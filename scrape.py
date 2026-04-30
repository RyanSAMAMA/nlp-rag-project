"""
Script de scraping Archelec depuis Internet Archive.

Pipeline :
  1. Récupère les identifiants filtrés par année via l'API scrape d'IA
  2. Échantillonne de façon stratifiée par département (round-robin)
  3. Pour chaque document, récupère les métadonnées + texte OCR (DjVuTXT)
  4. Soft-cap par parti (≤40% du total) pour éviter la surreprésentation
  5. Chunk → embed → insert dans Qdrant avec métadonnées enrichies

Usage :
  uv run python scrape.py --years 1967 1981 --per-year 200
  uv run python scrape.py --years 1967 1981 1978 --per-year 200
  uv run python scrape.py --resume                           # reprend le checkpoint
"""

import argparse
import json
import random
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path

import requests
from qdrant_client.models import PointStruct

from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE, get_qdrant_client
from data_processing.chunk import Chunker
from data_processing.convert import ConversionResults
from data_processing.embed import embedding_chunk_results
from data_processing.load import create_collection
from corpus.parties import normalize

IA_BASE = "https://archive.org"
IA_COLLECTION = "archiveselectoralesducevipof"
CHECKPOINT_FILE = ".scrape_checkpoint.json"

# A single party must not exceed this share of ingested docs
MAX_PARTY_SHARE = 0.40

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "archelec-rag-scraper/1.0"})

_DEPT_RE = re.compile(r"_L_\d{4}_\d+_([0-9A-Z]+)_")


def _dept_from_id(identifier: str) -> str:
    m = _DEPT_RE.search(identifier)
    return m.group(1) if m else "unknown"


def stratified_sample(identifiers: list[str], n: int) -> list[str]:
    """Round-robin across departments so every region is represented."""
    by_dept: dict[str, list[str]] = defaultdict(list)
    for ident in identifiers:
        by_dept[_dept_from_id(ident)].append(ident)

    # Shuffle within each dept for randomness
    rng = random.Random(42)
    for lst in by_dept.values():
        rng.shuffle(lst)

    # Round-robin interleave
    result: list[str] = []
    buckets = list(by_dept.values())
    i = 0
    while len(result) < n and any(buckets):
        bucket = buckets[i % len(buckets)]
        if bucket:
            result.append(bucket.pop(0))
        i += 1
        buckets = [b for b in buckets if b] or buckets  # compact when empty

    return result[:n]


# ---
# Internet Archive
# ---

def fetch_identifiers(year: int | None = None) -> list[str]:
    url = f"{IA_BASE}/services/search/v1/scrape"
    query = f'collection:{IA_COLLECTION} AND type:"profession de foi"'
    if year:
        query += f" AND year:{year}"
    params: dict = {"q": query, "count": 1000}
    ids: list[str] = []

    label = f"année {year}" if year else "toute la collection"
    print(f"  Récupération des identifiants ({label})...", end=" ", flush=True)
    while True:
        resp = SESSION.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        ids.extend(item["identifier"] for item in data.get("items", []))
        cursor = data.get("cursor")
        if not cursor:
            break
        params["cursor"] = cursor
        time.sleep(0.2)

    ids = [i for i in ids if "_L_" in i]
    print(f"{len(ids)} législatives trouvées.")
    return ids


def fetch_metadata(identifier: str) -> dict | None:
    url = f"{IA_BASE}/metadata/{identifier}"
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == 2:
                print(f"  ERREUR metadata {identifier}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def fetch_ocr(server: str, directory: str, files: list[dict]) -> tuple[str | None, str | None]:
    ocr_file = next((f for f in files if f.get("format") == "DjVuTXT"), None)
    if not ocr_file:
        return None, None
    url = f"https://{server}{directory}/{ocr_file['name']}"
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text, url
        except Exception as e:
            if attempt == 2:
                print(f"  ERREUR OCR {url}: {e}")
                return None, None
            time.sleep(2 ** attempt)
    return None, None


# ---
# Extraction des métadonnées
# ---

def extract_fields(raw: dict) -> dict:
    fields_of_interest = {
        "subject", "title", "type", "contexte-election", "contexte-tour",
        "cote", "date", "departement", "departement-nom", "circonscription",
    }
    candidate_suffixes = ("-titulaire", "-suppleant")

    result: dict = {}
    for key, value in raw.items():
        clean = key[3:] if len(key) > 3 and key[2] == "-" and key[:2].isalpha() else key
        if clean not in fields_of_interest and not any(clean.endswith(s) for s in candidate_suffixes):
            continue
        if value in ("NR", "", None):
            continue
        result[clean] = value
    return result


def build_payload(identifier: str, fields: dict, ocr_url: str, pdf_url: str | None) -> dict:
    def candidate(suffix: str) -> dict:
        return {k.replace(f"-{suffix}", ""): v for k, v in fields.items() if k.endswith(f"-{suffix}")}

    titulaire = candidate("titulaire")
    suppleant = candidate("suppleant")

    return {
        "source": "archelec",
        "id": identifier,
        "annee": str(fields.get("date", ""))[:4] or None,
        "date": str(fields.get("date", "")),
        "contexte_election": fields.get("contexte-election"),
        "contexte_tour": fields.get("contexte-tour"),
        "departement": fields.get("departement"),
        "departement_nom": fields.get("departement-nom"),
        "circonscription": fields.get("circonscription"),
        "titulaire_nom": titulaire.get("nom"),
        "titulaire_prenom": titulaire.get("prenom"),
        "titulaire_sexe": titulaire.get("sexe"),
        "titulaire_profession": titulaire.get("profession"),
        "titulaire_soutien": titulaire.get("soutien"),
        "suppleant_nom": suppleant.get("nom"),
        "suppleant_prenom": suppleant.get("prenom"),
        "ocr_url": ocr_url,
        "pdf_url": pdf_url,
    }


# ---
# Checkpoint
# ---

def load_checkpoint() -> set[str]:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set[str]) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done), f)


# ---
# Ingestion
# ---

def ingest_identifiers(
    identifiers: list[str],
    client,
    done: set[str],
    per_year: int | None = None,
    label: str = "",
) -> tuple[int, int, int]:
    ingested = skipped = errors = 0
    party_counts: dict[str, int] = defaultdict(int)
    dept_counts: dict[str, int] = defaultdict(int)

    # Pre-sample for department diversity
    candidates = [i for i in identifiers if i not in done]
    if per_year and len(candidates) > per_year * 3:
        # Over-sample 3× then let the party soft-cap trim the rest
        candidates = stratified_sample(candidates, per_year * 3)
    elif per_year:
        candidates = stratified_sample(candidates, len(candidates))

    total = len(candidates)

    for idx, identifier in enumerate(candidates, 1):
        if per_year and ingested >= per_year:
            break

        prefix = f"[{label} {idx}/{total}]" if label else f"[{idx}/{total}]"
        print(f"{prefix} {identifier}", end=" ... ", flush=True)

        meta = fetch_metadata(identifier)
        if not meta:
            errors += 1
            print("erreur metadata")
            continue

        server = meta.get("server", "")
        directory = meta.get("dir", "")
        files = meta.get("files", [])
        raw_metadata = meta.get("metadata", {})

        ocr_text, ocr_url = fetch_ocr(server, directory, files)
        if not ocr_text or len(ocr_text.strip()) < 50:
            skipped += 1
            print("pas d'OCR")
            continue

        pdf_file = next((f for f in files if f.get("format") == "Image Container PDF"), None)
        pdf_url = f"https://{server}{directory}/{pdf_file['name']}" if pdf_file else None

        fields = extract_fields(raw_metadata)
        payload = build_payload(identifier, fields, ocr_url, pdf_url)

        # Soft party cap
        party_key = normalize(payload.get("titulaire_soutien") or "")
        if ingested > 10:
            share = party_counts[party_key] / ingested
            if share > MAX_PARTY_SHARE:
                skipped += 1
                print(f"skip (parti {party_key} déjà {share:.0%})")
                continue

        conversion = ConversionResults(filename=identifier, conversion_type="ocr", result=ocr_text)
        chunk_results = Chunker(conversion)()
        if not chunk_results.chunks:
            skipped += 1
            print("chunks vides")
            continue

        embedding_results = embedding_chunk_results(chunk_results, model_name=EMBEDDING_MODEL)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_results.embeddings[i],
                    payload={**payload, "chunk": chunk_results.chunks[i]},
                )
                for i in range(len(chunk_results.chunks))
            ],
        )

        ingested += 1
        done.add(identifier)
        party_counts[party_key] += 1
        dept_counts[payload.get("departement_nom") or _dept_from_id(identifier)] += 1
        print(f"{len(chunk_results.chunks)} chunks  [{party_key} | {payload.get('departement_nom', '?')}]")

        if ingested % 50 == 0:
            save_checkpoint(done)
            print(f"\n  → Partis: { {k: v for k, v in sorted(party_counts.items(), key=lambda x: -x[1])[:5]} }")
            print(f"  → Depts ({len(dept_counts)} différents)\n")

        time.sleep(0.3)

    print(f"\n  Partis représentés :")
    for party, count in sorted(party_counts.items(), key=lambda x: -x[1]):
        pct = count / ingested * 100 if ingested else 0
        print(f"     {count:3d} ({pct:4.1f}%)  {party}")
    print(f"  Départements couverts : {len(dept_counts)}")

    return ingested, skipped, errors


# ---
# Main
# ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Archelec (Internet Archive) → Qdrant")
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--per-year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME, VECTOR_SIZE)

    done = load_checkpoint() if args.resume else set()
    if done:
        print(f"Reprise : {len(done)} documents déjà traités.")

    total_ingested = total_skipped = total_errors = 0

    if args.years:
        for year in args.years:
            print(f"\nannée {year}")
            ids = fetch_identifiers(year=year)
            i, s, e = ingest_identifiers(ids, client, done, per_year=args.per_year, label=str(year))
            total_ingested += i
            total_skipped += s
            total_errors += e
            print(f"  {i} ingérés, {s} ignorés, {e} erreurs")
    else:
        print("\ncollection complète")
        ids = fetch_identifiers()
        if args.limit:
            ids = ids[: args.limit]
        total_ingested, total_skipped, total_errors = ingest_identifiers(
            ids, client, done, per_year=args.per_year
        )

    save_checkpoint(done)
    print(f"\ntotal : {total_ingested} ingérés, {total_skipped} ignorés, {total_errors} erreurs.")


if __name__ == "__main__":
    main()
