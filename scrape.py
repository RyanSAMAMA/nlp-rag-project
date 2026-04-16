"""
Script de scraping Archelec depuis Internet Archive.

Pipeline :
  1. Récupère les identifiants filtrés par année via l'API scrape d'IA
  2. Pour chaque document, récupère les métadonnées + texte OCR (DjVuTXT)
  3. Chunk → embed → insert dans Qdrant avec métadonnées enrichies

Usage :
  uv run python scrape.py --years 1967 1981 --per-year 100   # 100 docs par année
  uv run python scrape.py --years 1967 --per-year 50         # test rapide
  uv run python scrape.py --resume                           # reprend le checkpoint
  uv run python scrape.py --limit 10                         # scrape tout, limité
"""

import argparse
import json
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

IA_BASE = "https://archive.org"
IA_COLLECTION = "archiveselectoralesducevipof"
CHECKPOINT_FILE = ".scrape_checkpoint.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "archelec-rag-scraper/1.0"})


# ---------------------------------------------------------------------------
# Internet Archive
# ---------------------------------------------------------------------------

def fetch_identifiers(year: int | None = None) -> list[str]:
    """Récupère les identifiants de la collection, optionnellement filtrés par année."""
    url = f"{IA_BASE}/services/search/v1/scrape"
    query = f'collection:{IA_COLLECTION} AND type:"profession de foi"'
    if year:
        query += f" AND year:{year}"
    params: dict = {"q": query, "count": 1000}
    ids: list[str] = []

    label = f"année {year}" if year else "toute la collection"
    print(f"  Récupération des identifiants ({label})...", end=" ")
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

    # Garde uniquement les législatives (identifiant contient _L_)
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
            time.sleep(2**attempt)
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
            time.sleep(2**attempt)
    return None, None


# ---------------------------------------------------------------------------
# Extraction des métadonnées
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint() -> set[str]:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set[str]) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done), f)


# ---------------------------------------------------------------------------
# Ingestion d'une liste d'identifiants
# ---------------------------------------------------------------------------

def ingest_identifiers(
    identifiers: list[str],
    client,
    done: set[str],
    per_year: int | None = None,
    label: str = "",
) -> tuple[int, int, int]:
    """
    Ingère une liste d'identifiants dans Qdrant.
    Retourne (ingested, skipped, errors).
    """
    ingested = skipped = errors = 0
    total = len(identifiers)
    party_counts: dict[str, int] = defaultdict(int)

    for idx, identifier in enumerate(identifiers, 1):
        if per_year and ingested >= per_year:
            break

        prefix = f"[{label} {idx}/{total}]" if label else f"[{idx}/{total}]"
        print(f"{prefix} {identifier}", end=" ... ")

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

        conversion = ConversionResults(filename=identifier, conversion_type="ocr", result=ocr_text)
        chunk_results = Chunker(conversion)()
        if not chunk_results.chunks:
            skipped += 1
            print("chunks vides")
            continue

        embedding_results = embedding_chunk_results(chunk_results, model_name=EMBEDDING_MODEL)
        fields = extract_fields(raw_metadata)
        payload = build_payload(identifier, fields, ocr_url, pdf_url)

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
        party = payload.get("titulaire_soutien") or "inconnu"
        party_counts[party] += 1
        print(f"{len(chunk_results.chunks)} chunks  [{party}]")

        if ingested % 50 == 0:
            save_checkpoint(done)

        time.sleep(0.3)

    if party_counts:
        print(f"  → Partis représentés :")
        for party, count in sorted(party_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"     {count:3d}  {party}")

    return ingested, skipped, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Archelec (Internet Archive) → Qdrant")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Années à ingérer (ex: --years 1967 1981)")
    parser.add_argument("--per-year", type=int, default=None,
                        help="Nombre max de documents par année (ex: --per-year 100)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limite globale (sans filtre année)")
    parser.add_argument("--resume", action="store_true",
                        help="Reprend depuis le dernier checkpoint")
    args = parser.parse_args()

    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME, VECTOR_SIZE)

    done = load_checkpoint() if args.resume else set()
    if done:
        print(f"Reprise : {len(done)} documents déjà traités.")

    total_ingested = total_skipped = total_errors = 0

    if args.years:
        for year in args.years:
            print(f"\n── Année {year} ──────────────────────────────")
            ids = fetch_identifiers(year=year)
            ids = [i for i in ids if i not in done]
            i, s, e = ingest_identifiers(ids, client, done, per_year=args.per_year, label=str(year))
            total_ingested += i
            total_skipped += s
            total_errors += e
            print(f"  {i} ingérés, {s} sans OCR, {e} erreurs")
    else:
        print("\n── Toute la collection ──────────────────────")
        ids = fetch_identifiers()
        ids = [i for i in ids if i not in done]
        if args.limit:
            ids = ids[: args.limit]
        total_ingested, total_skipped, total_errors = ingest_identifiers(ids, client, done)

    save_checkpoint(done)
    print(f"\n{'═'*50}")
    print(f"Total : {total_ingested} ingérés, {total_skipped} sans OCR, {total_errors} erreurs.")


if __name__ == "__main__":
    main()
