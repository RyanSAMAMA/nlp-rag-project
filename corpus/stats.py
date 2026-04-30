"""
Compute descriptive statistics on the Archelec corpus stored in Qdrant.
Saves a JSON summary and prints a report.

Usage:
    uv run python -m corpus.stats
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from config import COLLECTION_NAME, get_qdrant_client
from corpus.parties import normalize

DEPT_MAP: dict[str, str] = {
    "Basses-Alpes": "Alpes-de-Haute-Provence",
    "Basses Alpes": "Alpes-de-Haute-Provence",
    "Alpes de Haute-Provence": "Alpes-de-Haute-Provence",
}

STATS_PATH = Path("results/corpus_stats.json")


def _doc_id(chunk_id: str) -> str:
    parts = chunk_id.rsplit("_PF_", 1)
    return parts[0] if len(parts) == 2 else chunk_id


def fetch_all(client, collection_name: str) -> list[dict]:
    """Scroll the entire collection and return payloads."""
    records = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return [r.payload for r in records]


def compute_stats(payloads: list[dict]) -> dict:
    docs: dict[str, dict] = {}
    chunks_per_doc: dict[str, list[str]] = defaultdict(list)

    for p in payloads:
        chunk_id = p.get("id", "")
        doc_id = _doc_id(chunk_id)
        chunks_per_doc[doc_id].append(p.get("chunk", ""))

        if doc_id not in docs:
            docs[doc_id] = {
                "annee": p.get("annee", ""),
                "parti": normalize(p.get("titulaire_soutien", "") or ""),
                "departement": DEPT_MAP.get(
                    p.get("departement_nom", "") or "",
                    p.get("departement_nom", "") or "",
                ),
                "sexe": p.get("titulaire_sexe", "") or "",
                "profession": p.get("titulaire_profession", "") or "",
                "nom": p.get("titulaire_nom", "") or "",
                "prenom": p.get("titulaire_prenom", "") or "",
            }

    n_docs = len(docs)
    n_chunks = len(payloads)
    chunk_lengths = [len(p.get("chunk", "")) for p in payloads]

    by_year = Counter(d["annee"] for d in docs.values())
    by_parti = Counter(d["parti"] for d in docs.values() if d["parti"])
    by_dept = Counter(d["departement"] for d in docs.values() if d["departement"])
    by_sexe = Counter(d["sexe"] for d in docs.values() if d["sexe"])
    by_sexe_year: dict[str, Counter] = defaultdict(Counter)
    for d in docs.values():
        if d["sexe"] and d["annee"]:
            by_sexe_year[d["annee"]][d["sexe"]] += 1
    by_profession = Counter(d["profession"] for d in docs.values() if d["profession"])
    chunks_per_doc_counts = [len(v) for v in chunks_per_doc.values()]

    return {
        "n_documents": n_docs,
        "n_chunks": n_chunks,
        "by_year": dict(sorted(by_year.items())),
        "by_parti": dict(by_parti.most_common(25)),
        "by_departement": dict(by_dept.most_common(30)),
        "by_sexe": dict(by_sexe),
        "by_sexe_year": {y: dict(c) for y, c in by_sexe_year.items()},
        "by_profession": dict(by_profession.most_common(20)),
        "chunk_lengths": {
            "mean": round(sum(chunk_lengths) / len(chunk_lengths), 1),
            "median": sorted(chunk_lengths)[len(chunk_lengths) // 2],
            "min": min(chunk_lengths),
            "max": max(chunk_lengths),
            "all": chunk_lengths,
        },
        "chunks_per_doc": {
            "mean": round(sum(chunks_per_doc_counts) / len(chunks_per_doc_counts), 1),
            "median": sorted(chunks_per_doc_counts)[len(chunks_per_doc_counts) // 2],
            "min": min(chunks_per_doc_counts),
            "max": max(chunks_per_doc_counts),
            "all": chunks_per_doc_counts,
        },
    }


def main():
    STATS_PATH.parent.mkdir(exist_ok=True)
    client = get_qdrant_client()

    print(f"Fetching all chunks from '{COLLECTION_NAME}'…")
    payloads = fetch_all(client, COLLECTION_NAME)
    print(f"  → {len(payloads)} chunks fetched")

    stats = compute_stats(payloads)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n{'─' * 45}")
    print(f"  Documents uniques    : {stats['n_documents']}")
    print(f"  Chunks total         : {stats['n_chunks']}")
    print(f"  Par année            : {stats['by_year']}")
    print(f"  Genre                : {stats['by_sexe']}")
    print(f"  Longueur chunk moy.  : {stats['chunk_lengths']['mean']} chars")
    print(f"  Chunks / doc moy.    : {stats['chunks_per_doc']['mean']}")
    print(f"\nStats saved at {STATS_PATH}")


if __name__ == "__main__":
    main()
