"""
BM25 retrieval over an Archelec Qdrant collection.
The index is built lazily from all chunks in the collection and cached in memory.
"""
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from retrieving.retrieve import RetrievalResult


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _build_index(client: QdrantClient, collection_name: str) -> tuple[BM25Okapi, list[dict]]:
    """Fetch all chunks and build a BM25 index."""
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

    payloads = [r.payload for r in records]
    corpus = [_tokenize(p.get("chunk", "")) for p in payloads]
    index = BM25Okapi(corpus)
    return index, payloads


_index_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def get_index(client: QdrantClient, collection_name: str) -> tuple[BM25Okapi, list[dict]]:
    if collection_name not in _index_cache:
        print(f"  [BM25] Building index for '{collection_name}'…", flush=True)
        _index_cache[collection_name] = _build_index(client, collection_name)
        print(f"  [BM25] Index ready ({len(_index_cache[collection_name][1])} chunks)")
    return _index_cache[collection_name]


def retrieve_bm25(
    query: str,
    client: QdrantClient,
    collection_name: str,
    k: int = 10,
) -> list[RetrievalResult]:
    index, payloads = get_index(client, collection_name)
    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    results = []
    for i in top_indices:
        if scores[i] <= 0:
            continue
        p = payloads[i]
        results.append(
            RetrievalResult(
                chunk=p.get("chunk", ""),
                filename=p.get("id") or p.get("filename") or "unknown",
                score=float(scores[i]),
                annee=p.get("annee"),
                parti=p.get("titulaire_soutien"),
                departement_nom=p.get("departement_nom"),
                candidat_nom=p.get("titulaire_nom"),
                candidat_prenom=p.get("titulaire_prenom"),
            )
        )
    return results
