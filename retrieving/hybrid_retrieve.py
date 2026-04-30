"""
Hybrid retrieval combining dense (Qdrant) and BM25 via Reciprocal Rank Fusion (RRF).
"""
from qdrant_client import QdrantClient

from retrieving.retrieve import RetrievalResult, retrieve
from retrieving.bm25_retrieve import retrieve_bm25

RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


def retrieve_hybrid(
    query: str,
    client: QdrantClient,
    collection_name: str,
    k: int = 10,
    model_name: str = "intfloat/multilingual-e5-small",
    score_threshold: float = 0.0,
    alpha: float = 0.5,
) -> list[RetrievalResult]:
    """
    Fetch top 2*k from dense and BM25, fuse with RRF, return top k.
    alpha controls dense weight (1-alpha = BM25 weight).
    """
    fetch_k = k * 2

    dense_results = retrieve(
        query=query,
        client=client,
        collection_name=collection_name,
        k=fetch_k,
        model_name=model_name,
        score_threshold=score_threshold,
    )
    bm25_results = retrieve_bm25(
        query=query,
        client=client,
        collection_name=collection_name,
        k=fetch_k,
    )

    scores: dict[str, float] = {}
    result_map: dict[str, RetrievalResult] = {}

    for rank, r in enumerate(dense_results):
        scores[r.filename] = scores.get(r.filename, 0.0) + alpha * _rrf_score(rank)
        result_map[r.filename] = r

    for rank, r in enumerate(bm25_results):
        scores[r.filename] = scores.get(r.filename, 0.0) + (1 - alpha) * _rrf_score(rank)
        if r.filename not in result_map:
            result_map[r.filename] = r

    top_k = sorted(scores.items(), key=lambda x: -x[1])[:k]

    return [
        RetrievalResult(
            chunk=result_map[fn].chunk,
            filename=fn,
            score=score,
            annee=result_map[fn].annee,
            parti=result_map[fn].parti,
            departement_nom=result_map[fn].departement_nom,
            candidat_nom=result_map[fn].candidat_nom,
            candidat_prenom=result_map[fn].candidat_prenom,
        )
        for fn, score in top_k
    ]
