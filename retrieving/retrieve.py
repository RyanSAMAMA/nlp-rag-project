from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from data_processing.embed import get_model, DEFAULT_EMBEDDING_MODEL
from config import COLLECTION_NAME
from dataclasses import dataclass

DEFAULT_SCORE_THRESHOLD = 0.4


@dataclass
class RetrievalResult:
    chunk: str
    filename: str
    score: float
    annee: str | None = None
    parti: str | None = None
    departement_nom: str | None = None
    candidat_nom: str | None = None
    candidat_prenom: str | None = None


def retrieve(
    query: str,
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    k: int = 10,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    annee: str | None = None,
    parti: str | None = None,
    departement: str | None = None,
    candidat: str | None = None,
) -> list[RetrievalResult]:
    model = get_model(model_name)
    query_vector = model.encode(query).tolist()

    conditions = []
    if annee:
        conditions.append(FieldCondition(key="annee", match=MatchValue(value=str(annee))))
    if parti:
        conditions.append(FieldCondition(key="titulaire_soutien", match=MatchText(text=parti)))
    if departement:
        conditions.append(FieldCondition(key="departement_nom", match=MatchText(text=departement)))
    if candidat:
        conditions.append(FieldCondition(key="titulaire_nom", match=MatchText(text=candidat)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    return [
        RetrievalResult(
            chunk=point.payload["chunk"],
            filename=point.payload.get("id") or point.payload.get("filename") or "unknown",
            score=point.score,
            annee=point.payload.get("annee"),
            parti=point.payload.get("titulaire_soutien"),
            departement_nom=point.payload.get("departement_nom"),
            candidat_nom=point.payload.get("titulaire_nom"),
            candidat_prenom=point.payload.get("titulaire_prenom"),
        )
        for point in results.points
    ]
