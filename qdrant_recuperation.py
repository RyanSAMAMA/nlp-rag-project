from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# But : à partir d'un vector query, récuperer les k points les plus proches dans une collection donnée.


def qdrant_recuperation(
    collection_name: str,
    vector_query: list,
    k: int,
    qdrant_client: QdrantClient,
    binary_search=False,
):
    """
    Récupère les k points les plus proches d'un vector query dans une collection donnée.

    Args:
        collection_name (str): Le nom de la collection dans laquelle effectuer la recherche.
        vector_query (list): Le vecteur de requête pour lequel trouver les points les plus proches.
        k (int): Le nombre de points les plus proches à récupérer.
        qdrant_client (QdrantClient): Une instance du client Qdrant pour interagir avec la base de données.
        binary_search (bool): Si True, utilise une recherche binaire pour trouver les points les plus proches. Sinon, utilise une recherche linéaire.

    Returns:
        list: Une liste des points les plus proches, chacun étant un dictionnaire contenant l'id, le vecteur et le payload du point.
    """
    if binary_search:
        # Implémentation d'une recherche binaire pour trouver les points les plus proches
        # Note: La recherche binaire nécessite que les points soient triés par distance, ce qui n'est pas le cas dans Qdrant.
        # Par conséquent, cette option n'est pas applicable directement avec Qdrant et est laissée à titre indicatif.
        raise NotImplementedError(
            "La recherche binaire n'est pas implémentée pour Qdrant."
        )

    # Utilisation de la méthode de recherche linéaire pour trouver les k points les plus proches
    search_result = qdrant_client.query_points(
        collection_name=collection_name, query=vector_query, limit=k
    )

    return search_result
