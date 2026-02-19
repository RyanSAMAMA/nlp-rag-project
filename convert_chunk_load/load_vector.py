from sentence_transformers import SentenceTransformer
from convert_chunk_load.chunk import ChunkResults
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def embedding_chunks(
    chunks: list[str], model_name: str = "BAAI/bge-small-en-v1.5"
) -> list[float]:
    """
    Convert a list of chunks to a vector using bge-small-en-v1.5 model from the sentence-transformers library

    Args:
        chunks (list[str]): The list of chunks to convert

    Returns:
        list[float]: The vector representation of the chunks
    """
    model = SentenceTransformer(model_name)
    return model.encode(chunks).tolist()


def embedding_chunk_results(
    chunk_results: ChunkResults, model_name: str = "BAAI/bge-small-en-v1.5"
) -> list[float]:
    """
    Convert the chunks in a ChunkResults object to a vector using bge-small-en-v1.5 model from the sentence-transformers library

    Args:
        chunk_results (ChunkResults): The ChunkResults object containing the chunks to convert

    Returns:
        list[float]: The vector representation of the chunks
    """
    return embedding_chunks(chunk_results.chunks, model_name=model_name)


class QdrandLoader:
    """
    Class to load a vector into a Qdrant collection

    Args:
        vector (list[float]): The vector to be loaded
        collection_name (str): The name of the Qdrant collection to load the vector into
    """

    def __init__(self, vector: list[float], collection_name: str):
        self.collection_name = collection_name

    def load(self):
        """
        Load the vector into the Qdrant collection
        """
