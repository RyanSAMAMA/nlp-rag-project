from sentence_transformers import SentenceTransformer
from data_processing.chunk import ChunkResults
from config import EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL
from dataclasses import dataclass

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


@dataclass
class EmbeddingResults:
    """
    Dataclass to store the results of the embedding process

    Attributes:
        filename (str): The name of the file that was embedded
        embedding_model (str): The name of the embedding model that was used
        embeddings (list[list[float]]): The embeddings of the chunks
    """

    filename: str
    embedding_model: str
    embeddings: list[list[float]]


def embedding_chunks(
    chunks: list[str], model_name: str = DEFAULT_EMBEDDING_MODEL
) -> list[float]:
    """
    Convert a list of chunks to a vector using the sentence-transformers library

    Args:
        chunks (list[str]): The list of chunks to convert
        model_name (str): The name of the model to use

    Returns:
        list[float]: The vector representation of the chunks
    """
    model = get_model(model_name)
    return model.encode(chunks).tolist()


def embedding_chunk_results(
    chunk_results: ChunkResults, model_name: str = DEFAULT_EMBEDDING_MODEL
) -> list[float]:
    """
    Convert the chunks in a ChunkResults object to embeddings

    Args:
        chunk_results (ChunkResults): The ChunkResults object containing the chunks to convert
        model_name (str): The name of the model to use

    Returns:
        EmbeddingResults: The embeddings of the chunks
    """
    embedding_results = EmbeddingResults(
        filename=chunk_results.filename,
        embedding_model=model_name,
        embeddings=embedding_chunks(chunk_results.chunks, model_name),
    )
    return embedding_results
