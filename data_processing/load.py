from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from data_processing.embed import EmbeddingResults
from data_processing.chunk import ChunkResults
import uuid


def create_collection(client: QdrantClient, collection_name: str, size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=size,
                distance=Distance.COSINE,
            ),
        )


def insert_embeddings(
    client: QdrantClient,
    chunk_results: ChunkResults,
    embedding_results: EmbeddingResults,
    collection_name: str,
):
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding_results.embeddings[i],
                payload={
                    "filename": chunk_results.filename,
                    "chunk": chunk_results.chunks[i],
                },
            )
            for i in range(len(chunk_results.chunks))
        ],
    )
