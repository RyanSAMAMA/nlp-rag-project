from data_processing.convert import DoclingConverter
from data_processing.chunk import Chunker
from data_processing.embed import embedding_chunk_results
from data_processing.load import create_collection, insert_embeddings
from config import COLLECTION_NAME, EMBEDDING_MODEL, VECTOR_SIZE, get_qdrant_client
import sys


def ingest(filepath: str):
    print(f"Conversion de {filepath}...")
    conversion = DoclingConverter(filepath)()

    print("Chunking...")
    chunks = Chunker(conversion)()

    print("Embedding...")
    embeddings = embedding_chunk_results(chunks, model_name=EMBEDDING_MODEL)

    print("Chargement dans Qdrant...")
    client = get_qdrant_client()
    create_collection(client, COLLECTION_NAME, VECTOR_SIZE)
    insert_embeddings(client, chunks, embeddings, COLLECTION_NAME)

    print(f"{len(chunks.chunks)} chunks ingérés depuis '{chunks.filename}' dans la collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python ingest.py <chemin_du_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
