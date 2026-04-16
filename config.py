import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        api_key=os.environ["QDRANT_API"],
        url=os.environ["QDRANT_ENDPOINT"],
        check_compatibility=False,
    )
