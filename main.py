from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_recuperation import qdrant_recuperation
import os
from dotenv import load_dotenv

load_dotenv()

endpoint_qdrant = os.getenv("QDRANT_ENDPOINT")
api_qdrant = os.getenv("QDRANT_API")


qdrant_client = QdrantClient(
    url=endpoint_qdrant,
    api_key=api_qdrant,
)

# qdrant_client.create_collection(
#     collection_name="test",
#     vectors_config=VectorParams(size=4, distance=Distance.COSINE),
# )

# operation_info = qdrant_client.upsert(
#     collection_name="test",
#     wait=True,
#     points=[
#         PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#         PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#         PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#         PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
#         PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#         PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#     ],
# )

# print(operation_info)

print(qdrant_recuperation(collection_name="test", vector_query=[0.2, 0.1, 0.9, 0.7], k=1, qdrant_client=qdrant_client))