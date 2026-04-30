"""
Create Qdrant variant collections for ablation studies.
Reconstructs document texts from the existing 'documents' collection,
then re-chunks and re-embeds with different configurations.

Collections created:
  archelec_fixed_256   — FixedSizeChunker 256 chars  + e5-small
  archelec_fixed_1024  — FixedSizeChunker 1024 chars + e5-small
  archelec_e5large_512 — RecursiveChunker 512 chars  + e5-large

Usage:
    uv run python ingest_variants.py
    uv run python ingest_variants.py --collection archelec_fixed_256
"""
import argparse
import uuid
from collections import defaultdict
from dataclasses import dataclass

from qdrant_client.models import Distance, PointStruct, VectorParams

from config import COLLECTION_NAME, get_qdrant_client
from data_processing.chunk import ChunkResults, FixedSizeChunker, Chunker, rules
from data_processing.embed import embedding_chunks
from data_processing.convert import ConversionResults


@dataclass
class VariantConfig:
    collection_name: str
    embedding_model: str
    vector_size: int
    chunker: str
    chunk_size: int


VARIANTS: list[VariantConfig] = [
    VariantConfig(
        collection_name="archelec_fixed_256",
        embedding_model="intfloat/multilingual-e5-small",
        vector_size=384,
        chunker="fixed",
        chunk_size=256,
    ),
    VariantConfig(
        collection_name="archelec_fixed_1024",
        embedding_model="intfloat/multilingual-e5-small",
        vector_size=384,
        chunker="fixed",
        chunk_size=1024,
    ),
    VariantConfig(
        collection_name="archelec_e5large_512",
        embedding_model="intfloat/multilingual-e5-large",
        vector_size=1024,
        chunker="recursive",
        chunk_size=512,
    ),
]


def _pf_number(chunk_id: str) -> int:
    """Extract page number from EL..._PF_XX for sorting."""
    try:
        return int(chunk_id.rsplit("_PF_", 1)[1])
    except (IndexError, ValueError):
        return 0


def fetch_documents(client, collection_name: str) -> dict[str, dict]:
    """
    Returns {doc_id: {metadata, text}} where text is reconstructed
    from sorted chunks.
    """
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

    grouped: dict[str, list] = defaultdict(list)
    meta: dict[str, dict] = {}

    for r in records:
        p = r.payload
        chunk_id = p.get("id", "")
        doc_id = chunk_id.rsplit("_PF_", 1)[0] if "_PF_" in chunk_id else chunk_id
        grouped[doc_id].append((chunk_id, p.get("chunk", "")))
        if doc_id not in meta:
            meta[doc_id] = {k: v for k, v in p.items() if k != "chunk"}

    documents = {}
    for doc_id, chunks in grouped.items():
        chunks.sort(key=lambda x: _pf_number(x[0]))
        text = "\n\n".join(c for _, c in chunks)
        documents[doc_id] = {"meta": meta[doc_id], "text": text}

    return documents


def _ensure_collection(client, name: str, size: int):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
        print(f"  Created collection '{name}'")
    else:
        print(f"  Collection '{name}' already exists, skipping creation")


def ingest_variant(client, variant: VariantConfig, documents: dict[str, dict]):
    _ensure_collection(client, variant.collection_name, variant.vector_size)

    points = []
    chunker_fixed = FixedSizeChunker(chunk_size=variant.chunk_size) if variant.chunker == "fixed" else None

    total_chunks = 0
    for i, (doc_id, doc) in enumerate(documents.items()):
        text = doc["text"]
        meta = doc["meta"]

        if variant.chunker == "fixed":
            result = chunker_fixed.chunk(text, doc_id=doc_id)
            chunks = result.chunks
        else:
            # RecursiveChunker via Chunker class (needs ConversionResults)
            conversion = ConversionResults(filename=doc_id, conversion_type="reconstructed", result=text)
            chunker = Chunker(conversion, rules=rules, max_chunk_size=variant.chunk_size)
            result = chunker()
            chunks = result.chunks

        if not chunks:
            continue

        embeddings = embedding_chunks(chunks, model_name=variant.embedding_model)

        for chunk_text, embedding in zip(chunks, embeddings):
            payload = {**meta, "chunk": chunk_text}
            points.append(PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload))

        total_chunks += len(chunks)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(documents)} docs — {total_chunks} chunks so far")

    # Upsert in batches of 200
    batch_size = 200
    for start in range(0, len(points), batch_size):
        client.upsert(collection_name=variant.collection_name, points=points[start:start+batch_size])

    print(f"  ✓ {total_chunks} chunks ingested into '{variant.collection_name}'")


def main(target: str | None = None):
    client = get_qdrant_client()

    print(f"Fetching documents from '{COLLECTION_NAME}'…")
    documents = fetch_documents(client, COLLECTION_NAME)
    print(f"  → {len(documents)} unique documents reconstructed\n")

    variants = [v for v in VARIANTS if target is None or v.collection_name == target]
    if not variants:
        print(f"Unknown collection: {target}")
        return

    for variant in variants:
        print(f"{'─'*50}")
        print(f"  Variant : {variant.collection_name}")
        print(f"  Chunker : {variant.chunker} ({variant.chunk_size} chars)")
        print(f"  Model   : {variant.embedding_model}")
        print(f"{'─'*50}")
        ingest_variant(client, variant, documents)
        print()

    print("✓ All variants ingested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", help="Ingest only this collection")
    args = parser.parse_args()
    main(target=args.collection)
