"""
Local vector store ingestion script.

Reads pre-chunked .txt files + .txt.metadata.json sidecars from
LocalStack S3 (processed-chunks/), embeds each chunk via Bedrock Titan
Embed Text v2 (real AWS), and upserts into a ChromaDB persistent
collection at backend/.local-vectorstore/.

Idempotent: chunk IDs already in the collection are skipped, so you can
safely re-run after partial failures or pipeline additions.

Concurrency: ThreadPoolExecutor with MAX_WORKERS concurrent Titan calls.
Titan has a default TPS quota of ~10; 4 workers keeps us safely under it.

Usage:
  ENV=local uv run ingest_local.py

Prerequisites:
  - LocalStack running with processed-chunks/ populated
      (pipeline: npm run dev && npm run chunk)
  - AWS credentials configured for real Bedrock (Titan embeddings)
  - chromadb installed:  uv add chromadb
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config as BotoConfig
import chromadb

from app.config import load_config

S3_BUCKET = os.environ.get("S3_BUCKET", "haki-ai-data")
CHUNKS_PREFIX = "processed-chunks/"
COLLECTION_NAME = "haki_chunks"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"
# Titan Embed Text v2's default on-demand quota is ~10 TPS per account.
# MAX_WORKERS=2 with adaptive retries (see BOTO_CONFIG below) keeps the
# effective rate well under that; anything higher starts bleeding chunks
# to ThrottlingException on large re-embeds.
MAX_WORKERS = 2
BOTO_CONFIG = BotoConfig(
    retries={"max_attempts": 10, "mode": "adaptive"},
    read_timeout=30,
    connect_timeout=10,
)

# Persistent store lives at backend/.local-vectorstore (one level up from
# backend/app/) so the eval runner and ingest share the same collection.
# Before this fix, `dirname(__file__)` resolved to backend/app/, which
# silently split the corpus into two parallel ChromaDB stores — eval
# reads from backend/.local-vectorstore while ingest wrote to
# backend/app/.local-vectorstore. See `swahili_retrieval_ulizallama.plan.md`
# §5b for the symptom trail that uncovered this.
VECTORSTORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".local-vectorstore",
)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def list_chunk_keys(s3) -> list[str]:
    """Returns all .txt chunk keys; skips .metadata.json files and .complete markers."""
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=CHUNKS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".txt") and not key.endswith(".metadata.json"):
                keys.append(key)
    return keys


def read_chunk(s3, key: str) -> tuple[str, dict]:
    """
    Returns (text, metadata_attributes) for a chunk key.
    The sidecar is at <key>.metadata.json and uses Bedrock KB native format:
      { "metadataAttributes": { source, chapter, section, title, chunkId, pageImageKey } }
    """
    text = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")

    meta_key = key + ".metadata.json"
    meta_raw = s3.get_object(Bucket=S3_BUCKET, Key=meta_key)["Body"].read()
    meta = json.loads(meta_raw).get("metadataAttributes", {})

    return text, meta


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(bedrock, text: str) -> list[float]:
    """Calls Titan Embed Text v2 and returns the 1024-dim embedding vector."""
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    # LocalStack S3 for chunk text + metadata
    s3 = boto3.client(
        "s3",
        endpoint_url=config.localstack_endpoint,
        region_name=config.aws_region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    # Real AWS Bedrock Runtime for Titan embeddings (adaptive retry keeps
    # us polite when the account's TPS quota gets hot).
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
        config=BOTO_CONFIG,
    )

    # ChromaDB persistent collection (cosine similarity matches S3 Vectors default)
    chroma = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Discover chunks ───────────────────────────────────────────────────────

    print("Listing chunks from LocalStack S3...")
    keys = list_chunk_keys(s3)
    print(f"Found {len(keys)} chunks in S3")

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    print(f"Already embedded: {len(existing_ids)}\n")

    # Load text + metadata for chunks not yet in ChromaDB
    pending = []
    for key in keys:
        try:
            text, meta = read_chunk(s3, key)
            chunk_id = meta.get("chunkId") or key.removeprefix(CHUNKS_PREFIX).removesuffix(".txt")
            if chunk_id in existing_ids:
                continue
            pending.append({"id": chunk_id, "text": text, "meta": meta})
        except Exception as err:
            print(f"  [SKIP] {key}: {err}")

    if not pending:
        print(f"Nothing to embed — all {len(existing_ids)} chunks already in ChromaDB.")
        return

    # ── Embed and upsert ──────────────────────────────────────────────────────

    total = len(pending)
    print(f"Embedding {total} new chunks (workers={MAX_WORKERS})...")

    def process(chunk: dict) -> tuple[str, str, dict, list[float]]:
        vector = embed(bedrock, chunk["text"])
        return chunk["id"], chunk["text"], chunk["meta"], vector

    done = 0
    errors = 0
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, c): c for c in pending}
        for future in as_completed(futures):
            try:
                chunk_id, text, meta, vector = future.result()
                # Strip None values — ChromaDB metadata values must be str/int/float/bool
                clean_meta = {k: v for k, v in meta.items() if v is not None}
                collection.upsert(
                    ids=[chunk_id],
                    documents=[text],
                    embeddings=[vector],
                    metadatas=[clean_meta],
                )
                done += 1
                if done % 50 == 0 or done == total:
                    elapsed = time.monotonic() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  {done}/{total}  ({rate:.1f}/s  ETA {eta:.0f}s)")
            except Exception as err:
                errors += 1
                print(f"  [ERROR] {futures[future]['id']}: {err}")

    # ── Summary ───────────────────────────────────────────────────────────────

    elapsed = time.monotonic() - start
    print(f"\nDone. {done} embedded, {errors} errors in {elapsed:.0f}s.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")
    print(f"Store path: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    if os.environ.get("ENV") != "local":
        print("Error: set ENV=local to ingest against LocalStack")
        sys.exit(1)
    main()
