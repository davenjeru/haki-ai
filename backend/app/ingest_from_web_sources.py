"""
Web-sourced corpus ingestion.

Counterpart to ``app.ingest_local``, which handles the statute corpus
under ``processed-chunks/``. This script reads *crawled* Q&A content
(SheriaPlex forum threads today; KenyaLaw summaries / case-law briefs
tomorrow) from the ``faq-chunks/`` S3 prefix and upserts them into the
same ChromaDB collection used by the statute corpus, so retrieval can
merge both sources naturally.

Why a separate ingest script instead of extending ``ingest_local.py``?

- The two corpora have different provenance, different re-ingest
  cadences, and different throttling profiles (statute re-embeds run
  once per pipeline change; FAQ re-embeds run after every crawl).
- Keeping them in separate scripts makes it easy to re-embed only the
  FAQ half after a crawl without paying Titan calls on unchanged
  statute chunks.
- The filename signals intent: if we later add a KenyaLaw crawler or
  any other web source, we extend this script (and its --prefix
  argument) rather than overloading the statute ingest.

Metadata contract
-----------------
Each chunk carries a ``corpus`` attribute set by the crawler:
``"statute"`` (default) for statute PDFs, ``"faq"`` for SheriaPlex
threads. The retrieval pipeline filters on this attribute so the
statute specialists and the ``FAQAgent`` see disjoint slices of the
collection even though they share one vector store.

Usage
-----
::

    # Default: read s3://<bucket>/faq-chunks/ into haki_chunks collection
    ENV=local uv run python -m app.ingest_from_web_sources

    # Custom prefix (e.g. future KenyaLaw summaries at kenyalaw-chunks/)
    ENV=local uv run python -m app.ingest_from_web_sources --prefix kenyalaw-chunks/

    # Dry run — list what would be embedded, don't call Titan or Chroma
    ENV=local uv run python -m app.ingest_from_web_sources --dry-run
"""

from __future__ import annotations

import argparse
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
DEFAULT_PREFIX = "faq-chunks/"
COLLECTION_NAME = "haki_chunks"
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

# Titan Embed Text v2's default on-demand quota is ~10 TPS per account.
# MAX_WORKERS=2 + adaptive retry keeps the effective rate safely under
# that when re-embedding a couple hundred FAQ threads at once. Mirrors
# the throttling profile used by `app.ingest_local`.
MAX_WORKERS = 2
BOTO_CONFIG = BotoConfig(
    retries={"max_attempts": 10, "mode": "adaptive"},
    read_timeout=30,
    connect_timeout=10,
)

# Must match the path used by `app.ingest_local` and by the eval runner
# (`backend/evals/runner.py`) so all ingests write — and all queries
# read — the same collection. Before the ingest-path fix landed, these
# three had drifted into parallel stores.
VECTORSTORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".local-vectorstore",
)


# ── S3 helpers ────────────────────────────────────────────────────────────────


def list_chunk_keys(s3, prefix: str) -> list[str]:
    """Returns all ``.txt`` chunk keys under ``prefix``, skipping metadata sidecars."""
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith(".txt") and not key.endswith(".metadata.json"):
                keys.append(key)
    return keys


def read_chunk(s3, key: str) -> tuple[str, dict]:
    """
    Returns ``(text, metadata_attributes)`` for a chunk key.

    The sidecar lives at ``<key>.metadata.json`` in the Bedrock KB native
    format: ``{"metadataAttributes": {...}}``. We keep whatever the
    crawler emitted (corpus, source, section, category, url, chunkId,
    chunkType) so the retriever's ``metadata_filter`` works identically
    to the statute path.
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


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed crawled web-source chunks (default: SheriaPlex FAQs).",
    )
    p.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=(
            "S3 prefix to ingest. Defaults to 'faq-chunks/' (SheriaPlex). "
            "Override when adding new crawled corpora (e.g. 'kenyalaw-chunks/')."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List pending chunks and exit; no Titan or ChromaDB calls.",
    )
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    config = load_config()

    prefix = args.prefix if args.prefix.endswith("/") else args.prefix + "/"

    # LocalStack S3 for chunk text + metadata
    s3 = boto3.client(
        "s3",
        endpoint_url=config.localstack_endpoint,
        region_name=config.aws_region,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    print(f"Listing chunks from s3://{S3_BUCKET}/{prefix} ...")
    keys = list_chunk_keys(s3, prefix)
    print(f"Found {len(keys)} chunks in S3")

    if not keys:
        print(
            f"Nothing under {prefix}. If you expected FAQ content, run the "
            f"crawler first:  cd pipeline && npm run crawl -- --limit 200"
        )
        return 0

    if args.dry_run:
        print("\n[dry-run] Would embed the following chunks:")
        for k in keys[:20]:
            print(f"  {k}")
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")
        return 0

    # Real AWS Bedrock for Titan embeddings (adaptive retry on throttle).
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
        config=BOTO_CONFIG,
    )

    chroma = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    print(f"Already embedded (any corpus): {len(existing_ids)}\n")

    # Load text + metadata for chunks not yet in ChromaDB.
    pending: list[dict] = []
    for key in keys:
        try:
            text, meta = read_chunk(s3, key)
            chunk_id = meta.get("chunkId") or key.removeprefix(prefix).removesuffix(".txt")
            if chunk_id in existing_ids:
                continue
            pending.append({"id": chunk_id, "text": text, "meta": meta})
        except Exception as err:
            print(f"  [SKIP] {key}: {err}")

    if not pending:
        print(f"Nothing new under {prefix} — all chunks already in ChromaDB.")
        return 0

    total = len(pending)
    print(f"Embedding {total} new chunks from {prefix} (workers={MAX_WORKERS})...")

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
                # ChromaDB rejects None values in metadata; strip them.
                clean_meta = {k: v for k, v in meta.items() if v is not None}
                collection.upsert(
                    ids=[chunk_id],
                    documents=[text],
                    embeddings=[vector],
                    metadatas=[clean_meta],
                )
                done += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.monotonic() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  {done}/{total}  ({rate:.1f}/s  ETA {eta:.0f}s)")
            except Exception as err:
                errors += 1
                print(f"  [ERROR] {futures[future]['id']}: {err}")

    elapsed = time.monotonic() - start
    print(f"\nDone. {done} embedded, {errors} errors in {elapsed:.0f}s.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} vectors.")
    print(f"Store path: {VECTORSTORE_PATH}")
    return 0


if __name__ == "__main__":
    if os.environ.get("ENV") != "local":
        print("Error: set ENV=local to ingest against LocalStack")
        sys.exit(1)
    sys.exit(main())
