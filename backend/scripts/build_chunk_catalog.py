"""
Build `processed-chunks/_catalog.json` from S3 chunks + metadata sidecars.

The Lambda cold-start loader (`rag.catalog`) prefers this single-object
catalog over listing + fetching each chunk individually; with ~1200
chunks that reduces cold-start S3 latency from ~2400 GETs to exactly 1.

Run manually (requires AWS creds with read/write on the data bucket):
    uv run python -m scripts.build_chunk_catalog \\
        --bucket haki-ai-data \\
        --prefix processed-chunks/

Or via the ingestion_trigger Lambda whenever new chunks land (future).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3

_MAX_WORKERS = 32


def _make_s3_client():
    """Builds an S3 client that honours ENV=local so the LocalStack
    bucket (populated during dev ingestion) can be rebuilt without
    touching real AWS."""
    if os.environ.get("ENV") == "local":
        host = os.environ.get("LOCALSTACK_HOSTNAME", "localhost")
        port = os.environ.get("EDGE_PORT", "4566")
        return boto3.client(
            "s3",
            endpoint_url=f"http://{host}:{port}",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
    return boto3.client("s3")


def _chunk_id_from_key(key: str, prefix: str) -> str:
    return key.removeprefix(prefix).removesuffix(".txt")


def _list_chunk_keys(s3, bucket: str, prefix: str) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if (
                key.endswith(".txt")
                and not key.endswith(".metadata.json")
                and not key.endswith(".complete")
            ):
                keys.append(key)
    return keys


def _read_chunk(s3, bucket: str, key: str, prefix: str) -> dict | None:
    try:
        text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        meta_raw = s3.get_object(
            Bucket=bucket, Key=key + ".metadata.json"
        )["Body"].read()
        meta = (json.loads(meta_raw) or {}).get("metadataAttributes", {})
        chunk_id = meta.get("chunkId") or _chunk_id_from_key(key, prefix)
        return {"chunkId": chunk_id, "text": text, "metadata": meta}
    except Exception as err:
        print(f"[build_catalog] skipping {key}: {err}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="haki-ai-data")
    parser.add_argument("--prefix", default="processed-chunks/")
    parser.add_argument(
        "--output-key",
        default=None,
        help="Output S3 key. Defaults to `<prefix>_catalog.json`.",
    )
    args = parser.parse_args()

    s3 = _make_s3_client()
    output_key = args.output_key or f"{args.prefix}_catalog.json"

    print(f"[build_catalog] listing s3://{args.bucket}/{args.prefix} ...")
    keys = _list_chunk_keys(s3, args.bucket, args.prefix)
    print(f"[build_catalog] {len(keys)} chunk text files found")
    if not keys:
        print("[build_catalog] nothing to do", file=sys.stderr)
        return 1

    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_read_chunk, s3, args.bucket, k, args.prefix): k for k in keys
        }
        for fut in as_completed(futures):
            entry = fut.result()
            if entry is not None:
                out.append(entry)
    out.sort(key=lambda c: c["chunkId"])
    print(f"[build_catalog] built catalog with {len(out)} entries")

    body = json.dumps(out, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=args.bucket,
        Key=output_key,
        Body=body,
        ContentType="application/json",
        CacheControl="no-cache",
    )
    print(
        f"[build_catalog] wrote s3://{args.bucket}/{output_key} "
        f"({len(body):,} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
