"""
Chunk catalog loader.

The BM25 retriever and the TOC filter both need direct access to the corpus
metadata (id, text, attributes). This module loads that once from S3 at
cold start and caches it at module scope so subsequent invocations reuse it.

Catalog shape (list of dicts, order-stable):
    [
      {
        "chunkId":  "employment-act-2007-part-iii-section-40",
        "text":     "An employer who terminates...",
        "metadata": {
            "source": "Employment Act 2007",
            "chapter": "Part III \u2014 Termination of Contract",
            "section": "Section 40",
            "title":   "Termination of employment",
            "chunkType": "body",
            "pageImageKey": "page-images/employment-act-2007/page-40.pdf",
            ...
        }
      },
      ...
    ]

Loading strategy (fast path \u2192 fallback):
  1. Fast path   \u2014 read one S3 object `processed-chunks/_catalog.json`
                   produced by the pipeline (future optimisation).
  2. Fallback    \u2014 list `processed-chunks/*.txt` and read each chunk plus
                   its `.metadata.json` sidecar in a ThreadPool. With ~1200
                   chunks this takes ~1\u20133 s on a warm Lambda.

The catalog is loaded lazily on first access and memoised process-wide.
Callers can force a reload via `reset_catalog()` (used by tests).
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

_CHUNKS_PREFIX = "processed-chunks/"
_CATALOG_KEY = f"{_CHUNKS_PREFIX}_catalog.json"
_MAX_WORKERS = 16

_lock = threading.Lock()
_cached: dict[tuple[str, str], list[dict]] = {}


def _chunk_id_from_key(key: str) -> str:
    return key.removeprefix(_CHUNKS_PREFIX).removesuffix(".txt")


def _load_from_catalog_file(s3_client, bucket: str) -> list[dict] | None:
    """Returns the prebuilt catalog if present, else None."""
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=_CATALOG_KEY)
        raw = obj["Body"].read()
        payload = json.loads(raw)
        if isinstance(payload, list) and all(
            isinstance(c, dict) and "chunkId" in c and "text" in c for c in payload
        ):
            return payload
    except Exception:
        return None
    return None


def _list_chunk_keys(s3_client, bucket: str) -> list[str]:
    """Lists all `.txt` chunk keys under processed-chunks/, excluding sidecars."""
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=_CHUNKS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if (
                key.endswith(".txt")
                and not key.endswith(".metadata.json")
                and not key.endswith(".complete")
            ):
                keys.append(key)
    return keys


def _read_chunk(s3_client, bucket: str, key: str) -> dict | None:
    """Reads the chunk body + sidecar metadata. Returns None on error."""
    try:
        text = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
        meta_raw = s3_client.get_object(
            Bucket=bucket, Key=key + ".metadata.json"
        )["Body"].read()
        meta = (json.loads(meta_raw) or {}).get("metadataAttributes", {})
        chunk_id = meta.get("chunkId") or _chunk_id_from_key(key)
        return {"chunkId": chunk_id, "text": text, "metadata": meta}
    except Exception as err:
        print(f"[catalog] skipping {key}: {err}")
        return None


def _load_from_listing(s3_client, bucket: str) -> list[dict]:
    keys = _list_chunk_keys(s3_client, bucket)
    if not keys:
        return []
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {ex.submit(_read_chunk, s3_client, bucket, k): k for k in keys}
        for fut in as_completed(futures):
            entry = fut.result()
            if entry is not None:
                out.append(entry)
    # Stable order (by chunkId) so BM25 scoring is reproducible.
    out.sort(key=lambda c: c["chunkId"])
    return out


def get_catalog(s3_client, bucket: str) -> list[dict]:
    """
    Returns the chunk catalog, loading from S3 on first call and caching
    the result. Safe to call concurrently \u2014 a lock guards the load.
    """
    cache_key = (bucket, _CHUNKS_PREFIX)
    cached = _cached.get(cache_key)
    if cached is not None:
        return cached

    with _lock:
        cached = _cached.get(cache_key)
        if cached is not None:
            return cached

        catalog = _load_from_catalog_file(s3_client, bucket)
        if catalog is None:
            catalog = _load_from_listing(s3_client, bucket)
        _cached[cache_key] = catalog
        return catalog


def reset_catalog() -> None:
    """Drops the cached catalog. Used by tests to isolate runs."""
    with _lock:
        _cached.clear()


def set_catalog(catalog: list[dict], bucket: str = "haki-ai-data") -> None:
    """
    Injects a pre-built catalog. Used by tests that want to exercise BM25
    behaviour without touching S3.
    """
    with _lock:
        _cached[(bucket, _CHUNKS_PREFIX)] = list(catalog)
