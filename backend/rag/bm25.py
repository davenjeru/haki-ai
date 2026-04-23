"""
In-memory BM25 retriever.

Wraps the `rank_bm25` package with a tiny compatibility layer that mirrors
the `DenseRetriever.retrieve` interface so the hybrid pipeline can treat
sparse and dense retrieval symmetrically.

Tokenisation is intentionally simple \u2014 lowercase alphanumeric runs plus
short statutory-style tokens like \u201c40\u201d or \u201c12A\u201d preserved verbatim. No
Swahili stop-word list; keeping stop words in actually helps BM25 on
short queries like \u201cmy rights on probation\u201d.

Index is built lazily at the first `retrieve` call from the catalog and
cached at module scope. Call `reset_index()` to force a rebuild (tests).
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

_lock = threading.Lock()
_cached: dict[str, tuple["BM25Okapi", list[dict]]] = {}


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenisation. Keeps statutory numbers intact."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def _build_index(catalog: list[dict]) -> tuple["BM25Okapi", list[dict]]:
    """Builds a BM25Okapi over the catalog\u2019s chunk texts."""
    tokenised = [tokenize(c["text"]) for c in catalog]
    # `rank_bm25` explodes on empty corpora; guard and return a sentinel.
    if not tokenised or not any(tokenised):
        return BM25Okapi([["__empty__"]]), list(catalog)
    return BM25Okapi(tokenised), list(catalog)


def get_index(catalog: list[dict], cache_key: str = "default") -> tuple["BM25Okapi", list[dict]]:
    """
    Returns a cached (BM25, catalog) pair for the given cache_key, building
    lazily on first access. The cache_key lets us hold separate indexes for
    different catalogs without clashing.
    """
    cached = _cached.get(cache_key)
    if cached is not None:
        return cached
    with _lock:
        cached = _cached.get(cache_key)
        if cached is not None:
            return cached
        cached = _build_index(catalog)
        _cached[cache_key] = cached
        return cached


def reset_index() -> None:
    """Drops any cached indexes. Tests use this for isolation."""
    with _lock:
        _cached.clear()


def retrieve(
    query: str,
    catalog: list[dict],
    *,
    top_k: int = 30,
    cache_key: str = "default",
    metadata_filter: dict | None = None,
) -> list[dict]:
    """
    Runs BM25 over the catalog and returns the top_k matches, each shaped
    like the dense retrieval output so downstream code stays symmetric:

        {
          "content":  {"text": str},
          "metadata": {...},
          "location": {"type": "S3", "s3Location": {"uri": "..."}},
          "score":    float,
        }

    metadata_filter is an AND-of-equals dict \u2014 e.g.
    `{"source": "Employment Act 2007"}`. Entries whose metadata does not
    match every key/value are excluded before scoring so the BM25
    normalisation reflects the filtered corpus, not the whole corpus.
    """
    query = (query or "").strip()
    if not query or not catalog:
        return []

    filtered_catalog = _apply_filter(catalog, metadata_filter)
    if not filtered_catalog:
        return []

    # Re-index when a filter is applied (filtered corpus \u2192 different IDFs).
    if metadata_filter:
        bm25, cat = _build_index(filtered_catalog)
    else:
        bm25, cat = get_index(filtered_catalog, cache_key=cache_key)

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(
        range(len(cat)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results: list[dict] = []
    for idx in ranked:
        entry = cat[idx]
        meta = dict(entry.get("metadata") or {})
        chunk_id = entry["chunkId"]
        results.append({
            "content": {"text": entry.get("text", "")},
            "metadata": meta,
            "location": {
                "type": "S3",
                "s3Location": {
                    "uri": f"s3://haki-ai-data/processed-chunks/{chunk_id}.txt"
                },
            },
            "score": float(scores[idx]),
        })
    return results


def _apply_filter(catalog: list[dict], metadata_filter: dict | None) -> list[dict]:
    if not metadata_filter:
        return catalog
    out: list[dict] = []
    for entry in catalog:
        meta = entry.get("metadata") or {}
        if all(meta.get(k) == v for k, v in metadata_filter.items()):
            out.append(entry)
    return out
