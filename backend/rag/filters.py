"""
Retrieval-stage filters.

drop_toc(results):
    Excludes chunks flagged as table-of-contents / arrangement-of-sections.
    The primary signal is the `chunkType` metadata attribute set by the
    pipeline (`"toc" | "body"`). Older chunks that predate that attribute
    fall through a heuristic check on section + title so retroactive
    filtering still works until the pipeline is re-run.

dedup_by_section(results):
    Collapses multiple chunks with the same (source, section) to the first
    (highest-scoring) one. `splitToSegments` can produce 2\u20133 sibling chunks
    per long section; for answer generation we only need the top-ranked
    sibling because they share the same surrounding context.
"""

from __future__ import annotations

_TOC_HEURISTIC_SNIPPETS = (
    "arrangement of sections",
    "table of contents",
    "arrangement of articles",
)


def _is_toc(entry: dict) -> bool:
    meta = entry.get("metadata") or {}
    chunk_type = (meta.get("chunkType") or "").strip().lower()
    if chunk_type == "toc":
        return True
    # Heuristic fallback for chunks ingested before chunkType existed.
    section = (meta.get("section") or "").strip().lower()
    title = (meta.get("title") or "").strip().lower()
    for needle in _TOC_HEURISTIC_SNIPPETS:
        if needle in section or needle in title:
            return True
    return False


def drop_toc(results: list[dict]) -> list[dict]:
    """Filters out chunks flagged as TOC or that heuristically look like one."""
    return [r for r in results if not _is_toc(r)]


def dedup_by_section(results: list[dict]) -> list[dict]:
    """
    Keeps the first occurrence (i.e. the highest-ranked) for each
    (source, section) key. Chunks missing either field are kept verbatim
    \u2014 we never silently drop ambiguous metadata.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in results:
        meta = r.get("metadata") or {}
        source = (meta.get("source") or "").strip()
        section = (meta.get("section") or "").strip()
        if not source or not section:
            out.append(r)
            continue
        key = f"{source}|{section}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
