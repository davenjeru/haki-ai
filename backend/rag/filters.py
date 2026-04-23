"""
Retrieval-stage filters.

drop_toc(results):
    Excludes chunks flagged as table-of-contents / arrangement-of-sections.
    The primary signal is the `chunkType` metadata attribute set by the
    pipeline (`"toc" | "body"`). Older chunks that predate that attribute
    fall through a heuristic check on section + title so retroactive
    filtering still works until the pipeline is re-run.

drop_boilerplate(results):
    Excludes chunks whose `chunkType` is one of the generic-content
    classes — ``preamble`` (cover-page OCR garbage), ``short-title``
    (Section 1's "This Act may be cited as ...") and ``definitions``
    (the flat Interpretation list in Section 2). These chunks match
    almost any query via dense embedding because they contain every
    term defined in the Act, so they pollute the top-K for
    non-definitional questions. The heuristic fallback keeps retroactive
    filtering working for chunks ingested before the pipeline started
    tagging these categories.

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

# chunkType values that should be excluded from non-definitional retrieval.
# Kept as a frozenset so callers can extend it without touching filter
# logic (e.g. experimental `"schedule"` or `"footnote"` classes later).
_BOILERPLATE_TYPES: frozenset[str] = frozenset({"preamble", "short-title", "definitions"})

# Heuristics for retroactively classifying chunks that predate the
# chunkType attribute. Matched against (section, title) tuples after
# lowercasing. The chunker now tags these at ingest time, but we keep
# the fallback so stale corpora don't require an immediate re-chunk.
_BOILERPLATE_HEURISTICS: tuple[tuple[str, str], ...] = (
    ("preamble", ""),            # section="Preamble", any/blank title
    ("section 1", "short title"),
    ("section 2", "interpretation"),
    ("section 2", "definitions"),
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


def is_boilerplate(meta: dict) -> bool:
    """
    True if the *chunk metadata* looks like preamble / short-title /
    definitions. Accepts either the raw metadata dict (as stored on an
    ``EvalResult.retrieved_metadata`` entry) or a retrieval result that
    nests its metadata under ``"metadata"``.

    Exposed as a standalone helper so the audit tool (``evals.audit``)
    can classify top-K slots using the exact same rule the pipeline
    applies.
    """
    if not isinstance(meta, dict):
        return False
    # Accept both shapes: plain metadata dict, or a retrieval entry that
    # nests metadata under "metadata". Audit pulls the former out of
    # ``EvalResult.retrieved_metadata``; the live pipeline passes the latter.
    if "metadata" in meta and isinstance(meta["metadata"], dict):
        meta = meta["metadata"]

    chunk_type = (meta.get("chunkType") or "").strip().lower()
    if chunk_type in _BOILERPLATE_TYPES:
        return True

    section = (meta.get("section") or "").strip().lower()
    title = (meta.get("title") or "").strip().lower()
    for sec_needle, title_needle in _BOILERPLATE_HEURISTICS:
        if section == sec_needle and (not title_needle or title_needle in title):
            return True
    return False


def drop_toc(results: list[dict]) -> list[dict]:
    """Filters out chunks flagged as TOC or that heuristically look like one."""
    return [r for r in results if not _is_toc(r)]


def drop_boilerplate(results: list[dict]) -> list[dict]:
    """
    Filters out preamble / short-title / definitions chunks. These are
    extracted once at ingest time and tagged via the ``chunkType``
    metadata attribute; the heuristic fallback handles corpora that
    predate that attribute.
    """
    return [r for r in results if not is_boilerplate(r)]


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
