"""
Reciprocal Rank Fusion.

Merges multiple ranked lists (dense + BM25 \u00d7 each query variant) into a
single ranked list. Scoring follows the canonical RRF formula:

    score(doc) = \u03a3 over lists L: 1 / (k + rank_L(doc))

where `rank_L(doc)` is the 1-indexed position of the doc in list L (if
present). Documents absent from a list contribute nothing from that list.

Identity is based on the `chunkId` metadata field \u2014 or the S3 URI as a
fallback \u2014 so the same chunk retrieved twice by two query variants is
correctly merged into one entry with summed score.

`k = 60` is the value from the original Cormack et al. paper; it damps
the contribution of ranks beyond ~60 without over-weighting the top hit.
"""

from __future__ import annotations

from typing import Iterable

_RRF_K = 60


def _identity(result: dict) -> str:
    meta = result.get("metadata") or {}
    chunk_id = meta.get("chunkId")
    if chunk_id:
        return f"chunk:{chunk_id}"
    location = (result.get("location") or {}).get("s3Location", {})
    uri = location.get("uri")
    if uri:
        return f"uri:{uri}"
    return f"text:{(result.get('content') or {}).get('text', '')[:64]}"


def fuse(lists: Iterable[list[dict]], *, k: int = _RRF_K) -> list[dict]:
    """
    Fuses any number of ranked result lists using RRF.

    Input lists may overlap in content or be empty. Output is sorted by
    descending RRF score; each entry is the first dict seen for that
    identity, with `score` overwritten with the fused RRF value (and
    `denseScore` / `bm25Score` preserved under `componentScores` when the
    caller tagged them).
    """
    scores: dict[str, float] = {}
    exemplar: dict[str, dict] = {}
    component_scores: dict[str, list[tuple[str, float]]] = {}

    for ranked in lists:
        for rank, result in enumerate(ranked, start=1):
            ident = _identity(result)
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            if ident not in exemplar:
                exemplar[ident] = result
            src_tag = str(result.get("retriever", "unknown"))
            component_scores.setdefault(ident, []).append(
                (src_tag, float(result.get("score", 0.0)))
            )

    merged: list[dict] = []
    for ident, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        # Use the exemplar dict but overwrite score with the fused value.
        out = dict(exemplar[ident])
        out["score"] = scores[ident]
        out["componentScores"] = component_scores[ident]
        merged.append(out)
    return merged
