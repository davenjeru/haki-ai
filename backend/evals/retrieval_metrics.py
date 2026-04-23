"""
Retrieval-only metrics — Recall@k and MRR@k.

These exist because the Legal RAG Bench (arXiv 2603.01710, March 2026) and
Legal-DC (arXiv 2603.11772) findings both land on the same conclusion for
legal RAG: *retrieval quality dominates end-to-end performance*, and many
apparent "hallucinations" are retrieval failures in disguise. That means a
single aggregate RAGAS or LLM-judge number hides where the system actually
breaks — so we decompose.

We compare the top-K reranked chunks against each case's
``expected_sections`` (e.g. ``["Article 174"]``, ``["Section 35"]``). A
match is a substring hit in either the chunk text or the chunk metadata's
``section`` / ``chapter`` field — lenient enough to survive different
formatting conventions across statutes ("Section 40" vs "section 40"
vs a chunk body that repeats the section header).

Report stratified by language (english / swahili / mixed) so the known
Swahili weakness doesn't disappear into an aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .runner import EvalResult


# Default cut-offs picked to match the Legal-DC paper (Recall@20, MRR@5)
# and the pipeline's own defaults (fuse=20, rerank=5).
DEFAULT_RECALL_K = 20
DEFAULT_MRR_K = 10


def _haystack(text: str, meta: dict) -> str:
    """Joins chunk text + filterable metadata into a single lowercased blob."""
    parts: list[str] = [text]
    for key in ("section", "chapter", "title", "chunkId"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts).lower()


def _matches(needle: str, haystack: str) -> bool:
    return needle.strip().lower() in haystack if needle else False


def chunk_recall_at_k(
    result: EvalResult,
    *,
    k: int = DEFAULT_RECALL_K,
) -> float | None:
    """
    Fraction of the case's ``expected_sections`` that appear in the top-K
    retrieved chunks. Returns ``None`` when the case has no expected
    sections (refusal rows, open-ended questions) so aggregates can skip it.
    """
    expected = result.case.expected_sections or []
    if not expected:
        return None

    texts = result.retrieved_contexts[:k]
    metas = result.retrieved_metadata[:k]
    haystacks = [_haystack(t, m) for t, m in zip(texts, metas)]

    hits = 0
    for section in expected:
        if any(_matches(section, hay) for hay in haystacks):
            hits += 1
    return hits / len(expected)


def mrr_at_k(
    result: EvalResult,
    *,
    k: int = DEFAULT_MRR_K,
) -> float | None:
    """
    Reciprocal rank of the *first* retrieved chunk that matches any of the
    case's ``expected_sections``. Returns ``None`` for cases without
    expected sections; returns ``0.0`` when no match is found inside the
    top-K window so aggregates correctly punish misses.
    """
    expected = result.case.expected_sections or []
    if not expected:
        return None

    texts = result.retrieved_contexts[:k]
    metas = result.retrieved_metadata[:k]

    for idx, (text, meta) in enumerate(zip(texts, metas), start=1):
        hay = _haystack(text, meta)
        if any(_matches(section, hay) for section in expected):
            return 1.0 / idx
    return 0.0


@dataclass(frozen=True)
class RetrievalReport:
    """
    Aggregate retrieval scores. ``by_language`` splits out english /
    swahili / mixed so weak spots aren't averaged away.
    """
    recall_at_k: float
    mrr_at_k: float
    recall_k: int
    mrr_k: int
    case_count: int
    by_language: dict[str, dict[str, float | int]]


def _mean(values: Iterable[float | None]) -> float:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else 0.0


def summarize_retrieval(
    results: list[EvalResult],
    *,
    recall_k: int = DEFAULT_RECALL_K,
    mrr_k: int = DEFAULT_MRR_K,
) -> RetrievalReport:
    """Computes Recall@k / MRR@k overall and stratified by language."""
    scored: list[tuple[EvalResult, float | None, float | None]] = [
        (r, chunk_recall_at_k(r, k=recall_k), mrr_at_k(r, k=mrr_k))
        for r in results
    ]

    overall_recall = _mean(recall for _, recall, _ in scored)
    overall_mrr = _mean(mrr for _, _, mrr in scored)

    by_language: dict[str, dict[str, float | int]] = {}
    for language in ("english", "swahili", "mixed"):
        subset = [
            (recall, mrr)
            for r, recall, mrr in scored
            if r.case.language == language
        ]
        countable = [(rc, mr) for rc, mr in subset if rc is not None]
        if not countable:
            continue
        by_language[language] = {
            "recall_at_k": sum(rc for rc, _ in countable) / len(countable),
            "mrr_at_k": sum(mr for _, mr in countable) / len(countable),
            "count": len(countable),
        }

    countable = [(rc, mr) for _, rc, mr in scored if rc is not None]
    return RetrievalReport(
        recall_at_k=overall_recall,
        mrr_at_k=overall_mrr,
        recall_k=recall_k,
        mrr_k=mrr_k,
        case_count=len(countable),
        by_language=by_language,
    )


def format_retrieval_report(report: RetrievalReport) -> str:
    """Human-readable text block suitable for CLI output and markdown reports."""
    lines = [
        f"Retrieval metrics ({report.case_count} scorable cases)",
        f"  Recall@{report.recall_k:<3}: {report.recall_at_k:.3f}",
        f"  MRR@{report.mrr_k:<3}:    {report.mrr_at_k:.3f}",
    ]
    if report.by_language:
        lines.append("  By language:")
        for lang, stats in report.by_language.items():
            lines.append(
                f"    {lang:<8} n={int(stats['count']):<3} "
                f"Recall@{report.recall_k}={stats['recall_at_k']:.3f}  "
                f"MRR@{report.mrr_k}={stats['mrr_at_k']:.3f}"
            )
    return "\n".join(lines)
