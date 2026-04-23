"""
Retrieval audit — classify each golden case's top-5 into hit / noise /
rerank-loss so regressions in chunking or filtering are attributable.

The eval harness (``evals.run``) already produces per-case
``Recall@20`` / ``MRR@10``, but those are aggregates and hide *why* a
case missed. This module re-runs the pipeline for one category,
inspects each ``EvalResult.retrieved_metadata`` top-5, and prints a
triage table plus a matching markdown report.

Usage
-----
    uv run -m evals.audit --category land
    uv run -m evals.audit --category tenant --top-k 5

The output is designed to answer, for each case, exactly one question:
"is the expected section missing from the corpus, missing from the top-K
retrieval, polluted out by a boilerplate chunk, or reranked away?"

Failure classes
---------------
- ``hit``              \u2014 at least one ``expected_section`` found in top-K.
- ``noise-pollution``  \u2014 expected section missing AND \u2265 1 top-K slot is
                       occupied by a boilerplate chunk (Preamble / Short
                       title / Interpretation). The fix is chunk-hygiene
                       (plan step 6), not an embedding swap.
- ``rerank-loss``      \u2014 expected section missing AND no boilerplate in
                       top-K. Either the expected section is semantically
                       unreachable or rerank dropped it; the fix is the
                       structural rerank prior (step 7) or a better
                       embedding model (step 8).
- ``no-expected``      \u2014 case has empty ``expected_sections`` (refusal
                       or open-ended). Skipped from counts so aggregates
                       stay honest.

The classification uses the same boilerplate detector as
``rag.filters.drop_boilerplate`` so the audit stays in sync with what
the live pipeline filters.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import load_config
from rag.filters import is_boilerplate

from .loader import load_golden_set
from .runner import EvalResult, run_all


DEFAULT_TOP_K = 5
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


# ── Classification ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditRow:
    """One row per golden case \u2014 enough to diagnose why a case scored
    the way it did without re-reading the full markdown report."""

    case_id: str
    language: str
    expected_sections: list[str]
    retrieved_sections: list[str]       # e.g. "Section 5" | "Preamble"
    retrieved_chunk_types: list[str]    # e.g. "body" | "preamble" | "definitions"
    boilerplate_slots: int              # count of top-K that are boilerplate
    matched: list[str]                  # expected sections actually found in top-K
    failure_mode: str                   # "hit" | "noise-pollution" | "rerank-loss" | "no-expected"


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _section_matches(expected: str, meta: dict) -> bool:
    """
    String-matches ``expected`` (e.g. ``"Section 54"``) against the
    retrieved chunk's ``section`` metadata attribute. Falls back to the
    chunk id (``"land-act-2012-...-section-54-1"``) so pre-metadata
    chunks still score.
    """
    needle = _normalize(expected)
    if not needle:
        return False
    for key in ("section", "chunkId", "title", "chapter"):
        hay = _normalize(meta.get(key) or "")
        if needle in hay:
            return True
    return False


def classify(result: EvalResult, *, top_k: int = DEFAULT_TOP_K) -> AuditRow:
    """Builds an :class:`AuditRow` from a single ``EvalResult``."""
    meta_top_k = result.retrieved_metadata[:top_k]
    retrieved_sections = [str(m.get("section") or "?") for m in meta_top_k]
    retrieved_chunk_types = [str(m.get("chunkType") or "body") for m in meta_top_k]
    boilerplate_slots = sum(1 for m in meta_top_k if is_boilerplate(m))

    expected = list(result.case.expected_sections or [])
    if not expected:
        return AuditRow(
            case_id=result.case.id,
            language=result.case.language,
            expected_sections=[],
            retrieved_sections=retrieved_sections,
            retrieved_chunk_types=retrieved_chunk_types,
            boilerplate_slots=boilerplate_slots,
            matched=[],
            failure_mode="no-expected",
        )

    matched = [
        section
        for section in expected
        if any(_section_matches(section, m) for m in meta_top_k)
    ]

    if matched:
        mode = "hit"
    elif boilerplate_slots > 0:
        mode = "noise-pollution"
    else:
        mode = "rerank-loss"

    return AuditRow(
        case_id=result.case.id,
        language=result.case.language,
        expected_sections=expected,
        retrieved_sections=retrieved_sections,
        retrieved_chunk_types=retrieved_chunk_types,
        boilerplate_slots=boilerplate_slots,
        matched=matched,
        failure_mode=mode,
    )


# ── Rendering ─────────────────────────────────────────────────────────────────


_FAILURE_EMOJI = {
    "hit": "PASS",
    "noise-pollution": "NOISE",
    "rerank-loss": "MISS",
    "no-expected": "n/a",
}


def _fmt_row(row: AuditRow) -> str:
    expected = ", ".join(row.expected_sections) or "(none)"
    retrieved = ", ".join(row.retrieved_sections) or "(none)"
    matched = ", ".join(row.matched) or "-"
    return (
        f"  {row.case_id:<16} {row.language:<8} "
        f"{_FAILURE_EMOJI[row.failure_mode]:<6} "
        f"boilerplate={row.boilerplate_slots:<1} "
        f"matched=[{matched}] "
        f"expected=[{expected}] "
        f"retrieved=[{retrieved}]"
    )


def _summarise(rows: list[AuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {"hit": 0, "noise-pollution": 0, "rerank-loss": 0, "no-expected": 0}
    for row in rows:
        counts[row.failure_mode] += 1
    return counts


def format_report(
    category: str | None,
    rows: list[AuditRow],
    *,
    top_k: int,
) -> str:
    """Markdown report suitable for committing alongside eval runs."""
    ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    counts = _summarise(rows)
    scorable = len(rows) - counts["no-expected"]
    hit_rate = counts["hit"] / scorable if scorable else 0.0
    noise_rate = counts["noise-pollution"] / scorable if scorable else 0.0

    lines: list[str] = [
        f"# Retrieval audit \u2014 {category or 'all'} ({ts})",
        "",
        f"- Cases: {len(rows)}  |  Scorable: {scorable}  |  Top-K: {top_k}",
        f"- hit={counts['hit']}  noise-pollution={counts['noise-pollution']}  "
        f"rerank-loss={counts['rerank-loss']}  no-expected={counts['no-expected']}",
        f"- hit-rate={hit_rate:.1%}  noise-rate={noise_rate:.1%}",
        "",
        "| Case | Lang | Status | Boilerplate | Matched | Expected | Retrieved (top-K section/chunkType) |",
        "|------|------|--------|-------------|---------|----------|--------------------------------------|",
    ]
    for row in rows:
        retrieved_cells = " \u00b7 ".join(
            f"{sec} ({ct})"
            for sec, ct in zip(row.retrieved_sections, row.retrieved_chunk_types)
        ) or "(none)"
        lines.append(
            f"| `{row.case_id}` "
            f"| {row.language} "
            f"| {row.failure_mode} "
            f"| {row.boilerplate_slots}/{top_k} "
            f"| {', '.join(row.matched) or '\u2014'} "
            f"| {', '.join(row.expected_sections) or '\u2014'} "
            f"| {retrieved_cells} |"
        )
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-case retrieval audit for a golden-set category.",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Filter golden set to one category (e.g. land, tenant, constitution).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many retrieved chunks to inspect per case (default {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing the markdown report to disk.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    config = load_config()

    cases = load_golden_set()
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if not cases:
        print("No golden cases matched \u2014 nothing to audit.")
        return 0

    label = args.category or "all"
    print(f"[audit] running {len(cases)} case(s) in category={label}")
    results = run_all(cases, config=config)

    rows = [classify(r, top_k=args.top_k) for r in results]

    print()
    print(f"Retrieval audit \u2014 {label} \u2014 top-{args.top_k}")
    print("-" * 80)
    for row in rows:
        print(_fmt_row(row))
    print("-" * 80)
    counts = _summarise(rows)
    scorable = len(rows) - counts["no-expected"]
    if scorable:
        print(
            f"  hit={counts['hit']}/{scorable}  "
            f"noise-pollution={counts['noise-pollution']}/{scorable}  "
            f"rerank-loss={counts['rerank-loss']}/{scorable}  "
            f"(skipped {counts['no-expected']} no-expected)"
        )
    else:
        print("  no scorable cases")

    if not args.no_write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        out_path = REPORTS_DIR / f"audit-{label}-{ts}.md"
        out_path.write_text(format_report(args.category, rows, top_k=args.top_k), encoding="utf-8")
        print(f"[audit] markdown report written \u2192 {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
