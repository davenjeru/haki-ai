"""
Eval runner entrypoint.

Usage:
    uv run -m evals.run                 # LLM-judge only (fast, ~30 calls)
    uv run -m evals.run --ragas         # Also run RAGAS (heavy)
    uv run -m evals.run --limit 5       # First N cases (smoke test)
    uv run -m evals.run --category employment

Exits with status 0 always so CI wrappers can decide how to handle the
numeric score (e.g. the Phase 5c ``ci.yml`` workflow compares against a
baseline stored in the repo).
"""

from __future__ import annotations

import argparse
import sys

from clients import make_bedrock_runtime, make_cloudwatch
from app.config import load_config

from .llm_judge import judge
from .loader import load_golden_set
from .ragas_run import score_with_ragas
from .report import CaseScore, _aggregate_judge, _overall_mean, emit_cloudwatch_score, write_report
from .retrieval_metrics import format_retrieval_report, summarize_retrieval
from .runner import run_all


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Haki AI evaluation suite.")
    p.add_argument("--ragas", action="store_true", help="Also run RAGAS metrics (heavy).")
    p.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    p.add_argument(
        "--category",
        default=None,
        help="Restrict the run to one category (e.g. constitution, employment, land, criminal).",
    )
    p.add_argument(
        "--skip-cloudwatch",
        action="store_true",
        help="Don't emit the EvalScore metric (useful for local dev).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    config = load_config()

    cases = load_golden_set()
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("No golden cases selected; nothing to do.")
        return 0

    print(f"[evals] running {len(cases)} case(s) against {config.bedrock_model_id or '<no model>'}")
    results = run_all(cases, config=config)

    bedrock_runtime = make_bedrock_runtime(config)
    case_scores: list[CaseScore] = []
    for r in results:
        score = judge(
            question=r.case.question,
            language=r.case.language,
            candidate_answer=r.answer,
            candidate_citations=r.citations,
            reference_answer=r.case.reference_answer,
            expected_sources=r.case.expected_sources,
            retrieved_contexts=r.retrieved_contexts,
            bedrock_runtime=bedrock_runtime,
            model_id=config.bedrock_model_id,
        )
        case_scores.append(CaseScore(result=r, judge=score))

    ragas_scores = score_with_ragas(results) if args.ragas else None

    aggregate = _aggregate_judge([cs.judge for cs in case_scores])
    mean_score = _overall_mean(aggregate)

    retrieval = summarize_retrieval(results)

    out_path = write_report(case_scores, ragas=ragas_scores, retrieval=retrieval)
    print(f"[evals] report written → {out_path}")
    print(f"[evals] overall LLM-judge mean (0–5): {mean_score:.2f}")
    for axis, value in aggregate.items():
        print(f"        {axis:>25}: {value:.2f}")
    if ragas_scores:
        print("[evals] RAGAS:")
        for metric, value in ragas_scores.items():
            print(f"        {metric:>25}: {value:.3f}")
    print(f"[evals] {format_retrieval_report(retrieval)}")

    if not args.skip_cloudwatch:
        cloudwatch = make_cloudwatch(config)
        emit_cloudwatch_score(cloudwatch, mean_score)

    return 0


if __name__ == "__main__":
    sys.exit(main())
