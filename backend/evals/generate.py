"""
CLI entrypoint for synthesising a RAGAS test-set.

Usage:
    uv run -m evals.generate --size 50 --output generated_set.jsonl
    uv run -m evals.generate --size 10 --max-cost 1.00   # hard cap
    uv run -m evals.generate --dry-run                   # pre-flight only

The CLI bootstraps LangSmith (same path the app uses), runs a pre-flight
cost estimate, refuses to start if the estimate exceeds ``--max-cost``,
runs the generator, writes the generated cases to JSONL, and writes a
sibling ``*.cost.md`` with the authoritative token + cost aggregation
pulled from the LangSmith trace (falling back to the in-process
``BudgetTracker`` when tracing is off).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from app.config import load_config
from observability.tracing import bootstrap_langsmith

from .generation_cost import (
    BudgetExceededError,
    aggregate_from_langsmith,
    estimate_generation_cost,
)

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT = "generated_set.jsonl"
_DEFAULT_SUBSAMPLE = 200
_DEFAULT_SIZE = 50


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesise a bilingual RAG eval set via RAGAS.",
    )
    p.add_argument(
        "--size",
        type=int,
        default=_DEFAULT_SIZE,
        help=f"Number of questions to generate (default: {_DEFAULT_SIZE}).",
    )
    p.add_argument(
        "--subsample",
        type=int,
        default=_DEFAULT_SUBSAMPLE,
        help=(
            "Random stratified subsample of the corpus fed into RAGAS. "
            f"Smaller = cheaper KG extraction (default: {_DEFAULT_SUBSAMPLE})."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help="Output JSONL filename (written under backend/evals/).",
    )
    p.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Hard budget cap in USD. Pre-flight refuses to start when the "
            "estimate exceeds this; mid-run the BudgetTracker aborts when "
            "the running actual cost exceeds it."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the stratified subsample (default: 0).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the pre-flight cost estimate and exit without calling "
            "Bedrock. Useful for sanity-checking --size / --subsample."
        ),
    )
    p.add_argument(
        "--project",
        default=os.environ.get("LANGSMITH_PROJECT", "haki-ai"),
        help="LangSmith project name to read the trace from (default: haki-ai).",
    )
    return p.parse_args(argv)


def _resolve_output_path(filename: str) -> Path:
    """Writes to ``backend/evals/<filename>`` unless an absolute path is
    given. Keeps generated files next to the curated golden set.
    """
    if os.path.isabs(filename):
        return Path(filename)
    return Path(__file__).parent / filename


def _preflight(args: argparse.Namespace, config) -> float:
    """Computes, prints, and optionally enforces the pre-flight estimate."""
    estimate = estimate_generation_cost(
        num_chunks=args.subsample,
        testset_size=args.size,
        llm_model_id=config.bedrock_model_id,
        embed_model_id=config.embedding_model_id,
    )
    print(
        f"[gen] pre-flight estimate: ~${estimate:.2f} "
        f"(size={args.size}, subsample={args.subsample}, "
        f"llm={config.bedrock_model_id or '<unset>'})"
    )
    if args.max_cost is not None and estimate > args.max_cost:
        print(
            f"[gen] aborting: estimate ${estimate:.2f} exceeds "
            f"--max-cost ${args.max_cost:.2f}. Lower --size or --subsample, "
            f"or raise --max-cost."
        )
        sys.exit(2)
    return estimate


def _write_jsonl(path: Path, cases: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(case.to_jsonl())
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv or sys.argv[1:])
    config = load_config()

    bootstrap_langsmith(config)

    if config.bedrock_model_id == "":
        print("[gen] BEDROCK_MODEL_ID is unset in .env — aborting.")
        return 2

    estimate = _preflight(args, config)
    if args.dry_run:
        print("[gen] --dry-run set; exiting before calling Bedrock.")
        return 0

    # Lazy import so the help text + dry-run path don't force `ragas`
    # + `langchain-aws` to be installed.
    from .testset_generator import generate_testset

    print(
        f"[gen] generating {args.size} questions "
        f"(subsample={args.subsample}, max_cost={args.max_cost})"
    )

    try:
        result = generate_testset(
            config=config,
            size=args.size,
            subsample_size=args.subsample,
            max_cost=args.max_cost,
            seed=args.seed,
        )
    except BudgetExceededError as err:
        print(f"[gen] {err}")
        # Still report what we spent before aborting.
        report = aggregate_from_langsmith(
            trace_id="",
            project_name=args.project,
            fallback_tracker=None,
        )
        report.by_model = {}  # force fallback-to-local path below
        print(f"[gen] actual cost before abort: ${0.0:.2f} (no cases written)")
        return 3

    out_path = _resolve_output_path(args.output)
    _write_jsonl(out_path, result.cases)
    print(f"[gen] wrote {len(result.cases)} cases → {out_path}")

    # Authoritative cost aggregation via LangSmith trace, with local
    # tracker as a fallback so the report is never blank.
    cost_report = aggregate_from_langsmith(
        trace_id=result.trace_id or "",
        project_name=args.project,
        fallback_tracker=result.tracker,
    )
    cost_md = cost_report.to_markdown(title=f"Testset generation ({out_path.name})")
    cost_path = out_path.with_suffix(".cost.md")
    cost_path.write_text(cost_md, encoding="utf-8")
    print(f"[gen] cost report → {cost_path}")
    print(
        f"[gen] total cost: ${cost_report.total_cost:.4f} "
        f"({cost_report.total_tokens:,} tokens, source={cost_report.source})"
    )
    if cost_report.trace_url:
        print(f"[gen] trace: {cost_report.trace_url}")

    _print_summary(result.cases)
    return 0


def _print_summary(cases: list) -> None:
    """Category + language counts so the operator can eyeball the mix."""
    from collections import Counter

    cats = Counter(c.category for c in cases)
    langs = Counter(c.language for c in cases)
    print("[gen] category breakdown:")
    for cat, n in cats.most_common():
        print(f"       {cat:>15}: {n}")
    print("[gen] language breakdown:")
    for lang, n in langs.most_common():
        print(f"       {lang:>15}: {n}")


if __name__ == "__main__":
    sys.exit(main())
