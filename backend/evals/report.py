"""
Report writer + CloudWatch metric emitter.

Produces two artefacts per eval run:

1. ``evals/reports/{timestamp}.md`` — human-readable markdown with a
   summary table, per-category aggregates, and per-question diffs
   (question, reference, candidate, citations, scores, judge notes).

2. A single CloudWatch custom metric named ``EvalScore`` in the
   ``HakiAI`` namespace so the existing dashboard + alarms can catch
   regressions without any new wiring. The metric value is the
   LLM-judge mean across axes across all cases, on a 0–5 scale.

The report format is chosen for git-diffability: one row per axis in
the summary, one section per question underneath, no base64 blobs.
That way a reviewer can ``git diff evals/reports/`` between two runs
and immediately see which questions regressed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .llm_judge import AXES, JudgeScore
from .runner import EvalResult


_REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


@dataclass
class CaseScore:
    result: EvalResult
    judge: JudgeScore


def _aggregate_judge(scores: list[JudgeScore]) -> dict[str, float]:
    """Mean score per axis across the list. Empty list → zeros."""
    if not scores:
        return {axis: 0.0 for axis in AXES}
    return {
        axis: sum(getattr(s, axis) for s in scores) / len(scores)
        for axis in AXES
    }


def _overall_mean(agg: dict[str, float]) -> float:
    return sum(agg.values()) / len(agg) if agg else 0.0


def _fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _category_breakdown(case_scores: list[CaseScore]) -> dict[str, dict[str, float]]:
    by_cat: dict[str, list[JudgeScore]] = {}
    for cs in case_scores:
        by_cat.setdefault(cs.result.case.category, []).append(cs.judge)
    return {cat: _aggregate_judge(scores) for cat, scores in by_cat.items()}


def write_report(
    case_scores: list[CaseScore],
    *,
    ragas: dict[str, Any] | None = None,
    out_dir: str | None = None,
) -> str:
    """Writes the markdown report and returns the full output path."""
    out_dir = out_dir or _REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = os.path.join(out_dir, f"{stamp}.md")

    overall = _aggregate_judge([cs.judge for cs in case_scores])
    mean_score = _overall_mean(overall)
    category_scores = _category_breakdown(case_scores)

    lines: list[str] = []
    lines.append(f"# Haki AI — Evaluation Report")
    lines.append("")
    lines.append(f"Run: `{stamp}`  •  Cases: {len(case_scores)}  •  "
                 f"Overall (0–5): **{_fmt_num(mean_score)}**")
    lines.append("")

    lines.append("## Summary (LLM-judge)")
    lines.append("")
    lines.append("| Axis | Score (0–5) |")
    lines.append("| --- | --- |")
    for axis in AXES:
        lines.append(f"| {axis} | {_fmt_num(overall[axis])} |")
    lines.append("")

    if ragas:
        lines.append("## RAGAS metrics")
        lines.append("")
        lines.append("| Metric | Score (0–1) |")
        lines.append("| --- | --- |")
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            if name in ragas:
                lines.append(f"| {name} | {_fmt_num(ragas[name], 3)} |")
        lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    lines.append("| Category | accuracy | citation | tone | language |")
    lines.append("| --- | --- | --- | --- | --- |")
    for cat in sorted(category_scores):
        agg = category_scores[cat]
        lines.append(
            f"| {cat} | {_fmt_num(agg['accuracy'])} "
            f"| {_fmt_num(agg['citation_correctness'])} "
            f"| {_fmt_num(agg['tone'])} "
            f"| {_fmt_num(agg['language_appropriateness'])} |"
        )
    lines.append("")

    lines.append("## Per-question detail")
    lines.append("")
    for cs in case_scores:
        case = cs.result.case
        j = cs.judge
        lines.append(f"### `{case.id}` — {case.category} — {case.language}")
        lines.append("")
        lines.append(f"**Q:** {case.question}")
        lines.append("")
        lines.append("**Reference:**")
        lines.append("")
        lines.append(f"> {case.reference_answer}")
        lines.append("")
        lines.append("**Candidate:**")
        lines.append("")
        lines.append(f"> {cs.result.answer or '(empty)'}")
        lines.append("")
        if cs.result.citations:
            lines.append("**Citations:**")
            for c in cs.result.citations:
                lines.append(f"- {c.get('source', '?')} — {c.get('section', '?')}")
            lines.append("")
        lines.append(
            f"**Scores:** accuracy={j.accuracy}/5, "
            f"citation_correctness={j.citation_correctness}/5, "
            f"tone={j.tone}/5, "
            f"language_appropriateness={j.language_appropriateness}/5"
        )
        if j.notes:
            lines.append("")
            lines.append(f"_Notes: {j.notes}_")
        if cs.result.blocked:
            lines.append("")
            lines.append("_Guardrail blocked this response._")
        if cs.result.error:
            lines.append("")
            lines.append(f"_Error: {cs.result.error}_")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def emit_cloudwatch_score(cloudwatch, mean_score: float) -> None:
    """Publishes a single EvalScore metric so existing dashboards pick it up."""
    try:
        cloudwatch.put_metric_data(
            Namespace="HakiAI",
            MetricData=[{
                "MetricName": "EvalScore",
                "Value": float(mean_score),
                "Unit": "None",
            }],
        )
    except Exception as err:
        print(f"[report] CloudWatch put_metric_data failed: {err}")
