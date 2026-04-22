"""
LLM-as-judge scorer.

Compares a candidate answer to a reference answer on four 0–5 integer
axes using Claude on Bedrock:

    - accuracy
    - citation_correctness
    - tone
    - language_appropriateness

The judge prompt is ``prompts.JUDGE_PROMPT``. We use the same model we
use for generation so the judge is at parity with production; in a
future iteration we can swap to a larger model (Sonnet) via the
``JUDGE_MODEL_ID`` env var to reduce judge/generator overlap bias.

Failure semantics: any Bedrock error is caught and returns a score of 0
on every axis with the error message in ``notes``. The runner keeps
going so one flaky request does not destroy the whole eval run.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from prompts import JUDGE_PROMPT


AXES = ("accuracy", "citation_correctness", "tone", "language_appropriateness")


@dataclass
class JudgeScore:
    accuracy: int = 0
    citation_correctness: int = 0
    tone: int = 0
    language_appropriateness: int = 0
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "citation_correctness": self.citation_correctness,
            "tone": self.tone,
            "language_appropriateness": self.language_appropriateness,
            "notes": self.notes,
        }

    def mean(self) -> float:
        return sum(getattr(self, axis) for axis in AXES) / len(AXES)


def _parse_judge_response(text: str) -> JudgeScore:
    """Extracts a JSON score from the model's reply. Missing axes = 0."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return JudgeScore(notes="parse_failure: no JSON object found")
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as err:
        return JudgeScore(notes=f"parse_failure: {err}")

    def _clamp(value) -> int:
        try:
            return max(0, min(5, int(value)))
        except (TypeError, ValueError):
            return 0

    return JudgeScore(
        accuracy=_clamp(obj.get("accuracy")),
        citation_correctness=_clamp(obj.get("citation_correctness")),
        tone=_clamp(obj.get("tone")),
        language_appropriateness=_clamp(obj.get("language_appropriateness")),
        notes=str(obj.get("notes") or ""),
    )


def _build_user_message(
    *,
    question: str,
    language: str,
    candidate_answer: str,
    candidate_citations: list[dict],
    reference_answer: str,
    expected_sources: list[str],
    retrieved_contexts: list[str],
) -> str:
    """Renders every field the judge needs as plain structured text."""
    citations_lines = (
        "\n".join(
            f"  - {c.get('source', '?')} / {c.get('section', '?')}"
            for c in candidate_citations
        )
        or "  (none)"
    )
    retrieved_lines = (
        "\n\n".join(f"[ctx-{i + 1}] {t}" for i, t in enumerate(retrieved_contexts[:5]))
        or "(no contexts retrieved)"
    )
    sources_lines = "\n".join(f"  - {s}" for s in expected_sources) or "  (unspecified)"

    return (
        f"Question: {question}\n"
        f"Expected language: {language}\n"
        f"Expected sources:\n{sources_lines}\n\n"
        f"REFERENCE answer:\n{reference_answer}\n\n"
        f"CANDIDATE answer:\n{candidate_answer or '(empty)'}\n\n"
        f"CANDIDATE citations:\n{citations_lines}\n\n"
        f"Retrieved context chunks used by the candidate:\n{retrieved_lines}\n"
    )


def judge(
    *,
    question: str,
    language: str,
    candidate_answer: str,
    candidate_citations: list[dict],
    reference_answer: str,
    expected_sources: list[str],
    retrieved_contexts: list[str],
    bedrock_runtime,
    model_id: str | None = None,
    max_tokens: int = 400,
) -> JudgeScore:
    """Runs the LLM-judge on a single (question, candidate, reference) triple."""
    model = model_id or os.environ.get("JUDGE_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID")
    if not model:
        return JudgeScore(notes="no model id configured for judge")

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": JUDGE_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": _build_user_message(
                    question=question,
                    language=language,
                    candidate_answer=candidate_answer,
                    candidate_citations=candidate_citations,
                    reference_answer=reference_answer,
                    expected_sources=expected_sources,
                    retrieved_contexts=retrieved_contexts,
                ),
            }
        ],
    }
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as err:
        return JudgeScore(notes=f"bedrock_error: {err}")

    result = json.loads(response["body"].read())
    text = (result.get("content", [{}]) or [{}])[0].get("text", "")
    return _parse_judge_response(text)
