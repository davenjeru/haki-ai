"""
Loader for the golden evaluation set.

One JSONL file keeps the dataset diffable in git and lets CI run a subset
by category or language without a schema migration.

Each record has:
    id                -- stable id used in reports
    category          -- "constitution" | "employment" | "land"
    question          -- user turn text
    reference_answer  -- gold-standard answer used by the LLM judge
    expected_sources  -- list[str] of statute names; maps 1:1 onto the
                         specialist filter in backend/agents/specialists.py
    expected_sections -- list[str] (informational; used for reporting, not
                         for metadata filtering, so a statute-wide query
                         can still match)
    language          -- "english" | "swahili" | "mixed"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_sources: list[str]
    expected_sections: list[str]
    language: str

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenCase":
        return cls(
            id=d["id"],
            category=d["category"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            expected_sources=list(d.get("expected_sources") or []),
            expected_sections=list(d.get("expected_sections") or []),
            language=d.get("language", "english"),
        )


_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "golden_set.jsonl")


def load_golden_set(path: str | None = None) -> list[GoldenCase]:
    """Parses the JSONL dataset into a list of ``GoldenCase`` records."""
    path = path or _DEFAULT_PATH
    cases: list[GoldenCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                cases.append(GoldenCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as err:
                raise ValueError(f"{path}:{line_no}: {err}") from err
    return cases
