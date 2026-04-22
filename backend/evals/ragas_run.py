"""
RAGAS metrics — optional advanced-RAG scorer.

Computes the four core RAGAS metrics against the golden set:
    - faithfulness
    - answer_relevancy
    - context_precision
    - context_recall

RAGAS pulls a large dependency tree (langchain, datasets, pyarrow) that
we don't want in the Lambda runtime. It's declared in the ``evals``
optional dependency group so CI and local runs opt in explicitly:

    uv sync --group evals
    uv run -m evals.run --ragas

If the group isn't installed, :func:`score_with_ragas` returns ``None``
and the harness falls back to LLM-judge-only. This keeps ``uv run -m
evals.run`` useful for a fast iteration loop before committing to a
full, costly RAGAS sweep.
"""

from __future__ import annotations

from typing import Any

from .runner import EvalResult


def _lazy_import():
    """Imports RAGAS lazily. Returns None when the ``evals`` group isn't installed."""
    try:
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset  # type: ignore
        return {
            "evaluate": evaluate,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "Dataset": Dataset,
        }
    except ImportError as err:
        print(f"[ragas] optional dependency not installed ({err}); skipping RAGAS step.")
        return None


def score_with_ragas(results: list[EvalResult]) -> dict[str, Any] | None:
    """Runs RAGAS on the already-collected eval results.

    Returns a dict of ``{metric_name -> float}`` aggregated across the
    dataset, or ``None`` when RAGAS isn't installed / can't be loaded.
    Per-case scores are attached back onto each :class:`EvalResult` via
    its ``extra`` dict so the markdown report can display them.
    """
    lib = _lazy_import()
    if lib is None:
        return None

    Dataset = lib["Dataset"]
    records = [
        {
            "question": r.case.question,
            "answer": r.answer,
            "contexts": r.retrieved_contexts or [""],
            "ground_truth": r.case.reference_answer,
        }
        for r in results
    ]
    dataset = Dataset.from_list(records)

    # NB: by default RAGAS uses OpenAI. In an airgapped / Bedrock-only
    # environment the caller should configure a ``BedrockWrapper``-backed
    # LLM and embeddings via ``ragas.run_config`` and pass them through
    # ``evaluate(llm=..., embeddings=...)``. We leave that wiring to the
    # caller because it depends on how the eval is being invoked (CI vs.
    # local dev vs. nightly cron).
    metrics = [
        lib["faithfulness"],
        lib["answer_relevancy"],
        lib["context_precision"],
        lib["context_recall"],
    ]
    try:
        scores = lib["evaluate"](dataset, metrics=metrics)
    except Exception as err:
        print(f"[ragas] evaluate failed: {err}")
        return None

    per_case = scores.to_pandas() if hasattr(scores, "to_pandas") else None
    if per_case is not None:
        for i, result in enumerate(results):
            row = per_case.iloc[i].to_dict()
            result.extra["ragas"] = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}

    aggregate: dict[str, float] = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        value = getattr(scores, name, None) if not isinstance(scores, dict) else scores.get(name)
        if value is None and per_case is not None and name in per_case.columns:
            value = float(per_case[name].mean())
        if value is not None:
            aggregate[name] = float(value)
    return aggregate
