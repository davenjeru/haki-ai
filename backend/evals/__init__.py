"""
Haki AI — Evaluation harness (Phase 3).

Modules
-------
golden_set.jsonl
    30 curated Q&A pairs (10 Constitution, 10 Employment, 10 Land).
    10 of the 30 are Swahili or mixed-language so we exercise the
    Comprehend + bilingual-prompt path end-to-end.

runner
    Runs the advanced-RAG pipeline against every golden-set entry and
    captures {answer, citations, retrieved_contexts}. Uses ``run_rag``
    directly (rather than the full graph) so we can inject the right
    source filter per question and so RAGAS metrics have a clean,
    deterministic input.

llm_judge
    Pairwise LLM-as-judge comparing candidate vs. reference on four
    axes (accuracy, citation_correctness, tone, language_appropriateness).
    Uses the same Claude-on-Bedrock model as the generator so we inherit
    the existing IAM posture; the prompt lives in ``prompts.JUDGE_PROMPT``.

ragas_run
    Optional RAGAS metrics (faithfulness, answer_relevancy,
    context_precision, context_recall). RAGAS is a heavy dependency so it
    is loaded lazily behind an ImportError guard — if the ``evals``
    dependency group isn't installed, the RAGAS step is skipped with a
    clear warning and only the LLM-judge score is computed.

testset_generator / generate
    Synthetic test-set generation using RAGAS. ``testset_generator``
    loads the chunk catalog, builds budget-tracked Bedrock wrappers, and
    drives ``ragas.testset.TestsetGenerator`` with bilingual personas.
    ``generate`` is the CLI: ``uv run -m evals.generate --size 50``.

generation_cost
    ``BudgetTracker`` for mid-run cost accounting (hard caps on
    ``--max-cost``) plus a post-run aggregator that walks the LangSmith
    trace tree and produces an authoritative cost report, falling back
    to a local ``PRICE_TABLE`` when LangSmith prices aren't configured.

report
    Writes ``evals/reports/{timestamp}.md`` with per-question diffs and a
    summary table, then emits a single ``eval_score`` CloudWatch metric
    so regressions surface on the existing HakiAI dashboard.

run
    Entrypoint: ``uv run -m evals.run``.
"""

from .loader import load_golden_set

__all__ = ["load_golden_set"]
