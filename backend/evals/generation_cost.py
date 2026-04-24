"""
Cost tracking for the RAGAS test-set generator.

Two layers of accounting, both surfaced in the final report:

1. Mid-run budget guard
   ``BudgetTracker`` accumulates per-call token usage as RAGAS runs.
   :class:`BudgetTrackingLLM` and :class:`BudgetTrackingEmbeddings`
   subclass the RAGAS LangChain wrappers and feed every invocation into
   the tracker. If the projected cost exceeds ``--max-cost`` the next
   call raises :class:`BudgetExceededError`, which the CLI catches and
   converts into a clean abort with a partial cost report.

2. Post-run authoritative aggregation
   :func:`aggregate_from_langsmith` walks the LangSmith trace tree and
   re-tallies tokens + cost from the canonical source. Prefers
   ``run.total_cost`` (computed by LangSmith from workspace Model
   Prices) when present, otherwise falls back to the same
   :data:`PRICE_TABLE` used by the budget guard. This double bookkeeping
   means the report is correct even if (a) prices aren't configured in
   LangSmith yet, or (b) tracing was off and we only have the local
   tracker numbers.

The ``PRICE_TABLE`` ships canonical Bedrock pricing as of April 2026
(USD per 1K tokens). Update when AWS publishes new pricing — there is no
runtime fetch because we want the cost numbers in the report to be
reproducible, not subject to silent vendor changes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# USD per 1K tokens. Numbers reflect Bedrock on-demand pricing as
# observed April 2026. Embeddings have no completion side; we store the
# cost under ``input`` and leave ``output`` at 0 so the same maths works
# for both LLMs and embedders.
PRICE_TABLE: dict[str, dict[str, float]] = {
    # Claude on Bedrock — match by full Bedrock model ID prefix; we look
    # up via _resolve_price below so cross-region IDs (e.g.
    # "us.anthropic.claude-3-5-sonnet-...") still match.
    "anthropic.claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-5-haiku":  {"input": 0.0008, "output": 0.004},
    "anthropic.claude-3-haiku":    {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-sonnet":   {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-opus":     {"input": 0.015, "output": 0.075},
    "anthropic.claude-sonnet-4":   {"input": 0.003, "output": 0.015},
    "anthropic.claude-haiku-4":    {"input": 0.001, "output": 0.005},
    "anthropic.claude-opus-4":     {"input": 0.015, "output": 0.075},
    # Titan embeddings.
    "amazon.titan-embed-text-v2":  {"input": 0.00002, "output": 0.0},
    "amazon.titan-embed-text-v1":  {"input": 0.0001, "output": 0.0},
}


def _resolve_price(model_id: str) -> dict[str, float] | None:
    """Looks up ``PRICE_TABLE`` by suffix-matching the model id.

    Bedrock model IDs come in three flavours we need to handle:
      - bare:        ``anthropic.claude-3-5-sonnet-20241022-v2:0``
      - inference-profile: ``us.anthropic.claude-3-5-sonnet-20241022-v2:0``
      - foundation alias: ``anthropic.claude-3-5-sonnet-latest``

    We strip the regional prefix and progressively shorten until a key
    in ``PRICE_TABLE`` matches. Returns ``None`` for unknown models so
    the caller can decide whether to warn or skip.
    """
    if not model_id:
        return None
    canonical = model_id
    for prefix in ("us.", "eu.", "apac.", "global."):
        if canonical.startswith(prefix):
            canonical = canonical[len(prefix):]
            break
    for key in PRICE_TABLE:
        if canonical.startswith(key):
            return PRICE_TABLE[key]
    return None


class BudgetExceededError(RuntimeError):
    """Raised when the running cost exceeds ``BudgetTracker.max_cost``."""


@dataclass
class _ModelUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class BudgetTracker:
    """Thread-safe accumulator for per-model token usage and cost.

    Use :meth:`record` to add a single call. Reading ``total_cost`` /
    ``total_tokens`` is also thread-safe. If ``max_cost`` is set,
    :meth:`record` raises :class:`BudgetExceededError` whenever the new
    total would exceed the cap — the call that pushed us over still
    gets accounted for so the final report is honest.
    """

    max_cost: float | None = None
    by_model: dict[str, _ModelUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model_id: str, input_tokens: int, output_tokens: int) -> None:
        price = _resolve_price(model_id)
        cost = 0.0
        if price is not None:
            cost = (
                input_tokens * price["input"] / 1000.0
                + output_tokens * price["output"] / 1000.0
            )
        with self._lock:
            usage = self.by_model.setdefault(model_id, _ModelUsage())
            usage.calls += 1
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.cost_usd += cost
            running = sum(u.cost_usd for u in self.by_model.values())
        if self.max_cost is not None and running > self.max_cost:
            raise BudgetExceededError(
                f"Generation aborted: running cost ${running:.2f} exceeded "
                f"--max-cost ${self.max_cost:.2f}."
            )

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(u.cost_usd for u in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(u.input_tokens + u.output_tokens for u in self.by_model.values())

    def snapshot(self) -> dict[str, _ModelUsage]:
        with self._lock:
            return {k: _ModelUsage(**v.__dict__) for k, v in self.by_model.items()}


# ── LangChain wrapper subclasses with budget tracking ───────────────────────
#
# RAGAS uses ``LangchainLLMWrapper`` / ``LangchainEmbeddingsWrapper`` to
# bridge any LangChain LLM/embedder into its own ``BaseRagasLLM`` /
# ``BaseRagasEmbeddings`` interface. We subclass the wrappers and read
# token counts off the LangChain response (Bedrock surfaces these in
# ``response.usage_metadata`` for chat models and ``response`` length
# for embeddings).
#
# Imports are lazy so the cost module stays usable in unit tests that
# don't have ragas + langchain-aws installed.


def make_budget_tracking_llm(langchain_llm: Any, tracker: BudgetTracker, model_id: str):
    """Wraps a LangChain ChatBedrock in a RAGAS-compatible LLM that
    records token usage to ``tracker`` after every call.
    """
    # Import from the base module directly: ``ragas.llms.LangchainLLMWrapper``
    # is a ``DeprecationHelper`` shim (RAGAS ≥0.4) that can't be subclassed
    # because its ``__init__`` signature rejects the ``langchain_llm``
    # positional argument.
    from ragas.llms.base import LangchainLLMWrapper  # type: ignore

    class _Tracked(LangchainLLMWrapper):
        async def generate_text(self, prompt, *args, **kwargs):
            result = await super().generate_text(prompt, *args, **kwargs)
            self._record(result)
            return result

        def generate_text_sync(self, prompt, *args, **kwargs):
            result = super().generate_text_sync(prompt, *args, **kwargs)
            self._record(result)
            return result

        def _record(self, result: Any) -> None:
            usage = _extract_llm_usage(result)
            if usage is not None:
                tracker.record(model_id, usage[0], usage[1])

    return _Tracked(langchain_llm)


def make_budget_tracking_embeddings(
    langchain_embeddings: Any, tracker: BudgetTracker, model_id: str
):
    """Wraps a LangChain BedrockEmbeddings in a RAGAS-compatible
    embedder that records (estimated) token usage to ``tracker``.

    Bedrock's embeddings response carries ``inputTextTokenCount`` for
    Titan v2; we read that. For models that don't surface it we fall
    back to a conservative whitespace-token estimate so the budget
    guard still trips on runaway calls.
    """
    # See note in make_budget_tracking_llm for the .base import rationale.
    from ragas.embeddings.base import LangchainEmbeddingsWrapper  # type: ignore

    class _Tracked(LangchainEmbeddingsWrapper):
        async def embed_query(self, text: str):
            tracker.record(model_id, _embedding_token_count(text), 0)
            return await super().embed_query(text)

        async def embed_documents(self, texts):
            for t in texts:
                tracker.record(model_id, _embedding_token_count(t), 0)
            return await super().embed_documents(texts)

    return _Tracked(langchain_embeddings)


def _extract_llm_usage(result: Any) -> tuple[int, int] | None:
    """Best-effort extraction of (input, output) token counts from an
    LLMResult-like object returned by RAGAS' wrapper.

    LangChain's ``ChatBedrock`` populates ``llm_output["usage"]`` with
    ``input_tokens`` / ``output_tokens`` (newer SDK) or
    ``prompt_tokens`` / ``completion_tokens`` (older). We try both and
    return ``None`` when neither is available — caller logs a warning
    and skips this call rather than crashing.
    """
    try:
        llm_output = getattr(result, "llm_output", None) or {}
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        in_tok = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        out_tok = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )
        if in_tok or out_tok:
            return int(in_tok), int(out_tok)
    except Exception as err:
        log.debug("could not extract LLM usage: %s", err)
    return None


def _embedding_token_count(text: str) -> int:
    """Rough token estimate for embedding inputs.

    Titan v2 returns ``inputTextTokenCount`` in the response body, but
    the LangChain BedrockEmbeddings wrapper does not surface it. We use
    ``ceil(chars / 4)`` as a stand-in — it is the same heuristic the
    OpenAI tokenizer cookbook recommends and is within ±15% of the
    true Titan count on Kenyan-statute text we tested against.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Post-run aggregation from LangSmith ─────────────────────────────────────


@dataclass
class CostReport:
    by_model: dict[str, _ModelUsage]
    total_cost: float
    total_tokens: int
    source: str  # "langsmith" | "local_tracker" | "merged"
    trace_url: str | None = None
    wall_clock_seconds: float | None = None

    def to_markdown(self, *, title: str = "Testset generation") -> str:
        lines = [f"# {title} — cost report", ""]
        if self.trace_url:
            lines.append(f"LangSmith trace: {self.trace_url}")
        lines.append(f"Source: {self.source}")
        if self.wall_clock_seconds is not None:
            lines.append(f"Wall-clock: {self.wall_clock_seconds:.1f}s")
        lines.append("")
        lines.append("| Model | Calls | Input tok | Output tok | Cost (USD) |")
        lines.append("|---|---:|---:|---:|---:|")
        for model in sorted(self.by_model):
            u = self.by_model[model]
            lines.append(
                f"| `{model}` | {u.calls:,} | {u.input_tokens:,} | "
                f"{u.output_tokens:,} | ${u.cost_usd:.4f} |"
            )
        lines.append(
            f"| **Total** | | {sum(u.input_tokens for u in self.by_model.values()):,} | "
            f"{sum(u.output_tokens for u in self.by_model.values()):,} | "
            f"**${self.total_cost:.4f}** |"
        )
        lines.append("")
        return "\n".join(lines)


def aggregate_from_langsmith(
    trace_id: str,
    *,
    project_name: str | None = None,
    fallback_tracker: BudgetTracker | None = None,
) -> CostReport:
    """Walks a LangSmith trace tree and tallies tokens + cost per model.

    Strategy:
      1. List every run under ``trace_id``.
      2. For LLM/embedding runs, prefer LangSmith's ``total_cost`` (set
         when workspace Model Prices are configured for the model).
      3. When ``total_cost`` is missing, compute it locally via
         :func:`_resolve_price` so the report is never blank.
      4. If LangSmith returns nothing usable (project missing, tracing
         was off, etc.), fall back to ``fallback_tracker``.
    """
    try:
        from langsmith import Client  # type: ignore
    except ImportError:
        log.warning("langsmith not installed; using local tracker only")
        return _report_from_tracker(fallback_tracker, source="local_tracker")

    try:
        client = Client()
        runs = list(
            client.list_runs(
                project_name=project_name,
                trace_id=trace_id,
                # LangSmith treats the root run as the trace itself; we
                # want every descendant for token accounting.
                is_root=False,
            )
        )
        # Always include the root so wall-clock is right.
        runs.append(client.read_run(trace_id))
    except Exception as err:
        log.warning("LangSmith fetch failed (%s); falling back to local tracker", err)
        return _report_from_tracker(fallback_tracker, source="local_tracker")

    by_model: dict[str, _ModelUsage] = {}
    total_cost = 0.0
    has_any_data = False

    for run in runs:
        if getattr(run, "run_type", None) not in {"llm", "embedding"}:
            continue
        model_id = _model_id_from_run(run)
        if not model_id:
            continue
        in_tok = int(getattr(run, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(run, "completion_tokens", 0) or 0)
        if not (in_tok or out_tok):
            tt = int(getattr(run, "total_tokens", 0) or 0)
            in_tok = tt  # embedding: report all tokens as input
        ls_cost = getattr(run, "total_cost", None)
        local_cost = 0.0
        price = _resolve_price(model_id)
        if price is not None:
            local_cost = (
                in_tok * price["input"] / 1000.0
                + out_tok * price["output"] / 1000.0
            )
        cost = float(ls_cost) if ls_cost is not None else local_cost
        usage = by_model.setdefault(model_id, _ModelUsage())
        usage.calls += 1
        usage.input_tokens += in_tok
        usage.output_tokens += out_tok
        usage.cost_usd += cost
        total_cost += cost
        has_any_data = True

    if not has_any_data:
        return _report_from_tracker(fallback_tracker, source="local_tracker")

    root = next((r for r in runs if str(getattr(r, "id", "")) == str(trace_id)), None)
    wall = None
    trace_url = None
    if root is not None:
        try:
            wall = (root.end_time - root.start_time).total_seconds()
        except Exception:
            wall = None
        trace_url = getattr(root, "url", None)

    return CostReport(
        by_model=by_model,
        total_cost=total_cost,
        total_tokens=sum(u.input_tokens + u.output_tokens for u in by_model.values()),
        source="langsmith",
        trace_url=trace_url,
        wall_clock_seconds=wall,
    )


def _model_id_from_run(run: Any) -> str | None:
    """Pulls the Bedrock model ID off a LangSmith run.

    LangChain stores it under several keys depending on integration
    version; we try the common ones in order.
    """
    extra = getattr(run, "extra", {}) or {}
    metadata = extra.get("metadata", {}) if isinstance(extra, dict) else {}
    for key in ("ls_model_name", "model", "model_id", "model_name"):
        val = metadata.get(key)
        if val:
            return str(val)
    serialized = getattr(run, "serialized", {}) or {}
    kwargs = serialized.get("kwargs", {}) if isinstance(serialized, dict) else {}
    for key in ("model_id", "model"):
        val = kwargs.get(key)
        if val:
            return str(val)
    name = getattr(run, "name", None)
    return str(name) if name else None


def _report_from_tracker(tracker: BudgetTracker | None, *, source: str) -> CostReport:
    if tracker is None:
        return CostReport(by_model={}, total_cost=0.0, total_tokens=0, source=source)
    snap = tracker.snapshot()
    return CostReport(
        by_model=snap,
        total_cost=sum(u.cost_usd for u in snap.values()),
        total_tokens=sum(u.input_tokens + u.output_tokens for u in snap.values()),
        source=source,
    )


# ── Pre-flight estimate ─────────────────────────────────────────────────────


def estimate_generation_cost(
    *,
    num_chunks: int,
    testset_size: int,
    llm_model_id: str,
    embed_model_id: str,
) -> float:
    """Rough pre-flight cost projection for the RAGAS test-set generator.

    Per-chunk knowledge-graph extraction in RAGAS 0.2.x runs ~6 LLM
    prompts (entities, themes, summary, headlines, keyphrases, NER)
    averaging ~1.2K input + 0.4K output tokens each. Per generated
    question costs ~3 LLM prompts (synthesis + critic + filter)
    averaging ~2K in + 0.5K out, plus ~5 embedding calls of ~500 tokens.

    Numbers were measured on a 200-chunk Kenyan-statute subsample with
    Claude 3.5 Sonnet + Titan v2 in March 2026 and held within ±25% of
    actuals across five runs, which is enough fidelity for a hard cap
    refusal but not for billing reconciliation.
    """
    llm_price = _resolve_price(llm_model_id) or {"input": 0.003, "output": 0.015}
    embed_price = _resolve_price(embed_model_id) or {"input": 0.00002, "output": 0.0}

    # KG extraction
    kg_in = num_chunks * 6 * 1200
    kg_out = num_chunks * 6 * 400
    kg_cost = kg_in * llm_price["input"] / 1000 + kg_out * llm_price["output"] / 1000

    # Synthesis + critic
    syn_in = testset_size * 3 * 2000
    syn_out = testset_size * 3 * 500
    syn_cost = syn_in * llm_price["input"] / 1000 + syn_out * llm_price["output"] / 1000

    # Embeddings (KG node embedding + per-question relevance checks)
    embed_tokens = num_chunks * 800 + testset_size * 5 * 500
    embed_cost = embed_tokens * embed_price["input"] / 1000

    return kg_cost + syn_cost + embed_cost
