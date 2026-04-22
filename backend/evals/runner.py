"""
Eval runner — executes the advanced-RAG pipeline against every golden
case and returns structured candidate answers for scoring.

We intentionally bypass the supervisor/synthesizer graph and call
``run_rag`` directly:

- The golden set already tells us which statute(s) each question belongs
  to (``expected_sources``), so we can lock the retrieval filter and get
  deterministic, reproducible scores. Letting the supervisor route adds
  variance we can't attribute to the RAG pipeline under test.
- We still exercise the *same* pipeline the specialist subgraphs use —
  query expansion → hybrid retrieve → TOC filter → rerank → dedup →
  generate — so score deltas reflect real user-facing improvements.

Each result captures the candidate answer, citations, and the retrieved
contexts so downstream scorers (RAGAS, LLM-judge) can grade both the
answer and the supporting evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clients.adapters import BedrockRAGAdapter, LocalRAGAdapter
from clients import (
    make_bedrock_agent_runtime,
    make_bedrock_runtime,
    make_s3,
)
from app.config import Config, load_config
from prompts import build_system_prompt
from rag import (
    blocked_response,
    check_guardrail_block,
    extract_citations,
)
from rag.pipeline import run_rag

from .loader import GoldenCase


_VECTORSTORE_PATH_DEFAULT = ".local-vectorstore"


@dataclass
class EvalResult:
    case: GoldenCase
    answer: str
    citations: list[dict]
    retrieved_contexts: list[str]
    blocked: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _build_adapter(config: Config):
    bedrock_runtime = make_bedrock_runtime(config)
    bedrock_agent_runtime = make_bedrock_agent_runtime(config)
    s3 = make_s3(config)
    if config.is_local:
        import os
        vectorstore_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            _VECTORSTORE_PATH_DEFAULT,
        )
        return LocalRAGAdapter(
            bedrock_runtime=bedrock_runtime,
            bedrock_agent_runtime=bedrock_agent_runtime,
            embed_model=config.embedding_model_id,
            vectorstore_path=vectorstore_path,
            s3_client=s3,
            s3_bucket=config.s3_bucket,
            aws_region=config.aws_region,
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
            guardrail_id=config.guardrail_id,
            guardrail_version=config.guardrail_version,
        ), s3
    return BedrockRAGAdapter(
        bedrock_agent_runtime=bedrock_agent_runtime,
        bedrock_runtime=bedrock_runtime,
        config=config,
        s3_client=s3,
    ), s3


def _extract_retrieved_contexts(result: dict) -> list[str]:
    """Pulls the chunk texts out of the retrieval-augmented generation result."""
    contexts: list[str] = []
    for ref in result.get("citations") or []:
        text = (
            (ref.get("content") or {}).get("text")
            if isinstance(ref.get("content"), dict)
            else ref.get("text")
        )
        if isinstance(text, str) and text.strip():
            contexts.append(text.strip())
    return contexts


def run_case(case: GoldenCase, *, rag_adapter, s3_client, bucket: str, model_id: str) -> EvalResult:
    """Runs the advanced-RAG pipeline for a single golden case."""
    filt: dict | None = None
    if case.expected_sources:
        # Filter on the first source — the KB can only filter on one value
        # per key at a time and the golden set lists the primary statute
        # first. Cross-source questions (e.g. "non-citizen owning land")
        # still retrieve from the primary source, which is acceptable for
        # eval determinism.
        filt = {"source": case.expected_sources[0]}

    system_prompt = build_system_prompt(case.language)
    try:
        result = run_rag(
            query=case.question,
            system_prompt=system_prompt,
            model_id=model_id,
            rag_adapter=rag_adapter,
            metadata_filter=filt,
        )
    except Exception as err:
        return EvalResult(
            case=case,
            answer="",
            citations=[],
            retrieved_contexts=[],
            blocked=False,
            error=f"run_rag raised: {err}",
        )

    if check_guardrail_block(result):
        return EvalResult(
            case=case,
            answer=blocked_response(case.language),
            citations=[],
            retrieved_contexts=[],
            blocked=True,
        )

    citations = extract_citations(result, s3_client=s3_client, bucket=bucket)
    return EvalResult(
        case=case,
        answer=result.get("output", {}).get("text", ""),
        citations=citations,
        retrieved_contexts=_extract_retrieved_contexts(result),
    )


def run_all(cases: list[GoldenCase], config: Config | None = None) -> list[EvalResult]:
    """Runs every golden case sequentially and returns the raw results."""
    config = config or load_config()
    rag_adapter, s3 = _build_adapter(config)

    results: list[EvalResult] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[eval] {idx:>2}/{len(cases)} {case.id} ({case.language})")
        results.append(
            run_case(
                case,
                rag_adapter=rag_adapter,
                s3_client=s3,
                bucket=config.s3_bucket,
                model_id=config.bedrock_model_id,
            )
        )
    return results
