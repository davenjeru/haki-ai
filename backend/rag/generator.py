"""
Answer generation \u2014 final stage of the RAG pipeline.

Calls Claude via `bedrock-runtime.invoke_model` with the user query + the
final (reranked, filtered, deduplicated) chunk context. A Bedrock guardrail
is attached when a guardrail id is provided so the pipeline inherits the
same topic-denial boundary the legacy `retrieve_and_generate` path used.

Guardrail behaviour:
  - stopReason == "guardrail_intervened"       \u2192 blocked by policy
  - guardrailAction/STOP_REASON alt fields     \u2192 same, kept for compat

Returns (answer_text, stop_reason) so callers can branch on blocked vs.
normal completion without re-parsing the response body.
"""

from __future__ import annotations

import json


_MAX_CONTEXT_CHARS = 12_000
_MAX_OUTPUT_TOKENS = 1024


def build_context(results: list[dict]) -> str:
    """
    Renders reranked results into the `<context>` block the model reads.

    Each chunk is prefixed with a `[source \u2014 chapter \u2014 section \u2014 title]`
    header so the model can cite accurately. Results are truncated at
    `_MAX_CONTEXT_CHARS` to keep prompt cost predictable; if the top-K
    overruns, we still include full text for the highest-ranked chunks.
    """
    parts: list[str] = []
    budget = _MAX_CONTEXT_CHARS
    for r in results:
        meta = r.get("metadata") or {}
        header = " \u2014 ".join(
            part for part in (
                meta.get("source"),
                meta.get("chapter"),
                meta.get("section"),
                meta.get("title"),
            ) if part
        )
        body = (r.get("content") or {}).get("text", "")
        block = f"[{header}]\n{body}"
        if budget <= 0:
            break
        if len(block) > budget:
            block = block[:budget]
        parts.append(block)
        budget -= len(block) + 8  # +8 for the \"\\n\\n---\\n\\n\" separator
    return "\n\n---\n\n".join(parts)


def generate(
    query: str,
    system_prompt: str,
    context: str,
    model_id: str,
    bedrock_runtime,
    *,
    guardrail_id: str = "",
    guardrail_version: str = "",
    max_tokens: int = _MAX_OUTPUT_TOKENS,
) -> tuple[str, str]:
    """
    Invokes Claude with the built context and returns (text, stop_reason).
    """
    user_content = f"<context>\n{context}\n</context>\n\n{query}" if context else query
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_content}],
    }

    kwargs: dict = {
        "modelId": model_id,
        "body": json.dumps(body),
        "contentType": "application/json",
        "accept": "application/json",
    }
    if guardrail_id:
        kwargs["guardrailIdentifier"] = guardrail_id
        kwargs["guardrailVersion"] = guardrail_version or "DRAFT"

    response = bedrock_runtime.invoke_model(**kwargs)
    result = json.loads(response["body"].read())
    text = (result.get("content", [{}]) or [{}])[0].get("text", "")
    stop_reason = result.get("stop_reason", "end_turn")
    return text, stop_reason
