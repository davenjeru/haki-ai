"""
Chat-only node — used when classify_intent decides the turn is conversational.

Invokes Claude directly with the full message history and the chat system
prompt (prompts.build_chat_system_prompt). No retrieval, no citations, no
Bedrock KB session — just a standard multi-turn Anthropic completion.

The `messages` list from AgentState already contains every turn in the
LangGraph-persisted conversation, so Claude has full recall of things like
"my name is Dave" → "what is my name?".
"""

from __future__ import annotations

import json

from prompts import build_chat_system_prompt


# Claude on Bedrock expects a tight, non-empty messages array. Our state
# stores messages as plain [{role, content}] dicts; the content may be an
# empty string on error paths, which Claude rejects, so we filter those
# defensively rather than propagate a BadRequest to the user.
def _sanitize_messages(messages: list[dict]) -> list[dict]:
    clean: list[dict] = []
    for m in messages:
        content = (m.get("content") or "").strip()
        role = m.get("role")
        if not content or role not in ("user", "assistant"):
            continue
        clean.append({"role": role, "content": content})
    return clean


def invoke_chat(
    messages: list[dict],
    language: str,
    bedrock_runtime,
    model_id: str,
    max_tokens: int = 1024,
) -> str:
    """
    Returns the assistant's chat-only reply as a plain string.

    Args:
        messages:         Full conversation so far (oldest first).
        language:         "english", "swahili", or "mixed".
        bedrock_runtime:  boto3 bedrock-runtime client.
        model_id:         Bedrock model id (same as the RAG path model).
    """
    system_prompt = build_chat_system_prompt(language)
    clean = _sanitize_messages(messages)
    if not clean:
        # Defensive fallback: empty history shouldn't happen because the
        # handler always seeds one user turn, but guard anyway.
        clean = [{"role": "user", "content": "(empty)"}]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": clean,
    }
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result.get("content", [{}])[0].get("text", "")
