"""
Intent classifier — decides whether the current turn needs RAG retrieval.

Called from graph.py as the `classify_intent` node. Uses a cheap Haiku
InvokeModel call with a tight system prompt (CLASSIFIER_PROMPT) that asks
for a single-key JSON response: {"needs_rag": true|false}.

Conservative behaviour:
  - If the model returns malformed JSON, we default to needs_rag=True so
    the user still gets a proper answer on the legal path.
  - If the Bedrock call itself fails, we propagate the exception — the
    graph wrapper handles it as an unhandled error (same as today).
"""

from __future__ import annotations

import json
import re
from typing import Any


# Max characters from the conversation that we send to the classifier.
# Keeps the Haiku call fast and cheap even for long sessions.
_MAX_CONTEXT_CHARS = 4000

# Only the last N turns matter for routing intent.
_RECENT_TURN_LIMIT = 6


def _format_messages_for_classifier(messages: list[dict]) -> str:
    """
    Renders the conversation as a short transcript prefixed with role tags.
    The last N turns only, truncated to _MAX_CONTEXT_CHARS worst-case.
    """
    tail = messages[-_RECENT_TURN_LIMIT:]
    lines = []
    for m in tail:
        role = "U" if m.get("role") == "user" else "A"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    transcript = "\n".join(lines)
    if len(transcript) > _MAX_CONTEXT_CHARS:
        transcript = transcript[-_MAX_CONTEXT_CHARS:]
    return transcript


def _parse_needs_rag(text: str) -> bool:
    """
    Pulls the first JSON object out of the model's reply and reads needs_rag.
    Falls back to True (i.e. run RAG) if we cannot parse the response.
    """
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not match:
        return True
    try:
        obj: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return True
    val = obj.get("needs_rag")
    if isinstance(val, bool):
        return val
    return True


def classify_intent(
    messages: list[dict],
    bedrock_runtime,
    model_id: str,
    classifier_prompt: str,
) -> bool:
    """
    Returns True if the latest user turn needs the RAG pipeline.

    Args:
        messages:           Full conversation so far, [{role, content}, ...].
        bedrock_runtime:    boto3 bedrock-runtime client.
        model_id:           Haiku model id (cheap + fast).
        classifier_prompt:  System prompt (see prompts.CLASSIFIER_PROMPT).
    """
    transcript = _format_messages_for_classifier(messages)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32,
        "system": classifier_prompt,
        "messages": [{"role": "user", "content": transcript or "(empty)"}],
    }
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    text = result.get("content", [{}])[0].get("text", "")
    return _parse_needs_rag(text)
