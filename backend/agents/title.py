"""
Short-title generator for signed-in chat threads.

A single cheap Haiku call that turns the first (question, answer) turn into
a <= 6-word title for the sidebar. Designed to be called from the handler
*after* the synthesizer has succeeded, inside a try/except so a title
failure never fails the actual chat response.
"""

from __future__ import annotations

import json
import logging

from prompts import TITLE_GENERATOR_PROMPT

logger = logging.getLogger(__name__)


_MAX_INPUT_CHARS = 2400  # keep Haiku prompt cheap even on long legal answers
_MAX_TITLE_WORDS = 6
_FALLBACK_TITLE = "New chat"


def _clip(text: str, limit: int = _MAX_INPUT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _sanitize_title(raw: str) -> str:
    """Strips quotes/punctuation and clamps to :data:`_MAX_TITLE_WORDS` words."""
    candidate = (raw or "").strip().splitlines()[0] if raw else ""
    candidate = candidate.strip().strip('"').strip("'")
    candidate = candidate.rstrip(".!?,:;-")
    if not candidate:
        return _FALLBACK_TITLE
    words = candidate.split()
    if len(words) > _MAX_TITLE_WORDS:
        candidate = " ".join(words[:_MAX_TITLE_WORDS])
    if len(candidate) > 80:
        candidate = candidate[:77].rstrip() + "…"
    return candidate


def generate_title(
    question: str,
    answer: str,
    bedrock_runtime,
    model_id: str,
) -> str:
    """
    Generates a short title for the conversation. Never raises — any failure
    falls back to ``"New chat"`` so the caller can unconditionally persist
    the result alongside the thread row.
    """
    if not question.strip():
        return _FALLBACK_TITLE
    user_text = (
        f"Q: {_clip(question)}\n"
        f"A: {_clip(answer)}"
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 40,
        "system": TITLE_GENERATOR_PROMPT,
        "messages": [{"role": "user", "content": user_text}],
    }
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("Title generation Bedrock call failed: %s", err)
        return _FALLBACK_TITLE

    try:
        payload = json.loads(response["body"].read())
        text = (payload.get("content", [{}]) or [{}])[0].get("text", "")
    except Exception as err:  # noqa: BLE001
        logger.warning("Title generation response parse failed: %s", err)
        return _FALLBACK_TITLE

    return _sanitize_title(text)
