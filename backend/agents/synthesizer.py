"""
Synthesizer \u2014 merges multi-specialist answers into one.

Runs after the parallel specialist fan-out. Three behaviours:

  0 outputs       \u2192 returns a safe \u201cno answer\u201d blocked response (defensive;
                    shouldn't happen in practice because the supervisor
                    always returns at least one agent).
  1 output        \u2192 passthrough \u2014 the specialist answer is used verbatim.
                    No extra LLM call; keeps single-specialist latency at
                    parity with the Phase 1 pipeline.
  \u22652 outputs      \u2192 calls Claude with SYNTHESIZER_PROMPT on the concatenated
                    specialist answers. Citations are unioned (dedupe by
                    (source, section)) so the frontend shows every unique
                    chunk without duplicates.

Blocked specialists propagate: if any specialist returned blocked=True
we keep that specialist's refusal wording to preserve the guardrail UX.
"""

from __future__ import annotations

import json

from prompts import SYNTHESIZER_PROMPT


def _dedup_citations(citation_lists: list[list[dict]]) -> list[dict]:
    """Unions citation lists across specialists using (source, section) key."""
    seen: set[str] = set()
    out: list[dict] = []
    for citations in citation_lists:
        for c in citations:
            source = (c.get("source") or "").strip()
            section = (c.get("section") or "").strip()
            key = f"{source}|{section}" if source and section else c.get("chunkId", "")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(c)
    return out


def synthesize(
    outputs: list[dict],
    language: str,
    bedrock_runtime,
    model_id: str,
    *,
    max_tokens: int = 1024,
) -> dict:
    """
    Returns `{text, citations, blocked}` ready to merge into AgentState.
    """
    if not outputs:
        return {"text": "", "citations": [], "blocked": True}

    # Blocked fast-path: if any specialist is blocked, surface its wording.
    for o in outputs:
        if o.get("blocked"):
            return {"text": o.get("text", ""), "citations": [], "blocked": True}

    if len(outputs) == 1:
        o = outputs[0]
        return {
            "text": o.get("text", ""),
            "citations": list(o.get("citations") or []),
            "blocked": False,
        }

    merged_text = _call_synthesizer(outputs, language, bedrock_runtime, model_id, max_tokens)
    merged_citations = _dedup_citations([o.get("citations") or [] for o in outputs])
    return {"text": merged_text, "citations": merged_citations, "blocked": False}


def _build_user_message(outputs: list[dict], language: str) -> str:
    """Renders the specialist answers as a structured input for the synthesizer."""
    lines = [f"Target language: {language}.", ""]
    for o in outputs:
        lines.append(f"=== {o.get('agent', 'unknown').title()} specialist ===")
        lines.append(o.get("text", "").strip() or "(no answer)")
        lines.append("")
    return "\n".join(lines).rstrip()


def _call_synthesizer(
    outputs: list[dict],
    language: str,
    bedrock_runtime,
    model_id: str,
    max_tokens: int,
) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYNTHESIZER_PROMPT,
        "messages": [
            {"role": "user", "content": _build_user_message(outputs, language)}
        ],
    }
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as err:
        print(f"[synthesizer] Bedrock call failed, concatenating answers verbatim: {err}")
        return "\n\n".join(o.get("text", "") for o in outputs)

    result = json.loads(response["body"].read())
    return (result.get("content", [{}]) or [{}])[0].get("text", "")
