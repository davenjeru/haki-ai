"""
RAG orchestration — steps 3 and 4.

retrieve_and_generate() calls whichever RAG adapter is passed in:
  - LocalRAGAdapter  (is_local=True)  — ChromaDB + Bedrock InvokeModel
  - BedrockRAGAdapter (is_local=False) — real Bedrock KB agent runtime

Both adapters return the same response shape, so this module has no
env-specific logic. handler.py decides which adapter to instantiate.

check_guardrail_block() inspects the stopReason field that both adapters
return, so it also works identically in both environments.
"""

from prompts import BILINGUAL_REFUSAL


def retrieve_and_generate(
    query: str,
    system_prompt: str,
    model_id: str,
    rag_adapter,
    kb_session_id: str | None = None,
) -> dict:
    """
    Calls the RAG adapter and returns its response dict:
      {
        "output":    { "text": str },
        "citations": [ { "retrievedReferences": [...] } ],
        "stopReason": str,
        "sessionId": str,   # only on Bedrock path
      }

    Args:
        query:         The user's message.
        system_prompt: Built by prompts.build_system_prompt(language).
        model_id:      Bedrock model ID (e.g. "anthropic.claude-3-sonnet-...").
        rag_adapter:   LocalRAGAdapter or BedrockRAGAdapter instance.
        kb_session_id: Bedrock KB session id from a prior turn (prod only).
                       The adapter returns a new sessionId which callers
                       should persist in graph state for the next turn.
    """
    return rag_adapter.retrieve_and_generate(
        query, system_prompt, model_id, kb_session_id=kb_session_id
    )


def check_guardrail_block(rag_result: dict) -> bool:
    """
    Returns True if Bedrock Guardrails blocked the response.

    The signal differs by API:
      - retrieve_and_generate (Bedrock KB) returns `guardrailAction: "INTERVENED"`.
      - invoke_model with a guardrail set returns `stopReason: "guardrail_intervened"`.
    LocalRAGAdapter never sets either (it has no guardrail layer), so this
    always returns False locally — the system prompt is the only safety
    layer in local testing.
    """
    if rag_result.get("guardrailAction") == "INTERVENED":
        return True
    return rag_result.get("stopReason") == "guardrail_intervened"


def blocked_response(language: str) -> str:
    """
    Returns the bilingual refusal string used when a guardrail fires.
    Exposed here so handler.py has a single import for the blocked case.
    """
    return BILINGUAL_REFUSAL
