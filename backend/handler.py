"""
Haki AI — Lambda handler.

Entry point for the chat API. Orchestrates:
  1. Language detection (Comprehend)
  2. System prompt construction          [stub]
  3. RAG via Bedrock KB                  [stub]
  4. Guardrail block check               [stub]
  5. Citation extraction                 [stub]
  6. CloudWatch metrics                  [stub]
  7. Return response JSON

Environment concerns (LocalStack endpoints, fallbacks) live in
config.py, clients.py, and adapters.py — not here.
"""

import json

from config import load_config
from clients import make_comprehend
from adapters import ComprehendAdapter

# ── Language detection ────────────────────────────────────────────────────────

# Minimum confidence for a language to be considered dominant.
# Below this threshold, and when both English and Swahili are present,
# the message is treated as mixed. Kenyan code-switching commonly scores
# in the 0.6–0.85 range even for mostly-one-language text.
_DOMINANCE_THRESHOLD = 0.85


def detect_language(text: str, comprehend: ComprehendAdapter) -> str:
    """
    Returns "english", "swahili", or "mixed" for the given text.
    Delegates all env/fallback concerns to ComprehendAdapter.
    """
    languages = comprehend.detect_dominant_language(text)
    scores = {lang["LanguageCode"]: lang["Score"] for lang in languages}

    top = languages[0] if languages else {}
    top_code = top.get("LanguageCode")
    top_score = top.get("Score", 0.0)

    if top_code == "en" and top_score >= _DOMINANCE_THRESHOLD:
        return "english"
    if top_code == "sw" and top_score >= _DOMINANCE_THRESHOLD:
        return "swahili"
    if "en" in scores and "sw" in scores:
        return "mixed"
    if top_code == "sw":
        return "swahili"
    # Unknown language or no result — default to English so the user gets a response
    return "english"

# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Main Lambda entry point. Called by API Gateway HTTP API.

    Expected request body: { "message": "<user question>" }
    Response body:         { "response": str, "citations": list, "language": str, "blocked": bool }
    """
    try:
        config = load_config()
        comprehend = ComprehendAdapter(make_comprehend(config), config.is_local)

        body = json.loads(event.get("body") or "{}")
        user_message = body.get("message", "").strip()

        if not user_message:
            return _response(400, {"error": "message is required"})

        # Step 1: Language detection
        language = detect_language(user_message, comprehend)

        # Step 2: Build system prompt  [TODO]
        # system_prompt = build_system_prompt(language)

        # Step 3: Retrieve and generate via Bedrock KB  [TODO]
        # bedrock = make_bedrock_agent_runtime(config)
        # rag_result = retrieve_and_generate(user_message, system_prompt, config, bedrock)

        # Step 4: Check for guardrail block  [TODO]
        # if rag_result["stopReason"] == "guardrail_intervened":
        #     return _response(200, {"response": BILINGUAL_REFUSAL, "citations": [], "language": language, "blocked": True})

        # Step 5: Extract citations  [TODO]
        # citations = extract_citations(rag_result)

        # Step 6: Emit CloudWatch metrics  [TODO]
        # cloudwatch = make_cloudwatch(config)
        # emit_metrics(cloudwatch, language, latency_ms, blocked=False)

        return _response(200, {
            "response": "",       # filled in by step 3
            "citations": [],      # filled in by step 5
            "language": language,
            "blocked": False,
        })

    except Exception as err:
        print(f"Unhandled error: {err}")
        return _response(500, {"error": "Internal server error"})


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
