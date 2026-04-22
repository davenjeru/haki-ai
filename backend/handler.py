"""
Haki AI — Lambda handler.

Entry point for the chat API. Orchestrates:
  1. Language detection (Comprehend)
  2. System prompt construction (prompts.py)
  3. RAG via Bedrock KB or local ChromaDB (rag.py)
  4. Guardrail block check (rag.py)
  5. Citation extraction (citations.py)
  6. CloudWatch metrics (metrics.py)
  7. Return response JSON

Environment concerns (LocalStack endpoints, adapter selection) live in
config.py, clients.py, and adapters.py — not here.
"""

import json
import os

from config import load_config
from clients import make_comprehend, make_bedrock_agent_runtime, make_bedrock_runtime, make_cloudwatch, make_s3
from adapters import ComprehendAdapter, LocalRAGAdapter, BedrockRAGAdapter
from prompts import build_system_prompt
from rag import retrieve_and_generate, check_guardrail_block, blocked_response
from citations import extract_citations
from metrics import emit_metrics, now_ms, elapsed_ms

# Path to the local ChromaDB store — relative to this file
_VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), ".local-vectorstore")

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


# ── RAG adapter factory ───────────────────────────────────────────────────────

def _make_rag_adapter(config, in_process: bool = False):
    """
    Returns the appropriate RAG adapter for the current environment:

      is_local + chroma_host set  → LocalRAGAdapter (HTTP client → ChromaDB server)
      is_local + in_process=True  → LocalRAGAdapter (in-process PersistentClient)
      is_local=False              → BedrockRAGAdapter (real Bedrock KB)

    chroma_host is injected as a Lambda env var (CHROMA_HOST) pointing at
    host.docker.internal so the container can reach the ChromaDB HTTP server
    running on the host machine. This mirrors the prod path where Lambda
    calls Bedrock KB over the network to reach S3 Vectors.

    in_process=True is used only by test_e2e_local.py (runs outside Docker).
    """
    if config.is_local:
        bedrock_runtime = make_bedrock_runtime(config)
        return LocalRAGAdapter(
            bedrock_runtime,
            config.embedding_model_id,
            _VECTORSTORE_PATH,
            chroma_host=config.chroma_host if not in_process else "",
            chroma_port=config.chroma_port,
        )
    return BedrockRAGAdapter(make_bedrock_agent_runtime(config), config)


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    """
    Main Lambda entry point. Called by API Gateway HTTP API.

    Expected request body: { "message": "<user question>" }
    Response body:         { "response": str, "citations": list, "language": str, "blocked": bool }
    """
    config = load_config()
    cloudwatch = make_cloudwatch(config)
    start = now_ms()
    language = "english"  # default for metrics if we fail before detection

    try:
        body = json.loads(event.get("body") or "{}")
        user_message = body.get("message", "").strip()

        if not user_message:
            return _response(400, {"error": "message is required"})

        # Step 1: Language detection
        comprehend = ComprehendAdapter(make_comprehend(config), config.is_local)
        language = detect_language(user_message, comprehend)

        # Step 2: Build system prompt
        system_prompt = build_system_prompt(language)

        # Step 3: Retrieve and generate
        rag_adapter = _make_rag_adapter(config)
        rag_result = retrieve_and_generate(
            user_message, system_prompt, config.bedrock_model_id, rag_adapter
        )

        # Step 4: Check for guardrail block
        if check_guardrail_block(rag_result):
            emit_metrics(
                cloudwatch,
                language=language,
                latency_ms=elapsed_ms(start),
                blocked=True,
                citations=[],
            )
            return _response(200, {
                "response": blocked_response(language),
                "citations": [],
                "language": language,
                "blocked": True,
            })

        # Step 5: Extract citations
        s3 = make_s3(config)
        citations = extract_citations(rag_result, s3_client=s3, bucket=config.s3_bucket)
        response_text = rag_result.get("output", {}).get("text", "")

        # Step 6: Emit metrics
        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=False,
            citations=citations,
        )

        return _response(200, {
            "response": response_text,
            "citations": citations,
            "language": language,
            "blocked": False,
        })

    except Exception as err:
        print(f"Unhandled error: {err}")
        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=False,
            citations=[],
            failed=True,
        )
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
