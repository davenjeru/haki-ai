"""
Haki AI — Lambda handler.

Thin entry point that:
  1. Parses the request (message, sessionId).
  2. Invokes the compiled LangGraph with thread_id = sessionId.
  3. Maps the final graph state back to the existing JSON response shape.

All orchestration (language detection, classify_intent, rag_node, chat_node)
now lives in graph.py. Memory is persisted via DynamoDBSaver, keyed by the
frontend-provided sessionId.
"""

import json
import uuid

from clients import make_cloudwatch
from config import load_config
from graph import get_compiled_graph
from metrics import elapsed_ms, emit_metrics, now_ms


def lambda_handler(event, context):
    """
    Main Lambda entry point. Called by API Gateway HTTP API.

    Expected request body: { "message": "<user question>", "sessionId": "<uuid>" }
    Response body:         { "response", "citations", "language", "blocked", "sessionId" }
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

        session_id = (body.get("sessionId") or "").strip() or str(uuid.uuid4())

        graph = get_compiled_graph(config)
        final_state = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"configurable": {"thread_id": session_id}},
        )

        language = final_state.get("language", "english")
        blocked = bool(final_state.get("blocked", False))
        citations = final_state.get("citations", [])
        response_text = final_state.get("response_text", "")

        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=blocked,
            citations=citations,
        )

        return _response(200, {
            "response": response_text,
            "citations": citations,
            "language": language,
            "blocked": blocked,
            "sessionId": session_id,
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
