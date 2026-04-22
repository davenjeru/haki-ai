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

# NOTE: observability MUST be imported and bootstrapped before `graph` so
# LANGSMITH_API_KEY lands in the environment before LangChain initialises
# its tracer. Importing graph triggers the langchain import chain.
from observability import bootstrap_langsmith, run_traced_turn

bootstrap_langsmith(load_config())

from graph import get_compiled_graph, load_history  # noqa: E402
from metrics import elapsed_ms, emit_metrics, now_ms  # noqa: E402


def lambda_handler(event, context):
    """
    Main Lambda entry point. Called by API Gateway HTTP API.

    Routes:
      POST /chat                     → chat turn
      GET  /chat/history?sessionId=  → full persisted conversation
    """
    method, path = _method_and_path(event)
    if method == "GET" and path.rstrip("/") == "/chat/history":
        return _handle_history(event)
    return _handle_chat(event)


def _handle_chat(event):
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
        final_state = run_traced_turn(
            graph,
            {"messages": [{"role": "user", "content": user_message}]},
            {"configurable": {"thread_id": session_id}},
            session_id=session_id,
            env=config.environment,
            message_length=len(user_message),
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
        print(f"Unhandled error (chat): {err}")
        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=False,
            citations=[],
            failed=True,
        )
        return _response(500, {"error": "Internal server error"})


def _handle_history(event):
    """
    Returns { messages: [{id, role, content, citations?, language?, blocked?}], sessionId }.
    Safe for brand-new sessions — returns an empty list.
    """
    try:
        config = load_config()
        params = (event.get("queryStringParameters") or {})
        session_id = (params.get("sessionId") or "").strip()
        if not session_id:
            return _response(400, {"error": "sessionId is required"})
        messages = load_history(config, session_id)
        return _response(200, {"messages": messages, "sessionId": session_id})
    except Exception as err:
        print(f"Unhandled error (history): {err}")
        return _response(500, {"error": "Internal server error"})


def _method_and_path(event) -> tuple[str, str]:
    """
    Extracts (method, path) from an API Gateway v2 HTTP event. Falls back to
    top-level keys when called from server_local.py which hand-rolls the event.
    """
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method") or event.get("httpMethod") or event.get("method") or "POST"
    path = http.get("path") or event.get("rawPath") or event.get("path") or "/chat"
    return method.upper(), path


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }
