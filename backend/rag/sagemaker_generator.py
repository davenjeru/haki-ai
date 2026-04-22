"""
SageMaker-backed generator for the fine-tuned Llama-3.1-8B adapter.

Invoked by the RAG pipeline when ``USE_FINETUNED_MODEL=true`` and
``SAGEMAKER_ENDPOINT_NAME`` is set. Mirrors the signature of the Bedrock
``generate`` function so callers can swap them behind the same adapter
method.

Design notes
------------
- We call the endpoint with the JumpStart / LMI DJL-Inference JSON shape:
  ``{"inputs": "<prompt>", "parameters": {...}}``. The training script
  bundled with the Terraform module in ``infra/modules/ml`` trains a LoRA
  adapter on the JumpStart base image which expects this exact shape.
- Any exception is caught by the adapter layer, which falls back to
  Bedrock Claude. That is important because the endpoint is deployed in
  serverless mode and can cold-start with a 60s latency on the first hit
  of the day — we never want that to block a user query.
"""

from __future__ import annotations

import json


_MAX_OUTPUT_TOKENS = 1024


def _format_prompt(system_prompt: str, context: str, query: str) -> str:
    """Formats a Llama-3-Instruct-style prompt with optional <context>."""
    user_content = f"<context>\n{context}\n</context>\n\n{query}" if context else query
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate(
    query: str,
    system_prompt: str,
    context: str,
    endpoint_name: str,
    sagemaker_runtime,
    *,
    max_tokens: int = _MAX_OUTPUT_TOKENS,
) -> tuple[str, str]:
    """
    Invokes the SageMaker endpoint and returns ``(text, stop_reason)``.

    stop_reason mirrors the Bedrock stop_reason vocabulary so downstream
    code can branch the same way for either backend:
      - ``"end_turn"``  → normal completion
      - ``"max_tokens"``→ truncated at `max_tokens`
    """
    prompt = _format_prompt(system_prompt, context, query)
    body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
            "stop": ["<|eot_id|>"],
        },
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(body),
    )
    payload = json.loads(response["Body"].read())

    text = ""
    if isinstance(payload, list) and payload:
        text = (payload[0] or {}).get("generated_text", "") or ""
    elif isinstance(payload, dict):
        text = payload.get("generated_text", "") or payload.get("output", "") or ""

    text = text.strip()
    stop_reason = "max_tokens" if text and len(text.split()) >= max_tokens else "end_turn"
    return text, stop_reason
