"""
CloudWatch custom metrics — step 6.

emit_metrics() publishes to the HakiAI namespace. All metric names match
the CloudWatch alarms defined in infra/modules/observability/.

Metrics emitted per request:
  - SuccessfulRequests or FailedRequests (count)
  - ResponseLatency (milliseconds)
  - DetectedLanguage_english / _swahili / _mixed (count)
  - GuardrailBlock (count, only when blocked=True)
  - MissingCitations (count, only when citations=[])

When is_local=True the CloudWatch client points at LocalStack. LocalStack
supports PutMetricData, so metrics are emitted in both environments.
"""

import time

NAMESPACE = "HakiAI"


def emit_metrics(
    cloudwatch,
    *,
    language: str,
    latency_ms: float,
    blocked: bool,
    citations: list,
    failed: bool = False,
) -> None:
    """
    Publishes custom metrics for a single Lambda invocation.

    Args:
        cloudwatch:  boto3 CloudWatch client (from make_cloudwatch(config)).
        language:    "english", "swahili", or "mixed".
        latency_ms:  Wall-clock milliseconds from request start to RAG response.
        blocked:     True when the guardrail fired.
        citations:   The citations list from extract_citations() — used to detect
                     missing citations.
        failed:      True when the invocation raised an unhandled exception.
    """
    metric_data = [
        _count("FailedRequests" if failed else "SuccessfulRequests"),
        _value("ResponseLatency", latency_ms, unit="Milliseconds"),
        _count(f"DetectedLanguage_{language}"),
    ]

    if blocked:
        metric_data.append(_count("GuardrailBlock"))

    if not citations:
        metric_data.append(_count("MissingCitations"))

    cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)


def now_ms() -> float:
    """Returns current time in milliseconds. Call before the RAG step."""
    return time.monotonic() * 1000


def elapsed_ms(start_ms: float) -> float:
    """Returns milliseconds elapsed since start_ms."""
    return time.monotonic() * 1000 - start_ms


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count(name: str) -> dict:
    return {"MetricName": name, "Value": 1, "Unit": "Count"}


def _value(name: str, value: float, unit: str = "None") -> dict:
    return {"MetricName": name, "Value": value, "Unit": unit}
