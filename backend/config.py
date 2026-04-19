"""
Single source of truth for runtime configuration.

All environment variable reads happen here. The rest of the codebase
imports Config and never calls os.environ directly.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    is_local: bool
    localstack_endpoint: str
    aws_region: str
    knowledge_base_id: str
    guardrail_id: str
    guardrail_version: str
    bedrock_model_id: str


def load_config() -> Config:
    """
    Reads environment variables and returns an immutable Config.

    LOCALSTACK_HOSTNAME is injected by LocalStack into Lambda execution
    environments so the function can reach LocalStack from inside Docker.
    Falls back to 'localhost' for direct local invocation outside Docker.
    """
    is_local = os.environ.get("ENV") == "local"
    host = os.environ.get("LOCALSTACK_HOSTNAME", "localhost")
    port = os.environ.get("EDGE_PORT", "4566")

    return Config(
        is_local=is_local,
        localstack_endpoint=f"http://{host}:{port}",
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        knowledge_base_id=os.environ.get("KNOWLEDGE_BASE_ID", ""),
        guardrail_id=os.environ.get("GUARDRAIL_ID", ""),
        guardrail_version=os.environ.get("GUARDRAIL_VERSION", "DRAFT"),
        bedrock_model_id=os.environ.get("BEDROCK_MODEL_ID", ""),
    )
