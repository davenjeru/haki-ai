"""
Boto3 client factory.

All AWS clients are constructed here using Config. No other module
should call boto3.client() directly — add a make_X() function here instead.
"""

import boto3
from config import Config


def _localstack_kwargs(config: Config) -> dict:
    """Common kwargs for any boto3 client pointed at LocalStack."""
    return {
        "endpoint_url": config.localstack_endpoint,
        "region_name": config.aws_region,
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
    }


def make_comprehend(config: Config):
    """
    Comprehend client for language detection.
    Points at LocalStack when is_local=True.
    Note: DetectDominantLanguage is not yet implemented in LocalStack —
    the ComprehendAdapter in adapters.py handles that gracefully.
    """
    if config.is_local:
        return boto3.client("comprehend", **_localstack_kwargs(config))
    return boto3.client("comprehend", region_name=config.aws_region)


def make_bedrock_agent_runtime(config: Config):
    """
    Bedrock Agent Runtime client for retrieve_and_generate (RAG).
    Always points at real AWS — LocalStack does not support Bedrock.
    """
    return boto3.client("bedrock-agent-runtime", region_name=config.aws_region)


def make_bedrock_runtime(config: Config):
    """
    Bedrock Runtime client for direct model invocation (InvokeModel).
    Used locally for Titan embeddings + Claude generation via LocalRAGAdapter.
    Always points at real AWS — LocalStack does not support Bedrock.
    """
    return boto3.client("bedrock-runtime", region_name=config.aws_region)


def make_cloudwatch(config: Config):
    """
    CloudWatch client for emitting custom HakiAI metrics.
    Points at LocalStack when is_local=True.
    """
    if config.is_local:
        return boto3.client("cloudwatch", **_localstack_kwargs(config))
    return boto3.client("cloudwatch", region_name=config.aws_region)
