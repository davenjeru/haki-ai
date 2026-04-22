"""
Boto3 client factory.

All AWS clients are constructed here using Config. No other module
should call boto3.client() directly — add a make_X() function here instead.
"""

import boto3
from botocore.config import Config as BotoConfig
from app.config import Config


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


def make_ssm(config: Config):
    """
    SSM client used to fetch the LangSmith API key SecureString at Lambda
    cold start. Routes to LocalStack when is_local=True so local dev can
    mirror the prod path without calling real AWS.
    """
    if config.is_local:
        return boto3.client("ssm", **_localstack_kwargs(config))
    return boto3.client("ssm", region_name=config.aws_region)


def make_dynamodb_table(config: Config, table_name: str):
    """
    DynamoDB Table resource for the LangGraph checkpoint store.

    Returns a boto3 Table resource (not a client) because its higher-level
    serialization handles binary attributes and Python types directly —
    the checkpointer code stays concise.

    Points at LocalStack when is_local=True.
    """
    import boto3  # re-import for clarity: resource != client
    if config.is_local:
        return boto3.resource(
            "dynamodb",
            endpoint_url=config.localstack_endpoint,
            region_name=config.aws_region,
            aws_access_key_id="test",
            aws_secret_access_key="test",
        ).Table(table_name)
    return boto3.resource("dynamodb", region_name=config.aws_region).Table(table_name)


def make_sagemaker_runtime(config: Config):
    """
    SageMaker Runtime client used to call the fine-tuned Llama-3.1-8B
    endpoint when USE_FINETUNED_MODEL=true. LocalStack does not support
    SageMaker inference, so this always targets real AWS — the backend
    falls back to Bedrock automatically on any error.
    """
    return boto3.client("sagemaker-runtime", region_name=config.aws_region)


def make_s3(config: Config):
    """
    S3 client used for generating presigned GET URLs for per-page PDFs
    stored under page-images/ in the haki-ai data bucket.

    Path-style addressing is used locally so presigned URLs point at
    http://localhost:4566/<bucket>/<key> (browser-reachable from the host)
    rather than the virtual-host form (<bucket>.localhost:4566) which does
    not resolve via DNS. signature_version=s3v4 is required for LocalStack
    presigning to match what boto3 signs.
    """
    boto_cfg = BotoConfig(
        s3={"addressing_style": "path"},
        signature_version="s3v4",
    )
    if config.is_local:
        return boto3.client("s3", config=boto_cfg, **_localstack_kwargs(config))
    return boto3.client("s3", region_name=config.aws_region, config=boto_cfg)
