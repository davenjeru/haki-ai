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
    embedding_model_id: str
    chroma_host: str
    chroma_port: int
    s3_bucket: str
    checkpoints_table: str
    environment: str
    langsmith_ssm_parameter: str
    use_finetuned_model: bool
    sagemaker_endpoint_name: str


def load_config() -> Config:
    """
    Reads environment variables and returns an immutable Config.

    LOCALSTACK_HOSTNAME is injected by LocalStack into Lambda execution
    environments so the function can reach LocalStack from inside Docker.
    Falls back to 'localhost' for direct local invocation outside Docker.

    CHROMA_HOST / CHROMA_PORT point the local Lambda at the ChromaDB HTTP
    server running on the host machine. In Lambda, use host.docker.internal
    so the container can reach the host. Empty string = no ChromaDB (prod).
    """
    env = os.environ.get("ENV", "prod")
    is_local = env == "local"
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
        embedding_model_id=os.environ.get(
            "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        ),
        chroma_host=os.environ.get("CHROMA_HOST", ""),
        chroma_port=int(os.environ.get("CHROMA_PORT", "8000")),
        s3_bucket=os.environ.get("S3_BUCKET", "haki-ai-data"),
        checkpoints_table=os.environ.get("CHECKPOINTS_TABLE", "haki-ai-checkpoints"),
        environment=env,
        langsmith_ssm_parameter=os.environ.get("LANGSMITH_API_KEY_SSM_PARAMETER", ""),
        use_finetuned_model=os.environ.get("USE_FINETUNED_MODEL", "").lower() in {"1", "true", "yes"},
        sagemaker_endpoint_name=os.environ.get("SAGEMAKER_ENDPOINT_NAME", ""),
    )
