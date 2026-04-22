variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment: 'local' (LocalStack) or 'prod' (real AWS)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["local", "prod"], var.environment)
    error_message = "environment must be 'local' or 'prod'."
  }
}

variable "project_name" {
  description = "Project prefix used in all resource names"
  type        = string
  default     = "haki-ai"
}

variable "alert_email" {
  description = "Email address for CloudWatch alarm notifications"
  type        = string
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model for generation. Must be an inference-profile ID (prefixed `us.`) since Claude 4.x models no longer support on-demand throughput."
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "embedding_model_id" {
  description = "Bedrock embedding model for Bedrock KB"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "chroma_host" {
  description = "ChromaDB HTTP server host for local Lambda → ChromaDB access. Empty in prod."
  type        = string
  default     = ""
}

variable "chroma_port" {
  description = "ChromaDB HTTP server port"
  type        = string
  default     = "8000"
}

# ── LangSmith observability ──────────────────────────────────────────────────
# The API key is stored in SSM Parameter Store (SecureString) so it stays out
# of Terraform state once applied. Provide it via `TF_VAR_langsmith_api_key`
# in CI, or a gitignored terraform.tfvars.local during manual applies.
# Leave empty to disable tracing entirely — the Lambda code no-ops gracefully.

variable "langsmith_api_key" {
  description = "LangSmith API key. Empty string disables tracing."
  type        = string
  sensitive   = true
  default     = ""
}

variable "langsmith_project" {
  description = "LangSmith project name that traces are filed under."
  type        = string
  default     = "haki-ai"
}

variable "langsmith_endpoint" {
  description = "LangSmith API endpoint (US region by default)."
  type        = string
  default     = "https://api.smith.langchain.com"
}
