variable "project_name" { type = string }
variable "aws_region" { type = string }
variable "environment" { type = string }
variable "knowledge_base_id" { type = string }
variable "guardrail_id" { type = string }
variable "guardrail_version" { type = string }
variable "bedrock_model_id" { type = string }
variable "chroma_host" {
  type    = string
  default = ""
}
variable "chroma_port" {
  type    = string
  default = "8000"
}
variable "checkpoints_table_name" {
  description = "DynamoDB table backing the LangGraph checkpoint store."
  type        = string
}
variable "checkpoints_table_arn" {
  description = "ARN of the DynamoDB checkpoints table (used for scoping IAM)."
  type        = string
}

variable "chat_threads_table_name" {
  description = "DynamoDB table holding the per-user thread index (signed-in sidebar)."
  type        = string
}
variable "chat_threads_table_arn" {
  description = "ARN of the chat-threads DynamoDB table (used for scoping IAM)."
  type        = string
}

variable "clerk_publishable_key" {
  description = "Clerk publishable key. The backend uses it to auto-derive the issuer and JWKS URL for JWT verification."
  type        = string
  default     = ""
}

variable "data_bucket_arn" {
  description = "ARN of the S3 data bucket. Lambda needs GetObject on page-images/* so presigned citation URLs resolve."
  type        = string
}

# ── LangSmith tracing ────────────────────────────────────────────────────────

variable "langsmith_ssm_parameter_name" {
  description = "Name of the SSM SecureString holding the LangSmith API key. Empty disables tracing."
  type        = string
  default     = ""
}

variable "langsmith_ssm_parameter_arn" {
  description = "ARN of the SSM SecureString. Used to scope Lambda IAM. Empty disables tracing."
  type        = string
  default     = ""
}

variable "langsmith_project" {
  description = "LangSmith project name. Only used when tracing is enabled."
  type        = string
  default     = "haki-ai"
}

variable "langsmith_endpoint" {
  description = "LangSmith API endpoint. Only used when tracing is enabled."
  type        = string
  default     = "https://api.smith.langchain.com"
}

# ── Auto-ingestion trigger (Phase 5b) ────────────────────────────────────────

variable "data_bucket_name" {
  description = "Name (not ARN) of the data bucket. Used to scope the EventBridge rule that fires ingestion jobs on S3 uploads."
  type        = string
  default     = ""
}

variable "ingestion_kb_id" {
  description = "Knowledge Base id to run ingestion against. Empty disables the auto-ingestion trigger (e.g. local runs)."
  type        = string
  default     = ""
}

variable "ingestion_data_source_id" {
  description = "Bedrock KB data-source id. Empty disables the auto-ingestion trigger."
  type        = string
  default     = ""
}
