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
  description = "Bedrock foundation model for generation"
  type        = string
  default     = "anthropic.claude-sonnet-4-6"
}

variable "embedding_model_id" {
  description = "Bedrock embedding model for Bedrock KB"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}
