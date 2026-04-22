variable "project_name" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "is_local" {
  description = "When true, skips S3 Vectors resources (not supported by LocalStack)"
  type        = bool
  default     = false
}

variable "cors_allowed_origins" {
  description = "Origins allowed to GET page-image PDFs from the data bucket (browser CORS)."
  type        = list(string)
  default     = ["*"]
}
