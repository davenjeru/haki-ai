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
