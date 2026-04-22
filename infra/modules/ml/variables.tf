variable "project_name" { type = string }
variable "data_bucket_id" { type = string }

# Keep the SageMaker endpoint off by default — Terraform still provisions
# IAM + dataset staging so the fine-tune can be kicked off without a
# follow-up apply, but no inference endpoint is billed until the operator
# has a trained adapter ready and explicitly sets this to true.
variable "deploy_endpoint" {
  type        = bool
  default     = false
  description = "Provision the SageMaker serverless endpoint that serves the fine-tuned Llama-3.1-8B adapter."
}
