output "data_bucket_id" {
  description = "S3 bucket for raw PDFs and processed chunks"
  value       = module.storage.data_bucket_id
}

output "knowledge_base_id" {
  description = "Bedrock Knowledge Base ID — used by Lambda and start_ingestion_job"
  value       = length(module.ai) > 0 ? module.ai[0].knowledge_base_id : ""
}

output "data_source_id" {
  description = "Bedrock KB data source ID — used by start_ingestion_job"
  value       = length(module.ai) > 0 ? module.ai[0].data_source_id : ""
}

output "guardrail_id" {
  description = "Bedrock Guardrail ID"
  value       = length(module.ai) > 0 ? module.ai[0].guardrail_id : ""
}

output "api_endpoint" {
  description = "API Gateway endpoint URL"
  value       = module.api.api_endpoint
}

output "lambda_name" {
  description = "Lambda function name"
  value       = module.compute.lambda_name
}

# ── Web distribution ─────────────────────────────────────────────────────────

output "web_bucket" {
  description = "S3 bucket the built frontend is synced to"
  value       = length(module.web) > 0 ? module.web[0].site_bucket_name : ""
}

output "web_distribution_id" {
  description = "CloudFront distribution ID (used for cache invalidation after a deploy)"
  value       = length(module.web) > 0 ? module.web[0].distribution_id : ""
}

output "web_url" {
  description = "Public URL of the deployed frontend"
  value       = length(module.web) > 0 ? "https://${module.web[0].distribution_domain_name}" : ""
}

# ── CI/CD ────────────────────────────────────────────────────────────────────

output "github_actions_role_arn" {
  description = "IAM role ARN for the GitHub Actions deploy workflow. Paste into the `AWS_ROLE_TO_ASSUME` repo Variable."
  value       = length(module.github_oidc) > 0 ? module.github_oidc[0].role_arn : ""
}
