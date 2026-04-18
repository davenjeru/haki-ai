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
