output "knowledge_base_id" {
  value = aws_bedrockagent_knowledge_base.this.id
}

output "data_source_id" {
  value = aws_bedrockagent_data_source.laws.data_source_id
}

output "guardrail_id" {
  value = aws_bedrock_guardrail.this.guardrail_id
}

output "guardrail_arn" {
  value = aws_bedrock_guardrail.this.guardrail_arn
}

output "guardrail_version" {
  value = aws_bedrock_guardrail_version.this.version
}
