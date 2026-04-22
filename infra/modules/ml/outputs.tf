output "sagemaker_role_arn" {
  description = "ARN of the role shared by the training job and endpoint."
  value       = aws_iam_role.sagemaker.arn
}

output "training_dataset_prefix" {
  description = "S3 prefix (inside the data bucket) where the fine-tune JSONL is uploaded."
  value       = "models/haki-ai-finetune/dataset/"
}

output "training_output_prefix" {
  description = "S3 prefix where SageMaker writes the trained adapter artefact."
  value       = "models/haki-ai-finetune/output/"
}

output "endpoint_name" {
  description = "Name of the SageMaker Serverless endpoint (only populated when deploy_endpoint = true)."
  value       = try(aws_sagemaker_endpoint.finetuned[0].name, "")
}
