#!/usr/bin/env bash
# Kick off the Haki AI fine-tune job.
#
# Pre-reqs:
#   1. Phase 4a crawler has already uploaded faq-chunks/** to the data bucket.
#   2. `aws sts get-caller-identity` works and the role has access to the
#      SageMaker JumpStart Llama-3.1-8B-Instruct model (you must have
#      accepted the model licence once in the AWS console).
#
# This script:
#   1. Builds the JSONL dataset and uploads it to s3://…/models/haki-ai-finetune/dataset/train.jsonl
#   2. Starts a SageMaker JumpStart qLoRA training job against that dataset
#   3. Waits for completion and prints the S3 URI of the trained adapter
#
# After the job finishes, flip `deploy_endpoint = true` in Terraform
# (or pass `-var deploy_endpoint=true`) and re-apply to spin up the
# serverless inference endpoint. Then set USE_FINETUNED_MODEL=true and
# SAGEMAKER_ENDPOINT_NAME=<project>-finetuned in the backend env to route
# the "tone" answers through the fine-tune.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-haki-ai}"
BUCKET_NAME="${DATA_BUCKET:-${PROJECT_NAME}-data}"
JOB_NAME="${PROJECT_NAME}-finetune-$(date +%Y%m%d-%H%M%S)"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g5.2xlarge}"

ROLE_ARN="$(aws iam get-role --role-name "${PROJECT_NAME}-sagemaker" \
  --query 'Role.Arn' --output text)"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[run-finetune] 1/3 building training JSONL"
cd "$REPO_ROOT/backend"
uv run python -m scripts.prepare_finetune_data --bucket "$BUCKET_NAME"

DATASET_URI="s3://${BUCKET_NAME}/models/haki-ai-finetune/dataset/"
OUTPUT_URI="s3://${BUCKET_NAME}/models/haki-ai-finetune/output/"

echo "[run-finetune] 2/3 starting SageMaker training job ${JOB_NAME}"
cat > /tmp/finetune-params.json <<EOF
{
  "TrainingJobName": "${JOB_NAME}",
  "RoleArn": "${ROLE_ARN}",
  "AlgorithmSpecification": {
    "TrainingImage": "763104351884.dkr.ecr.${REGION}.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04",
    "TrainingInputMode": "File"
  },
  "HyperParameters": {
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "epoch": "3",
    "learning_rate": "0.0002",
    "per_device_train_batch_size": "1",
    "gradient_accumulation_steps": "4",
    "lora_r": "16",
    "lora_alpha": "32",
    "peft_type": "lora",
    "load_in_4bit": "true",
    "max_input_length": "2048"
  },
  "InputDataConfig": [
    {
      "ChannelName": "training",
      "DataSource": {
        "S3DataSource": {
          "S3DataType": "S3Prefix",
          "S3Uri": "${DATASET_URI}",
          "S3DataDistributionType": "FullyReplicated"
        }
      },
      "ContentType": "application/jsonl"
    }
  ],
  "OutputDataConfig": {
    "S3OutputPath": "${OUTPUT_URI}"
  },
  "ResourceConfig": {
    "InstanceType": "${INSTANCE_TYPE}",
    "InstanceCount": 1,
    "VolumeSizeInGB": 50
  },
  "StoppingCondition": {
    "MaxRuntimeInSeconds": 14400
  }
}
EOF

aws sagemaker create-training-job --cli-input-json file:///tmp/finetune-params.json --region "$REGION"

echo "[run-finetune] 3/3 waiting for ${JOB_NAME} …"
aws sagemaker wait training-job-completed-or-stopped --training-job-name "$JOB_NAME" --region "$REGION"

STATUS="$(aws sagemaker describe-training-job \
  --training-job-name "$JOB_NAME" --region "$REGION" \
  --query 'TrainingJobStatus' --output text)"

echo "[run-finetune] training job finished with status: $STATUS"
if [[ "$STATUS" != "Completed" ]]; then
  exit 1
fi

echo "[run-finetune] adapter artefact: ${OUTPUT_URI}"
echo "[run-finetune] next: terraform apply -var deploy_endpoint=true"
