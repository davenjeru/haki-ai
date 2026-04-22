############################################################
# ML: SageMaker qLoRA fine-tune + serverless inference endpoint
#
# Phase 4b — we fine-tune Llama-3.1-8B-Instruct (via JumpStart)
# on the SheriaPlex FAQ corpus to teach the model the "tone" a
# Kenyan legal-aid assistant should use. The trained adapter is
# written to s3://<data>/models/haki-ai-finetune/ and served on
# a SageMaker Serverless Inference endpoint so it scales to zero
# between eval runs.
#
# The backend picks up the endpoint via USE_FINETUNED_MODEL=true +
# SAGEMAKER_ENDPOINT_NAME; generator.py falls back to Claude on
# Bedrock on any error so the production path degrades gracefully.
############################################################

locals {
  # Toggle deploy_endpoint=true once you've run the training job
  # (see scripts/run-finetune.sh) and have a model artefact in S3.
  # Keeping it off by default keeps `terraform apply` idempotent and
  # cheap for reviewers who just want to inspect the IaC.
  deploy_endpoint = var.deploy_endpoint

  training_prefix = "models/haki-ai-finetune/"
  dataset_prefix  = "models/haki-ai-finetune/dataset/"
  model_data_url  = "s3://${var.data_bucket_id}/${local.training_prefix}output/model.tar.gz"
}

# ── IAM role shared by training jobs + endpoints ────────────────────────────

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project_name}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

# SageMaker needs S3 r/w on the data bucket (for training data in, model
# artefacts out) and CloudWatch Logs writes for training + inference.
data "aws_iam_policy_document" "sagemaker_perms" {
  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${var.data_bucket_id}",
      "arn:aws:s3:::${var.data_bucket_id}/*",
    ]
  }

  statement {
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }

  # Needed by JumpStart training images that pull the base model from
  # SageMaker's public container registry.
  statement {
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "${var.project_name}-sagemaker-inline"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sagemaker_perms.json
}

# ── Dataset staging path (for visibility in the console) ──────────────────

resource "aws_s3_object" "dataset_readme" {
  bucket  = var.data_bucket_id
  key     = "${local.dataset_prefix}README.md"
  content = <<-EOT
    # Haki AI — fine-tune dataset

    This prefix receives the SheriaPlex JSONL produced by
    `backend/scripts/prepare_finetune_data.py`. Each line is a record of
    the form `{"instruction": ..., "input": ..., "output": ...}`.

    Run the training job with:

        scripts/run-finetune.sh

    The job writes the LoRA adapter to
    `s3://${var.data_bucket_id}/${local.training_prefix}output/model.tar.gz`.
  EOT

  content_type = "text/markdown"
}

# ── Inference model + serverless endpoint (opt-in) ────────────────────────

# JumpStart hosts the Llama-3.1-8B image; the exact URI is region-specific
# and can change over time. Point at the container by data lookup rather
# than hardcoding — if the data source fails we skip endpoint creation
# cleanly instead of breaking `terraform apply`.
data "aws_sagemaker_prebuilt_ecr_image" "llama" {
  count           = local.deploy_endpoint ? 1 : 0
  repository_name = "djl-inference"
  image_tag       = "0.28.0-lmi10.0.0-cu124"
}

resource "aws_sagemaker_model" "finetuned" {
  count = local.deploy_endpoint ? 1 : 0

  name               = "${var.project_name}-finetuned-llama"
  execution_role_arn = aws_iam_role.sagemaker.arn

  primary_container {
    image          = data.aws_sagemaker_prebuilt_ecr_image.llama[0].registry_path
    model_data_url = local.model_data_url

    environment = {
      HF_MODEL_ID                = "meta-llama/Meta-Llama-3.1-8B-Instruct"
      OPTION_QUANTIZE            = "awq"
      OPTION_MAX_MODEL_LEN       = "4096"
      OPTION_ENABLE_LORA         = "true"
      OPTION_LORA_ADAPTERS_S3URL = "s3://${var.data_bucket_id}/${local.training_prefix}output/"
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "finetuned" {
  count = local.deploy_endpoint ? 1 : 0

  name = "${var.project_name}-finetuned-config"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.finetuned[0].name

    serverless_config {
      # Scale-to-zero by default — keeps eval runs cheap and dev-friendly.
      # Bump memory + concurrency once the endpoint is promoted to the
      # live request path (see USE_FINETUNED_MODEL toggle in backend).
      max_concurrency   = 2
      memory_size_in_mb = 6144
    }
  }
}

resource "aws_sagemaker_endpoint" "finetuned" {
  count = local.deploy_endpoint ? 1 : 0

  name                 = "${var.project_name}-finetuned"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.finetuned[0].name
}
