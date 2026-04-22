terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.41.0"
    }
  }

  # Remote state — used for prod only.
  # For local runs use: terraform init -backend=false
  # Create manually before prod init:
  #   aws s3api create-bucket --bucket haki-ai-terraform-state --region us-east-1
  backend "s3" {
    bucket = "haki-ai-terraform-state"
    key    = "terraform.tfstate"
    region = "us-east-1"
  }
}

locals {
  is_local       = var.environment == "local"
  localstack_url = "http://localhost:4566"
}

provider "aws" {
  region = var.aws_region

  # LocalStack overrides — no-ops when environment = "prod"
  access_key                  = local.is_local ? "test" : null
  secret_key                  = local.is_local ? "test" : null
  skip_credentials_validation = local.is_local
  skip_metadata_api_check     = local.is_local
  skip_requesting_account_id  = local.is_local
  s3_use_path_style           = local.is_local

  dynamic "endpoints" {
    for_each = local.is_local ? [1] : []
    content {
      s3             = local.localstack_url
      iam            = local.localstack_url
      lambda         = local.localstack_url
      apigateway     = local.localstack_url
      apigatewayv2   = local.localstack_url
      cloudwatch     = local.localstack_url
      cloudwatchlogs = local.localstack_url
      sns            = local.localstack_url
      sts            = local.localstack_url
      dynamodb       = local.localstack_url
      ssm            = local.localstack_url
    }
  }
}

# ── Storage: S3 data bucket + S3 Vectors bucket ──────────────────────────────
# S3 Vectors is not supported by LocalStack — skipped locally.

module "storage" {
  source       = "./modules/storage"
  project_name = var.project_name
  aws_region   = var.aws_region
  is_local     = local.is_local
}

# ── LangSmith API key (SSM SecureString) ─────────────────────────────────────
# Lives at the root so both the compute module (Lambda IAM + env var) and
# anything else that might need it can depend on it without creating a cycle
# through the observability module. Only created when a key is supplied —
# an empty var.langsmith_api_key disables tracing cleanly.

resource "aws_ssm_parameter" "langsmith_api_key" {
  count = var.langsmith_api_key == "" ? 0 : 1

  name        = "/${var.project_name}/langsmith/api-key"
  description = "LangSmith API key used by the Lambda for tracing."
  type        = "SecureString"
  value       = var.langsmith_api_key
}

locals {
  langsmith_ssm_parameter_name = length(aws_ssm_parameter.langsmith_api_key) > 0 ? aws_ssm_parameter.langsmith_api_key[0].name : ""
  langsmith_ssm_parameter_arn  = length(aws_ssm_parameter.langsmith_api_key) > 0 ? aws_ssm_parameter.langsmith_api_key[0].arn : ""
}

# ── AI: Bedrock KB + Guardrails ───────────────────────────────────────────────
# Bedrock is not supported by LocalStack — skipped locally.

module "ai" {
  count = local.is_local ? 0 : 1

  source             = "./modules/ai"
  project_name       = var.project_name
  aws_region         = var.aws_region
  data_bucket_arn    = module.storage.data_bucket_arn
  data_bucket_id     = module.storage.data_bucket_id
  vector_bucket_arn  = module.storage.vector_bucket_arn
  vector_index_arn   = module.storage.vector_index_arn
  embedding_model_id = var.embedding_model_id
}

# ── Compute: Lambda ───────────────────────────────────────────────────────────

module "compute" {
  source                       = "./modules/compute"
  project_name                 = var.project_name
  aws_region                   = var.aws_region
  environment                  = var.environment
  knowledge_base_id            = local.is_local ? "" : module.ai[0].knowledge_base_id
  guardrail_id                 = local.is_local ? "" : module.ai[0].guardrail_id
  guardrail_version            = local.is_local ? "DRAFT" : module.ai[0].guardrail_version
  bedrock_model_id             = var.bedrock_model_id
  chroma_host                  = var.chroma_host
  chroma_port                  = var.chroma_port
  checkpoints_table_name       = module.storage.checkpoints_table_name
  checkpoints_table_arn        = module.storage.checkpoints_table_arn
  data_bucket_arn              = module.storage.data_bucket_arn
  data_bucket_name             = module.storage.data_bucket_id
  ingestion_kb_id              = local.is_local ? "" : module.ai[0].knowledge_base_id
  ingestion_data_source_id     = local.is_local ? "" : module.ai[0].data_source_id
  langsmith_ssm_parameter_name = local.langsmith_ssm_parameter_name
  langsmith_ssm_parameter_arn  = local.langsmith_ssm_parameter_arn
  langsmith_project            = var.langsmith_project
  langsmith_endpoint           = var.langsmith_endpoint
}

# ── API: API Gateway HTTP ─────────────────────────────────────────────────────

module "api" {
  source            = "./modules/api"
  project_name      = var.project_name
  lambda_arn        = module.compute.lambda_arn
  lambda_invoke_arn = module.compute.lambda_invoke_arn
}

# ── Observability: CloudWatch + SNS ──────────────────────────────────────────

module "observability" {
  source       = "./modules/observability"
  project_name = var.project_name
  alert_email  = var.alert_email
  lambda_name  = module.compute.lambda_name
}

# ── ML: SageMaker fine-tuning (future) ───────────────────────────────────────
# SageMaker is not supported by LocalStack — skipped locally.

module "ml" {
  count = local.is_local ? 0 : 1

  source         = "./modules/ml"
  project_name   = var.project_name
  data_bucket_id = module.storage.data_bucket_id
}

# ── Web: S3 + CloudFront hosting for the built Vite SPA ──────────────────────
# LocalStack's CloudFront support is limited, so the web module is prod-only.
# For local frontend dev, `npm run dev` hits the Vite dev server directly.

module "web" {
  count = local.is_local ? 0 : 1

  source       = "./modules/web"
  project_name = var.project_name
  price_class  = var.cloudfront_price_class
}
