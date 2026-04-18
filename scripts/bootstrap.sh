#!/usr/bin/env bash
set -euo pipefail

REGION="us-east-1"
STATE_BUCKET="haki-ai-terraform-state"
INFRA_DIR="$(cd "$(dirname "$0")/../infra" && pwd)"

# ── 1. Create Terraform state bucket (idempotent) ─────────────────────────────
if aws s3api head-bucket --bucket "$STATE_BUCKET" 2>/dev/null; then
  echo "State bucket already exists: $STATE_BUCKET"
else
  echo "Creating Terraform state bucket: $STATE_BUCKET"
  aws s3api create-bucket \
    --bucket "$STATE_BUCKET" \
    --region "$REGION"

  # Enable versioning so state history is preserved
  aws s3api put-bucket-versioning \
    --bucket "$STATE_BUCKET" \
    --versioning-configuration Status=Enabled

  # Block all public access
  aws s3api put-public-access-block \
    --bucket "$STATE_BUCKET" \
    --public-access-block-configuration \
      BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

  echo "State bucket ready."
fi

# ── 2. Terraform init + apply ─────────────────────────────────────────────────
cd "$INFRA_DIR"
terraform init -reconfigure
terraform apply "$@"
