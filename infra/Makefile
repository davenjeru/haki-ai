LOCAL_VARS = -var-file=terraform.tfvars.local

# ── Both envs share the s3 backend (haki-ai-terraform-state). Environments
# are isolated via Terraform workspaces:
#   local   → state key  env:/local/terraform.tfstate  (LocalStack resources)
#   default → state key  terraform.tfstate             (real AWS / prod)
# Switch by calling `terraform workspace select local|default` — no reinit.

.PHONY: init local-init local-plan local-apply local-destroy local-output \
        plan apply destroy output

init:
	terraform init -reconfigure

# ── Local (LocalStack) ────────────────────────────────────────────────────────
# Each target selects (or creates) the `local` workspace so the local state
# file lives in the s3 backend under env:/local/.

local-init: init
	terraform workspace select -or-create local

local-plan: local-init
	terraform plan $(LOCAL_VARS)

local-apply: local-init
	terraform apply $(LOCAL_VARS) -auto-approve

local-destroy: local-init
	terraform destroy $(LOCAL_VARS) -auto-approve

local-output: local-init
	terraform output

# ── Prod (real AWS) ───────────────────────────────────────────────────────────
# Prod lives in the `default` workspace. Targets below explicitly select it so
# a previous `workspace select local` doesn't leak into a prod command.
# Provide the LangSmith API key via TF_VAR_langsmith_api_key before apply.

_prod-init: init
	terraform workspace select default

plan: _prod-init
	terraform plan

apply: _prod-init
	terraform apply

destroy: _prod-init
	terraform destroy

output: _prod-init
	terraform output

# ── Web deploy (prod frontend: build + S3 sync + CloudFront invalidation) ────
# Usage:
#   make deploy-web    # from infra/
# Requires: VITE_API_BASE_URL set in frontend/.env.production (committed).

.PHONY: deploy-web

deploy-web: _prod-init
	@BUCKET=$$(terraform output -raw web_bucket); \
	 DIST=$$(terraform output -raw web_distribution_id); \
	 URL=$$(terraform output -raw web_url); \
	 if [ -z "$$BUCKET" ] || [ -z "$$DIST" ]; then \
	   echo "Error: web module outputs are empty. Run 'make apply' first." >&2; exit 1; \
	 fi; \
	 echo ">>> building frontend"; \
	 (cd ../frontend && npm ci --prefer-offline --no-audit --silent && npm run build); \
	 echo ">>> syncing dist/ → s3://$$BUCKET/"; \
	 aws s3 sync ../frontend/dist/ "s3://$$BUCKET/" --delete --only-show-errors \
	   --cache-control "public, max-age=31536000, immutable" \
	   --exclude 'index.html' --exclude '*.html'; \
	 aws s3 sync ../frontend/dist/ "s3://$$BUCKET/" --only-show-errors \
	   --cache-control "public, max-age=0, must-revalidate" \
	   --exclude '*' --include '*.html'; \
	 echo ">>> invalidating CloudFront ($$DIST)"; \
	 aws cloudfront create-invalidation --distribution-id "$$DIST" --paths '/*' --query 'Invalidation.Id' --output text; \
	 echo ">>> deployed to $$URL"
