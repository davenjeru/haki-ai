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
