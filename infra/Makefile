LOCAL_VARS = -var-file=terraform.tfvars.local

# ── Local (LocalStack) ────────────────────────────────────────────────────────

.PHONY: local-init local-plan local-apply local-destroy local-output

local-init:
	@printf 'terraform {\n  backend "local" {\n    path = "local.tfstate"\n  }\n}\n' > local_backend_override.tf
	terraform init -reconfigure

local-plan: local-init
	terraform plan $(LOCAL_VARS)

local-apply: local-init
	terraform apply $(LOCAL_VARS) -auto-approve

local-destroy: local-init
	terraform destroy $(LOCAL_VARS) -auto-approve

local-output:
	terraform output

# ── Prod (real AWS) ───────────────────────────────────────────────────────────

.PHONY: init plan apply destroy

init:
	terraform init

plan:
	terraform plan

apply:
	terraform apply

destroy:
	terraform destroy
