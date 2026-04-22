variable "project_name" {
  description = "Prefix for the IAM role name."
  type        = string
}

variable "repository" {
  description = "GitHub repo in `owner/name` form. The assume-role trust policy is locked to this value so other repos cannot assume this role."
  type        = string
}

variable "allowed_branches" {
  description = "List of branch names on `var.repository` that may assume this role via a plain (non-environment) job. Keep this tight — it gates who can deploy."
  type        = list(string)
  default     = ["main"]
}

variable "allowed_environments" {
  description = "List of GitHub Environment names on `var.repository` that may assume this role. Jobs using `jobs.<id>.environment: <name>` get OIDC tokens whose `sub` claim matches `repo:<r>:environment:<name>` — those MUST be listed here for the workflow to assume the role."
  type        = list(string)
  default     = ["production"]
}
