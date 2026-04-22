variable "project_name" {
  description = "Prefix for the IAM role name."
  type        = string
}

variable "repository" {
  description = "GitHub repo in `owner/name` form. The assume-role trust policy is locked to this value so other repos cannot assume this role."
  type        = string
}

variable "allowed_branches" {
  description = "List of branch names on `var.repository` that may assume this role. Keep this tight — it gates who can deploy."
  type        = list(string)
  default     = ["main"]
}
