output "role_arn" {
  description = "ARN to paste into the GitHub Actions `AWS_ROLE_TO_ASSUME` Variable."
  value       = aws_iam_role.github_actions.arn
}

output "role_name" {
  description = "Name of the role — useful for debugging."
  value       = aws_iam_role.github_actions.name
}
