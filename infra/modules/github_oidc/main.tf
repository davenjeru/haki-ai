############################################################
# GitHub Actions OIDC role
#
# Trust model: GitHub's OIDC provider (already installed in
# the account account-wide) issues a short-lived token to the
# workflow, and this role trusts exactly the repo + branch we
# care about. No long-lived access keys live anywhere.
#
# The `deploy.yml` + `eval-nightly.yml` workflows both assume
# this role via `aws-actions/configure-aws-credentials@v4`.
# `ci.yml` runs `terraform fmt -check` / `validate` which does
# not need AWS creds at all, so it doesn't assume this role.
############################################################

# The OIDC provider is a singleton per account; use a data
# source so we never try to re-create it.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Scope the trust to exactly `refs/heads/<branch>` in the
    # configured repo. This blocks PR builds, forks, and other
    # branches from assuming the deploy role.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        for b in var.allowed_branches :
        "repo:${var.repository}:ref:refs/heads/${b}"
      ]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.project_name}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.assume.json

  description = "Role assumed by GitHub Actions OIDC for Haki AI deploys."
}

# Deploy needs to touch ~every service in the stack (Lambda,
# API Gateway, Bedrock, SageMaker, S3, CloudFront, DynamoDB,
# CloudWatch, EventBridge, SNS, SSM, IAM, Comprehend). For a
# demo project we attach AdministratorAccess; a production
# setup would replace this with a service-scoped policy
# tracked in `infra/modules/github_oidc/policy.json`.
resource "aws_iam_role_policy_attachment" "admin" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
