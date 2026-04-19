# ── IAM execution role ────────────────────────────────────────────────────────

resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs — required for Lambda to write logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # Comprehend — language detection (resource-level ARNs not supported)
      {
        Effect   = "Allow"
        Action   = ["comprehend:DetectDominantLanguage"]
        Resource = "*"
      },
      # CloudWatch — custom metrics for the HakiAI namespace
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      # Bedrock — RAG via Knowledge Base (steps 3–5; stubs until handler is complete)
      {
        Effect = "Allow"
        Action = [
          "bedrock:RetrieveAndGenerate",
          "bedrock:Retrieve",
          "bedrock:InvokeModel",
          "bedrock:ApplyGuardrail",
        ]
        Resource = "*"
      },
    ]
  })
}

# ── Lambda packaging ──────────────────────────────────────────────────────────
# Zips the full backend/ directory. boto3 is already in the Lambda runtime.
# Excludes the uv virtual env, cache, and test files — these must not be deployed.

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../backend"
  output_path = "${path.module}/lambda.zip"

  excludes = [
    ".venv",
    "__pycache__",
    "*.pyc",
    "test_*.py",   # test runners — committed to git, not needed in Lambda
    "*_local.py",  # local-only scripts — committed to git, not needed in Lambda
    "pyproject.toml",
    "uv.lock",
  ]
}

# ── Lambda function ───────────────────────────────────────────────────────────

resource "aws_lambda_function" "handler" {
  function_name    = "${var.project_name}-handler"
  role             = aws_iam_role.lambda_exec.arn
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  # 30s is well within the Bedrock KB p95 latency budget
  timeout     = 30
  memory_size = 512

  environment {
    variables = {
      ENV               = var.environment
      KNOWLEDGE_BASE_ID = var.knowledge_base_id
      GUARDRAIL_ID      = var.guardrail_id
      GUARDRAIL_VERSION = var.guardrail_version
      BEDROCK_MODEL_ID  = var.bedrock_model_id
      CHROMA_HOST       = var.chroma_host
      CHROMA_PORT       = var.chroma_port
    }
  }
}
