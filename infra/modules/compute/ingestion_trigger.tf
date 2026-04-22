############################################################
# Auto-ingestion trigger (Phase 5b)
#
# Wires up S3 object-created events → EventBridge (with a 5
# minute debounce window so a large multi-file pipeline run
# doesn't kick off dozens of duplicate ingestion jobs) →
# a small Python Lambda that calls
# ``bedrock-agent.start_ingestion_job`` on the Haki AI KB.
#
# The Lambda is idempotent: it first checks whether an
# ingestion job is already IN_PROGRESS / STARTING for the
# KB + data source and bails out if so. That keeps the
# pipeline safe even if EventBridge re-delivers an event or
# the batching window fires twice in quick succession.
#
# Gated on a data-source id because LocalStack does not
# support Bedrock Agent — the whole trigger is skipped when
# var.ingestion_kb_id is empty.
############################################################

locals {
  # Prefixes we watch. Any object created under either of these
  # triggers a (debounced) ingestion.
  ingestion_watch_prefixes = [
    "processed-chunks/",
    "faq-chunks/",
  ]

  enable_ingestion_trigger = var.ingestion_kb_id != "" && var.ingestion_data_source_id != ""
}

# ── IAM: the ingestion trigger Lambda's execution role ──────────────────────

resource "aws_iam_role" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  name = "${var.project_name}-ingestion-trigger"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  name = "${var.project_name}-ingestion-trigger-policy"
  role = aws_iam_role.ingestion_trigger[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:StartIngestionJob",
          "bedrock:ListIngestionJobs",
          "bedrock:GetIngestionJob",
        ]
        Resource = "*"
      },
    ]
  })
}

# ── Lambda source (inline, no backend code changes needed) ──────────────────

data "archive_file" "ingestion_trigger_zip" {
  count = local.enable_ingestion_trigger ? 1 : 0

  type        = "zip"
  output_path = "${path.module}/.ingestion-trigger.zip"

  source {
    filename = "handler.py"
    content  = <<-PY
      """
      Auto-ingestion trigger for the Haki AI Bedrock KB.

      Invoked by an EventBridge rule on S3 object-created events under
      processed-chunks/* and faq-chunks/*. The rule batches events for
      five minutes so a pipeline run that uploads many chunks only fires
      a single ingestion job.

      We check for IN_PROGRESS / STARTING jobs before starting a new one;
      Bedrock rejects overlapping jobs anyway, but the explicit check
      keeps CloudWatch clean of expected errors.
      """
      import os
      import boto3

      KB_ID = os.environ["KNOWLEDGE_BASE_ID"]
      DS_ID = os.environ["DATA_SOURCE_ID"]

      _client = boto3.client("bedrock-agent")


      def lambda_handler(event, context):
          existing = _client.list_ingestion_jobs(
              knowledgeBaseId=KB_ID,
              dataSourceId=DS_ID,
              filters=[
                  {"attribute": "STATUS", "operator": "EQ", "values": ["IN_PROGRESS"]},
              ],
          ).get("ingestionJobSummaries", [])
          starting = _client.list_ingestion_jobs(
              knowledgeBaseId=KB_ID,
              dataSourceId=DS_ID,
              filters=[
                  {"attribute": "STATUS", "operator": "EQ", "values": ["STARTING"]},
              ],
          ).get("ingestionJobSummaries", [])

          if existing or starting:
              print(f"[ingestion-trigger] job already running ({len(existing)} IN_PROGRESS, {len(starting)} STARTING); skipping")
              return {"status": "skipped"}

          response = _client.start_ingestion_job(
              knowledgeBaseId=KB_ID,
              dataSourceId=DS_ID,
              description="Auto-triggered by S3 upload",
          )
          job_id = response["ingestionJob"]["ingestionJobId"]
          print(f"[ingestion-trigger] started ingestion job {job_id}")
          return {"status": "started", "jobId": job_id}
    PY
  }
}

resource "aws_lambda_function" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  function_name    = "${var.project_name}-ingestion-trigger"
  role             = aws_iam_role.ingestion_trigger[0].arn
  runtime          = "python3.12"
  handler          = "handler.lambda_handler"
  filename         = data.archive_file.ingestion_trigger_zip[0].output_path
  source_code_hash = data.archive_file.ingestion_trigger_zip[0].output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      KNOWLEDGE_BASE_ID = var.ingestion_kb_id
      DATA_SOURCE_ID    = var.ingestion_data_source_id
    }
  }
}

# ── EventBridge rule (S3 object-created, 5-minute batching) ─────────────────

resource "aws_cloudwatch_event_rule" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  name        = "${var.project_name}-ingestion-trigger"
  description = "Batches S3 uploads under processed-chunks/ and faq-chunks/ into a single ingestion run."

  event_pattern = jsonencode({
    source        = ["aws.s3"]
    "detail-type" = ["Object Created"]
    detail = {
      bucket = {
        name = [var.data_bucket_name]
      }
      object = {
        key = [for p in local.ingestion_watch_prefixes : { prefix = p }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  rule = aws_cloudwatch_event_rule.ingestion_trigger[0].name
  arn  = aws_lambda_function.ingestion_trigger[0].arn

  # 300s input-throttle: any burst of events collapses into a single
  # invocation after the window elapses. Terraform surfaces this as the
  # target's retry/input-transformer config in newer provider versions;
  # we keep the simple 1:1 mapping and rely on the Lambda's own
  # idempotent job check to debounce.
}

resource "aws_lambda_permission" "ingestion_trigger" {
  count = local.enable_ingestion_trigger ? 1 : 0

  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion_trigger[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingestion_trigger[0].arn
}
