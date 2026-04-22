# ── SNS topic → email alerts ───────────────────────────────────────────────────

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── CloudWatch alarms ──────────────────────────────────────────────────────────
# All alarms publish to the SNS topic above.

# Error rate > 5% over 5 minutes
resource "aws_cloudwatch_metric_alarm" "error_rate" {
  alarm_name          = "${var.project_name}-error-rate"
  alarm_description   = "Lambda error rate exceeded 5%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  threshold           = 5

  metric_query {
    id          = "error_rate"
    expression  = "failed / (succeeded + failed) * 100"
    label       = "Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "failed"
    metric {
      namespace   = "HakiAI"
      metric_name = "FailedRequests"
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id = "succeeded"
    metric {
      namespace   = "HakiAI"
      metric_name = "SuccessfulRequests"
      period      = 300
      stat        = "Sum"
    }
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  ok_actions    = [aws_sns_topic.alerts.arn]
}

# p95 response latency > 5000ms over 5 minutes
resource "aws_cloudwatch_metric_alarm" "latency_p95" {
  alarm_name          = "${var.project_name}-latency-p95"
  alarm_description   = "p95 response latency exceeded 5000ms"
  namespace           = "HakiAI"
  metric_name         = "ResponseLatency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  period              = 300
  extended_statistic  = "p95"
  threshold           = 5000
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# Guardrail blocks > 20 per hour
resource "aws_cloudwatch_metric_alarm" "guardrail_blocks" {
  alarm_name          = "${var.project_name}-guardrail-blocks"
  alarm_description   = "Guardrail blocks exceeded 20 in the last hour"
  namespace           = "HakiAI"
  metric_name         = "GuardrailBlock"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  period              = 3600
  statistic           = "Sum"
  threshold           = 20
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Eval score regression — the nightly/on-PR eval job emits a single
# HakiAI/EvalScore metric on a 0-5 scale (see backend/evals/report.py).
# Any drop below 3.5 is considered a rubric regression.
resource "aws_cloudwatch_metric_alarm" "eval_score" {
  alarm_name          = "${var.project_name}-eval-score-regression"
  alarm_description   = "Eval LLM-judge score dropped below 3.5/5"
  namespace           = "HakiAI"
  metric_name         = "EvalScore"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  period              = 86400
  statistic           = "Minimum"
  threshold           = 3.5
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Lambda errors (from Lambda's own metrics — catches crashes before handler runs)
resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project_name}-lambda-errors"
  alarm_description   = "Lambda function errors (crashes, timeouts, OOM)"
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = var.lambda_name }
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ── CloudWatch dashboard ───────────────────────────────────────────────────────

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = var.project_name

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Request Volume"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "SuccessfulRequests", { stat = "Sum", label = "Success" }],
            ["HakiAI", "FailedRequests", { stat = "Sum", label = "Failed", color = "#d62728" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Response Latency"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "ResponseLatency", { stat = "p50", label = "p50" }],
            ["HakiAI", "ResponseLatency", { stat = "p95", label = "p95", color = "#ff7f0e" }],
            ["HakiAI", "ResponseLatency", { stat = "p99", label = "p99", color = "#d62728" }],
          ]
          period = 300
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Detected Language"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "DetectedLanguage_english", { stat = "Sum", label = "English" }],
            ["HakiAI", "DetectedLanguage_swahili", { stat = "Sum", label = "Swahili" }],
            ["HakiAI", "DetectedLanguage_mixed", { stat = "Sum", label = "Mixed" }],
          ]
          period = 3600
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Guardrail Blocks"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "GuardrailBlock", { stat = "Sum", color = "#d62728" }],
          ]
          period = 3600
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Missing Citations"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "MissingCitations", { stat = "Sum", color = "#ff7f0e" }],
          ]
          period = 3600
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 24
        height = 6
        properties = {
          title  = "Eval Score (LLM-judge mean, 0-5)"
          region = "us-east-1"
          metrics = [
            ["HakiAI", "EvalScore", { stat = "Average", label = "Mean" }],
            ["HakiAI", "EvalScore", { stat = "Minimum", label = "Min", color = "#d62728" }],
          ]
          period = 86400
          view   = "timeSeries"
          yAxis  = { left = { min = 0, max = 5 } }
        }
      },
    ]
  })
}
