# ── CloudWatch log group ───────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 7
}

# ── HTTP API ───────────────────────────────────────────────────────────────────
# API Gateway HTTP API (v2) — lower latency and cost than REST API (v1).
# CORS is configured here at the API level; Lambda does not need to set headers.

resource "aws_apigatewayv2_api" "chat" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "PATCH", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }
}

# ── Lambda integration ─────────────────────────────────────────────────────────
# AWS_PROXY passes the full HTTP request to Lambda and returns its response as-is.
# payload_format_version 2.0 is the HTTP API default and is simpler than 1.0.

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.chat.id
  integration_type       = "AWS_PROXY"
  integration_uri        = var.lambda_invoke_arn
  payload_format_version = "2.0"
}

# ── Route: POST /chat ──────────────────────────────────────────────────────────

resource "aws_apigatewayv2_route" "chat" {
  api_id    = aws_apigatewayv2_api.chat.id
  route_key = "POST /chat"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# ── Route: GET /chat/history ───────────────────────────────────────────────────
# Returns the persisted conversation for a given sessionId so the frontend can
# rehydrate the chat UI after a page refresh. Lambda reads the LangGraph
# checkpointer and re-presigns citation pageImageUrls on every call.

resource "aws_apigatewayv2_route" "chat_history" {
  api_id    = aws_apigatewayv2_api.chat.id
  route_key = "GET /chat/history"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# ── Routes: /chat/threads (signed-in only) ─────────────────────────────────────
# The thread index powers the "your chats" sidebar and only responds to
# requests that carry a verified Clerk session JWT. The Lambda rejects with
# 401 when the Bearer header is missing or invalid; API Gateway is not
# configured with a JWT authorizer because Clerk instances are per-app and
# we want the publishable-key-derived verification to stay in one place.

resource "aws_apigatewayv2_route" "chat_threads_list" {
  api_id    = aws_apigatewayv2_api.chat.id
  route_key = "GET /chat/threads"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "chat_threads_rename" {
  api_id    = aws_apigatewayv2_api.chat.id
  route_key = "PATCH /chat/threads"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "chat_threads_claim" {
  api_id    = aws_apigatewayv2_api.chat.id
  route_key = "POST /chat/threads/claim"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

# ── $default stage ─────────────────────────────────────────────────────────────
# auto_deploy = true — changes are deployed immediately without a manual deploy step.
# $default stage means the route is reachable at the root of the API endpoint.

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.chat.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      ip               = "$context.identity.sourceIp"
      requestTime      = "$context.requestTime"
      httpMethod       = "$context.httpMethod"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}

# ── Lambda invoke permission ───────────────────────────────────────────────────
# Grants API Gateway permission to call the Lambda function.
# source_arn restricts the grant to this specific API only.

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_arn
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.chat.execution_arn}/*/*"
}
