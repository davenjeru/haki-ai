output "api_endpoint" {
  description = "API Gateway HTTP API base URL — append /chat for the chat endpoint"
  value       = aws_apigatewayv2_stage.default.invoke_url
}
