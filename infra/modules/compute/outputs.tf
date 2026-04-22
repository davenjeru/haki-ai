output "lambda_arn" { value = aws_lambda_function.handler.arn }
output "lambda_invoke_arn" { value = aws_lambda_function.handler.invoke_arn }
output "lambda_name" { value = aws_lambda_function.handler.function_name }
