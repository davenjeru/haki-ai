output "data_bucket_id" {
  value = aws_s3_bucket.data.id
}

output "data_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "vector_bucket_arn" {
  value = length(aws_s3vectors_vector_bucket.kb) > 0 ? aws_s3vectors_vector_bucket.kb[0].vector_bucket_arn : ""
}

output "vector_index_arn" {
  value = length(aws_s3vectors_index.kb) > 0 ? aws_s3vectors_index.kb[0].index_arn : ""
}
