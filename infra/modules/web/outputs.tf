output "site_bucket_name" {
  description = "Name of the S3 bucket holding the built frontend."
  value       = aws_s3_bucket.site.id
}

output "distribution_id" {
  description = "CloudFront distribution ID. Used by `aws cloudfront create-invalidation`."
  value       = aws_cloudfront_distribution.site.id
}

output "distribution_domain_name" {
  description = "CloudFront domain (d1234.cloudfront.net). This is the public URL of the app."
  value       = aws_cloudfront_distribution.site.domain_name
}
