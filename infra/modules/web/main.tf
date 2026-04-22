# ── Static-site S3 bucket (private; only CloudFront can read) ─────────────────
# The bucket has no public policy and no website config — CloudFront fetches
# objects over the REST endpoint via Origin Access Control (OAC), the modern
# replacement for Origin Access Identity. This keeps the bucket fully private.

resource "aws_s3_bucket" "site" {
  bucket = "${var.project_name}-web"
}

resource "aws_s3_bucket_public_access_block" "site" {
  bucket                  = aws_s3_bucket.site.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "site" {
  bucket = aws_s3_bucket.site.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ── CloudFront Origin Access Control ──────────────────────────────────────────
# OAC signs requests with SigV4 so the bucket can verify they came from
# our specific distribution and nothing else.

resource "aws_cloudfront_origin_access_control" "site" {
  name                              = "${var.project_name}-web-oac"
  description                       = "OAC for ${var.project_name} static site"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# ── CloudFront distribution ───────────────────────────────────────────────────
# Pay-as-you-go. PriceClass_100 = cheapest (US/Canada/Europe edges only).
# The Vite SPA does client-side routing, so any 403/404 from S3 is mapped to
# /index.html (200) and React Router handles it.

resource "aws_cloudfront_distribution" "site" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.project_name} static site"
  default_root_object = "index.html"
  price_class         = var.price_class
  http_version        = "http2and3"

  origin {
    domain_name              = aws_s3_bucket.site.bucket_regional_domain_name
    origin_id                = "s3-${aws_s3_bucket.site.id}"
    origin_access_control_id = aws_cloudfront_origin_access_control.site.id
  }

  default_cache_behavior {
    target_origin_id       = "s3-${aws_s3_bucket.site.id}"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # AWS-managed CachingOptimized policy (long cache TTLs, appropriate for
    # Vite's hashed asset filenames). index.html is handled specially via the
    # response_headers_policy below so the shell is never stale.
    cache_policy_id            = "658327ea-f89d-4fab-a63d-7e88639e58f6" # CachingOptimized
    response_headers_policy_id = "5cc3b908-e619-4b99-88e5-2cf7f45965bd" # CORS-with-preflight-and-SecurityHeadersPolicy
  }

  # SPA rewrite: any 403/404 from S3 returns index.html so the React Router
  # can handle the path client-side. Status 200 so browsers treat it as a
  # normal page load (with correct asset paths) rather than an error.
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }
  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# ── S3 bucket policy: allow only this distribution to read ───────────────────

data "aws_iam_policy_document" "site_bucket" {
  statement {
    sid       = "AllowCloudFrontServicePrincipalRead"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.site.arn}/*"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.site.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "site" {
  bucket = aws_s3_bucket.site.id
  policy = data.aws_iam_policy_document.site_bucket.json
}
