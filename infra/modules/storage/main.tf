# ── S3 data bucket (raw PDFs + pre-chunked .txt files for Bedrock KB) ─────────

resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── S3 Vectors bucket + index (vector store backend for Bedrock KB) ───────────
# Not supported by LocalStack — skipped when is_local = true.

resource "aws_s3vectors_vector_bucket" "kb" {
  count = var.is_local ? 0 : 1

  vector_bucket_name = "${var.project_name}-vectors"
  force_destroy      = true
}

# Titan Embed Text v2 produces 1024-dimensional float32 vectors.
# Cosine similarity is standard for semantic text search.
resource "aws_s3vectors_index" "kb" {
  count = var.is_local ? 0 : 1

  index_name         = "${var.project_name}-index"
  vector_bucket_name = aws_s3vectors_vector_bucket.kb[0].vector_bucket_name
  data_type          = "float32"
  dimension          = 1024
  distance_metric    = "cosine"
}
