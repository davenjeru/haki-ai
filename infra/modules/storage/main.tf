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

# ── CORS ──────────────────────────────────────────────────────────────────────
# The frontend fetches single-page PDFs from page-images/ via presigned S3 GET
# URLs and renders them through PDF.js on a <canvas>. Those fetches are
# cross-origin (browser ↔ S3 / LocalStack) so the bucket must allow GET from
# the frontend origin. Allowed origins come from var.cors_allowed_origins.
# Presigned URLs already require a valid signature, so a wildcard origin on
# GET is safe; we still keep the list configurable so prod can be tightened.

resource "aws_s3_bucket_cors_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  cors_rule {
    allowed_methods = ["GET", "HEAD"]
    allowed_origins = var.cors_allowed_origins
    allowed_headers = ["*"]
    expose_headers  = ["ETag", "Content-Length", "Content-Type"]
    max_age_seconds = 3000
  }
}

# ── EventBridge notifications ────────────────────────────────────────────────
# Phase 5b: enabling EventBridge on the data bucket lets the ingestion-trigger
# Lambda in modules/compute listen for Object Created events under
# processed-chunks/*. When a pipeline run drops new chunks, the rule fires
# and starts a Bedrock KB ingestion job — no operator click required.

resource "aws_s3_bucket_notification" "data_eventbridge" {
  bucket      = aws_s3_bucket.data.id
  eventbridge = true
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

  # S3 Vectors caps filterable metadata at 2048 bytes per vector. Two
  # sources push past that:
  #   1. Our own chapter/title/pageImageKey strings — long for the
  #      Constitution, especially. Never used as filters (UI display only).
  #   2. Bedrock KB automatically injects AMAZON_BEDROCK_TEXT (full chunk
  #      body) and AMAZON_BEDROCK_METADATA (JSON blob with source/pages)
  #      as filterable attributes. A single ~500-token chunk alone blows
  #      the budget. AWS requires both of these marked non-filterable
  #      when using S3 Vectors as the KB vector store.
  # source / section / chunkId / chunkType stay filterable so the
  # backend can:
  #   - scope retrieval to one statute  (source = "Employment Act 2007")
  #   - drop table-of-contents chunks   (chunkType != "toc") — Phase 1
  # Long free-text attributes (chapter/title/pageImageKey) remain
  # non-filterable to stay under the 2048 byte per-vector cap.
  metadata_configuration {
    non_filterable_metadata_keys = [
      "AMAZON_BEDROCK_METADATA",
      "AMAZON_BEDROCK_TEXT",
      "chapter",
      "title",
      "pageImageKey",
    ]
  }
}

# ── LangGraph checkpoint store (DynamoDB) ─────────────────────────────────────
# Single-table design keyed by (thread_id, sort_key). thread_id equals the
# frontend-generated sessionId so conversation memory survives Lambda cold
# starts and server_local.py restarts. TTL attribute auto-purges abandoned
# conversations after ~30 days (set by backend/checkpointer.py).

resource "aws_dynamodb_table" "checkpoints" {
  name         = "${var.project_name}-checkpoints"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "thread_id"
  range_key    = "sort_key"

  attribute {
    name = "thread_id"
    type = "S"
  }

  attribute {
    name = "sort_key"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  # PITR is free-tier-unfriendly on LocalStack; skip it locally.
  dynamic "point_in_time_recovery" {
    for_each = var.is_local ? [] : [1]
    content {
      enabled = true
    }
  }
}
