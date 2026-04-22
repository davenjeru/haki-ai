data "aws_caller_identity" "current" {}

# ── IAM role for Bedrock KB to read from S3 ───────────────────────────────────

resource "aws_iam_role" "bedrock_kb" {
  name = "${var.project_name}-bedrock-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "aws:SourceAccount" = data.aws_caller_identity.current.account_id
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "bedrock_kb_s3" {
  name = "${var.project_name}-bedrock-kb-s3"
  role = aws_iam_role.bedrock_kb.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          var.data_bucket_arn,
          "${var.data_bucket_arn}/*"
        ]
      },
      {
        # S3 Vectors permissions for read/write of embeddings
        Effect = "Allow"
        Action = [
          "s3vectors:GetIndex",
          "s3vectors:ListIndexes",
          "s3vectors:PutVectors",
          "s3vectors:GetVectors",
          "s3vectors:DeleteVectors",
          "s3vectors:QueryVectors"
        ]
        Resource = "${var.vector_bucket_arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.embedding_model_id}"
      }
    ]
  })
}

# ── Bedrock Knowledge Base ─────────────────────────────────────────────────────

resource "aws_bedrockagent_knowledge_base" "this" {
  name        = "${var.project_name}-kb"
  description = "Kenyan law knowledge base — Constitution 2010, Employment Act 2007, Land Act 2012"
  role_arn    = aws_iam_role.bedrock_kb.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.embedding_model_id}"
    }
  }

  storage_configuration {
    type = "S3_VECTORS"
    s3_vectors_configuration {
      index_arn = var.vector_index_arn
    }
  }
}

# ── Bedrock KB data source (points at processed-chunks/ in S3) ────────────────

resource "aws_bedrockagent_data_source" "laws" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.this.id
  name              = "kenyan-laws"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = var.data_bucket_arn
      inclusion_prefixes = ["processed-chunks/"]
    }
  }

  # We pre-chunk in the pipeline — tell Bedrock KB not to re-chunk
  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "NONE"
    }
  }
}

# ── Bedrock Guardrail ─────────────────────────────────────────────────────────

resource "aws_bedrock_guardrail" "this" {
  name        = "${var.project_name}-guardrail"
  description = "Defence-in-depth safety layer: blocks hateful, violent, and sexual content. Scope (Kenyan-law only) is enforced by the system prompt + classifier, not by topic policy."

  blocked_input_messaging   = "Mimi ni msaidizi wa kisheria wa Kenya tu. / I can only help with Kenyan legal matters."
  blocked_outputs_messaging = "Mimi ni msaidizi wa kisheria wa Kenya tu. / I can only help with Kenyan legal matters."

  # Note: an earlier version of this resource used a topic_policy_config
  # with a negative definition ("Any question not related to Kenyan law").
  # Negative topic definitions are over-aggressive in practice — the
  # classifier flagged legitimate questions like "What does section 40 of
  # the Employment Act say?" as off-topic and blocked them. Scope is now
  # enforced exclusively by:
  #   1. prompts.build_system_prompt() — the model's behavioural instruction
  #   2. classifier.classify_intent() — routes non-legal chat away from RAG
  #
  # Bedrock requires at least one policy on every guardrail, so we keep
  # a HIGH-severity harmful-content filter as the single active policy.
  # This acts as a baseline safety net without affecting the core legal Q&A.
  content_policy_config {
    # Explicit tier avoids a TF provider drift warning where the API
    # defaults to CLASSIC but the resource schema leaves it null.
    tier_config {
      tier_name = "CLASSIC"
    }
    filters_config {
      type            = "HATE"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "VIOLENCE"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "SEXUAL"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
    filters_config {
      type            = "INSULTS"
      input_strength  = "HIGH"
      output_strength = "HIGH"
    }
  }
}

resource "aws_bedrock_guardrail_version" "this" {
  guardrail_arn = aws_bedrock_guardrail.this.guardrail_arn
  description   = "v1"
}
