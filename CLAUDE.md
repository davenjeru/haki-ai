# Haki AI — Project Context

## What this is
A Kenyan legal aid agent that answers questions about Kenyan law
in English and Swahili, with citations to specific Act, Chapter, and Section.

## Stack
- Frontend: React (chat UI + citation renderer + language-aware placeholder)
- Backend: Python Lambda (agent orchestrator)
- IaC: Terraform (modular)
- RAG pipeline: custom chunking pipeline (TypeScript, LiteParse OCR + Haiku
  LLM extraction) → S3 pre-chunked .txt files → Bedrock KB (embedding +
  retrieval + generation via retrieve_and_generate)
- Vector store: S3 Vectors (storage backend for Bedrock KB)
- Raw data: S3 standard bucket (law PDFs + fine-tuning data)
- Language detection: AWS Comprehend (returns english / swahili / mixed)
- Guardrails: Amazon Bedrock Guardrails (topic denial for out-of-scope)
- Fine-tuning: SageMaker (qLoRA / PEFT)
- Monitoring: CloudWatch (custom metrics, alarms, dashboard, SNS alerts)

## Laws covered
- Constitution of Kenya 2010
- Employment Act 2007
- Land Act 2012

## Project structure
haki-ai/
├── frontend/
├── backend/
│   ├── handler.py         # Lambda entry point — thin orchestrator
│   ├── config.py          # single source of truth for env vars
│   ├── clients.py         # boto3 factory (make_comprehend, make_bedrock, etc.)
│   ├── adapters.py        # client wrappers (papers over LocalStack limitations)
│   ├── prompts.py         # system prompt builder (step 2)           [TODO]
│   ├── rag.py             # retrieve_and_generate + guardrail (steps 3–4) [TODO]
│   ├── citations.py       # citation extraction (step 5)             [TODO]
│   ├── metrics.py         # CloudWatch metric emission (step 6)      [TODO]
│   └── test_language_detection.py
├── pipeline/              # local data prep (TypeScript, LiteParse)
│   └── src/
├── data/
│   └── raw/              # Kenyan law PDFs (committed — required for pipeline)
└── infra/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── terraform.tfvars
    └── modules/
        ├── storage/       # S3 standard + S3 Vectors buckets
        ├── compute/       # Lambda + IAM role  ✓ implemented
        ├── api/           # API Gateway HTTP + CORS + CloudWatch logs [TODO]
        ├── ai/            # Bedrock KB + Guardrails
        ├── ml/            # SageMaker qLoRA job
        └── observability/ # CloudWatch metrics, alarms, dashboard, SNS [TODO]

## Key architecture decisions
- Chunking is fully custom (not delegated to Bedrock KB):
    1. LiteParse OCR extracts raw text per page in 10-page batches
    2. Claude Haiku (Bedrock InvokeModel) identifies section boundaries per page
    3. assembleChunks() walks pages in order, accumulating lines per section
    4. splitToSegments() splits long sections at paragraph breaks (~500 tokens)
    5. Pre-chunked .txt files + .txt.metadata.json sidecars uploaded to S3
  Bedrock KB only handles embedding + retrieval + generation (retrieve_and_generate).
- S3 Vectors is the vector store backend for Bedrock KB (serverless,
  cheapest option, GA December 2025)
- Bedrock KB syncs from pre-chunked .txt files in S3 via start_ingestion_job
- Each chunk has a .txt.metadata.json sidecar (Bedrock KB native format)
  carrying { source, chapter, section, title } for citation accuracy
- Language detection via Comprehend → dynamic system prompt per language
- Three-layer guardrails:
    1. System prompt instructs model to refuse out-of-scope
    2. Bedrock Guardrails intercepts before model sees request
    3. Lambda checks stopReason == "guardrail_intervened"
- Bilingual refusal: "Mimi ni msaidizi wa kisheria wa Kenya tu. /
  I can only help with Kenyan legal matters."
- Every response must cite Act + Chapter + Section

## Lambda core logic (handler.py)
1. Detect language (Comprehend) → english / swahili / mixed
2. Build system prompt with language instruction
3. Call bedrock_agent_runtime.retrieve_and_generate with KB id + model
4. Check for guardrail block (stopReason)
5. Extract citations from response
6. Log custom CloudWatch metrics
7. Return { response, citations, language, blocked }

## PDF pipeline (pipeline/)
- Runtime: TypeScript / Node.js (local scripts, not Lambda)
- Two-script pipeline — run in order:

### Script 1: `npm run dev` (src/run.ts) — Page extraction
- LiteParse (@llamaindex/liteparse) with OCR, processes in 10-page batches
- pdf-lib extracts each page as a single-page PDF (no rendering, fast)
- Uploads per page:
    page-images/{shortId}/page-{n}.pdf   ← single-page PDF for citation carousel
    page-text/{shortId}/page-{n}.txt     ← raw OCR text for LLM chunking
- Also uploads raw-laws/{filename} (original PDF)

### Script 2: `npm run chunk` (src/chunk-laws.ts) — LLM-assisted chunking
- For each page, calls Claude Haiku (us.anthropic.claude-haiku-4-5-20251001-v1:0)
  via Bedrock InvokeModel to extract structured section metadata as JSON:
    { chapterOrPart: string|null, sections: [{ number, title, bodyStartLine }] }
- Haiku results cached to S3 at page-extractions/{shortId}/page-{n}.json
  so re-runs skip already-processed pages (resume-safe after throttling)
- Concurrency capped at 2 with exponential backoff + jitter on ThrottlingException
- assembleChunks() walks extractions in page order, accumulating lines per section
  across page boundaries
- Uploads per section chunk:
    processed-chunks/{chunkId}.txt
    processed-chunks/{chunkId}.txt.metadata.json  ← Bedrock KB sidecar format
- Writes processed-chunks/{shortId}/.complete marker after successful upload
  (idempotency: re-runs skip completed laws entirely)

### Chunk metadata sidecar format (Bedrock KB native)
```json
{
  "metadataAttributes": {
    "source": "Employment Act 2007",
    "chapter": "Part III — Termination of Contract",
    "section": "Section 40",
    "title": "Termination of employment",
    "chunkId": "employment-act-2007-part-iii-section-40",
    "pageImageKey": "page-images/employment-act-2007/page-40.pdf"
  }
}
```

### S3 layout
    raw-laws/           ← original PDFs
    page-images/        ← single-page PDFs per page per law
    page-text/          ← raw OCR text per page per law
    page-extractions/   ← cached Haiku JSON extractions (resume support)
    processed-chunks/   ← .txt + .txt.metadata.json for Bedrock KB

### Run order
terraform apply → npm run dev → npm run chunk → start_ingestion_job

### To re-process a law
aws s3 rm s3://haki-ai-data/processed-chunks/{shortId}/.complete
aws s3 rm s3://haki-ai-data/processed-chunks/ --recursive --exclude "*" --include "{shortId}-*"
aws s3 rm s3://haki-ai-data/page-extractions/{shortId}/ --recursive

## Backend module structure (backend/)
Three separation layers keep handler.py a thin orchestrator:
- config.py   — load_config() reads all env vars once; returns frozen Config dataclass
- clients.py  — make_comprehend(), make_bedrock_agent_runtime(), make_cloudwatch()
                 all boto3 clients constructed here, nowhere else
- adapters.py — ComprehendAdapter wraps client; handles LocalStack "not supported"
                 fallback so business logic is environment-agnostic

Adding a new AWS service: add make_X() to clients.py, add XAdapter if LocalStack
limitations apply, inject into lambda_handler — no changes to business logic files.

## Local testing strategy
- LocalStack Pro (paid): Lambda, API Gateway, S3, CloudWatch, IAM
  - Start: localstack start -d (no docker-compose — not set up)
  - Comprehend DetectDominantLanguage not yet implemented in LocalStack Pro
    v2026.3.x — ComprehendAdapter falls back to "english" locally
- Bedrock always hits real AWS (LocalStack does not support it)
- S3 Vectors skipped locally (count = 0 in Terraform) — no ChromaDB replacement
- ENV=local switches all boto3 clients to LocalStack endpoint
- LOCALSTACK_HOSTNAME env var (injected by LocalStack into Lambda) resolves
  the correct internal Docker hostname; falls back to localhost for direct calls
- Pipeline uses AWS_ENDPOINT_URL=http://localhost:4566 for LocalStack S3;
  Bedrock endpoint explicitly hardcoded in llm-extract.ts to bypass it
- uv manages Python dependencies (Python 3.12)

## CloudWatch custom metrics (namespace: HakiAI)
- SuccessfulRequests, FailedRequests
- ResponseLatency (ms)
- DetectedLanguage_english / _swahili / _mixed
- GuardrailBlock, MissingCitations, LowConfidenceRetrieval

## CloudWatch alarms → SNS email alerts
- Error rate > 5%
- p95 latency > 5000ms
- GuardrailBlocks > 20/hr

## Terraform state
- Remote backend: S3 bucket haki-ai-terraform-state
- Create manually before terraform init

## Frontend (frontend/)
- React + Vite + Tailwind v4 (@tailwindcss/vite plugin)
- Custom color tokens defined in index.css via @theme block:
  --color-bg, --color-elevated, --color-border, --color-muted,
  --color-accent, --color-accent-bright, --color-citation, --color-strong, etc.
- Components: ChatApp, MessageThread, Composer, LanguageBadge,
  CitationBlock, PageCarousel
- PageCarousel: displays single-page PDFs from page-images/ S3 prefix
  as a citation carousel (useState, no extra deps)
- Mock mode: VITE_USE_MOCK=true or no VITE_API_BASE_URL set
- Real API: POST to VITE_API_BASE_URL + VITE_CHAT_PATH (/chat default)

## Current status

### Done
- Pipeline (both scripts) complete, tested, and verified locally against LocalStack
  - 1196 chunks + metadata sidecars across all 3 laws
  - Resume-safe: S3 cache for Haiku extractions, .complete markers, local temp dir
- Pipeline refactored: dead code removed, module-level doc comments added
- Backend architecture established (config / clients / adapters / handler pattern)
- Backend step 1 complete: language detection via Comprehend with mock + LocalStack tests
- infra/modules/compute implemented: Lambda (Python 3.12) + IAM role + policy
  - Permissions: CloudWatch Logs, Comprehend, CloudWatch metrics, Bedrock
  - archive_file zips backend/ excluding .venv, tests, uv files
- End-to-end Lambda invocation verified on LocalStack Pro

### Remaining — backend
- prompts.py   step 2: build_system_prompt(language)
- rag.py       steps 3–4: retrieve_and_generate + guardrail block check
- citations.py step 5: extract_citations(rag_response)
- metrics.py   step 6: emit_metrics() → HakiAI CloudWatch namespace
- Wire all steps into handler.py; LocalStack end-to-end test

### Remaining — infrastructure
- infra/modules/api/main.tf        API Gateway HTTP + CORS + Lambda integration
- infra/modules/observability/     CloudWatch alarms + dashboard + SNS email alerts

### Remaining — prod deployment
- make apply → recreate prod infra on real AWS
- npm run dev + npm run chunk → pipeline against real S3
- start_ingestion_job → index chunks into Bedrock KB
- Connect frontend: set VITE_API_BASE_URL to API Gateway URL
