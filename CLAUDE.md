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
│   ├── adapters.py        # ComprehendAdapter + LocalRAGAdapter + BedrockRAGAdapter
│   ├── prompts.py         # system prompt builder (step 2)
│   ├── rag.py             # retrieve_and_generate + guardrail check (steps 3–4)
│   ├── citations.py       # citation extraction (step 5)
│   ├── metrics.py         # CloudWatch metric emission (step 6)
│   ├── ingest_local.py    # one-time local ingestion: S3 chunks → ChromaDB [local only]
│   ├── server_local.py    # local HTTP server for frontend dev (port 8080) [local only]
│   ├── test_unit.py       # 35 unit tests — no AWS required [local only]
│   ├── test_e2e_local.py  # in-process end-to-end RAG test [local only]
│   └── test_language_detection.py  # language detection test runner [local only]
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
        ├── compute/       # Lambda + IAM role
        ├── api/           # API Gateway HTTP + CORS + CloudWatch logs
        ├── ai/            # Bedrock KB + Guardrails
        ├── ml/            # SageMaker qLoRA job
        └── observability/ # CloudWatch alarms, dashboard, SNS alerts

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

### Run order (prod)
terraform apply → npm run dev → npm run chunk → start_ingestion_job

### Run order (local)
localstack start -d → terraform apply → npm run dev → npm run chunk → ENV=local uv run ingest_local.py

### To re-process a law
aws s3 rm s3://haki-ai-data/processed-chunks/{shortId}/.complete
aws s3 rm s3://haki-ai-data/processed-chunks/ --recursive --exclude "*" --include "{shortId}-*"
aws s3 rm s3://haki-ai-data/page-extractions/{shortId}/ --recursive

## Backend module structure (backend/)
Three separation layers keep handler.py a thin orchestrator:
- config.py   — load_config() reads all env vars once; returns frozen Config dataclass
                 Fields: is_local, localstack_endpoint, aws_region, knowledge_base_id,
                 guardrail_id, guardrail_version, bedrock_model_id, embedding_model_id,
                 chroma_host, chroma_port
- clients.py  — make_comprehend(), make_bedrock_agent_runtime(), make_bedrock_runtime(),
                 make_cloudwatch() — all boto3 clients constructed here, nowhere else
- adapters.py — ComprehendAdapter: wraps Comprehend, falls back to "english" on LocalStack
                 LocalRAGAdapter: mimics Bedrock KB retrieve_and_generate locally via
                   ChromaDB + Titan embed + Claude InvokeModel. Selects backend at init:
                   - CHROMA_HOST set → _ChromaHttpClient (stdlib urllib, no chromadb package)
                   - CHROMA_HOST empty → chromadb.PersistentClient (in-process)
                 BedrockRAGAdapter: wraps bedrock-agent-runtime retrieve_and_generate

Adding a new AWS service: add make_X() to clients.py, add XAdapter if LocalStack
limitations apply, inject into lambda_handler — no changes to business logic files.

## Local testing strategy

### Two local paths — choose based on what you're testing

**Path A — server_local.py (recommended for frontend dev + RAG quality)**
- Runs handler in-process on the host machine with real AWS credentials
- ChromaDB PersistentClient reads .local-vectorstore/ directly (no HTTP server needed)
- Port 8080; set LOCAL_API_URL=http://localhost:8080 in frontend/.env.local
- Start: `cd backend && ENV=local uv run server_local.py`
- Full stack: Terminal 1 above, Terminal 2: `cd frontend && npm run dev`
- Verified working: 5 citations returned from Employment Act on real questions

**Path B — LocalStack Lambda (for infrastructure/wiring verification)**
- Lambda runs inside Docker with fake credentials (LocalStack injects test/test)
- CHROMA_HOST=host.docker.internal lets container reach ChromaDB HTTP server on host
- ChromaDB must be running: `uv run chroma run --path .local-vectorstore --port 8000 --host 0.0.0.0`
- Bedrock InvokeModel fails inside Lambda because LocalStack overrides credentials —
  use Path A for real RAG responses
- Invoke via: `make local-apply` then invoke Lambda through LocalStack API Gateway

### Local infrastructure
- LocalStack Pro (paid): Lambda, API Gateway, S3, CloudWatch, IAM
  - Start: localstack start -d (no docker-compose — not set up)
  - Comprehend DetectDominantLanguage not yet implemented in LocalStack Pro
    v2026.3.x — ComprehendAdapter falls back to "english" locally
- Bedrock always hits real AWS (LocalStack does not support it)
- Local vector store at backend/.local-vectorstore/ (gitignored, excluded from Lambda zip)
  - Populated once: ENV=local uv run ingest_local.py (reads from LocalStack S3)
  - 1196 chunks embedded and verified
- ENV=local switches boto3 clients to LocalStack endpoint
- LOCALSTACK_HOSTNAME env var (injected by LocalStack into Lambda) resolves
  the correct internal Docker hostname; falls back to localhost for direct calls
- Pipeline uses AWS_ENDPOINT_URL=http://localhost:4566 for LocalStack S3;
  Bedrock endpoint explicitly hardcoded in llm-extract.ts to bypass it
- uv manages Python dependencies (Python 3.12)

### Test scripts (all committed to git, excluded from Lambda zip)
- `uv run test_unit.py`        — 35 unit tests, no AWS required, runs in 0.001s
- `ENV=local uv run test_e2e_local.py` — in-process end-to-end RAG (real Bedrock)
- `ENV=local uv run test_language_detection.py` — language detection against LocalStack

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
- Local: make local-apply (uses local.tfstate, no remote backend)

## Frontend (frontend/)
- React + Vite + Tailwind v4 (@tailwindcss/vite plugin)
- Custom color tokens defined in index.css via @theme block:
  --color-bg, --color-elevated, --color-border, --color-muted,
  --color-accent, --color-accent-bright, --color-citation, --color-strong, etc.
- Components: ChatApp, MessageThread, Composer, LanguageBadge,
  CitationBlock, PageCarousel
- PageCarousel: displays single-page PDFs from page-images/ S3 prefix
  as a citation carousel (useState, no extra deps)
- API modes (chatClient.ts):
  - VITE_USE_MOCK=true or VITE_API_BASE_URL unset → mock mode (no backend)
  - VITE_API_BASE_URL="" (empty) + LOCAL_API_URL set → real API via Vite proxy
  - VITE_API_BASE_URL="https://..." → real API called directly (production)
- Local dev: set LOCAL_API_URL=http://localhost:8080 in frontend/.env.local

## Current status

### Done
- Pipeline (both scripts) complete, tested, and verified locally against LocalStack
  - 1196 chunks + metadata sidecars across all 3 laws
  - Resume-safe: S3 cache for Haiku extractions, .complete markers, local temp dir
- Pipeline refactored: dead code removed, module-level doc comments added
- Backend fully implemented — all 6 handler steps wired:
  - prompts.py: build_system_prompt(language) — english / swahili / mixed variants
  - rag.py: retrieve_and_generate() + check_guardrail_block() — env-agnostic
  - citations.py: extract_citations() — deduplicates by chunkId, includes pageImageKey
  - metrics.py: emit_metrics() → HakiAI CloudWatch namespace
  - handler.py: thin orchestrator, all steps wired, CORS headers
- Backend architecture: config / clients / adapters / handler separation
- Local RAG adapter implemented: ChromaDB + Titan Embed Text v2 + Claude InvokeModel
  - LocalRAGAdapter uses _ChromaHttpClient (stdlib urllib) when CHROMA_HOST is set
  - Falls back to chromadb.PersistentClient when CHROMA_HOST is empty (in-process)
  - Returns Bedrock KB-compatible response shape; citations.py/handler.py env-agnostic
  - 1196 chunks ingested and verified; end-to-end RAG verified (5 citations on test query)
- server_local.py: in-process HTTP server on port 8080 for frontend local dev
  - No LocalStack required; runs with real AWS credentials on host
  - Verified: real RAG responses with citations returned to frontend
- infra/modules/compute: Lambda (Python 3.12) + IAM role + policy
  - CHROMA_HOST=host.docker.internal + CHROMA_PORT=8000 in Lambda env vars
  - archive_file zips backend/ excluding .venv, test_*.py, *_local.py, uv files
- infra/modules/api: API Gateway HTTP + CORS + Lambda integration + access logs
- infra/modules/observability: CloudWatch alarms + dashboard + SNS email alerts
  - 4 alarms: error rate, p95 latency (extended_statistic), guardrail blocks, lambda errors
- Frontend Vite proxy: LOCAL_API_URL (Node-only) proxies /chat to backend
- Unit tests: 35 tests covering all business logic modules, run in 0.001s
- End-to-end verified locally: frontend → server_local.py → ChromaDB → Bedrock → citations

### Remaining — prod deployment
- make apply → create prod infra on real AWS
- npm run dev + npm run chunk → pipeline against real S3
- start_ingestion_job → index chunks into Bedrock KB
- Connect frontend: set VITE_API_BASE_URL to API Gateway URL
