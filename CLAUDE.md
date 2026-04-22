# Haki AI — Project Context

## What this is
A Kenyan legal-aid agent that answers questions about Kenyan law in
English and Swahili, with citations to specific Act, Chapter, and
Section. Built around a two-tier multi-agent LangGraph with an
advanced RAG pipeline (hybrid retrieval + rerank) and an optional
fine-tuned Llama-3.1-8B fallback.

## Stack
- Frontend: React + Vite + Tailwind v4, full EN/SW i18n with auto
  language switch driven by the backend's Comprehend detection.
- Backend: Python 3.12 Lambda orchestrating a LangGraph StateGraph
  with DynamoDB checkpointing for multi-turn memory.
- RAG: pre-chunking pipeline (TypeScript + LiteParse OCR + Haiku
  extraction) → Bedrock KB over S3 Vectors → advanced-RAG pipeline
  (query expansion, hybrid BM25+dense, RRF, TOC filter, Cohere rerank).
- Data sources: Kenyan Acts (Constitution 2010, Employment 2007,
  Land 2012) + SheriaPlex forum Q&A crawl + KenyaLaw case summaries,
  all tagged with a filterable `corpus` metadata key.
- Fine-tuning: SageMaker qLoRA on Llama-3.1-8B-Instruct with a
  `USE_FINETUNED_MODEL` feature flag and Bedrock fallback.
- Evaluation: RAGAS (faithfulness / answer_relevancy /
  context_precision / context_recall) + LLM-as-judge on 4 axes.
- IaC: Terraform modular. GitHub Actions for CI, deploy, nightly evals.
- Observability: LangSmith per-turn tracing + CloudWatch metrics,
  alarms, dashboards, and SNS alerts.

## System architecture

```mermaid
flowchart TB
  subgraph Client
    fe[React + Vite frontend<br/>EN/SW i18n + LanguageToggle]
  end

  subgraph AWS
    apigw[API Gateway HTTP]
    lam[Lambda: app.handler]
    ddb[(DynamoDB<br/>LangGraph checkpoints)]
    kb[(Bedrock KB<br/>S3 Vectors index)]
    br[Bedrock Claude + Titan + Rerank]
    cmp[Comprehend]
    sm[SageMaker endpoint<br/>Llama-3.1-8B qLoRA]
    s3[(S3 haki-ai-data<br/>raw-laws, chunks, faq-chunks)]
    cw[(CloudWatch<br/>metrics + alarms)]
    sns[SNS email alerts]
  end

  fe -->|/chat| apigw --> lam
  lam --> ddb
  lam --> cmp
  lam --> kb --> s3
  lam --> br
  lam -.USE_FINETUNED_MODEL.-> sm
  lam --> cw --> sns

  subgraph Pipeline
    raw[(data/raw PDFs)] --> pipe[pipeline: OCR + Haiku extract + chunk]
    pipe --> s3
    s3 --> ing[ingestion_trigger Lambda]
    ing --> kb
  end

  subgraph Dev
    crawl[pipeline crawl-sheriaplex] --> s3
    ls[(LangSmith traces)]
    lam -.optional.-> ls
  end
```

## LangGraph agent flow

```mermaid
flowchart LR
  start[user turn] --> lang[detect_language<br/>Comprehend]
  lang --> sup[supervisor<br/>Haiku routing]
  sup -->|chat| chat[ChatAgent]
  sup -->|1–3 specialists| fan{{Send fan-out}}
  fan --> c[ConstitutionAgent]
  fan --> e[EmploymentAgent]
  fan --> l[LandAgent]
  fan --> f[FAQAgent]
  c --> pipe[rag.pipeline]
  e --> pipe
  l --> pipe
  f --> pipe
  pipe --> syn[synthesizer]
  chat --> syn
  syn --> outn[response + citations + language]
```

## Project layout
```
haki-ai/
├── frontend/              # React + Vite + Tailwind, i18n
├── backend/
│   ├── app/               # handler.py, graph.py, config.py, server_local.py, ingest_local.py
│   ├── agents/            # supervisor, specialists, chat, synthesizer, classifier
│   ├── rag/               # query_expansion, hybrid_retriever, bm25, rrf, filters, reranker, citations, generator, sagemaker_generator
│   ├── clients/           # boto3 factory + ComprehendAdapter / BedrockRAGAdapter / LocalRAGAdapter
│   ├── memory/            # DynamoDBSaver checkpointer
│   ├── observability/     # LangSmith tracing + CloudWatch metrics
│   ├── prompts/           # all LLM prompts
│   ├── evals/             # golden_set.jsonl + RAGAS + llm_judge + report writer
│   ├── tests/             # unit + e2e
│   └── scripts/           # offline helpers (prepare_finetune_data.py)
├── pipeline/              # TypeScript: PDF → pages → chunks + SheriaPlex crawler
├── data/raw/              # committed Kenyan law PDFs
├── infra/
│   └── modules/
│       ├── storage/       # S3 + S3 Vectors index (corpus filterable)
│       ├── compute/       # Lambda + DynamoDB + ingestion_trigger
│       ├── api/           # API Gateway HTTP
│       ├── ai/            # Bedrock KB + Guardrails
│       ├── ml/            # SageMaker fine-tune + endpoint (scale-to-zero)
│       └── observability/ # alarms + dashboard + SNS
├── .github/workflows/     # ci.yml, deploy.yml, eval-nightly.yml
└── scripts/               # bootstrap.sh, run-finetune.sh
```

## Per-package docs
Each backend package has its own diagram-led one-pager:

- [backend/app](backend/app/CLAUDE.md) — entry points (Lambda, local server, ingestion CLI)
- [backend/agents](backend/agents/CLAUDE.md) — supervisor + specialists + synthesizer
- [backend/rag](backend/rag/CLAUDE.md) — advanced RAG pipeline stages
- [backend/clients](backend/clients/CLAUDE.md) — boto3 factory + adapters
- [backend/memory](backend/memory/CLAUDE.md) — DynamoDB checkpointer
- [backend/observability](backend/observability/CLAUDE.md) — tracing + metrics
- [backend/prompts](backend/prompts/CLAUDE.md) — LLM prompt templates
- [backend/evals](backend/evals/CLAUDE.md) — RAGAS + LLM-judge harness
- [backend/tests](backend/tests/CLAUDE.md) — test suite

## Key architecture decisions

### Advanced RAG (Phase 1)
The single `retrieve_and_generate` call was split into a pipeline
(`rag/pipeline.py`) with: HyDE-style query expansion (3 variants),
dense retrieval via Bedrock KB + in-memory BM25 over `processed-chunks`,
Reciprocal Rank Fusion, TOC filter (chunks tagged `chunkType=toc` at
pipeline time are excluded), Cohere Rerank v3.5 on Bedrock, citation
dedup by `(source, section)`, then Claude `invoke_model` for generation.
`LocalRAGAdapter` runs the same pipeline against ChromaDB so dev/prod
parity holds.

### Two-tier multi-agent (Phase 2)
`SupervisorRouter` (Haiku, JSON) picks 1–3 specialists or `chat`.
Each specialist is a sub-agent that runs the advanced-RAG pipeline
scoped by `source` / `corpus` filter. `Synthesizer` merges ≥2
specialist outputs; single-specialist turns pass through verbatim.
Fan-out uses LangGraph's `Send()` with an `operator.add` reducer on
`specialist_outputs`.

### Evaluation harness (Phase 3)
30-question golden set, RAGAS (optional dep), LLM-as-judge on 4 axes,
markdown reports + `EvalScore` CloudWatch metric. CI blocks PRs on
eval-score regressions >5%.

### Problem-solution fit (Phase 4)
- SheriaPlex forum + KenyaLaw case summaries are crawled into
  `faq-chunks/` with `corpus=faq` metadata so the `FAQAgent` can
  filter them and statute specialists can exclude them.
- Fine-tune: SageMaker qLoRA on Llama-3.1-8B-Instruct, deployed as a
  serverless scale-to-zero endpoint. `USE_FINETUNED_MODEL=true`
  routes generation to it with automatic Bedrock fallback on error.
- Frontend i18n covers ~30 UI strings in EN + SW, auto-switches locale
  on detected Swahili responses, `manualLocaleOverride` persists user
  intent in `localStorage`.

### Deployment maturity (Phase 5)
- `make setup` bootstraps state bucket, copies `.env.example` → `.env`,
  installs backend/pipeline/frontend deps, and applies the local
  LocalStack infra — one command from a fresh clone.
- `ingestion_trigger` Lambda debounces S3 EventBridge events for
  `processed-chunks/` + `faq-chunks/` and calls
  `bedrock-agent.start_ingestion_job` — dropping a new Act PDF and
  running the pipeline re-indexes the KB with no human in the loop.
- `.github/workflows/ci.yml` (tests + evals on PR),
  `deploy.yml` (OIDC → terraform apply + deploy-web + ingest
  net-new PDFs), `eval-nightly.yml` (RAGAS report posted to main).

### Separation of concerns (Phase 6)
`backend/` is packaged by concern: `app/`, `agents/`, `rag/`,
`clients/`, `memory/`, `observability/`, `prompts/`, `evals/`,
`tests/`. Lambda ships only the runtime-relevant subset (tests, evals,
scripts, `*_local.py` excluded). Each package has its own
`CLAUDE.md` with a mermaid diagram.

## Chunk metadata sidecar (Bedrock KB native)
```json
{
  "metadataAttributes": {
    "source": "Employment Act 2007",
    "chapter": "Part III — Termination of Contract",
    "section": "Section 40",
    "title": "Termination of employment",
    "chunkId": "employment-act-2007-part-iii-section-40",
    "pageImageKey": "page-images/employment-act-2007/page-40.pdf",
    "chunkType": "body",
    "corpus": "statute"
  }
}
```

`corpus ∈ {"statute", "faq"}` and `chunkType ∈ {"body", "toc"}` are
filterable; `chapter` / `title` / `pageImageKey` /
`AMAZON_BEDROCK_TEXT` / `AMAZON_BEDROCK_METADATA` are non-filterable
to stay under the 2048-byte S3 Vectors filterable-metadata cap.

## Local testing strategy

### Paths
- **`make dev`** — LocalStack + ChromaDB + backend in-process
  (`python -m app.server_local`, port 8080) + Vite dev server.
  Real AWS creds for Bedrock; everything else against LocalStack.
- **LocalStack Lambda** — for infra/wiring verification only;
  Bedrock calls fail inside the Lambda container by design.

### Commands
- `make test` — `unittest discover -s tests -p 'test_unit.py'` (backend)
  + `tsc --noEmit` (frontend) + `npm test` (pipeline crawler).
- `make eval` — `uv run -m evals.run` against the golden set.
- `make ingest-local` — hydrate `.local-vectorstore/` from LocalStack S3.

## CloudWatch metrics (namespace: HakiAI)
- Request/response: `SuccessfulRequests`, `FailedRequests`,
  `ResponseLatency`.
- Language: `DetectedLanguage_english` / `_swahili` / `_mixed`.
- Quality: `GuardrailBlock`, `MissingCitations`,
  `LowConfidenceRetrieval`, `EvalScore`.

## Terraform state
- Remote backend: S3 bucket `haki-ai-terraform-state`.
- Workspaces: `default` → prod, `local` → LocalStack. Flip with
  `terraform workspace select`.
- Prod apply needs `TF_VAR_langsmith_api_key` in the environment
  (CI reads it from GitHub Secrets).

## Current status
Phase 1–6 of the rubric-alignment plan are complete: advanced RAG,
two-tier multi-agent, eval harness, SheriaPlex corpus + SageMaker
fine-tune scaffolding, Swahili UI, clone-and-run Makefile, auto
ingestion, three GitHub Actions workflows, backend refactor, and
per-package diagram-led docs. Remaining work is prod rollout and any
final demo polish.
