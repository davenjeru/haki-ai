# Haki AI — root Makefile.
#
# Orchestrates the four sub-projects (infra/, backend/, pipeline/,
# frontend/) so a new developer can clone and run without having to
# remember the order of tools.
#
# Fast path for reviewers:
#     make setup       # one-shot: deps + Terraform state bucket + env
#     make dev         # LocalStack + chroma + backend server + frontend

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

REPO_ROOT := $(shell pwd)

.PHONY: help setup deps env bootstrap local-apply dev backend-dev frontend-dev \
        pipeline-dev ingest-local ingest-web crawl test eval audit clean check-tools

help:
	@echo "Haki AI — common targets:"
	@echo "  make setup          Install all deps, create .env, bootstrap Terraform state."
	@echo "  make dev            Run the full local stack (LocalStack + backend + frontend)."
	@echo "  make test           Run backend + frontend + pipeline tests."
	@echo "  make eval           Run the RAG evaluation harness against the golden set."
	@echo "  make audit CATEGORY=land   Per-case retrieval triage (hit / noise / rerank-loss)."
	@echo "  make ingest-local   Ingest statute chunks (processed-chunks/) into local ChromaDB."
	@echo "  make ingest-web     Ingest crawled web sources (faq-chunks/) into local ChromaDB."
	@echo "  make crawl LIMIT=200   Crawl SheriaPlex forum threads into LocalStack S3."
	@echo ""
	@echo "One-off plumbing:"
	@echo "  make deps           Install backend/pipeline/frontend dependencies."
	@echo "  make env            Copy .env.example → .env (skips if .env exists)."
	@echo "  make bootstrap      Create Terraform S3 state bucket."
	@echo "  make local-apply    terraform apply against LocalStack."
	@echo "  make clean          Remove build artefacts and local vectorstore."

# ── Setup ────────────────────────────────────────────────────────────────────
# One-shot target for cloning the repo and running it end-to-end. Safe
# to re-run — every step is idempotent.

setup: check-tools env deps bootstrap local-apply
	@echo ""
	@echo "──────────────────────────────────────────────────────────"
	@echo "✓ Haki AI is set up locally."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Add your LocalStack Pro token + LangSmith key to .env"
	@echo "  2. Pipeline: cd pipeline && npm run dev (downloads + OCRs PDFs)"
	@echo "  3. Ingest:   make ingest-local"
	@echo "  4. Run app:  make dev"
	@echo "──────────────────────────────────────────────────────────"

check-tools:
	@command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	@command -v node >/dev/null 2>&1 || { echo "ERROR: node not installed."; exit 1; }
	@command -v terraform >/dev/null 2>&1 || { echo "ERROR: terraform not installed."; exit 1; }
	@command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI not installed."; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not installed (needed for LocalStack)."; exit 1; }

env:
	@if [ -f .env ]; then \
	  echo "[setup] .env already exists, skipping copy"; \
	else \
	  cp .env.example .env; \
	  echo "[setup] created .env — edit it to add your LocalStack / LangSmith keys"; \
	fi

deps:
	@echo "[setup] installing backend deps (uv sync)"
	cd backend && uv sync
	@echo "[setup] installing pipeline deps (npm ci)"
	cd pipeline && npm ci --prefer-offline --no-audit
	@echo "[setup] installing frontend deps (npm ci)"
	cd frontend && npm ci --prefer-offline --no-audit

bootstrap:
	@echo "[setup] ensuring Terraform state bucket exists"
	./scripts/bootstrap.sh -auto-approve || true

local-apply:
	@echo "[setup] terraform apply against LocalStack"
	cd infra && $(MAKE) local-apply

# ── Dev loop ─────────────────────────────────────────────────────────────────

dev:
	@echo "Starting LocalStack + ChromaDB + backend server + frontend."
	@echo "Stop with Ctrl+C (cleans up all child processes)."
	@( \
	  trap 'kill 0' SIGINT SIGTERM EXIT; \
	  cd pipeline && docker compose up -d 2>/dev/null || true; \
	  cd $(REPO_ROOT)/backend && uv run python -m app.server_local & \
	  cd $(REPO_ROOT)/frontend && npm run dev & \
	  wait \
	)

backend-dev:
	cd backend && uv run python -m app.server_local

frontend-dev:
	cd frontend && npm run dev

pipeline-dev:
	cd pipeline && npm run dev

ingest-local:
	cd backend && uv run python -m app.ingest_local

# Ingest crawled web sources (SheriaPlex FAQs today; future KenyaLaw
# summaries tomorrow) into the same ChromaDB collection as statutes,
# tagged corpus="faq" so the FAQAgent can filter on it. Pass PREFIX=...
# to target a different crawled corpus.
ingest-web:
	cd backend && uv run python -m app.ingest_from_web_sources \
	  $(if $(PREFIX),--prefix $(PREFIX))

# Crawl SheriaPlex forum threads into s3://<bucket>/faq-chunks/.
# Safe defaults: 1.5s delay between requests + a hard LIMIT cap (default
# 40 via the script, override with LIMIT=200 to cover every thread
# discoverable from the forum index).
crawl:
	cd pipeline && npm run crawl -- $(if $(LIMIT),--limit $(LIMIT))

# ── Tests + evals ────────────────────────────────────────────────────────────

test:
	cd backend && uv run python -m unittest discover -s tests -p 'test_unit.py' -v
	cd frontend && npx tsc --noEmit
	cd pipeline && npm test --silent

eval:
	cd backend && uv run python -m evals.run

# Per-case retrieval audit — re-runs the pipeline for one category and
# classifies each top-5 as hit / noise-pollution / rerank-loss. Used to
# decide whether a retrieval regression is a chunker / filter bug or a
# genuine embedding-coverage gap. Set CATEGORY=<name> to narrow scope.
#
#   make audit CATEGORY=land
#   make audit CATEGORY=tenant TOP_K=10
audit:
	cd backend && uv run python -m evals.audit \
	  $(if $(CATEGORY),--category $(CATEGORY)) \
	  $(if $(TOP_K),--top-k $(TOP_K))

# ── Housekeeping ─────────────────────────────────────────────────────────────

clean:
	rm -rf backend/.local-vectorstore
	rm -rf backend/__pycache__
	rm -rf frontend/dist pipeline/dist
	rm -rf infra/modules/compute/lambda.zip infra/modules/compute/lambda-layer.zip
	@echo "[clean] done"
