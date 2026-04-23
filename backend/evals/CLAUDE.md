# backend/evals — RAG evaluation harness

## Purpose
Phase 3 deliverable: a quantitative "did we improve the system" story
for the bootcamp rubric. Runs RAGAS + LLM-as-judge against a 30-item
golden set and writes a markdown report to `evals/reports/`. Also
emits a single `EvalScore` CloudWatch metric so regressions surface
on the main dashboard.

## Files
- `golden_set.jsonl` — 30 curated Q&A pairs (10 Constitution, 10
  Employment, 10 Land; ≥10 are Swahili / mixed). Each entry:
  `{question, reference_answer, expected_sources, language}`.
- `loader.py` — parses the golden set and validates each record.
- `runner.py` — invokes the compiled graph against every question
  and captures `{answer, citations, retrieved_contexts, language}`.
- `ragas_run.py` — optional RAGAS metrics: `faithfulness`,
  `answer_relevancy`, `context_precision`, `context_recall`. Uses
  Claude on Bedrock as the judge LLM. Skipped cleanly if `ragas` is
  not installed.
- `llm_judge.py` — LLM-as-judge scoring on 4 axes (accuracy,
  citation correctness, tone, language appropriateness), 0–5 each.
- `report.py` — writes `reports/{timestamp}.md` with the summary
  table + per-question diffs. Pushes `EvalScore` to CloudWatch.
- `run.py` — CLI entry: `uv run -m evals.run` (invoked by
  `make eval` locally and `eval-nightly.yml` in CI).
- `testset_generator.py` — RAGAS synthetic test-set generation.
  Loads chunks via `rag.catalog.get_catalog`, stratified-subsamples
  by statute, wires budget-tracked `ChatBedrock` + `BedrockEmbeddings`,
  runs two personas (English legal practitioner + Swahili
  pro-se litigant) at a 70/30 single-hop/multi-hop split, and maps
  rows back onto the `GoldenCase` schema.
- `generate.py` — CLI for the above: `make gen-testset SIZE=50
  MAX_COST=5`. Runs a pre-flight estimate, refuses to start if it
  exceeds `--max-cost`, writes `generated_set.jsonl` plus a sibling
  `generated_set.cost.md` report.
- `generation_cost.py` — `BudgetTracker` + `PRICE_TABLE` for mid-run
  hard-cap aborts, plus a post-run `aggregate_from_langsmith()` that
  walks the trace tree to produce the authoritative cost breakdown.

## Internal data flow

```mermaid
flowchart LR
  gs[golden_set.jsonl] --> ld[loader]
  ld --> rn[runner → graph.invoke per Q]
  rn --> ra[ragas_run optional]
  rn --> lj[llm_judge]
  ra --> rp[report]
  lj --> rp
  rp --> md[reports/*.md]
  rp --> cw[(CloudWatch EvalScore)]
```

## Conventions
- Never imports from `app.handler` — goes directly through the
  compiled LangGraph so evals reflect the agent, not the HTTP shim.
- Report paths include a timestamp so nightly runs don't clobber
  each other.
- The generated test-set is kept **separate** from `golden_set.jsonl`.
  Treat `generated_set.jsonl` as an exploration set — hand-pick rows
  worth promoting into the curated golden set; never auto-merge.
- All Bedrock-spending paths (the generator, RAGAS scoring) must run
  under a LangSmith `@traceable` scope so `generation_cost.py` /
  the nightly job can attribute spend back to a single trace.
