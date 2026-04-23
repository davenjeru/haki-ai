# backend/agents — Two-tier multi-agent orchestration

## Purpose
Implements the Phase 2 supervisor + specialists pattern. The supervisor
uses Claude Haiku to route a user turn to 1–3 specialists (or `chat`
for chit-chat). Each specialist runs the Phase 1 advanced-RAG pipeline
scoped to its own law. A synthesizer merges multi-specialist answers.

## Files
- `supervisor.py` — Haiku JSON router. Emits
  `{"agents": ["employment", "constitution"], "reason": "..."}`.
  Falls back to the chat path on malformed output so the graph never
  dead-ends.
- `specialists.py` — factory for `ConstitutionAgent`, `EmploymentAgent`,
  `LandAgent`. Each scopes the RAG pipeline by a `source` metadata
  filter and returns `{response_text, citations, blocked,
  stop_reason}`.
- `chat.py` — conversational (no-RAG) path. Direct Claude
  `invoke_model` call with history — used when `supervisor` routes to
  `chat`. Zero citations by design.
- `classifier.py` — legacy binary intent classifier (`needs_rag`)
  kept for backward compatibility with older prompts/tests.
- `synthesizer.py` — merges ≥2 specialist outputs into one response:
  dedups citations by `(source, section)`, makes a single Haiku call
  to unify language/tone. Passes a single specialist through verbatim
  (no extra LLM call).

## Internal data flow

```mermaid
flowchart TB
  in[user turn + language] --> sup[supervisor.route]
  sup -->|chat| chat[chat.answer]
  sup -->|specialists list| fan[Send() to each]
  fan --> c[ConstitutionAgent]
  fan --> e[EmploymentAgent]
  fan --> l[LandAgent]
  c --> syn[synthesizer.merge]
  e --> syn
  l --> syn
  syn --> out[merged answer + citations]
  chat --> out
```

## Conventions
- Specialists never own their retrieval code — they delegate to
  `rag.pipeline.run_pipeline` so one advanced-RAG change ships to all
  of them at once.
- Every specialist returns the same tuple shape so the `operator.add`
  reducer on `specialist_outputs` can accumulate them in parallel.
