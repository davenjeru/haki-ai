# backend/agents — Two-tier multi-agent orchestration

## Purpose
Implements the Phase 2 supervisor + specialists pattern. The supervisor
uses Claude Haiku to route a user turn to 1–3 specialists (or `chat`
for chit-chat). Each specialist runs the Phase 1 advanced-RAG pipeline
scoped to its **legal domain** — a curated set of Kenyan primary
sources covering the same area of law. A synthesizer merges
multi-specialist answers.

## Files
- `supervisor.py` — Haiku JSON router. Emits
  `{"agents": ["employment", "constitution"], "reason": "..."}`.
  Falls back to the chat path on malformed output so the graph never
  dead-ends. `KNOWN_AGENTS` must stay in lockstep with
  `specialists.AGENT_REGISTRY`.
- `specialists.py` — domain specialists built from `AGENT_REGISTRY`:
  `constitution`, `employment`, `land` (Land Act + Cap. 301),
  `criminal` (Penal Code + CPC + Sexual Offences), `family`
  (Marriage + Children), `contracts` (Contract Act + Consumer
  Protection). Each scopes the RAG pipeline via a list-valued
  `source` metadata filter (Bedrock KB `in` clause in prod, Chroma
  `$in` locally) and returns
  `{response_text, citations, blocked, stop_reason}`.
- `chat.py` — conversational (no-RAG) path. Direct Claude
  `invoke_model` call with history — used when `supervisor` routes to
  `chat`. Zero citations by design.
- `classifier.py` — legacy binary intent classifier (`needs_rag`)
  kept for backward compatibility with older prompts/tests.
- `synthesizer.py` — merges ≥2 specialist outputs into one response:
  dedups citations by `(source, section)`, makes a single Haiku call
  to unify language/tone. Passes a single specialist through verbatim
  (no extra LLM call).

## Domain → source map

| Specialist     | Sources (OR-joined via `in`) |
|----------------|------------------------------|
| `constitution` | Constitution of Kenya 2010 |
| `employment`   | Employment Act 2007 |
| `land`         | Land Act 2012, Landlord and Tenant Act (Cap. 301) |
| `criminal`     | Penal Code (Cap. 63), Criminal Procedure Code (Cap. 75), Sexual Offences Act 2006 |
| `family`       | Marriage Act 2014, Children Act 2022 |
| `contracts`    | Law of Contract Act, Consumer Protection Act 2012 |
| `chat`         | (no retrieval) |

Adding a new statute to an existing domain is a one-line edit to the
filter list. A genuinely new domain requires entries in both
`AGENT_REGISTRY` and `supervisor.KNOWN_AGENTS`, plus a new bullet in
`SUPERVISOR_PROMPT`.

## Internal data flow

```mermaid
flowchart TB
  in[user turn + language] --> sup[supervisor.route]
  sup -->|chat| chat[chat.answer]
  sup -->|specialists list| fan[Send() to each]
  fan --> c[constitution]
  fan --> e[employment]
  fan --> l[land + tenancy]
  fan --> cr[criminal]
  fan --> fm[family]
  fan --> co[contracts + consumer]
  c --> syn[synthesizer.merge]
  e --> syn
  l --> syn
  cr --> syn
  fm --> syn
  co --> syn
  syn --> out[merged answer + citations]
  chat --> out
```

## Conventions
- Specialists never own their retrieval code — they delegate to
  `rag.pipeline.run_rag` so one advanced-RAG change ships to all of
  them at once.
- Every specialist returns the same dict shape so the
  `_specialist_outputs_reducer` on `specialist_outputs` can accumulate
  them in parallel while still supporting the per-turn reset sentinel.
- The `source` metadata filter is always a **list**, even for
  single-statute specialists, so adding another statute later doesn't
  require changing the filter shape.
