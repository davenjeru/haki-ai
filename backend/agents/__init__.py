"""
Multi-agent orchestration package.

Two-tier design:

  Tier 1 (Supervisor)
    `supervisor.py` routes each user turn to one or more specialists based
    on a cheap Haiku call. Possible specialists: constitution, employment,
    land, chat.

  Tier 2 (Specialist sub-agents)
    Each specialist is a compiled LangGraph subgraph that runs the Phase 1
    advanced-RAG pipeline filtered to its own statute(s). Specialists share
    the builder in `specialists.py` \u2014 only their `source` metadata filter
    and display name differ.

When the supervisor fires \u22652 specialists (cross-cutting questions), the
`synthesizer.py` node merges their answers into a single unified response
with unioned citations. Single-specialist runs bypass the synthesizer and
surface the specialist answer verbatim.

The `chat` specialist wraps the existing chat_node logic unchanged \u2014 it
covers greetings, memory lookups, and the bilingual off-topic refusal.
"""

from .specialists import AGENT_REGISTRY, build_specialist
from .supervisor import route_supervisor
from .synthesizer import synthesize

__all__ = [
    "AGENT_REGISTRY",
    "build_specialist",
    "route_supervisor",
    "synthesize",
]
