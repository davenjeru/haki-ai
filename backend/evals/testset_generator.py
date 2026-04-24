"""
RAGAS synthetic test-set generator for Haki AI.

Loads the Kenyan-statute chunk catalog from S3 (same path the prod
retriever uses, so generated questions map 1:1 to what the agent can
actually retrieve), subsamples the corpus, and runs
``ragas.testset.TestsetGenerator`` to synthesise bilingual
(question, reference_answer, reference_contexts) triples.

Output schema matches :class:`evals.loader.GoldenCase` so the generated
file is a drop-in input to the rest of the eval harness — though in
practice we keep it separate (``generated_set.jsonl``) and hand-pick
promising rows into ``golden_set.jsonl``.

RAGAS and ``langchain-aws`` are imported lazily because they live in the
opt-in ``evals`` dependency group. A clear error message points the
caller at ``uv sync --group evals`` when the group isn't installed.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from app.config import Config
from clients import make_bedrock_runtime, make_s3_listing

from .generation_cost import (
    BudgetTracker,
    make_budget_tracking_embeddings,
    make_budget_tracking_llm,
)

log = logging.getLogger(__name__)

_DEFAULT_SUBSAMPLE_SIZE = 200
_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

# chunkTypes safe to anchor questions on. Excludes toc/preamble/short-title
# (boilerplate — questions about them test the chunker, not the retriever),
# and excludes `definitions` (definitional questions circularly anchor on
# the Interpretation section, which §5b of the retrieval plan flagged as
# a retrieval false-positive magnet).
#
# Legacy chunks without a chunkType tag (Land Act, Landlord-Tenant, and
# the FAQ corpus ingested before the boundary-bleed fix) are treated as
# ``body`` for backwards compatibility, matching ``rag.filters``'
# ``is_boilerplate`` default.
_ANCHORABLE_CHUNK_TYPES: frozenset[str] = frozenset({"body"})


# ── Persona definitions ────────────────────────────────────────────────────
#
# Two personas mixed 50/50 so RAGAS produces an even English/Swahili
# split (chosen interactively at --config time; see CLI). Persona
# descriptions are prompted into the question synthesiser by RAGAS and
# materially affect phrasing, formality, and code-switching density.

ENGLISH_PERSONA = (
    "A junior Kenyan advocate researching a client matter. Asks precise, "
    "formal legal questions in English, typically naming the statute and "
    "article they want to understand. Expects answers grounded in the "
    "Constitution and Acts of Parliament."
)

SWAHILI_PERSONA = (
    "A Swahili-speaking pro-se litigant in Nairobi with no legal training. "
    "Asks questions entirely in Swahili using everyday legal vocabulary "
    "(haki, sheria, katiba, kifungu). May mix in one or two English legal "
    "terms where a direct Swahili equivalent is uncommon. Wants plain-"
    "language explanations of their rights."
)


# ── Corpus loading ──────────────────────────────────────────────────────────


@dataclass
class _Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def load_corpus(
    config: Config,
    *,
    anchorable_chunk_types: frozenset[str] = _ANCHORABLE_CHUNK_TYPES,
) -> list[_Chunk]:
    """Reads the full chunk catalog from S3 (LocalStack when ``is_local``).

    Reuses :func:`rag.catalog.get_catalog` so the generator sees the same
    corpus the retriever does — if a chunk isn't retrievable in prod,
    RAGAS shouldn't be allowed to anchor a question on it.

    Filters to ``chunkType in anchorable_chunk_types`` (default: body
    only). Without this filter RAGAS anchors ~30% of questions on TOC
    chunks whose metadata lists a single section (e.g. "Section 398")
    but whose body text reels through every index entry (Sections 101,
    138, 273, 329, ...). The generated expected_sections then points
    at the TOC chunk's metadata rather than where the answer actually
    lives. Chunks missing the chunkType tag (older ingests pre-dating
    the boundary-bleed fix) default to ``body`` for backwards compat.
    """
    from rag.catalog import get_catalog

    s3 = make_s3_listing(config)
    entries = get_catalog(s3, config.s3_bucket)

    total = 0
    skipped_by_type: dict[str, int] = {}
    chunks: list[_Chunk] = []
    for e in entries:
        total += 1
        text = e.get("text")
        if not text:
            continue
        metadata = e.get("metadata") or {}
        chunk_type = metadata.get("chunkType") or "body"
        if chunk_type not in anchorable_chunk_types:
            skipped_by_type[chunk_type] = skipped_by_type.get(chunk_type, 0) + 1
            continue
        chunks.append(
            _Chunk(
                chunk_id=e["chunkId"],
                text=text,
                metadata=metadata,
            )
        )

    skip_summary = (
        ", ".join(f"{t}={n}" for t, n in sorted(skipped_by_type.items()))
        or "none"
    )
    log.info(
        "loaded %d/%d chunks from s3://%s/processed-chunks/ (anchorable=%s, skipped=%s)",
        len(chunks),
        total,
        config.s3_bucket,
        ",".join(sorted(anchorable_chunk_types)),
        skip_summary,
    )
    return chunks


def subsample(
    chunks: list[_Chunk],
    *,
    size: int = _DEFAULT_SUBSAMPLE_SIZE,
    seed: int = 0,
) -> list[_Chunk]:
    """Stratified subsample: evenly distributes across ``metadata.source``.

    Without stratification a random sample from a ~1.2K-chunk corpus
    skews toward the largest statute (Employment Act) and under-samples
    the Constitution and specialty codes. Stratifying by source keeps
    every category represented in the generated set roughly
    proportionally to its share of the corpus.
    """
    if size >= len(chunks):
        return list(chunks)

    rng = random.Random(seed)
    by_source: dict[str, list[_Chunk]] = {}
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        by_source.setdefault(source, []).append(c)

    # Proportional allocation with a floor of 1 per source so no statute
    # silently drops out. Any leftover slots go to the largest groups.
    total = len(chunks)
    allocations = {
        src: max(1, round(size * len(bucket) / total))
        for src, bucket in by_source.items()
    }
    overflow = sum(allocations.values()) - size
    if overflow > 0:
        for src in sorted(allocations, key=lambda s: -len(by_source[s])):
            if overflow <= 0:
                break
            if allocations[src] > 1:
                allocations[src] -= 1
                overflow -= 1

    picked: list[_Chunk] = []
    for src, n in allocations.items():
        bucket = by_source[src]
        picked.extend(rng.sample(bucket, min(n, len(bucket))))
    rng.shuffle(picked)
    return picked


# ── RAGAS wiring ────────────────────────────────────────────────────────────


def _lazy_imports():
    """Imports ragas + langchain-aws lazily. Raises a clear error with
    the install hint when the ``evals`` group isn't synced.
    """
    try:
        from langchain_aws import ChatBedrock, BedrockEmbeddings  # type: ignore
        from langchain_core.documents import Document  # type: ignore
        from ragas.run_config import RunConfig  # type: ignore
        from ragas.testset import TestsetGenerator  # type: ignore
        from ragas.testset.persona import Persona  # type: ignore
        from ragas.testset.synthesizers import (  # type: ignore
            SingleHopSpecificQuerySynthesizer,
            MultiHopAbstractQuerySynthesizer,
        )
        return {
            "ChatBedrock": ChatBedrock,
            "BedrockEmbeddings": BedrockEmbeddings,
            "Document": Document,
            "RunConfig": RunConfig,
            "TestsetGenerator": TestsetGenerator,
            "Persona": Persona,
            "SingleHopSpecificQuerySynthesizer": SingleHopSpecificQuerySynthesizer,
            "MultiHopAbstractQuerySynthesizer": MultiHopAbstractQuerySynthesizer,
        }
    except ImportError as err:
        raise RuntimeError(
            "RAGAS test-set generator requires the 'evals' dependency group. "
            "Install with:  uv sync --group evals"
        ) from err


def _build_documents(chunks: list[_Chunk], Document) -> list:
    """Converts ``_Chunk`` records into LangChain ``Document`` objects.

    Preserves every metadata key so we can map back to
    ``expected_sources`` / ``expected_sections`` during output
    post-processing.
    """
    docs = []
    for c in chunks:
        docs.append(
            Document(
                page_content=c.text,
                metadata={**c.metadata, "chunk_id": c.chunk_id},
            )
        )
    return docs


def _build_generator(
    config: Config,
    tracker: BudgetTracker,
    lib: dict[str, Any],
) -> Any:
    """Constructs the ``TestsetGenerator`` with budget-tracked Bedrock
    wrappers and the two Haki-AI personas.
    """
    bedrock_runtime = make_bedrock_runtime(config)
    llm_model_id = config.bedrock_model_id
    embed_model_id = config.embedding_model_id or _DEFAULT_EMBED_MODEL

    if not llm_model_id:
        raise RuntimeError(
            "BEDROCK_MODEL_ID is not set; cannot run RAGAS test-set generation."
        )

    chat = lib["ChatBedrock"](
        model_id=llm_model_id,
        client=bedrock_runtime,
        # RAGAS synthesiser prompts can be long; keep max_tokens generous.
        model_kwargs={"temperature": 0.3, "max_tokens": 2048},
    )
    embed = lib["BedrockEmbeddings"](
        model_id=embed_model_id,
        client=bedrock_runtime,
    )

    ragas_llm = make_budget_tracking_llm(chat, tracker, llm_model_id)
    ragas_embed = make_budget_tracking_embeddings(embed, tracker, embed_model_id)

    personas = [
        lib["Persona"](name="english_lawyer", role_description=ENGLISH_PERSONA),
        lib["Persona"](name="swahili_litigant", role_description=SWAHILI_PERSONA),
    ]

    return lib["TestsetGenerator"](
        llm=ragas_llm,
        embedding_model=ragas_embed,
        persona_list=personas,
    )


def _query_distribution(lib: dict[str, Any], ragas_llm: Any) -> list:
    """70% single-hop specific, 30% multi-hop abstract.

    The mix matches the golden set's texture (most questions quote a
    specific article; a few are cross-statute synthesis). We don't use
    ``MultiHopSpecificQuerySynthesizer`` because the Kenyan-statute
    corpus has relatively few cross-statute joins and RAGAS fails a lot
    of candidates with "cannot find sufficient multi-hop anchors".
    """
    return [
        (lib["SingleHopSpecificQuerySynthesizer"](llm=ragas_llm), 0.7),
        (lib["MultiHopAbstractQuerySynthesizer"](llm=ragas_llm), 0.3),
    ]


# ── Schema mapping ──────────────────────────────────────────────────────────

# Maps statute name (in chunk metadata) → golden-set category bucket.
# Unknowns fall through to ``"uncategorised"`` — keeps the generated
# file parseable even when the pipeline adds new sources.
_SOURCE_TO_CATEGORY = {
    "Constitution of Kenya 2010": "constitution",
    "Employment Act 2007": "employment",
    "Land Registration Act 2012": "land",
    "Land Act 2012": "land",
    "Landlord and Tenant Act (Cap. 301)": "tenancy",
    "Sexual Offences Act 2006": "criminal",
    "Penal Code (Cap. 63)": "criminal",
    "Criminal Procedure Code (Cap. 75)": "criminal",
    "Children Act 2022": "family",
    "Marriage Act 2014": "family",
    "Law of Contract Act": "contract",
    "Consumer Protection Act 2012": "consumer",
}


def _infer_category(sources: list[str]) -> str:
    for src in sources:
        if src in _SOURCE_TO_CATEGORY:
            return _SOURCE_TO_CATEGORY[src]
    return "uncategorised"


# Swahili markers; presence of any indicates the language isn't pure
# English. Tuned against the golden set: every Swahili/mixed case hits
# at least one of these tokens, zero English cases do.
_SWAHILI_MARKERS = re.compile(
    r"\b(ni|nina|je|haki|sheria|katiba|kifungu|mimi|mtu|mke|mume|"
    r"mwananchi|raia|ndoa|watoto|hakimu|mahakama|mshtakiwa|wakili|"
    r"kwa|nani|nini|gani|vipi|kuhusu|inasema|ambayo|ambao|kama|"
    r"lakini|yangu|yako|wangu|wao|ya|wa)\b",
    re.IGNORECASE,
)


def _detect_language(question: str) -> str:
    """Classifies a generated question as english / swahili / mixed.

    We look at Swahili-token density rather than running Comprehend at
    generation time: (a) LocalStack's Comprehend doesn't implement
    language detection; (b) batch classification here would cost real
    AWS money; (c) the marker list is small and surgical enough that
    the false-positive rate on Kenyan-English legal prose is <5%.
    """
    swahili_hits = len(_SWAHILI_MARKERS.findall(question))
    words = re.findall(r"\w+", question)
    if not words:
        return "english"
    density = swahili_hits / len(words)
    if density >= 0.5:
        return "swahili"
    if density >= 0.1:
        return "mixed"
    return "english"


@dataclass
class GeneratedCase:
    """Mirrors :class:`evals.loader.GoldenCase` exactly.

    Kept as its own dataclass so we can ship ragas-specific fields in
    the extras (e.g. `persona`, `synthesiser_type`) under a sidecar if
    needed, without changing the golden-set schema.
    """

    id: str
    category: str
    question: str
    reference_answer: str
    expected_sources: list[str]
    expected_sections: list[str]
    language: str

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _map_sample(idx: int, sample: Any) -> GeneratedCase | None:
    """Converts a RAGAS testset row into a :class:`GeneratedCase`.

    Returns ``None`` for rows we can't safely use (empty question,
    missing reference, no retrievable contexts) — RAGAS occasionally
    emits these and we'd rather drop them than ship broken eval cases.
    """
    eval_sample = getattr(sample, "eval_sample", sample)
    question = (
        getattr(eval_sample, "user_input", None)
        or getattr(eval_sample, "question", None)
        or ""
    ).strip()
    reference = (
        getattr(eval_sample, "reference", None)
        or getattr(eval_sample, "ground_truth", None)
        or ""
    ).strip()
    contexts = (
        getattr(eval_sample, "reference_contexts", None)
        or getattr(eval_sample, "contexts", None)
        or []
    )
    if not question or not reference or not contexts:
        return None

    # ``reference_contexts`` on the RAGAS sample is a list of strings
    # (chunk text). The originating Document's metadata isn't attached
    # to the sample directly in 0.2.x; we pull it from the sample's
    # ``synthesizer_name`` trace when available and fall back to an
    # empty sources list. The CLI then enriches via the raw node list
    # below.
    sources: list[str] = []
    sections: list[str] = []

    case_id = f"gen-{idx:03d}"
    category = _infer_category(sources)
    language = _detect_language(question)

    return GeneratedCase(
        id=case_id,
        category=category,
        question=question,
        reference_answer=reference,
        expected_sources=sources,
        expected_sections=sections,
        language=language,
    )


def _match_context_to_chunk(
    ctx: str,
    corpus: list[_Chunk],
    prefix_index: dict[str, _Chunk],
) -> _Chunk | None:
    """Best-effort reverse-lookup from a RAGAS reference-context string
    to the corpus ``_Chunk`` it came from.

    RAGAS 0.2.x stores ``reference_contexts`` as plain strings rather
    than node references, and it occasionally mutates the text
    (whitespace collapse, prefix trimming when it concatenates multiple
    hops, header stripping). A single prefix-match therefore misses
    ~30% of contexts and silently ships empty ``expected_sources``.

    The strategy below tries four increasingly fuzzy matches. Order
    matters: earlier strategies are O(1) hash lookups; the substring
    fallback is O(N) over the corpus and only runs when the faster
    strategies miss.
    """
    if not ctx or not isinstance(ctx, str):
        return None

    # 1. Exact prefix(80). Hits the common case in O(1).
    match = prefix_index.get(ctx[:80])
    if match is not None:
        return match

    # 2. Exact prefix(40). Covers cases where RAGAS trimmed a short
    # prefix (e.g. a header line) before storing.
    short_prefix = ctx[:40]
    for key, chunk in prefix_index.items():
        if key.startswith(short_prefix):
            return chunk

    # 3. Substring match: first 120 chars of the context appear
    # verbatim inside a chunk. Handles RAGAS' "glue two chunks" hops
    # where ``ctx`` = ``chunk_A.text + "\n\n" + chunk_B.text`` and
    # the prefix is part of chunk_A's middle.
    probe = ctx[:120].strip()
    if len(probe) >= 60:
        for chunk in corpus:
            if chunk.text and probe in chunk.text:
                return chunk

    # 4. Reverse substring: a chunk's first 120 chars appear inside
    # the context. Handles the inverse: RAGAS prepended its own summary
    # + the full chunk text.
    for chunk in corpus:
        if chunk.text and len(chunk.text) >= 60:
            head = chunk.text[:120].strip()
            if head and head in ctx:
                return chunk

    return None


def _enrich_from_nodes(
    cases: list[GeneratedCase],
    testset: Any,
    corpus: list[_Chunk],
) -> tuple[list[GeneratedCase], int]:
    """Fills ``expected_sources`` / ``expected_sections`` by matching
    each case's ``reference_contexts`` text back to chunk metadata.

    Returns ``(enriched_cases, dropped)`` where ``dropped`` counts rows
    whose contexts couldn't be matched to any corpus chunk — those
    rows are omitted rather than shipped with empty citations, because
    downstream eval scoring treats an empty ``expected_sources`` as a
    refusal case and computes misleading metrics against it.
    """
    prefix_index = {c.text[:80]: c for c in corpus if c.text}

    samples = list(getattr(testset, "samples", []) or [])
    enriched: list[GeneratedCase] = []
    dropped = 0
    for case, sample in zip(cases, samples):
        eval_sample = getattr(sample, "eval_sample", sample)
        contexts = (
            getattr(eval_sample, "reference_contexts", None)
            or getattr(eval_sample, "contexts", None)
            or []
        )
        sources: list[str] = []
        sections: list[str] = []
        unmatched_ctxs = 0
        for ctx in contexts:
            match = _match_context_to_chunk(ctx, corpus, prefix_index)
            if match is None:
                unmatched_ctxs += 1
                continue
            src = match.metadata.get("source")
            sec = match.metadata.get("section")
            if src and src not in sources:
                sources.append(src)
            if sec and sec not in sections:
                sections.append(sec)

        if not sources:
            log.warning(
                "dropping %s: enrichment found no source chunk "
                "(contexts=%d, unmatched=%d)",
                case.id,
                len(contexts),
                unmatched_ctxs,
            )
            dropped += 1
            continue

        enriched.append(
            GeneratedCase(
                id=case.id,
                category=_infer_category(sources),
                question=case.question,
                reference_answer=case.reference_answer,
                expected_sources=sources,
                expected_sections=sections,
                language=case.language,
            )
        )

    # Any cases beyond what ``samples`` covered (shouldn't happen,
    # defensive) pass through unchanged — they have empty sources
    # already and will fail the same filter above if we ever reach
    # this path, but we keep it here to avoid swallowing edge cases.
    for extra in cases[len(samples):]:
        if extra.expected_sources:
            enriched.append(extra)
        else:
            dropped += 1

    return enriched, dropped


# ── Public entrypoint ───────────────────────────────────────────────────────


@dataclass
class GenerationResult:
    cases: list[GeneratedCase]
    trace_id: str | None
    tracker: BudgetTracker
    corpus_size: int
    subsample_size: int


def generate_testset(
    *,
    config: Config,
    size: int,
    subsample_size: int = _DEFAULT_SUBSAMPLE_SIZE,
    max_cost: float | None = None,
    seed: int = 0,
    trace_name: str = "ragas_testset_generation",
) -> GenerationResult:
    """Runs the full generation pipeline end-to-end.

    The whole synthesis is wrapped in a ``@traceable`` span named
    ``ragas_testset_generation`` so LangSmith captures every downstream
    LLM/embedding call under a single trace. The trace id is returned
    so the CLI can query LangSmith for authoritative token counts.
    """
    lib = _lazy_imports()

    corpus = load_corpus(config)
    if not corpus:
        raise RuntimeError(
            f"No chunks found in s3://{config.s3_bucket}/processed-chunks/ — "
            "run `make ingest-local` first."
        )
    sampled = subsample(corpus, size=subsample_size, seed=seed)
    log.info("subsampled %d → %d chunks for generation", len(corpus), len(sampled))

    tracker = BudgetTracker(max_cost=max_cost)
    generator = _build_generator(config, tracker, lib)
    documents = _build_documents(sampled, lib["Document"])
    distribution = _query_distribution(lib, generator.llm)

    # ``@traceable`` is a no-op when LANGSMITH_TRACING=false; we still
    # attach it so the trace is captured whenever the env is configured
    # (CI, dev with .env key). Metadata gets stamped onto the root run
    # so "how much did this cost" is visible in the LangSmith UI too.
    try:
        from langsmith import traceable  # type: ignore
        from langsmith.run_helpers import get_current_run_tree  # type: ignore
    except ImportError:
        traceable = None  # type: ignore
        get_current_run_tree = None  # type: ignore

    trace_id_holder: dict[str, str | None] = {"id": None}

    # Bedrock on-demand quotas for Claude Haiku 4.5 in us-east-1 are
    # tight enough (~200 RPM account-wide) that the RAGAS default
    # max_workers=16 trips ThrottlingException after ~60-90s of KG
    # extraction. Cap workers to 4 and bump retries so transient
    # throttles recover instead of aborting a paid run.
    run_config = lib["RunConfig"](
        max_workers=4,
        max_retries=8,
        timeout=300,
    )

    def _invoke_generator():
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=size,
            query_distribution=distribution,
            run_config=run_config,
        )
        if get_current_run_tree is not None:
            rt = get_current_run_tree()
            if rt is not None:
                trace_id_holder["id"] = str(getattr(rt, "trace_id", rt.id))
                rt.add_metadata(
                    {
                        "corpus_size": len(corpus),
                        "subsample_size": len(sampled),
                        "testset_size_requested": size,
                        "testset_size_generated": len(
                            list(getattr(testset, "samples", []) or [])
                        ),
                        "llm_model_id": config.bedrock_model_id,
                        "embed_model_id": config.embedding_model_id,
                        "running_cost_usd": tracker.total_cost,
                    }
                )
        return testset

    if traceable is not None:
        testset = traceable(
            name=trace_name,
            tags=["testset-generation", "ragas", "haki-ai"],
        )(_invoke_generator)()
    else:
        testset = _invoke_generator()

    # Map → enrich → drop unmatched
    samples = list(getattr(testset, "samples", []) or [])
    raw_cases = [_map_sample(i + 1, s) for i, s in enumerate(samples)]
    raw_cases = [c for c in raw_cases if c is not None]
    cases, dropped = _enrich_from_nodes(raw_cases, testset, corpus)
    if dropped:
        log.warning(
            "dropped %d/%d generated rows with unmatchable reference contexts",
            dropped,
            len(raw_cases),
        )

    return GenerationResult(
        cases=cases,
        trace_id=trace_id_holder["id"],
        tracker=tracker,
        corpus_size=len(corpus),
        subsample_size=len(sampled),
    )
