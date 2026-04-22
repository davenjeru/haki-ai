"""
System prompt builder (step 2).

build_system_prompt(language) returns a language-appropriate system prompt
for the retrieve_and_generate call. The prompt:
  - Instructs the model to answer only Kenyan law questions
  - Sets the response language based on Comprehend detection
  - Requires every answer to cite Act + Chapter/Part + Section
  - Provides a bilingual refusal for out-of-scope questions

The guardrail layer (step 4) is the safety backstop; this prompt is the
model's behavioural instruction, not the security boundary.
"""

# Refusal string used both in the system prompt and as the blocked response
# returned when stopReason == "guardrail_intervened" (handler step 4).
BILINGUAL_REFUSAL = (
    "Mimi ni msaidizi wa kisheria wa Kenya tu. / "
    "I can only help with Kenyan legal matters."
)

_CITATION_RULE = (
    "Every answer MUST include a citation in the format: "
    "[Act name — Chapter/Part — Section number and title]. "
    "If you cannot find a relevant law in the provided context, say so clearly "
    "rather than answering from general knowledge."
)

_SCOPE_RULE = (
    f"You are a Kenyan legal aid assistant. You answer questions about Kenyan law only. "
    f"For any question that is not about Kenyan law, respond exactly with: "
    f'"{BILINGUAL_REFUSAL}"'
)

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "english": "Respond in English.",
    "swahili": "Jibu kwa Kiswahili.",
    "mixed": (
        "The user is mixing English and Swahili. "
        "Respond in both languages: give your full answer in English first, "
        "then repeat the key points in Swahili."
    ),
}


def build_system_prompt(language: str) -> str:
    """
    Returns the system prompt for the given detected language.

    Args:
        language: "english", "swahili", or "mixed" — from detect_language().

    Returns:
        A complete system prompt string ready to pass to retrieve_and_generate.
    """
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["english"])
    return "\n\n".join([_SCOPE_RULE, lang_instruction, _CITATION_RULE])


# ── Chat-only system prompt (no retrieval, no citations) ──────────────────────
# Used by chat_node when classify_intent decides the latest turn does not need
# RAG (e.g. "my name is Dave", "what is my name?", greetings). The model still
# refuses topics that fall outside Kenyan legal aid — the conversational tone
# allowance only covers meta-questions about the chat itself or friendly chit-
# chat that precedes a real legal question.

_CHAT_SCOPE_RULE = (
    "You are Haki AI, a Kenyan legal aid assistant. "
    "You only hold conversations that support answering Kenyan law questions "
    "(Constitution of Kenya 2010, Employment Act 2007, Land Act 2012). "
    "Small talk, clarifying questions, and recalling details the user has "
    "shared earlier in this conversation are all fine. "
    "For any substantive non-legal question (e.g. weather, sports, other "
    f'jurisdictions, medical/financial advice), respond exactly with: "{BILINGUAL_REFUSAL}"'
)

_CHAT_STYLE_RULE = (
    "Keep replies short and natural. Do NOT invent legal citations; this turn "
    "was classified as conversational, so the user is not asking for a legal "
    "answer. If the user asks a legal question, acknowledge it and invite them "
    "to restate it — the next turn will be routed through the legal retrieval "
    "pipeline."
)


def build_chat_system_prompt(language: str) -> str:
    """
    Returns the system prompt used by chat_node (no retrieval, no citations).
    """
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["english"])
    return "\n\n".join([_CHAT_SCOPE_RULE, lang_instruction, _CHAT_STYLE_RULE])


# ── Classifier prompt (routes each turn to rag_node or chat_node) ─────────────
# Returned verbatim as the system prompt for a cheap Haiku call. The model is
# instructed to respond with a strict single-key JSON object so the caller can
# parse reliably.

CLASSIFIER_PROMPT = """You are a routing classifier for a Kenyan legal aid chatbot.

Given the conversation so far, decide whether the USER'S MOST RECENT MESSAGE
requires retrieval from the Kenyan legal corpus (Constitution of Kenya 2010,
Employment Act 2007, Land Act 2012) to be answered well.

Respond with exactly one JSON object and nothing else:
  {"needs_rag": true}  or  {"needs_rag": false}

Rules:
- needs_rag = true   when the latest message asks about Kenyan law,
                     statutes, rights, procedures, cases, or specific sections.
- needs_rag = false  when the latest message is conversational:
                     greetings, thanks, introductions ("my name is X"),
                     memory lookups about earlier chat ("what is my name"),
                     clarifying questions about the bot itself, or smalltalk.
- If the latest message is clearly off-topic (other jurisdictions, weather,
  sports), still return needs_rag=false — chat_node will handle the refusal.

Examples:
  U: "Hi there"                                          -> {"needs_rag": false}
  U: "My name is Dave"                                   -> {"needs_rag": false}
  U: "What is my name?"                                  -> {"needs_rag": false}
  U: "Thanks!"                                           -> {"needs_rag": false}
  U: "What does section 40 of the Employment Act say?"   -> {"needs_rag": true}
  U: "What are my rights if I am fired without notice?"  -> {"needs_rag": true}
  U: "Sheria ya ajira inasema nini kuhusu kufukuzwa?"    -> {"needs_rag": true}
  U: "What about probation?" (after a legal question)    -> {"needs_rag": true}
  U: "How's the weather in Nairobi?"                     -> {"needs_rag": false}
"""
