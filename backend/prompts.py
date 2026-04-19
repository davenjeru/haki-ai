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
