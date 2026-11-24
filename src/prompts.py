"""
prompts.py — Canonical prompt definitions for all inference paths.

Single source of truth for SYSTEM_PROMPT and user prompt construction.
Used by: infer_local.py, infer_groq.py, infer_gemini.py, and dataset.py
(for SFT training formatting).

IMPORTANT: Any change here affects ALL models and training data.
Do not modify without understanding the downstream impact on:
  - Training data formatting (dataset.py)
  - Local inference (infer_local.py)
  - Groq/Llama inference (infer_groq.py)
  - Gemini inference (infer_gemini.py)
"""

SYSTEM_PROMPT: str = (
    "You are a sentiment classifier. Classify the sentiment of the given "
    "text as exactly one of: positive, neutral, or negative. "
    "Respond with ONLY the sentiment label, nothing else."
)


def build_user_prompt(text: str) -> str:
    """Build the user prompt for sentiment classification.

    Args:
        text: The tweet/text to classify.

    Returns:
        Formatted user prompt string.
    """
    return (
        "Classify the sentiment of the following text as positive, "
        "neutral, or negative.\n\nText: " + text.strip()
    )
