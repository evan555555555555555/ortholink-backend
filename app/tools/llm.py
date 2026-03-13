"""
OrthoLink LLM Tool
gpt-4o structured output wrapper.
HC-9: gpt-4o for generation (NOT embedding).
"""

import json
import logging
import time
from typing import Optional, Type, TypeVar

from openai import OpenAI, RateLimitError
from pydantic import BaseModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ─── Exponential Backoff for 429 Rate Limits ────────────────────────────
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds; doubles each retry (2s, 4s, 8s)


def _retry_on_rate_limit(func):
    """Decorator: retries OpenAI calls up to 3× on 429 with exponential backoff."""
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                last_exc = e
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "OpenAI 429 rate limit (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("OpenAI 429 rate limit: exhausted %d retries", _MAX_RETRIES)
        raise last_exc  # type: ignore[misc]
    return wrapper

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Lazy-initialize OpenAI client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


@_retry_on_rate_limit
def structured_completion(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> T:
    """
    Get a structured response from gpt-4o parsed into a Pydantic model.

    HC-9 ENFORCED: Only gpt-4o used for generation.
    """
    settings = get_settings()
    client = _get_client()

    response = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    parsed = json.loads(content)
    return response_model.model_validate(parsed)


@_retry_on_rate_limit
def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    Get a plain text response from gpt-4o.

    HC-9 ENFORCED: Only gpt-4o used for generation.
    """
    settings = get_settings()
    client = _get_client()

    response = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response")

    return content


@_retry_on_rate_limit
def classify_with_llm(
    item_text: str,
    regulation_text: str,
    country: str,
    device_class: str,
) -> dict:
    """
    Use gpt-4o to classify a distributor document item against a regulation.
    Returns classification with confidence and explanation.
    """
    # CFS is classified via retrieval: seed_cfs_regulatory.py indexes CFS as a regulatory
    # requirement per country, so semantic search surfaces it; no hardcoded list here.
    system_prompt = (
        "You are a senior regulatory affairs specialist with 15 years of experience in "
        "medical device regulations across 50+ countries. You classify document requirements "
        "with precision and ZERO inference — only facts from the provided regulatory text.\n\n"
        "ZERO-INFERENCE PROTOCOL:\n"
        "Before classifying, you MUST internally map:\n"
        "  Column A (Regulatory Requirement from provided text) → Column B (Distributor Item)\n"
        "If no Column A entry exists for the distributor item, classify as EXTRA.\n"
        "If a Column A entry exists but Column B has no match, classify as MISSING.\n"
        "You MUST quote the exact regulatory text excerpt that justifies your classification.\n\n"
        "EVIDENCE REQUIREMENTS:\n"
        "- Every classification MUST include a direct quote from the regulatory text (≤50 words)\n"
        "- If you cannot find supporting text, set status=EXTRA and confidence=0.5\n"
        "- Never infer requirements that are not explicitly stated in the provided text\n"
        "- Never fabricate citations — cite only text you were given\n\n"
        "MANDATORY OVERRIDE RULES (Class II/III devices):\n"
        "- ISO 13485 QMS Certificate → ALWAYS REQUIRED (never EXTRA)\n"
        "- CE Certificate / Declaration of Conformity → ALWAYS REQUIRED (never EXTRA)\n"
        "- Power of Attorney / AR Agreement → ALWAYS REQUIRED for foreign manufacturers\n"
        "- Free Sale Certificate → ALWAYS REQUIRED for export markets\n\n"
        "BUSINESS vs REGULATORY distinction:\n"
        "Items like 'Bank Statement', 'Financial Guarantee', 'Insurance Certificate', "
        "'Business License' are BUSINESS due-diligence items → classify as OPTIONAL.\n\n"
        "REVOKED REGULATION RULE (zero tolerance):\n"
        "If regulatory text indicates a law was REVOKED/REPEALED/SUPERSEDED:\n"
        "(1) status='MISSING' + rejection_code='ERR_REVOKED_LAW'\n"
        "(2) Citation field: ONLY the current replacement regulation\n"
        "(3) Explanation: state which law was revoked and what replaced it\n"
        "NEVER classify as REQUIRED if legal basis is a revoked law."
    )

    user_prompt = f"""Classify the following document item against the regulatory requirement.

Document Item (from distributor): "{item_text}"
Regulatory Requirement Text: "{regulation_text}"
Country: {country}
Device Class: {device_class}

STEP 1 — Internal Mapping (think before answering):
Map the distributor item to the closest regulatory requirement from the text above.
If no match exists in the regulatory text, the item is EXTRA.

STEP 2 — Classification:
- REQUIRED: The regulatory text explicitly mandates this item
- EXTRA: No regulatory basis found in the provided text
- MISSING: A required item is absent from the distributor's submission
- OPTIONAL: Recommended but not legally required per the text

STEP 3 — Evidence:
You MUST quote the exact phrase from the regulatory text that supports your classification.

Respond in JSON format:
{{
    "status": "REQUIRED|EXTRA|MISSING|OPTIONAL",
    "confidence": 0.0-1.0,
    "explanation": "Brief explanation with direct quote from regulatory text",
    "citation": "Specific regulation article/section (CURRENT law only)",
    "evidence_quote": "Exact phrase from the regulatory text (≤50 words)",
    "semantic_match": true/false,
    "rejection_code": "ERR_REVOKED_LAW|ERR_MISSING_SIGNATURE|ERR_MISSING_DOC|ERR_EXTRA_DOC|null"
}}"""

    settings = get_settings()
    client = _get_client()

    response = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response for classification")

    return json.loads(content)


# Alias for CrewAI DVA agent tool (PRD C1)
classify_item = classify_with_llm


@_retry_on_rate_limit
def compare_clause_to_standard(
    clause_text: str,
    standard_reference: str,
    country: str,
    regulation_snippets: str,
) -> dict:
    """
    Compare a document clause to regulatory standard. Returns severity, citation, suggestion.
    Used by CRA agent. No hardcoded classifications; grounded in regulation_snippets.
    """
    import re as _re

    system_prompt = (
        "You are a medical device compliance auditor. Compare the document clause to the "
        "regulatory snippets. Output JSON only — no markdown fences, no extra text. "
        "Never fabricate citations; use only text from the provided regulation snippets. "
        "You MUST quote the exact regulatory snippet text that supports your finding. "
        "If no snippet supports a finding, do not report it. Zero inference — facts only. "
        "Severity: CRITICAL (non-compliance risk), MAJOR (significant gap), "
        "MINOR (improvement), INFO (note only)."
    )
    user_prompt = f"""Document clause to review:
{clause_text[:2000]}

Standard: {standard_reference}
Country: {country}

Regulatory snippets (use for citation):
{regulation_snippets[:4000]}

Respond in JSON:
{{
  "clause": "brief excerpt of clause",
  "severity": "CRITICAL|MAJOR|MINOR|INFO",
  "citation": "exact reference from snippets",
  "suggestion": "remediation if needed",
  "standard_reference": "{standard_reference}",
  "confidence": 0.0 to 1.0
}}"""

    settings = get_settings()
    client = _get_client()

    # Use response_format=json_object to guarantee valid JSON from OpenAI
    response = client.chat.completions.create(
        model=settings.openai_generation_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM returned empty response for clause comparison")

    # Strip markdown fences if present (defensive)
    text = content.strip()
    if "```" in text:
        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    return json.loads(text)
