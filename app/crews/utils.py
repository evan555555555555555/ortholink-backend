"""
OrthoLink Crews — Shared Utilities
Eliminates duplicate code across crew modules (parse_llm_json appeared verbatim in
pms_plan, capa_analysis, risk_analysis, and technical_dossier).

Public API:
  extract_clean_json(text)             Universal Extraction Manifold — strip ALL noise
  parse_llm_json(raw, key=None)        Strip markdown fences, parse JSON dict/list
  multi_query_faiss(store, queries, …) Run N queries, deduplicate, return chunk strings
  build_regulation_context(chunks, …)  Join chunks into regulation context for prompts
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Universal Extraction Manifold ──────────────────────────────────────
# Strips markdown formatting, conversational preambles/postscripts, and
# any non-JSON noise from LLM outputs. Returns a clean JSON string ready
# for Pydantic validation or json.loads().


def extract_clean_json(text: str) -> str:
    """
    Universal Extraction Manifold — isolate pure JSON from LLM output.

    Strategy (ordered by specificity):
      1. Extract from markdown code fences (```json ... ```)
      2. Strip conversational preamble/postscript around JSON
      3. Find first { or [ and match to closing } or ]
      4. Validate the extracted string is parseable JSON
      5. Raise ValueError if nothing parseable found

    Returns:
        Clean JSON string (validated parseable).

    Raises:
        ValueError: If no valid JSON can be extracted.
    """
    if not text or not text.strip():
        raise ValueError("Extraction Manifold: empty input")

    cleaned = text.strip()

    # Strategy 1: Markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # Fence content wasn't valid JSON — continue

    # Strategy 2: Strip common conversational noise patterns
    # LLMs often prepend "Here is the JSON:" or append "Let me know if..."
    noise_patterns = [
        r"^(?:Here\s+(?:is|are)\s+.*?:)\s*",        # "Here is the JSON:"
        r"^(?:Sure[,!.]?\s+.*?:)\s*",                 # "Sure, here you go:"
        r"^(?:The\s+(?:result|output|response).*?:)\s*",  # "The result is:"
        r"\s*(?:Let me know|Feel free|Is there|Hope this).*$",  # trailing chat
    ]
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    # Strategy 3: Direct parse (after noise removal)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Strategy 4: Find outermost JSON object { ... }
    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        candidate = obj_match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Strategy 5: Find outermost JSON array [ ... ]
    arr_match = re.search(r"\[[\s\S]*\]", cleaned)
    if arr_match:
        candidate = arr_match.group()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    logger.error(
        "CRITICAL: Extraction Manifold failure — no valid JSON in %d chars: %.200s",
        len(text), text,
    )
    raise ValueError(
        f"Extraction Manifold: no valid JSON found in LLM output ({len(text)} chars)"
    )


def parse_llm_json(raw: str, key: str | None = None) -> dict | list:
    """
    Extract and parse JSON from raw LLM output, stripping markdown code fences.

    Uses extract_clean_json() for robust extraction, then optionally
    extracts a sub-key from the parsed dict.

    Returns:
      dict or list; {} on any failure (never raises).
    """
    if not raw:
        return {}

    try:
        clean = extract_clean_json(raw)
        data = json.loads(clean)
        if isinstance(data, (dict, list)):
            if key is not None and isinstance(data, dict):
                return data.get(key, {})
            return data
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("parse_llm_json: extraction failed: %s (input len=%d)", exc, len(raw))

    return {}


def multi_query_faiss(
    store,
    queries: list[str],
    country: str,
    device_class: Optional[str] = None,
    top_k: int = 5,
    max_chunks: int = 50,
) -> list[str]:
    """
    Run multiple FAISS queries and return deduplicated, formatted chunk strings.

    Deduplication is by chunk_id (falls back to first 40 chars of text).
    Each chunk is formatted as "[section_path_or_doc_id] text[:400]".

    Args:
        store:        VectorStore instance (from get_vector_store())
        queries:      Search query strings
        country:      ISO country code for filtering ("US", "EU", "JP", …)
        device_class: Optional device class filter ("I", "IIa", "IIb", "III")
        top_k:        Results retrieved per query
        max_chunks:   Hard cap on total returned chunks (stops early when reached)

    Returns:
        Deduplicated list of "[source] text" strings, capped at max_chunks.
    """
    seen: set[str] = set()
    chunks: list[str] = []

    for q in queries:
        if len(chunks) >= max_chunks:
            break
        try:
            results = store.search(
                q, country=country, device_class=device_class or None, top_k=top_k
            )
        except Exception as exc:
            logger.debug("FAISS query failed for '%s': %s", q, exc)
            continue

        for r in results:
            cid = r.get("chunk_id") or r.get("text", "")[:40]
            if cid in seen:
                continue
            seen.add(cid)
            src = r.get("section_path") or r.get("document_id") or ""
            chunks.append(f"[{src}] {r.get('text', '')[:400]}")
            if len(chunks) >= max_chunks:
                break

    return chunks


def multi_query_faiss_raw(
    store,
    queries: list[str],
    country: str,
    device_class: Optional[str] = None,
    top_k: int = 5,
    max_chunks: int = 50,
) -> list[dict]:
    """
    Like multi_query_faiss, but returns raw dicts with score/text/document_id/etc.
    Used by verify_claims which needs scores for confidence calculation.
    """
    seen: set[str] = set()
    chunks: list[dict] = []

    for q in queries:
        if len(chunks) >= max_chunks:
            break
        try:
            results = store.search(
                q, country=country, device_class=device_class or None, top_k=top_k
            )
        except Exception as exc:
            logger.debug("FAISS query failed for '%s': %s", q, exc)
            continue

        for r in results:
            cid = r.get("chunk_id") or r.get("text", "")[:40]
            if cid in seen:
                continue
            seen.add(cid)
            chunks.append(r)
            if len(chunks) >= max_chunks:
                break

    return chunks


def build_regulation_context(chunks: list[str], max_chunks: int = 40) -> str:
    """
    Join chunk strings into a regulation context block for LLM prompts.

    Args:
        chunks:     List of "[source] text" strings from multi_query_faiss
        max_chunks: Limit on chunks to include

    Returns:
        Double-newline separated string for insertion into LLM prompts.
    """
    return "\n\n".join(chunks[:max_chunks])
