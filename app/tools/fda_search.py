"""
OrthoLink FDA 510(k) Predicate Heuristic Engine.

Flow: (1) Query the public OpenFDA API (device/510k) by product_code to get candidate
510(k) devices. (2) For each candidate, take the indications_for_use text and embed it
with text-embedding-3-large. (3) Embed the user’s device profile (device_name + intended_use)
with the same model. (4) Rank candidates by cosine similarity between profile and
indications_for_use embeddings and return the top 3 (k_number, applicant, decision_date,
similarity). This gives a repeatable, citation-ready way to surface predicate devices
by intended use for US strategy and CRA/RSA context, without ad-hoc rules.
"""

import asyncio
import json
import logging

import numpy as np
import requests
from crewai.tools import tool

from app.tools.embeddings import embed_text

logger = logging.getLogger(__name__)

OPENFDA_510K_URL = "https://api.fda.gov/device/510k.json"
DEFAULT_LIMIT = 50
TOP_N = 3


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumed L2-normalized for dot = cosine)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = float(np.dot(a_flat, b_flat))
    return max(0.0, min(1.0, dot))


def _fetch_510k_sync(product_code: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Query openFDA 510(k) API by product_code (sync for use in CrewAI tool from thread)."""
    search = f"product_code:{product_code}" if product_code else ""
    params = {"search": search, "limit": limit} if search else {"limit": limit}
    try:
        resp = requests.get(OPENFDA_510K_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results") or []
    except (requests.RequestException, Exception) as e:
        logger.warning("openFDA 510(k) request failed: %s", e)
        return []


async def _fetch_510k_async(product_code: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Async wrapper: run sync fetch in thread."""
    return await asyncio.to_thread(_fetch_510k_sync, product_code, limit)


def _indications_text(record: dict) -> str:
    """Extract indications-for-use text from a 510(k) record for embedding."""
    # openFDA fields may be nested; common names
    for key in ("indications_for_use", "statement_of_indications_for_use", "indications_for_use_label"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val:
            return " ".join(str(x) for x in val).strip()
    device_name = record.get("device_name") or record.get("device_name_original") or ""
    return device_name


def _run_embed_and_rank_sync(
    records: list[dict],
    device_profile_text: str,
) -> list[tuple[dict, float]]:
    """Synchronous: embed profile and each indications text, rank by cosine. Run in thread."""
    if not device_profile_text.strip() or not records:
        return []
    profile_emb = embed_text(device_profile_text[:4000])
    if profile_emb.size == 0:
        return []
    profile_flat = profile_emb.flatten()
    norm = np.linalg.norm(profile_flat)
    if norm > 0:
        profile_flat = profile_flat / norm
    scored: list[tuple[dict, float]] = []
    for rec in records:
        text = _indications_text(rec)
        if not text:
            continue
        try:
            ind_emb = embed_text(text[:4000])
            if ind_emb.size == 0:
                continue
            ind_flat = ind_emb.flatten()
            n = np.linalg.norm(ind_flat)
            if n > 0:
                ind_flat = ind_flat / n
            sim = _cosine_similarity(profile_flat, ind_flat)
            scored.append((rec, float(sim)))
        except Exception as e:
            logger.debug("Embed/score skip record: %s", e)
            continue
    scored.sort(key=lambda x: -x[1])
    return scored[:TOP_N]


def search_fda_predicates_sync(
    product_code: str,
    device_name: str = "",
    intended_use: str = "",
) -> list[dict]:
    """
    Sync: query openFDA 510(k) by product_code; rank by semantic similarity of indications_for_use
    to (device_name + intended_use). Returns top 3 matches. Safe to call from CrewAI tool (thread).
    HC-1: text-embedding-3-large for embeddings.
    """
    profile = f"{device_name}\n{intended_use}".strip() or "medical device"
    records = _fetch_510k_sync(product_code or "UNKNOWN", limit=DEFAULT_LIMIT)
    if not records:
        return []
    scored = _run_embed_and_rank_sync(records, profile)
    out: list[dict] = []
    for rec, sim in scored:
        out.append({
            "k_number": rec.get("k_number") or rec.get("k_numbers") or "",
            "applicant": rec.get("applicant") or rec.get("applicant_original") or "",
            "decision_date": rec.get("decision_date") or rec.get("decision_date_original") or "",
            "device_name": rec.get("device_name") or rec.get("device_name_original") or "",
            "indications_for_use": _indications_text(rec)[:500],
            "similarity": round(sim, 4),
        })
    return out


async def search_fda_predicates(
    product_code: str,
    device_name: str = "",
    intended_use: str = "",
) -> list[dict]:
    """Async: run sync predicate search in thread to avoid blocking event loop."""
    return await asyncio.to_thread(
        search_fda_predicates_sync,
        product_code,
        device_name,
        intended_use,
    )


@tool
def search_fda_predicates_tool(product_code: str, device_name: str = "", intended_use: str = "") -> str:
    """
    Search FDA 510(k) database by product_code. Returns top 3 predicate devices ranked by
    semantic similarity of indications_for_use to the given device_name and intended_use.
    Use for US market strategy and predicate cross-reference. Returns JSON array with
    k_number, applicant, decision_date, similarity.
    """
    results = search_fda_predicates_sync(product_code, device_name, intended_use)
    return json.dumps(results)
