"""
OrthoLink RAA Crew — Monitor Regulations
POST /api/v1/alerts/check-changes (called via alert route)
Scheduled: can be run via cron / background task.

PRD §4.6: Sequential CrewAI crew.
  1. Scraper Agent  — fetches regulatory source URL, computes content hash
  2. Diff Agent     — compares hash to stored; if changed, extracts delta text
  3. Summarizer     — produces ChangeSummary (gpt-4o); applies confidence gate ≥ 0.7
  4. Notifier       — soft-deactivates old chunks, re-embeds new, fires AlertEvent

All re-embedding uses text-embedding-3-large (HC-1).
Never deletes chunks — sets is_active=False + valid_to (HC-6).
"""

import logging
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from pydantic import BaseModel

from app.agents.raa_agent import DISCLAIMER
from app.tools.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tools exposed to CrewAI agents
# ─────────────────────────────────────────────────────────────────────────────

@tool
def fetch_and_hash_url(url: str) -> str:
    """
    Fetch regulatory source URL and return JSON with {text, sha256_hash, url}.
    Used by the Scraper agent to detect content changes.
    """
    import json
    from app.tools.alert_tools import scrape_and_hash
    try:
        result = scrape_and_hash(url)
        return json.dumps(result)
    except Exception as e:
        logger.warning("CRITICAL: fetch_and_hash_url failed for %s: %s", url, e)
        return json.dumps({"error": str(e), "url": url, "text": "", "sha256_hash": ""})


@tool
def get_stored_hash(country: str, document_id: str) -> str:
    """
    Return the stored sha256 hash for a given country + document_id from the vector store.
    Returns 'NOT_FOUND' if no chunks exist for this document.
    """
    store = get_vector_store()
    chunks = list(store.search(document_id, country=country, top_k=1))
    if not chunks:
        return "NOT_FOUND"
    return chunks[0].get("chunk_hash", "NOT_FOUND")


@tool
def soft_deactivate_chunks(country: str, document_id: str) -> str:
    """
    Soft-deactivate all active chunks for country+document_id.
    Sets is_active=False and valid_to=today. Never deletes. Returns count deactivated.
    """
    # Import here to avoid circular imports at module level
    from app.tools.alert_tools import soft_deactivate_chunks as _do_deactivate
    from app.ingestion.monitored_docs import get_monitored_doc

    store = get_vector_store()
    try:
        doc = get_monitored_doc(country, document_id) or {}
        source_url = doc.get("source_url", "")
        chunks = store.get_chunks_by_document(
            country=country,
            document_id=document_id,
            source_url=source_url,
            active_only=True,
        )
        chunk_ids = [c["chunk_id"] for c in chunks if c.get("chunk_id")]
        if not chunk_ids:
            return f"No active chunks found for {country}/{document_id} — nothing to deactivate."
        _do_deactivate(store, chunk_ids)
        return f"Soft-deactivated {len(chunk_ids)} chunks for {country}/{document_id}"
    except Exception as e:
        logger.warning(f"soft_deactivate_chunks failed: {e}")
        return f"Deactivation failed: {e}"


@tool
def reembed_and_store(country: str, document_id: str, regulation_name: str, text: str) -> str:
    """
    Chunk and re-embed new regulatory text using text-embedding-3-large.
    Returns count of new chunks stored.
    """
    from app.ingestion.chunker import chunk_regulatory_text
    from app.ingestion.embedder import embed_and_index_chunks

    try:
        chunks = chunk_regulatory_text(
            text=text,
            regulation_name=regulation_name,
            country=country,
            document_id=document_id,
        )
        embed_and_index_chunks(chunks)
        return f"Stored {len(chunks)} new chunks for {country}/{document_id}"
    except Exception as e:
        logger.warning(f"reembed_and_store failed: {e}")
        return f"Re-embedding failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Crew builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_monitor_crew(country: str, document_id: str, source_url: str) -> Crew:
    """Build a sequential CrewAI Crew for monitoring one regulatory document."""

    scraper_agent = Agent(
        role="Regulatory Scraper",
        goal=f"Fetch the regulatory document at {source_url} and report its hash.",
        backstory=(
            "You fetch government regulatory URLs and compute SHA-256 content hashes "
            "to detect when documents change."
        ),
        tools=[fetch_and_hash_url, get_stored_hash],
        llm="gpt-4o",
        max_iter=5,
        verbose=True,
    )

    diff_agent = Agent(
        role="Regulatory Diff Analyst",
        goal="Compare stored and fetched hashes; extract changed text if different.",
        backstory=(
            "You compare document versions and identify what text changed. "
            "If hashes match, report NO_CHANGE. If they differ, summarise the delta."
        ),
        tools=[get_stored_hash],
        llm="gpt-4o",
        max_iter=5,
        verbose=True,
    )

    summarizer_agent = Agent(
        role="Change Summarizer",
        goal="Write a concise, accurate change summary. Apply confidence gate ≥ 0.7.",
        backstory=(
            "You summarize regulatory changes for medical device compliance teams. "
            "Output JSON: {\"summary\": \"...\", \"confidence\": 0.85}. "
            "If confidence < 0.7, set summary to 'NEEDS_HUMAN_REVIEW'."
        ),
        tools=[],
        llm="gpt-4o",
        max_iter=5,
        verbose=True,
    )

    notifier_agent = Agent(
        role="Regulatory Notifier",
        goal=(
            f"Soft-deactivate old chunks and re-embed new text for "
            f"{country}/{document_id}. Return alert details."
        ),
        backstory=(
            "You update the vector store: old chunks are soft-deactivated (is_active=False), "
            "new text is re-embedded with text-embedding-3-large. You never delete chunks."
        ),
        tools=[soft_deactivate_chunks, reembed_and_store],
        llm="gpt-4o",
        max_iter=5,
        verbose=True,
    )

    task_scrape = Task(
        description=(
            f"Fetch the regulatory URL: {source_url}\n"
            f"Also retrieve the stored hash for country={country}, document_id={document_id}.\n"
            "Report: {\"fetched_hash\": \"...\", \"stored_hash\": \"...\", \"text\": \"...\"}"
        ),
        expected_output="JSON with fetched_hash, stored_hash, and text fields.",
        agent=scraper_agent,
    )

    task_diff = Task(
        description=(
            "Compare the fetched_hash to the stored_hash from the previous task.\n"
            "If they match: output {\"changed\": false}.\n"
            "If they differ: output {\"changed\": true, \"old_text\": \"...\", \"new_text\": \"...\"}."
        ),
        expected_output="JSON with changed boolean and optional text fields.",
        agent=diff_agent,
        context=[task_scrape],
    )

    task_summarize = Task(
        description=(
            "If changed=false from the previous task, output {\"summary\": \"NO_CHANGE\", \"confidence\": 1.0}.\n"
            "If changed=true, summarize what changed in 1-2 sentences.\n"
            "Apply confidence gate: if confidence < 0.7, set summary='NEEDS_HUMAN_REVIEW'.\n"
            "Output JSON: {\"summary\": \"...\", \"confidence\": 0.85}"
        ),
        expected_output="JSON with summary and confidence fields.",
        agent=summarizer_agent,
        context=[task_diff],
    )

    task_notify = Task(
        description=(
            f"If summary=='NO_CHANGE': do nothing, output {{\"alerts_emitted\": 0}}.\n"
            f"Otherwise:\n"
            f"1. Call soft_deactivate_chunks(country='{country}', document_id='{document_id}')\n"
            f"2. Call reembed_and_store(country='{country}', document_id='{document_id}', "
            f"   regulation_name='{document_id}', text=<new_text from diff task>)\n"
            f"3. Output {{\"alerts_emitted\": 1, \"summary\": \"...\", \"confidence\": ...}}"
        ),
        expected_output="JSON with alerts_emitted, summary, confidence.",
        agent=notifier_agent,
        context=[task_scrape, task_diff, task_summarize],
    )

    return Crew(
        agents=[scraper_agent, diff_agent, summarizer_agent, notifier_agent],
        tasks=[task_scrape, task_diff, task_summarize, task_notify],
        process=Process.sequential,
        verbose=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

class MonitorResult(BaseModel):
    """Result from running the monitor crew for one document."""

    country: str
    document_id: str
    changed: bool = False
    alerts_emitted: int = 0
    change_summary: str = ""
    confidence: float = 0.0
    status: str = "completed"
    disclaimer: str = DISCLAIMER


def run_monitor_crew(
    country: str,
    document_id: str,
    source_url: str,
    regulation_name: Optional[str] = None,
) -> MonitorResult:
    """
    Run the RAA sequential monitoring crew for one regulatory document.

    Args:
        country: ISO country code (e.g. "US", "UA")
        document_id: Unique identifier for this regulatory document
        source_url: URL to fetch the current regulatory text from
        regulation_name: Human-readable name of the regulation

    Returns:
        MonitorResult with change detection, summary, and confidence.
    """
    import json

    if not source_url:
        return MonitorResult(
            country=country,
            document_id=document_id,
            status="skipped",
            change_summary="No source URL provided.",
        )

    try:
        crew = _build_monitor_crew(country, document_id, source_url)
        crew_output = crew.kickoff(inputs={
            "country": country,
            "document_id": document_id,
            "source_url": source_url,
            "regulation_name": regulation_name or document_id,
        })

        # Extract final task output (notifier task)
        tasks_output = getattr(crew_output, "tasks_output", None) or []
        raw = ""
        if tasks_output:
            raw = getattr(tasks_output[-1], "raw", "") or str(tasks_output[-1])

        # Parse JSON result
        changed = False
        alerts_emitted = 0
        summary = ""
        confidence = 0.0

        if raw:
            try:
                from app.crews.utils import extract_clean_json
                clean = extract_clean_json(raw)
                data = json.loads(clean)
                alerts_emitted = int(data.get("alerts_emitted", 0))
                summary = data.get("summary", "")
                confidence = float(data.get("confidence", 0.0))
                changed = alerts_emitted > 0 or (summary and summary not in ("NO_CHANGE", ""))
            except Exception as pe:
                logger.warning(f"monitor crew output parse error: {pe}, raw={raw[:200]}")

        return MonitorResult(
            country=country,
            document_id=document_id,
            changed=changed,
            alerts_emitted=alerts_emitted,
            change_summary=summary,
            confidence=confidence,
            status="completed",
        )

    except Exception as e:
        logger.error(f"monitor_regulations crew failed: {e}", exc_info=True)
        return MonitorResult(
            country=country,
            document_id=document_id,
            status="failed",
            change_summary=f"Crew execution failed: {e}",
        )
