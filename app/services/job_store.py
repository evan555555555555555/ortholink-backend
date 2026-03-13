"""
In-memory job store for async crew execution.

# TODO M3: Replace with Redis-backed store (aioredis).
# In-memory store loses jobs on restart. Acceptable for M1 single-instance
# deployment; not acceptable at scale.

Long-running crew runs (DVA, RSA) must not block the FastAPI event loop.
Jobs are enqueued with a job_id; background task runs crew and stores result.
GET /api/v1/jobs/{job_id} returns status (pending | running | completed | failed | NOT_FOUND) and result.
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

# Jobs older than this are evicted on the next create_job() call.
_TTL = timedelta(hours=1)

# Legal disclaimer injected into every AI agent result (PRD §7 — regulatory accuracy)
_DISCLAIMER = (
    "OrthoLink provides regulatory intelligence based on indexed official sources. "
    "Information is current as of the last ingestion date. "
    "This is not legal advice — always verify with your regulatory counsel "
    "and the relevant competent authority before submission."
)

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    job_id: str
    status: str  # pending | running | completed | failed
    agent: str
    created_at: str
    updated_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


_store: dict[str, JobRecord] = {}
_lock = threading.Lock()


def _evict_expired() -> None:
    """Remove terminal jobs (completed/failed) older than _TTL. Caller must hold _lock."""
    cutoff = (datetime.now(timezone.utc) - _TTL).isoformat()
    expired = [
        jid for jid, rec in _store.items()
        if rec.status in ("completed", "failed") and rec.updated_at < cutoff
    ]
    for jid in expired:
        del _store[jid]
    if expired:
        logger.debug("job_store: evicted %d expired jobs", len(expired))


def create_job(agent: str) -> str:
    """Create a new job; returns job_id. Evicts expired jobs first."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        _evict_expired()
        _store[job_id] = JobRecord(
            job_id=job_id,
            status="pending",
            agent=agent,
            created_at=now,
            updated_at=now,
        )
    return job_id


def set_running(job_id: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id].status = "running"
            _store[job_id].updated_at = datetime.now(timezone.utc).isoformat()


def set_completed(job_id: str, result: dict) -> None:
    # Inject legal disclaimer into every AI agent result
    result["_disclaimer"] = _DISCLAIMER

    # Cryptographically sign the payload (fast — pure hashing, no I/O)
    try:
        from app.services.crypto_signer import sign_payload
        result = sign_payload(result)
    except Exception:
        pass  # Never let signing break job completion

    with _lock:
        if job_id in _store:
            _store[job_id].status = "completed"
            _store[job_id].result = result
            _store[job_id].updated_at = datetime.now(timezone.utc).isoformat()

    # Fire background integrity check (non-blocking daemon thread)
    _schedule_integrity_check(job_id, result)


def _schedule_integrity_check(job_id: str, result: dict) -> None:
    """
    Spawn a daemon thread to auto-verify the job result against FAISS.
    Injects `_integrity` report into the job result when complete (~1-3s).
    Never blocks the calling thread or the FastAPI event loop.
    """
    def _run():
        try:
            from app.services.integrity_guard import auto_verify_result
            country = result.get("country", "")
            device_class = result.get("device_class", "")
            report = auto_verify_result(result, country=country, device_class=device_class)
            if report:
                with _lock:
                    if job_id in _store and _store[job_id].result is not None:
                        _store[job_id].result["_integrity"] = report
                        logger.debug(
                            "IntegrityGuard: job %s → %s (%s/%s verified)",
                            job_id[:8],
                            report.get("overall_verdict"),
                            report.get("verified", 0),
                            report.get("claims_checked", 0),
                        )
        except Exception as e:
            logger.debug("IntegrityGuard background check %s: %s", job_id[:8], e)

    t = threading.Thread(target=_run, daemon=True, name=f"integrity-{job_id[:8]}")
    t.start()


def set_failed(job_id: str, error: str) -> None:
    with _lock:
        if job_id in _store:
            _store[job_id].status = "failed"
            _store[job_id].error = error
            _store[job_id].updated_at = datetime.now(timezone.utc).isoformat()


def get_job(job_id: str) -> Optional[JobRecord]:
    with _lock:
        return _store.get(job_id)


def get_job_response(job_id: str) -> Optional[dict]:
    """Return dict suitable for API response."""
    rec = get_job(job_id)
    if not rec:
        return None
    out = {
        "job_id": rec.job_id,
        "status": rec.status,
        "agent": rec.agent,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
    }
    if rec.result is not None:
        out["result"] = rec.result
    if rec.error is not None:
        out["error"] = rec.error
    return out


def get_all_recent_jobs(limit: int = 50, agent: Optional[str] = None) -> list[dict]:
    """Return most recent completed jobs, newest first. Optionally filter by agent."""
    with _lock:
        records = list(_store.values())
    completed = [
        r for r in records
        if r.status == "completed" and (agent is None or r.agent == agent)
    ]
    completed.sort(key=lambda r: r.updated_at, reverse=True)
    return [
        {
            "job_id": r.job_id,
            "agent": r.agent,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "result": r.result or {},
        }
        for r in completed[:limit]
    ]


def get_latest_job(agent: str) -> Optional[dict]:
    """Return the most recently completed job for a given agent type."""
    jobs = get_all_recent_jobs(limit=1, agent=agent)
    return jobs[0] if jobs else None
