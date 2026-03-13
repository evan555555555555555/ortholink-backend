"""
OrthoLink Omni-Strike Stress Test
==================================
Fires all 12 agents concurrently using asyncio + aiohttp.
Validates HTTP status codes, response structure, defensive architecture
invariants, and finishes with a dashboard health check.

Usage:
    .venv/bin/python3 omni_stress_test.py [--base-url URL] [--token JWT]

If --token is omitted, reads VITE_AUTH_TOKEN from frontend-vite/.env.
Requires: pip install aiohttp
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError:
    sys.exit("aiohttp not installed. Run: .venv/bin/pip install aiohttp")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE = "http://localhost:8001"
API_PREFIX = "/api/v1"
JOB_POLL_INTERVAL = 2.0
JOB_POLL_TIMEOUT = 200.0

# ---------------------------------------------------------------------------
# Test payloads — Class III Titanium Spinal Fixation System
# ---------------------------------------------------------------------------
DEVICE = "Spinal Fixation System (Ti-6Al-4V ELI)"
DEVICE_USE = "Posterior spinal fusion and stabilization in skeletally mature patients"
COUNTRY = "US"
DEVICE_CLASS = "III"

# Minimal CSV for DVA (distributor gap analysis)
DVA_CSV = (
    "Item,Status\n"
    "510(k) Clearance Letter,Present\n"
    "Quality Management System Certificate,Present\n"
    "Labeling Compliance Declaration,Present\n"
    "UDI Registration Confirmation,Missing\n"
    "Establishment Registration,Present\n"
)

# Minimal text document for CRA (compliance review)
CRA_DOC = (
    "QUALITY MANUAL — Spinal Fixation System\n"
    "Section 4.2: Quality Management System\n"
    "The organization maintains a QMS in compliance with 21 CFR 820.\n"
    "Design controls per 21 CFR 820.30 are applied to all Class III devices.\n"
    "Post-market surveillance is conducted per 21 CFR 803 (MDR).\n"
    "Risk management follows ISO 14971:2019 throughout the product lifecycle.\n"
)

# Claims for Verify agent
VERIFY_CLAIMS = (
    "21 CFR 820.30 requires design controls for Class III devices\n"
    "ISO 14971:2019 mandates risk management throughout product lifecycle\n"
    "510(k) clearance is sufficient for Class III spinal implants\n"
    "FDA requires UDI labeling for all medical devices"
)

# Problem statement for CAPA
CAPA_PROBLEM = (
    "Three field complaints received: titanium pedicle screw fracture "
    "at 18 months post-implantation in patients with BMI > 35. "
    "Metallurgical analysis indicates fatigue failure at thread root."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_token_from_env() -> str | None:
    """Try to read VITE_AUTH_TOKEN from frontend .env."""
    candidates = [
        Path(__file__).resolve().parent.parent / "frontend-vite" / ".env",
        Path(__file__).resolve().parent / ".." / "frontend-vite" / ".env",
    ]
    for p in candidates:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("VITE_AUTH_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Agent strike functions
# ---------------------------------------------------------------------------
async def _poll_job(
    session: aiohttp.ClientSession,
    base: str,
    job_id: str,
    headers: dict,
    timeout: float = JOB_POLL_TIMEOUT,
) -> dict[str, Any]:
    """Poll GET /jobs/{job_id} until completed or timeout."""
    url = f"{base}{API_PREFIX}/jobs/{job_id}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with session.get(url, headers=headers) as r:
            if r.status != 200:
                await asyncio.sleep(JOB_POLL_INTERVAL)
                continue
            data = await r.json()
            status = data.get("status", "")
            if status in ("completed", "failed"):
                return data
        await asyncio.sleep(JOB_POLL_INTERVAL)
    return {"status": "timeout", "error": f"Job {job_id} did not complete within {timeout}s"}


async def strike_roa(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """ROA — generate-checklist (sync, form-data)."""
    url = f"{base}{API_PREFIX}/generate-checklist"
    form = aiohttp.FormData()
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("device_type", "orthopedic_implant")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "ROA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_rma(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """RMA — risk-analysis (sync mode)."""
    url = f"{base}{API_PREFIX}/risk-analysis"
    form = aiohttp.FormData()
    form.add_field("device_description", DEVICE)
    form.add_field("intended_use", DEVICE_USE)
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "RMA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_capa(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """CAPA — capa analysis (sync mode)."""
    url = f"{base}{API_PREFIX}/capa"
    form = aiohttp.FormData()
    form.add_field("problem_statement", CAPA_PROBLEM)
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "CAPA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_pms(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """PMS — pms-plan (sync mode)."""
    url = f"{base}{API_PREFIX}/pms-plan"
    form = aiohttp.FormData()
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("device_type", "orthopedic_implant")
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "PMS", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_tda(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """TDA — technical-dossier (sync mode)."""
    url = f"{base}{API_PREFIX}/technical-dossier"
    form = aiohttp.FormData()
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("device_type", "orthopedic_implant")
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "TDA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_swarm(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """Swarm — swarm-analysis (sync mode, with CAPA trigger)."""
    url = f"{base}{API_PREFIX}/swarm-analysis"
    form = aiohttp.FormData()
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("device_type", "orthopedic_implant")
    form.add_field("problem_statement", CAPA_PROBLEM)
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "SWARM", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_rsa(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """RSA — plan-strategy (sync mode)."""
    url = f"{base}{API_PREFIX}/plan-strategy"
    form = aiohttp.FormData()
    form.add_field("device_name", DEVICE)
    form.add_field("target_markets", "US,EU,JP")
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "RSA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_verify(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """Verify — verify-claims (sync, FAISS-only, no LLM)."""
    url = f"{base}{API_PREFIX}/verify-claims"
    form = aiohttp.FormData()
    form.add_field("claims_text", VERIFY_CLAIMS)
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "VERIFY", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_briefing(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """Briefing — briefing/run (sync, no LLM)."""
    url = f"{base}{API_PREFIX}/briefing/run"
    t0 = time.monotonic()
    async with session.post(url, headers=headers) as r:
        body = await r.json()
        return {"agent": "BRIEFING", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_raa(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """RAA — FDA recalls live feed (GET, no LLM)."""
    url = f"{base}{API_PREFIX}/alerts/live/fda-recalls"
    params = {"device_name": "spinal", "limit": "5", "days_back": "180"}
    t0 = time.monotonic()
    async with session.get(url, headers=headers, params=params) as r:
        body = await r.json()
        return {"agent": "RAA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_dva(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """DVA — verify-distributor (file upload, sync mode)."""
    url = f"{base}{API_PREFIX}/verify-distributor"
    form = aiohttp.FormData()
    form.add_field(
        "file",
        DVA_CSV.encode(),
        filename="submission_checklist.csv",
        content_type="text/csv",
    )
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    form.add_field("async_mode", "false")
    t0 = time.monotonic()
    async with session.post(url, headers=headers, data=form) as r:
        body = await r.json()
        return {"agent": "DVA", "status": r.status, "ms": _ms(t0), "body": body}


async def strike_cra(session: aiohttp.ClientSession, base: str, headers: dict) -> dict:
    """CRA — review-document (file upload, SSE stream)."""
    url = f"{base}{API_PREFIX}/review-document"
    form = aiohttp.FormData()
    form.add_field(
        "file",
        CRA_DOC.encode(),
        filename="quality_manual.txt",
        content_type="text/plain",
    )
    form.add_field("standard", "FDA 21 CFR 820")
    form.add_field("country", COUNTRY)
    form.add_field("device_class", DEVICE_CLASS)
    t0 = time.monotonic()
    events: list[dict] = []
    try:
        async with session.post(url, headers=headers, data=form) as r:
            if r.status != 200:
                text = await r.text()
                return {"agent": "CRA", "status": r.status, "ms": _ms(t0), "body": {"error": text}}
            async for line in r.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded.startswith("data:"):
                    payload = decoded[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
    except Exception as exc:
        return {"agent": "CRA", "status": "ERROR", "ms": _ms(t0), "body": {"error": str(exc)}}
    return {
        "agent": "CRA",
        "status": 200,
        "ms": _ms(t0),
        "body": {"sse_events": len(events), "events": events},
    }


def _ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def _validate_verify(body: dict) -> list[str]:
    """Check verify-claims for defensive architecture invariants."""
    issues: list[str] = []
    result = body.get("result", body)
    verifications = result.get("verifications", [])

    for v in verifications:
        claim = v.get("claim", "")
        verdict = v.get("verdict", "")
        # 510(k) for Class III should NOT be VERIFIED — it's PMA territory
        if "510(k)" in claim and "Class III" in claim and verdict == "VERIFIED":
            issues.append(
                f"CRITICAL: '510(k) for Class III' was VERIFIED — should be "
                f"UNVERIFIED or CONTRADICTED (PMA required for Class III)"
            )
    return issues


def _validate_dashboard(data: dict) -> list[str]:
    """Validate dashboard defensive architecture flags."""
    issues: list[str] = []
    arch = data.get("defensive_architecture", {})
    for flag in ("country_isolation", "revoked_law_filter", "static_fallback_floor",
                 "cross_country_contamination_blocked"):
        if arch.get(flag) is not True:
            issues.append(f"DEFENSIVE FLAG '{flag}' is not True")
    if arch.get("cache_key_format") != "faiss:v3:{COUNTRY}:{SHA256}":
        issues.append("Cache key format is not v3")
    return issues


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def run_omni_strike(base_url: str, token: str):
    print(f"\n{'='*64}")
    print(f"  ORTHOLINK OMNI-STRIKE STRESS TEST")
    print(f"  Target: {base_url}")
    print(f"  Device: {DEVICE}")
    print(f"  Market: {COUNTRY} | Class: {DEVICE_CLASS}")
    print(f"  Time:   {_ts()}")
    print(f"{'='*64}\n")

    headers = _hdr(token)
    connector = aiohttp.TCPConnector(limit=6, limit_per_host=4)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        # ----- Pre-flight: Dashboard health check -----
        print("[PRE-FLIGHT] Checking system dashboard...")
        dash_url = f"{base_url}{API_PREFIX}/dashboard/system-status"
        try:
            async with session.get(dash_url, headers=headers) as r:
                if r.status == 200:
                    dash = await r.json()
                    health = dash.get("global_health")
                    verdict = dash.get("global_verdict")
                    print(f"  Global Health : {'NOMINAL' if health else 'DEGRADED'}")
                    print(f"  Verdict       : {verdict}")
                    cache_status = dash.get("components", {}).get("redis_v3_cache", {}).get("status")
                    faiss_status = dash.get("components", {}).get("faiss_vector_store", {}).get("status")
                    hmac_status = dash.get("components", {}).get("hmac_integrity_engine", {}).get("status")
                    print(f"  Redis Cache   : {cache_status}")
                    print(f"  FAISS Store   : {faiss_status}")
                    print(f"  HMAC Engine   : {hmac_status}")
                else:
                    print(f"  Dashboard returned {r.status} — proceeding anyway")
        except Exception as exc:
            print(f"  Dashboard unreachable: {exc} — proceeding anyway")

        # ----- Concurrent strike (staggered in 3 waves to avoid TCP saturation) -----
        print(f"\n[{_ts()}] LAUNCHING 12-AGENT OMNI-STRIKE (3 waves)...\n")

        # Wave 1: Fast/sync agents (no LLM or lightweight)
        wave1 = [
            strike_verify(session, base_url, headers),
            strike_briefing(session, base_url, headers),
            strike_raa(session, base_url, headers),
            strike_dva(session, base_url, headers),
        ]
        # Wave 2: LLM-heavy sync agents
        wave2 = [
            strike_roa(session, base_url, headers),
            strike_rma(session, base_url, headers),
            strike_cra(session, base_url, headers),
        ]
        # Wave 3: Async/polled LLM agents
        wave3 = [
            strike_capa(session, base_url, headers),
            strike_pms(session, base_url, headers),
            strike_tda(session, base_url, headers),
            strike_swarm(session, base_url, headers),
            strike_rsa(session, base_url, headers),
        ]

        global_t0 = time.monotonic()
        results1 = await asyncio.gather(*wave1, return_exceptions=True)
        await asyncio.sleep(3)
        results2 = await asyncio.gather(*wave2, return_exceptions=True)
        await asyncio.sleep(3)
        results3 = await asyncio.gather(*wave3, return_exceptions=True)
        results = list(results1) + list(results2) + list(results3)
        global_ms = _ms(global_t0)

        # ----- Results table -----
        print(f"\n{'='*64}")
        print(f"  {'AGENT':<12} | {'STATUS':>6} | {'LATENCY':>10} | NOTES")
        print(f"{'-'*64}")

        ok_count = 0
        fail_count = 0
        all_issues: list[str] = []

        for res in results:
            if isinstance(res, Exception):
                print(f"  {'EXCEPTION':<12} | {'ERR':>6} | {'—':>10} | {res}")
                fail_count += 1
                continue

            agent = res.get("agent", "?")
            status = res.get("status", "?")
            ms = res.get("ms", 0)
            body = res.get("body", {})

            # Determine success: 200 or 202 are both acceptable
            is_ok = status in (200, 202)
            icon = "OK" if is_ok else "FAIL"
            if is_ok:
                ok_count += 1
            else:
                fail_count += 1

            # Extract notes
            notes = ""
            if agent == "VERIFY":
                verifs = body.get("result", body).get("verifications", [])
                verdicts = [v.get("verdict", "?") for v in verifs]
                notes = f"verdicts={verdicts}"
                issues = _validate_verify(body)
                all_issues.extend(issues)
            elif agent == "CRA":
                notes = f"sse_events={body.get('sse_events', '?')}"
            elif agent == "RAA":
                notes = f"recalls={body.get('count', '?')}"
            elif agent == "BRIEFING":
                brief_result = body.get("result", body)
                risk = brief_result.get("risk_level", "?")
                notes = f"risk_level={risk}"
            elif agent == "DVA":
                dva_result = body.get("result", body)
                items = dva_result.get("total_items", "?")
                notes = f"items={items}"
            elif agent == "SWARM":
                swarm_result = body.get("result", body)
                agents_run = swarm_result.get("agents_run", "?")
                notes = f"agents_run={agents_run}"
            elif agent == "RMA":
                rma_result = body.get("result", body)
                verdict = rma_result.get("overall_verdict", "?")
                notes = f"verdict={verdict}"

            print(f"  {agent:<12} | {icon:>6} | {ms:>8.1f}ms | {notes}")

        print(f"{'-'*64}")
        print(f"  TOTAL: {ok_count}/12 OK, {fail_count}/12 FAILED")
        print(f"  Wall-clock time: {global_ms:.1f}ms")
        print(f"{'='*64}")

        # ----- Defensive architecture validation -----
        if all_issues:
            print(f"\n  DEFENSIVE ARCHITECTURE ISSUES:")
            for issue in all_issues:
                print(f"    {issue}")

        # ----- Post-strike: Dashboard integrity check -----
        print(f"\n[{_ts()}] POST-STRIKE DASHBOARD CHECK...")
        try:
            async with session.get(dash_url, headers=headers) as r:
                if r.status == 200:
                    dash = await r.json()
                    health = dash.get("global_health")
                    latency = dash.get("latency_ms", "?")
                    cache = dash.get("components", {}).get("redis_v3_cache", {})
                    faiss_comp = dash.get("components", {}).get("faiss_vector_store", {})

                    print(f"  Global Health  : {'NOMINAL' if health else 'DEGRADED'}")
                    print(f"  Dashboard lat. : {latency}ms")
                    print(f"  Cache status   : {cache.get('status', '?')}")
                    print(f"  Cache version  : {cache.get('version', '?')}")
                    print(f"  Cache key fmt  : {cache.get('key_format', '?')}")
                    print(f"  FAISS chunks   : {faiss_comp.get('total_chunks', '?')}")
                    print(f"  FAISS verdict  : {faiss_comp.get('verdict', '?')}")

                    # Validate defensive flags
                    dash_issues = _validate_dashboard(dash)
                    if dash_issues:
                        print(f"\n  DASHBOARD ISSUES:")
                        for di in dash_issues:
                            print(f"    {di}")
                    else:
                        print(f"  Defensive arch : ALL FLAGS NOMINAL")
                else:
                    print(f"  Dashboard returned {r.status}")
        except Exception as exc:
            print(f"  Dashboard check failed: {exc}")

        # ----- Final verdict -----
        print(f"\n{'='*64}")
        if fail_count == 0 and not all_issues:
            print("  VERDICT: ALL SYSTEMS NOMINAL — OMNI-STRIKE SURVIVED")
        elif fail_count == 0:
            print("  VERDICT: ALL AGENTS OK — DEFENSIVE ISSUES DETECTED")
        else:
            print(f"  VERDICT: {fail_count} AGENT(S) FAILED — REVIEW REQUIRED")
        print(f"{'='*64}\n")

        # ----- Write unified_field_results.txt -----
        import datetime as dt
        with open("unified_field_results.txt", "w") as f:
            f.write(f"OrthoLink Omni Stress Test — Unified Field Results\n")
            f.write(f"Run: {dt.datetime.now().isoformat()}\n")
            f.write(f"Target: {base_url}\n")
            f.write(f"Duration: {global_ms:.1f}ms\n")
            f.write(f"Result: {ok_count}/{ok_count + fail_count} agents passed\n")
            f.write(f"{'='*60}\n\n")
            for res in results:
                if isinstance(res, Exception):
                    f.write(f"[FAIL] EXCEPTION: {res}\n")
                else:
                    agent = res.get("agent", "?")
                    status = res.get("status", "?")
                    ms = res.get("ms", 0)
                    icon = "PASS" if status in (200, 202) else "FAIL"
                    f.write(f"[{icon}] {agent}: status={status}, latency={ms:.1f}ms\n")
            f.write(f"\n{'='*60}\n")
            if fail_count == 0 and not all_issues:
                f.write("UNIFIED FIELD THEORY CONFIRMED — ALL AGENTS OPERATIONAL\n")
            elif all_issues:
                f.write(f"ISSUES FOUND:\n")
                for iss in all_issues:
                    f.write(f"  - {iss}\n")
            if fail_count > 0:
                f.write(f"{fail_count} agent(s) failed — review required\n")
        print(f"  Results written to unified_field_results.txt")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OrthoLink 12-Agent Omni-Strike")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="Backend URL")
    parser.add_argument("--token", default=None, help="JWT bearer token")
    args = parser.parse_args()

    token = args.token
    if not token:
        token = _load_token_from_env()
    if not token:
        sys.exit(
            "No JWT token. Provide --token or set VITE_AUTH_TOKEN in frontend-vite/.env"
        )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(run_omni_strike(args.base_url, token))


if __name__ == "__main__":
    main()
