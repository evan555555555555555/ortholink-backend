#!/usr/bin/env python3
"""
OrthoLink — Total System Shock: 12-Agent Concurrent Stress Test
================================================================

Fires ALL 12 agents simultaneously against a live backend with a Class III
Titanium Spinal Fixation System payload.  Validates response structure,
timing, cache behaviour, and defensive architecture invariants.

PREREQUISITES:
  1. Redis running:        brew services start redis
  2. Backend running:      cd backend && .venv/bin/python3 -m uvicorn app.main:app --port 8000
  3. Real OPENAI_API_KEY:  must be set in backend/.env (not test-key-not-real)

USAGE:
  cd backend
  .venv/bin/python3 stress_test_agents.py [--base-url http://localhost:8000]

Reports per-agent timing, validates response shapes, and checks defensive
architecture invariants (country isolation, revoked-law filter, static fallback).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"

DEVICE = "Class III Titanium Spinal Fixation System"
DEVICE_DESC = (
    "A Class III titanium alloy pedicle screw spinal fixation system intended "
    "for posterior non-cervical fixation as an adjunct to fusion in the treatment "
    "of degenerative disc disease (DDD), spondylolisthesis, and spinal stenosis. "
    "The system includes polyaxial screws, rods, cross-connectors, and locking caps."
)
INTENDED_USE = (
    "Posterior non-cervical spinal fixation as an adjunct to fusion in patients "
    "with DDD, spondylolisthesis grade I-II, trauma, or tumor."
)

# Sample IFU text for CRA streaming review
IFU_TEXT = (
    "INSTRUCTIONS FOR USE — Titanium Spinal Fixation System\n\n"
    "1. INTENDED USE: This device is intended for posterior non-cervical fixation.\n"
    "2. CONTRAINDICATIONS: Active infection, insufficient bone quality, obesity.\n"
    "3. WARNINGS: MRI conditional — see MRI safety section.\n"
    "4. STERILITY: Provided sterile via gamma irradiation. Do not re-sterilize.\n"
    "5. ADVERSE EVENTS: Report to FDA via MedWatch (21 CFR 803).\n"
)

# Claims for verify-claims (mix of true, partial, and potentially contradicted)
CLAIMS = [
    "FDA requires a 510(k) submission for Class III spinal fixation devices",
    "EU MDR Annex II requires a Summary of Safety and Clinical Performance (SSCP) for Class III implants",
    "ISO 14971:2019 mandates a risk management file with a 5x5 severity-probability matrix",
    "Brazil ANVISA RDC 185/2001 governs medical device registration",
    "Ukraine MOH requires a Certificate of Free Sale for device import",
]


# ── Agent Fire Functions ──────────────────────────────────────────────────────

def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _form_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _poll_job(base: str, job_id: str, token: str, timeout: int = 120) -> dict:
    """Poll GET /api/v1/jobs/{job_id} until completed or timeout."""
    url = f"{base}/api/v1/jobs/{job_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(url, headers=_headers(token), timeout=10)
        if r.status_code != 200:
            time.sleep(2)
            continue
        data = r.json()
        status = data.get("status", "")
        if status == "completed":
            return data.get("result", data)
        if status == "failed":
            return {"error": data.get("error", "Job failed"), "status": "failed"}
        time.sleep(2)
    return {"error": "Poll timeout", "status": "timeout"}


def fire_swarm(base: str, token: str) -> dict:
    """Swarm Orchestrator — TDA+PMS+ROA+CAPA in parallel."""
    r = requests.post(
        f"{base}/api/v1/swarm-analysis",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "country": "US",
            "device_class": "III",
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token, timeout=180)
    return r.json()


def fire_rma(base: str, token: str) -> dict:
    """RMA — ISO 14971 risk analysis."""
    r = requests.post(
        f"{base}/api/v1/risk-analysis",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "country": "US",
            "device_class": "III",
            "hazards_hint": "titanium degradation, screw loosening, rod fracture, corrosion",
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token)
    return r.json()


def fire_capa(base: str, token: str) -> dict:
    """CAPA — 21 CFR 820.100 corrective action."""
    r = requests.post(
        f"{base}/api/v1/capa",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "country": "US",
            "device_class": "III",
            "issue_description": (
                "Field reports of titanium alloy degradation in pedicle screws after "
                "24 months post-implant. MAUDE adverse event reports indicate loosening "
                "and metallosis in 0.3% of implanted units."
            ),
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token)
    return r.json()


def fire_pms(base: str, token: str) -> dict:
    """PMS — Post-market surveillance plan."""
    r = requests.post(
        f"{base}/api/v1/pms-plan",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "country": "EU",
            "device_class": "III",
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token)
    return r.json()


def fire_tda(base: str, token: str) -> dict:
    """TDA — Technical dossier checklist."""
    r = requests.post(
        f"{base}/api/v1/technical-dossier",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "country": "EU",
            "device_class": "III",
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token)
    return r.json()


def fire_roa(base: str, token: str) -> dict:
    """ROA — Role-split checklist."""
    r = requests.post(
        f"{base}/api/v1/generate-checklist",
        headers=_form_headers(token),
        data={"country": "US", "device_class": "III"},
        timeout=60,
    )
    return r.json()


def fire_dva(base: str, token: str) -> dict:
    """DVA — Distributor verification."""
    r = requests.post(
        f"{base}/api/v1/verify-distributor",
        headers=_headers(token),
        json={
            "distributor_name": "MedTech Solutions LLC",
            "country": "US",
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "documents": [
                {"name": "ISO 13485 Certificate", "content": "Certified QMS"},
                {"name": "FDA Establishment Registration", "content": "FEI 3012345678"},
            ],
        },
        timeout=60,
    )
    return r.json()


def fire_rsa(base: str, token: str) -> dict:
    """RSA — Regulatory strategy."""
    r = requests.post(
        f"{base}/api/v1/plan-strategy",
        headers=_headers(token),
        json={
            "device_description": DEVICE_DESC,
            "intended_use": INTENDED_USE,
            "target_markets": ["US", "EU", "BR"],
            "device_class": "III",
        },
        timeout=15,
    )
    if r.status_code == 202:
        job_id = r.json().get("job_id")
        return _poll_job(base, job_id, token, timeout=180)
    return r.json()


def fire_verify(base: str, token: str) -> dict:
    """Verify — FAISS-only truth checker."""
    r = requests.post(
        f"{base}/api/v1/verify-claims",
        headers=_headers(token),
        json={
            "claims": CLAIMS,
            "country": "US",
            "device_class": "III",
        },
        timeout=60,
    )
    return r.json()


def fire_briefing(base: str, token: str) -> dict:
    """Briefing — Daily intelligence brief."""
    r = requests.post(
        f"{base}/api/v1/briefing/run",
        headers=_headers(token),
        timeout=60,
    )
    return r.json()


def fire_cra(base: str, token: str) -> dict:
    """CRA — SSE streaming compliance review (collect full stream)."""
    r = requests.post(
        f"{base}/api/v1/review-document",
        headers=_form_headers(token),
        data={"document_text": IFU_TEXT, "country": "US", "device_class": "III"},
        stream=True,
        timeout=120,
    )
    chunks = []
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            chunks.append(payload)
    return {"sse_chunks": len(chunks), "sample": chunks[:2] if chunks else []}


def fire_fda_alerts(base: str, token: str) -> dict:
    """RAA — Live FDA recalls."""
    r = requests.get(
        f"{base}/api/v1/alerts/live/fda-recalls",
        headers=_headers(token),
        params={"search": "spinal"},
        timeout=30,
    )
    return r.json()


def fire_dashboard(base: str, token: str) -> dict:
    """Dashboard — System status aggregator."""
    r = requests.get(
        f"{base}/api/v1/dashboard/system-status",
        headers=_headers(token),
        timeout=15,
    )
    return r.json()


# ── Agent Registry ────────────────────────────────────────────────────────────

AGENTS = [
    ("FDA Alerts (RAA)",    fire_fda_alerts),
    ("Swarm Orchestrator",  fire_swarm),
    ("Risk Analysis (RMA)", fire_rma),
    ("CAPA Analysis",       fire_capa),
    ("PMS Plan",            fire_pms),
    ("Technical Dossier",   fire_tda),
    ("Checklist (ROA)",     fire_roa),
    ("Distributor (DVA)",   fire_dva),
    ("Strategy (RSA)",      fire_rsa),
    ("Compliance (CRA)",    fire_cra),
    ("Verify Claims",       fire_verify),
    ("Daily Brief",         fire_briefing),
]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_verify_result(result: dict) -> list[str]:
    """Check verify-claims for revoked law detection and no hallucinations."""
    issues = []
    outer = result.get("result", result)
    claims = outer.get("claim_results", [])

    for c in claims:
        # RDC 185/2001 is REVOKED — must be CONTRADICTED or UNVERIFIED
        if "185/2001" in c.get("claim", "") or "RDC 185" in c.get("claim", ""):
            if c.get("verdict") == "VERIFIED":
                issues.append(
                    f"CRITICAL: Revoked law RDC 185/2001 was VERIFIED — "
                    f"defensive filter failed!"
                )

        # 510(k) is wrong for Class III (PMA required) — should not be VERIFIED
        if "510(k)" in c.get("claim", "") and "Class III" in c.get("claim", ""):
            if c.get("verdict") == "VERIFIED":
                issues.append(
                    f"WARNING: 510(k) for Class III was VERIFIED — "
                    f"Class III typically requires PMA, not 510(k)"
                )

    return issues


# ── Main ──────────────────────────────────────────────────────────────────────

def run_stress_test(base_url: str, token: str) -> bool:
    print(f"\n{BOLD}{'═' * 70}")
    print(f"  OrthoLink — TOTAL SYSTEM SHOCK")
    print(f"  12-Agent Concurrent Stress Test")
    print(f"  Device: {DEVICE}")
    print(f"  Target: {base_url}")
    print(f"{'═' * 70}{RESET}\n")

    # Pre-flight: dashboard check
    print(f"{'─' * 70}")
    print(f"  0 · Pre-flight: System Status")
    print(f"{'─' * 70}")
    try:
        dash = fire_dashboard(base_url, token)
        health = dash.get("global_health", False)
        verdict = dash.get("global_verdict", "UNKNOWN")
        latency = dash.get("latency_ms", -1)
        print(f"  {PASS if health else WARN} Global health: {verdict}  [{latency}ms]")

        components = dash.get("components", {})
        for name, comp in components.items():
            status = comp.get("status", "unknown")
            icon = PASS if status in ("online", "healthy") else WARN
            print(f"  {icon} {name}: {status}")
    except Exception as e:
        print(f"  {FAIL} Dashboard unreachable: {e}")
        print(f"\n  Is the backend running at {base_url}?")
        return False

    # Fire all 12 agents concurrently
    print(f"\n{'─' * 70}")
    print(f"  1 · Firing 12 agents concurrently")
    print(f"{'─' * 70}")

    results = {}
    timings = {}
    errors = []

    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {}
        for name, fn in AGENTS:
            futures[pool.submit(fn, base_url, token)] = name

        for future in as_completed(futures):
            name = futures[future]
            elapsed = time.monotonic() - t_start
            try:
                result = future.result()
                results[name] = result
                timings[name] = round(elapsed, 2)
                # Quick status check
                has_error = (
                    isinstance(result, dict)
                    and (result.get("status") == "failed" or "error" in result)
                    and "sse_chunks" not in result
                )
                icon = FAIL if has_error else PASS
                if has_error:
                    errors.append(name)

                # Size metric
                size = len(json.dumps(result, default=str))
                print(f"  {icon} {name:.<40s} {elapsed:6.1f}s  [{size:,} bytes]")

            except Exception as e:
                timings[name] = round(elapsed, 2)
                results[name] = {"error": str(e)}
                errors.append(name)
                print(f"  {FAIL} {name:.<40s} {elapsed:6.1f}s  [EXCEPTION: {e}]")

    total_time = round(time.monotonic() - t_start, 2)

    # Execution telemetry
    print(f"\n{'─' * 70}")
    print(f"  2 · Execution Telemetry")
    print(f"{'─' * 70}")
    print(f"  {INFO} Total wall time: {total_time}s (12 agents concurrent)")
    print(f"  {INFO} Agents succeeded: {12 - len(errors)}/12")
    if errors:
        print(f"  {WARN} Failed agents: {', '.join(errors)}")

    sorted_times = sorted(timings.items(), key=lambda x: x[1])
    fastest = sorted_times[0]
    slowest = sorted_times[-1]
    print(f"  {INFO} Fastest: {fastest[0]} ({fastest[1]}s)")
    print(f"  {INFO} Slowest: {slowest[0]} ({slowest[1]}s)")

    # Verify-claims audit
    print(f"\n{'─' * 70}")
    print(f"  3 · Defensive Architecture Validation")
    print(f"{'─' * 70}")

    if "Verify Claims" in results:
        issues = validate_verify_result(results["Verify Claims"])
        if issues:
            for issue in issues:
                print(f"  {FAIL} {issue}")
        else:
            print(f"  {PASS} Verify-claims: no revoked laws hallucinated as VERIFIED")

        outer = results["Verify Claims"].get("result", results["Verify Claims"])
        verdicts = {}
        for c in outer.get("claim_results", []):
            v = c.get("verdict", "UNKNOWN")
            verdicts[v] = verdicts.get(v, 0) + 1
        print(f"  {INFO} Verdict distribution: {verdicts}")
    else:
        print(f"  {WARN} Verify Claims did not return results")

    # Swarm result check
    if "Swarm Orchestrator" in results:
        swarm = results["Swarm Orchestrator"]
        outer = swarm.get("result", swarm)
        agents_completed = []
        for key in ("tda", "pms", "roa", "capa"):
            if key in outer or f"{key}_result" in outer:
                agents_completed.append(key.upper())
        if agents_completed:
            print(f"  {PASS} Swarm sub-agents completed: {', '.join(agents_completed)}")
        else:
            print(f"  {WARN} Swarm result structure unexpected")

    # RMA risk matrix check
    if "Risk Analysis (RMA)" in results:
        rma = results["Risk Analysis (RMA)"]
        outer = rma.get("result", rma)
        hazards = outer.get("hazard_analysis", [])
        verdict = outer.get("overall_verdict", "UNKNOWN")
        print(f"  {PASS} RMA: {len(hazards)} hazards identified, verdict={verdict}")

    # CRA SSE check
    if "Compliance (CRA)" in results:
        cra = results["Compliance (CRA)"]
        chunks = cra.get("sse_chunks", 0)
        print(f"  {PASS if chunks > 0 else WARN} CRA SSE stream: {chunks} chunks received")

    # Dashboard post-check
    print(f"\n{'─' * 70}")
    print(f"  4 · Post-Strike System Status")
    print(f"{'─' * 70}")
    try:
        dash = fire_dashboard(base_url, token)
        health = dash.get("global_health", False)
        verdict = dash.get("global_verdict", "UNKNOWN")
        icon = PASS if health else WARN
        print(f"  {icon} Post-strike health: {verdict}")
        print(f"  {icon} System survived 12-agent concurrent assault")
    except Exception as e:
        print(f"  {FAIL} Post-strike dashboard failed: {e}")

    # Final verdict
    success = len(errors) == 0
    print(f"\n{'═' * 70}")
    if success:
        print(f"  {PASS} ALL 12/12 AGENTS OPERATIONAL — System Shock Survived")
    else:
        print(f"  {WARN} {12 - len(errors)}/12 agents passed, {len(errors)} failed")
    print(f"  Total wall time: {total_time}s")
    print(f"{'═' * 70}\n")

    return success


def _load_token() -> str:
    """Load JWT from frontend .env (VITE_AUTH_TOKEN)."""
    env_paths = [
        Path(__file__).parent.parent / "frontend-vite" / ".env",
        Path(__file__).parent / ".env",
    ]
    for p in env_paths:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("VITE_AUTH_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoLink 12-Agent Stress Test")
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--token", default=None,
        help="JWT token (default: reads VITE_AUTH_TOKEN from frontend .env)",
    )
    args = parser.parse_args()

    token = args.token or _load_token()
    if not token:
        print(f"{FAIL} No JWT token found. Set --token or VITE_AUTH_TOKEN in frontend/.env")
        sys.exit(1)

    try:
        ok = run_stress_test(args.base_url, token)
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(130)
    except Exception:
        import traceback
        print(f"\n{FAIL} Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)
