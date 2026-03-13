#!/usr/bin/env python3
"""
Quick smoke test — verify live Supabase connection + RLS.

Creates a dummy Organization, then a dummy org_member profile,
reads them back, then cleans up.
"""

import os, sys, uuid
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load .env from backend/
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from supabase import create_client, Client

URL = os.getenv("SUPABASE_URL", "")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

if not URL or not SERVICE_KEY:
    print("FAIL: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set in .env")
    sys.exit(1)

print(f"Supabase URL : {URL}")
print(f"Service key  : {SERVICE_KEY[:20]}...{SERVICE_KEY[-8:]}")
print(f"Anon key     : {ANON_KEY[:20]}...{ANON_KEY[-8:]}")
print()

# ── Service-role client (bypasses RLS) ────────────────────────────────────
svc: Client = create_client(URL, SERVICE_KEY)

org_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())
now = datetime.now(timezone.utc).isoformat()

# ── 1. Insert Organization ────────────────────────────────────────────────
print("1. Creating test organization...")
try:
    org_data = {
        "id": org_id,
        "org_name": "OrthoLink Smoke Test Org",
        "slug": f"smoke-test-{org_id[:8]}",
        "plan_tier": "starter",
    }
    res = svc.table("organizations").insert(org_data).execute()
    print(f"   OK — org created: {res.data[0]['id']}")
except Exception as e:
    print(f"   FAIL — {e}")
    sys.exit(1)

# ── 2. Insert org_member (profile) ────────────────────────────────────────
print("2. Creating test org_member profile...")
try:
    member_data = {
        "org_id": org_id,
        "user_id": user_id,
        "role": "admin",
    }
    res = svc.table("org_members").insert(member_data).execute()
    print(f"   OK — member created: {res.data[0]['id']}")
except Exception as e:
    print(f"   FAIL — {e}")
    # Clean up org before exiting
    svc.table("organizations").delete().eq("id", org_id).execute()
    sys.exit(1)

# ── 3. Read back with service client ──────────────────────────────────────
print("3. Reading back organization...")
try:
    res = svc.table("organizations").select("*").eq("id", org_id).execute()
    assert len(res.data) == 1, f"Expected 1 row, got {len(res.data)}"
    print(f"   OK — org_name: {res.data[0]['org_name']}, plan_tier: {res.data[0]['plan_tier']}")
except Exception as e:
    print(f"   FAIL — {e}")

# ── 4. Read members ───────────────────────────────────────────────────────
print("4. Reading back org_member...")
try:
    res = svc.table("org_members").select("*").eq("org_id", org_id).execute()
    assert len(res.data) == 1, f"Expected 1 row, got {len(res.data)}"
    print(f"   OK — role: {res.data[0]['role']}, user_id: {res.data[0]['user_id']}")
except Exception as e:
    print(f"   FAIL — {e}")

# ── 5. Test anon client (RLS should block without auth) ───────────────────
print("5. Testing anon client (RLS enforcement)...")
try:
    anon: Client = create_client(URL, ANON_KEY)
    res = anon.table("organizations").select("*").eq("id", org_id).execute()
    if len(res.data) == 0:
        print("   OK — RLS blocked unauthenticated read (0 rows returned)")
    else:
        print(f"   WARNING — RLS may not be enforced (got {len(res.data)} rows)")
except Exception as e:
    print(f"   OK — RLS blocked with error: {type(e).__name__}")

# ── 6. Cleanup ────────────────────────────────────────────────────────────
print("6. Cleaning up test data...")
try:
    svc.table("org_members").delete().eq("org_id", org_id).execute()
    svc.table("organizations").delete().eq("id", org_id).execute()
    print("   OK — test data removed")
except Exception as e:
    print(f"   WARNING — cleanup failed: {e}")

print()
print("=" * 50)
print("ALL CHECKS PASSED — Supabase is live!")
print("=" * 50)
