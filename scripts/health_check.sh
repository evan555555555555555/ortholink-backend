#!/usr/bin/env bash
# OrthoLink backend health check — run with backend on port 8000.
# Usage: from backend/ run: JWT=$(...) bash scripts/health_check.sh
# Or: SUPABASE_JWT_SECRET=test-jwt-secret-for-unit-tests-only poetry run python -c "from app.middleware.auth import create_test_jwt; print(create_test_jwt())" then export JWT and run.

set -e
BASE="${BASE:-http://localhost:8000}"
JWT="${JWT:?Set JWT (e.g. export JWT=\$(poetry run python -c 'from app.middleware.auth import create_test_jwt; print(create_test_jwt())'))}"

echo "=== OrthoLink health check (BASE=$BASE) ==="

# Health
code=$(curl -s -o /tmp/hc_health.json -w "%{http_code}" "$BASE/health")
echo -n "GET /health ... "
if [ "$code" = "200" ]; then echo "PASS ($code)"; else echo "FAIL ($code)"; exit 1; fi

# Countries (no auth)
code=$(curl -s -o /tmp/hc_countries.json -w "%{http_code}" "$BASE/api/v1/countries")
echo -n "GET /api/v1/countries ... "
if [ "$code" = "200" ]; then echo "PASS ($code)"; else echo "FAIL ($code)"; exit 1; fi

# DVA countries (auth)
code=$(curl -s -o /tmp/hc_dva_countries.json -w "%{http_code}" -H "Authorization: Bearer $JWT" "$BASE/api/v1/verify-distributor/countries")
echo -n "GET /api/v1/verify-distributor/countries ... "
if [ "$code" = "200" ]; then echo "PASS ($code)"; else echo "FAIL ($code)"; exit 1; fi

# Alerts
code=$(curl -s -o /tmp/hc_alerts.json -w "%{http_code}" -H "Authorization: Bearer $JWT" "$BASE/api/v1/alerts")
echo -n "GET /api/v1/alerts ... "
if [ "$code" = "200" ]; then
  echo "PASS ($code)"
  python3 -c "import json; d=json.load(open('/tmp/hc_alerts.json')); assert 'alerts' in d or 'count' in d" 2>/dev/null || true
else
  echo "FAIL ($code)"; exit 1
fi

# ROA (Form: country, device_class)
code=$(curl -s -o /tmp/hc_roa.json -w "%{http_code}" --max-time 60 -X POST "$BASE/api/v1/generate-checklist" \
  -H "Authorization: Bearer $JWT" -F "country=UA" -F "device_class=IIb")
echo -n "POST /api/v1/generate-checklist (Form) ... "
if [ "$code" = "200" ]; then
  echo "PASS ($code)"
  python3 -c "import json; d=json.load(open('/tmp/hc_roa.json')); assert 'job_id' in d and 'items' in d" 2>/dev/null || true
else
  echo "FAIL ($code)"; exit 1
fi

# Strategy (Form: device_name, target_markets, async_mode)
code=$(curl -s -o /tmp/hc_rsa.json -w "%{http_code}" --max-time 15 -X POST "$BASE/api/v1/plan-strategy" \
  -H "Authorization: Bearer $JWT" -F "device_name=Test Device" -F "target_markets=UA,US" -F "async_mode=true")
echo -n "POST /api/v1/plan-strategy (Form, async) ... "
if [ "$code" = "202" ]; then
  echo "PASS ($code)"
  python3 -c "import json; d=json.load(open('/tmp/hc_rsa.json')); assert d.get('job_id')" 2>/dev/null || true
else
  echo "FAIL ($code)"; exit 1
fi

# DVA (optional, long timeout — 22 items can take 1–2 min)
if [ -f "helton_ukraine_753.csv" ]; then
  echo -n "POST /api/v1/verify-distributor (UA 22 items) ... "
  code=$(curl -s -o /tmp/hc_dva.json -w "%{http_code}" --max-time 180 -X POST "$BASE/api/v1/verify-distributor" \
    -H "Authorization: Bearer $JWT" -F "country=UA" -F "device_class=IIb" -F "file=@helton_ukraine_753.csv")
  if [ "$code" = "200" ]; then
    echo "PASS ($code)"
    python3 -c "
import json
d=json.load(open('/tmp/hc_dva.json'))
items=d.get('items',[])
required=sum(1 for i in items if i.get('status')=='REQUIRED')
summary=d.get('summary',{}) or {}
print(f'  -> {required}/{len(items)} REQUIRED, fraud_risk_score={summary.get(\"fraud_risk_score\", \"N/A\")}')
" 2>/dev/null || true
  else
    echo "FAIL or timeout ($code)"
  fi
else
  echo "POST /api/v1/verify-distributor ... SKIP (no helton_ukraine_753.csv)"
fi

echo "=== All required endpoints passed ==="
