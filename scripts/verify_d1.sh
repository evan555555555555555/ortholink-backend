#!/usr/bin/env bash
# D1 verification: run with server up (e.g. uvicorn app.main:app --reload or docker compose up).
# Usage: BASE_URL=http://localhost:8000 ./scripts/verify_d1.sh

set -e
BASE_URL="${BASE_URL:-http://localhost:8000}"

echo "=== D1.1 GET /health ==="
curl -s "${BASE_URL}/health" && echo "" || true
echo ""

echo "=== D1.2 GET /api/v1/countries ==="
curl -s "${BASE_URL}/api/v1/countries" && echo "" || true
echo ""

echo "=== D1.3 POST /api/v1/auth/magic-link ==="
curl -s -X POST "${BASE_URL}/api/v1/auth/magic-link" \
  -H "Content-Type: application/json" \
  -d '{"email": "helton@brmextremities.com"}' && echo "" || true

echo ""
echo "D1 checks requested. Verify: health has status ok and app OrthoLink; countries non-empty after ingestion; magic-link returns message."
