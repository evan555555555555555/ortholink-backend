#!/usr/bin/env bash
# Run ingestion with backend/.env loaded (OPENAI_API_KEY required).
# Usage: cd backend && bash scripts/run_ingest_with_env.sh scripts/ingest_ukraine.py --file ../ukraine-regs/resolution-753.pdf --document-id UA-RES-753 --language uk

set -e
cd "$(dirname "$0")/.."
set -a
[ -f .env ] && . ./.env
set +a
exec poetry run python "$@"
