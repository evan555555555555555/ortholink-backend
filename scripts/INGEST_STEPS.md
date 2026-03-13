# Ingestion steps (run in order)

Ensure `backend/.env` exists and contains `OPENAI_API_KEY=sk-...`.

## STEP 1 — Ukraine (from local PDFs)

From repo root (or from `backend` with paths adjusted):

```bash
cd backend

# Load .env then run each ingestion (or run with: bash scripts/run_ingest_with_env.sh scripts/ingest_ukraine.py ...)
bash scripts/run_ingest_with_env.sh scripts/ingest_ukraine.py --file ../ukraine-regs/resolution-753.pdf --document-id UA-RES-753 --language uk
bash scripts/run_ingest_with_env.sh scripts/ingest_ukraine.py --file ../ukraine-regs/resolution-754.pdf --document-id UA-RES-754 --language uk
bash scripts/run_ingest_with_env.sh scripts/ingest_ukraine.py --file ../ukraine-regs/resolution-755.pdf --document-id UA-RES-755 --language uk

# Audit: ALL 10 must PASS
poetry run python -m app.ingestion.chunk_audit --country UA --sample 10
```

## STEP 2 — USA (eCFR)

```bash
cd backend
bash scripts/run_ingest_with_env.sh scripts/ingest_usa.py
poetry run python -m app.ingestion.chunk_audit --country US --sample 10
```

## STEP 3 — India (PDF)

```bash
cd backend
bash scripts/run_ingest_with_env.sh scripts/ingest_india.py --file ../india-regs/india-mdr-2017.pdf
poetry run python -m app.ingestion.chunk_audit --country IN --sample 10
```

## STEP 4 — Verify brain

```bash
cd backend
poetry run python scripts/verify_brain.py
```

Expected: `UA: 5 chunks found`, `US: 5 chunks found`, `IN: 5 chunks found`.

## STEP 5 — Acceptance test

Terminal 1:

```bash
cd backend && poetry run uvicorn app.main:app --reload --port 8000
```

Terminal 2 (with a valid JWT for the running server):

```bash
cd backend
export ORTHOLINK_TEST_JWT="<your-test-jwt>"
poetry run pytest tests/test_acceptance_ukraine.py -v -m integration --server-url http://localhost:8000
```

## STEP 6 — Docker

```bash
docker compose up --build
curl http://localhost:8000/health
bash backend/scripts/verify_d1.sh
```

## STEP 7 — Full test suite

```bash
cd backend
poetry run pytest tests/ -v --ignore=tests/test_acceptance_ukraine.py
```

Expected: 94+ passed, 2 skipped.
