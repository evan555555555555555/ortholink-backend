# OrthoLink Backend

Regulatory Intelligence Platform for Medical Devices.  
Five AI Agents. Fifteen Countries. Three User Types.

## M1: Foundation + DVA

This milestone implements:

- **FastAPI Backend** with CORS, JWT auth, RBAC (Admin/Reviewer/Viewer)
- **Supabase Schema** with 12 tables and Row-Level Security (RLS)
- **DVA Agent** (Distributor Verification Agent) — CrewAI sequential crew
- **Ingestion Pipeline** — scrape, validate, translate, chunk, embed, audit
- **FAISS Vector Store** with country isolation (HC-5)
- **Anti-Hallucination** guards (HC-1 through HC-10)
- **Audit Logging** — append-only, immutable (HC-6)

## Quick Start

```bash
# 1. Install dependencies
cd backend
poetry install

# 2. Copy environment config
cp .env.example .env
# Edit .env with your Supabase and OpenAI credentials

# 3. Run Supabase migration
# Apply supabase/migrations/20260305000000_ortholink_m1_schema.sql to your Supabase project

# 4. Start the development server
poetry run uvicorn app.main:app --reload --port 8000

# 5. Verify
curl http://localhost:8000/health
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── agents/              # CrewAI agents
│   │   └── dva_agent.py     # DVA: Distributor Verification Agent
│   ├── core/
│   │   ├── config.py        # Pydantic Settings
│   │   └── anti_hallucination.py  # Confidence gating, citations, refusals
│   ├── crews/
│   │   └── verify_distributor.py  # DVA sequential crew + GapAnalysisReport
│   ├── ingestion/
│   │   ├── scraper.py       # BeautifulSoup + Selenium
│   │   ├── scraper_validator.py  # Content quality checks
│   │   ├── translator.py    # gpt-4o translation (not Google Translate)
│   │   ├── chunker.py       # Article/Section hierarchy chunker
│   │   ├── embedder.py      # Batch embedding + FAISS indexer
│   │   └── chunk_audit.py   # Post-ingestion validation
│   ├── middleware/
│   │   ├── auth.py          # JWT verification via Supabase
│   │   └── rbac.py          # Role-based access control
│   ├── routes/
│   │   ├── health.py        # Health check endpoints
│   │   ├── dva.py           # POST /api/v1/verify-distributor
│   │   ├── query.py         # GET /api/v1/query
│   │   └── admin.py         # Admin provisioning
│   ├── services/
│   │   ├── audit_logger.py  # Append-only audit log (HC-6)
│   │   ├── supabase_client.py  # Supabase client singleton
│   │   ├── usage_metering.py   # Usage tracking / trial limits
│   │   └── email_service.py    # Resend email service
│   └── tools/
│       ├── embeddings.py    # text-embedding-3-large (HC-1)
│       ├── similarity.py    # Cosine similarity (0.82 threshold)
│       ├── llm.py           # gpt-4o structured output (HC-9)
│       └── vector_store.py  # FAISS + country isolation (HC-5)
├── scripts/
│   ├── provision_customer.py  # CLI: create new org + admin
│   └── ingest_country.py     # CLI: ingest regulatory data
├── tests/
│   └── ...
├── data/
│   ├── raw/               # Scraped regulatory text
│   ├── processed/         # Chunked/translated text
│   ├── embeddings/        # FAISS index files
│   └── archive/           # Previous index versions
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/detailed` | Detailed health with dependency status |
| POST | `/api/v1/verify-distributor` | DVA gap analysis (upload CSV) |
| GET | `/api/v1/verify-distributor/countries` | List indexed countries |
| GET | `/api/v1/query` | Plain-English regulatory Q&A |
| POST | `/api/v1/admin/provision` | Provision new organization |
| POST | `/api/v1/export-pdf` | WeasyPrint PDF (checklists; `report_type=checklist`) |

**PDF export (WeasyPrint):** Requires system libs (pango, cairo). On macOS without them, `test_pdf_export` skips; this is expected. PDF generation works in Docker and in CI images that install those libs. Not a code defect.

## Anti-Hallucination Constraints (HC-1 through HC-10)

| ID | Constraint | Enforcement |
|----|-----------|-------------|
| HC-1 | text-embedding-3-large ONLY | `embeddings.py` + config; grep-verifiable |
| HC-2 | ZERO hardcoded dicts (COMMERCIAL_ITEMS, UA_ITEM_TERMS) | Classification is semantic + Set Theory gate only |
| HC-3 | Chunking preserves legal hierarchy | Article/Section/Clause in chunker; audit script |
| HC-4 | Safe refusal: out-of-scope → status REFUSED | `is_out_of_scope` + `QueryResponse.status = "REFUSED"` |
| HC-5 | Secrets in env; .env in .gitignore | Never commit .env |
| HC-6 | Daily commits to dev branch | Git workflow |
| HC-7 | Every output: "Reference tool only. Verify with official sources." | `GapAnalysisReport.disclaimer`; all agent outputs |
| HC-8 | Append-only audit_log; REVOKE DELETE | Supabase migration |
| HC-9 | Multi-tenant RLS; org isolation | Supabase RLS |
| HC-10 | API keys never in frontend; all AI via FastAPI | Proxy only |

## Verification Commands (PRD)

Run before release:

```bash
# HC-1: Only text-embedding-3-large
grep -r 'text-embedding' app/ | grep -v 'large'  # Should return NOTHING

# HC-2: No hardcoded classification dictionaries
grep -rn 'COMMERCIAL_ITEMS\|UA_ITEM_TERMS\|KNOWN_ITEMS' app/  # Should return NOTHING

# HC-5: .env never committed
git log --all --full-history -- .env  # Should return NOTHING

# Tests
poetry run python -m pytest tests/ -v
```

## Data encryption and transmission

**No application-level encryption or TLS configuration** — There is no custom crypto, TLS version pinning, or SSL options in the OrthoLink app code. Transmission security is not implemented in the repo; it is delegated to the transport layer and deployment.

**TLS via HTTPS** — All outbound calls use HTTPS URLs, so traffic is encrypted in transit by normal TLS (whatever the client and server negotiate):

- **Supabase:** `supabase_url` is expected to be `https://*.supabase.co`; the Supabase client does not set any SSL options.
- **OpenAI, PostHog, Sentry, Resend:** Used via their SDKs/default endpoints, which are HTTPS.
- **Backend API:** The FastAPI app does not enable TLS itself; Docker Compose exposes the backend on port 8000 with no SSL. Encryption for client–backend traffic is intended to come from deployment: the PRD and `.cursorrules` specify "HTTPS via Let's Encrypt" and "encrypted in transit (TLS 1.3+)", i.e. a reverse proxy (e.g. nginx/Caddy) in front of the app, which is not present in this repo.

**Summary:** For transmission, the project relies on TLS over HTTPS (Supabase and other services over HTTPS; backend over HTTPS only when a TLS-terminating proxy is used in production). There is no extra data encryption for transmission implemented in the codebase beyond that.

## Set Theory Gate (DVA)

The DVA pipeline uses a **Set Theory gate** to drive final classification: a distributor item is "in" the regulatory baseline iff semantic similarity to at least one baseline chunk ≥ 0.82. If `set_theory_result` is False, status is **EXTRA** (no LLM override). If True, LLM classifies as REQUIRED or OPTIONAL. See `app/crews/verify_distributor.py`.

## Certificate of Free Sale (CFS)

CFS/CFG and equivalents are **regulatory** documents. The LLM prompt instructs: always classify as REQUIRED. See `app/tools/llm.py` and `tests/test_cfs_required.py`.

## Ingestion and acceptance

Set `OPENAI_API_KEY` in `.env` (and optionally `SUPABASE_*` for auth). Then:

```bash
# Ukraine (A2): resolutions 753, 754, 755 — requires network to zakon.rada.gov.ua or use --file
poetry run python scripts/ingest_ukraine.py

# USA (A3): 21 CFR 801, 803, 806, 820
poetry run python scripts/ingest_usa.py

# India (A4): CDSCO PDF
poetry run python scripts/ingest_india.py --file path/to/Medical-Devices-Rules-2017.pdf

# Audit after each country (all 10 samples must pass before proceeding)
poetry run python -m app.ingestion.chunk_audit --country UA --sample 10
poetry run python -m app.ingestion.chunk_audit --country US --sample 10
poetry run python -m app.ingestion.chunk_audit --country IN --sample 10
```

D1 verification (with server running):

```bash
BASE_URL=http://localhost:8000 ./scripts/verify_d1.sh
```

Acceptance tests (live server + `ORTHOLINK_TEST_JWT` + populated vector store):

```bash
poetry run pytest tests/test_acceptance_ukraine.py -v -m integration
```

## Docker

```bash
docker compose up --build
```

Create `backend/.env` with real keys (or use env vars) so the app can connect to OpenAI and Supabase.
