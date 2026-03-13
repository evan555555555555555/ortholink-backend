-- ============================================================
-- OrthoLink Migration 001 — RAA Alert Store Tables
-- Run against your Supabase project (SQL Editor or CLI):
--   psql $DATABASE_URL -f 001_raa_tables.sql
--   OR copy-paste into Supabase Dashboard → SQL Editor
-- ============================================================

-- Enable UUID extension (already enabled on Supabase by default)
create extension if not exists "pgcrypto";


-- ──────────────────────────────────────────────────────────────
-- raa_alerts
--   Stores every AlertEvent emitted by the RAA crew.
--   alert_store.py reads / writes this table.
-- ──────────────────────────────────────────────────────────────
create table if not exists raa_alerts (
    id              uuid primary key default gen_random_uuid(),
    country         text not null,
    document_id     text,
    payload         jsonb not null default '{}',
    notified_orgs   text[] not null default '{}',
    created_at      timestamptz not null default now()
);

-- Speed up recent-first queries (alert_store.py: .order("created_at", desc=True))
create index if not exists raa_alerts_created_at_idx
    on raa_alerts (created_at desc);

-- Speed up per-country queries (GET /api/v1/alerts/{country})
create index if not exists raa_alerts_country_idx
    on raa_alerts (country);

-- Speed up per-org queries (filter notified_orgs array)
create index if not exists raa_alerts_notified_orgs_gin_idx
    on raa_alerts using gin (notified_orgs);

comment on table raa_alerts is
    'RAA crew alert events — one row per detected regulatory change.';
comment on column raa_alerts.country      is 'ISO country code (uppercase, e.g. US, EU, UA).';
comment on column raa_alerts.document_id  is 'Stable document slug matching monitored_docs.py.';
comment on column raa_alerts.payload      is 'Full AlertEvent.model_dump() JSON.';
comment on column raa_alerts.notified_orgs is 'org_id list of orgs that were notified of this change.';


-- ──────────────────────────────────────────────────────────────
-- raa_subscriptions
--   Tracks which orgs want alerts for which countries.
--   alert_store.py reads / upserts this table.
-- ──────────────────────────────────────────────────────────────
create table if not exists raa_subscriptions (
    org_id      text not null,
    country     text not null,
    created_at  timestamptz not null default now(),
    primary key (org_id, country)
);

-- Speed up "get all orgs subscribed to country X" queries
create index if not exists raa_subscriptions_country_idx
    on raa_subscriptions (country);

comment on table raa_subscriptions is
    'Per-org alert subscriptions — which countries to watch.';
comment on column raa_subscriptions.org_id   is 'Org identifier from the JWT (sub or org claim).';
comment on column raa_subscriptions.country  is 'ISO country code (uppercase).';


-- ──────────────────────────────────────────────────────────────
-- Row Level Security (RLS)
--   Supabase best practice: enable RLS and create policies.
--   The backend always uses the SERVICE_ROLE_KEY (bypasses RLS),
--   so these policies are for direct client queries only.
-- ──────────────────────────────────────────────────────────────
alter table raa_alerts       enable row level security;
alter table raa_subscriptions enable row level security;

-- Service role bypasses RLS automatically; no extra policy needed.
-- Add anon / authenticated policies here if you expose these tables
-- to the client SDK directly (currently not required by this app).

-- Example: allow authenticated users to read their org's alerts
-- (Uncomment and adjust if you add Supabase Auth integration)
--
-- create policy "users read own org alerts"
--     on raa_alerts for select
--     using (auth.jwt() ->> 'org_id' = any(notified_orgs));
--
-- create policy "users manage own subscriptions"
--     on raa_subscriptions for all
--     using (auth.jwt() ->> 'org_id' = org_id);
