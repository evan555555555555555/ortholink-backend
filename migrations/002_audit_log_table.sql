-- ============================================================
-- OrthoLink Migration 002 — Audit Log Table
-- Stores every action logged by audit_logger.py.
-- ============================================================

create extension if not exists "pgcrypto";

create table if not exists audit_logs (
    id            uuid primary key default gen_random_uuid(),
    action        text not null,
    org_id        text not null default '',
    user_id       text,
    resource_type text,
    resource_id   text,
    details       jsonb not null default '{}',
    created_at    timestamptz not null default now()
);

create index if not exists audit_logs_org_id_idx    on audit_logs (org_id);
create index if not exists audit_logs_action_idx    on audit_logs (action);
create index if not exists audit_logs_created_at_idx on audit_logs (created_at desc);

alter table audit_logs enable row level security;

comment on table audit_logs is
    'Immutable audit trail for all OrthoLink agent actions.';
comment on column audit_logs.action        is 'Action name (e.g. raa_alert, cra_review_started).';
comment on column audit_logs.org_id        is 'Org that performed the action.';
comment on column audit_logs.resource_type is 'Type of resource (e.g. alert, document, distributor).';
comment on column audit_logs.resource_id   is 'ID of the resource being acted on.';
comment on column audit_logs.details       is 'Arbitrary action metadata as JSON.';
