-- LEGACY (schema v1). Since v2 the backend keeps runs in memory and the UI reads the API
-- (GET /runs/{id}); these tables are not used. Kept for reference only.
-- Lab-to-Market schema, schema_version 1. Mirrors backend/app/models.py.
-- Run in the Supabase SQL editor. Safe to re-run.
--
-- Access: anon/authenticated may SELECT every table except llm_calls.
-- Writes come only from the backend with the service_role key (bypasses RLS).
-- Ledger tables use (run_id, id) keys so ids stay short ("c1", "s2") and
-- readable inside report text ({{c1}}) and LLM prompts.

-- ---------------------------------------------------------------- tables

create table if not exists public.runs (
  id             text primary key default gen_random_uuid()::text,
  schema_version int not null default 1,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now(),
  status         text not null default 'queued' check (status in (
                   'queued','scoping','needs_input','refused','awaiting_gate1',
                   'researching','modelling','awaiting_gate2','done','failed')),
  idea_text      text not null,
  context        jsonb not null default '{}'::jsonb,
  scope          jsonb,
  interpretation jsonb,
  assessment     jsonb,
  error          text
);

create table if not exists public.steps (
  run_id      text not null references public.runs(id) on delete cascade,
  id          text not null,
  name        text not null check (name in (
                'scoping','researching','verifying','modelling','ranking','experiments','composing')),
  status      text not null check (status in ('running','done','failed','skipped')),
  message     text,
  started_at  timestamptz not null default now(),
  finished_at timestamptz,
  primary key (run_id, id)
);

create table if not exists public.sources (
  run_id           text not null references public.runs(id) on delete cascade,
  id               text not null,
  url              text not null,
  domain           text not null,
  title            text,
  credibility_tier text not null default 'unknown' check (credibility_tier in (
                     'government','peer_reviewed','industry_report','company_page','blog','unknown')),
  fetched_at       timestamptz not null default now(),
  text_hash        text,
  fetch_ok         boolean not null default true,
  error            text,
  primary key (run_id, id)
);

create table if not exists public.claims (
  run_id            text not null references public.runs(id) on delete cascade,
  id                text not null,
  text              text not null,
  type              text not null check (type in ('fact','estimate','assumption')),
  origin            text not null default 'research' check (origin in ('research','user','model')),
  driver            text check (driver in (
                      'n_target_orgs','share_need_budget','adoption_rate','price_per_org','achievable_share')),
  value             double precision,
  unit              text,
  p10               double precision,
  p50               double precision,
  p90               double precision,
  source_id         text,
  quote             text,
  derivation        jsonb,
  evidence_strength double precision not null check (evidence_strength between 0 and 1),
  sensitivity       double precision check (sensitivity between 0 and 1),
  status            text not null default 'unverified' check (status in ('verified','unverified')),
  note              text,
  primary key (run_id, id),
  -- Backstop for the label rules; Pydantic is the main check.
  constraint fact_needs_verified_quote check (
    type <> 'fact' or (source_id is not null and quote is not null and status = 'verified')),
  constraint only_facts_verified check (type = 'fact' or status = 'unverified'),
  constraint estimate_needs_derivation check (
    type <> 'estimate' or (derivation is not null and p10 is not null and p50 is not null and p90 is not null)),
  constraint user_figures_are_assumptions check (origin <> 'user' or type = 'assumption')
);

create table if not exists public.experiments (
  run_id            text not null references public.runs(id) on delete cascade,
  id                text not null,
  assumption_id     text not null,
  hypothesis        text not null,
  method            text not null check (method in (
                      'customer_interviews','landing_page_smoke_test','letter_of_intent',
                      'pricing_test','technical_benchmark','regulatory_expert_call')),
  metric            text not null,
  pass_threshold    text not null,
  kill_criterion    text not null,
  effort_days       double precision not null check (effort_days >= 0),
  cost_eur          double precision not null check (cost_eur >= 0),
  updates_claim_ids text[] not null default '{}',
  priority          double precision not null check (priority between 0 and 1),
  status            text not null default 'proposed' check (status in ('proposed','accepted','edited','rejected')),
  primary key (run_id, id)
);

create table if not exists public.approvals (
  id         text primary key default gen_random_uuid()::text,
  run_id     text not null references public.runs(id) on delete cascade,
  gate       int not null check (gate in (1, 2)),
  approved   boolean not null,
  payload    jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.eval_results (
  id          text primary key default gen_random_uuid()::text,
  created_at  timestamptz not null default now(),
  eval_run_id text not null,
  case_id     text not null,
  system      text not null check (system in ('pipeline','baseline','analyst')),
  metrics     jsonb not null default '{}'::jsonb,
  passed      boolean,
  notes       text,
  latency_s   double precision,
  cost_eur    double precision
);
create index if not exists eval_results_eval_run_id on public.eval_results (eval_run_id);

create table if not exists public.llm_calls (
  id            text primary key default gen_random_uuid()::text,
  run_id        text references public.runs(id) on delete set null,
  created_at    timestamptz not null default now(),
  purpose       text not null,
  model         text not null,
  input_tokens  int,
  output_tokens int,
  latency_ms    int,
  ok            boolean not null,
  error         text,
  request       jsonb,
  response      jsonb
);
create index if not exists llm_calls_run_id on public.llm_calls (run_id);

-- ---------------------------------------------------------------- updated_at

create or replace function public.set_updated_at() returns trigger
language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end
$$;

drop trigger if exists runs_set_updated_at on public.runs;
create trigger runs_set_updated_at before update on public.runs
  for each row execute function public.set_updated_at();

-- ---------------------------------------------------------------- access

grant usage on schema public to anon, authenticated;
grant select on public.runs, public.steps, public.sources, public.claims,
  public.experiments, public.approvals, public.eval_results to anon, authenticated;
grant all on all tables in schema public to service_role;

do $$
declare t text;
begin
  foreach t in array array['runs','steps','sources','claims','experiments','approvals','eval_results','llm_calls'] loop
    execute format('alter table public.%I enable row level security', t);
  end loop;
  -- llm_calls gets no policy: only the service_role key can read it.
  foreach t in array array['runs','steps','sources','claims','experiments','approvals','eval_results'] loop
    execute format('drop policy if exists "public read" on public.%I', t);
    execute format('create policy "public read" on public.%I for select to anon, authenticated using (true)', t);
  end loop;
end
$$;

-- ---------------------------------------------------------------- realtime

do $$
declare t text;
begin
  foreach t in array array['runs','steps','sources','claims','experiments'] loop
    if not exists (
      select 1 from pg_publication_tables
      where pubname = 'supabase_realtime' and schemaname = 'public' and tablename = t
    ) then
      execute format('alter publication supabase_realtime add table public.%I', t);
    end if;
  end loop;
end
$$;
