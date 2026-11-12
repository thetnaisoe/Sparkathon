# Sparkathon 2026 · Topic 8 "From Lab to Market"

Backend for the claim-ledger assessment tool. The Lovable app lives in its own repo; its contract is in [CONTRACT.md](CONTRACT.md).

```
backend/app/models.py      contract (Pydantic) – single source of truth
backend/app/main.py        FastAPI app (stub mode: serves fixtures)
fixtures/                  demo runs + generated schema.json
sql/schema.sql             Supabase tables, RLS, Realtime
sql/seed_fixtures.sql      generated: loads the demo runs into Supabase
lovable/                   files + prompt sequence for Dev A (contract.ts, fixtures.ts generated)
```

## Run locally

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --reload        # http://127.0.0.1:8000/docs
```

After changing `models.py` or a fixture, regenerate the contract files:

```bash
.venv/bin/python scripts/export_contract.py   # schema.json, seed_fixtures.sql, lovable/src/lib/{contract,fixtures}.ts
```

Then send Dev A the new `lovable/src/lib/*.ts`.

## Supabase

In the SQL editor, run `sql/schema.sql`, then `sql/seed_fixtures.sql`. Both are safe to re-run.

## Deploy (Render)

New → Blueprint → this repo; `render.yaml` sets build/start commands and `/health`. Python version comes from `.python-version`.
