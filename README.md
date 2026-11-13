# Sparkathon 2026 · Topic 8 "From Lab to Market"

Backend for the claim-ledger assessment tool. The Lovable app lives in its own repo; its contract is in [CONTRACT.md](CONTRACT.md).

```
backend/app/models.py      contract (Pydantic) – single source of truth
backend/app/main.py        FastAPI app (runs, gates, results, demo reset, evals)
backend/app/pipeline/      scope, research, model, learning loop, experiments, composer
backend/evals/             code-only evidence + LLM baseline (run_evals.py)
fixtures/                  demo runs + generated schema.json
sql/schema.sql             legacy v1 Supabase schema (storage is in-memory since v2)
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
.venv/bin/python scripts/build_fixtures.py    # recompute the demo runs with the real code
.venv/bin/python scripts/export_contract.py   # schema.json, lovable/src/lib/{contract,fixtures}.ts
.venv/bin/python evals/run_evals.py           # code-only evidence (add --llm when a key is set)
```

Then send Dev A the new `lovable/src/lib/*.ts`.

## LLM key

Set on Render (and in `backend/.env` locally): `LLM_PROVIDER` (`anthropic` or `openai` for any OpenAI-compatible API), `LLM_API_KEY`, optional `LLM_MODEL` / `LLM_BASE_URL`, and `TAVILY_API_KEY` for web research. Check with `.venv/bin/python scripts/llm_smoke.py`. Without a key, scoping fails with a clear message; the demo runs and the learning loop still work.

## Deploy (Render)

New → Blueprint → this repo; `render.yaml` sets build/start commands and `/health`. Python version comes from `.python-version`.
