"""FastAPI entry point.

Stage 0: stub mode. GET /runs/{id} serves the fixtures in /fixtures so Dev A can
wire the UI against the real URL. Stage 1 replaces the stubs with the pipeline
runner and Supabase reads.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    SCHEMA_VERSION,
    ApproveRequest,
    EvalResult,
    RunCreate,
    RunCreated,
    RunState,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
FIXTURE_FILES = ("gate1.json", "assessment.json", "needs_input.json", "refused.json")


def load_fixtures() -> dict[str, RunState]:
    states: dict[str, RunState] = {}
    for name in FIXTURE_FILES:
        state = RunState.model_validate_json((FIXTURES_DIR / name).read_text(encoding="utf-8"))
        states[state.run.id] = state
    return states


FIXTURES = load_fixtures()

app = FastAPI(title="Lab-to-Market API", version=str(SCHEMA_VERSION))

# No cookies or auth headers are used, so a wildcard origin is safe and avoids
# CORS surprises with Lovable preview domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "schema_version": SCHEMA_VERSION, "mode": "stub"}


@app.post("/runs", response_model=RunCreated, status_code=202)
def create_run(body: RunCreate) -> RunCreated:
    # Stub: no pipeline yet. Points the UI at the Gate 1 fixture.
    return RunCreated(run_id="demo-gate1", status="awaiting_gate1")


@app.get("/runs/{run_id}", response_model=RunState)
def get_run(run_id: str) -> RunState:
    state = FIXTURES.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return state


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, body: ApproveRequest) -> dict:
    if run_id not in FIXTURES:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    # Stub: accepts and validates the payload but does not resume anything yet.
    return {"ok": True, "run_id": run_id, "gate": body.gate, "approved": body.approved}


@app.get("/evals/latest", response_model=list[EvalResult])
def latest_evals() -> list[EvalResult]:
    return []
