"""FastAPI entry point.

Live runs go through the pipeline (app/runner.py). The demo runs from
/fixtures are loaded at start-up so the UI always has data to show; reset them
with POST /demo/reset.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .env import load_env

load_env()

from . import llm, runner  # noqa: E402
from .models import (  # noqa: E402
    SCHEMA_VERSION,
    ApproveRequest,
    EvalResult,
    ResultCreate,
    RunCreate,
    RunCreated,
    RunState,
)
from .store import RunRecord, RunStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

EVALS_FILE = Path(__file__).resolve().parents[1] / "evals" / "results" / "latest.json"

store = RunStore()
DEMO_IDS = store.load_demo()

app = FastAPI(title="Lab-to-Market API", version=str(SCHEMA_VERSION))

# No cookies or auth headers are used, so a wildcard origin is safe and avoids
# CORS surprises with Lovable preview domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _record(run_id: str) -> RunRecord:
    record = store.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return record


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "llm_configured": llm.is_configured(),
        "search_configured": bool(os.getenv("TAVILY_API_KEY", "").strip()),
        "runs": store.count(),
    }


@app.post("/runs", response_model=RunCreated, status_code=202)
def create_run(body: RunCreate) -> RunCreated:
    record = runner.create_run(store, body)
    return RunCreated(run_id=record.state.run.id, status=record.state.run.status)


@app.get("/runs/{run_id}", response_model=RunState)
def get_run(run_id: str) -> RunState:
    return _record(run_id).snapshot()


@app.post("/runs/{run_id}/approve", response_model=RunState)
def approve(run_id: str, body: ApproveRequest) -> RunState:
    record = _record(run_id)
    try:
        runner.approve(record, body)
    except runner.RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.snapshot()


@app.post("/runs/{run_id}/results", response_model=RunState)
def add_result(run_id: str, body: ResultCreate) -> RunState:
    record = _record(run_id)
    try:
        runner.add_result(record, body)
    except runner.RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.snapshot()


@app.post("/demo/reset")
def reset_demo() -> dict:
    return {"reset": store.load_demo()}


@app.get("/evals/latest", response_model=list[EvalResult])
def latest_evals() -> list[EvalResult]:
    if not EVALS_FILE.exists():
        return []
    return [EvalResult.model_validate(row) for row in json.loads(EVALS_FILE.read_text(encoding="utf-8"))]
