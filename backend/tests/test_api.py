"""API and state machine, without an LLM key (the offline paths must work end to end)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import app, store

client = TestClient(app)
FINAL = {"needs_input", "refused", "awaiting_gate1", "awaiting_gate2", "done", "failed"}


@pytest.fixture(autouse=True)
def fresh_demo(monkeypatch):
    for key in ("LLM_PROVIDER", "LLM_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    llm.set_client(None)
    store.load_demo()
    yield
    llm.set_client(None)


def wait(run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["run"]["status"] in FINAL:
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not settle")


def test_health_reports_configuration():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["schema_version"] == 2
    assert body["llm_configured"] is False and body["runs"] >= 5


@pytest.mark.parametrize("run_id", ["demo-gate1", "demo-success", "demo-updated", "demo-needs-input", "demo-refused"])
def test_demo_runs_are_served(run_id):
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["run"]["id"] == run_id


def test_unknown_run_is_404():
    assert client.get("/runs/nope").status_code == 404


def test_create_run_rejects_empty_idea():
    assert client.post("/runs", json={"idea_text": ""}).status_code == 422


def test_new_run_without_llm_fails_with_a_clear_message():
    created = client.post("/runs", json={"idea_text": "A drone that inspects wind-turbine blades."})
    assert created.status_code == 202
    body = wait(created.json()["run_id"])
    assert body["run"]["status"] == "failed"
    assert "not configured" in body["run"]["error"]


def test_gate2_then_results_flip_the_verdict():
    decisions = [{"experiment_id": x, "decision": "accept"} for x in ("x1", "x2", "x3")]
    approved = client.post("/runs/demo-success/approve", json={"gate": 2, "decisions": decisions})
    assert approved.status_code == 200
    assert approved.json()["run"]["status"] == "done"
    assert all(x["locked_at"] for x in approved.json()["experiments"])

    result = client.post("/runs/demo-success/results",
                         json={"experiment_id": "x1", "successes": 2, "trials": 10, "mocked": True})
    assert result.status_code == 200
    body = result.json()
    event = body["updates"][-1]
    assert (event["verdict_before"], event["verdict_after"]) == ("continue", "pivot")
    assert body["run"]["assessment"]["decision"]["verdict"] == "pivot"


def test_results_before_gate2_are_refused():
    response = client.post("/runs/demo-success/results",
                           json={"experiment_id": "x1", "successes": 2, "trials": 10, "mocked": True})
    assert response.status_code == 409
    assert "Gate 2" in response.json()["detail"]


def test_results_must_say_whether_they_are_mocked():
    response = client.post("/runs/demo-updated/results", json={"experiment_id": "x2", "successes": 2, "trials": 15})
    assert response.status_code == 422


def test_gate_requests_must_match_status():
    assert client.post("/runs/demo-success/approve", json={
        "gate": 1, "approved": False, "note": "x"}).status_code == 409
    assert client.post("/runs/demo-refused/approve", json={
        "gate": 2, "decisions": [{"experiment_id": "x1", "decision": "accept"}]}).status_code == 409


def test_gate2_needs_every_decision():
    response = client.post("/runs/demo-success/approve",
                           json={"gate": 2, "decisions": [{"experiment_id": "x1", "decision": "accept"}]})
    assert response.status_code == 409


def test_demo_reset_restores_fixtures():
    decisions = [{"experiment_id": x, "decision": "accept"} for x in ("x1", "x2", "x3")]
    client.post("/runs/demo-success/approve", json={"gate": 2, "decisions": decisions})
    assert client.get("/runs/demo-success").json()["run"]["status"] == "done"
    client.post("/demo/reset")
    assert client.get("/runs/demo-success").json()["run"]["status"] == "awaiting_gate2"


def test_gate1_to_gate2_offline_uses_templates_and_says_so():
    gate1 = client.get("/runs/demo-gate1").json()
    response = client.post("/runs/demo-gate1/approve",
                           json={"gate": 1, "approved": True, "interpretation": gate1["run"]["interpretation"]})
    assert response.status_code == 200
    body = wait("demo-gate1")
    assert body["run"]["status"] == "awaiting_gate2", body["run"]["error"]
    a = body["run"]["assessment"]
    assert any("standard template" in s for s in a["limitations"])
    assert any("No web research was run" in s for s in a["limitations"])
    assert a["evidence_floor_met"] is False and a["headline"] is None and a["data_needed"]
    assert 1 <= len(body["experiments"]) <= 3
    steps = {s["name"]: s["status"] for s in body["steps"]}
    assert steps["researching"] == "skipped" and steps["composing"] == "done"


def test_gate1_rejects_missing_driver():
    gate1 = client.get("/runs/demo-gate1").json()
    interp = gate1["run"]["interpretation"]
    interp["assumptions"] = interp["assumptions"][:4]
    response = client.post("/runs/demo-gate1/approve", json={"gate": 1, "approved": True, "interpretation": interp})
    assert response.status_code == 409


def test_cors_allows_browser_origin():
    response = client.options(
        "/runs",
        headers={"Origin": "https://example.lovable.app", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
