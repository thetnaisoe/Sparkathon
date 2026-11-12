from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_run_returns_run_id():
    response = client.post("/runs", json={"idea_text": "A drone that inspects wind-turbine blades."})
    assert response.status_code == 202
    assert response.json() == {"run_id": "demo-gate1", "status": "awaiting_gate1"}


def test_create_run_rejects_empty_idea():
    assert client.post("/runs", json={"idea_text": ""}).status_code == 422


def test_get_each_fixture():
    for run_id in ("demo-gate1", "demo-success", "demo-needs-input", "demo-refused"):
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200, run_id
        assert response.json()["run"]["id"] == run_id


def test_unknown_run_is_404():
    assert client.get("/runs/nope").status_code == 404


def test_approve_gate2():
    body = {"gate": 2, "decisions": [{"experiment_id": "x1", "decision": "accept"}]}
    response = client.post("/runs/demo-success/approve", json=body)
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_approve_rejects_bad_payload():
    response = client.post("/runs/demo-gate1/approve", json={"gate": 1, "approved": True})
    assert response.status_code == 422


def test_cors_allows_browser_origin():
    response = client.options(
        "/runs",
        headers={"Origin": "https://example.lovable.app", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
