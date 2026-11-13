"""Build the demo fixtures from hand-written inputs, computing every derived number with the real code.

    cd backend && .venv/bin/python scripts/build_fixtures.py

Writes fixtures/{gate1,assessment,updated,needs_input,refused}.json. Every number
in them is invented demo data (sources use .example domains), but the model
output, tornado, verdict, ranks and the update are computed, not typed.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.models import (  # noqa: E402
    SCHEMA_VERSION,
    ApproveRequest,
    ExperimentRule,
    ResultCreate,
    RunState,
)
from app.pipeline.assess import refresh  # noqa: E402
from app.pipeline.gates import approve_plan  # noqa: E402
from app.pipeline.learn import apply_result  # noqa: E402
from app.pipeline.rules import describe_rule  # noqa: E402

FIXTURES = ROOT / "fixtures"
VIABILITY_THRESHOLD = 150_000.0  # EUR per year, the demo's pre-registered threshold

IDEA = ("We built a drone that inspects wind-turbine blades with thermal and RGB cameras and detects "
        "defects with an on-board AI model, so operators no longer need rope-access teams. "
        "We think the market is worth EUR 5 billion.")


def ts(minute: int, second: int = 0) -> str:
    return datetime(2026, 11, 12, 18, minute, second, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


SCOPE = {
    "verdict": "in_scope",
    "reasons": ["A concrete technology with identifiable B2B customers (wind farm operators and O&M providers)."],
    "clarifying_questions": [],
    "refusal_kind": None,
}

INTERPRETATION = {
    "summary": ("Autonomous drones with thermal and RGB cameras inspect wind-turbine blades and flag defects "
                "with an on-board AI model, replacing rope-access inspection teams."),
    "technology": "Drone-mounted thermal/RGB imaging with on-board defect classification",
    "trl_guess": 4,
    "target_customers": ["Wind farm operators", "Independent operations and maintenance (O&M) providers"],
    "market_drivers": [
        "Ageing onshore wind fleet needs more frequent inspection",
        "Shortage and cost of rope-access technicians",
        "Pressure to cut turbine downtime",
    ],
    "sector": "energy",
    "region": "EU",
    "assumptions": [
        {"id": "a1", "text": "Wind farm operators and independent O&M providers in the EU", "driver": "n_target_orgs",
         "included": True, "unit": "organisations", "p10": 1200, "p50": 2000, "p90": 3000},
        {"id": "a2", "text": "Share of them with both the need and a budget for automated blade inspection",
         "driver": "share_need_budget", "included": True, "unit": "share", "p10": 0.3, "p50": 0.5, "p90": 0.7},
        {"id": "a3", "text": "Share adopting drone-based blade inspection within five years",
         "driver": "adoption_rate", "included": True, "unit": "share", "p10": 0.1, "p50": 0.25, "p90": 0.45},
        {"id": "a4", "text": "Annual price per organisation", "driver": "price_per_org", "included": True,
         "unit": "EUR per organisation per year", "p10": 8000, "p50": 20000, "p90": 45000},
        {"id": "a5", "text": "Share of adopting organisations we can win within five years",
         "driver": "achievable_share", "included": True, "unit": "share", "p10": 0.02, "p50": 0.05, "p90": 0.1},
    ],
}

SOURCES = [
    {"id": "s1", "url": "https://windreport.example/fixture/annual-statistics", "domain": "windreport.example",
     "title": "(fixture) Wind energy in Europe: annual statistics", "credibility_tier": "industry_report",
     "fetched_at": ts(1, 20), "text_hash": "sha256:fixture-s1", "fetch_ok": True, "error": None},
    {"id": "s2", "url": "https://stats.example/fixture/electricity-enterprises", "domain": "stats.example",
     "title": "(fixture) Enterprises in electricity generation by activity", "credibility_tier": "government",
     "fetched_at": ts(1, 25), "text_hash": "sha256:fixture-s2", "fetch_ok": True, "error": None},
    {"id": "s3", "url": "https://droneblog.example/fixture/blade-inspection-cost", "domain": "droneblog.example",
     "title": "(fixture) What does a blade inspection cost?", "credibility_tier": "blog",
     "fetched_at": ts(1, 31), "text_hash": "sha256:fixture-s3", "fetch_ok": True, "error": None},
    {"id": "s4", "url": "https://paywalled.example/fixture/om-market-study", "domain": "paywalled.example",
     "title": None, "credibility_tier": "unknown", "fetched_at": ts(1, 40), "text_hash": None,
     "fetch_ok": False, "error": "Timed out after 8 s"},
]


def claim(cid: str, text: str, ctype: str, **kw) -> dict:
    base = {"id": cid, "text": text, "type": ctype, "origin": "research", "driver": None, "value": None,
            "unit": None, "p10": None, "p50": None, "p90": None, "source_id": None, "quote": None,
            "derivation": None, "evidence_strength": 0.1, "sensitivity": None, "status": "unverified",
            "note": None}
    base.update(kw)
    return base


CLAIMS = [
    claim("c1", "About 2,000 enterprises operate or maintain wind generation assets in the EU.", "fact",
          value=2000, unit="organisations", source_id="s2", status="verified", evidence_strength=0.7,
          quote="around 2,000 enterprises operate or maintain wind generation assets in the EU"),
    claim("c2", "A manual rope-access blade inspection costs around EUR 1,000 per turbine.", "fact",
          value=1000, unit="EUR per turbine", source_id="s3", status="verified", evidence_strength=0.4,
          quote="a rope-access team typically charges around EUR 1,000 per turbine for a blade inspection",
          note="Blog source: evidence strength capped at 0.4."),
    claim("c3", "An operator's portfolio comprises about 30 turbines on average.", "fact",
          value=30, unit="turbines per organisation", source_id="s1", status="verified", evidence_strength=0.7,
          quote="the average portfolio comprises about 30 turbines"),
    claim("c4", "Target organisations in the EU (operators and O&M providers).", "estimate", origin="model",
          driver="n_target_orgs", unit="organisations", p10=1500, p50=2000, p90=2600, evidence_strength=0.7,
          derivation={"formula": "c1 × [0.75, 1.0, 1.3] (operators and O&M providers overlap)",
                      "input_claim_ids": ["c1"]}),
    claim("c5", "Annual price per organisation.", "estimate", origin="model", driver="price_per_org",
          unit="EUR per organisation per year", p10=12000, p50=20000, p90=36000, evidence_strength=0.1,
          derivation={"formula": "c2 × c3 × c8", "input_claim_ids": ["c2", "c3", "c8"]}),
    claim("c6", "Share of target organisations with both the need and a budget for automated blade inspection.",
          "assumption", origin="model", driver="share_need_budget", unit="share", p10=0.3, p50=0.5, p90=0.7,
          evidence_strength=0.1, note="Guess: no source found for budgets."),
    claim("c7", "Share of target organisations that adopt drone-based blade inspection within five years.",
          "assumption", origin="model", driver="adoption_rate", unit="share", p10=0.1, p50=0.25, p90=0.45,
          evidence_strength=0.25, note="Analogy: drone adoption in power-line inspection."),
    claim("c8", "Customers pay this share of their current manual inspection spend.", "assumption",
          origin="model", unit="share of current spend", p10=0.5, p50=0.67, p90=1.0, evidence_strength=0.1),
    claim("c9", "Share of adopting organisations we can win within five years.", "assumption", origin="model",
          driver="achievable_share", unit="share", p10=0.02, p50=0.05, p90=0.1, evidence_strength=0.1),
    claim("c10", "The market is worth EUR 5 billion (stated in the idea).", "assumption", origin="user",
          value=5_000_000_000, unit="EUR", evidence_strength=0.1, note="User-supplied figure; see the top-down check."),
    claim("c12", "Drone inspection cuts turbine downtime during inspection by about 70%.", "assumption",
          value=0.7, unit="share", source_id="s3", quote="reduces downtime by up to 70%", evidence_strength=0.1,
          note="Demoted from fact: this quote was not found on the fetched page."),
]

EXPERIMENTS = [
    ("x1", "c6", "customer_interviews",
     "At least half of interviewed operators and O&M providers have both the need and a budget line for automated blade inspection.",
     "Interviewed target organisations with need and budget, out of 10",
     ExperimentRule(trials=10, pass_at_least=5, fail_below=3, on_fail="pivot"), 7, 150),
    ("x2", "c7", "customer_interviews",
     "Operators plan to switch from rope access to drone inspection within three years.",
     "Interviewed operators with a budgeted switch plan, out of 15",
     ExperimentRule(trials=15, pass_at_least=6, fail_below=3, on_fail="kill"), 7, 0),
    ("x3", "c9", "letter_of_intent",
     "Adopting O&M providers sign a non-binding letter of intent for a paid pilot.",
     "Signed letters of intent, out of 20 providers contacted",
     ExperimentRule(trials=20, pass_at_least=3, fail_below=1, on_fail="kill"), 10, 0),
]

STEP_PLAN = [
    ("st1", "scoping", "In scope: wind-turbine blade inspection drones", (0, 0), (0, 12)),
    ("st2", "researching", "6 searches, 3 of 4 pages fetched (1 timed out)", (1, 0), (2, 10)),
    ("st3", "verifying", "3 of 4 quotes verified; 1 claim demoted to assumption", (2, 10), (2, 20)),
    ("st4", "modelling", "5,000 Monte Carlo draws, seed 42", (2, 20), (2, 25)),
    ("st5", "ranking", "Weakest testable assumptions ranked", (2, 25), (2, 26)),
    ("st6", "experiments", "3 experiment cards with rules fixed in advance", (2, 26), (3, 0)),
    ("st7", "composing", "Report composed from the ledger", (3, 0), (3, 40)),
]


def steps(run_id: str, upto: int) -> list[dict]:
    return [{"id": sid, "run_id": run_id, "name": name, "status": "done", "message": msg,
             "started_at": ts(*a), "finished_at": ts(*b)} for sid, name, msg, a, b in STEP_PLAN[:upto]]


def base_run(run_id: str, status: str) -> dict:
    return {"id": run_id, "schema_version": SCHEMA_VERSION, "created_at": ts(0), "updated_at": ts(3, 40),
            "status": status, "idea_text": IDEA, "context": {"sector": "energy", "region": "EU", "trl": 4},
            "scope": SCOPE, "interpretation": INTERPRETATION, "assessment": None, "error": None}


def with_run_id(data: dict, run_id: str) -> dict:
    data = copy.deepcopy(data)
    data["run"]["id"] = run_id
    for table in ("steps", "sources", "claims", "experiments", "updates"):
        for row in data.get(table, []):
            row["run_id"] = run_id
    return data


def build_success() -> RunState:
    rid = "demo-success"
    experiments = []
    for xid, target, method, hyp, metric, rule, effort, cost in EXPERIMENTS:
        pass_text, kill_text = describe_rule(rule)
        experiments.append({"id": xid, "run_id": rid, "assumption_id": target, "hypothesis": hyp, "method": method,
                            "metric": metric, "rule": rule.model_dump(), "pass_threshold": pass_text,
                            "kill_criterion": kill_text, "effort_days": effort, "cost_eur": cost,
                            "updates_claim_ids": [target, "c11"], "priority": 0.0, "status": "proposed"})
    raw = {
        "run": {**base_run(rid, "modelling")},
        "steps": steps(rid, 7),
        "sources": [{**s, "run_id": rid} for s in SOURCES],
        "claims": [{**c, "run_id": rid} for c in CLAIMS],
        "experiments": [],
        "updates": [],
    }
    # Same order as the pipeline: model first, then experiments on the weakest shares, then refresh.
    state = RunState.model_validate(raw)
    refresh(state, threshold=VIABILITY_THRESHOLD, output_claim_id="c11",
            cannot_determine=["How beyond-visual-line-of-sight (BVLOS) drone rules will develop in Germany",
                              "Prices of competing robotic blade-crawler services"],
            extra_limitations=["Only public web pages were used; no paid databases or interviews."],
            fit_claims=True)
    state = RunState.model_validate({**state.model_dump(mode="json"), "experiments": experiments})
    refresh(state)
    state.run.status = "awaiting_gate2"
    return RunState.model_validate(state.model_dump())


def build_updated(success: RunState) -> RunState:
    state = RunState.model_validate(with_run_id(success.model_dump(mode="json"), "demo-updated"))
    when = datetime(2026, 11, 12, 18, 10, tzinfo=timezone.utc)
    approve_plan(state, ApproveRequest(gate=2, decisions=[
        {"experiment_id": x.id, "decision": "accept"} for x in state.experiments]), now=when)
    apply_result(state, ResultCreate(experiment_id="x1", successes=2, trials=10, mocked=True,
                                     note="Mocked demo value: 2 of 10 interviewees have need and budget."),
                 now=datetime(2026, 11, 12, 18, 20, tzinfo=timezone.utc))
    refresh(state)
    state.run.updated_at = datetime(2026, 11, 12, 18, 20, tzinfo=timezone.utc)
    return RunState.model_validate(state.model_dump())


def build_gate1() -> RunState:
    raw = {"run": {**base_run("demo-gate1", "awaiting_gate1"), "updated_at": ts(0, 12)},
           "steps": steps("demo-gate1", 1)}
    return RunState.model_validate(raw)


def bump(name: str) -> RunState:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    data["run"]["schema_version"] = SCHEMA_VERSION
    data.setdefault("updates", [])
    return RunState.model_validate(data)


def write(name: str, state: RunState) -> None:
    (FIXTURES / name).write_text(json.dumps(state.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    print(f"wrote fixtures/{name}")


if __name__ == "__main__":
    success = build_success()
    write("gate1.json", build_gate1())
    write("assessment.json", success)
    write("updated.json", build_updated(success))
    write("needs_input.json", bump("needs_input.json"))
    write("refused.json", bump("refused.json"))
