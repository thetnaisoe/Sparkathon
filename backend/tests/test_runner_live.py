"""A full live run through the real LLM code path, with scripted model replies instead of a key."""

from __future__ import annotations

import pytest

from app import llm, runner
from app.models import ApproveRequest, ResultCreate, RunCreate, RunContext
from app.store import RunStore

IDEA = ("A low-cost acoustic sensor that detects bearing wear in wind turbines before failure. "
        "The market is worth EUR 12 billion.")


def scope_reply(**overrides) -> dict:
    drivers = {
        "n_target_orgs": ("Wind farm operators in the EU", "organisations", 800, 1500, 3000),
        "share_need_budget": ("Operators with unplanned downtime and a maintenance budget", "share", 0.2, 0.4, 0.6),
        "adoption_rate": ("Operators that would adopt acoustic monitoring", "share", 0.05, 0.15, 0.3),
        "price_per_org": ("Annual subscription per operator", "EUR per organisation per year", 5000, 15000, 40000),
        "achievable_share": ("Share of adopters we could win", "share", 0.02, 0.08, 0.2),
    }
    reply = {
        "verdict": "in_scope",
        "reasons": ["A sensor product with identifiable B2B customers."],
        "clarifying_questions": [],
        "refusal_kind": None,
        "summary": "Acoustic bearing-wear detection for wind turbines.",
        "technology": "Acoustic emission sensing with edge ML",
        "trl_guess": 4,
        "target_customers": ["Wind farm operators"],
        "market_drivers": ["Rising O&M costs"],
        "sector": "Energy",
        "region": "EU",
        "drivers": [
            {"driver": k, "text": t, "unit": u, "p10": a, "p50": b, "p90": c,
             "basis": "analogy" if k == "adoption_rate" else "guess", "rationale": f"Why {k}.",
             "search_queries": [f"{k} statistics"]}
            for k, (t, u, a, b, c) in drivers.items()
        ],
        "user_figures": [{"text": "The market is worth EUR 12 billion", "value": 12e9, "unit": "EUR"}],
        "cannot_determine": ["Warranty terms of turbine OEMs"],
        "injection_suspected": False,
    }
    reply.update(overrides)
    return reply


def experiments_reply(system, user, schema) -> dict:
    # The runner numbers claims from the Gate 1 assumptions: c2 need, c3 adoption, c5 win share.
    plan = [("c2", "customer_interviews", 10, 5, 3, "pivot"),
            ("c3", "customer_interviews", 15, 5, 2, "kill"),
            ("c5", "letter_of_intent", 20, 3, 1, "kill")]
    return {
        "experiments": [
            {"claim_id": cid, "method": m, "hypothesis": f"Hypothesis for {cid}.",
             "metric": f"Number of the {n} contacted operators who say yes", "trials": n,
             "pass_at_least": p, "fail_below": f, "on_fail": a, "effort_days": 5, "cost_eur": 0}
            for cid, m, n, p, f, a in plan
        ],
        "viability_threshold_eur": 100000,
        "viability_reason": "Two founders need about this much revenue to keep going.",
    }


@pytest.fixture
def store(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    yield RunStore()
    llm.set_client(None)


def use(replies: dict) -> llm.FakeProvider:
    provider = llm.FakeProvider(replies)
    llm.set_client(llm.LLMClient(provider))
    return provider


def test_live_run_from_idea_to_learning(store):
    use({"ScopeReply": [scope_reply()], "ExperimentsReply": experiments_reply})
    record = runner.create_run(store, RunCreate(idea_text=IDEA, context=RunContext(region="EU")), background=False)
    state = record.snapshot()
    assert state.run.status == "awaiting_gate1", state.run.error
    assert [a.driver for a in state.run.interpretation.assumptions][0] == "n_target_orgs"

    runner.approve(record, ApproveRequest(gate=1, interpretation=state.run.interpretation), background=False)
    state = record.snapshot()
    assert state.run.status == "awaiting_gate2", state.run.error
    a = state.run.assessment
    assert a.decision.viability_threshold == 100000
    assert not any("standard template" in s for s in a.limitations)
    assert any("No web research" in s for s in a.limitations)
    assert a.market_model.top_down_check is not None and a.market_model.top_down_check.ratio > 1
    user_claim = next(c for c in state.claims if c.origin == "user")
    assert user_claim.type == "assumption" and user_claim.value == 12e9
    assert {x.assumption_id for x in state.experiments} == {"c2", "c3", "c5"}
    assert all(x.pass_threshold.endswith(f"of {x.rule.trials}") for x in state.experiments)

    runner.approve(record, ApproveRequest(gate=2, decisions=[
        {"experiment_id": x.id, "decision": "accept"} for x in state.experiments]))
    assert record.snapshot().run.status == "done"

    x_need = next(x for x in record.snapshot().experiments if x.assumption_id == "c2")
    runner.add_result(record, ResultCreate(experiment_id=x_need.id, successes=1, trials=10, mocked=True))
    state = record.snapshot()
    event = state.updates[-1]
    assert event.outcome == "fail" and event.verdict_after == "pivot"
    assert event.posterior.p50 < event.prior.p50


def test_scope_refusal_and_questions(store):
    use({"ScopeReply": [scope_reply(verdict="refuse", refusal_kind="harmful_or_dual_use",
                                    reasons=["Covert tracking of people."], drivers=[], user_figures=[])]})
    record = runner.create_run(store, RunCreate(idea_text="Track protesters with drones."), background=False)
    assert record.snapshot().run.status == "refused"

    use({"ScopeReply": [scope_reply(verdict="needs_info", reasons=["Too vague."],
                                    clarifying_questions=["Who pays?"], drivers=[], user_figures=[])]})
    record = runner.create_run(store, RunCreate(idea_text="An AI app for farmers."), background=False)
    assert record.snapshot().run.status == "needs_input"


def test_provider_refusal_becomes_a_refusal(store):
    use({"ScopeReply": [llm.LLMRefused("declined")]})
    record = runner.create_run(store, RunCreate(idea_text="Something the provider declines."), background=False)
    state = record.snapshot()
    assert state.run.status == "refused"
    assert state.run.scope.refusal_kind == "harmful_or_dual_use"


def test_experiment_llm_failure_falls_back_to_labelled_templates(store):
    use({"ScopeReply": [scope_reply()], "ExperimentsReply": [llm.LLMError("timeout")]})
    record = runner.create_run(store, RunCreate(idea_text=IDEA), background=False)
    runner.approve(record, ApproveRequest(gate=1, interpretation=record.snapshot().run.interpretation),
                   background=False)
    state = record.snapshot()
    assert state.run.status == "awaiting_gate2", state.run.error
    assert any("standard template" in s for s in state.run.assessment.limitations)


def test_scope_llm_error_fails_the_run_with_a_message(store):
    use({"ScopeReply": [llm.LLMError("provider down")]})
    record = runner.create_run(store, RunCreate(idea_text=IDEA), background=False)
    state = record.snapshot()
    assert state.run.status == "failed" and "provider down" in state.run.error
