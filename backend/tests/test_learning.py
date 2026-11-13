"""Code-only core: distributions, Monte Carlo, rules, verdict, updates, re-ranking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from app.models import ApproveRequest, ExperimentRule, ResultCreate, RunState
from app.pipeline.assess import refresh
from app.pipeline.beliefs import Beta, SplitLognormal, fit_beta
from app.pipeline.gates import GateError, approve_plan
from app.pipeline.learn import ResultError, apply_result, verdict_for
from app.pipeline.model import run_model
from app.pipeline.rules import describe_rule, evaluate_rule

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
NOW = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)


def demo(name: str = "assessment.json") -> RunState:
    return RunState.model_validate_json((FIXTURES / name).read_text(encoding="utf-8"))


def approved_demo() -> RunState:
    state = demo()
    approve_plan(state, ApproveRequest(gate=2, decisions=[
        {"experiment_id": x.id, "decision": "accept"} for x in state.experiments]), now=NOW)
    return state


def enter(state: RunState, xid: str, k: int, n: int, mocked: bool = True) -> RunState:
    apply_result(state, ResultCreate(experiment_id=xid, successes=k, trials=n, mocked=mocked), now=NOW)
    refresh(state)
    return RunState.model_validate(state.model_dump())  # every invariant still holds


# ------------------------------------------------------------------ distributions


@pytest.mark.parametrize("rng", [(0.3, 0.5, 0.7), (0.1, 0.25, 0.45), (0.02, 0.05, 0.1), (0.6, 0.8, 0.95)])
def test_beta_fit_reproduces_stated_range(rng):
    # A Beta has two parameters, so a skewed three-point range is matched as closely as possible;
    # the ledger then shows the fitted range (assess.fit_share_claims), never the unfitted one.
    q = fit_beta(*rng).quantiles()
    assert np.allclose(q, rng, atol=0.02)


def test_assumption_prior_is_capped_at_twenty_observations():
    narrow = fit_beta(0.25, 0.3, 0.35)
    capped = fit_beta(0.25, 0.3, 0.35, max_strength=20)
    assert narrow.strength > 20
    assert capped.strength == pytest.approx(20)
    assert capped.mean() == pytest.approx(narrow.mean())
    lo, _, hi = capped.quantiles()
    assert lo < 0.25 and hi > 0.35  # widened, same centre


def test_split_lognormal_hits_all_three_quantiles():
    s = SplitLognormal(1500, 2000, 2600)
    draws = s.ppf(np.random.default_rng(0).random(200_000))
    assert np.allclose(np.percentile(draws, [10, 50, 90]), [1500, 2000, 2600], rtol=0.01)


def test_beta_binomial_update_is_conjugate():
    assert Beta(2, 3).update(4, 10) == Beta(6, 9)
    with pytest.raises(ValueError):
        Beta(2, 3).update(11, 10)


# ------------------------------------------------------------------ Monte Carlo


def test_monte_carlo_is_reproducible():
    a, b = run_model(demo()), run_model(demo())
    assert np.array_equal(a.output, b.output)
    assert [bar.driver for bar in a.tornado] == [bar.driver for bar in b.tornado]


def test_tornado_is_sorted_and_sensitivities_normalised():
    state = demo()
    swings = [b.swing for b in state.run.assessment.market_model.tornado]
    assert swings == sorted(swings, reverse=True)
    sens = [c.sensitivity for c in state.claims if c.driver]
    assert max(sens) == pytest.approx(1.0)


def test_output_claim_matches_monte_carlo():
    state = demo()
    out = next(c for c in state.claims if c.id == state.run.assessment.market_model.output_claim_id)
    r = run_model(state).output_range
    assert (out.p10, out.p50, out.p90) == pytest.approx((r.p10, r.p50, r.p90))


# ------------------------------------------------------------------ rules and verdict


RULE = ExperimentRule(trials=10, pass_at_least=5, fail_below=3, on_fail="pivot")


@pytest.mark.parametrize("k,n,expected", [
    (2, 10, "fail"), (3, 10, "inconclusive"), (4, 10, "inconclusive"), (5, 10, "pass"),
    (5, 6, "pass"),        # already reached the pass count before all trials
    (0, 9, "fail"),        # cannot reach fail_below any more
    (1, 7, "pending"),     # still undecided mid-sample
    (0, 0, "pending"),
    (4, 20, "fail"),       # beyond the planned sample, compared as rates (20% < 30%)
    (10, 20, "pass"),
])
def test_rule_evaluation(k, n, expected):
    assert evaluate_rule(RULE, k, n) == expected


def test_rule_text_comes_from_the_rule():
    assert describe_rule(RULE) == ("5 or more of 10", "Fewer than 3 of 10 → pivot")


def test_verdict_order():
    assert verdict_for({"a": "fail", "b": "fail"}, {"a": "pivot", "b": "kill"}, 0.9) == "kill"
    assert verdict_for({"a": "fail"}, {"a": "pivot"}, 0.1) == "pivot"
    assert verdict_for({"a": "pass"}, {"a": "kill"}, 0.4) == "kill"
    assert verdict_for({"a": "pending"}, {"a": "kill"}, 0.5) == "continue"


# ------------------------------------------------------------------ the learning loop


def test_demo_starts_provisional():
    d = demo().run.assessment.decision
    assert d.provisional and d.verdict == "continue"


def test_results_need_gate_two():
    state = demo()
    with pytest.raises(ResultError, match="Gate 2"):
        apply_result(state, ResultCreate(experiment_id="x1", successes=2, trials=10, mocked=True), now=NOW)


def test_rejected_experiment_takes_no_results():
    state = demo()
    approve_plan(state, ApproveRequest(gate=2, decisions=[
        {"experiment_id": "x1", "decision": "reject"},
        {"experiment_id": "x2", "decision": "accept"},
        {"experiment_id": "x3", "decision": "accept"}]), now=NOW)
    with pytest.raises(ResultError, match="not approved"):
        apply_result(state, ResultCreate(experiment_id="x1", successes=2, trials=10, mocked=True), now=NOW)


def test_gate_two_needs_a_decision_for_every_experiment():
    state = demo()
    with pytest.raises(GateError, match="Missing"):
        approve_plan(state, ApproveRequest(gate=2, decisions=[{"experiment_id": "x1", "decision": "accept"}]),
                     now=NOW)


def test_gate_two_edit_rewrites_card_text_and_locks():
    state = demo()
    new_rule = {"trials": 12, "pass_at_least": 6, "fail_below": 4, "on_fail": "kill"}
    approve_plan(state, ApproveRequest(gate=2, viability_threshold=200000, decisions=[
        {"experiment_id": "x1", "decision": "edit", "rule": new_rule},
        {"experiment_id": "x2", "decision": "accept"},
        {"experiment_id": "x3", "decision": "accept"}]), now=NOW)
    x1 = state.experiments[0]
    assert x1.status == "edited" and x1.locked_at == NOW
    assert (x1.pass_threshold, x1.kill_criterion) == ("6 or more of 12", "Fewer than 4 of 12 → kill")
    assert state.run.assessment.decision.viability_threshold == 200000
    assert state.run.status == "done"


def test_negative_result_flips_verdict_and_narrows_range():
    state = enter(approved_demo(), "x1", 2, 10)
    event = state.updates[-1]
    assert event.outcome == "fail"
    assert (event.verdict_before, event.verdict_after) == ("continue", "pivot")
    assert event.posterior.p50 < event.prior.p50
    assert event.output_after.p50 < event.output_before.p50
    assert (event.output_after.p90 - event.output_after.p10) < (event.output_before.p90 - event.output_before.p10)
    target = next(c for c in state.claims if c.id == event.claim_id)
    obs = next(c for c in state.claims if c.id == event.observation_claim_id)
    assert target.type == "estimate" and target.derivation.input_claim_ids == [obs.id]
    assert obs.type == "assumption" and obs.origin == "experiment" and "MOCKED" in obs.text
    assert state.run.assessment.decision.provisional is False


def test_finished_experiment_leaves_ranking():
    state = enter(approved_demo(), "x1", 2, 10)
    x1 = next(x for x in state.experiments if x.id == "x1")
    assert x1.rank is None and x1.decision_flip_prob is None
    ranks = sorted(x.rank for x in state.experiments if x.rank is not None)
    assert ranks == list(range(1, len(ranks) + 1))
    assert all(0 <= x.decision_flip_prob <= 1 for x in state.experiments if x.rank is not None)


def test_decided_rule_is_not_reinterpreted_by_later_results():
    state = enter(approved_demo(), "x1", 2, 10)   # fail
    state = enter(state, "x1", 10, 10)            # contradictory batch
    x1 = next(x for x in state.experiments if x.id == "x1")
    assert x1.outcome == "fail"                   # the rule fixed in advance still decides
    assert state.run.assessment.decision.verdict == "pivot"
    assert state.updates[-1].posterior.p50 > state.updates[-2].posterior.p50  # but the belief learns


def test_tiny_sample_moves_less_than_large_sample():
    small = enter(approved_demo(), "x1", 1, 2).updates[-1]
    large = enter(approved_demo(), "x1", 10, 20).updates[-1]
    shift = lambda e: abs(e.posterior.p50 - e.prior.p50)  # noqa: E731
    assert shift(small) < shift(large) or abs(small.posterior.p50 - 0.5) < abs(large.posterior.p50 - 0.5)
    assert small.posterior.alpha + small.posterior.beta < large.posterior.alpha + large.posterior.beta


def test_good_results_can_pass_and_keep_continue():
    state = enter(approved_demo(), "x1", 8, 10)
    assert state.updates[-1].outcome == "pass"
    assert state.run.assessment.decision.verdict == "continue"


def test_kill_rule_fail_kills():
    state = enter(approved_demo(), "x3", 0, 20)   # no letters of intent at all
    assert state.updates[-1].outcome == "fail"
    assert state.run.assessment.decision.verdict == "kill"


def test_update_is_deterministic():
    a = enter(approved_demo(), "x2", 3, 15).updates[-1]
    b = enter(approved_demo(), "x2", 3, 15).updates[-1]
    assert a.model_dump(exclude={"created_at"}) == b.model_dump(exclude={"created_at"})


@pytest.mark.parametrize("xid,k,n", [("x1", 1, 10), ("x2", 1, 15), ("x3", 0, 20), ("x1", 0, 3)])
def test_evidence_below_the_estimate_never_raises_expected_revenue(xid, k, n):
    before = approved_demo()
    claim = next(c for c in before.claims if c.id == next(x for x in before.experiments if x.id == xid).assumption_id)
    prior = run_model(before).beliefs[claim.driver]
    assert k / n < min(prior.mean(), prior.quantiles()[1])
    after = enter(before.model_copy(deep=True), xid, k, n)
    post = after.updates[-1].posterior
    assert post.alpha / (post.alpha + post.beta) < prior.mean()  # E[output] = product of driver means
    assert post.p50 <= prior.quantiles()[1]
