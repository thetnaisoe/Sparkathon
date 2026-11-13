"""Evidence for the learning loop, written to evals/results/latest.json (served by GET /evals/latest).

    cd backend && .venv/bin/python evals/run_evals.py            # code-only checks (no key needed)
    LLM_PROVIDER=... LLM_API_KEY=... .venv/bin/python evals/run_evals.py --llm   # + baseline and T5-T12

Code-only checks
  E1 rule consistency  6 scripted result sets; the verdict must equal what the rule fixed in advance implies
  E2 direction         results below the current estimate never raise the share's median or expected revenue;
                       how often the output's P50 still rises (it can, when a result mainly removes uncertainty)
  E3 tiny sample       2 trials move the belief less than 20 trials at the same rate
  E4 contradiction     a decided rule is not reinterpreted by a later, opposite batch
  E5 reproducibility   same inputs give byte-identical outputs
With --llm
  B1 baseline          one unassisted prompt gets the same idea, rule and result and must name the verdict
  T5/T6/T7/T9/T10/T12  scope behaviour on edge cases (cases.yaml)

This is prototype evidence on a small number of cases, not statistical proof.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.env import load_env  # noqa: E402

load_env()
from app import llm  # noqa: E402
from app.models import ApproveRequest, EvalResult, ResultCreate, RunContext, RunState  # noqa: E402
from app.pipeline.assess import refresh  # noqa: E402
from app.pipeline.beliefs import fit_beta  # noqa: E402
from app.pipeline.gates import approve_plan  # noqa: E402
from app.pipeline.learn import apply_result  # noqa: E402
from app.pipeline.model import run_model  # noqa: E402

FIXTURE = ROOT.parent / "fixtures" / "assessment.json"
OUT = ROOT / "evals" / "results" / "latest.json"
CASES = ROOT / "evals" / "cases.yaml"
NOW = datetime(2026, 11, 13, 9, 0, tzinfo=timezone.utc)

# (case id, experiment, successes, trials, description)
SCRIPTED = [
    ("E1.1", "x1", 8, 10, "clear pass"),
    ("E1.2", "x1", 2, 10, "clear fail (pivot rule)"),
    ("E1.3", "x1", 5, 10, "borderline: exactly the pass count"),
    ("E1.4", "x1", 3, 10, "borderline: exactly the fail line, inconclusive"),
    ("E1.5", "x3", 0, 20, "clear fail (kill rule)"),
    ("E1.6", "x2", 1, 8, "partial sample that can no longer reach the fail line"),
]


def demo_state() -> RunState:
    state = RunState.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    approve_plan(state, ApproveRequest(gate=2, decisions=[
        {"experiment_id": x.id, "decision": "accept"} for x in state.experiments]), now=NOW)
    return state


def enter(state: RunState, xid: str, k: int, n: int) -> RunState:
    apply_result(state, ResultCreate(experiment_id=xid, successes=k, trials=n, mocked=True), now=NOW)
    refresh(state)
    return state


def rule_implied_verdict(state: RunState, xid: str, k: int, n: int) -> str:
    """Independent re-implementation of what the rules fixed in advance imply (counts only)."""
    x = next(e for e in state.experiments if e.id == xid)
    r = x.rule
    if n >= r.trials:
        failed = k * r.trials < r.fail_below * n
    else:
        failed = k + (r.trials - n) < r.fail_below
    if failed:
        return r.on_fail
    p_viable = run_model(state).p_at_least(state.run.assessment.decision.viability_threshold)
    return "continue" if p_viable >= 0.5 else "kill"


class BaselineVerdict(BaseModel):
    verdict: Literal["continue", "pivot", "kill"]
    reason: str


def baseline_verdict(client: llm.LLMClient, state: RunState, xid: str, k: int, n: int) -> str:
    x = next(e for e in state.experiments if e.id == xid)
    user = (
        f"Startup idea: {state.run.idea_text}\n\n"
        f"Before running an experiment the team fixed this rule: {x.hypothesis} Metric: {x.metric}. "
        f"Pass if {x.pass_threshold}. {x.kill_criterion}. Otherwise inconclusive.\n"
        f"They also decided to continue only if obtainable revenue in year five is likely to reach "
        f"EUR {state.run.assessment.decision.viability_threshold:,.0f} a year.\n"
        f"Result: {k} of {n}.\n\nWhat is the verdict now: continue, pivot or kill? Reply as JSON."
    )
    reply = client.structured(BaselineVerdict, system="You are an experienced startup advisor.", user=user,
                              purpose="eval_baseline", run_id=None)
    return reply.verdict


def e1_rule_consistency(client: Optional[llm.LLMClient]) -> list[EvalResult]:
    rows = []
    for case_id, xid, k, n, desc in SCRIPTED:
        started = time.monotonic()
        state = enter(demo_state(), xid, k, n)
        expected = rule_implied_verdict(state, xid, k, n)
        got = state.run.assessment.decision.verdict
        rows.append(row(case_id, "pipeline", {"match": float(got == expected)}, got == expected,
                        f"{desc}: {k} of {n} on {xid}; expected {expected}, pipeline {got}", started))
        if client is not None:
            started = time.monotonic()
            try:
                b = baseline_verdict(client, demo_state(), xid, k, n)
                rows.append(row(case_id, "baseline", {"match": float(b == expected)}, b == expected,
                                f"{desc}: expected {expected}, single prompt {b}", started))
            except llm.LLMError as exc:
                rows.append(row(case_id, "baseline", {}, None, f"baseline call failed: {exc}", started))
    return rows


def e2_direction() -> list[EvalResult]:
    started = time.monotonic()
    state = RunState.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    cases = share_up = mean_up = p50_up = 0
    worst = 0.0
    for driver in ("share_need_budget", "adoption_rate", "achievable_share"):
        for rng in [(0.3, 0.5, 0.7), (0.1, 0.25, 0.45), (0.02, 0.05, 0.1), (0.05, 0.3, 0.8), (0.6, 0.8, 0.95)]:
            prior = fit_beta(*rng, max_strength=20)
            for n in (1, 2, 3, 5, 10, 20, 50):
                for k in range(n + 1):
                    if not k / n < min(prior.quantiles()[1], prior.mean()):
                        continue
                    cases += 1
                    post = prior.update(k, n)
                    share_up += post.quantiles()[1] > prior.quantiles()[1] + 1e-12
                    mean_up += post.mean() > prior.mean()  # E[revenue] = product of independent driver means
                    before = run_model(state, overrides={driver: prior}).output_range.p50
                    after = run_model(state, overrides={driver: post}).output_range.p50
                    if after > before + 1e-9:
                        p50_up += 1
                        worst = max(worst, (after - before) / before)
    return [
        row("E2.1", "pipeline", {"cases": cases, "violations": share_up}, share_up == 0,
            "Results below the current estimate never raise the tested share's median.", started),
        row("E2.2", "pipeline", {"cases": cases, "violations": mean_up}, mean_up == 0,
            "Results below the current estimate never raise expected revenue (product of driver means).", started),
        row("E2.3", "pipeline", {"cases": cases, "p50_rises": p50_up, "worst_relative_rise": round(worst, 4)}, None,
            "Honest finding, not a pass/fail check: the output's P50 can still rise when a result mainly removes "
            "uncertainty (the median of a product of skewed factors grows as their spread shrinks).", started),
    ]


def e3_tiny_sample() -> EvalResult:
    started = time.monotonic()
    small = enter(demo_state(), "x1", 1, 2).updates[-1]
    large = enter(demo_state(), "x1", 10, 20).updates[-1]
    w_small = (small.trials) / (small.posterior.alpha + small.posterior.beta)
    w_large = (large.trials) / (large.posterior.alpha + large.posterior.beta)
    ok = w_small < w_large
    return row("E3", "pipeline", {"data_weight_n2": round(w_small, 3), "data_weight_n20": round(w_large, 3)}, ok,
               "Two trials carry less weight in the posterior than twenty at the same rate.", started)


def e4_contradiction() -> EvalResult:
    started = time.monotonic()
    state = enter(demo_state(), "x1", 2, 10)
    state = enter(state, "x1", 10, 10)
    x1 = next(x for x in state.experiments if x.id == "x1")
    moved_up = state.updates[-1].posterior.p50 > state.updates[-2].posterior.p50
    ok = x1.outcome == "fail" and state.run.assessment.decision.verdict == "pivot" and moved_up
    return row("E4", "pipeline", {"belief_moved_up": float(moved_up)}, ok,
               "A decided rule stays decided after an opposite batch; the belief still learns from it.", started)


def e5_reproducible() -> EvalResult:
    started = time.monotonic()
    a = enter(demo_state(), "x2", 3, 15).model_dump_json()
    b = enter(demo_state(), "x2", 3, 15).model_dump_json()
    return row("E5", "pipeline", {}, a == b, "Same inputs give identical outputs (seed 42, fixed draws).", started)


def llm_scope_cases(client: llm.LLMClient) -> list[EvalResult]:
    from app.pipeline.scope import run_scope

    rows = []
    for case in yaml.safe_load(CASES.read_text(encoding="utf-8")):
        started = time.monotonic()
        try:
            draft = run_scope(client, case["idea"], RunContext(), run_id=f"eval-{case['id']}")
        except llm.LLMError as exc:
            rows.append(row(case["id"], "pipeline", {}, False, f"scope call failed: {exc}", started))
            continue
        checks = case["expect"]
        ok = draft.scope.verdict == checks["verdict"]
        if "refusal_kind" in checks:
            ok = ok and draft.scope.refusal_kind == checks["refusal_kind"]
        if checks.get("injection_suspected"):
            ok = ok and draft.injection_suspected
        if checks.get("user_figure"):
            ok = ok and bool(draft.user_figures)
        rows.append(row(case["id"], "pipeline", {"passed": float(ok)}, ok,
                        f"{case['name']}: verdict {draft.scope.verdict}, refusal {draft.scope.refusal_kind}, "
                        f"injection {draft.injection_suspected}, user figures {len(draft.user_figures)}", started))
    return rows


_EVAL_RUN = "eval-" + uuid.uuid4().hex[:8]


def row(case_id: str, system: str, metrics: dict, passed: Optional[bool], notes: str, started: float) -> EvalResult:
    return EvalResult(id=f"{_EVAL_RUN}-{case_id}-{system}", created_at=datetime.now(timezone.utc),
                      eval_run_id=_EVAL_RUN, case_id=case_id, system=system,
                      metrics={k: float(v) for k, v in metrics.items()}, passed=passed, notes=notes,
                      latency_s=round(time.monotonic() - started, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="also run the LLM baseline and scope cases")
    args = parser.parse_args()
    client = None
    if args.llm:
        try:
            client = llm.get_client()
        except llm.LLMNotConfigured as exc:
            print(f"--llm requested but {exc}")
            return 2
    rows = e1_rule_consistency(client) + e2_direction() + [e3_tiny_sample(), e4_contradiction(), e5_reproducible()]
    if client is not None:
        rows += llm_scope_cases(client)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps([r.model_dump(mode="json") for r in rows], indent=2) + "\n", encoding="utf-8")
    for r in rows:
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[r.passed]
        print(f"{mark:4} {r.case_id:6} {r.system:8} {r.notes}")
    checks = [r for r in rows if r.passed is not None]
    print(f"\n{sum(r.passed for r in checks)}/{len(checks)} checks passed; written to {OUT.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
