"""Run state machine.

    POST /runs           -> queued -> scoping -> needs_input | refused | awaiting_gate1
    approve gate 1       -> researching -> modelling -> awaiting_gate2      (background thread)
    send back at gate 1  -> scoping again with the user's note              (background thread)
    approve gate 2       -> done (rules locked)                             (synchronous)
    send back at gate 2  -> new experiment plan with the note                (background thread)
    POST /results        -> Beta-Binomial update, verdict, re-rank           (synchronous)

A run cannot pass a gate without an approval request in the right status.
Background failures put the run in `failed` with the error, never crash the server.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from . import llm
from .models import (
    PROPORTION_DRIVERS,
    ApproveRequest,
    Claim,
    Derivation,
    Interpretation,
    ResultCreate,
    Run,
    RunCreate,
    RunState,
    ScopeResult,
    Step,
)
from .pipeline.assess import _output_claim, fit_share_claims, refresh
from .pipeline.gates import GateError, approve_plan
from .pipeline.learn import ResultError, apply_result
from .pipeline.model import DRIVER_ORDER, ModelError, run_model, sensitivities
from .store import RunRecord, RunStore

log = logging.getLogger("runner")

ANALOGY_STRENGTH = 0.25
GUESS_STRENGTH = 0.1
MAX_TARGETS = 3


class RunConflict(ValueError):
    """The request does not fit the run's current status (HTTP 409)."""


def now() -> datetime:
    return datetime.now(timezone.utc)


def _search_configured() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def _client() -> Optional[llm.LLMClient]:
    try:
        return llm.get_client()
    except llm.LLMNotConfigured:
        return None


# --------------------------------------------------------------------------- helpers


def _id_allocator(state: RunState) -> Callable[[str], str]:
    used = {c.id for c in state.claims} | {s.id for s in state.sources} | {x.id for x in state.experiments}
    counters: dict[str, int] = {}

    def new_id(prefix: str) -> str:
        n = counters.get(prefix, 0)
        while True:
            n += 1
            candidate = f"{prefix}{n}"
            if candidate not in used:
                used.add(candidate)
                counters[prefix] = n
                return candidate

    return new_id


def _start_step(state: RunState, name: str, message: Optional[str] = None) -> Step:
    step = Step(id=f"st{len(state.steps) + 1}", run_id=state.run.id, name=name, status="running",
                message=message, started_at=now())
    state.steps.append(step)
    state.run.updated_at = step.started_at
    return step


def _finish_step(state: RunState, step: Step, status: str = "done", message: Optional[str] = None) -> None:
    step.status = status
    if message is not None:
        step.message = message
    step.finished_at = now()
    state.run.updated_at = step.finished_at


def _fail(record: RunRecord, message: str) -> None:
    with record.lock:
        state = record.state
        for step in state.steps:
            if step.status == "running":
                _finish_step(state, step, "failed", message)
        state.run.status = "failed"
        state.run.error = message
        state.run.updated_at = now()
    log.error("run %s failed: %s", record.state.run.id, message)


def _spawn(target: Callable[..., None], *args) -> None:
    threading.Thread(target=target, args=args, daemon=True).start()


# --------------------------------------------------------------------------- scoping


def create_run(store: RunStore, body: RunCreate, *, background: bool = True) -> RunRecord:
    t = now()
    state = RunState(run=Run(id=store.new_id(), created_at=t, updated_at=t, status="queued",
                             idea_text=body.idea_text, context=body.context))
    record = RunRecord(state=state)
    store.put(record)
    if background:
        _spawn(scope_run, record, None)
    else:
        scope_run(record, None)
    return record


def scope_run(record: RunRecord, note: Optional[str]) -> None:
    from .pipeline.scope import run_scope

    with record.lock:
        state = record.state
        state.run.status = "scoping"
        state.run.error = None
        step = _start_step(state, "scoping", "Checking scope and reading the idea")
        idea, context, run_id = state.run.idea_text, state.run.context, state.run.id
    client = _client()
    if client is None:
        _fail(record, "The AI step is not configured on the server (missing LLM_PROVIDER / LLM_API_KEY).")
        return
    try:
        draft = run_scope(client, idea, context, run_id=run_id, note=note)
    except llm.LLMError as exc:
        _fail(record, f"Scoping failed: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("scope crashed")
        _fail(record, f"Scoping failed unexpectedly: {exc}")
        return

    with record.lock:
        state = record.state
        record.scope_draft = draft
        state.run.scope = draft.scope
        verdict = draft.scope.verdict
        if verdict == "in_scope" and draft.interpretation is not None:
            state.run.interpretation = draft.interpretation
            record.original_assumptions = {
                a.id: (a.p10, a.p50, a.p90) for a in draft.interpretation.assumptions
            }
            state.run.status = "awaiting_gate1"
            _finish_step(state, step, message="In scope: " + draft.interpretation.technology)
        elif verdict == "needs_info":
            state.run.status = "needs_input"
            _finish_step(state, step, message="Needs more information")
        else:
            state.run.status = "refused"
            _finish_step(state, step, message=f"Refused: {draft.scope.refusal_kind or 'out of scope'}")
        record.state = RunState.model_validate(state.model_dump())


# --------------------------------------------------------------------------- gate 1 -> plan


def _claims_from_interpretation(record: RunRecord, interpretation: Interpretation) -> list[Claim]:
    state = record.state
    draft = record.scope_draft
    basis = getattr(draft, "basis", {}) or {}
    rationale = getattr(draft, "rationale", {}) or {}
    claims: list[Claim] = []
    n = 0
    for a in interpretation.assumptions:
        if not a.included:
            continue
        n += 1
        edited = (a.driver is not None and a.id in record.original_assumptions
                  and record.original_assumptions[a.id] != (a.p10, a.p50, a.p90))
        strength = ANALOGY_STRENGTH if basis.get(a.driver or "") == "analogy" and not edited else GUESS_STRENGTH
        note = rationale.get(a.driver or "") or None
        if edited:
            note = "Range edited by the user at Gate 1."
        claims.append(Claim(
            id=f"c{n}", run_id=state.run.id, text=a.text, type="assumption",
            origin="user" if edited else "model", driver=a.driver, unit=a.unit,
            p10=a.p10, p50=a.p50, p90=a.p90, evidence_strength=strength, note=note,
        ))
    for fig in getattr(draft, "user_figures", []) or []:
        n += 1
        claims.append(Claim(
            id=f"c{n}", run_id=state.run.id, text=f"{fig.text} (stated in the idea)", type="assumption",
            origin="user", value=fig.value, unit=fig.unit, evidence_strength=GUESS_STRENGTH,
            note="User-supplied figure; see the top-down check.",
        ))
    return claims


def approve(record: RunRecord, body: ApproveRequest, *, background: bool = True) -> None:
    with record.lock:
        state = record.state
        status = state.run.status
        if body.gate == 1:
            if status != "awaiting_gate1":
                raise RunConflict(f"The run is {status}, not awaiting gate 1.")
            if not body.approved:
                state.run.status = "scoping"
                target, args = scope_run, (record, body.note)
            else:
                drivers = [a.driver for a in body.interpretation.assumptions if a.driver]
                missing = [d for d in DRIVER_ORDER if d not in drivers]
                if missing:
                    raise RunConflict(f"The interpretation is missing driver assumptions: {', '.join(missing)}.")
                if len(drivers) != len(set(drivers)):
                    raise RunConflict("Each market-model driver may appear only once in the interpretation.")
                state.run.interpretation = body.interpretation
                state.run.status = "researching"
                target, args = plan_run, (record,)
        else:
            if status != "awaiting_gate2":
                raise RunConflict(f"The run is {status}, not awaiting gate 2.")
            if body.approved:
                try:
                    approve_plan(state, body, now=now())
                except GateError as exc:
                    raise RunConflict(str(exc)) from exc
                record.state = RunState.model_validate(state.model_dump())
                return
            state.run.status = "modelling"
            target, args = replan_run, (record, body.note)
        state.run.updated_at = now()
    if background:
        _spawn(target, *args)
    else:
        target(*args)


def plan_run(record: RunRecord) -> None:
    try:
        _plan(record)
    except Exception as exc:
        log.exception("planning crashed")
        _fail(record, f"Assessment failed: {exc}")


def _plan(record: RunRecord) -> None:
    from .pipeline.experiments import Target, propose_experiments

    client = _client()
    with record.lock:
        state = record.state
        state.claims = _claims_from_interpretation(record, state.run.interpretation)
        state.sources, state.experiments, state.updates = [], [], []
        state.run.assessment = None
        interpretation = state.run.interpretation
        draft = record.scope_draft
        run_id = state.run.id

    cannot = list(getattr(draft, "cannot_determine", []) or [])
    limitations: list[str] = []
    if getattr(draft, "injection_suspected", False):
        limitations.append("The idea text contained instructions aimed at the AI; they were ignored.")

    # ---- research (optional; failures become gaps)
    with record.lock:
        step = _start_step(record.state, "researching", "Searching the web for the model drivers")
    if client is not None and _search_configured():
        from .pipeline.research.research import run_research

        with record.lock:
            drivers = {c.driver: c for c in record.state.claims if c.driver}
            new_id = _id_allocator(record.state)
        queries = getattr(draft, "search_queries", {}) or {
            d: [f"{c.text} {interpretation.region or ''}".strip()] for d, c in drivers.items()}
        result = run_research(client, interpretation, drivers, queries, run_id=run_id, new_id=new_id)
        with record.lock:
            state = record.state
            state.sources = list(result.sources)
            state.claims.extend(result.facts)
            by_driver = {c.driver: c for c in state.claims if c.driver}
            for est in result.estimates:
                target = by_driver.get(est.driver)
                facts = [c for c in result.facts if c.id in est.fact_claim_ids]
                if target is None or not facts:
                    continue
                target.type = "estimate"
                target.origin = "research"
                target.p10, target.p50, target.p90 = est.p10, est.p50, est.p90
                target.unit = est.unit or target.unit
                target.derivation = Derivation(formula=est.formula, input_claim_ids=[f.id for f in facts])
                target.evidence_strength = min(f.evidence_strength for f in facts)
                target.note = "Estimated from verified sources."
            cannot.extend(result.gaps)
            if result.injection_suspected:
                limitations.append("A fetched web page contained instructions aimed at the AI; they were ignored.")
            _finish_step(state, step, message=(f"{result.pages_fetched} pages fetched, {result.pages_failed} failed; "
                                               f"{result.quotes_verified} of {result.quotes_checked} quotes verified"))
    else:
        reason = "the AI step is not configured" if client is None else "web search is not configured"
        limitations.append(f"No web research was run ({reason}), so every driver is an assumption.")
        with record.lock:
            _finish_step(record.state, step, "skipped", f"Skipped: {reason}")

    # ---- model and ranking
    with record.lock:
        state = record.state
        state.run.status = "modelling"
        step = _start_step(state, "modelling", "Monte Carlo: 5,000 draws, seed 42")
        RunState.model_validate(state.model_dump())  # fail early on a broken ledger
        run = run_model(state)
        fit_share_claims(state, run)
        run = run_model(state)
        new_id = _id_allocator(state)
        output = _output_claim(state, new_id("c"), run)
        _finish_step(state, step)
        rank_step = _start_step(state, "ranking", "Ranking weak, influential assumptions")
        sens = sensitivities(run.tornado)
        targets = []
        for d in PROPORTION_DRIVERS:
            c = next(c for c in state.claims if c.driver == d)
            if c.type == "assumption":
                targets.append(Target(claim=c, priority=round(sens[c.id] * (1 - c.evidence_strength), 4)))
        targets.sort(key=lambda t: t.priority, reverse=True)
        targets = targets[:MAX_TARGETS]
        _finish_step(state, rank_step, message=f"{len(targets)} testable assumptions selected")
        exp_step = _start_step(state, "experiments", "Writing experiments with rules fixed in advance")

    plan = propose_experiments(client, interpretation, targets, output, run_id=run_id)

    with record.lock:
        state = record.state
        state.experiments = plan.experiments
        if plan.used_template:
            limitations.append("Experiment cards come from a standard template because the AI step was unavailable.")
        _finish_step(state, exp_step, message=f"{len(plan.experiments)} experiments; threshold reason: "
                                              f"{plan.viability_reason}")
        compose_step = _start_step(state, "composing", "Composing the report from the ledger")
        refresh(state, threshold=plan.viability_threshold, output_claim_id=output.id,
                cannot_determine=cannot, extra_limitations=limitations)
        _finish_step(state, compose_step)
        state.run.status = "awaiting_gate2"
        record.state = RunState.model_validate(state.model_dump())


def replan_run(record: RunRecord, note: Optional[str]) -> None:
    """Gate 2 send-back: write a new experiment plan (same model)."""
    from .pipeline.experiments import Target, propose_experiments

    try:
        client = _client()
        with record.lock:
            state = record.state
            step = _start_step(state, "experiments", "Rewriting the experiment plan" + (f": {note}" if note else ""))
            claims = {c.id: c for c in state.claims}
            targets = [Target(claim=claims[x.assumption_id], priority=x.priority) for x in state.experiments]
            output = claims[state.run.assessment.market_model.output_claim_id]
            interpretation = state.run.interpretation
        plan = propose_experiments(client, interpretation, targets, output, run_id=state.run.id)
        with record.lock:
            state = record.state
            state.experiments = plan.experiments
            refresh(state, threshold=plan.viability_threshold)
            _finish_step(state, step)
            state.run.status = "awaiting_gate2"
            record.state = RunState.model_validate(state.model_dump())
    except Exception as exc:
        log.exception("replanning crashed")
        _fail(record, f"Replanning failed: {exc}")


# --------------------------------------------------------------------------- results


def add_result(record: RunRecord, body: ResultCreate) -> None:
    with record.lock:
        state = RunState.model_validate(record.state.model_dump())  # work on a copy; commit on success
        try:
            apply_result(state, body, now=now())
            refresh(state, rank=False)  # apply_result has just ranked against the same model
        except (ResultError, ModelError) as exc:
            raise RunConflict(str(exc)) from exc
        state.run.updated_at = now()
        record.state = RunState.model_validate(state.model_dump())
