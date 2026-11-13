"""Recompute everything derived from the ledger: model, output, floor, verdict, ranks, report.

`refresh` is called after planning (Gate 1), after Gate 2 edits and after every
entered result. It is deterministic: the same claims give the same assessment.
"""

from __future__ import annotations

from typing import Optional

from ..models import (
    PROPORTION_DRIVERS,
    Assessment,
    Claim,
    Derivation,
    MarketModel,
    RunState,
)
from .beliefs import Beta
from .compose import compose
from .learn import decide, rank_experiments
from .model import (
    DRAWS,
    DRIVER_LABEL,
    DRIVER_ORDER,
    OUTPUT_LABEL,
    OUTPUT_UNIT,
    SEED,
    ModelRun,
    evidence_floor,
    latest_posterior,
    run_model,
    sensitivities,
    top_down_check,
)

BASE_LIMITATIONS = [
    "Prototype output: figures are model ranges, not forecasts or calibrated probabilities.",
    "A verified quote shows that a page said something, not that it is true.",
    "Drivers are treated as independent; correlations between them are not modelled.",
    "Results entered by the team are not verified by us; mocked demo results are labelled as such.",
]


def fit_share_claims(state: RunState, run: ModelRun) -> None:
    """Show the distribution the model actually uses on share assumptions that were never updated."""
    for d in PROPORTION_DRIVERS:
        cid = run.claim_ids[d]
        claim = next(c for c in state.claims if c.id == cid)
        belief = run.beliefs[d]
        if claim.type != "assumption" or not isinstance(belief, Beta) or latest_posterior(state, cid):
            continue
        p10, p50, p90 = belief.quantiles()
        if (claim.p10, claim.p50, claim.p90) != (p10, p50, p90):
            claim.p10, claim.p50, claim.p90 = p10, p50, p90
        fitted = f"Model uses Beta({belief.alpha:.2f}, {belief.beta:.2f})."
        if not claim.note or "Model uses Beta(" not in claim.note:
            claim.note = f"{claim.note} {fitted}".strip() if claim.note else fitted


def _output_claim(state: RunState, output_claim_id: str, run: ModelRun) -> Claim:
    drivers = [next(c for c in state.claims if c.id == run.claim_ids[d]) for d in DRIVER_ORDER]
    formula = (" × ".join(f"{DRIVER_LABEL[d]} ({run.claim_ids[d]})" for d in DRIVER_ORDER)
               + f"; Monte Carlo, {DRAWS:,} draws, seed {SEED}")
    out = next((c for c in state.claims if c.id == output_claim_id), None)
    if out is None:
        out = Claim(id=output_claim_id, run_id=state.run.id, text=f"{OUTPUT_LABEL}.", type="estimate",
                    origin="model", unit=OUTPUT_UNIT, p10=0, p50=0, p90=0,
                    derivation=Derivation(formula=formula, input_claim_ids=[c.id for c in drivers]),
                    evidence_strength=0.0)
        state.claims.append(out)
    r = run.output_range
    out.p10, out.p50, out.p90 = r.p10, r.p50, r.p90
    out.derivation = Derivation(formula=formula, input_claim_ids=[c.id for c in drivers])
    out.evidence_strength = min(c.evidence_strength for c in drivers)
    return out


def refresh(
    state: RunState,
    *,
    threshold: Optional[float] = None,
    output_claim_id: Optional[str] = None,
    cannot_determine: Optional[list[str]] = None,
    extra_limitations: Optional[list[str]] = None,
    fit_claims: bool = False,
    rank: bool = True,
) -> ModelRun:
    """Rebuild state.run.assessment from the claims and experiments (mutates `state`).

    `rank=False` keeps the experiments' current ranks (apply_result has just computed them).
    """
    previous = state.run.assessment
    if threshold is None:
        if previous is None or previous.decision is None:
            raise ValueError("a viability threshold is needed for the first assessment")
        threshold = previous.decision.viability_threshold
    if output_claim_id is None:
        if previous is None or previous.market_model is None:
            raise ValueError("an output claim id is needed for the first assessment")
        output_claim_id = previous.market_model.output_claim_id

    run = run_model(state)
    if fit_claims:
        fit_share_claims(state, run)
        run = run_model(state)

    output = _output_claim(state, output_claim_id, run)
    sens = sensitivities(run.tornado)
    for c in state.claims:
        c.sensitivity = round(sens[c.id], 4) if c.driver is not None and c.id in sens else None

    claims = {c.id: c for c in state.claims}
    for x in state.experiments:
        target = claims[x.assumption_id]
        x.priority = round((target.sensitivity or 0.0) * (1 - target.evidence_strength), 4)

    floor_met, lacking = evidence_floor(state, run.tornado)
    data_needed: list[str] = []
    if not floor_met:
        data_needed = [f"A published, quotable figure for {DRIVER_LABEL[d]}." for d in lacking]
        if not data_needed:
            data_needed = ["A second independent, quotable source for the most influential drivers."]
    top_down = top_down_check(state, run.beliefs)
    decision = decide(state, run, threshold)
    if rank:
        rank_experiments(state, run, decision.verdict, threshold)
    headline, sections = compose(state, run, decision, output_claim_id=output.id,
                                 floor_met=floor_met, top_down=top_down)

    if previous is not None:
        cannot = previous.cannot_determine if cannot_determine is None else cannot_determine
        limitations = previous.limitations if extra_limitations is None else BASE_LIMITATIONS + extra_limitations
        critic = previous.critic_notes
    else:
        cannot = cannot_determine or []
        limitations = BASE_LIMITATIONS + (extra_limitations or [])
        critic = []

    state.run.assessment = Assessment(
        evidence_floor_met=floor_met,
        headline=headline,
        market_model=MarketModel(output_label=OUTPUT_LABEL, output_claim_id=output.id, draws=DRAWS,
                                 seed=SEED, tornado=run.tornado, top_down_check=top_down),
        decision=decision,
        sections=sections,
        cannot_determine=cannot,
        data_needed=data_needed,
        limitations=limitations,
        critic_notes=critic,
    )
    return run
