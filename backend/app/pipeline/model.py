"""Bottom-up market model: Monte Carlo, tornado, sensitivity, evidence floor, top-down check.

Output (obtainable annual revenue in year five) =
    target organisations × share with need and budget × adoption rate
    × share we can win × annual price per organisation

Each driver gets one fixed stream of uniforms (seed 42, 5,000 draws) and is
sampled by inverse CDF, so re-running the model after an update changes only
the updated driver's draws (common random numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models import PROPORTION_DRIVERS, Claim, OutputRange, RunState, TopDownCheck, TornadoBar
from .beliefs import PRIOR_MAX_STRENGTH, Belief, Beta, SplitLognormal, fit_beta

DRAWS = 5000
SEED = 42
DRIVER_ORDER: tuple[str, ...] = (
    "n_target_orgs",
    "share_need_budget",
    "adoption_rate",
    "achievable_share",
    "price_per_org",
)
DRIVER_LABEL: dict[str, str] = {
    "n_target_orgs": "target organisations",
    "share_need_budget": "share with need and budget",
    "adoption_rate": "adoption within five years",
    "achievable_share": "share we can win",
    "price_per_org": "annual price per organisation",
}
OUTPUT_LABEL = "Obtainable annual revenue in year five"
OUTPUT_UNIT = "EUR per year"


class ModelError(ValueError):
    """The claims do not form a complete market model."""


def uniforms(draws: int = DRAWS, seed: int = SEED) -> dict[str, np.ndarray]:
    return {d: np.random.default_rng([seed, i]).random(draws) for i, d in enumerate(DRIVER_ORDER)}


UNIFORMS = uniforms()  # fixed per driver; shared by every model run


def driver_claims(state: RunState) -> dict[str, Claim]:
    found = {c.driver: c for c in state.claims if c.driver is not None}
    missing = [d for d in DRIVER_ORDER if d not in found]
    if missing:
        raise ModelError(f"missing driver claims: {', '.join(missing)}")
    return found


def latest_posterior(state: RunState, claim_id: str) -> Optional[Beta]:
    for event in reversed(state.updates):
        if event.claim_id == claim_id:
            return Beta(event.posterior.alpha, event.posterior.beta)
    return None


def belief_for(state: RunState, claim: Claim) -> Belief:
    """The distribution the model uses for a driver claim."""
    if claim.p10 is None or claim.p50 is None or claim.p90 is None:
        raise ModelError(f"driver claim {claim.id} has no range")
    if claim.driver in PROPORTION_DRIVERS:
        posterior = latest_posterior(state, claim.id)
        if posterior is not None:
            return posterior
        cap = PRIOR_MAX_STRENGTH if claim.type == "assumption" else None
        return fit_beta(claim.p10, claim.p50, claim.p90, max_strength=cap)
    if claim.p10 <= 0:
        raise ModelError(f"driver claim {claim.id} needs a positive range")
    return SplitLognormal(claim.p10, claim.p50, claim.p90)


@dataclass
class ModelRun:
    beliefs: dict[str, Belief]
    samples: dict[str, np.ndarray]
    output: np.ndarray
    output_range: OutputRange
    tornado: list[TornadoBar]
    claim_ids: dict[str, str]  # driver -> claim id

    def p_at_least(self, threshold: float) -> float:
        return float(np.mean(self.output >= threshold))


def output_range(output: np.ndarray) -> OutputRange:
    p10, p50, p90 = np.percentile(output, [10, 50, 90])
    return OutputRange(p10=float(p10), p50=float(p50), p90=float(p90))


def run_model(state: RunState, overrides: Optional[dict[str, Belief]] = None) -> ModelRun:
    claims = driver_claims(state)
    beliefs: dict[str, Belief] = {d: belief_for(state, claims[d]) for d in DRIVER_ORDER}
    if overrides:
        beliefs.update(overrides)
    samples = {d: beliefs[d].ppf(UNIFORMS[d]) for d in DRIVER_ORDER}
    output = np.prod(np.vstack([samples[d] for d in DRIVER_ORDER]), axis=0)
    return ModelRun(
        beliefs=beliefs,
        samples=samples,
        output=output,
        output_range=output_range(output),
        tornado=tornado(beliefs, {d: claims[d].id for d in DRIVER_ORDER}),
        claim_ids={d: claims[d].id for d in DRIVER_ORDER},
    )


def tornado(beliefs: dict[str, Belief], claim_ids: dict[str, str]) -> list[TornadoBar]:
    """Output when one driver moves to its P10 / P90 with the others at their medians."""
    q = {d: beliefs[d].quantiles() for d in DRIVER_ORDER}
    medians = {d: q[d][1] for d in DRIVER_ORDER}

    def output_with(driver: str, value: float) -> float:
        return float(np.prod([value if d == driver else medians[d] for d in DRIVER_ORDER]))

    bars = []
    for d in DRIVER_ORDER:
        low, high = output_with(d, q[d][0]), output_with(d, q[d][2])
        bars.append(TornadoBar(driver=d, claim_id=claim_ids[d], low_output=low, high_output=high,
                               swing=abs(high - low)))
    bars.sort(key=lambda b: b.swing, reverse=True)  # stable: ties keep DRIVER_ORDER
    return bars


def sensitivities(bars: list[TornadoBar]) -> dict[str, float]:
    """Normalised swing per driver claim id (largest = 1)."""
    top = max((b.swing for b in bars), default=0.0)
    return {b.claim_id: (b.swing / top if top > 0 else 0.0) for b in bars}


def verified_facts_behind(state: RunState, claim_id: str) -> set[str]:
    """Verified fact ids reachable from a claim through derivations."""
    claims = {c.id: c for c in state.claims}
    seen: set[str] = set()
    facts: set[str] = set()
    stack = [claim_id]
    while stack:
        cid = stack.pop()
        if cid in seen or cid not in claims:
            continue
        seen.add(cid)
        c = claims[cid]
        if c.type == "fact" and c.status == "verified":
            facts.add(cid)
        if c.derivation is not None:
            stack.extend(c.derivation.input_claim_ids)
    return facts


def evidence_floor(state: RunState, bars: list[TornadoBar]) -> tuple[bool, list[str]]:
    """Floor: the top 3 drivers by swing need at least 2 verified facts behind them.

    Returns (met, driver keys among the top 3 that have no verified fact).
    """
    top = bars[:3]
    facts: set[str] = set()
    lacking = []
    for bar in top:
        behind = verified_facts_behind(state, bar.claim_id)
        facts |= behind
        if not behind:
            lacking.append(bar.driver)
    return len(facts) >= 2, lacking


def _is_money(unit: Optional[str]) -> bool:
    return bool(unit) and ("EUR" in unit.upper() or "€" in unit)


def top_down_check(state: RunState, beliefs: dict[str, Belief]) -> Optional[TopDownCheck]:
    """Compare the largest money figure the user asserted with the bottom-up serviceable market."""
    figures = [c for c in state.claims
               if c.origin == "user" and c.value is not None and c.value > 0 and _is_money(c.unit)]
    if not figures:
        return None
    figure = max(figures, key=lambda c: c.value or 0.0)
    serviceable = float(
        beliefs["n_target_orgs"].quantiles()[1]
        * beliefs["share_need_budget"].quantiles()[1]
        * beliefs["price_per_org"].quantiles()[1]
    )
    if serviceable <= 0:
        return None
    ratio = float(figure.value) / serviceable
    if ratio >= 1:
        relation = f"{ratio:,.0f} times" if ratio >= 10 else f"{ratio:,.1f} times"
        note = (f"The figure in the idea is {relation} the bottom-up serviceable market "
                "(target organisations × share with need and budget × annual price). "
                "It stays a user-supplied assumption and is not used in the model.")
    else:
        note = (f"The figure in the idea is {1 / ratio:,.1f} times smaller than the bottom-up "
                "serviceable market. It stays a user-supplied assumption and is not used in the model.")
    return TopDownCheck(claim_id=figure.id, top_down_value=float(figure.value),
                        bottom_up_value=serviceable, ratio=ratio, note=note)
