"""The learning loop: verdicts from rules fixed in advance, Beta-Binomial updates, re-ranking.

Verdict (code, never the model), in order:
  1. an experiment failed a rule whose on_fail is "kill"   -> kill
  2. an experiment failed a rule whose on_fail is "pivot"  -> pivot
  3. less than 50% of Monte Carlo draws reach the viability threshold -> kill
  4. otherwise -> continue (provisional until the first result is entered)

Once a rule has decided (pass or fail) it stays decided: later results still
update the belief, but they cannot reinterpret the decision after the fact.

Ranking of open experiments approximates the value of information: the
probability that the experiment's next batch changes the verdict (Beta-Binomial
predictive over its outcomes), divided by its effort in days.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from scipy import special

from ..models import (
    BetaBelief,
    Claim,
    Decision,
    Derivation,
    Experiment,
    FailAction,
    ResultCreate,
    RuleOutcome,
    RunState,
    UpdateEvent,
    Verdict,
)
from .beliefs import Beta, beta_binomial_pmf
from .model import DRIVER_LABEL, DRIVER_ORDER, UNIFORMS, ModelRun, run_model
from .rules import evaluate_rule
from .text import METHOD_LABEL, fmt_money, fmt_pct

ACTIVE_STATUSES = ("proposed", "accepted", "edited")
OPEN_OUTCOMES = ("pending", "inconclusive")
DECIDED_OUTCOMES = ("pass", "fail")
MAX_FLIP_POINTS = 41  # outcome grid for large samples (keeps ranking fast)
MIN_EFFORT_DAYS = 0.5
MOCKED_STRENGTH = 0.1
ENTERED_STRENGTH = 0.25  # the team's own result: real, but we cannot verify it


class ResultError(ValueError):
    """A result cannot be accepted (wrong state, unknown or unapproved experiment)."""


# --------------------------------------------------------------------------- verdict


def verdict_for(outcomes: dict[str, RuleOutcome], on_fail: dict[str, FailAction], p_viable: float) -> Verdict:
    failed = [xid for xid, o in outcomes.items() if o == "fail"]
    if any(on_fail[xid] == "kill" for xid in failed):
        return "kill"
    if failed:
        return "pivot"
    if p_viable < 0.5:
        return "kill"
    return "continue"


def active_experiments(state: RunState) -> list[Experiment]:
    return [x for x in state.experiments if x.status in ACTIVE_STATUSES]


def decide(state: RunState, run: ModelRun, threshold: float) -> Decision:
    active = active_experiments(state)
    p_viable = run.p_at_least(threshold)
    verdict = verdict_for({x.id: x.outcome for x in active}, {x.id: x.rule.on_fail for x in active}, p_viable)
    provisional = not any(x.trials_done > 0 for x in active)

    rule_reasons: list[str] = []
    for x in active:
        label = METHOD_LABEL[x.method].lower()
        if x.outcome == "fail":
            rule_reasons.append(
                f"{x.id} ({label}) failed its rule: {x.successes} of {x.trials_done} "
                f"(rule fixed in advance: {x.kill_criterion})."
            )
        elif x.outcome == "pass":
            rule_reasons.append(
                f"{x.id} ({label}) passed its rule: {x.successes} of {x.trials_done} "
                f"(needed {x.pass_threshold})."
            )
        elif x.outcome == "inconclusive":
            rule_reasons.append(
                f"{x.id} ({label}) is inconclusive so far: {x.successes} of {x.trials_done}."
            )
    viability = (
        f"Chance of reaching {fmt_money(threshold)} a year: {fmt_pct(p_viable)} "
        f"(the rule fixed in advance needs at least 50% to continue)."
    )
    failed_rules = [r for r in rule_reasons if " failed its rule" in r]
    if failed_rules:
        reasons = failed_rules + [viability] + [r for r in rule_reasons if r not in failed_rules]
    else:
        reasons = [viability] + rule_reasons
    if provisional:
        reasons.append("Provisional: no experiment results yet, so this rests on assumptions.")
    return Decision(verdict=verdict, provisional=provisional, reasons=reasons,
                    viability_threshold=threshold, p_viable=p_viable)


# --------------------------------------------------------------------------- ranking


def _remaining_trials(x: Experiment) -> int:
    return x.rule.trials - x.trials_done if x.trials_done < x.rule.trials else x.rule.trials


def _outcome_grid(trials: int, pmf: np.ndarray) -> list[tuple[int, float]]:
    """(k, probability) pairs; large samples are bucketed onto an evenly spaced grid."""
    if trials + 1 <= MAX_FLIP_POINTS:
        return [(k, float(pmf[k])) for k in range(trials + 1)]
    grid = np.unique(np.round(np.linspace(0, trials, MAX_FLIP_POINTS)).astype(int))
    nearest = np.abs(np.arange(trials + 1)[:, None] - grid[None, :]).argmin(axis=1)
    mass = np.bincount(nearest, weights=pmf, minlength=len(grid))
    return [(int(k), float(m)) for k, m in zip(grid, mass)]


def flip_probability(state: RunState, run: ModelRun, x: Experiment, threshold: float, current: Verdict) -> float:
    """Probability that the experiment's next batch changes the verdict."""
    claims = {c.id: c for c in state.claims}
    driver = claims[x.assumption_id].driver
    belief = run.beliefs[driver]
    if not isinstance(belief, Beta):
        raise TypeError(f"{x.id} tests {driver}, which is not a share")
    active = active_experiments(state)
    outcomes = {e.id: e.outcome for e in active}
    on_fail = {e.id: e.rule.on_fail for e in active}
    others = np.prod(np.vstack([run.samples[d] for d in DRIVER_ORDER if d != driver]), axis=0)
    # A draw clears the threshold when share >= threshold / others. With inverse-CDF sampling
    # (share = ppf(u)) that is u >= cdf(threshold / others): the same test as re-sampling,
    # but the forward CDF is far cheaper than the inverse.
    with np.errstate(divide="ignore"):
        cut = np.clip(np.where(others > 0, threshold / others, np.inf), 0.0, 1.0)
    u = UNIFORMS[driver]
    n = _remaining_trials(x)
    flip = 0.0
    for k, mass in _outcome_grid(n, beta_binomial_pmf(n, belief)):
        if mass <= 0:
            continue
        posterior = belief.update(k, n)
        p_viable = float(np.mean(u >= special.betainc(posterior.alpha, posterior.beta, cut)))
        outcome = x.outcome if x.outcome in DECIDED_OUTCOMES else evaluate_rule(
            x.rule, x.successes + k, x.trials_done + n)
        if verdict_for({**outcomes, x.id: outcome}, on_fail, p_viable) != current:
            flip += mass
    return float(min(max(flip, 0.0), 1.0))


def rank_experiments(state: RunState, run: ModelRun, verdict: Verdict, threshold: float) -> list[str]:
    """Set decision_flip_prob and rank on every experiment. Returns open ids, best first."""
    scored = []
    for x in state.experiments:
        x.rank = None
        x.decision_flip_prob = None
        if x.status in ACTIVE_STATUSES and x.outcome in OPEN_OUTCOMES:
            p = flip_probability(state, run, x, threshold, verdict)
            x.decision_flip_prob = round(p, 4)
            score = p / max(x.effort_days, MIN_EFFORT_DAYS)
            scored.append((-score, x.cost_eur, -x.priority, x.id, x))
    scored.sort(key=lambda t: t[:4])
    for i, (*_, x) in enumerate(scored, start=1):
        x.rank = i
    return [t[3] for t in scored]


def current_order(state: RunState) -> list[str]:
    ranked = [x for x in state.experiments if x.rank is not None]
    return [x.id for x in sorted(ranked, key=lambda x: x.rank)]


# --------------------------------------------------------------------------- results


def _belief(b: Beta) -> BetaBelief:
    p10, p50, p90 = b.quantiles()
    return BetaBelief(alpha=b.alpha, beta=b.beta, p10=p10, p50=p50, p90=p90)


def _next_claim_id(state: RunState) -> str:
    used = {c.id for c in state.claims}
    n = len(state.claims) + 1
    while f"c{n}" in used:
        n += 1
    return f"c{n}"


def apply_result(state: RunState, body: ResultCreate, *, now: datetime) -> UpdateEvent:
    """Record one experiment result and update the tested share (mutates `state`).

    The caller re-composes the assessment afterwards (assess.refresh).
    """
    if state.run.status != "done" or state.run.assessment is None or state.run.assessment.decision is None:
        raise ResultError("Approve the plan at Gate 2 before entering results.")
    x = next((e for e in state.experiments if e.id == body.experiment_id), None)
    if x is None:
        raise ResultError(f"Unknown experiment {body.experiment_id}.")
    if x.status not in ("accepted", "edited") or x.locked_at is None:
        raise ResultError(f"Experiment {x.id} was not approved at Gate 2.")

    threshold = state.run.assessment.decision.viability_threshold
    claims = {c.id: c for c in state.claims}
    target = claims[x.assumption_id]
    driver = target.driver

    before = run_model(state)
    decision_before = decide(state, before, threshold)
    rank_before = current_order(state)
    prior = before.beliefs[driver]
    if not isinstance(prior, Beta):
        raise ResultError(f"{x.id} tests {driver}, which is not a share")
    posterior = prior.update(body.successes, body.trials)

    obs = Claim(
        id=_next_claim_id(state),
        run_id=state.run.id,
        text=(f"Result of {x.id} ({METHOD_LABEL[x.method].lower()}): {body.successes} of {body.trials}"
              f" — {x.metric}" + (" [MOCKED demo value]" if body.mocked else "")),
        type="assumption",
        origin="experiment",
        value=body.successes / body.trials,
        unit="share",
        evidence_strength=MOCKED_STRENGTH if body.mocked else ENTERED_STRENGTH,
        note=body.note or ("Mocked result for the demo; not real evidence." if body.mocked
                           else "Entered by the team; we cannot verify it."),
    )
    state.claims.append(obs)

    earlier = [u for u in state.updates if u.claim_id == target.id]
    first_prior = earlier[0].prior if earlier else _belief(prior)
    obs_ids = [u.observation_claim_id for u in earlier] + [obs.id]
    successes = sum(u.successes for u in earlier) + body.successes
    trials = sum(u.trials for u in earlier) + body.trials
    obs_claims = [c for c in state.claims if c.id in obs_ids]
    p10, p50, p90 = posterior.quantiles()
    target.type = "estimate"
    target.p10, target.p50, target.p90 = p10, p50, p90
    target.derivation = Derivation(
        formula=(f"Beta-Binomial update of the {DRIVER_LABEL[driver]} assumption: prior "
                 f"Beta({first_prior.alpha:.2f}, {first_prior.beta:.2f}) + {successes} of {trials} "
                 f"({', '.join(obs_ids)}) → Beta({posterior.alpha:.2f}, {posterior.beta:.2f})"),
        input_claim_ids=obs_ids,
    )
    target.evidence_strength = min(c.evidence_strength for c in obs_claims)
    mocked_any = body.mocked or any(u.mocked for u in earlier)
    target.note = (f"Updated by experiment results ({', '.join(obs_ids)})"
                   + (", including mocked demo values." if mocked_any else "."))

    was_decided = x.outcome in DECIDED_OUTCOMES
    x.successes += body.successes
    x.trials_done += body.trials
    if not was_decided:
        x.outcome = evaluate_rule(x.rule, x.successes, x.trials_done)

    after = run_model(state, overrides={driver: posterior})
    decision_after = decide(state, after, threshold)
    rank_after = rank_experiments(state, after, decision_after.verdict, threshold)

    event = UpdateEvent(
        id=f"u{len(state.updates) + 1}",
        run_id=state.run.id,
        created_at=now,
        experiment_id=x.id,
        claim_id=target.id,
        observation_claim_id=obs.id,
        successes=body.successes,
        trials=body.trials,
        mocked=body.mocked,
        outcome=x.outcome,
        prior=_belief(prior),
        posterior=_belief(posterior),
        output_before=before.output_range,
        output_after=after.output_range,
        p_viable_before=decision_before.p_viable,
        p_viable_after=decision_after.p_viable,
        verdict_before=decision_before.verdict,
        verdict_after=decision_after.verdict,
        rank_before=rank_before,
        rank_after=rank_after,
    )
    state.updates.append(event)
    return event
