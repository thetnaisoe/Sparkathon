"""Deterministic report composer.

The report may only restate the ledger: every number enters through a
{{claim_id}} token, so ReportSection's validator can reject any number that
does not trace to a claim. No string literal in this file may contain a digit.
"""

from __future__ import annotations

from typing import Optional

from ..models import Decision, ReportSection, RunState, TopDownCheck
from .model import DRIVER_LABEL, DRIVER_ORDER, ModelRun
from .text import METHOD_LABEL


def _t(claim_id: str) -> str:
    return "{{" + claim_id + "}}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _verdict_sentence(state: RunState, decision: Decision) -> str:
    failed = [x for x in state.experiments if x.status != "rejected" and x.outcome == "fail"]
    if decision.verdict == "kill":
        if any(x.rule.on_fail == "kill" for x in failed):
            text = "Kill: an experiment failed a rule that was fixed in advance to mean the idea does not work."
        else:
            text = "Kill: the model no longer gives at least an even chance of reaching the viability threshold."
    elif decision.verdict == "pivot":
        text = "Pivot: an experiment failed a rule that was fixed in advance to mean this is the wrong segment."
    else:
        text = "Continue: no rule fixed in advance has failed, and the model clears the viability threshold."
    if decision.provisional:
        text += " This is provisional, because nothing has been tested yet."
    return text


def compose(
    state: RunState,
    run: ModelRun,
    decision: Decision,
    *,
    output_claim_id: str,
    floor_met: bool,
    top_down: Optional[TopDownCheck],
) -> tuple[Optional[str], list[ReportSection]]:
    """Return (headline, sections)."""
    claims = {c.id: c for c in state.claims}
    ids = run.claim_ids
    headline = f"Obtainable annual revenue in year five: {_t(output_claim_id)}." if floor_met else None

    sections: list[ReportSection] = []

    first = _verdict_sentence(state, decision)
    if floor_met:
        first += f" Obtainable annual revenue in year five is {_t(output_claim_id)}."
    else:
        first += (f" On assumptions alone the model gives {_t(output_claim_id)}; the evidence is too thin "
                  "for a headline figure.")
    top_two = [_t(b.claim_id) for b in run.tornado[:2]]
    first += f" The answer moves most with {_join(top_two)}."
    sections.append(ReportSection(title="Verdict", body=first))

    model_text = "; ".join(
        f"{DRIVER_LABEL[d]} {_t(ids[d])}" for d in DRIVER_ORDER
    )
    model_text = model_text[0].upper() + model_text[1:] + "."
    facts = [c.id for c in state.claims if c.type == "fact"]
    if facts:
        model_text += f" Verified facts behind the model: {_join([_t(f) for f in facts])}."
    else:
        model_text += " No verified fact supports these drivers yet, so they rest on assumptions."
    sections.append(ReportSection(title="Market model", body=model_text))

    weak = sorted(
        (claims[ids[d]] for d in DRIVER_ORDER),
        key=lambda c: (c.sensitivity or 0.0) * (1 - c.evidence_strength),
        reverse=True,
    )[:3]
    sections.append(ReportSection(
        title="Weakest assumptions",
        body=(f"{_join([_t(c.id) for c in weak])} combine the most influence with the least evidence, "
              "so they are the first to test."),
    ))

    plans = []
    for x in sorted((x for x in state.experiments if x.status != "rejected"),
                    key=lambda x: (x.rank is None, x.rank or 0, x.id)):
        action = "pivot" if x.rule.on_fail == "pivot" else "kill"
        if x.outcome in ("pass", "fail"):
            plans.append(f"{METHOD_LABEL[x.method]} on {_t(x.assumption_id)} has decided: {x.outcome}.")
        else:
            plans.append(f"{METHOD_LABEL[x.method]} on {_t(x.assumption_id)}: if it fails its rule, "
                         f"the verdict becomes {action}.")
    if plans:
        sections.append(ReportSection(title="What would change our mind", body=" ".join(plans)))

    learned = []
    for u in state.updates:
        kind = "A mocked result" if u.mocked else "A result"
        if u.verdict_before != u.verdict_after:
            change = f"changed the verdict from {u.verdict_before} to {u.verdict_after}"
        else:
            change = f"kept the verdict at {u.verdict_after}"
        learned.append(f"{kind} for {_t(u.claim_id)} ({_t(u.observation_claim_id)}) {change}.")
    if learned:
        sections.append(ReportSection(title="What we learned", body=" ".join(learned)))

    if top_down is not None:
        direction = "far above" if top_down.ratio >= 1 else "below"
        sections.append(ReportSection(
            title="Top-down cross-check",
            body=(f"The figure in the idea ({_t(top_down.claim_id)}) is {direction} what the bottom-up "
                  "model supports. It is kept as a user-supplied assumption and does not enter the model."),
        ))
    return headline, sections
