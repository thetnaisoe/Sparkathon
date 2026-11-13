"""Gate 2: the user fixes the experiment plan and its rules before any result exists."""

from __future__ import annotations

from datetime import datetime

from ..models import ApproveRequest, RunState
from .assess import refresh
from .rules import describe_rule


class GateError(ValueError):
    """The approval does not fit the run's current state."""


def approve_plan(state: RunState, body: ApproveRequest, *, now: datetime) -> None:
    """Apply Gate 2 decisions, lock the rules and move the run to `done` (mutates `state`)."""
    if body.gate != 2 or not body.approved:
        raise GateError("approve_plan handles an approved gate 2 only")
    if state.run.status != "awaiting_gate2":
        raise GateError(f"The run is {state.run.status}, not awaiting gate 2.")
    by_id = {x.id: x for x in state.experiments}
    decided = [d.experiment_id for d in body.decisions]
    unknown = [i for i in decided if i not in by_id]
    if unknown:
        raise GateError(f"Unknown experiments: {', '.join(unknown)}.")
    if len(set(decided)) != len(decided):
        raise GateError("Each experiment needs exactly one decision.")
    missing = [i for i in by_id if i not in decided]
    if missing:
        raise GateError(f"Missing a decision for: {', '.join(missing)}.")

    for d in body.decisions:
        x = by_id[d.experiment_id]
        if d.decision == "reject":
            x.status = "rejected"
            continue
        if d.decision == "edit":
            if d.hypothesis is not None:
                x.hypothesis = d.hypothesis
            if d.metric is not None:
                x.metric = d.metric
            if d.rule is not None:
                x.rule = d.rule
                x.pass_threshold, x.kill_criterion = describe_rule(d.rule)
            x.status = "edited"
        else:
            x.status = "accepted"
        x.locked_at = now

    state.run.status = "done"
    state.run.updated_at = now
    refresh(state, threshold=body.viability_threshold)
