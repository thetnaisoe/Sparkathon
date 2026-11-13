"""Experiment rules fixed in advance: plain-language text and evaluation.

The text shown on a card and the rule the code evaluates come from the same
ExperimentRule, so they cannot disagree.
"""

from __future__ import annotations

from ..models import ExperimentRule, RuleOutcome


def describe_rule(rule: ExperimentRule) -> tuple[str, str]:
    """Return (pass_threshold, kill_criterion) text for an experiment card."""
    action = "pivot" if rule.on_fail == "pivot" else "kill"
    return (
        f"{rule.pass_at_least} or more of {rule.trials}",
        f"Fewer than {rule.fail_below} of {rule.trials} → {action}",
    )


def evaluate_rule(rule: ExperimentRule, successes: int, trials_done: int) -> RuleOutcome:
    """Evaluate the rule on all results entered so far (exact integer arithmetic).

    Before the planned number of trials is reached, the outcome is decided only
    when it can no longer change: pass once successes reach pass_at_least, fail
    once even all remaining trials succeeding could not reach fail_below.
    From the planned number of trials on, the success rate is compared with the
    rule's thresholds as rates.
    """
    if trials_done <= 0:
        return "pending"
    if successes < 0 or successes > trials_done:
        raise ValueError("need 0 <= successes <= trials_done")
    if trials_done < rule.trials:
        if successes >= rule.pass_at_least:
            return "pass"
        if successes + (rule.trials - trials_done) < rule.fail_below:
            return "fail"
        return "pending"
    if successes * rule.trials >= rule.pass_at_least * trials_done:
        return "pass"
    if successes * rule.trials < rule.fail_below * trials_done:
        return "fail"
    return "inconclusive"
