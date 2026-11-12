"""Label rules and contract checks (build plan section 8, unit tests)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models import (
    ApproveRequest,
    Assessment,
    Claim,
    ExperimentDecision,
    ReportSection,
    RunState,
    ScopeResult,
    find_orphan_numbers,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def claim(**overrides) -> dict:
    base = {
        "id": "c1",
        "run_id": "r1",
        "text": "Some claim",
        "type": "assumption",
        "evidence_strength": 0.1,
    }
    base.update(overrides)
    return base


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ fixtures


@pytest.mark.parametrize("name", ["gate1.json", "assessment.json", "needs_input.json", "refused.json"])
def test_fixtures_validate(name):
    RunState.model_validate(load(name))


# ------------------------------------------------------------------ label rules


def test_fact_without_quote_is_rejected():
    with pytest.raises(ValidationError, match="source_id and quote"):
        Claim.model_validate(claim(type="fact", source_id="s1", status="verified", evidence_strength=0.7))


def test_fact_without_source_is_rejected():
    with pytest.raises(ValidationError, match="source_id and quote"):
        Claim.model_validate(claim(type="fact", quote="x", status="verified", evidence_strength=0.7))


def test_unverified_fact_is_rejected():
    with pytest.raises(ValidationError, match="verified quote"):
        Claim.model_validate(claim(type="fact", source_id="s1", quote="x", evidence_strength=0.7))


def test_only_facts_can_be_verified():
    with pytest.raises(ValidationError, match="only facts"):
        Claim.model_validate(claim(status="verified"))


def test_estimate_without_derivation_is_rejected():
    with pytest.raises(ValidationError, match="derivation"):
        Claim.model_validate(claim(type="estimate", p10=1, p50=2, p90=3, evidence_strength=0.5))


def test_estimate_without_range_is_rejected():
    derivation = {"formula": "c2 x c3", "input_claim_ids": ["c2", "c3"]}
    with pytest.raises(ValidationError, match="p10, p50 and p90"):
        Claim.model_validate(claim(type="estimate", derivation=derivation, evidence_strength=0.5))


def test_derivation_needs_inputs():
    with pytest.raises(ValidationError):
        Claim.model_validate(
            claim(type="estimate", p10=1, p50=2, p90=3, derivation={"formula": "x", "input_claim_ids": []})
        )


def test_range_must_be_ordered():
    with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
        Claim.model_validate(claim(p10=3, p50=2, p90=1))


def test_partial_range_is_rejected():
    with pytest.raises(ValidationError, match="all of p10"):
        Claim.model_validate(claim(p10=1, p90=3))


def test_model_driver_needs_range():
    with pytest.raises(ValidationError, match="model drivers"):
        Claim.model_validate(claim(driver="adoption_rate"))


def test_assumption_strength_is_capped():
    with pytest.raises(ValidationError, match="at most"):
        Claim.model_validate(claim(evidence_strength=0.5))


def test_user_figures_are_assumptions():
    with pytest.raises(ValidationError, match="supplied by the user"):
        Claim.model_validate(
            claim(
                type="fact",
                origin="user",
                source_id="s1",
                quote="x",
                status="verified",
                evidence_strength=0.7,
            )
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Claim.model_validate(claim(confidence=0.9))


# ------------------------------------------------------------------ run-level references


def test_fact_strength_capped_by_source_tier():
    data = load("assessment.json")
    c2 = next(c for c in data["claims"] if c["id"] == "c2")  # cites a blog
    c2["evidence_strength"] = 0.7
    with pytest.raises(ValidationError, match="blog cap"):
        RunState.model_validate(data)


def test_fact_on_failed_fetch_is_rejected():
    data = load("assessment.json")
    c1 = next(c for c in data["claims"] if c["id"] == "c1")
    c1["source_id"] = "s4"  # the source that timed out
    with pytest.raises(ValidationError, match="not fetched"):
        RunState.model_validate(data)


def test_unknown_claim_reference_in_report_is_rejected():
    data = load("assessment.json")
    data["run"]["assessment"]["sections"][0]["body"] += " See {{c99}}."
    with pytest.raises(ValidationError, match="unknown claim c99"):
        RunState.model_validate(data)


def test_experiment_must_target_an_assumption():
    data = load("assessment.json")
    data["experiments"][0]["assumption_id"] = "c1"  # a fact
    with pytest.raises(ValidationError, match="target an assumption"):
        RunState.model_validate(data)


def test_status_must_match_content():
    data = load("assessment.json")
    data["run"]["assessment"] = None
    with pytest.raises(ValidationError, match="needs an assessment"):
        RunState.model_validate(data)


# ------------------------------------------------------------------ report rule


def test_orphan_numbers_are_found():
    assert find_orphan_numbers("Revenue is {{c11}} from 2,000 firms") == ["2,000"]
    assert find_orphan_numbers("Revenue is {{c11}}.") == []


def test_report_section_rejects_raw_numbers():
    with pytest.raises(ValidationError, match="must come from claims"):
        ReportSection(title="Bottom line", body="Revenue could reach EUR 240,000.")


def test_no_headline_without_evidence_floor():
    with pytest.raises(ValidationError, match="no headline"):
        Assessment(evidence_floor_met=False, headline="{{c1}}", data_needed=["x"], limitations=["x"])


def test_floor_not_met_requires_data_needed():
    with pytest.raises(ValidationError, match="data needed"):
        Assessment(evidence_floor_met=False, limitations=["x"])


# ------------------------------------------------------------------ scope and gates


def test_needs_info_requires_questions():
    with pytest.raises(ValidationError, match="clarifying question"):
        ScopeResult(verdict="needs_info", reasons=["vague"])


def test_refusal_requires_kind():
    with pytest.raises(ValidationError, match="refusal_kind"):
        ScopeResult(verdict="refuse", reasons=["harmful"])


def test_gate1_approval_requires_interpretation():
    with pytest.raises(ValidationError, match="interpretation"):
        ApproveRequest(gate=1, approved=True)


def test_gate1_send_back_needs_no_interpretation():
    ApproveRequest(gate=1, approved=False, note="Wrong customer segment")


def test_gate2_approval_requires_decisions():
    with pytest.raises(ValidationError, match="decision"):
        ApproveRequest(gate=2, approved=True)


def test_edit_decision_must_change_something():
    with pytest.raises(ValidationError, match="at least one field"):
        ExperimentDecision(experiment_id="x1", decision="edit")
