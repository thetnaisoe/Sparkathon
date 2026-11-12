"""Shared contract between the backend and the Lovable app.

These Pydantic models are the single source of truth. To change them: message
Dev A first, bump SCHEMA_VERSION, regenerate fixtures/schema.json and update
fixtures/*.json in the same commit.

The label rules ("code, not the model, enforces the label") live in the
validators below, so nothing that breaks them can be stored or served.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

RunStatus = Literal[
    "queued",
    "scoping",
    "needs_input",
    "refused",
    "awaiting_gate1",
    "researching",
    "modelling",
    "awaiting_gate2",
    "done",
    "failed",
]
ScopeVerdict = Literal["in_scope", "needs_info", "refuse"]
RefusalKind = Literal["harmful_or_dual_use", "physically_infeasible", "outside_scope"]
StepName = Literal[
    "scoping", "researching", "verifying", "modelling", "ranking", "experiments", "composing"
]
StepStatus = Literal["running", "done", "failed", "skipped"]
ClaimType = Literal["fact", "estimate", "assumption"]
ClaimStatus = Literal["verified", "unverified"]
ClaimOrigin = Literal["research", "user", "model"]
CredibilityTier = Literal[
    "government", "peer_reviewed", "industry_report", "company_page", "blog", "unknown"
]
DriverKey = Literal[
    "n_target_orgs", "share_need_budget", "adoption_rate", "price_per_org", "achievable_share"
]
ExperimentMethod = Literal[
    "customer_interviews",
    "landing_page_smoke_test",
    "letter_of_intent",
    "pricing_test",
    "technical_benchmark",
    "regulatory_expert_call",
]
ExperimentStatus = Literal["proposed", "accepted", "edited", "rejected"]
EvalSystem = Literal["pipeline", "baseline", "analyst"]

# Evidence rules (starting values from the build plan; tune later).
ASSUMPTION_MAX_STRENGTH = 0.25
TIER_CAP: dict[str, float] = {
    "government": 1.0,
    "peer_reviewed": 1.0,
    "industry_report": 0.85,
    "company_page": 0.5,
    "blog": 0.4,
    "unknown": 0.3,
}

# Report text references claims as {{claim_id}}; numbers may only enter through these.
CLAIM_REF = re.compile(r"\{\{([A-Za-z0-9_-]+)\}\}")
_NUMBER = re.compile(r"\d[\d.,]*")


def claim_refs(text: str) -> list[str]:
    return CLAIM_REF.findall(text)


def find_orphan_numbers(text: str) -> list[str]:
    """Numbers in `text` that do not come from a {{claim_id}} reference."""
    return _NUMBER.findall(CLAIM_REF.sub("", text))


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- run input


class RunContext(Contract):
    sector: Optional[str] = None
    region: Optional[str] = None
    trl: Optional[int] = Field(default=None, ge=1, le=9)


class RunCreate(Contract):
    """Body of POST /runs."""

    idea_text: str = Field(min_length=1, max_length=12000)
    context: RunContext = Field(default_factory=RunContext)


class RunCreated(Contract):
    """Response of POST /runs."""

    run_id: str
    status: RunStatus


# --------------------------------------------------------------------------- scoping / Gate 1


class ScopeResult(Contract):
    verdict: ScopeVerdict
    reasons: list[str] = Field(min_length=1)
    clarifying_questions: list[str] = Field(default_factory=list)
    refusal_kind: Optional[RefusalKind] = None

    @model_validator(mode="after")
    def _consistent(self) -> ScopeResult:
        if self.verdict == "needs_info" and not self.clarifying_questions:
            raise ValueError("needs_info requires at least one clarifying question")
        if self.verdict == "refuse" and self.refusal_kind is None:
            raise ValueError("refuse requires a refusal_kind")
        if self.verdict != "refuse" and self.refusal_kind is not None:
            raise ValueError("refusal_kind is only allowed when verdict is refuse")
        return self


class DraftAssumption(Contract):
    """An assumption proposed at Gate 1. The user can edit or untick it."""

    id: str
    text: str
    driver: Optional[DriverKey] = None
    included: bool = True


class Interpretation(Contract):
    """The interpreted idea the user reviews and edits at Gate 1."""

    summary: str
    technology: str
    trl_guess: Optional[int] = Field(default=None, ge=1, le=9)
    target_customers: list[str]
    market_drivers: list[str]  # qualitative trends behind demand
    sector: Optional[str] = None
    region: Optional[str] = None
    assumptions: list[DraftAssumption] = Field(default_factory=list)


# --------------------------------------------------------------------------- ledger


class Step(Contract):
    id: str
    run_id: str
    name: StepName
    status: StepStatus
    message: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None


class Source(Contract):
    id: str
    run_id: str
    url: str
    domain: str
    title: Optional[str] = None
    credibility_tier: CredibilityTier = "unknown"
    fetched_at: datetime
    text_hash: Optional[str] = None  # sha256 of the extracted text; None if the fetch failed
    fetch_ok: bool = True
    error: Optional[str] = None


class Derivation(Contract):
    formula: str = Field(min_length=1)
    input_claim_ids: list[str] = Field(min_length=1)


class Claim(Contract):
    id: str
    run_id: str
    text: str
    type: ClaimType
    origin: ClaimOrigin = "research"
    driver: Optional[DriverKey] = None  # set when the claim is a market-model input
    value: Optional[float] = None
    unit: Optional[str] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    source_id: Optional[str] = None
    quote: Optional[str] = None
    derivation: Optional[Derivation] = None
    evidence_strength: float = Field(ge=0, le=1)
    sensitivity: Optional[float] = Field(default=None, ge=0, le=1)
    status: ClaimStatus = "unverified"
    note: Optional[str] = None

    @model_validator(mode="after")
    def _enforce_label(self) -> Claim:
        if self.type == "fact":
            if not (self.source_id and self.quote):
                raise ValueError("a fact needs source_id and quote")
            if self.status != "verified":
                raise ValueError("a fact needs a verified quote; demote it to an assumption")
        elif self.status == "verified":
            raise ValueError("only facts can have status 'verified'")

        if self.type == "estimate" and self.derivation is None:
            raise ValueError("an estimate needs a derivation (formula + input_claim_ids)")

        if self.type == "assumption" and self.evidence_strength > ASSUMPTION_MAX_STRENGTH:
            raise ValueError(f"an assumption's evidence_strength is at most {ASSUMPTION_MAX_STRENGTH}")

        if self.origin == "user" and self.type != "assumption":
            raise ValueError("figures supplied by the user are always assumptions")

        bounds = (self.p10, self.p50, self.p90)
        given = [b is not None for b in bounds]
        if any(given) and not all(given):
            raise ValueError("give all of p10, p50 and p90, or none")
        if (self.type == "estimate" or self.driver is not None) and not all(given):
            raise ValueError("estimates and model drivers need p10, p50 and p90")
        if all(given) and not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("need p10 <= p50 <= p90")
        return self


class Experiment(Contract):
    id: str
    run_id: str
    assumption_id: str
    hypothesis: str
    method: ExperimentMethod
    metric: str
    pass_threshold: str
    kill_criterion: str
    effort_days: float = Field(ge=0)
    cost_eur: float = Field(ge=0)
    updates_claim_ids: list[str] = Field(default_factory=list)  # estimates this experiment would update
    priority: float = Field(ge=0, le=1)  # normalised swing x (1 - evidence_strength)
    status: ExperimentStatus = "proposed"


# --------------------------------------------------------------------------- assessment


class TornadoBar(Contract):
    driver: DriverKey
    claim_id: str
    low_output: float  # median output with this driver at its P10, others at median
    high_output: float  # median output with this driver at its P90, others at median
    swing: float = Field(ge=0)


class TopDownCheck(Contract):
    claim_id: str  # the top-down figure being checked
    top_down_value: float
    bottom_up_value: float
    ratio: float  # top_down_value / bottom_up_value
    note: str


class MarketModel(Contract):
    output_label: str
    output_claim_id: str  # estimate claim holding the P10/P50/P90 of the output
    draws: int = 5000
    seed: int = 42
    tornado: list[TornadoBar]  # largest swing first
    top_down_check: Optional[TopDownCheck] = None

    @model_validator(mode="after")
    def _sorted(self) -> MarketModel:
        swings = [bar.swing for bar in self.tornado]
        if swings != sorted(swings, reverse=True):
            raise ValueError("tornado bars must be sorted by swing, largest first")
        return self


class ReportSection(Contract):
    title: str
    body: str  # markdown; reference claims as {{claim_id}}; no raw numbers

    @model_validator(mode="after")
    def _no_orphan_numbers(self) -> ReportSection:
        orphans = find_orphan_numbers(self.title) + find_orphan_numbers(self.body)
        if orphans:
            raise ValueError(f"numbers must come from claims via {{{{claim_id}}}}; found {orphans}")
        return self


class Assessment(Contract):
    evidence_floor_met: bool
    headline: Optional[str] = None  # None when the evidence floor is not met
    market_model: Optional[MarketModel] = None  # None when no model was built (e.g. infeasible)
    sections: list[ReportSection] = Field(default_factory=list)
    cannot_determine: list[str] = Field(default_factory=list)
    data_needed: list[str] = Field(default_factory=list)  # filled when the floor is not met
    limitations: list[str] = Field(min_length=1)
    critic_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _floor(self) -> Assessment:
        if not self.evidence_floor_met:
            if self.headline is not None:
                raise ValueError("no headline when the evidence floor is not met")
            if not self.data_needed:
                raise ValueError("list the data needed when the evidence floor is not met")
        if self.headline is not None and find_orphan_numbers(self.headline):
            raise ValueError("headline numbers must come from claims via {{claim_id}}")
        return self


# --------------------------------------------------------------------------- run


class Run(Contract):
    id: str
    schema_version: int = SCHEMA_VERSION
    created_at: datetime
    updated_at: datetime
    status: RunStatus
    idea_text: str
    context: RunContext = Field(default_factory=RunContext)
    scope: Optional[ScopeResult] = None
    interpretation: Optional[Interpretation] = None
    assessment: Optional[Assessment] = None
    error: Optional[str] = None

    @model_validator(mode="after")
    def _status_matches_content(self) -> Run:
        verdict = self.scope.verdict if self.scope else None
        if self.status == "refused" and verdict != "refuse":
            raise ValueError("status refused needs scope.verdict refuse")
        if self.status == "needs_input" and verdict != "needs_info":
            raise ValueError("status needs_input needs scope.verdict needs_info")
        if self.status == "awaiting_gate1" and (verdict != "in_scope" or self.interpretation is None):
            raise ValueError("status awaiting_gate1 needs an in_scope verdict and an interpretation")
        if self.status in ("awaiting_gate2", "done") and self.assessment is None:
            raise ValueError(f"status {self.status} needs an assessment")
        return self


class RunState(Contract):
    """Response of GET /runs/{id}: the run plus every ledger row that belongs to it."""

    run: Run
    steps: list[Step] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> RunState:
        rid = self.run.id
        for kind, rows in (
            ("step", self.steps),
            ("source", self.sources),
            ("claim", self.claims),
            ("experiment", self.experiments),
        ):
            ids = [row.id for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids")
            for row in rows:
                if row.run_id != rid:
                    raise ValueError(f"{kind} {row.id} belongs to run {row.run_id}, not {rid}")

        sources = {s.id: s for s in self.sources}
        claims = {c.id: c for c in self.claims}

        for c in self.claims:
            if c.source_id is not None and c.source_id not in sources:
                raise ValueError(f"claim {c.id} cites unknown source {c.source_id}")
            if c.type == "fact":
                src = sources[c.source_id]
                if not src.fetch_ok:
                    raise ValueError(f"fact {c.id} cites a source that was not fetched")
                cap = TIER_CAP[src.credibility_tier]
                if c.evidence_strength > cap:
                    raise ValueError(
                        f"fact {c.id}: evidence_strength {c.evidence_strength} exceeds the "
                        f"{src.credibility_tier} cap of {cap}"
                    )
            if c.derivation is not None:
                for ref in c.derivation.input_claim_ids:
                    if ref == c.id or ref not in claims:
                        raise ValueError(f"claim {c.id} derives from unknown claim {ref}")

        for x in self.experiments:
            target = claims.get(x.assumption_id)
            if target is None or target.type != "assumption":
                raise ValueError(f"experiment {x.id} must target an assumption claim")
            for ref in x.updates_claim_ids:
                if ref not in claims:
                    raise ValueError(f"experiment {x.id} updates unknown claim {ref}")

        a = self.run.assessment
        if a is not None:
            refs = claim_refs(a.headline or "")
            for section in a.sections:
                refs += claim_refs(section.body)
            if a.market_model is not None:
                mm = a.market_model
                out = claims.get(mm.output_claim_id)
                if out is None or out.type != "estimate":
                    raise ValueError("market_model.output_claim_id must be an estimate claim")
                refs += [bar.claim_id for bar in mm.tornado]
                if mm.top_down_check is not None:
                    refs.append(mm.top_down_check.claim_id)
            for ref in refs:
                if ref not in claims:
                    raise ValueError(f"assessment references unknown claim {ref}")
        return self


# --------------------------------------------------------------------------- gates


class ExperimentDecision(Contract):
    experiment_id: str
    decision: Literal["accept", "edit", "reject"]
    # Only for decision == "edit"; omitted fields stay unchanged.
    hypothesis: Optional[str] = None
    metric: Optional[str] = None
    pass_threshold: Optional[str] = None
    kill_criterion: Optional[str] = None

    @model_validator(mode="after")
    def _edit_has_changes(self) -> ExperimentDecision:
        edits = (self.hypothesis, self.metric, self.pass_threshold, self.kill_criterion)
        if self.decision == "edit" and all(e is None for e in edits):
            raise ValueError("an edit decision must change at least one field")
        if self.decision != "edit" and any(e is not None for e in edits):
            raise ValueError("only edit decisions may carry changed fields")
        return self


class ApproveRequest(Contract):
    """Body of POST /runs/{id}/approve."""

    gate: Literal[1, 2]
    approved: bool = True  # False = "send back"
    note: Optional[str] = None
    interpretation: Optional[Interpretation] = None  # gate 1: the (edited) interpretation
    decisions: list[ExperimentDecision] = Field(default_factory=list)  # gate 2

    @model_validator(mode="after")
    def _payload_matches_gate(self) -> ApproveRequest:
        if self.gate == 1:
            if self.decisions:
                raise ValueError("gate 1 takes an interpretation, not experiment decisions")
            if self.approved and self.interpretation is None:
                raise ValueError("approving gate 1 requires the interpretation")
        else:
            if self.interpretation is not None:
                raise ValueError("gate 2 takes experiment decisions, not an interpretation")
            if self.approved and not self.decisions:
                raise ValueError("approving gate 2 requires a decision for each experiment")
        return self


# --------------------------------------------------------------------------- evals


class EvalResult(Contract):
    id: str
    created_at: datetime
    eval_run_id: str  # one id per eval batch
    case_id: str  # e.g. "T1"
    system: EvalSystem
    metrics: dict[str, Optional[float]] = Field(default_factory=dict)
    passed: Optional[bool] = None
    notes: Optional[str] = None
    latency_s: Optional[float] = None
    cost_eur: Optional[float] = None
