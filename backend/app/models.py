"""Shared contract between the backend and the Lovable app.

These Pydantic models are the single source of truth. To change them: message
Dev A first, bump SCHEMA_VERSION, run scripts/build_fixtures.py and
scripts/export_contract.py, and commit the regenerated files together.

The label rules ("code, not the model, enforces the label") live in the
validators below, so nothing that breaks them can be stored or served.

v2 adds the learning loop: each experiment carries a rule fixed in advance
("k of n"), results update the tested assumption, and a Decision (continue,
pivot or kill) is computed by code from those rules.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 2

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
# experiment = a result the team typed in after running an experiment (we cannot verify it)
ClaimOrigin = Literal["research", "user", "model", "experiment"]
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
RuleOutcome = Literal["pending", "pass", "fail", "inconclusive"]
FailAction = Literal["pivot", "kill"]
Verdict = Literal["continue", "pivot", "kill"]
EvalSystem = Literal["pipeline", "baseline", "analyst"]

# Drivers that are shares of a population (0..1). Only these can be tested with a
# "k of n" experiment and updated with a Beta-Binomial model.
PROPORTION_DRIVERS: tuple[str, ...] = ("share_need_budget", "adoption_rate", "achievable_share")
MAGNITUDE_DRIVERS: tuple[str, ...] = ("n_target_orgs", "price_per_org")

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


def _check_range(p10: Optional[float], p50: Optional[float], p90: Optional[float]) -> bool:
    """Validate an optional P10/P50/P90 triple. Returns True if all three are set."""
    given = [b is not None for b in (p10, p50, p90)]
    if any(given) and not all(given):
        raise ValueError("give all of p10, p50 and p90, or none")
    if all(given) and not (p10 <= p50 <= p90):
        raise ValueError("need p10 <= p50 <= p90")
    return all(given)


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
    """An assumption proposed at Gate 1. The user can edit it; non-driver ones can be unticked.

    Assumptions with a `driver` are market-model inputs and must stay included.
    """

    id: str
    text: str
    driver: Optional[DriverKey] = None
    included: bool = True
    unit: Optional[str] = None
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None

    @model_validator(mode="after")
    def _range(self) -> DraftAssumption:
        has_range = _check_range(self.p10, self.p50, self.p90)
        if self.driver is not None:
            if not has_range:
                raise ValueError("a driver assumption needs p10, p50 and p90")
            if not self.included:
                raise ValueError("a driver assumption is a model input and cannot be excluded")
        return self


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

        if self.origin in ("user", "experiment") and self.type != "assumption":
            raise ValueError("figures entered by the user or the team are always assumptions")

        has_range = _check_range(self.p10, self.p50, self.p90)
        if (self.type == "estimate" or self.driver is not None) and not has_range:
            raise ValueError("estimates and model drivers need p10, p50 and p90")
        if self.driver in PROPORTION_DRIVERS and has_range and not (0 <= self.p10 and self.p90 <= 1):
            raise ValueError(f"{self.driver} is a share and must lie between 0 and 1")
        return self


class ExperimentRule(Contract):
    """Decision rule fixed in advance (locked at Gate 2) and evaluated by code.

    With the planned number of trials: pass if successes >= pass_at_least,
    fail if successes < fail_below, otherwise inconclusive.
    """

    trials: int = Field(ge=1, le=10000)
    pass_at_least: int = Field(ge=1)
    fail_below: int = Field(ge=1)
    on_fail: FailAction = "kill"

    @model_validator(mode="after")
    def _ordered(self) -> ExperimentRule:
        if not (self.fail_below <= self.pass_at_least <= self.trials):
            raise ValueError("need fail_below <= pass_at_least <= trials")
        return self


class Experiment(Contract):
    id: str
    run_id: str
    assumption_id: str  # the proportion-driver claim this experiment tests
    hypothesis: str
    method: ExperimentMethod
    metric: str
    rule: ExperimentRule
    pass_threshold: str  # written by code from `rule`
    kill_criterion: str  # written by code from `rule`
    effort_days: float = Field(ge=0)
    cost_eur: float = Field(ge=0)
    updates_claim_ids: list[str] = Field(default_factory=list)  # claims this experiment would update
    priority: float = Field(ge=0, le=1)  # normalised swing x (1 - evidence_strength)
    status: ExperimentStatus = "proposed"
    locked_at: Optional[datetime] = None  # set at Gate 2; the rule cannot change afterwards
    successes: int = Field(default=0, ge=0)  # cumulative results entered so far
    trials_done: int = Field(default=0, ge=0)
    outcome: RuleOutcome = "pending"
    decision_flip_prob: Optional[float] = Field(default=None, ge=0, le=1)  # chance the next result changes the verdict
    rank: Optional[int] = Field(default=None, ge=1)  # 1 = run next; None when finished or rejected

    @model_validator(mode="after")
    def _counts(self) -> Experiment:
        if self.successes > self.trials_done:
            raise ValueError("successes cannot exceed trials_done")
        return self


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


class Decision(Contract):
    """The verdict, computed by code from the rules fixed in advance. Never by the model."""

    verdict: Verdict
    provisional: bool  # true until the first experiment result is entered
    reasons: list[str] = Field(min_length=1)  # which rules fired, in plain words
    viability_threshold: float = Field(gt=0)  # same unit as the output; continue only if P50 >= this
    p_viable: float = Field(ge=0, le=1)  # share of Monte Carlo draws at or above the threshold


class Assessment(Contract):
    evidence_floor_met: bool
    headline: Optional[str] = None  # None when the evidence floor is not met
    market_model: Optional[MarketModel] = None  # None when no model was built (e.g. infeasible)
    decision: Optional[Decision] = None  # None when no model was built
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
        if (self.market_model is None) != (self.decision is None):
            raise ValueError("decision and market_model come together")
        return self


# --------------------------------------------------------------------------- learning loop


class BetaBelief(Contract):
    """Belief about a share, as a Beta(alpha, beta) distribution and its quantiles."""

    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    p10: float = Field(ge=0, le=1)
    p50: float = Field(ge=0, le=1)
    p90: float = Field(ge=0, le=1)


class OutputRange(Contract):
    p10: float
    p50: float
    p90: float


class UpdateEvent(Contract):
    """One entered result: prior -> evidence -> posterior, and what it changed."""

    id: str
    run_id: str
    created_at: datetime
    experiment_id: str
    claim_id: str  # the driver claim that was updated
    observation_claim_id: str  # the claim holding the entered result
    successes: int = Field(ge=0)
    trials: int = Field(ge=1)
    mocked: bool
    outcome: RuleOutcome  # the experiment's rule outcome after this result (all results so far)
    prior: BetaBelief
    posterior: BetaBelief
    output_before: OutputRange
    output_after: OutputRange
    p_viable_before: float = Field(ge=0, le=1)
    p_viable_after: float = Field(ge=0, le=1)
    verdict_before: Verdict
    verdict_after: Verdict
    rank_before: list[str]  # open experiment ids, best first
    rank_after: list[str]


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
    updates: list[UpdateEvent] = Field(default_factory=list)  # oldest first

    @model_validator(mode="after")
    def _references(self) -> RunState:
        rid = self.run.id
        for kind, rows in (
            ("step", self.steps),
            ("source", self.sources),
            ("claim", self.claims),
            ("experiment", self.experiments),
            ("update", self.updates),
        ):
            ids = [row.id for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids")
            for row in rows:
                if row.run_id != rid:
                    raise ValueError(f"{kind} {row.id} belongs to run {row.run_id}, not {rid}")

        sources = {s.id: s for s in self.sources}
        claims = {c.id: c for c in self.claims}
        experiments = {x.id: x for x in self.experiments}

        drivers = [c.driver for c in self.claims if c.driver is not None]
        if len(drivers) != len(set(drivers)):
            raise ValueError("each market-model driver must belong to exactly one claim")

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
            if target is None or target.driver not in PROPORTION_DRIVERS or target.type == "fact":
                raise ValueError(
                    f"experiment {x.id} must test a share driver (assumption or updated estimate)"
                )
            for ref in x.updates_claim_ids:
                if ref not in claims:
                    raise ValueError(f"experiment {x.id} updates unknown claim {ref}")

        for u in self.updates:
            if u.experiment_id not in experiments:
                raise ValueError(f"update {u.id} refers to unknown experiment {u.experiment_id}")
            for ref in (u.claim_id, u.observation_claim_id):
                if ref not in claims:
                    raise ValueError(f"update {u.id} refers to unknown claim {ref}")

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


# --------------------------------------------------------------------------- gates and results


class ExperimentDecision(Contract):
    experiment_id: str
    decision: Literal["accept", "edit", "reject"]
    # Only for decision == "edit"; omitted fields stay unchanged.
    hypothesis: Optional[str] = None
    metric: Optional[str] = None
    rule: Optional[ExperimentRule] = None

    @model_validator(mode="after")
    def _edit_has_changes(self) -> ExperimentDecision:
        edits = (self.hypothesis, self.metric, self.rule)
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
    viability_threshold: Optional[float] = Field(default=None, gt=0)  # gate 2: optional edit

    @model_validator(mode="after")
    def _payload_matches_gate(self) -> ApproveRequest:
        if self.gate == 1:
            if self.decisions or self.viability_threshold is not None:
                raise ValueError("gate 1 takes an interpretation, not experiment decisions")
            if self.approved and self.interpretation is None:
                raise ValueError("approving gate 1 requires the interpretation")
        else:
            if self.interpretation is not None:
                raise ValueError("gate 2 takes experiment decisions, not an interpretation")
            if self.approved and not self.decisions:
                raise ValueError("approving gate 2 requires a decision for each experiment")
        return self


class ResultCreate(Contract):
    """Body of POST /runs/{id}/results: the outcome of running one experiment."""

    experiment_id: str
    successes: int = Field(ge=0)
    trials: int = Field(ge=1, le=10000)
    mocked: bool  # required: say whether this is a real result or a demo value
    note: Optional[str] = None

    @model_validator(mode="after")
    def _counts(self) -> ResultCreate:
        if self.successes > self.trials:
            raise ValueError("successes cannot exceed trials")
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
