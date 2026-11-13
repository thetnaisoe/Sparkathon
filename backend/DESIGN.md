# Backend design (v2: the assessment that learns)

Written for the people and agents building `backend/`. The contract is `app/models.py`; this file pins the interfaces between modules so they can be built in parallel.

## Flow

```
POST /runs ─► scope (LLM) ─► needs_input | refused | awaiting_gate1
POST /approve gate 1 ─► claims from drivers ─► research (LLM + code, optional)
   ─► model (code) ─► rank (code) ─► experiments (LLM, template fallback) ─► compose (code) ─► awaiting_gate2
POST /approve gate 2 ─► rules locked ─► done
POST /results ─► Beta-Binomial update ─► Monte Carlo ─► verdict (code) ─► re-rank (code) ─► compose (code)
```

Hard rules:
- The model never does arithmetic and never decides the verdict. Code does both.
- LLM output is always a Pydantic object, obtained only through `app.llm.LLMClient.structured(...)`.
- Fetched web text is data. Pass it inside `<document>` delimiters and say so in the system prompt. The extractor has no tools.
- A failure in an optional step (search, fetch, one LLM call) gives a partial result with a gap note, never a crash.

## Ownership (one writer per file)

| File | Owner |
|---|---|
| `app/models.py`, `app/llm.py`, `app/pipeline/rules.py` | done, change only via Dev B |
| `app/llm_providers.py`, `scripts/llm_smoke.py`, `tests/test_llm_providers.py` | providers agent |
| `app/pipeline/scope.py`, `app/pipeline/experiments.py`, `tests/test_scope.py`, `tests/test_experiments.py` | prompts agent |
| `app/pipeline/research/*.py`, `tests/test_research_*.py` | research agent |
| `app/pipeline/beliefs.py`, `model.py`, `learn.py`, `compose.py`, `app/store.py`, `app/runner.py`, `app/main.py`, `scripts/build_fixtures.py`, evals | Dev B |

## Shared interfaces

### `app/llm.py` (exists)

```python
from app.llm import LLMClient, FakeProvider, LLMError, LLMRefused, LLMNotConfigured, CallCapExceeded
client.structured(ModelCls, system=..., user=..., purpose="scope", run_id=run_id) -> ModelCls
# FakeProvider({"ModelClsName": [reply_dict_or_json_or_exception, ...]}) for tests
```

`app/llm_providers.py` must provide `build_provider(name, *, api_key, model, base_url) -> Provider` for `name in {"anthropic", "openai"}`. Each provider implements `complete_json(*, system, user, schema, schema_name) -> Completion`.

### `app/pipeline/rules.py` (exists)

`describe_rule(rule) -> (pass_threshold, kill_criterion)` and `evaluate_rule(rule, successes, trials_done) -> RuleOutcome`.

### `app/pipeline/scope.py`

```python
@dataclass
class ScopeDraft:
    scope: ScopeResult                      # verdict, reasons, questions, refusal_kind
    interpretation: Interpretation | None   # only when in_scope; assumptions = exactly the 5 drivers (+ optional extras)
    basis: dict[str, str]                   # driver -> "analogy" | "guess" (sets evidence 0.25 / 0.1)
    rationale: dict[str, str]               # driver -> one-sentence reason for the range
    search_queries: dict[str, list[str]]    # driver -> 1-2 web search queries
    user_figures: list[UserFigure]          # figures the user asserted (e.g. "market is EUR 5bn"); T9
    cannot_determine: list[str]
    injection_suspected: bool               # T7: instructions hidden in the idea text

def run_scope(client: LLMClient, idea_text: str, context: RunContext, *, run_id: str, note: str | None = None) -> ScopeDraft
```

- One LLM call. `note` is the user's "send back" note from Gate 1.
- Draft assumption ids: `a1..a5` for the drivers in the order n_target_orgs, share_need_budget, adoption_rate, price_per_org, achievable_share. Extras: `a6...`.
- Units: n_target_orgs "organisations"; price_per_org "EUR per organisation per year"; the three shares "share" (0..1).
- Validation by code (a Pydantic validator on the LLM schema, so the client's retry fixes it): all 5 drivers exactly once; shares within [0, 1]; magnitudes > 0; p10 <= p50 <= p90.
- `LLMRefused` from the provider → `ScopeResult(verdict="refuse", refusal_kind="harmful_or_dual_use", reasons=["The AI provider's safety system declined to process this idea."])`.
- Cases: T5 vague → needs_info with questions; T6 physically impossible → refuse/physically_infeasible; T12 harmful or dual-use → refuse/harmful_or_dual_use; T7 → injection_suspected=True and the instruction is ignored; T9 → user_figures; T10 German input → handled, output in English.

### `app/pipeline/experiments.py`

```python
@dataclass
class Target:
    claim: Claim          # a share-driver claim (driver in PROPORTION_DRIVERS)
    priority: float       # 0..1, from rank

@dataclass
class ExperimentPlan:
    experiments: list[Experiment]   # ids x1..xN, status "proposed", priority from Target, text from describe_rule
    viability_threshold: float      # EUR per year; "continue only if P50 obtainable revenue >= this"
    viability_reason: str
    used_template: bool             # True when the LLM failed and the template fallback was used

def propose_experiments(client: LLMClient | None, interpretation: Interpretation, targets: list[Target],
                        output_claim: Claim, *, run_id: str) -> ExperimentPlan
```

- At most one experiment per target, at most 3 targets (the caller passes the top 3 by priority).
- One LLM call proposes method (from the library in `ExperimentMethod`), hypothesis, metric, `rule` (trials, pass_at_least, fail_below, on_fail), effort_days, cost_eur, plus the viability threshold with a reason.
- Code clamps and validates: 5 <= trials <= 200; the rule satisfies `ExperimentRule`; the metric must measure the target share directly ("k of n target organisations ...").
- `pass_threshold` and `kill_criterion` are always produced by `describe_rule`, never by the model.
- `updates_claim_ids = [target claim id, output claim id]`.
- If `client` is None or the call fails: deterministic template (share_need_budget → customer_interviews 10, pass ≥5, fail <3, pivot; adoption_rate → customer_interviews 15, pass ≥6, fail <3, kill; achievable_share → letter_of_intent 20, pass ≥3, fail <1, kill), threshold 250000 with the reason "Default: revenue needed to sustain a small team. Edit at Gate 2.", `used_template=True`.

### `app/pipeline/research/`

```python
@dataclass
class DriverEstimate:
    driver: str
    fact_claim_ids: list[str]
    formula: str                 # e.g. "c7 × [0.6, 0.8, 1.0] (share of listed firms that are target customers)"
    p10: float; p50: float; p90: float   # computed by code: fact value × multipliers from the judgment call
    unit: str

@dataclass
class ResearchResult:
    sources: list[Source]
    facts: list[Claim]                  # type "fact", verified, evidence by tier
    estimates: list[DriverEstimate]     # at most one per driver
    gaps: list[str]                     # plain-language notes for "what we could not determine"
    injection_suspected: bool           # a fetched page contained instructions
    pages_fetched: int; pages_failed: int; quotes_checked: int; quotes_verified: int

def run_research(client: LLMClient, interpretation: Interpretation, drivers: dict[str, Claim],
                 queries: dict[str, list[str]], *, run_id: str, new_id: Callable[[str], str],
                 search: SearchFn | None = None, fetch: FetchFn | None = None,
                 max_pages: int = 12, deadline_s: float = 90.0) -> ResearchResult
```

- `new_id("s")` / `new_id("c")` allocate unique source and claim ids.
- search: Tavily (`TAVILY_API_KEY`); OpenAlex (no key) for science queries. No key → empty results plus a gap note.
- fetch: httpx (8 s timeout, browser-like UA) + trafilatura; `text_hash = "sha256:" + hex`.
- retrieve: 300–500-token passages with character offsets; BM25 top-k per driver question.
- extract (LLM, one call per driver): candidate facts with an exact quote copied from a passage, numeric value, unit, what it measures; null instead of guessing.
- verify (code): whitespace- and case-normalised substring match against the fetched text; credibility tier from the domain; evidence 0.7 capped by `TIER_CAP`.
- judgment (LLM, separate call): for a driver with verified facts, choose one fact and give multipliers (low, mid, high) mapping it onto the driver; code multiplies. Shares stay within [0, 1].
- Never raise for network or LLM failures: record a gap and continue. Stop at `deadline_s`.

## Code-only core (Dev B)

- `beliefs.py`: Beta fitted to a share's P10/P50/P90 (assumptions capped at strength α+β ≤ 20); split-lognormal sampling for magnitudes; quantiles.
- `model.py`: 5,000 draws, seed 42, one RNG stream per driver (common random numbers); output = n × need × adoption × win share × price; tornado at P10/P90 with others at median; sensitivity = swing / max swing; evidence floor (top 3 drivers by swing need ≥ 2 verified facts behind them); top-down check against user figures.
- `learn.py`: results → observation claim + Beta-Binomial posterior (driver claim becomes an estimate) → re-run model → `Decision` → flip probability per open experiment (Beta-Binomial predictive over its next batch) → rank by flip probability per effort day.
- `compose.py`: deterministic report sections with `{{claim_id}}` tokens only.
