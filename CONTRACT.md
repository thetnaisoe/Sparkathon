# API contract for the Lovable app (schema v2: the assessment that learns)

The source of truth is `backend/app/models.py`. Types for the app are generated into `lovable/src/lib/contract.ts`, and the setup steps and prompts are in [lovable/PROMPTS.md](lovable/PROMPTS.md).

Base URL: `https://lab2market-api.onrender.com` (already set in `api.ts`).

## Endpoints

| Call | Body | Returns |
|---|---|---|
| `GET /health` | – | `{status, schema_version, llm_configured, search_configured, runs}` |
| `POST /runs` | `{idea_text, context: {sector?, region?, trl?}}` | `202 {run_id, status}` |
| `GET /runs/{id}` | – | `RunState` = `{run, steps, sources, claims, experiments, updates}` |
| `POST /runs/{id}/approve` | Gate 1 or Gate 2 payload (below) | `RunState` |
| `POST /runs/{id}/results` | `{experiment_id, successes, trials, mocked, note?}` | `RunState` with a new entry in `updates` |
| `POST /demo/reset` | – | `{reset: [ids]}`: puts the demo runs back to their starting state |
| `GET /evals/latest` | – | `EvalResult[]` for the Evidence page |

Errors: `409` means the request doesn't fit the run's status, for example results before Gate 2 or a missing Gate 2 decision. `422` means the body is invalid. Show `error.detail` from `api.ts`.

Storage is in memory on the backend; Supabase is not used any more. Poll `GET /runs/{id}` every 3 s while `isWorking(status)` is true.

## Demo runs

| run id | use it for |
|---|---|
| `demo-gate1` | Gate 1 review. Approving it runs the real pipeline offline, using template experiments. |
| `demo-success` | Assessment + Experiments at Gate 2 (`awaiting_gate2`). Approve it, then enter results. |
| `demo-updated` | After Gate 2 plus one mocked result ("2 of 10"). Use it to design the before/after panel. |
| `demo-needs-input` | Clarifying questions |
| `demo-refused` | RefusalCard |

All numbers in the demo runs are invented, but everything derived from them (ranges, tornado, verdict, update) is computed by the real code.

## Which screen for which `run.status`

| status | screen |
|---|---|
| `queued`, `scoping`, `researching`, `modelling` | Run progress |
| `needs_input` | clarifying questions; answering starts a **new** run |
| `refused` | RefusalCard |
| `awaiting_gate1` | Gate 1 |
| `awaiting_gate2` | Assessment + Experiments, approve the plan |
| `done` | Assessment + approved plan + **results forms** |
| `failed` | error state with `run.error` |

## The learning loop (new in v2)

**`run.assessment.decision`** is the verdict, computed by code from the rules fixed in advance and never by the AI.
- `verdict` is `continue`, `pivot` or `kill`.
- `provisional` is true until the first result is entered.
- `reasons` explains which rules fired, in plain words.
- `viability_threshold` (EUR per year) and `p_viable` give "chance of reaching it".

**Each experiment** has:
- `rule`: `{trials, pass_at_least, fail_below, on_fail}`. `pass_threshold` and `kill_criterion` are the same rule as text.
- `locked_at`: set at Gate 2. After that the rule can't change.
- `successes` and `trials_done`: the totals entered so far.
- `outcome`: `pending`, `pass`, `fail` or `inconclusive`. Once it is pass or fail it stays decided.
- `decision_flip_prob`: the chance that the next batch changes the verdict.
- `rank`: 1 means run this next; `null` means finished or rejected.

**Each entered result** adds an `UpdateEvent` to `updates` (oldest first):
- `prior` → `posterior`: the tested share as a Beta, with p10/p50/p90.
- `output_before` → `output_after`: revenue range.
- `p_viable_before` → `p_viable_after`.
- `verdict_before` → `verdict_after`.
- `rank_before` → `rank_after`: experiment ids, best first.
- `mocked`: whether the result was a demo value.

It also adds an observation claim (`origin: "experiment"`, always an Assumption). The tested claim becomes an Estimate whose derivation shows the update.

## Claims (the ledger)

These are unchanged from v1, plus one new origin: `origin: "experiment"` marks a result entered by the team. Show it with a "Team result" tag, and **"MOCKED"** when the text says so. `origin: "user"` stays "User-supplied". The drawer works as before: a fact shows its source and quote, an estimate shows its formula and inputs, an assumption shows its note and evidence.

## Report text

`assessment.headline` and `assessment.sections[].body` contain `{{claim_id}}` tokens. Render them with `splitClaimRefs`. The text contains no other numbers.

## Gate payloads

```json
{ "gate": 1, "approved": true, "interpretation": { "...": "the edited run.interpretation" } }
{ "gate": 1, "approved": false, "note": "Wrong customer segment" }
{ "gate": 2, "approved": true, "viability_threshold": 200000, "decisions": [
    { "experiment_id": "x1", "decision": "accept" },
    { "experiment_id": "x2", "decision": "edit", "rule": { "trials": 12, "pass_at_least": 6, "fail_below": 4, "on_fail": "kill" } },
    { "experiment_id": "x3", "decision": "reject" } ] }
```

Gate 1 assumptions now carry `unit`, `p10`, `p50` and `p90`. Ranges may be edited. Assumptions with a `driver` are model inputs and cannot be unticked, and all five drivers must stay present.

Gate 2 needs one decision per experiment. An `edit` may change `hypothesis`, `metric` and `rule`. `viability_threshold` is optional.

## What changed from v1

- `approve` returns the full `RunState`, not `{ok}`.
- Experiments gained `rule`, `locked_at`, `outcome`, `successes`, `trials_done`, `decision_flip_prob` and `rank`.
- The Gate 2 edit fields `pass_threshold` and `kill_criterion` were replaced by `rule`.
- New: `assessment.decision`, `RunState.updates`, `POST /results`, `POST /demo/reset`, and the `demo-updated` run.
