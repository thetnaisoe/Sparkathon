# API contract for the Lovable app (schema v1)

The source of truth is `backend/app/models.py`. If anything here disagrees with it, the code wins. Changes are announced by message and bump `schema_version`.

## Endpoints

Base URL: *(sent separately after deploy)*. Keep it in one config constant.

| Call | Body | Returns | Right now (stub) |
|---|---|---|---|
| `GET /health` | – | `{status, schema_version, mode}` | works |
| `POST /runs` | `{idea_text, context: {sector?, region?, trl?}}` | `202 {run_id, status}` | always returns `demo-gate1` |
| `GET /runs/{id}` | – | `{run, steps, sources, claims, experiments}` | serves the 4 demo runs below |
| `POST /runs/{id}/approve` | see "Gate payloads" | `{ok, run_id, gate, approved}` | validates the payload, does nothing else |
| `GET /evals/latest` | – | `EvalResult[]` | empty list |

Types and a ready-made client: copy `lovable/src/lib/*.ts` into the Lovable repo. Setup steps and the prompt sequence are in [lovable/PROMPTS.md](lovable/PROMPTS.md).

## Demo runs

| run id | file | use it for |
|---|---|---|
| `demo-gate1` | `fixtures/gate1.json` | Gate 1 review |
| `demo-success` | `fixtures/assessment.json` | Run progress, Assessment, Experiments (Gate 2) |
| `demo-needs-input` | `fixtures/needs_input.json` | clarifying questions |
| `demo-refused` | `fixtures/refused.json` | RefusalCard |

All numbers in the fixtures are invented, and sources use `.example` domains. Never show them as real data.

## Which screen for which `run.status`

| status | screen |
|---|---|
| `queued`, `scoping`, `researching`, `modelling` | Run progress (`steps`, `sources` as they arrive) |
| `needs_input` | show `run.scope.reasons` and `run.scope.clarifying_questions`. Answering starts a **new** run with the answers appended to `idea_text` |
| `refused` | RefusalCard: `run.scope.reasons`, `run.scope.refusal_kind` |
| `awaiting_gate1` | Gate 1: edit `run.interpretation` |
| `awaiting_gate2` | Assessment + Experiments with Accept / Edit / Reject |
| `done` | Assessment + the approved plan |
| `failed` | error state with `run.error` |

## Claims (the ledger)

- **Badge** from `type`: `fact` / `estimate` / `assumption`.
- `origin: "user"` adds a "User-supplied" tag. These are always assumptions.
- **Drawer:**
  - fact: the source (`sources` row with `id == source_id`: `url`, `title`, `credibility_tier`) and the `quote` to highlight
  - estimate: `derivation.formula` and the claims listed in `derivation.input_claim_ids`
  - assumption: `note` and `evidence_strength`
- **Number:** if `p10/p50/p90` are set, show a range with `unit`; otherwise show `value` with `unit`. Some claims have neither.
- `evidence_strength` and `sensitivity` are 0–1. Sort the ledger by `sensitivity`. It's `null` for claims that aren't model drivers.
- Always show `note` when it's set (for example "Demoted from fact: quote not found").

## Report text

`assessment.headline` and `assessment.sections[].body` contain tokens like `{{c11}}`. Replace each token with that claim's formatted value or range plus a small badge, and make it open the drawer. The backend guarantees that the text contains no other numbers.

## Assessment

- `evidence_floor_met: false` → no `headline`. Show "Insufficient evidence", show `data_needed`, and label every range as assumption-based.
- `market_model.tornado` is already sorted by `swing`, largest first. Each bar has `low_output` and `high_output`. The output range is the claim with `id == market_model.output_claim_id`.
- `market_model.top_down_check` → show `ratio` and `note`. It can be `null`. `market_model` itself is `null` when no model was built.
- `cannot_determine` → "What we could not determine" panel. `limitations` → LimitationsBanner. `critic_notes` → skeptic panel.

## Gate payloads (`POST /runs/{id}/approve`)

```json
{ "gate": 1, "approved": true, "interpretation": { "...": "the edited run.interpretation" } }
{ "gate": 1, "approved": false, "note": "Wrong customer segment" }
{ "gate": 2, "approved": true, "decisions": [
    { "experiment_id": "x1", "decision": "accept" },
    { "experiment_id": "x2", "decision": "edit", "pass_threshold": "4 or more of 10" },
    { "experiment_id": "x3", "decision": "reject" } ] }
```

At Gate 1, unticked assumptions stay in `interpretation.assumptions` with `included: false`. At Gate 2, an `edit` decision carries only the fields that changed (`hypothesis`, `metric`, `pass_threshold`, `kill_criterion`).

## Supabase (once the project exists)

- Tables have the same shapes as above: `runs` (with `scope`, `interpretation` and `assessment` as JSON), `steps`, `sources`, `claims`, `experiments`, `eval_results`. Use the **anon** key, which can read.
- Realtime is on for `runs`, `steps`, `sources`, `claims` and `experiments`. Subscribe with the filter `run_id=eq.<id>`, or `id=eq.<id>` on `runs`. If Realtime fails, poll `GET /runs/{id}` every 3 s.
- The four demo runs are in the database too, under the same ids.
