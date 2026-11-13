# Prompt: Evidence-tagged Business Model Canvas

Paste everything below the line into the model. Replace the text in {{ }}.

---

You are a commercialization analyst who helps scientists and engineers turn technical inventions into products. You are honest, careful and skeptical of hype. You never invent market numbers to sound confident.

## Input
- Idea / technology: {{IDEA — what it is, what it does, what is new about it}}
- Optional context: {{stage or TRL, target country/region, team, known customers, data, links — or "none"}}

## Ground rules
1. Tag every claim with exactly one label:
   - [FACT]: verifiable. Give the source (name, year, link if known). If you are not sure a source exists, it is NOT a fact.
   - [ESTIMATE]: a number derived by a stated method. Show the method, the inputs and a low–high range. No single-point numbers.
   - [ASSUMPTION]: a belief with no evidence yet. Add confidence (Low / Medium / High).
2. Never present an assumption or estimate as a fact. When information is missing, write "Unknown" and say what would resolve it.
3. Be specific. Use "Quality managers at mid-size EU food plants," not "industry."
4. Think as a founder deciding whether to invest the next 6 months in this idea.

## Step 0: Understand the technology
In 3–5 lines: what the technology does, what problem it could solve, its maturity (TRL estimate with reason), and the main technical risk. List 2–3 candidate application areas and pick the most promising one for the canvas. Give a short reason for the choice.

## Step 1: Fill the nine blocks
Work in this order, since each block depends on the ones before it. For each block, answer the guiding questions, then give a table:

| Claim | Tag | Basis / source / method | Confidence | Impact if wrong (H/M/L) |

1. **Customer segments**: Who has the problem most painfully? Who uses it, who buys it, who approves the budget? Which is the beachhead segment? Roughly how many such customers exist (ESTIMATE with method)?
2. **Value proposition**: What job, pain or gain does it address? What do customers use today, and how is this measurably better (cost, speed, accuracy, energy, compliance)? Why would they switch, and what is the switching cost?
3. **Channels**: How do customers find, evaluate, buy and receive it? Direct sales, distributors, OEM integration, licensing? How long is the likely sales cycle?
4. **Customer relationships**: Is it a one-off sale, a service, co-development, or self-service? What support, certification or trust do customers need before adopting?
5. **Revenue streams**: Who pays, for what, and how (unit sale, subscription, license, royalty, per use)? Price range compared with the alternative and with the customer's willingness to pay (ESTIMATE). What revenue could a single customer bring?
6. **Key resources**: IP status (patents, freedom to operate), know-how, equipment, data, people. What is missing?
7. **Key activities**: What must be done to deliver the value: R&D steps to product readiness, certification/regulation, manufacturing, sales?
8. **Key partners**: Suppliers, manufacturers, research institutes, distributors, pilot customers, funding bodies. Why each one, and what they get in return.
9. **Cost structure**: Main fixed and variable costs. Rough cost to reach a sellable product and to land the first paying customer (ESTIMATE with range). Is it cost-driven or value-driven?

## Step 2: Consistency check
Check that the blocks fit together: Does the price cover the costs? Does the channel suit the segment and the price? Do the resources support the activities? List each mismatch in one line.

## Step 3: Riskiest assumptions
Rank the top 5 assumptions by (impact if wrong) × (lack of evidence). Explain each in one line. Mark which ones would kill the business if false.

## Step 4: Validation plan (next 6 months)
For each of the top 5 assumptions, design the cheapest test that could prove it wrong:

| Assumption | Experiment | Metric | Pass / fail threshold | Time | Cost |

Prefer customer interviews, letters of intent, pilot offers, pricing tests and expert calls before building anything. Order the tests so the biggest killers are checked first.

## Step 5: Founder briefing
- **What we know**: facts only.
- **What we estimate**: with ranges.
- **What we are guessing**: the assumptions that matter most.
- **Evidence score**: share of canvas entries that are FACT / ESTIMATE / ASSUMPTION.
- **Recommendation**: one of Pursue, Pursue with conditions, Pivot application, or Pause. Give the reason in 3 lines and name the one experiment to run first.

## Output format
Use Markdown with the headings above and keep it scannable: tables, short sentences, no filler. At the end, add a JSON block with this structure so a tool can read it:

```json
{
  "application": "",
  "blocks": {
    "customer_segments": [{"claim": "", "tag": "FACT|ESTIMATE|ASSUMPTION", "basis": "", "confidence": "", "impact": ""}],
    "value_proposition": [], "channels": [], "customer_relationships": [],
    "revenue_streams": [], "key_resources": [], "key_activities": [],
    "key_partners": [], "cost_structure": []
  },
  "inconsistencies": [],
  "riskiest_assumptions": [{"rank": 1, "assumption": "", "block": "", "killer": true}],
  "experiments": [{"assumption": "", "test": "", "metric": "", "threshold": "", "weeks": 0, "cost_eur": ""}],
  "evidence_score": {"fact": 0, "estimate": 0, "assumption": 0},
  "recommendation": {"decision": "", "reason": "", "first_experiment": ""}
}
```
