// Display helpers for the claim ledger. Written by Dev B; keep in sync with backend/app/models.py.
import type { Claim, ExperimentMethod, RunState, RunStatus } from "./contract";

// ------------------------------------------------------------------ which screen

export type Screen = "progress" | "needs_input" | "refused" | "gate1" | "assessment" | "failed";

export function screenFor(status: RunStatus): Screen {
  switch (status) {
    case "needs_input":
      return "needs_input";
    case "refused":
      return "refused";
    case "awaiting_gate1":
      return "gate1";
    case "awaiting_gate2":
    case "done":
      return "assessment";
    case "failed":
      return "failed";
    default:
      return "progress"; // queued, scoping, researching, modelling
  }
}

/** Keep polling GET /runs/{id} only while the backend is working. */
export function isWorking(status: RunStatus): boolean {
  return screenFor(status) === "progress";
}

// ------------------------------------------------------------------ report text

export type TextPart = { kind: "text"; text: string } | { kind: "claim"; claimId: string };

/** Split report text on {{claim_id}} tokens so each claim can render as a clickable badge. */
export function splitClaimRefs(text: string): TextPart[] {
  const parts: TextPart[] = [];
  let last = 0;
  for (const match of text.matchAll(/\{\{([A-Za-z0-9_-]+)\}\}/g)) {
    const start = match.index ?? 0;
    if (start > last) parts.push({ kind: "text", text: text.slice(last, start) });
    parts.push({ kind: "claim", claimId: match[1] });
    last = start + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

export function claimById(state: RunState, id: string): Claim | undefined {
  return state.claims.find((c) => c.id === id);
}

// ------------------------------------------------------------------ numbers

const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

function formatter(unit: string | null): (n: number) => string {
  if (unit && /share/i.test(unit)) return (n) => `${Math.round(n * 100)}%`;
  if (unit && unit.startsWith("EUR")) return (n) => `€${compact.format(n)}`;
  return (n) => compact.format(n);
}

function unitSuffix(unit: string | null): string {
  if (!unit || /share/i.test(unit)) return "";
  const rest = unit.startsWith("EUR") ? unit.slice(3).trim() : unit;
  return rest ? ` ${rest}` : "";
}

/** "€240K (€70K–€820K) per year", "25% (10%–45%)", "2K organisations", or null if the claim has no number. */
export function formatClaimValue(claim: Claim): string | null {
  const fmt = formatter(claim.unit);
  const suffix = unitSuffix(claim.unit);
  if (claim.p10 !== null && claim.p50 !== null && claim.p90 !== null) {
    return `${fmt(claim.p50)} (${fmt(claim.p10)}–${fmt(claim.p90)})${suffix}`;
  }
  if (claim.value !== null) return `${fmt(claim.value)}${suffix}`;
  return null;
}

/** Format a market-model output (tornado ends, top-down values) in the output claim's unit. */
export function formatAmount(n: number, unit: string | null): string {
  return formatter(unit)(n);
}

// ------------------------------------------------------------------ labels

export const METHOD_LABEL: Record<ExperimentMethod, string> = {
  customer_interviews: "Customer interviews",
  landing_page_smoke_test: "Landing-page smoke test",
  letter_of_intent: "Letter-of-intent request",
  pricing_test: "Pricing test",
  technical_benchmark: "Technical benchmark vs incumbent",
  regulatory_expert_call: "Regulatory expert call",
};
