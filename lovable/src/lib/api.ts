// Typed client for the Lab-to-Market backend. Written by Dev B; types come from ./contract.
// Only change the two constants below.
import type { ApproveRequest, EvalResult, ResultCreate, RunCreate, RunCreated, RunState } from "./contract";
import { FIXTURES } from "./fixtures";

/** Backend URL. No trailing slash. */
export const API_BASE_URL = "https://lab2market-api.onrender.com";

/** true = serve the demo runs from ./fixtures without calling the backend. */
export const USE_FIXTURES = false;

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`API error ${status}: ${detail}`);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the backend. It may be waking up; try again in a minute.");
  }
  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // keep the raw text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

function fixture(runId: string): RunState {
  const state = FIXTURES[runId];
  if (!state) throw new ApiError(404, `run ${runId} not found`);
  return state;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () =>
    request<{ status: string; schema_version: number; llm_configured: boolean; search_configured: boolean }>(
      "/health",
    ),

  createRun: async (body: RunCreate): Promise<RunCreated> =>
    USE_FIXTURES ? { run_id: "demo-gate1", status: "awaiting_gate1" } : request<RunCreated>("/runs", post(body)),

  getRun: async (runId: string): Promise<RunState> =>
    USE_FIXTURES ? fixture(runId) : request<RunState>(`/runs/${encodeURIComponent(runId)}`),

  /** Gate 1 or Gate 2. Returns the updated run. 409 = wrong status or missing decisions (see error.detail). */
  approve: async (runId: string, body: ApproveRequest): Promise<RunState> =>
    USE_FIXTURES ? fixture(runId) : request<RunState>(`/runs/${encodeURIComponent(runId)}/approve`, post(body)),

  /** Enter an experiment result (only after Gate 2). Returns the updated run with a new entry in `updates`. */
  addResult: async (runId: string, body: ResultCreate): Promise<RunState> =>
    USE_FIXTURES
      ? fixture("demo-updated")
      : request<RunState>(`/runs/${encodeURIComponent(runId)}/results`, post(body)),

  /** Put the demo runs back to their starting state. */
  resetDemo: async (): Promise<{ reset: string[] }> =>
    USE_FIXTURES ? { reset: Object.keys(FIXTURES) } : request<{ reset: string[] }>("/demo/reset", post({})),

  latestEvals: async (): Promise<EvalResult[]> => (USE_FIXTURES ? [] : request<EvalResult[]>("/evals/latest")),
};
