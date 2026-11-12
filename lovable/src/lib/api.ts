// Typed client for the Lab-to-Market backend. Written by Dev B; types come from ./contract.
// Only change the two constants below.
import type { ApproveRequest, EvalResult, RunCreate, RunCreated, RunState } from "./contract";
import { FIXTURES } from "./fixtures";

/** Backend URL. Dev B sends it after deploy. No trailing slash. */
export const API_BASE_URL = "https://REPLACE-ME.onrender.com";

/** true = serve the demo runs from ./fixtures without calling the backend. */
export const USE_FIXTURES = true;

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
    throw new ApiError(response.status, await response.text());
  }
  return (await response.json()) as T;
}

function fixture(runId: string): RunState {
  const state = FIXTURES[runId];
  if (!state) throw new ApiError(404, `run ${runId} not found`);
  return state;
}

export const api = {
  health: () => request<{ status: string; schema_version: number; mode: string }>("/health"),

  createRun: async (body: RunCreate): Promise<RunCreated> =>
    USE_FIXTURES
      ? { run_id: "demo-gate1", status: "awaiting_gate1" }
      : request<RunCreated>("/runs", { method: "POST", body: JSON.stringify(body) }),

  getRun: async (runId: string): Promise<RunState> =>
    USE_FIXTURES ? fixture(runId) : request<RunState>(`/runs/${encodeURIComponent(runId)}`),

  approve: async (runId: string, body: ApproveRequest): Promise<{ ok: boolean }> =>
    USE_FIXTURES
      ? { ok: true }
      : request<{ ok: boolean }>(`/runs/${encodeURIComponent(runId)}/approve`, {
          method: "POST",
          body: JSON.stringify(body),
        }),

  latestEvals: async (): Promise<EvalResult[]> =>
    USE_FIXTURES ? [] : request<EvalResult[]>("/evals/latest"),
};
