"""One tiny structured LLM call to check the configured provider end to end.

    cd backend
    LLM_PROVIDER=anthropic LLM_API_KEY=... .venv/bin/python scripts/llm_smoke.py
    LLM_PROVIDER=mistral LLM_API_KEY=... LLM_MODEL=mistral-large-latest .venv/bin/python scripts/llm_smoke.py

Prints the provider, model, parsed reply, latency and token usage. Never prints the key.
Exit codes: 0 ok, 1 call failed, 2 not configured, 3 refused.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field  # noqa: E402

from app.env import load_env  # noqa: E402

load_env()
from app.llm import LLMError, LLMNotConfigured, LLMRefused, get_client  # noqa: E402


class SmokeReply(BaseModel):
    capital: str = Field(description="Capital city of the country")
    continent: Literal["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
    confident: bool


def main() -> int:
    if not os.getenv("LLM_PROVIDER", "").strip() or not os.getenv("LLM_API_KEY", "").strip():
        print("LLM not configured: set LLM_PROVIDER (anthropic | openai | azure | mistral | gemini | groq | "
              "openrouter) and LLM_API_KEY; LLM_MODEL is required for the OpenAI-compatible ones.", file=sys.stderr)
        return 2
    try:
        client = get_client()
    except LLMNotConfigured as exc:
        print(f"LLM not configured: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"LLM setup failed: {exc}", file=sys.stderr)
        return 1

    provider = client.provider
    print(f"provider: {provider.name}")
    print(f"model:    {provider.model}")
    started = time.monotonic()
    try:
        reply = client.structured(
            SmokeReply,
            system="You answer geography questions as JSON.",
            user="What is the capital of France, and on which continent is France?",
            purpose="smoke",
            run_id="smoke",
        )
    except LLMRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    except LLMError as exc:
        print(f"call failed: {exc}", file=sys.stderr)
        return 1
    latency = time.monotonic() - started
    calls = [c for c in client.calls if c.run_id == "smoke"]
    tokens_in = sum(c.input_tokens or 0 for c in calls)
    tokens_out = sum(c.output_tokens or 0 for c in calls)
    print(f"reply:    {json.dumps(reply.model_dump())}")
    print(f"latency:  {latency:.1f}s over {len(calls)} attempt(s)")
    print(f"tokens:   in={tokens_in} out={tokens_out}")
    if getattr(provider, "mode", None):
        print(f"json mode: {provider.mode}")
    if getattr(provider, "fallbacks", None) is not None:
        print(f"fallbacks: {'on' if provider.fallbacks else 'off'}; effort: {getattr(provider, 'effort', None) or 'model default'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
