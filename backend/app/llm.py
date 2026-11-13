"""Provider-agnostic structured LLM calls.

Every call returns a validated Pydantic object. The client enforces a per-run
call cap, retries once when the reply does not match the schema, and logs every
call. Providers (Anthropic, OpenAI-compatible) live in app/llm_providers.py and
are chosen with environment variables:

    LLM_PROVIDER   anthropic | openai          (openai = any OpenAI-compatible API:
                                                Azure v1, Mistral, Gemini, Groq, ...)
    LLM_API_KEY    the provider key
    LLM_MODEL      optional model id (each provider has a default)
    LLM_BASE_URL   optional, OpenAI-compatible providers only
    LLM_MAX_CALLS_PER_RUN   default 40

Tests use FakeProvider, which replays scripted replies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The call failed (network, provider error, invalid reply after the retry)."""


class LLMNotConfigured(LLMError):
    """No provider is configured (missing LLM_PROVIDER or LLM_API_KEY)."""


class CallCapExceeded(LLMError):
    """The run has used up its LLM call budget."""


class LLMRefused(LLMError):
    """The provider's own safety system declined the request."""


@dataclass
class LLMCall:
    """One logged provider call (one per attempt, including the schema retry)."""

    run_id: Optional[str]
    purpose: str
    provider: str
    model: str
    ok: bool
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Completion:
    text: str  # the raw JSON text returned by the model
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class Provider(Protocol):
    name: str
    model: str

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        model_cls: Optional[type[BaseModel]] = None,
    ) -> Completion:
        """Ask for one JSON object matching `schema` (the JSON Schema of `model_cls`).

        Return the raw JSON text. Raise LLMRefused when the provider's safety
        system declines, LLMError for any other failure. Transient errors
        (429, 5xx, timeouts) should already be retried inside the provider.
        """
        ...


def _strip_fences(text: str) -> str:
    """Remove ```json fences some models put around JSON."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class LLMClient:
    def __init__(self, provider: Provider, max_calls_per_run: int = 40) -> None:
        self.provider = provider
        self.max_calls_per_run = max_calls_per_run
        self.calls: list[LLMCall] = []
        self._lock = threading.Lock()

    def calls_for(self, run_id: Optional[str]) -> int:
        with self._lock:
            return sum(1 for c in self.calls if c.run_id == run_id)

    def _reserve(self, run_id: Optional[str]) -> None:
        if run_id is not None and self.calls_for(run_id) >= self.max_calls_per_run:
            raise CallCapExceeded(f"run {run_id} reached its cap of {self.max_calls_per_run} LLM calls")

    def _log(self, record: LLMCall) -> None:
        with self._lock:
            self.calls.append(record)
        log.info(
            "llm call run=%s purpose=%s ok=%s %sms in=%s out=%s err=%s",
            record.run_id, record.purpose, record.ok, record.latency_ms,
            record.input_tokens, record.output_tokens, record.error,
        )

    def structured(
        self,
        model_cls: type[T],
        *,
        system: str,
        user: str,
        purpose: str,
        run_id: Optional[str] = None,
    ) -> T:
        """Call the model and return a validated `model_cls`. One retry on a schema mismatch."""
        schema = model_cls.model_json_schema()
        prompt = user
        last_error = ""
        for attempt in range(2):
            self._reserve(run_id)
            started = time.monotonic()
            try:
                completion = self.provider.complete_json(
                    system=system,
                    user=prompt,
                    schema=schema,
                    schema_name=model_cls.__name__,
                    model_cls=model_cls,
                )
            except LLMError as exc:
                self._log(LLMCall(run_id, purpose, self.provider.name, self.provider.model, False,
                                  int((time.monotonic() - started) * 1000), error=str(exc)[:500]))
                raise
            except Exception as exc:  # provider bug or SDK error not mapped by the adapter
                self._log(LLMCall(run_id, purpose, self.provider.name, self.provider.model, False,
                                  int((time.monotonic() - started) * 1000), error=repr(exc)[:500]))
                raise LLMError(f"{purpose}: provider error: {exc}") from exc

            latency = int((time.monotonic() - started) * 1000)
            try:
                result = model_cls.model_validate(json.loads(_strip_fences(completion.text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:1500]
                self._log(LLMCall(run_id, purpose, self.provider.name, self.provider.model, False,
                                  latency, completion.input_tokens, completion.output_tokens,
                                  error=f"invalid reply: {last_error[:300]}"))
                prompt = (
                    f"{user}\n\nYour previous reply did not match the required JSON schema. "
                    f"Error:\n{last_error}\nReply again with one JSON object that matches the schema."
                )
                continue
            self._log(LLMCall(run_id, purpose, self.provider.name, self.provider.model, True,
                              latency, completion.input_tokens, completion.output_tokens))
            return result
        raise LLMError(f"{purpose}: reply did not match the schema after one retry: {last_error[:300]}")


# --------------------------------------------------------------------------- test double


Reply = Any  # str (raw JSON), dict (serialised for you), BaseModel, Exception, or callable


class FakeProvider:
    """Replays scripted replies, keyed by schema name (the Pydantic class name).

    `replies[name]` is a list consumed in order (the last entry repeats) or a
    callable `(system, user, schema) -> Reply`. Exceptions are raised.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, replies: dict[str, list[Reply] | Callable[..., Reply]]) -> None:
        self.replies = {k: (list(v) if isinstance(v, list) else v) for k, v in replies.items()}
        self.requests: list[dict[str, Any]] = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        model_cls: Optional[type[BaseModel]] = None,
    ) -> Completion:
        self.requests.append({"system": system, "user": user, "schema_name": schema_name})
        if schema_name not in self.replies:
            raise LLMError(f"FakeProvider has no reply for {schema_name}")
        source = self.replies[schema_name]
        if callable(source):
            reply = source(system, user, schema)
        else:
            reply = source.pop(0) if len(source) > 1 else source[0]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, BaseModel):
            reply = reply.model_dump(mode="json")
        if not isinstance(reply, str):
            reply = json.dumps(reply)
        return Completion(text=reply, input_tokens=len(user) // 4, output_tokens=len(reply) // 4)


# --------------------------------------------------------------------------- configuration

_client: Optional[LLMClient] = None
_client_lock = threading.Lock()


def provider_from_env() -> Optional[Provider]:
    """Build the configured provider, or None when no key is set."""
    name = os.getenv("LLM_PROVIDER", "").strip().lower()
    key = os.getenv("LLM_API_KEY", "").strip()
    if not name or not key:
        return None
    from . import llm_providers  # imported lazily so tests never need the SDKs

    return llm_providers.build_provider(
        name,
        api_key=key,
        model=os.getenv("LLM_MODEL", "").strip() or None,
        base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
    )


def get_client() -> LLMClient:
    """The process-wide client. Raises LLMNotConfigured when no provider is set."""
    global _client
    with _client_lock:
        if _client is None:
            provider = provider_from_env()
            if provider is None:
                raise LLMNotConfigured("set LLM_PROVIDER and LLM_API_KEY to enable the LLM steps")
            _client = LLMClient(provider, int(os.getenv("LLM_MAX_CALLS_PER_RUN", "40")))
        return _client


def set_client(client: Optional[LLMClient]) -> None:
    """Install a client explicitly (tests, evals)."""
    global _client
    with _client_lock:
        _client = client


def is_configured() -> bool:
    if _client is not None:
        return True
    return bool(os.getenv("LLM_PROVIDER", "").strip() and os.getenv("LLM_API_KEY", "").strip())
