"""LLM provider adapters: Anthropic and OpenAI-compatible APIs.

Each provider implements the `Provider` protocol from app.llm: one call that
asks for a single JSON object and returns the raw JSON text plus token usage.
Validation against the Pydantic model and the one schema retry happen in
`LLMClient.structured`, not here.

Configuration (all optional except what `build_provider` receives):

    LLM_TIMEOUT_S               per-attempt HTTP timeout in seconds (default 60)
    LLM_EFFORT                  Anthropic only: low | medium | high | xhigh | max;
                                sent only when set (Claude Opus 5.5 defaults to medium)
    LLM_ANTHROPIC_FALLBACKS     Anthropic only: "0" turns off server-side refusal
                                fallbacks (default on, `fallbacks: "default"`)

Transient failures (429, 5xx, 529, connection errors, timeouts) are retried by
the SDKs themselves (`max_retries=2`). Whatever is still failing afterwards is
raised as `LLMError`; a safety decline is raised as `LLMRefused`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from pydantic import BaseModel

from .llm import Completion, LLMError, LLMNotConfigured, LLMRefused

log = logging.getLogger("llm.providers")

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_MAX_TOKENS = 16000
ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_RETRIES = 2
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

# OpenAI-compatible aliases: name -> default base_url (None = the SDK default, api.openai.com).
OPENAI_COMPATIBLE: dict[str, Optional[str]] = {
    "openai": None,
    "azure": None,  # Azure OpenAI v1 needs LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1/
    "mistral": "https://api.mistral.ai/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
}

_KEYLIKE = re.compile(r"\b(?:sk|key|gsk|AIza)[-_A-Za-z0-9*]{8,}")


DEFAULT_TIMEOUT_S = 60.0

# HTTP 400 classification. Only these messages change provider state; any other
# 400 (prompt too long, bad max_tokens, ...) is raised at once.
_ANTHROPIC_BETA_HINTS = ("fallback", "anthropic-beta", ANTHROPIC_FALLBACK_BETA)
_SCHEMA_HINTS = ("output_config.format", "output_format", "json_schema", "schema")
_OPENAI_FORMAT_HINTS = ("response_format", "json_schema", "json_object", "schema")
_CONTENT_FILTER_HINTS = ("content_filter", "responsibleaipolicyviolation", "content management policy")


def _timeout(override: Optional[float] = None) -> float:
    """Per-attempt timeout: explicit argument, else LLM_TIMEOUT_S, else 60 s (minimum 5 s)."""
    if override is not None:
        return max(5.0, float(override))
    try:
        return max(5.0, float(os.getenv("LLM_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _error_text(exc: Exception) -> str:
    """Lower-cased error text: str(exc) plus the body's error message/code/param when present."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else None
    if isinstance(err, dict):
        parts += [str(err.get(k) or "") for k in ("message", "code", "param", "type")]
    parts += [str(getattr(exc, k, None) or "") for k in ("code", "param")]
    return " ".join(parts).lower()


def _mentions(text: str, hints: tuple[str, ...]) -> bool:
    return any(h.lower() in text for h in hints)


def _redact(message: str, api_key: Optional[str]) -> str:
    """Strip the key (and anything key-shaped) from an error message before it is logged or raised."""
    if api_key:
        message = message.replace(api_key, "[redacted]")
    return _KEYLIKE.sub("[redacted]", message)[:400]


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Reply with one JSON object only: no prose, no markdown, no code fences. "
        "It must match this JSON Schema:\n" + json.dumps(schema, separators=(",", ":"))
    )


# --------------------------------------------------------------------------- Anthropic


class AnthropicProvider:
    """Claude via the Messages API with structured outputs (`output_config.format`).

    Thinking is left at the model default (adaptive, always on for Claude Opus 5.5);
    no sampling parameters are sent. Self-healing on HTTP 400, decided by the
    error text: if the fallback beta is rejected it is switched off for the
    process; if a schema is rejected by structured outputs, that schema is sent
    in the system prompt instead. Any other 400 is raised at once, state unchanged.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        client: Any = None,
        fallbacks: Optional[bool] = None,
        effort: Optional[str] = None,
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
        timeout: Optional[float] = None,
    ) -> None:
        import anthropic

        self._sdk = anthropic
        self._api_key = api_key
        self.model = model or ANTHROPIC_DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic(
            api_key=api_key, base_url=base_url, max_retries=MAX_RETRIES, timeout=_timeout(timeout)
        )
        if fallbacks is None:
            fallbacks = os.getenv("LLM_ANTHROPIC_FALLBACKS", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.fallbacks = fallbacks
        effort = (effort if effort is not None else os.getenv("LLM_EFFORT", "")).strip().lower() or None
        if effort is not None and effort not in EFFORT_LEVELS:
            log.warning("ignoring LLM_EFFORT=%r (expected one of %s)", effort, ", ".join(EFFORT_LEVELS))
            effort = None
        self.effort = effort
        self._prompt_schema_only: set[str] = set()  # schema names the API rejected as a format

    def _format(self, schema: dict[str, Any], schema_name: str) -> Optional[dict[str, Any]]:
        if schema_name in self._prompt_schema_only:
            return None
        try:
            return {"type": "json_schema", "schema": self._sdk.transform_schema(schema)}
        except Exception as exc:  # schema shape the SDK cannot make strict
            log.warning("anthropic: schema %s not usable as output format (%s); using prompt", schema_name, exc)
            self._prompt_schema_only.add(schema_name)
            return None

    def _request(self, system: str, user: str, schema: dict[str, Any], fmt: Optional[dict[str, Any]]) -> dict[str, Any]:
        output_config: dict[str, Any] = {}
        if fmt is not None:
            output_config["format"] = fmt
        else:
            system = f"{system}\n\n{_schema_instruction(schema)}"
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    def _send(self, kwargs: dict[str, Any]) -> Any:
        if self.fallbacks:
            return self.client.beta.messages.create(
                **kwargs, betas=[ANTHROPIC_FALLBACK_BETA], fallbacks="default"
            )
        return self.client.messages.create(**kwargs)

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        model_cls: Optional[type[BaseModel]] = None,
    ) -> Completion:
        sdk = self._sdk
        fmt = self._format(schema, schema_name)
        response = None
        for _ in range(3):  # at most: drop fallback beta, drop format, then succeed or raise
            try:
                response = self._send(self._request(system, user, schema, fmt))
                break
            except sdk.BadRequestError as exc:
                text = _error_text(exc)
                if self.fallbacks and _mentions(text, _ANTHROPIC_BETA_HINTS):
                    log.warning("anthropic: fallback beta rejected; disabling fallbacks: %s", _redact(str(exc), self._api_key))
                    self.fallbacks = False
                elif fmt is not None and _mentions(text, _SCHEMA_HINTS):
                    log.warning("anthropic: structured output rejected for %s; sending schema in prompt: %s",
                                schema_name, _redact(str(exc), self._api_key))
                    self._prompt_schema_only.add(schema_name)
                    fmt = None
                else:
                    raise LLMError(f"anthropic: bad request (400): {_redact(str(exc), self._api_key)}") from exc
            except sdk.AuthenticationError as exc:
                raise LLMError("anthropic: authentication failed (401); check LLM_API_KEY") from exc
            except sdk.PermissionDeniedError as exc:
                raise LLMError(f"anthropic: permission denied (403): {_redact(str(exc), self._api_key)}") from exc
            except sdk.NotFoundError as exc:
                raise LLMError(f"anthropic: model {self.model!r} not found (404)") from exc
            except sdk.RateLimitError as exc:
                raise LLMError("anthropic: rate limited (429) after retries") from exc
            except sdk.APIStatusError as exc:
                raise LLMError(f"anthropic: API error {exc.status_code} after retries: "
                               f"{_redact(str(exc), self._api_key)}") from exc
            except sdk.APITimeoutError as exc:
                raise LLMError("anthropic: request timed out after retries") from exc
            except sdk.APIConnectionError as exc:
                raise LLMError("anthropic: connection error after retries") from exc
            except sdk.AnthropicError as exc:
                raise LLMError(f"anthropic: {_redact(str(exc), self._api_key)}") from exc
        if response is None:
            raise LLMError("anthropic: request rejected (400) in every mode")
        return self._read(response)

    def _read(self, response: Any) -> Completion:
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LLMRefused(f"anthropic: the model declined (category={category or 'unspecified'})")
        if stop in ("max_tokens", "model_context_window_exceeded"):
            raise LLMError(f"anthropic: reply truncated (stop_reason={stop})")
        text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text")
        if not text.strip():
            raise LLMError(f"anthropic: empty reply (stop_reason={stop})")
        usage = getattr(response, "usage", None)
        iterations = getattr(usage, "iterations", None) or []
        if any(getattr(it, "type", None) == "fallback_message" for it in iterations):
            log.info("anthropic: request served by fallback model %s", getattr(response, "model", "?"))
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )


# --------------------------------------------------------------------------- OpenAI-compatible


class OpenAICompatibleProvider:
    """Any Chat Completions API (OpenAI, Azure v1, Mistral, Gemini, Groq, OpenRouter).

    JSON modes, tried in order until the provider accepts one; the mode that
    worked is remembered for later calls. Only a 400/422 about the response
    format moves to the next mode; a content-filter 400 (Azure) is a refusal and
    any other 400/422 is raised at once.
      1. `response_format: json_schema` (non-strict, the schema as given)
      2. `response_format: json_object` with the schema in the system prompt
      3. plain call with the schema and "one JSON object only" in the system prompt
    No sampling parameters or token limits are sent (reasoning models reject them).
    """

    MODES = ("json_schema", "json_object", "plain")

    def __init__(
        self,
        api_key: str,
        model: Optional[str],
        base_url: Optional[str] = None,
        *,
        name: str = "openai",
        client: Any = None,
        timeout: Optional[float] = None,
    ) -> None:
        if not model:
            raise LLMNotConfigured(f"{name}: set LLM_MODEL (OpenAI-compatible providers have no default model)")
        import openai

        self._sdk = openai
        self._api_key = api_key
        self.name = name
        self.model = model
        self.client = client or openai.OpenAI(
            api_key=api_key, base_url=base_url, max_retries=MAX_RETRIES, timeout=_timeout(timeout)
        )
        self.mode = self.MODES[0]

    @staticmethod
    def _schema_name(name: str) -> str:
        return (re.sub(r"[^A-Za-z0-9_-]", "_", name) or "reply")[:64]

    def _request(self, mode: str, system: str, user: str, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self._schema_name(schema_name), "schema": schema, "strict": False},
            }
        else:
            system = f"{system}\n\n{_schema_instruction(schema)}"
            if mode == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
        kwargs["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return kwargs

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str,
        model_cls: Optional[type[BaseModel]] = None,
    ) -> Completion:
        sdk = self._sdk
        modes = self.MODES[self.MODES.index(self.mode):]
        last = ""
        for mode in modes:
            try:
                response = self.client.chat.completions.create(**self._request(mode, system, user, schema, schema_name))
            except (sdk.BadRequestError, sdk.UnprocessableEntityError) as exc:
                text = _error_text(exc)
                last = _redact(str(exc), self._api_key)
                if _mentions(text, _CONTENT_FILTER_HINTS):
                    raise LLMRefused(f"{self.name}: request blocked by the content filter: {last}"[:300]) from exc
                if not _mentions(text, _OPENAI_FORMAT_HINTS):
                    raise LLMError(f"{self.name}: bad request ({exc.status_code}): {last}") from exc
                log.warning("%s: %s mode rejected (%s); trying the next mode", self.name, mode, last)
                continue
            except sdk.AuthenticationError as exc:
                raise LLMError(f"{self.name}: authentication failed (401); check LLM_API_KEY") from exc
            except sdk.PermissionDeniedError as exc:
                raise LLMError(f"{self.name}: permission denied (403): {_redact(str(exc), self._api_key)}") from exc
            except sdk.NotFoundError as exc:
                raise LLMError(f"{self.name}: model {self.model!r} or endpoint not found (404)") from exc
            except sdk.RateLimitError as exc:
                raise LLMError(f"{self.name}: rate limited (429) after retries") from exc
            except sdk.APIStatusError as exc:
                raise LLMError(f"{self.name}: API error {exc.status_code} after retries: "
                               f"{_redact(str(exc), self._api_key)}") from exc
            except sdk.APITimeoutError as exc:
                raise LLMError(f"{self.name}: request timed out after retries") from exc
            except sdk.APIConnectionError as exc:
                raise LLMError(f"{self.name}: connection error after retries") from exc
            except sdk.OpenAIError as exc:
                raise LLMError(f"{self.name}: {_redact(str(exc), self._api_key)}") from exc
            if mode != self.mode:
                log.info("%s: using JSON mode %s from now on", self.name, mode)
                self.mode = mode
            return self._read(response)
        raise LLMError(f"{self.name}: request rejected in every JSON mode: {last}")

    def _read(self, response: Any) -> Completion:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(f"{self.name}: reply had no choices")
        choice = choices[0]
        message = choice.message
        if getattr(message, "refusal", None) or choice.finish_reason == "content_filter":
            raise LLMRefused(f"{self.name}: the model declined ({getattr(message, 'refusal', None) or 'content_filter'})"[:300])
        if choice.finish_reason == "length":
            raise LLMError(f"{self.name}: reply truncated (finish_reason=length)")
        text = message.content or ""
        if not text.strip():
            raise LLMError(f"{self.name}: empty reply (finish_reason={choice.finish_reason})")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )


# --------------------------------------------------------------------------- factory


def build_provider(
    name: str,
    *,
    api_key: Optional[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> AnthropicProvider | OpenAICompatibleProvider:
    """Build the provider for `name` ("anthropic", "openai" or an OpenAI-compatible alias).

    Aliases (azure, mistral, groq, gemini, openrouter) use the OpenAI-compatible
    adapter with a default base_url; an explicit `base_url` always wins.
    """
    key = (name or "").strip().lower()
    if not api_key:
        raise LLMNotConfigured(f"{key or 'LLM_PROVIDER'}: LLM_API_KEY is not set")
    if key == "anthropic":
        return AnthropicProvider(api_key, model, base_url)
    if key == "azure" and not base_url:
        raise LLMNotConfigured("azure: set LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1/")
    if key in OPENAI_COMPATIBLE:
        return OpenAICompatibleProvider(api_key, model, base_url or OPENAI_COMPATIBLE[key], name=key)
    raise LLMNotConfigured(
        f"unknown LLM_PROVIDER {name!r}; use one of: anthropic, " + ", ".join(OPENAI_COMPATIBLE)
    )
