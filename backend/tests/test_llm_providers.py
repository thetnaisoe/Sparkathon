"""Provider adapters, tested against fake SDK clients (no network, no keys)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Literal, Optional

import anthropic
import httpx2
import openai
import pytest
from pydantic import BaseModel

from app.llm import LLMClient, LLMError, LLMNotConfigured, LLMRefused
from app.llm_providers import (
    ANTHROPIC_DEFAULT_MODEL,
    ANTHROPIC_FALLBACK_BETA,
    AnthropicProvider,
    OpenAICompatibleProvider,
    _timeout,
    build_provider,
)

KEY = "sk-test-0123456789abcdef"


class Reply(BaseModel):
    verdict: Literal["yes", "no"]
    note: Optional[str] = None


SCHEMA = Reply.model_json_schema()
GOOD = json.dumps({"verdict": "yes", "note": "fine"})
SAMPLING = {"temperature", "top_p", "top_k"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Provider env switches must not leak in from the developer's shell."""
    for var in ("LLM_EFFORT", "LLM_ANTHROPIC_FALLBACKS", "LLM_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)


def _status_error(cls: type, status: int, message: str, **extra: Any) -> Exception:
    request = httpx2.Request("POST", "https://example.invalid/v1")
    body = {"error": {"message": message, **extra}}
    response = httpx2.Response(status, request=request, json=body)
    # The openai SDK hands the exception the inner error object (`body.get("error", body)`).
    return cls(message, response=response, body=body["error"] if issubclass(cls, openai.APIError) else body)


class _Recorder:
    """Callable that records kwargs and plays scripted results (exceptions are raised)."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result


# --------------------------------------------------------------------------- Anthropic


def _claude(text: str = GOOD, stop: str = "end_turn", category: Optional[str] = None) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop,
        stop_details=SimpleNamespace(category=category) if stop == "refusal" else None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, iterations=None),
        model=ANTHROPIC_DEFAULT_MODEL,
    )


def _anthropic(results: list[Any], *, fallbacks: bool = True, effort: Optional[str] = None):
    plain, beta = _Recorder(results), _Recorder(results)
    client = SimpleNamespace(messages=SimpleNamespace(create=plain),
                             beta=SimpleNamespace(messages=SimpleNamespace(create=beta)))
    provider = AnthropicProvider(KEY, client=client, fallbacks=fallbacks, effort=effort)
    return provider, plain, beta


def _call(provider: Any) -> Any:
    return provider.complete_json(system="SYS", user="USER", schema=SCHEMA, schema_name="Reply", model_cls=Reply)


def test_anthropic_request_shape_with_fallbacks():
    provider, plain, beta = _anthropic([_claude()])
    completion = _call(provider)
    assert completion.text == GOOD
    assert (completion.input_tokens, completion.output_tokens) == (11, 7)
    assert not plain.calls and len(beta.calls) == 1
    req = beta.calls[0]
    assert req["model"] == ANTHROPIC_DEFAULT_MODEL == "claude-opus-5-5"
    assert req["system"] == "SYS"
    assert req["messages"] == [{"role": "user", "content": "USER"}]
    assert req["max_tokens"] == 16000
    assert req["betas"] == [ANTHROPIC_FALLBACK_BETA] and req["fallbacks"] == "default"
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["properties"]) == {"verdict", "note"}
    assert "effort" not in req["output_config"]
    assert not SAMPLING & set(req) and "thinking" not in req


def test_anthropic_without_fallbacks_uses_plain_endpoint_and_effort():
    provider, plain, beta = _anthropic([_claude()], fallbacks=False, effort="low")
    _call(provider)
    assert not beta.calls and len(plain.calls) == 1
    req = plain.calls[0]
    assert "betas" not in req and "fallbacks" not in req
    assert req["output_config"]["effort"] == "low"


def test_anthropic_env_switches(monkeypatch):
    monkeypatch.setenv("LLM_ANTHROPIC_FALLBACKS", "0")
    monkeypatch.setenv("LLM_EFFORT", "bogus")
    provider = AnthropicProvider(KEY, client=object())
    assert provider.fallbacks is False and provider.effort is None
    monkeypatch.setenv("LLM_EFFORT", "HIGH")
    assert AnthropicProvider(KEY, client=object()).effort == "high"


def test_anthropic_refusal_raises_refused():
    provider, _, _ = _anthropic([_claude(text="", stop="refusal", category="bio")])
    with pytest.raises(LLMRefused, match="bio"):
        _call(provider)


def test_anthropic_max_tokens_raises_error():
    provider, _, _ = _anthropic([_claude(text='{"verdict": "y', stop="max_tokens")])
    with pytest.raises(LLMError, match="truncated") as info:
        _call(provider)
    assert not isinstance(info.value, LLMRefused)


def test_anthropic_rejected_fallback_beta_is_switched_off():
    err = _status_error(anthropic.BadRequestError, 400, "Unexpected value(s) for the `anthropic-beta` header")
    provider, plain, beta = _anthropic([err, _claude()])
    plain.results = [_claude()]
    assert _call(provider).text == GOOD
    assert provider.fallbacks is False and len(beta.calls) == 1 and len(plain.calls) == 1
    assert "output_config" in plain.calls[0]


def test_anthropic_rejected_schema_moves_to_prompt_and_is_remembered():
    err = _status_error(anthropic.BadRequestError, 400, "output_config.format: schema is too complex")
    provider, plain, _ = _anthropic([err, _claude(), _claude()], fallbacks=False)
    _call(provider)
    second = plain.calls[1]
    assert "output_config" not in second
    assert "JSON Schema" in second["system"] and '"verdict"' in second["system"]
    _call(provider)
    assert "output_config" not in plain.calls[2]


@pytest.mark.parametrize("message", ["prompt is too long: 260000 tokens > 200000 maximum",
                                     "max_tokens: 999999 > 128000, which is the maximum allowed"])
def test_anthropic_unrelated_400_raises_at_once_without_state_change(message):
    provider, plain, beta = _anthropic([_status_error(anthropic.BadRequestError, 400, message)])
    with pytest.raises(LLMError, match="400"):
        _call(provider)
    assert len(beta.calls) == 1 and not plain.calls
    assert provider.fallbacks is True and provider._prompt_schema_only == set()
    provider, plain, _ = _anthropic([_status_error(anthropic.BadRequestError, 400, message)], fallbacks=False)
    with pytest.raises(LLMError):
        _call(provider)
    assert len(plain.calls) == 1 and provider._prompt_schema_only == set()


def test_anthropic_errors_map_to_llm_error_without_key():
    err = _status_error(anthropic.AuthenticationError, 401, f"invalid x-api-key {KEY}")
    provider, _, _ = _anthropic([err])
    with pytest.raises(LLMError) as info:
        _call(provider)
    assert KEY not in str(info.value)
    request = httpx2.Request("POST", "https://example.invalid/v1")
    provider, _, _ = _anthropic([anthropic.APIConnectionError(request=request)])
    with pytest.raises(LLMError, match="connection"):
        _call(provider)
    provider, _, _ = _anthropic([_status_error(anthropic.InternalServerError, 529, "overloaded")])
    with pytest.raises(LLMError, match="529"):
        _call(provider)


def test_anthropic_through_llm_client():
    provider, _, _ = _anthropic([_claude()])
    result = LLMClient(provider).structured(Reply, system="S", user="U", purpose="t", run_id="r")
    assert result == Reply(verdict="yes", note="fine")


# --------------------------------------------------------------------------- OpenAI-compatible


def _chat(text: Optional[str] = GOOD, finish: str = "stop", refusal: Optional[str] = None) -> Any:
    message = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3))


def _openai(results: list[Any]):
    create = _Recorder(results)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAICompatibleProvider(KEY, "some-model", client=client, name="mistral"), create


def test_openai_json_schema_request_shape():
    provider, create = _openai([_chat()])
    completion = _call(provider)
    assert completion.text == GOOD and (completion.input_tokens, completion.output_tokens) == (5, 3)
    req = create.calls[0]
    assert req["model"] == "some-model"
    assert req["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    rf = req["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"] == {"name": "Reply", "schema": SCHEMA, "strict": False}
    assert not SAMPLING & set(req) and "max_tokens" not in req


def test_openai_fallback_chain_and_memory():
    bad = _status_error(openai.BadRequestError, 400, "response_format json_schema is not supported")
    bad2 = _status_error(openai.UnprocessableEntityError, 422, "json_object unsupported")
    provider, create = _openai([bad, bad2, _chat(), _chat()])
    assert _call(provider).text == GOOD
    modes = [c.get("response_format", {}).get("type") for c in create.calls]
    assert modes == ["json_schema", "json_object", None]
    plain_system = create.calls[2]["messages"][0]["content"]
    assert plain_system.startswith("SYS") and "one JSON object only" in plain_system and '"verdict"' in plain_system
    assert "JSON" in create.calls[1]["messages"][0]["content"]  # json_object mode needs the word JSON
    assert provider.mode == "plain"
    _call(provider)
    assert len(create.calls) == 4 and "response_format" not in create.calls[3]


def test_openai_json_object_remembered():
    bad = _status_error(openai.BadRequestError, 400, "unsupported response_format")
    provider, create = _openai([bad, _chat(), _chat()])
    _call(provider)
    _call(provider)
    assert [c["response_format"]["type"] for c in create.calls] == ["json_schema", "json_object", "json_object"]


def test_openai_all_modes_rejected():
    provider, _ = _openai([_status_error(openai.BadRequestError, 400, "invalid schema", param="response_format")])
    with pytest.raises(LLMError, match="every JSON mode"):
        _call(provider)


def test_openai_unrelated_400_raises_at_once():
    err = _status_error(openai.BadRequestError, 400, "This model's maximum context length is 128000 tokens",
                        code="context_length_exceeded", param="messages")
    provider, create = _openai([err])
    with pytest.raises(LLMError, match="bad request") as info:
        _call(provider)
    assert not isinstance(info.value, LLMRefused)
    assert len(create.calls) == 1 and provider.mode == "json_schema"


def test_openai_azure_content_filter_400_is_refusal():
    err = _status_error(openai.BadRequestError, 400,
                        "The response was filtered due to the prompt triggering Azure OpenAI's content management "
                        "policy.", code="content_filter", param="prompt")
    assert err.code == "content_filter"
    provider, create = _openai([err])
    with pytest.raises(LLMRefused):
        _call(provider)
    assert len(create.calls) == 1 and provider.mode == "json_schema"


def test_timeout_default_and_override(monkeypatch):
    assert _timeout() == 60.0
    assert _timeout(20) == 20.0
    monkeypatch.setenv("LLM_TIMEOUT_S", "30")
    assert _timeout() == 30.0
    p = build_provider("openai", api_key=KEY, model="gpt-x", base_url="https://example.invalid/v1")
    assert p.client.timeout == 30.0
    assert OpenAICompatibleProvider(KEY, "m", timeout=10).client.timeout == 10.0
    assert AnthropicProvider(KEY, timeout=12).client.timeout == 12.0


@pytest.mark.parametrize("reply", [_chat(text=None, refusal="I can't help with that"),
                                   _chat(text="", finish="content_filter")])
def test_openai_refusals(reply):
    provider, _ = _openai([reply])
    with pytest.raises(LLMRefused):
        _call(provider)


def test_openai_truncation_and_errors():
    provider, _ = _openai([_chat(text='{"verdict"', finish="length")])
    with pytest.raises(LLMError, match="truncated"):
        _call(provider)
    provider, _ = _openai([_status_error(openai.RateLimitError, 429, "slow down")])
    with pytest.raises(LLMError, match="429"):
        _call(provider)


def test_openai_requires_model():
    with pytest.raises(LLMNotConfigured):
        OpenAICompatibleProvider(KEY, None, client=object())
    with pytest.raises(LLMNotConfigured):
        build_provider("openai", api_key=KEY, model=None, base_url=None)


# --------------------------------------------------------------------------- factory


def test_build_provider():
    p = build_provider("Anthropic", api_key=KEY, model=None, base_url=None)
    assert isinstance(p, AnthropicProvider) and p.model == "claude-opus-5-5" and p.name == "anthropic"
    p = build_provider("anthropic", api_key=KEY, model="claude-sonnet-5-5", base_url=None)
    assert p.model == "claude-sonnet-5-5"
    p = build_provider("openai", api_key=KEY, model="gpt-x", base_url="https://example.invalid/v1")
    assert isinstance(p, OpenAICompatibleProvider) and p.model == "gpt-x"
    assert str(p.client.base_url).startswith("https://example.invalid/v1")
    p = build_provider("mistral", api_key=KEY, model="mistral-large-latest", base_url=None)
    assert p.name == "mistral" and "api.mistral.ai" in str(p.client.base_url)


def test_build_provider_errors():
    with pytest.raises(LLMNotConfigured, match="unknown"):
        build_provider("cohere", api_key=KEY, model="m", base_url=None)
    with pytest.raises(LLMNotConfigured):
        build_provider("anthropic", api_key="", model=None, base_url=None)
    with pytest.raises(LLMNotConfigured, match="LLM_BASE_URL"):
        build_provider("azure", api_key=KEY, model="gpt-x", base_url=None)
