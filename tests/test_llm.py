"""LLM layer: schema validation, budget, caching, and the exact wire format of both provider backends."""

import json
from types import SimpleNamespace

import pytest

from ospm.llm import BudgetExceeded, LLMError, UsageMeter, make_llm, price_for, validate
from ospm.llm.anthropic_llm import FALLBACK_BETA, AnthropicLLM
from ospm.llm.fake import FakeLLM
from ospm.llm.openai_llm import OpenAILLM

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["bug", "feature"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "n": {"type": "integer"},
    },
    "required": ["label", "tags", "n"],
    "additionalProperties": False,
}
GOOD = {"label": "bug", "tags": ["a"], "n": 1}


def test_validate_subset():
    assert validate(GOOD, SCHEMA) == []
    assert validate({"label": "nope", "tags": [], "n": 1}, SCHEMA)
    assert validate({"label": "bug", "tags": [1], "n": 1}, SCHEMA)
    assert validate({"label": "bug", "tags": [], "n": True}, SCHEMA)  # bool is not an integer
    assert validate({"label": "bug", "tags": []}, SCHEMA)  # missing
    assert validate({**GOOD, "extra": 1}, SCHEMA)  # additionalProperties


def test_meter_enforces_budget_and_prices():
    m = UsageMeter(budget_usd=0.001)
    m.check()
    m.record("gpt-6-luna", 1_000_000, 0)  # $0.10
    with pytest.raises(BudgetExceeded):
        m.check()
    assert price_for("gpt-6-luna") == (0.10, 0.50)
    assert price_for("unknown-model") == (10.0, 50.0)  # conservative default


def test_fake_llm_cache_and_validation(memory):
    calls = []
    llm = FakeLLM(lambda s, u, sch: calls.append(1) or GOOD, cache=memory)
    assert llm.complete_json("sys", "user", SCHEMA) == GOOD
    assert llm.complete_json("sys", "user", SCHEMA) == GOOD
    assert len(calls) == 1 and llm.meter.cache_hits == 1
    bad = FakeLLM(lambda s, u, sch: {"label": "nope", "tags": [], "n": 0})
    with pytest.raises(LLMError, match="schema"):
        bad.complete_json("s", "u", SCHEMA)
    assert bad.meter.errors == 1


def test_budget_blocks_calls_before_they_happen():
    llm = FakeLLM(lambda s, u, sch: GOOD, meter=UsageMeter(budget_usd=0.0))
    with pytest.raises(BudgetExceeded):
        llm.complete_json("s", "u", SCHEMA)
    assert llm.requests == []


def test_make_llm_none_and_auto(monkeypatch):
    monkeypatch.delenv("OSPM_LLM", raising=False)
    assert make_llm() is None
    assert make_llm("none") is None
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert make_llm("auto") is None
    with pytest.raises(ValueError):
        make_llm("bogus")


# ---------------------------------------------------------------------------------------------- fake clients
class _OpenAIClient:
    def __init__(self, resp=None, exc=None):
        self.kwargs = None
        self.resp, self.exc = resp, exc
        self.responses = self

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp


def test_openai_request_shape_and_usage():
    resp = SimpleNamespace(status="completed", output_text=json.dumps(GOOD),
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))
    client = _OpenAIClient(resp)
    llm = OpenAILLM("gpt-6-luna", client=client)
    assert llm.complete_json("SYS", "USER", SCHEMA, name="triage", effort="low") == GOOD
    kw = client.kwargs
    assert kw["model"] == "gpt-6-luna" and kw["instructions"] == "SYS" and kw["input"] == "USER"
    assert kw["text"]["format"] == {"type": "json_schema", "name": "triage", "schema": SCHEMA, "strict": True}
    assert kw["reasoning"] == {"effort": "low"}
    assert llm.meter.input_tokens == 100 and llm.meter.output_tokens == 20
    assert llm.meter.cost_usd == pytest.approx(100 * 0.10 / 1e6 + 20 * 0.50 / 1e6)


def test_openai_errors_become_llm_errors():
    with pytest.raises(LLMError, match="request failed"):
        OpenAILLM(client=_OpenAIClient(exc=RuntimeError("boom"))).complete_json("s", "u", SCHEMA)
    incomplete = SimpleNamespace(status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                                 output_text="", usage=None)
    with pytest.raises(LLMError, match="incomplete"):
        OpenAILLM(client=_OpenAIClient(incomplete)).complete_json("s", "u", SCHEMA)


class _AnthropicClient:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []
        self.messages = SimpleNamespace(create=self._create("messages"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create("beta")))

    def _create(self, kind):
        def create(**kw):
            self.calls.append((kind, kw))
            return self.resp
        return create


def _anthropic_resp(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop, stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, cache_creation_input_tokens=5, cache_read_input_tokens=7,
                              output_tokens=3),
    )


def test_anthropic_request_shape_with_fallbacks():
    client = _AnthropicClient(_anthropic_resp(json.dumps(GOOD)))
    llm = AnthropicLLM("claude-opus-5-5", client=client)
    assert llm.complete_json("SYS", "USER", SCHEMA, effort="high") == GOOD
    kind, kw = client.calls[0]
    assert kind == "beta" and kw["betas"] == [FALLBACK_BETA]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert kw["messages"] == [{"role": "user", "content": "USER"}]
    assert kw["extra_body"]["fallbacks"] == "default"
    assert kw["extra_body"]["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "high"}
    assert "thinking" not in kw  # Opus 5.5 thinking is always on; effort is the control
    assert llm.meter.input_tokens == 22 and llm.meter.output_tokens == 3


def test_anthropic_without_fallbacks_and_refusal():
    client = _AnthropicClient(_anthropic_resp(json.dumps(GOOD)))
    AnthropicLLM(client=client, fallbacks=False).complete_json("s", "u", SCHEMA)
    kind, kw = client.calls[0]
    assert kind == "messages" and "fallbacks" not in kw["extra_body"] and "betas" not in kw
    with pytest.raises(LLMError, match="refused"):
        AnthropicLLM(client=_AnthropicClient(_anthropic_resp("", "refusal"))).complete_json("s", "u", SCHEMA)
    with pytest.raises(LLMError, match="max_tokens"):
        AnthropicLLM(client=_AnthropicClient(_anthropic_resp("{", "max_tokens"))).complete_json("s", "u", SCHEMA)


# ---------------------------------------------------------------------------------------------- real SDKs, mocked HTTP
httpx = pytest.importorskip("httpx")


def test_anthropic_sdk_wire_format():
    anthropic = pytest.importorskip("anthropic")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(GOOD)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    out = AnthropicLLM("claude-opus-5-5", client=client).complete_json("SYS", "USER", SCHEMA, effort="medium")
    assert out == GOOD
    assert seen["url"].endswith("/v1/messages?beta=true") or seen["url"].endswith("/v1/messages")
    assert FALLBACK_BETA in seen["headers"]["anthropic-beta"]
    body = seen["body"]
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "medium"
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_sdk_wire_format():
    openai = pytest.importorskip("openai")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6-luna", "status": "completed",
            "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": json.dumps(GOOD), "annotations": []}]}],
            "usage": {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        })

    client = openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = OpenAILLM("gpt-6-luna", client=client)
    assert llm.complete_json("SYS", "USER", SCHEMA, name="x", effort="low") == GOOD
    assert seen["url"].endswith("/responses")
    body = seen["body"]
    assert body["text"]["format"]["strict"] is True and body["text"]["format"]["schema"] == SCHEMA
    assert body["reasoning"] == {"effort": "low"} and body["instructions"] == "SYS"
    assert llm.meter.input_tokens == 12
