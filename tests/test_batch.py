import json
from types import SimpleNamespace

import pytest

from ospm.llm import UsageMeter
from ospm.llm.base import BudgetExceeded
from ospm.llm.batch import build_requests, complete_json_batch

SCHEMA = {"type": "object", "properties": {"p": {"type": "number"}}, "required": ["p"], "additionalProperties": False}


def _ok(cid, text, tin=100, tout=50):
    return {"custom_id": cid, "error": None, "response": {"status_code": 200, "body": {
        "status": "completed", "usage": {"input_tokens": tin, "output_tokens": tout},
        "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": text}]}]}}}


class FakeClient:
    def __init__(self, lines):
        self.lines, self.submitted, self.polls = lines, None, 0
        self.files = SimpleNamespace(create=self._create_file, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

    def _create_file(self, file, purpose):
        assert purpose == "batch"
        self.submitted = [json.loads(x) for x in file[1].getvalue().decode().splitlines()]
        return SimpleNamespace(id="file-in")

    def _create_batch(self, input_file_id, endpoint, completion_window):
        assert endpoint == "/v1/responses"
        return SimpleNamespace(id="batch-1")

    def _retrieve(self, batch_id):
        self.polls += 1
        done = self.polls > 1
        return SimpleNamespace(status="completed" if done else "in_progress", output_file_id="file-out" if done else None,
                               error_file_id=None, request_counts=SimpleNamespace(completed=0, total=3))

    def _content(self, fid):
        return SimpleNamespace(text="\n".join(json.dumps(x) for x in self.lines))


def test_batch_parses_records_cost_at_half_price_and_resumes(tmp_path):
    lines = [_ok("a", '{"p": 0.7}'), _ok("b", "not json"),
             {"custom_id": "c", "error": {"message": "boom"}, "response": None}]
    client = FakeClient(lines)
    meter = UsageMeter(1.0)
    items = [("a", "sys", "u1"), ("b", "sys", "u2"), ("c", "sys", "u3")]
    state = tmp_path / "state.json"
    out = complete_json_batch("gpt-6-luna", items, SCHEMA, meter=meter, max_tokens=100, state_path=state,
                              poll_s=0, client=client, log=lambda m: None)
    assert out["a"] == {"p": 0.7}
    assert "non-JSON" in out["b"] and "boom" in out["c"]
    assert [r["custom_id"] for r in client.submitted] == ["a", "b", "c"]
    assert client.submitted[0]["body"]["text"]["format"]["strict"] is True
    # two requests with usage: 200 in / 100 out tokens at half of (0.10, 0.50) per M
    assert meter.cost_usd == pytest.approx(0.5 * (200 * 0.10 + 100 * 0.50) / 1e6)
    assert json.loads(state.read_text())["batch_id"] == "batch-1"
    # a re-run resumes the recorded batch instead of submitting a new one
    client2 = FakeClient(lines)
    complete_json_batch("gpt-6-luna", items, SCHEMA, meter=UsageMeter(1.0), state_path=state, poll_s=0,
                        client=client2, log=lambda m: None)
    assert client2.submitted is None


def test_batch_refuses_when_worst_case_exceeds_budget():
    meter = UsageMeter(0.001)
    with pytest.raises(BudgetExceeded):
        complete_json_batch("gpt-6.1-sol", [("a", "s", "u" * 4000)], SCHEMA, meter=meter, max_tokens=16000,
                            client=FakeClient([]), log=lambda m: None)


def test_build_requests_effort_optional():
    r = build_requests("m", [("x", "s", "u")], SCHEMA, effort=None)[0]
    assert r["url"] == "/v1/responses" and "reasoning" not in r["body"]


def test_cached_input_is_billed_at_the_cache_price():
    from ospm.llm.base import call_cost
    # gpt-6-luna: 1M prompt tokens of which 900k cached -> 100k * $0.10 + 900k * $0.01 per M
    assert call_cost("gpt-6-luna", 1_000_000, 0, cached_tokens=900_000) == pytest.approx(0.01 + 0.009)
    # anthropic cache writes cost 1.25x input
    assert call_cost("claude-opus-5-5", 1_000_000, 0, cache_write_tokens=1_000_000) == pytest.approx(5.0)
    meter = UsageMeter()
    assert meter.record("gpt-6-luna", 1000, 0, cached_tokens=1000) == pytest.approx(1000 * 0.01 / 1e6)
    assert meter.snapshot()["cached_tokens"] == 1000
