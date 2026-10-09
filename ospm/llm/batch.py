"""OpenAI Batch API for non-interactive JSON calls (labelling, judging, reflection): half the list price.

Batches complete asynchronously (minutes to hours, 24h at most), so this suits one-shot calls only. Multi-turn agent
episodes, where each turn depends on the previous observation, keep using the synchronous API.

    results = complete_json_batch("gpt-6.1-sol", [(id, system, user), ...], schema, meter=meter,
                                  state_path=Path("data/batches/probe.json"))

`state_path` records the batch id, so re-running after an interruption resumes the same batch instead of paying again.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Callable

from .base import BudgetExceeded, UsageMeter, price_for, validate

BATCH_DISCOUNT = 0.5
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def build_requests(model: str, items: list[tuple[str, str, str]], schema: dict, *, name: str = "output",
                   effort: str | None = "medium", max_tokens: int = 16000) -> list[dict[str, Any]]:
    reqs = []
    for cid, system, user in items:
        body: dict[str, Any] = {"model": model, "instructions": system, "input": user, "max_output_tokens": max_tokens,
                                "text": {"format": {"type": "json_schema", "name": name, "schema": schema,
                                                    "strict": True}}}
        if effort:
            body["reasoning"] = {"effort": effort}
        reqs.append({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": body})
    return reqs


def parse_output_line(line: dict[str, Any], schema: dict) -> tuple[dict | None, str, int, int, int]:
    """(parsed json or None, error, input, output, cached input tokens) for one line of a batch output file."""
    resp = line.get("response") or {}
    body = resp.get("body") or {}
    usage = body.get("usage") or {}
    tin, tout = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    tcached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    if line.get("error") or resp.get("status_code") != 200:
        return None, f"request failed: {line.get('error') or body.get('error')}", tin, tout, tcached
    if body.get("status") != "completed":
        return None, f"response {body.get('status')}: {body.get('incomplete_details')}", tin, tout, tcached
    text = "".join(c.get("text", "") for item in body.get("output", []) if item.get("type") == "message"
                   for c in item.get("content", []) if c.get("type") == "output_text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, f"non-JSON output: {text[:200]!r}", tin, tout, tcached
    problems = validate(data, schema)
    return (None, f"schema: {problems[:3]}", tin, tout, tcached) if problems else (data, "", tin, tout, tcached)


def estimate_cost(model: str, items: list[tuple[str, str, str]], max_tokens: int) -> float:
    """Upper bound on the batch cost (~4 chars per input token, every request using max_tokens)."""
    pin, pout = price_for(model)
    tin = sum(len(s) + len(u) for _, s, u in items) / 4
    return BATCH_DISCOUNT * (tin * pin + len(items) * max_tokens * pout) / 1e6


def complete_json_batch(model: str, items: list[tuple[str, str, str]], schema: dict, *, meter: UsageMeter,
                        name: str = "output", effort: str | None = "medium", max_tokens: int = 16000,
                        state_path: Path | None = None, poll_s: float = 20.0, timeout_s: float = 24 * 3600,
                        client: Any = None, log: Callable[[str], None] = print) -> dict[str, dict | str]:
    """Run `items` (custom_id, system, user) through the Batch API. Returns custom_id -> parsed dict, or an error
    string for requests that failed. Refuses to submit if the worst-case cost would exceed the meter's budget."""
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    state = json.loads(state_path.read_text()) if state_path and state_path.exists() else {}
    batch_id = state.get("batch_id")
    if not batch_id:
        worst = estimate_cost(model, items, max_tokens)
        if meter.budget_usd is not None and meter.cost_usd + worst > meter.budget_usd:
            raise BudgetExceeded(f"batch worst case ${worst:.3f} would exceed the ${meter.budget_usd} budget "
                                 f"(spent ${meter.cost_usd:.3f}); lower max_tokens or raise the budget")
        lines = "\n".join(json.dumps(r) for r in build_requests(model, items, schema, name=name, effort=effort,
                                                                  max_tokens=max_tokens))
        f = client.files.create(file=("batch.jsonl", io.BytesIO(lines.encode())), purpose="batch")
        batch_id = client.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h").id
        log(f"[batch] submitted {batch_id}: {len(items)} requests to {model} (worst case ${worst:.3f})")
        if state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"batch_id": batch_id, "model": model, "n": len(items)}))
    t0 = time.time()
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in TERMINAL:
            break
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"batch {batch_id} still {b.status}; re-run to resume")
        rc = getattr(b, "request_counts", None)
        log(f"[batch] {batch_id} {b.status} {getattr(rc, 'completed', 0)}/{getattr(rc, 'total', '?')}")
        time.sleep(poll_s)
    out: dict[str, dict | str] = {}
    for fid in (b.output_file_id, b.error_file_id):
        if not fid:
            continue
        for raw in client.files.content(fid).text.splitlines():
            if not raw.strip():
                continue
            line = json.loads(raw)
            data, err, tin, tout, tcached = parse_output_line(line, schema)
            meter.record(model, tin, tout, discount=BATCH_DISCOUNT, cached_tokens=tcached)
            if err:
                meter.record_error()
            out[line["custom_id"]] = data if data is not None else err
    for cid, *_ in items:
        out.setdefault(cid, f"missing from batch output (batch {b.status})")
    log(f"[batch] {batch_id} {b.status}: {sum(isinstance(v, dict) for v in out.values())}/{len(items)} ok")
    return out
