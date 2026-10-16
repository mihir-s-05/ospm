"""Derived latency of an episode: the critical path through its agents, as in DeepSeek Harness's Agent Teams.

Wall-clock time is a poor objective here: it depends on rate-limit retries, how many episodes share the API and the
machine, and the provider's queueing. Instead every step is costed from what it did:

    model time = new prompt tokens / PREFILL_TPS + output tokens / DECODE_TPS + CALL_OVERHEAD_S
    tool time  = the measured duration of each bash / tool call (or NOMINAL_TOOL_S per call for traces recorded before
                 durations were, or with measured=False); coordination calls are instant

"New" prompt tokens assume a warm prefix cache: a step only prefills what was added since the previous step. A child
starts when it is spawned (after its parent's model turn) and runs its steps in sequence; a parent that receives a
child's `finished` notification (from wait_for_agents or auto-delivered before a step) cannot continue before that
child's derived finish. The episode's latency is the entry agent's derived finish. Approximation: a child resumed by
a later message continues its own timeline instead of waiting for that message.
"""

from __future__ import annotations

import json
import re
from typing import Any

PREFILL_TPS = 4000.0  # nominal rates; only ratios between systems matter
DECODE_TPS = 80.0
CALL_OVERHEAD_S = 0.5
NOMINAL_TOOL_S = 1.0
INSTANT = {"spawn_agent", "send_message", "wait_for_agents", "close_agent", "message_parent", "post_note",
           "read_notes", "task_create", "task_update", "task_list"}
_NOTIFICATION = re.compile(r"<notification>(.*?)</notification>", re.DOTALL)


def _finished(text: str) -> list[str]:
    """Agent ids of the `finished` notifications in a wait_for_agents result or an auto-delivered message."""
    notes: list[Any] = []
    try:
        notes = json.loads(text).get("notifications") or []
    except (json.JSONDecodeError, AttributeError):
        for m in _NOTIFICATION.findall(text or ""):
            try:
                notes.append(json.loads(m))
            except json.JSONDecodeError:
                pass
    return [n["agent_id"] for n in notes if isinstance(n, dict) and n.get("type") == "finished" and n.get("agent_id")]


def _spawned(text: str) -> str:
    try:
        return str(json.loads(text).get("agent_id") or "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def _finish(trace: dict, start: float, measured: bool) -> float:
    children = {c.get("agent_id"): c for c in trace.get("children") or []}
    done: dict[str, float] = {}
    t, prev_in = start, 0
    for s in trace.get("steps") or []:
        results = s.get("results") or []
        if s.get("event") == "message":
            for aid in _finished(results[0].get("output", "") if results else ""):
                t = max(t, done.get(aid, t))
            continue
        if s.get("event") == "compaction":
            prev_in = 0
        n_in = int(s.get("input_tokens") or 0)
        if n_in or s.get("output_tokens"):
            new = n_in - prev_in if n_in >= prev_in else n_in
            t += new / PREFILL_TPS + int(s.get("output_tokens") or 0) / DECODE_TPS + CALL_OVERHEAD_S
            prev_in = n_in
        by_id = {r.get("id"): r for r in results}
        tools = 0.0
        for c in s.get("calls") or []:
            r, name = by_id.get(c.get("id"), {}), c.get("name")
            if name == "spawn_agent" and (aid := _spawned(r.get("output", ""))) in children:
                done[aid] = _finish(children[aid], t, measured)
            elif name == "wait_for_agents":
                for aid in _finished(r.get("output", "")):
                    tools = max(tools, done.get(aid, t) - t)
            elif name not in INSTANT:
                tools += float(r["s"]) if measured and "s" in r else NOMINAL_TOOL_S
        t += tools
    return t


def derived_latency(trace: dict | None, measured: bool = True) -> float:
    """Critical-path seconds of an episode trace (Trace.to_dict()); 0.0 for no trace."""
    return round(_finish(trace, 0.0, measured), 2) if trace else 0.0


def timed(trace: dict | None) -> bool:
    """Whether every tool call of the episode has a measured duration (traces before durations were recorded do not)."""
    if not trace:
        return False
    for s in trace.get("steps") or []:
        by_id = {r.get("id"): r for r in s.get("results") or []}
        if any(c.get("name") not in INSTANT and "s" not in by_id.get(c.get("id"), {}) for c in s.get("calls") or []):
            return False
    return all(timed(c) for c in trace.get("children") or [])
