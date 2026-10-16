"""Derived latency: the critical path through an episode's agents."""

import json

from ospm.harness.latency import CALL_OVERHEAD_S, DECODE_TPS, NOMINAL_TOOL_S, PREFILL_TPS, derived_latency, timed


def step(n, inp, out, calls=(), event=""):
    """calls: (name, output, seconds or None)"""
    return {"n": n, "input_tokens": inp, "output_tokens": out, "event": event,
            "calls": [{"name": c[0], "id": f"c{n}{i}"} for i, c in enumerate(calls)],
            "results": [{"id": f"c{n}{i}", "output": c[1], **({"s": c[2]} if c[2] is not None else {})}
                        for i, c in enumerate(calls)]}


def model(new_in, out):
    return new_in / PREFILL_TPS + out / DECODE_TPS + CALL_OVERHEAD_S


def agent(aid, steps, children=()):
    return {"agent_id": aid, "steps": steps, "children": list(children)}


def test_single_agent_prefills_only_new_tokens_and_adds_tool_time():
    tr = agent("r", [step(1, 4000, 80, [("bash", "x", 2.0)]), step(2, 6000, 160, [("bash", "y", 3.0)])])
    assert derived_latency(tr) == round(model(4000, 80) + 2 + model(2000, 160) + 3, 2)
    assert derived_latency(tr, measured=False) == round(model(4000, 80) + model(2000, 160) + 2 * NOMINAL_TOOL_S, 2)
    assert timed(tr) and not timed(agent("r", [step(1, 10, 1, [("bash", "x", None)])]))


def test_parallel_children_overlap_and_the_parent_waits_for_the_slowest():
    slow = agent("a", [step(1, 4000, 80, [("bash", "", 30.0)])])
    fast = agent("b", [step(1, 4000, 80, [("bash", "", 5.0)])])
    spawn = [("spawn_agent", json.dumps({"agent_id": "a"}), 0.0), ("spawn_agent", json.dumps({"agent_id": "b"}), 0.0)]
    waited = json.dumps({"notifications": [{"type": "finished", "agent_id": "a"}, {"type": "finished", "agent_id": "b"}]})
    orch = agent("o", [step(1, 4000, 80, spawn), step(2, 5000, 80, [("wait_for_agents", waited, 99.0)]),
                       step(3, 6000, 80)], [slow, fast])
    t1 = model(4000, 80)
    child_end = t1 + model(4000, 80) + 30.0  # the slow child; the fast one overlaps it
    assert derived_latency(orch) == round(max(child_end, t1 + model(1000, 80)) + model(1000, 80), 2)
    # run in sequence (spawn, wait, spawn, wait) the same children take longer
    seq = agent("o", [step(1, 4000, 80, spawn[:1]), step(2, 5000, 80, [("wait_for_agents", waited, 0.0)]),
                      step(3, 6000, 80, spawn[1:]), step(4, 7000, 80, [("wait_for_agents", waited, 0.0)])],
                [slow, fast])
    assert derived_latency(seq) > derived_latency(orch)


def test_an_auto_delivered_notification_is_a_dependency():
    child = agent("a", [step(1, 4000, 80, [("bash", "", 50.0)])])
    note = '<notification>{"type": "finished", "agent_id": "a"}</notification>'
    orch = agent("o", [step(1, 4000, 80, [("spawn_agent", json.dumps({"agent_id": "a"}), 0.0)]),
                       {"n": 1, "input_tokens": 0, "output_tokens": 0, "event": "message", "calls": [],
                        "results": [{"id": "", "output": note}]},
                       step(2, 5000, 80)], [child])
    assert derived_latency(orch) == round(model(4000, 80) + model(4000, 80) + 50.0 + model(1000, 80), 2)
    assert derived_latency(None) == 0.0
