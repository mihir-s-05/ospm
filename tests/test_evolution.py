"""GEPA engine and mutation operators."""

import random

import pytest

from ospm.agents import CITriageAgent
from ospm.benchmarks import Example, Task, get_task
from ospm.evolution import GEPA, Candidate, GEPAConfig
from ospm.evolution.operators import (
    PARAM_BOUNDS,
    LLMReflector,
    RuleInducer,
    Trace,
    merge,
    mutate_exemplars,
    mutate_params,
)
from ospm.genome import Genome, Rule
from ospm.llm.fake import FakeLLM
from ospm.models import CIRun


# ---------------------------------------------------------------------------------------------- operators
def _trace(text, gold, pred, scores=None):
    return Trace("x", text, gold, pred, 0.0, [], "", "wrong", scores or {})


def test_rule_inducer_learns_specific_phrase_and_respects_lift():
    texts = {f"t{i}": f"job failed: ECONNRESET while downloading package {i}" for i in range(4)}
    texts.update({f"u{i}": f"job failed: assertion error in test number {i}" for i in range(6)})
    labels = {k: {"category": "infra_flaky" if k.startswith("t") else "test_failure"} for k in texts}
    ind = RuleInducer(CITriageAgent, texts, labels, min_pos=3)
    rule = ind.induce_positive(_trace("ERROR read ECONNRESET", {"category": "infra_flaky"}, {}), "category",
                               "infra_flaky", margin=0.5)
    assert rule is not None and rule.any_of == ["econnreset"] and rule.value == "infra_flaky"
    assert 0.4 <= rule.weight <= 2.5 and rule.origin == "induced"
    # a phrase shared by every class has no lift and is never induced
    assert ind.induce_positive(_trace("job failed", {"category": "infra_flaky"}, {}), "category",
                               "infra_flaky", 0.5) is None


def test_rule_inducer_vetoes_misfiring_rule():
    texts = {"a": "print(x) in src/lib.py", "b": "print(y) in src/core.py", "c": "file tests conftest print(z)"}
    labels = {"a": {"findings": ["debug_print"]}, "b": {"findings": ["debug_print"]}, "c": {"findings": []}}
    ind = RuleInducer(CITriageAgent, texts, labels, min_pos=1)
    rule = Rule(id="p", field="findings", value="debug_print", any_of=["print"])
    vetoed = ind.induce_veto(rule, _trace(texts["c"], {"findings": []}, {"findings": ["debug_print"]}))
    assert vetoed is not None and vetoed.none_of and vetoed.none_of[0] in {"tests", "conftest", "tests conftest"}
    assert not vetoed.matches(texts["c"].lower(), __import__("ospm.text", fromlist=["normalize"]).normalize(texts["c"]))


def test_mutate_params_stays_in_bounds():
    rng = random.Random(1)
    g = Genome(agent="triage", instructions="", params={"knn_k": 15, "min_sim": 0.6, "knn_weight": 0.0})
    for _ in range(50):
        child = mutate_params(g, rng)
        if child is None:
            continue
        for k, v in child.params.items():
            lo, hi, is_int = PARAM_BOUNDS[k]
            assert lo <= v <= hi and (not is_int or v == int(v))
        g = child


def test_mutate_exemplars_targets_failed_labels():
    traces = [Trace("e1", "t", {"kind": "docs"}, {"kind": "bug"}, 0.0, [], "", "")]
    train = {"e1": {"kind": "docs"}, "e2": {"kind": "docs"}, "e3": {"kind": "bug"}}
    child = mutate_exemplars(Genome(agent="triage", instructions=""), traces, train, random.Random(0))
    assert child.exemplars == ["e2"]  # the failing example itself is never pinned


def test_merge_takes_changed_modules_and_unions_playbooks():
    base = Genome(agent="triage", instructions="base", params={"knn_k": 5})
    base.add_rule(Rule(id="r0", field="kind", value="bug", any_of=["crash"]))
    a = base.clone()
    a.instructions = "improved instructions"
    a.add_rule(Rule(id="ra", field="kind", value="docs", any_of=["readme"]))
    b = base.clone()
    b.params["knn_k"] = 8
    b.add_rule(Rule(id="rb", field="kind", value="question", any_of=["how do i"]))
    child = merge(a, b, base, a_better=True)
    assert child.instructions == "improved instructions" and child.params["knn_k"] == 8
    assert {r.id for r in child.rules} == {"r0", "ra", "rb"}
    assert child.parents == [a.id, b.id]
    assert merge(a, a.clone(), base, True) is None  # nothing new


def test_llm_reflector_applies_validated_edits():
    def responder(system, user, schema):
        assert "Execution traces" in user and "Correct answer" in user
        return {
            "diagnosis": "misses flaky network errors",
            "instructions": "Treat transient network errors as infra_flaky.",
            "add_rules": [
                {"field": "category", "value": "infra_flaky", "any_of": ["econnreset"], "all_of": [], "none_of": [],
                 "pattern": "", "weight": 9.0, "rationale": "network"},
                {"field": "category", "value": "infra_flaky", "any_of": [], "all_of": [], "none_of": [],
                 "pattern": "(", "weight": 1.0, "rationale": "invalid regex is dropped"},
            ],
            "remove_rule_ids": ["timeout"],
            "reweight": [{"id": "mypy", "weight": 0.01}],
        }
    g = CITriageAgent.seed_genome()
    child = LLMReflector(FakeLLM(responder), CITriageAgent).mutate(
        g, [Trace("c", "read ECONNRESET", {"category": "infra_flaky"}, {"category": "timeout"}, 0.0, [], "", "x")])
    assert child.instructions == "Treat transient network errors as infra_flaky."
    ids = {r.id: r for r in child.rules}
    assert "timeout" not in ids and ids["mypy"].weight == 0.1
    new = [r for r in child.rules if r.origin == "llm"]
    assert len(new) == 1 and new[0].weight == 3.0 and new[0].any_of == ["econnreset"]
    assert child.parents == [g.id] and child.origin == "llm-reflection"
    # nothing to learn from perfect traces -> no call
    assert LLMReflector(FakeLLM(responder), CITriageAgent).mutate(
        g, [Trace("c", "t", {"category": "lint"}, {"category": "lint"}, 1.0, [], "", "")]) is None


# ---------------------------------------------------------------------------------------------- engine
def test_pareto_front_and_selection():
    task = get_task("ci_triage")
    gepa = GEPA(task)
    ids = [e.id for e in gepa.val]
    g1, g2, g3 = (Genome(agent="ci_triage", instructions=str(i)) for i in range(3))
    c1 = Candidate(g1, {i: 1.0 if n == 0 else 0.0 for n, i in enumerate(ids)}, "seed", 0)
    c2 = Candidate(g2, {i: 1.0 if n == 1 else 0.0 for n, i in enumerate(ids)}, "x", 1)
    c3 = Candidate(g3, {i: 0.0 for i in ids}, "x", 2)  # dominated by both
    front = gepa.pareto_front([c1, c2, c3])
    assert set(front) == {g1.id, g2.id}
    assert all(gepa.select_parent([c1, c2, c3]).id in front for _ in range(20))


def test_gepa_offline_improves_and_is_deterministic(memory):
    task = get_task("ci_triage")
    r1 = GEPA(task, config=GEPAConfig(max_iterations=20, seed=0), memory=memory).run()
    r2 = GEPA(get_task("ci_triage"), config=GEPAConfig(max_iterations=20, seed=0)).run()
    assert r1.best.mean >= r1.seed.mean
    assert r1.best.mean > r1.seed.mean  # CI triage has learnable lexical structure
    assert r1.best.id == r2.best.id and [h.get("status") for h in r1.history] == [h.get("status") for h in r2.history]
    assert set(r1.test) == {"seed", "best"}
    run = memory.runs("ci_triage")[0]
    assert run["summary"]["best_id"] == r1.best.id
    statuses = {row["status"] for row in memory.lineage(r1.run_id)}
    assert "accepted" in statuses
    for c in r1.candidates:  # every accepted candidate is persisted
        assert memory.load_genome(c.id) is not None


def test_gepa_respects_metric_call_budget():
    r = GEPA(get_task("triage"), config=GEPAConfig(max_iterations=100, max_metric_calls=150, seed=0)).run()
    assert r.stop_reason == "max_metric_calls"
    assert r.metric_calls < 150 + 2 * 60 + 40  # may finish the iteration in flight (+ final test eval)


def test_gepa_operator_validation():
    with pytest.raises(ValueError):
        GEPA(get_task("ci_triage"), config=GEPAConfig(operators=("reflection",)))  # no reflection LLM
    with pytest.raises(ValueError):
        GEPA(Task("ci_triage", [], get_task("ci_triage").project, {}))


def test_gepa_with_llm_reflection_uses_reflection_operator():
    seen = []

    def responder(system, user, schema):
        seen.append(user)
        return {"diagnosis": "d", "instructions": "new policy", "remove_rule_ids": [], "reweight": [],
                "add_rules": [{"field": "category", "value": "infra_flaky", "any_of": ["econnreset", "503", "502"],
                               "all_of": [], "none_of": [], "pattern": "", "weight": 1.5, "rationale": "r"}]}
    r = GEPA(get_task("ci_triage"), reflection_llm=FakeLLM(responder),
             config=GEPAConfig(max_iterations=6, seed=0)).run()
    assert r.history[0]["op"] in ("reflection", "merge") or any(h.get("op") == "reflection" for h in r.history)
    assert seen, "reflection LLM was never called"
    assert r.llm_usage["reflection"]["calls"] >= 1


def test_tiny_custom_task_end_to_end(project):
    """A task built from scratch (as feedback-derived tasks are) runs through GEPA."""
    logs = {"dependency": "ERROR: ResolutionImpossible conflicting dependencies",
            "lint": "would reformat src/x.py", "infra_flaky": "Error: read ECONNRESET"}
    exs, split = [], {}
    for i in range(15):
        cat = list(logs)[i % 3]
        ex = Example(f"c{i}", CIRun(f"job{i}", "failure", f"{logs[cat]} run {i}", run_id=f"c{i}"), {"category": cat})
        exs.append(ex)
        split[ex.id] = ("train", "val", "train", "test", "train")[i % 5]
    r = GEPA(Task("ci_triage", exs, project, split), config=GEPAConfig(max_iterations=10, seed=1)).run()
    assert 0.0 <= r.best.mean <= 1.0 and r.best.mean >= r.seed.mean


def test_policy_llm_failures_abort_cleanly_and_are_not_cached():
    from ospm.evolution import EvaluationAborted, evaluate
    from ospm.llm import UsageMeter

    task = get_task("ci_triage")
    calls = {"n": 0}

    def flaky(system, user, schema):
        calls["n"] += 1
        if calls["n"] > 30:  # healthy for the seed's val/train evaluation, then the API starts failing
            return {"category": "not-a-category"}
        return {"category": "test_failure", "root_cause": "", "suggested_fix": "", "failing_tests": [],
                "rationale": "r"}

    llm = FakeLLM(flaky, meter=UsageMeter(None))
    r = GEPA(task, policy_llm=llm, config=GEPAConfig(max_iterations=10, seed=0, workers=1)).run()
    assert r.stop_reason.startswith("aborted")
    assert r.best is r.seed or r.best.mean >= r.seed.mean
    # degraded fallback decisions are never memoised
    cache = {}
    agent = task.make_agent(llm=FakeLLM(lambda s, u, sch: {"bad": 1}))
    res = evaluate(agent, task.subset("val")[:2], cache=cache)
    assert res.llm_errors == 2 and cache == {}
    with pytest.raises(EvaluationAborted):
        GEPA(task, policy_llm=FakeLLM(lambda s, u, sch: {"bad": 1})).run()
