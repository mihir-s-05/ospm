"""Team evolution (ospm.learning.team_evolution): edits at three levels, per-agent credit, cost-aware acceptance."""

from __future__ import annotations

import json

from ospm.learning.team_evolution import TeamEvoConfig, TeamEvolver, apply_edit, credit_table, episode_credit, level_of
from ospm.llm.fake import FakeLLM
from ospm.replay.semantic import Matcher
from ospm.replay.tasks import ReplayTask
from ospm.team.roles import seed_library


def edit(op, target, text="", index=-1, description="", contract=""):
    return {"op": op, "target": target, "index": index, "text": text, "description": description,
            "contract": contract, "rationale": "r", "source_tasks": []}


def test_edits_at_each_level_are_immutable_and_validated():
    g = seed_library()
    g2 = apply_edit(g, edit("add_rule", "orchestrator", "Do not re-read files a specialist was asked to review."))
    assert level_of(edit("add_rule", "orchestrator")) == "team" and g2.roles["orchestrator"].playbook[-1].startswith("Do not")
    assert g.roles["orchestrator"].playbook == [] and g2.id != g.id  # the input is untouched
    g3 = apply_edit(g2, edit("edit_rule", "orchestrator", "Never re-read delegated files.", index=0))
    assert g3.roles["orchestrator"].playbook == ["Never re-read delegated files."]
    assert apply_edit(g3, edit("remove_rule", "orchestrator", index=0)).roles["orchestrator"].playbook == []
    assert apply_edit(g3, edit("edit_rule", "orchestrator", "x", index=5)) is None  # bad index
    g4 = apply_edit(g, edit("set_description", "security", "Only for auth, crypto, deserialisation changes."))
    assert level_of(edit("set_description", "security")) == "agent" and g4.roles["security"].description.startswith("Only")
    g5 = apply_edit(g, edit("add_type", "API compat", "Check public API breaks.", description="API changes"))
    assert "api_compat" in g5.specialists() and level_of(edit("add_type", "x")) == "library"
    assert apply_edit(g, edit("add_type", "review", "x", description="d")) is None  # name taken
    g6 = apply_edit(g, edit("remove_type", "security"))
    assert "security" not in g6.specialists() and "security" in g.specialists()
    lone = g
    for r in [r for r in g.specialists() if r not in ("review",)]:
        lone = apply_edit(lone, edit("remove_type", r)) or lone
    assert apply_edit(lone, edit("remove_type", "review")) is None  # the last reviewer stays


def task(tid, pr, positive, as_of, path="a.py"):
    concerns = [{"path": path, "line": 10, "body": "races on the cache", "themes": [], "nit": False}]
    return ReplayTask(tid, "concern", as_of, "sha", "event", {"positive": positive,
                                                              "concerns": concerns if positive else []}, pr=pr)


def matcher(tmp_path):
    m = Matcher(tmp_path / "sem.jsonl", workers=1)
    m._ask = lambda path, concern, comment: 0.95 if "race" in comment["body"] else 0.02
    return m


def child(role, comments, cost=0.01, steps=5):
    return {"role": role, "cost": cost, "steps": [{}] * steps, "exit_status": "Submitted",
            "submission": json.dumps({"verdict": "comment", "comments": comments, "report": "r"}), "children": []}


def test_credit_attributes_findings_cost_and_clean_pr_comments(tmp_path):
    m = matcher(tmp_path)
    race = {"path": "a.py", "line": 12, "body": "this races"}
    tr = {"cost": 0.02, "steps": [{"calls": [{"name": "spawn_agent"}, {"name": "spawn_agent"}, {"name": "bash"}]}],
          "children": [child("review", [race]), child("security", [], cost=0.004, steps=7)]}
    cr = episode_credit(tr, task("t1", 1, True, "2026-01-01"), m)
    assert cr["agents"][0]["found"] == [0] and cr["agents"][1]["found"] == []
    assert cr["orchestrator"] == {"cost": 0.02, "bash_steps": 1, "coordination": {"spawn_agent": 2}}
    tr_clean = {"cost": 0.01, "steps": [], "children": [child("review", [race]), child("security", [])]}
    table = credit_table([(task("t1", 1, True, "2026-01-01"), cr),
                          (task("t2", 2, False, "2026-01-02"), episode_credit(tr_clean, task("t2", 2, False, "x"), m))])
    assert table["review"]["found"] == 1 and table["review"]["found_alone"] == 1
    assert table["review"]["clean_pr_comments"] == 1 and table["security"]["found"] == 0
    assert table["security"]["spawned"] == 2 and table["security"]["mean_steps"] == 6


def evolver(tmp_path, quality, cost, edits):
    """quality(genome, task) and cost(genome) drive a fake runner; the reflector proposes `edits` once."""
    seen = []

    def run(t, ctx="", skills=(), role_ctx=(), genome=None):
        seen.append((t.id, genome.id))
        return quality(genome, t), {"comments": []}

    run.cost_of = lambda t, ctx="", skills=(), role_ctx=(), genome=None: cost(genome)
    run.trace_of = lambda t, ctx="", skills=(), role_ctx=(), genome=None: {"cost": cost(genome), "steps": [],
                                                                           "children": []}
    it = iter([edits])
    past = [task(f"p{i}", 100 + i, True, f"2026-01-0{i + 1}") for i in range(4)] + \
           [task(f"n{i}", 200 + i, False, f"2026-01-0{i + 1}") for i in range(4)]
    evo = TeamEvolver(None, run, FakeLLM(lambda s, u, sc: {"edits": next(it, [])}), matcher(tmp_path),
                      seed_library(), TeamEvoConfig(batch=4, workers=1), past=past, log=lambda m: None)
    return evo, seen


TRAIN = [task("tr1", 1, True, "2026-02-01"), task("tr2", 2, False, "2026-02-02")]


def test_a_cost_cut_that_keeps_quality_is_adopted(tmp_path):
    cut = edit("remove_type", "security")
    evo, _ = evolver(tmp_path, lambda g, t: 0.5, lambda g: 0.03 if "security" in g.specialists() else 0.02, [cut])
    evo.learn(TRAIN)
    rec = evo.edits[0]
    assert rec["status"] == "adopted" and rec["validation"]["accepted_for"] == "cost"
    assert "security" not in evo.genome.specialists()


def test_quality_gains_must_not_cost_too_much_and_losses_are_rejected(tmp_path):
    rule = edit("add_rule", "orchestrator", "Spawn the tests type for changes without tests.")
    better = lambda g, t: 0.8 if g.roles["orchestrator"].playbook else 0.5  # noqa: E731
    evo, _ = evolver(tmp_path, better, lambda g: 0.05 if g.roles["orchestrator"].playbook else 0.02, [rule])
    evo.learn(TRAIN)
    assert evo.edits[0]["status"] == "rejected" and evo.edits[0]["validation"]["cost_ratio"] == 2.5
    evo, _ = evolver(tmp_path, better, lambda g: 0.025 if g.roles["orchestrator"].playbook else 0.02, [rule])
    evo.learn(TRAIN)
    assert evo.edits[0]["status"] == "adopted" and evo.edits[0]["validation"]["accepted_for"] == "quality"
    worse = lambda g, t: 0.4 if "security" not in g.specialists() else 0.5  # noqa: E731
    evo, _ = evolver(tmp_path, worse, lambda g: 0.01 if "security" not in g.specialists() else 0.03,
                     [edit("remove_type", "security")])
    evo.learn(TRAIN)
    assert evo.edits[0]["status"] == "rejected"  # cheaper, but finds less


def test_validation_uses_other_known_prs_never_the_source(tmp_path):
    e = edit("add_rule", "orchestrator", "rule")
    e["source_tasks"] = ["p0"]
    evo, seen = evolver(tmp_path, lambda g, t: 0.5, lambda g: 0.02, [e])
    evo.learn(TRAIN)
    val = evo.edits[0]["validation"]["tasks"]
    assert "p0" not in val and len([v for v in val if v.startswith(("p", "tr1"))]) == 3
    assert all(v in {t for t, _ in seen} for v in val)
