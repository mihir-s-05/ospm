"""Post-mortem skill evolution (ospm.learning.postmortem): digests, immutable edits, the validated learning loop."""

from __future__ import annotations

from ospm.learning.interventions import InterventionLearner, LearnConfig, PRView
from ospm.learning.postmortem import SkillLearner, apply_edit, trace_digest
from ospm.learning.skills import TOOLS_NOTE, Skill
from ospm.llm.fake import FakeLLM
from ospm.replay.tasks import ReplayTask


def task(tid, pr, positive, as_of, path="src/mod.py"):
    concerns = [{"path": path, "line": 10, "body": "handle the edge case", "themes": ["correctness"], "nit": False}]
    return ReplayTask(tid, "concern", as_of, "sha" + tid, "event",
                      {"positive": positive, "concerns": concerns if positive else []}, pr=pr, split="train",
                      meta={"base": "base"})


def edit(op="create", name="src-area", part="both", body="Check CHECK-EDGE cases.", paths=("src/*.py",),
         sources=("p1",), description="Reviewing changes under src/"):
    return {"op": op, "role": "review", "part": part, "name": name, "description": description, "paths": list(paths), "body": body,
            "diagnosis": "d", "source_tasks": list(sources)}


def test_trace_digest():
    trace = {"steps": [{"calls": [{"name": "bash", "arguments": {"command": "skill src-area && sed -n 1,40p src/mod.py"}}]},
                       {"calls": [{"name": "bash", "arguments": {"command": "git diff base..HEAD"}}]}],
             "children": [{"steps": [{"calls": [{"name": "bash", "arguments": {"command": "skill other"}}]}],
                           "children": []}]}
    d = trace_digest(trace, ["src/mod.py", "src/other.py"])
    assert d["steps"] == 2 and d["loaded"] == ["src-area", "other"]
    assert d["opened"] == {"src/mod.py": True, "src/other.py": False} and len(d["commands"]) == 3
    assert trace_digest(None, ["a"])["commands"] == []


def test_apply_edit_is_immutable_and_targets_parts():
    lib, new = apply_edit([], edit())
    assert [s.name for s in lib] == ["src-area"] and new is lib[0] and new.version == 1
    lib2, v2 = apply_edit(lib, edit("edit", part="description", description="New when", body="ignored"))
    assert v2.description == "New when" and v2.body == new.body and v2.version == 2
    assert new.description == "Reviewing changes under src/"  # the old version is untouched
    lib3, v3 = apply_edit(lib2, edit("edit", part="body", body="New procedure", description="ignored"))
    assert v3.body == "New procedure" and v3.description == "New when"
    assert apply_edit(lib3, edit("edit", part="body", body="New procedure")) == (lib3, None)  # no-op
    lib4, gone = apply_edit(lib3, edit("delete"))
    assert lib4 == [] and gone is v3
    assert apply_edit([], edit("delete"))[1] is None and apply_edit([], edit(name="!!"))[1] is None
    assert apply_edit([], edit(body=""))[1] is None  # a new skill needs a body


def make(train, past, proposals, neg_comments_with_skill=False, entry="review"):
    calls = []

    def run(t, ctx, skills=(), role_ctx=()):
        calls.append((t.id, tuple(s.name for s in skills)))
        helped = any("CHECK-EDGE" in s.body for s in skills)
        if t.truth["positive"]:
            c = t.truth["concerns"][0]
            return (1.0, {"comments": [{"path": c["path"], "line": c["line"], "body": "x"}]}) if helped else (
                0.0, {"comments": []})
        if helped and neg_comments_with_skill:
            return 0.5, {"comments": [{"path": "src/mod.py", "line": 3, "body": "nit"}]}
        return 1.0, {"comments": []}

    def trace_of(t, ctx, skills=(), role_ctx=()):
        cmds = [f"skill {s.name}" for s in skills] + ["git diff base..HEAD"]
        return {"steps": [{"calls": [{"name": "bash", "arguments": {"command": c}}]} for c in cmds], "children": []}

    it = iter(proposals)
    learner = SkillLearner(None, run, FakeLLM(lambda s, u, sc: {"edits": next(it, [])}),
                           LearnConfig(workers=1, batch=4, max_candidates_per_step=4), log=lambda m: None, past=past,
                           trace_of=trace_of, entry=entry)
    for t in train + past:
        learner._views[t.id] = PRView(["src/mod.py"], "x = 1")
    return learner, calls


TRAIN = [task("p1", 1, True, "2024-01-05T00:00:00Z"), task("n1", 2, False, "2024-01-06T00:00:00Z")]
PAST = [task("p0", 3, True, "2024-01-01T00:00:00Z"), task("q0", 4, True, "2024-01-02T00:00:00Z"),
        task("n0", 5, False, "2024-01-03T00:00:00Z"), task("p1b", 1, True, "2024-01-04T00:00:00Z")]


def test_learner_adopts_a_skill_that_helps_other_prs_and_excludes_the_source_pr():
    learner, calls = make(TRAIN, PAST, [[edit()]])
    learner.learn(TRAIN)
    assert [s.name for s in learner.skills] == ["src-area"]
    rec = learner.edits[0]
    assert rec["status"] == "adopted" and rec["validation"]["pos_gain"] == 1.0
    assert rec["validation"]["n_pos"] == 2 and rec["validation"]["n_neg"] == 2
    assert not any(tid.startswith("p1") for tid in rec["validation"]["tasks"])  # all rounds of PR 1 excluded
    assert rec["validation"]["loaded"] == "4/4"
    ctx, offer, role_ctx = learner.serve(TRAIN[0], "skills")
    assert ctx.startswith(TOOLS_NOTE) and "- src-area:" in ctx and [s.name for s in offer] == ["src-area"]
    assert role_ctx == () and learner.serve(TRAIN[0], "tool") == (TOOLS_NOTE, (), ())
    assert learner.serve(TRAIN[0], "none") == ("", (), ())


def test_learner_rejects_edits_without_gain_or_with_false_alarms():
    learner, _ = make(TRAIN, PAST, [[edit(body="Look around.")]])
    learner.learn(TRAIN)
    assert learner.skills == [] and learner.edits[0]["status"] == "rejected"
    learner, _ = make(TRAIN, PAST, [[edit()]], neg_comments_with_skill=True)
    learner.learn(TRAIN)
    assert learner.skills == [] and learner.edits[0]["validation"]["fp_increase"] > 0
    lonely = [task("p9", 9, True, "2024-01-01T00:00:00Z", path="docs/x.md")]
    learner, _ = make(TRAIN, lonely, [[edit()]])
    for t in lonely:
        learner._views[t.id] = PRView(["docs/x.md"], "")
    learner.learn(TRAIN)
    assert "no independent support" in learner.edits[0]["validation"]["reason"]


def test_postmortems_cover_misses_and_false_alarms_only():
    learner, _ = make(TRAIN, PAST, [])
    hit = (1.0, {"comments": [{"path": "src/mod.py", "line": 10}]})
    items = learner.postmortem_items(TRAIN + [task("p2", 6, True, "2024-01-07T00:00:00Z")],
                                     [(0.0, {"comments": []}), (0.5, {"comments": [{"path": "a", "line": 1}]}), hit])
    assert [(i["task"], i["positive"], len(i["missed"])) for i in items] == [("p1", True, 1), ("n1", False, 0)]
    assert items[0]["digest"]["opened"] == {"src/mod.py": False}


def test_remember_fills_the_memory_baseline_like_learn():
    learner = InterventionLearner(None, lambda t, c: (0.0, {}), FakeLLM(lambda s, u, sc: {"interventions": []}),
                                  LearnConfig(workers=1), log=lambda m: None)
    learner.remember(TRAIN + PAST)
    assert learner.memory.n == 4  # one concern per positive task
    assert isinstance(Skill("A b", "d", "b").name, str)


def test_team_entry_serves_each_role_its_skills_and_digests_coordination():
    from ospm.learning.postmortem import team_digest

    lib = [Skill("coord", "brief the reviewer", "spawn review with the registry path", paths=["src/*"],
                 role="orchestrator"),
           Skill("rev", "review src", "CHECK-EDGE", paths=["src/*"]), Skill("sec", "security", "x", role="security")]
    learner, _ = make(TRAIN, PAST, [], entry="orchestrator")
    learner.skills = lib
    ctx, offer, role_ctx = learner.serve(TRAIN[0], "skills")
    assert "- coord:" in ctx and "- rev:" not in ctx and {s.name for s in offer} == {"coord", "rev", "sec"}
    rc = dict(role_ctx)
    assert "- rev:" in rc["review"] and "- sec:" in rc["security"] and "- coord:" not in rc["review"]
    assert learner.serve(TRAIN[0], "tool")[2] and not learner.serve(TRAIN[0], "tool")[1]
    trace = {"steps": [{"calls": [{"name": "spawn_agent", "arguments": {"role": "review", "prompt": "look at X"}},
                                  {"name": "bash", "arguments": {"command": "ls"}}]}],
             "children": [{"agent_id": "review-2", "role": "review", "exit_status": "Submitted",
                           "submission": '{"report": "checked X"}', "children": [],
                           "steps": [{"calls": [{"name": "bash", "arguments": {"command": "skill rev"}}]}]}]}
    d = team_digest(trace)
    assert d["coordination"] == ['spawn_agent({"role": "review", "prompt": "look at X"})']
    assert d["specialists"][0]["loaded"] == ["rev"] and "checked X" in d["specialists"][0]["final_message"]


def test_team_edit_of_an_orchestrator_skill_is_validated_where_that_role_is_offered():
    e = edit(name="coord")
    e["role"] = "orchestrator"
    learner, calls = make(TRAIN, PAST, [[e]], entry="orchestrator")
    learner.learn(TRAIN)
    rec = learner.edits[0]
    assert rec["edit"]["role"] == "orchestrator" and rec["status"] == "adopted"
    assert learner.skills[0].role == "orchestrator"
