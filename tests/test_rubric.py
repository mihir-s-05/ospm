"""Rubric mining (ospm.learning.rubric) with the Jev judges stubbed out."""

from __future__ import annotations

import json

from ospm.learning import rubric
from ospm.learning.skills import offered, rubric_block
from ospm.replay import recurrence

from test_recurrence import write_tasks


def test_rules_seen_on_enough_prs_before_the_test_period_become_area_skills(tmp_path, monkeypatch):
    note = "please add a release note in docs/releases for this"
    write_tasks(tmp_path, [
        (1, 1, "train", "docs/releases/6.1.txt", note),
        (2, 2, "train", "docs/releases/6.1.txt", "add a release note in docs/releases"),
        (3, 3, "val", "docs/releases/6.1.txt", "release note missing in docs/releases"),
        (4, 4, "train", "django/db/query.py", "this lookup is quadratic in the number of joins"),
        (5, 5, "train", "django/db/query.py", "this lookup is quadratic"),  # recurs, but on 2 PRs only
        (6, 9, "test", "docs/releases/6.2.txt", note),  # the test period never feeds the rubric
    ])

    def same_rule(self, path, earlier, later):
        return 0.9 if any(w in earlier["body"] and w in later["body"] for w in ("release", "quadratic")) else 0.1

    monkeypatch.setattr(recurrence.RuleJudge, "_ask", same_rule)
    monkeypatch.setattr(recurrence.SpecificJudge, "_ask", lambda self, p, e, l: 0.9)
    groups = rubric.clusters(tmp_path, sample=10, k=8, min_prs=3)
    assert [sorted(c["pr"] for c in g) for g in groups] == [[1, 2, 3]]

    written = {"areas": [{"name": "Release notes", "description": "Changes that need a release note.",
                          "paths": ["docs/*", "django/*"], "items": [
                              {"rule": "Every user-facing change has a release note",
                               "applies_when": "behaviour changes", "check": "a docs/releases/<version>.txt entry",
                               "example": note, "clusters": [0]}]},
                         {"name": "empty", "description": "", "paths": [], "items": []}]}
    skills = rubric.to_skills(written, groups)
    assert [s.name for s in skills] == ["rubric-release-notes"] and skills[0].provenance == ["pr:1", "pr:2", "pr:3"]
    assert "1. **Every user-facing change has a release note**" in skills[0].body
    mine = offered(skills, ["django/db/models.py"])
    assert mine and "## Review rubric" in rubric_block(mine) and skills[0].body in rubric_block(mine)
    assert not offered(skills, ["tests/test_x.py"])


def test_mcnemar_and_benjamini_hochberg():
    from ospm.learning.stats import bh, mcnemar, paired

    assert mcnemar(0, 0) == 1.0 and abs(mcnemar(5, 3) - 0.7266) < 1e-4 and abs(mcnemar(10, 0) - 2 / 1024) < 1e-12
    assert mcnemar(3, 5) == mcnemar(5, 3)
    assert [round(q, 4) for q in bh([0.01, 0.04, 0.03])] == [0.03, 0.04, 0.04]
    rows = {"none": [{"score": 0.0}, {"score": 1.0}, {"score": 0.5}],
            "a": [{"score": 1.0}, {"score": 1.0}, {"score": 0.0}]}
    assert paired(rows, "none") == {"a": {"wins": 1, "losses": 1, "ties": 1, "p": 1.0, "p_bh": 1.0}}


ITEM = {"rule": "Handle the edge case", "applies_when": "src changes", "check": "CHECK-EDGE in src/mod.py",
        "example": "handle the edge case"}


def area(items=(ITEM,), name="src"):
    return rubric.area_skill(name, "Source modules.", ["src/*.py"], list(items))


def test_item_edits_are_immutable_and_validated_for_shape():
    lib = [area()]
    new_lib, subject, item = rubric.apply_item_edit(lib, {"op": "add_item", "area": "src", "index": -1,
                                                          **{**ITEM, "rule": "Second rule"}})
    assert [len(s.items) for s in new_lib] == [2] and subject.version == 2 and "2. **Second rule**" in subject.body
    assert len(lib[0].items) == 1 and item["rule"] == "Second rule"
    narrower = {"rule": "", "applies_when": "", "check": "narrower", "example": ""}
    edited, s2, _ = rubric.apply_item_edit(new_lib, {"op": "edit_item", "area": "rubric-src", "index": 1, **narrower})
    assert s2.items[0] == {**ITEM, "check": "narrower"} and s2.items[1]["rule"] == "Second rule"
    assert rubric.apply_item_edit(edited, {"op": "edit_item", "area": "src", "index": 1, **s2.items[0]})[1] is None
    assert rubric.apply_item_edit(lib, {"op": "edit_item", "area": "src", "index": 5, **ITEM})[1] is None
    gone, old, removed = rubric.apply_item_edit(lib, {"op": "remove_item", "area": "src", "index": 1, **ITEM})
    assert gone == [] and old is lib[0] and removed == ITEM
    new_area = {"op": "add_item", "area": "Docs pages", "index": -1, **ITEM, "area_description": "",
                "area_paths": []}
    assert rubric.apply_item_edit(lib, new_area)[1] is None  # a new area needs a description and globs
    lib2, docs, _ = rubric.apply_item_edit(lib, {**new_area, "area_description": "Docs.", "area_paths": ["docs/*"]})
    assert docs.name == "rubric-docs-pages" and len(lib2) == 2 and docs.validation["source"] == "post-mortem"


def test_rubric_learner_adopts_an_item_the_reviewer_then_raises_on_other_prs():
    from test_postmortem import PAST, TRAIN

    from ospm.learning.interventions import LearnConfig, PRView
    from ospm.llm.fake import FakeLLM

    def run(t, ctx, skills=(), role_ctx=()):
        if t.truth["positive"]:
            c = t.truth["concerns"][0]
            return (1.0, {"comments": [{"path": c["path"], "line": c["line"], "body": "x"}]}) if "CHECK-EDGE" in ctx \
                else (0.0, {"comments": []})
        return 1.0, {"comments": []}

    seen = []
    proposals = iter([[{"op": "add_item", "area": "src", "index": -1, **ITEM, "area_description": "",
                        "area_paths": [], "diagnosis": "d", "source_tasks": ["p1"]}]])

    def llm(system, user, schema):
        seen.append(user)
        return {"edits": next(proposals, [])}

    learner = rubric.RubricLearner(None, run, FakeLLM(llm), LearnConfig(workers=1, batch=4), log=lambda m: None,
                                   past=PAST, trace_of=lambda *a: None)
    learner.skills = [area(items=[{**ITEM, "rule": "Unrelated", "check": "naming", "example": "rename it"}])]
    for t in TRAIN + PAST:
        learner._views[t.id] = PRView(["src/mod.py"], "x = 1")
    learner.learn(TRAIN)
    rec = learner.edits[0]
    v = rec["validation"]
    assert rec["status"] == "adopted" and (v["hits_before"], v["hits_after"]) == (0, v["trials"]) and v["n_pos"], rec
    assert not any(tid.startswith("p1") for tid in rec["validation"]["tasks"])  # all rounds of the source PR excluded
    assert [len(s.items) for s in learner.skills] == [2]
    assert "earlier: PR #3" in seen[0] or "earlier: PR #4" in seen[0]  # similar concerns of earlier PRs as evidence
    ctx = learner.serve(TRAIN[0], "rubric")[0]
    assert "## Review rubric" in ctx and "CHECK-EDGE" in ctx


def test_rubric_validation_needs_support_counts_hits_over_runs_and_remembers_rejections():
    from test_postmortem import PAST, TRAIN

    from ospm.learning.interventions import LearnConfig, PRView
    from ospm.llm.fake import FakeLLM

    def run(t, ctx, skills=(), role_ctx=(), genome=None, sample=0):
        c = t.truth["concerns"][0] if t.truth["positive"] else None
        hit = c is not None and "CHECK-EDGE" in ctx and sample == 1  # the item helps on some runs only
        return float(hit), {"comments": [{"path": c["path"], "line": c["line"], "body": "x"}] if hit else []}

    class Judge:  # past concerns apply every item except the unhelpful one
        def probabilities(self, pairs):
            return [0.0 if "Unhelpful" in item["body"] else 1.0 for _, _, item in pairs]

    def add(rule, check):
        return {"op": "add_item", "area": "src", "index": -1, **ITEM, "rule": rule, "check": check,
                "area_description": "", "area_paths": [], "diagnosis": "d", "source_tasks": []}

    seen = []
    proposals = iter([[add("Handle the edge case", "CHECK-EDGE"), add("Unhelpful naming rule", "naming")],
                      [add("Unhelpful rule about naming", "names")]])

    def llm(system, user, schema):
        seen.append(user)
        return {"edits": next(proposals, [])}

    cfg = LearnConfig(workers=1, batch=1, validate_max_tasks=6, samples=2, min_support=2, min_hit_gain=2)
    learner = rubric.RubricLearner(None, run, FakeLLM(llm), cfg, log=lambda m: None, past=PAST,
                                   trace_of=lambda *a, **k: None, judge=Judge())
    learner.skills = [area(items=[{**ITEM, "rule": "Unrelated", "check": "naming", "example": "rename it"}])]
    for t in TRAIN + PAST:
        learner._views[t.id] = PRView(["src/mod.py"], "x = 1")
    learner.learn(TRAIN, epochs=2)
    good, unhelpful, again = learner.edits
    v = good["validation"]
    assert good["status"] == "adopted" and v["samples"] == 2 and v["trials"] == 2 * v["n_pos"] >= 4
    assert (v["hits_before"], v["hits_after"]) == (0, v["n_pos"])  # raised on the second run of each PR
    assert unhelpful["validation"]["reason"].startswith("not recurring")
    assert again["validation"]["reason"] == "repeats a rejected edit"
    assert "# Edits already rejected" in seen[1] and "Unhelpful naming rule" in seen[1]
    assert [h["epoch"] for h in learner.history] == [1, 1, 2, 2]


def test_rubric_team_adds_one_specialist_per_area():
    from ospm.team.roles import seed_library

    seed = seed_library()
    team = rubric.rubric_team(seed, [area(), area(name="docs")])
    assert set(team.roles) - set(seed.roles) == {"rubric_src", "rubric_docs"} and team.id != seed.id
    src = team.roles["rubric_src"]
    assert "src/*.py" in src.description and ITEM["check"] in src.system_template
    assert team.roles["rubric_src"].contract == "review" and "rubric_src" in team.specialists()


def test_runner_samples_are_separate_episodes_with_their_own_traces(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_postmortem import TRAIN

    from ospm.learning import experiment
    from ospm.llm import UsageMeter

    dirs = []

    def run_task(repo, g, model, task, trace_dir=None, **kw):
        dirs.append(str(trace_dir))
        trace = SimpleNamespace(to_dict=lambda: {"steps": [], "n": len(dirs)})
        return SimpleNamespace(exit_status="submitted", output={"comments": []}, cost=0.01, trace=trace), 1.0, ""

    monkeypatch.setattr(experiment, "run_task", run_task)
    run = experiment.make_runner(None, None, UsageMeter(), tmp_path / "cache.jsonl", tmp_path / "traces")
    for sample in (0, 1, 1, 0):
        run(TRAIN[0], "ctx", sample=sample)
    assert len(dirs) == 2 and dirs[1].endswith("s1") and dirs[0] == str(tmp_path / "traces")  # each cached once
    keys = [json.loads(line)["key"] for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert keys[1] == keys[0] + "|s1"
    assert run.trace_of(TRAIN[0], "ctx", sample=1)["n"] == 2 and run.trace_of(TRAIN[0], "ctx")["n"] == 1
