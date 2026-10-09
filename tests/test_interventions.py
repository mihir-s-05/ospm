"""Tests for ospm.learning.interventions and ospm.learning.experiment (no git, no network)."""

from __future__ import annotations

from ospm.learning import experiment
from ospm.learning.interventions import (
    Intervention,
    InterventionLearner,
    LearnConfig,
    LessonMemory,
    PRView,
    applies,
    context_block,
    propose,
    select,
)
from ospm.llm.fake import FakeLLM
from ospm.replay.tasks import ReplayTask


def iv(id="iv-1", paths=(), keywords=(), status="active", gain=0.0, trigger="when X"):
    return Intervention(id, trigger, list(paths), list(keywords), "check it", "ask it", status=status,
                        validation={"gain": gain})


# ------------------------------------------------------------------ applies / select
def test_applies_path_glob():
    v = PRView(["src/a/b.py", "README.md"], "")
    assert applies(iv(paths=["src/*.py"]), v)
    assert applies(iv(paths=["*.md"]), v)
    assert not applies(iv(paths=["tests/*"]), v)
    assert applies(iv(), PRView([], ""))  # no scope -> applies


def test_applies_keyword_identifier_boundaries():
    i = iv(keywords=["compile"])
    assert not applies(i, PRView(["a.py"], "x = recompile_all()"))
    assert not applies(i, PRView(["a.py"], "compiled = 1"))
    assert not applies(i, PRView(["a.py"], "_compile"))
    assert applies(i, PRView(["a.py"], "m = torch.compile(model)"))
    assert applies(iv(keywords=["torch.compile"]), PRView(["a.py"], "torch.compile(m)"))
    assert not applies(iv(keywords=["torch.compile"]), PRView(["a.py"], "torchXcompile(m)"))  # '.' is escaped
    assert applies(iv(keywords=["nope", "compile"]), PRView(["a.py"], "compile"))


def test_select_status_rank_and_k():
    v = PRView(["a.py"], "")
    ivs = [iv("a", status="candidate", gain=9), iv("b", status="rejected", gain=9), iv("c", gain=0.1),
           iv("d", gain=0.5), iv("e", gain=0.5), iv("f", paths=["zzz/*"], gain=5)]
    assert [x.id for x in select(ivs, v)] == ["d", "e", "c"]
    assert [x.id for x in select(ivs, v, k=1)] == ["d"]


# ------------------------------------------------------------------ context_block
def test_context_block():
    assert context_block([]) == ""
    a, b = iv("iv-a", paths=["src/*"], trigger="uses locks"), iv("iv-b", trigger="adds API")
    out = context_block([a, b])
    assert "[iv-a] WHEN uses locks (files: src/*)" in out and "[iv-b] WHEN adds API:" in out
    assert "CHECK check it" in out and "ASK: ask it" in out


# ------------------------------------------------------------------ propose
def _spec(**kw):
    d = {"trigger": "t", "paths": ["src/*"], "keywords": ["kw"], "check": "c", "action": "a", "stage": "review",
         "source_tasks": ["t1"]}
    d.update(kw)
    return d


MISSES = [{"task": "t1", "files": ["src/a.py"], "concern": {"path": "src/a.py", "line": 3, "body": "b"},
           "agent_comments": []}]


def test_propose_basic_and_deterministic_ids():
    llm = FakeLLM(lambda s, u, sc: {"interventions": [_spec(), _spec(trigger="other")]})
    out = propose(llm, MISSES, [])
    assert len(out) == 2 and all(isinstance(x, Intervention) for x in out)
    assert out[0].id.startswith("iv-") and out[0].id != out[1].id and out[0].status == "candidate"
    again = propose(FakeLLM(lambda s, u, sc: {"interventions": [_spec()]}), MISSES, [])
    assert again[0].id == out[0].id
    # order of paths/keywords does not matter for id
    a = propose(FakeLLM(lambda s, u, sc: {"interventions": [_spec(paths=["a", "b"], keywords=["x", "y"])]}), MISSES, [])
    b = propose(FakeLLM(lambda s, u, sc: {"interventions": [_spec(paths=["b", "a"], keywords=["y", "x"])]}), MISSES, [])
    assert a[0].id == b[0].id


def test_propose_empty_misses_no_call():
    llm = FakeLLM(lambda s, u, sc: {"interventions": [_spec()]})
    assert propose(llm, [], []) == [] and llm.requests == []


def test_propose_truncates_and_cleans():
    spec = _spec(trigger="t" * 500, check="c" * 900, action="a" * 900,
                 paths=["", "p1"] + [f"p{i}" for i in range(2, 20)],
                 keywords=["", "  ", "k1"] + [f"k{i}" for i in range(2, 20)],
                 source_tasks=[f"s{i}" for i in range(20)])
    (x,) = propose(FakeLLM(lambda s, u, sc: {"interventions": [spec]}), MISSES, [])
    assert len(x.trigger) == 300 and len(x.check) == 400 and len(x.action) == 400
    assert "" not in x.paths and len(x.paths) == 8 and x.paths[0] == "p1"
    assert "" not in x.keywords and "  " not in x.keywords and len(x.keywords) == 8 and x.keywords[0] == "k1"
    assert len(x.provenance) == 8


def test_propose_max_new_and_duplicates():
    specs = [_spec(trigger=f"t{i}") for i in range(5)]
    llm = FakeLLM(lambda s, u, sc: {"interventions": specs})
    out = propose(llm, MISSES, [], max_new=3)
    assert len(out) == 3
    again = propose(llm, MISSES, out, max_new=5)  # the first three already exist
    assert [x.trigger for x in again] == ["t3", "t4"]


def test_propose_swallows_schema_failure():
    llm = FakeLLM(lambda s, u, sc: {"interventions": [{"trigger": "only this"}]})
    assert propose(llm, MISSES, []) == []
    llm = FakeLLM(lambda s, u, sc: {"wrong": 1})
    assert propose(llm, MISSES, []) == []
    llm = FakeLLM(lambda s, u, sc: {"interventions": [_spec(stage="bogus")]})
    assert propose(llm, MISSES, []) == []


# ------------------------------------------------------------------ LessonMemory
def test_lesson_memory():
    m = LessonMemory()
    assert m.context(PRView(["a.py"], "anything")) == ""
    m.add("t1", {"path": "src/scheduler/queue.py",
                 "body": "please add a regression test for the scheduler queue overflow"})
    m.add("t2", {"path": "docs/index.rst", "body": "changelog entry missing for documentation build"})
    ctx = m.context(PRView(["src/scheduler/queue.py"], "def push(queue): scheduler queue overflow handling"))
    assert "scheduler queue overflow" in ctx and ctx.startswith("\n\nLessons")
    assert m.context(PRView(["zzz.bin"], "qqqq wwww")) == ""


# ------------------------------------------------------------------ InterventionLearner end to end
def mk_task(tid, positive, as_of, path="src/mod.py", line=10):
    concerns = [{"path": path, "line": line, "body": "please handle the edge case", "themes": ["correctness"],
                 "nit": False}] if positive else []
    return ReplayTask(tid, "concern", as_of, "sha" + tid, "event", {"positive": positive, "concerns": concerns},
                      split="train", meta={"base": "base"})


TRIGGER = "TRIGGERX the change touches src"


def make_learner(paths, neg_comment_under_ctx=False, trigger=TRIGGER, source_tasks=("t1",), max_fire=1.0):
    tasks = [mk_task("p1", True, "2024-01-01T00:00:00Z"), mk_task("n1", False, "2024-01-02T00:00:00Z"),
             mk_task("p2", True, "2024-01-03T00:00:00Z")]
    calls = []

    def run(task, ctx):
        calls.append((task.id, ctx))
        if task.truth["positive"]:
            if "TRIGGERX" in ctx:
                c = task.truth["concerns"][0]
                return 1.0, {"comments": [{"path": c["path"], "line": c["line"], "body": "x"}]}
            return 0.0, {"comments": []}
        if neg_comment_under_ctx and ctx:
            return 0.75, {"comments": [{"path": "src/mod.py", "line": 1, "body": "x"}]}
        return 1.0, {"comments": []}

    spec = _spec(trigger=trigger, paths=paths, keywords=[], source_tasks=list(source_tasks))
    llm = FakeLLM(lambda s, u, sc: {"interventions": [spec]})
    # every fixture PR touches src/mod.py, so the broadness gate is disabled unless a test sets it
    learner = InterventionLearner(None, run, llm, LearnConfig(max_negative_fire_rate=max_fire, workers=1),
                                  log=lambda m: None)
    for t in tasks:
        learner._views[t.id] = PRView(["src/mod.py"], "x = 1")  # no git
    return learner, tasks, calls, llm


def test_learner_activates_candidate_with_gain():
    learner, tasks, calls, llm = make_learner(["src/*.py"])
    res = learner.learn(tasks[::-1])  # order is sorted internally
    assert len(learner.interventions) == 1
    c = learner.interventions[0]
    assert c.status == "active", c.validation
    assert c.validation["gain"] > 0 and c.validation["fp_increase"] == 0 and c.validation["n_pos"] == 2
    assert learner.active() == [c] and res["interventions"][0]["status"] == "active"
    assert "p1" in llm.requests[0]["user"] and "p2" in llm.requests[0]["user"]
    assert [h["task"] for h in res["history"]] == ["p1", "n1", "p2"]
    # memory baseline got the training concerns (one per positive task), none for negatives
    assert learner.memory.n == 2
    assert "edge case" in learner.memory.context(PRView(["src/mod.py"], "edge case"))
    # the active intervention is served only for the 'interventions' system
    assert TRIGGER in learner.serve_context(tasks[0], "interventions")
    assert learner.serve_context(tasks[0], "none") == ""
    assert "Lessons" in learner.serve_context(tasks[0], "memory")


def test_learner_rejects_on_false_positive_increase():
    learner, tasks, _, _ = make_learner(["src/*.py"], neg_comment_under_ctx=True)
    learner.learn(tasks)
    c = learner.interventions[0]
    assert c.status == "rejected"
    assert c.validation["fp_increase"] > 0 and c.validation["gain"] > 0
    assert learner.active() == []


def test_learner_rejects_when_trigger_matches_no_past_positive():
    learner, tasks, _, _ = make_learner(["docs/*"])
    learner.learn(tasks)
    c = learner.interventions[0]
    assert c.status == "rejected" and "no independent support" in c.validation["reason"]
    assert learner.memory.n == 2  # memory still stored the concerns


def test_learner_rejects_without_gain():
    learner, tasks, _, _ = make_learner(["src/*.py"], trigger="no marker here")
    learner.learn(tasks)
    c = learner.interventions[0]
    assert c.status == "rejected" and c.validation["gain"] == 0


def test_learner_never_validates_on_its_source_prs():
    # induced from p1 only: validation may use p2 (independent) but never p1
    learner, tasks, calls, _ = make_learner(["src/*.py"], source_tasks=("p1",))
    learner.learn(tasks)
    c = learner.interventions[0]
    assert "p1" not in c.validation["tasks"] and c.validation["n_pos"] == 1 and c.status == "active"
    # induced from both positives: no independent positive remains -> rejected
    learner, tasks, _, _ = make_learner(["src/*.py"], source_tasks=("p1", "p2"))
    learner.learn(tasks)
    c = learner.interventions[0]
    assert c.status == "rejected" and "no independent support" in c.validation["reason"]


def test_learner_rejects_overbroad_triggers():
    learner, tasks, _, _ = make_learner(["src/*.py"], max_fire=0.5)  # fires on 100% of clean past PRs
    learner.learn(tasks)
    c = learner.interventions[0]
    assert c.status == "rejected" and c.validation["reason"] == "trigger too broad"
    assert c.validation["negative_fire_rate"] == 1.0


# ------------------------------------------------------------------ experiment
def test_pick_balanced_chronological():
    ts = [mk_task(f"p{i}", True, f"2024-01-{i + 1:02d}T00:00:00Z") for i in range(6)]
    ts += [mk_task(f"n{i}", False, f"2024-02-{i + 1:02d}T00:00:00Z") for i in range(6)]
    old = experiment.pick(ts, 5, newest=False)
    assert [t.id for t in old] == ["p0", "p1", "p2", "n0", "n1"]  # 3 pos + 2 neg, sorted by time
    new = experiment.pick(ts, 4, newest=True)
    assert [t.id for t in new] == ["p4", "p5", "n4", "n5"]
    assert [t.as_of for t in new] == sorted(t.as_of for t in new)
    assert [t.id for t in experiment.pick(ts[:2], 10, newest=False)] == ["p0", "p1"]  # fewer than asked


def test_summarise():
    rows = [
        {"positive": True, "score": 1.0, "recall": 1.0, "n_comments": 2},
        {"positive": True, "score": 0.0, "recall": 0.0, "n_comments": 0},
        {"positive": True, "score": 0.0, "recall": None, "n_comments": 4},
        {"positive": False, "score": 1.0, "recall": None, "n_comments": 0},
        {"positive": False, "score": 0.5, "recall": None, "n_comments": 1},
        {"positive": False, "score": 1.0, "recall": None, "n_comments": 0},
        {"positive": False, "score": 0.0, "recall": None, "n_comments": 3},
    ]
    s = experiment.summarise(rows)
    assert s["n"] == 7 and s["pos_n"] == 3 and s["neg_n"] == 4
    assert s["mean_score"] == round(3.5 / 7, 4)
    assert s["pos_recall"] == round(1 / 3, 4)
    assert s["pos_mean_comments"] == 2.0
    assert s["neg_silent_rate"] == 0.5 and s["neg_mean_comments"] == 1.0
    empty = experiment.summarise([])
    assert empty["n"] == 0 and empty["mean_score"] is None and empty["neg_silent_rate"] is None


def test_runner_shares_one_episode_for_identical_concurrent_requests(monkeypatch, tmp_path):
    import threading
    import time
    from types import SimpleNamespace

    import ospm.learning.experiment as exp
    from ospm.llm import UsageMeter

    calls = []

    def fake_run_task(repo, genome, model, task, **kw):
        calls.append(kw["extra_context"])
        time.sleep(0.05)  # widen the race window
        return SimpleNamespace(output={}, exit_status="Submitted", cost=0.01, trace=SimpleNamespace(to_dict=dict)), float(len(calls)), ""

    monkeypatch.setattr(exp, "run_task", fake_run_task)
    monkeypatch.setattr(exp, "task_output", lambda task, out, entry: {})
    run = exp.make_runner(None, None, UsageMeter(1.0), tmp_path / "cache.jsonl", tmp_path / "traces")
    task = SimpleNamespace(id="concern:1@abc:2")
    results = []
    threads = [threading.Thread(target=lambda: results.append(run(task, ""))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len({r[0] for r in results}) == 1
    run(task, "other context")
    assert len(calls) == 2
    # the on-disk cache is reloaded by a new runner
    run2 = exp.make_runner(None, None, UsageMeter(1.0), tmp_path / "cache.jsonl", tmp_path / "traces")
    assert run2(task, "") == results[0] and len(calls) == 2


# ------------------------------------------------------------------ provenance and validation pool
def test_propose_resolves_source_ids_without_prefix():
    misses = [{**MISSES[0], "task": "concern:42@abcdef12:7"}, {**MISSES[0], "task": "concern:42@abcdef12:8"},
              {**MISSES[0], "task": "concern:9@00000000:1"}]
    spec = _spec(source_tasks=["42@abcdef12:7", "42@abcdef12", "unknown"])
    (x,) = propose(FakeLLM(lambda s, u, sc: {"interventions": [spec]}), misses, [])
    assert x.provenance == ["concern:42@abcdef12:7", "concern:42@abcdef12:8", "unknown"]


def test_pr_key():
    from ospm.learning.interventions import pr_key
    assert pr_key("concern:42@abcdef12:7") == 42 == pr_key("42@abcdef12") == pr_key("concern-neg:42@abcdef12")
    assert pr_key("p1") == "p1" and pr_key("anything", pr=5) == 5


def _pr_task(tid, pr, positive, as_of):
    t = mk_task(tid, positive, as_of)
    t.pr = pr
    return t


def _pool_learner(train, past, gain_on=("pos",), max_fire=0.5):
    calls = []

    def run(task, ctx):
        calls.append((task.id, bool(ctx)))
        if task.truth["positive"] and ctx and "pos" in gain_on:
            c = task.truth["concerns"][0]
            return 1.0, {"comments": [{"path": c["path"], "line": c["line"], "body": "x"}]}
        if not task.truth["positive"]:
            if ctx and "neg" in gain_on:
                return 1.0, {"comments": []}  # the context merely made the agent quieter on a clean PR
            return (0.75, {"comments": [{"path": "a", "line": 1}]}) if "neg" in gain_on else (1.0, {"comments": []})
        return 0.0, {"comments": []}

    spec = _spec(trigger=TRIGGER, paths=["src/*.py"], keywords=[], source_tasks=["7@aaaa:1"])
    learner = InterventionLearner(None, run, FakeLLM(lambda s, u, sc: {"interventions": [spec]}),
                                  LearnConfig(workers=1, max_negative_fire_rate=max_fire),
                                  log=lambda m: None, past=past)
    for t in train + past:
        learner._views[t.id] = PRView(["src/mod.py"], "x = 1")
    return learner, calls


def test_validation_excludes_every_round_of_the_source_pr_and_uses_one_positive_per_pr():
    train = [_pr_task("concern:7@aaaa:1", 7, True, "2024-03-01T00:00:00Z")]
    past = [_pr_task("concern:7@aaaa:2", 7, True, "2024-02-01T00:00:00Z"),  # same PR, other round
            _pr_task("concern:8@bbbb:1", 8, True, "2024-02-02T00:00:00Z"),
            _pr_task("concern:8@bbbb:2", 8, True, "2024-02-03T00:00:00Z"),  # same PR as above
            _pr_task("concern:9@cccc:1", 9, True, "2024-02-04T00:00:00Z"),
            _pr_task("concern:10@dddd:1", 10, True, "2024-04-01T00:00:00Z")]  # after the step: not yet known
    learner, _ = _pool_learner(train, past)
    learner.learn(train)
    c = learner.interventions[0]
    assert c.status == "active", c.validation
    assert c.validation["tasks"] == ["concern:9@cccc:1", "concern:8@bbbb:2"]


def test_adoption_needs_gain_on_positives_not_just_quieter_negatives():
    train = [_pr_task("concern:7@aaaa:1", 7, True, "2024-03-01T00:00:00Z")]
    past = [_pr_task("concern:8@bbbb:1", 8, True, "2024-02-02T00:00:00Z"),
            _pr_task("concern-neg:11@eeee", 11, False, "2024-02-05T00:00:00Z")]
    learner, _ = _pool_learner(train, past, gain_on=("neg",), max_fire=1.0)
    learner.learn(train)
    c = learner.interventions[0]
    assert c.validation["gain"] > 0 and c.validation["pos_gain"] == 0 and c.status == "rejected"
