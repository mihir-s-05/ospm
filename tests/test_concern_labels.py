import re

from ospm.llm.fake import FakeLLM
from ospm.replay.concern_labels import TAXONOMY, apply_labels, label_concerns, main
from ospm.replay.tasks import ReplayTask, load_tasks, save_tasks


def _task(tid, concerns, positive=True, kind="concern"):
    cs = [{"path": "a.py", "line": i + 1, "body": b, "themes": ["x"], "nit": False} for i, b in enumerate(concerns)]
    return ReplayTask(tid, kind, "2024-01-01T00:00:00Z", "abc", "ev", {"positive": positive, "concerns": cs})


def _tasks():
    return [_task("concern:1", ["add a test", "typo in docstring"]),
            _task("concern:2", ["why is this needed?"]),
            _task("concern-neg:3", [], positive=False),
            _task("review:4", ["ignored"], kind="review")]


def _responder(calls):
    def r(system, user, schema):
        calls.append(user)
        labels = []
        for m in re.finditer(r"^\[([^\]]+)\] \S+\n(.*)$", user, re.MULTILINE):
            key, body = m.group(1), m.group(2)
            if "test" in body:
                labels.append({"key": key, "primary": "test_gap",
                               "secondary": ["test_gap", "docs", "design", "typing"],
                               "actionable": True, "needs_context": False})
            elif "typo" in body:
                labels.append({"key": key, "primary": "docs", "secondary": [], "actionable": True,
                               "needs_context": False})
            else:
                labels.append({"key": key, "primary": "clarification", "secondary": [], "actionable": False,
                               "needs_context": True})
        labels.append({"key": "bogus#9", "primary": "other", "secondary": [], "actionable": False,
                       "needs_context": False})
        return {"labels": labels}
    return r


def test_taxonomy():
    assert len(TAXONOMY) == 15 and list(TAXONOMY)[0] == "correctness_bug" and list(TAXONOMY)[-1] == "other"


def test_batching_and_mapping():
    calls = []
    llm = FakeLLM(_responder(calls))
    labels = label_concerns(_tasks(), llm, batch_size=2)
    assert len(calls) == 2
    assert set(labels) == {"concern:1#0", "concern:1#1", "concern:2#0"}
    a = labels["concern:1#0"]
    assert a["primary"] == "test_gap" and a["secondary"] == ["docs", "design"] and a["actionable"] is True
    assert labels["concern:1#1"]["primary"] == "docs"
    assert labels["concern:2#0"] == {"primary": "clarification", "secondary": [], "actionable": False,
                                     "needs_context": True}


def test_invalid_response_leaves_batch_unlabelled():
    n = {"i": 0}

    def r(system, user, schema):
        n["i"] += 1
        if n["i"] == 1:
            return {"labels": [{"key": "x"}]}
        return _responder([])(system, user, schema)

    labels = label_concerns(_tasks(), FakeLLM(r), batch_size=2)
    assert set(labels) == {"concern:2#0"}


def test_apply_labels_and_negatives_untouched():
    tasks = _tasks()
    apply_labels(tasks, label_concerns(tasks, FakeLLM(_responder([])), batch_size=25))
    c = tasks[0].truth["concerns"][0]
    assert c["llm_labels"]["primary"] == "test_gap" and c["themes"] == ["x"]
    assert tasks[2].truth == {"positive": False, "concerns": []}
    assert "llm_labels" not in tasks[3].truth["concerns"][0]


def test_cli(tmp_path, monkeypatch, capsys):
    save_tasks(_tasks(), tmp_path / "tasks.jsonl")
    llm = FakeLLM(_responder([]))
    monkeypatch.setattr("ospm.llm.make_llm", lambda *a, **k: llm)
    assert main(["--data", str(tmp_path)]) == 0
    assert '"labelled": 3' in capsys.readouterr().out
    assert "llm_labels" in load_tasks(tmp_path / "tasks.jsonl")[1].truth["concerns"][0]


def test_label_concerns_batch_maps_results_back(monkeypatch, tmp_path):
    import ospm.llm.batch as batch_mod
    from ospm.llm import UsageMeter
    from ospm.replay.concern_labels import label_concerns_batch

    seen = {}

    def fake_batch(model, items, schema, **kw):
        seen["n"] = len(items)
        return {cid: {"labels": [{"key": "c0", "primary": "docs", "secondary": ["docs", "design"],
                                  "actionable": True, "needs_context": False}]} if cid == "0" else "failed"
                for cid, _s, _u in items}

    monkeypatch.setattr(batch_mod, "complete_json_batch", fake_batch)
    tasks = _tasks()
    labels = label_concerns_batch(tasks, "gpt-6-luna", UsageMeter(1.0), tmp_path / "s.json", batch_size=1,
                                  log=lambda m: None)
    assert seen["n"] >= 2  # one request per chunk
    assert len(labels) == 1  # failed chunks are skipped, not guessed
    (lab,) = labels.values()
    assert lab["primary"] == "docs" and lab["secondary"] == ["design"]
