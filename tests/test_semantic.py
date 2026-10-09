"""Semantic concern matching (ospm.replay.semantic) with the Jev call stubbed out."""

from __future__ import annotations

from ospm.replay.semantic import Matcher, missed_concerns, semantic_metrics, semantic_score
from ospm.replay.tasks import ReplayTask


def task(positive=True):
    concerns = [{"path": "a.py", "line": 10, "body": "races on the cache", "themes": [], "nit": False},
                {"path": "b.py", "line": 5, "body": "missing test", "themes": [], "nit": False},
                {"path": "a.py", "line": 1, "body": "nit: rename", "themes": [], "nit": True}]
    return ReplayTask("concern:1@abc:2", "concern", "2026-01-01T00:00:00Z", "sha", "event",
                      {"positive": positive, "concerns": concerns if positive else []}, pr=1)


def matcher(tmp_path, answers):
    m = Matcher(tmp_path / "sem.jsonl", workers=1)
    calls = []

    def ask(path, concern, comment):
        calls.append((path, concern["body"], comment["body"]))
        return answers.get((concern["body"], comment["body"]), 0.05)

    m._ask = ask
    return m, calls


def test_only_same_file_pairs_are_judged_and_answers_are_cached(tmp_path):
    out = {"comments": [{"path": "a.py", "line": 40, "body": "concurrent writers corrupt the cache"},
                        {"path": "c.py", "line": 1, "body": "missing test"}]}
    m, calls = matcher(tmp_path, {("races on the cache", "concurrent writers corrupt the cache"): 0.93})
    assert m.concern_scores(task().truth["concerns"][:2], out) == [0.93, 0.0]  # c.py is never compared to b.py
    assert calls == [("a.py", "races on the cache", "concurrent writers corrupt the cache")]
    m2, calls2 = matcher(tmp_path, {})
    assert m2.concern_scores(task().truth["concerns"][:2], out) == [0.93, 0.0] and calls2 == []  # from disk


def test_metrics_missed_and_score_use_the_threshold_and_budget(tmp_path):
    t = task()
    out = {"comments": [{"path": "a.py", "line": 40, "body": "x"}, {"path": "b.py", "line": 7, "body": "y"}]}
    m, _ = matcher(tmp_path, {("races on the cache", "x"): 0.9, ("missing test", "y"): 0.7})
    assert semantic_metrics(m, t, out) == {"sem_recall": 0.5, "soft_recall": 0.8}  # nits excluded
    assert [c["body"] for c in missed_concerns(m, t, out)] == ["missing test"]
    assert semantic_score(m, t, out) == 0.5
    noisy = {"comments": out["comments"] + [{"path": "z.py", "line": 1, "body": str(i)} for i in range(4)]}
    assert semantic_score(m, t, noisy) == 0.5 * 3 / (3 + 3)  # 6 comments, budget 3
    clean = task(positive=False)
    assert semantic_score(m, clean, {"comments": []}) == 1.0 and semantic_metrics(m, clean, {})["sem_recall"] is None
