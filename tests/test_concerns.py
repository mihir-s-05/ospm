"""Review-concern tasks on a synthetic repository: mining, chronological split, scoring, metrics."""

import os
import subprocess

import pytest

from ospm.replay.concerns import (
    COMMENT_BUDGET,
    chronological_split,
    concern_metrics,
    concern_tasks,
    drop_ai_concerns,
    is_ai_text,
    is_nit,
    score_concerns,
    themes_of,
)
from ospm.replay.store import HistoryStore
from ospm.replay.tasks import ReplayTask
from ospm.replay.workspace import ReplayRepo

PATH = "src/mod.py"
R1_AT = "2026-05-06T09:00:00Z"


def _git(cwd, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x")
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("concerns")
    src = root / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "src").mkdir()
    body = "".join(f"x{i} = {i}\n" for i in range(30))
    (src / PATH).write_text(body)
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "initial", date="2026-04-20T10:00:00Z")
    c1 = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-qb", "feature")
    (src / PATH).write_text(body + "def first(xs):\n    return xs[0]\n")
    _git(src, "commit", "-qam", "add first()", date="2026-05-04T10:00:00Z")
    f1 = _git(src, "rev-parse", "HEAD")
    (src / PATH).write_text(body + "def first(xs):\n    return xs[0] if xs else None\n")
    _git(src, "commit", "-qam", "handle empty", date="2026-05-05T10:00:00Z")
    f2 = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-q", "main")
    (src / "later.txt").write_text("later\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "future main commit", date="2026-06-01T10:00:00Z")
    d = root / "data"
    d.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(d / "repo.git")], check=True)

    s = HistoryStore(d / "history.db")
    item = dict(is_pr=1, author_association="NONE", reactions=0, labels_json="[]", in_window=1, updated_at=None,
                state="closed", state_reason=None, closed_at="2026-06-02T10:00:00Z", body="")
    s.upsert("items", [
        {**item, "number": 1, "title": "PR A", "author": "carol", "created_at": "2026-05-04T11:00:00Z"},
        {**item, "number": 2, "title": "PR B", "author": "carol", "created_at": "2026-05-04T12:00:00Z"},
        {**item, "number": 3, "title": "PR C", "author": "dependabot[bot]", "created_at": "2026-05-04T13:00:00Z"},
        {**item, "number": 4, "title": "PR D", "author": "carol", "created_at": "2026-05-04T14:00:00Z"},
        {**item, "number": 5, "title": "PR E", "author": "carol", "created_at": "2026-05-04T15:00:00Z"},
        {**item, "number": 6, "title": "PR F", "author": "carol", "created_at": "2026-05-04T16:00:00Z"},
        {**item, "number": 7, "title": "PR G", "author": "carol", "created_at": "2026-05-04T17:00:00Z"},
    ])
    s.upsert("pulls", [{"number": n, "head_sha": f1, "head_ref": "feature", "head_repo": "carol/x", "base_ref": "main",
                        "base_sha": c1, "merged_at": "2026-06-02T10:00:00Z" if n in (1, 2, 5, 6) else None,
                        "merge_commit_sha": None, "draft": 0, "have_head": 1} for n in (1, 2, 3, 4, 5, 6, 7)])

    def review(id_, number, author, assoc, state, at, sha):
        return {"id": id_, "number": number, "author": author, "author_association": assoc, "state": state,
                "body": "", "submitted_at": at, "commit_id": sha}

    s.upsert("reviews", [
        review(10, 1, "maint", "MEMBER", "CHANGES_REQUESTED", R1_AT, f1),
        review(11, 1, "maint", "MEMBER", "APPROVED", "2026-05-08T09:00:00Z", f2),
        review(20, 2, "maint", "MEMBER", "APPROVED", "2026-05-07T09:00:00Z", f1),
        review(30, 3, "maint", "MEMBER", "CHANGES_REQUESTED", "2026-05-07T10:00:00Z", f1),
        review(40, 4, "rando", "NONE", "CHANGES_REQUESTED", "2026-05-07T11:00:00Z", f1),
        # GitHub reports this approval against f2, a commit made a day *after* the approval: not reviewable then
        review(60, 6, "maint", "MEMBER", "APPROVED", "2026-05-07T12:00:00Z", f1),
        review(70, 7, "maint", "MEMBER", "CHANGES_REQUESTED", "2026-05-07T13:00:00Z", f1),
        review(50, 5, "maint", "MEMBER", "APPROVED", "2026-05-04T12:00:00Z", f2),
    ])

    def rc(id_, number, review_id, author, assoc, line, text, reply=None):
        return {"id": id_, "number": number, "review_id": review_id, "author": author, "author_association": assoc,
                "path": PATH, "line": line, "original_line": line, "side": "RIGHT", "diff_hunk": "", "body": text,
                "created_at": f"2026-05-06T08:{id_ % 60:02d}:00Z", "commit_id": f1, "in_reply_to": reply}

    s.upsert("review_comments", [
        rc(100, 1, 10, "maint", "MEMBER", 3, "nit: rename this variable"),
        rc(101, 1, 10, "maint", "MEMBER", 32, "this crashes when the list is empty"),
        rc(102, 1, 10, "carol", "NONE", 32, "good point, will fix", reply=101),
        rc(103, 1, 10, "maint", "MEMBER", 10, "LGTM"),
        rc(300, 3, 30, "maint", "MEMBER", 32, "this crashes when the list is empty"),
        rc(600, 6, 60, "maint", "MEMBER", 32, "[by Codex] **Severity: style | Confidence: High** rename this"),
        rc(601, 6, 60, "maint", "MEMBER", 5, "**Severity: bug | Confidence: Medium** unchecked index"),
        rc(700, 7, 70, "maint", "MEMBER", 5, "[by Claude] **Severity: nit | Confidence: Low** tidy"),
        rc(701, 7, 70, "maint", "MEMBER", 32, "please also cover the empty list"),
        rc(400, 4, 40, "rando", "NONE", 32, "this crashes when the list is empty"),
    ])
    s.close()
    return {"dir": d, "c1": c1, "f1": f1, "f2": f2}


@pytest.fixture(scope="module")
def tasks(data):
    return {t.id.split("@")[0]: t for t in concern_tasks(ReplayRepo(data["dir"]))}


def test_mining(data, tasks):
    # nothing for the bot PR (3), the NONE reviewer (4) or the approval reported against a later commit (5)
    # PR 6 was reviewed only with AI-written comments: neither a positive task nor a clean approval
    assert sorted(tasks) == ["concern-neg:2", "concern:1", "concern:7"]
    pos, neg = tasks["concern:1"], tasks["concern-neg:2"]
    assert pos.id == f"concern:1@{data['f1'][:8]}:10" and neg.id == f"concern-neg:2@{data['f1'][:8]}"
    assert pos.kind == "concern" and pos.pr == 1 and pos.sha == data["f1"]
    assert pos.truth["positive"] is True and neg.truth["positive"] is False and neg.truth["concerns"] == []
    cs = pos.truth["concerns"]
    assert [c["body"] for c in cs] == ["nit: rename this variable", "this crashes when the list is empty"]
    assert [c["nit"] for c in cs] == [True, False]
    assert "correctness" in cs[1]["themes"] and cs[1]["line"] == 32 and cs[1]["path"] == PATH
    assert pos.as_of == "2026-05-06T08:59:59Z" and pos.meta["base"] == data["c1"]
    assert neg.as_of == "2026-05-07T08:59:59Z"


def test_ai_text_is_dropped_from_ground_truth(tasks):
    assert "concern:6" not in tasks and "concern-neg:6" not in tasks
    assert [c["body"] for c in tasks["concern:7"].truth["concerns"]] == ["please also cover the empty list"]


def test_drop_ai_concerns_in_place():
    ai = {"path": "a", "line": 1, "body": "[by Codex] **Severity: style | Confidence: High** x", "themes": [], "nit": False}
    human = {"path": "a", "line": 2, "body": "why not use a set here?", "themes": [], "nit": False}
    mk = lambda i, cs, pos=True: ReplayTask(i, "concern", "2026-05-01T00:00:00Z", "s", "e",  # noqa: E731
                                            {"positive": pos, "concerns": cs}, split="test")
    out = drop_ai_concerns([mk("a", [ai]), mk("b", [ai, human]), mk("c", [], pos=False)])
    assert [t.id for t in out] == ["b", "c"] and out[0].truth["concerns"] == [human] and out[0].split == "test"
    assert is_ai_text("[by Claude] x") and not is_ai_text("add a test please")


def test_helpers():
    assert themes_of("this crashes when the list is empty") == ["correctness"]
    assert themes_of("hello") == ["other"]
    assert is_nit("Nit: x") and not is_nit("this is not a nit")


def _fake(i):
    return ReplayTask(f"t{i}", "concern", f"2026-05-{i + 1:02d}T00:00:00Z", "sha", "ev", {})


def test_chronological_split():
    ts = [_fake(i) for i in (7, 2, 9, 0, 5, 1, 8, 3, 6, 4)]  # shuffled
    out = chronological_split(ts)
    assert [t.as_of for t in out] == sorted(t.as_of for t in ts)
    assert [t.split for t in out] == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    assert [t.id for t in out if t.split == "test"] == ["t8", "t9"]


def _c(path=PATH, line=32):
    return {"path": path, "line": line, "body": "x"}


def test_score_positive(tasks):
    pos = tasks["concern:1"]
    good = {"comments": [_c(line=35)]}  # within 5 lines of line 32
    assert score_concerns(pos, good)[0] == 1.0
    m = concern_metrics(pos, good)
    assert m["recall"] == 1.0 and m["n_gold"] == 1 and m["matched"] == 1  # the nit is not gold
    assert m["located_precision"] == 1.0 and m["matched_themes"] == ["correctness"] and m["missed_themes"] == []

    noisy = {"comments": [_c()] + [_c(path=f"other{i}.py", line=1) for i in range(5)]}
    over = 6 - COMMENT_BUDGET
    expected = 1.0 * COMMENT_BUDGET / (COMMENT_BUDGET + over)
    assert score_concerns(pos, noisy)[0] == pytest.approx(expected) and expected < 1.0
    assert concern_metrics(pos, noisy)["located_precision"] == pytest.approx(1 / 6)

    assert score_concerns(pos, {"comments": []})[0] == 0.0
    assert score_concerns(pos, None)[0] == 0.0
    assert score_concerns(pos, {"comments": [_c(line=40)]})[0] == 0.0  # too far
    assert score_concerns(pos, {"comments": [_c(path="other.py")]})[0] == 0.0  # wrong file


def test_metrics_miss(tasks):
    pos = tasks["concern:1"]
    m = concern_metrics(pos, {"comments": [_c(line=3)]})  # lands on the nit only
    assert m["recall"] == 0.0 and m["matched"] == 0
    assert m["located_precision"] == 1.0  # a nit location still counts as located
    assert m["matched_themes"] == [] and m["missed_themes"] == ["correctness"]
    m = concern_metrics(pos, {"comments": []})
    assert m["located_precision"] is None and m["n_comments"] == 0


def test_score_negative(tasks):
    neg = tasks["concern-neg:2"]
    assert score_concerns(neg, {"comments": []})[0] == 1.0
    assert score_concerns(neg, {"comments": [_c(), _c(line=1)]})[0] == pytest.approx(1 - 2 / (COMMENT_BUDGET + 1))
    assert score_concerns(neg, {"comments": [_c()] * 10})[0] == 0.0
    assert score_concerns(neg, None)[0] == 0.0
    m = concern_metrics(neg, {"comments": [_c()]})
    assert m["recall"] is None and m["located_precision"] == 0.0 and m["n_gold"] == 0
