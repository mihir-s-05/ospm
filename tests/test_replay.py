"""Replay environment on a synthetic repository: time-travel guarantees, task mining, scoring, mirror paging."""

import io
import json
import os
import subprocess
import sys

import pytest

from ospm.replay.store import HistoryStore, trim_log
from ospm.replay.tasks import (
    build_tasks,
    ci_tasks,
    review_tasks,
    score_ci,
    score_review,
    score_triage,
    task_output,
    triage_tasks,
)
from ospm.replay.workspace import ReplayRepo


def _git(cwd, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x")
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("replay")
    src = root / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "src" / "_pytest").mkdir(parents=True)
    (src / "src" / "_pytest" / "fixtures.py").write_text("def old():\n    return 1\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "initial", date="2026-04-20T10:00:00Z")
    c1 = _git(src, "rev-parse", "HEAD")
    # a PR branch, reviewed on 2026-05-06, merged later
    _git(src, "checkout", "-qb", "feature")
    (src / "src" / "_pytest" / "fixtures.py").write_text("def old():\n    return 1\n\ndef new():\n    return eval('2')\n")
    _git(src, "commit", "-qam", "add new()", date="2026-05-04T10:00:00Z")
    pr_sha = _git(src, "rev-parse", "HEAD")
    _git(src, "checkout", "-q", "main")
    (src / "changelog").mkdir()
    (src / "changelog" / "10.bugfix.rst").write_text("Fixed the thing.\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "fix #10 (future)", date="2026-06-01T10:00:00Z")
    _git(src, "merge", "-q", "--no-ff", "feature", "-m", "merge feature", date="2026-06-02T10:00:00Z")
    future = _git(src, "rev-parse", "HEAD")
    d = root / "data"
    d.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(d / "repo.git")], check=True)
    _git(d / "repo.git", "update-ref", "refs/pull/12/head", pr_sha)

    s = HistoryStore(d / "history.db")
    item = dict(is_pr=0, author_association="NONE", reactions=0, labels_json="[]", in_window=1, updated_at=None)
    s.upsert("items", [
        {**item, "number": 10, "title": "fixture crash", "body": "crashes on teardown", "author": "alice",
         "state": "closed", "state_reason": "completed", "created_at": "2026-05-01T09:00:00Z",
         "closed_at": "2026-06-01T10:00:00Z"},
        {**item, "number": 11, "title": "fixture teardown crash again", "body": "same crash", "author": "bob",
         "state": "closed", "state_reason": "duplicate", "created_at": "2026-05-03T09:00:00Z",
         "closed_at": "2026-05-03T12:00:00Z"},
        {**item, "number": 12, "is_pr": 1, "title": "add new()", "body": "adds new", "author": "carol",
         "state": "closed", "state_reason": None, "created_at": "2026-05-04T11:00:00Z",
         "closed_at": "2026-06-02T10:00:00Z"},
    ])
    s.upsert("issue_events", [
        {"id": 1, "number": 10, "event": "labeled", "actor": "maint", "label": "type: bug",
         "created_at": "2026-05-02T09:00:00Z", "commit_id": None, "extra_json": "{}"},
        {"id": 2, "number": 10, "event": "labeled", "actor": "maint", "label": "topic: fixtures",
         "created_at": "2026-05-02T09:01:00Z", "commit_id": None, "extra_json": "{}"},
        {"id": 3, "number": 11, "event": "closed", "actor": "maint", "label": None,
         "created_at": "2026-05-03T12:00:00Z", "commit_id": None, "extra_json": json.dumps({"state_reason": "duplicate"})},
    ])
    s.upsert("comments", [
        {"id": 1, "number": 10, "author": "maint", "author_association": "MEMBER",
         "created_at": "2026-05-05T09:00:00Z", "body": "FUTURE-COMMENT: confirmed, fixing"},
        {"id": 2, "number": 11, "author": "maint", "author_association": "MEMBER",
         "created_at": "2026-05-03T11:59:00Z", "body": "Duplicate of #10"},
    ])
    s.upsert("pulls", [{"number": 12, "head_sha": pr_sha, "head_ref": "feature", "head_repo": "carol/x",
                        "base_ref": "main", "base_sha": c1, "merged_at": "2026-06-02T10:00:00Z",
                        "merge_commit_sha": future, "draft": 0, "have_head": 1}])
    s.upsert("reviews", [{"id": 7, "number": 12, "author": "maint", "author_association": "MEMBER",
                          "state": "CHANGES_REQUESTED", "body": "please avoid eval",
                          "submitted_at": "2026-05-06T09:00:00Z", "commit_id": pr_sha}])
    s.upsert("review_comments", [{"id": 70, "number": 12, "review_id": 7, "author": "maint",
                                  "author_association": "MEMBER", "path": "src/_pytest/fixtures.py", "line": 5,
                                  "original_line": 5, "side": "RIGHT", "diff_hunk": "", "body": "eval is unsafe",
                                  # drafted an hour before the review was submitted (pending review)
                                  "created_at": "2026-05-06T08:00:00Z", "commit_id": pr_sha, "in_reply_to": None}])
    s.upsert("runs", [
        {"id": 100, "attempt": 2, "name": "test", "event": "pull_request", "head_sha": pr_sha, "head_branch": "feature",
         "pr_numbers_json": "[12]", "status": "completed", "conclusion": "success",
         "created_at": "2026-05-04T12:00:00Z", "updated_at": None, "html_url": ""}])
    s.upsert("jobs", [
        {"id": 1000, "run_id": 100, "attempt": 1, "name": "test (py312)", "conclusion": "failure",
         "started_at": "2026-05-04T12:00:00Z", "completed_at": "2026-05-04T12:05:00Z",
         "steps_json": json.dumps([{"name": "Test", "conclusion": "failure"}])},
        {"id": 1001, "run_id": 100, "attempt": 2, "name": "test (py312)", "conclusion": "success",
         "started_at": "2026-05-04T13:00:00Z", "completed_at": "2026-05-04T13:05:00Z", "steps_json": "[]"}])
    s.put_log(1000, "2026-05-04T12:04:00Z FAILED testing/test_a.py::test_x - AssertionError\nError: exit code 1")
    s.set_meta("repo", "demo/demo")
    s.close()
    return {"dir": d, "pr_sha": pr_sha, "future": future, "c1": c1}


def test_store_as_of_hides_the_future(data):
    s = HistoryStore(data["dir"] / "history.db", readonly=True)
    it = s.item(10, "2026-05-01T12:00:00Z")
    assert it["state"] == "open" and it["labels"] == [] and "labels_json" not in it
    assert s.item(10, "2026-05-02T10:00:00Z")["labels"] == ["type: bug", "topic: fixtures"]
    assert s.comments(10, "2026-05-04T00:00:00Z") == []
    assert s.item(11, "2026-05-02T00:00:00Z") is None  # did not exist yet
    assert [r["number"] for r in s.search_items("fixture teardown crash", "2026-05-02T00:00:00Z")] == [10]
    assert s.job_log(1000).startswith("2026-05-04") and s.has_log(1000)
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        s.upsert("items", [{"number": 99}])  # read-only
    s.close()


def _hist(data, *args, as_of, extra=None):
    env = dict(os.environ, OSPM_HIST_DB=str(data["dir"] / "history.db"), OSPM_HIST_ASOF=as_of,
               OSPM_HIST_GIT=str(data["dir"] / "repo.git"), **(extra or {}))
    return subprocess.run([sys.executable, "-m", "ospm.replay.hist", *args], env=env, capture_output=True,
                          text=True, encoding="utf-8").stdout


def test_hist_cli_respects_replay_time(data):
    early = _hist(data, "issue", "10", as_of="2026-05-01T12:00:00Z")
    assert "fixture crash" in early and "FUTURE-COMMENT" not in early and "state: open" in early
    assert "FUTURE-COMMENT" in _hist(data, "issue", "10", as_of="2026-05-06T00:00:00Z")
    pr = _hist(data, "pr", "12", "--diff", as_of="2026-05-05T00:00:00Z",
               extra={"OSPM_HIST_PR": "12", "OSPM_HIST_PR_SHA": data["pr_sha"]})
    assert "eval('2')" in pr and "please avoid eval" not in pr  # the review is in the future
    draft_time = _hist(data, "pr", "12", as_of="2026-05-06T08:30:00Z",
                       extra={"OSPM_HIST_PR": "12", "OSPM_HIST_PR_SHA": data["pr_sha"]})
    assert "eval is unsafe" not in draft_time  # pending-review drafts stay hidden until the review is submitted
    assert "eval is unsafe" in _hist(data, "pr", "12", as_of="2026-05-06T09:30:00Z")
    # without the pin, an unmerged PR's latest head is not shown (it may postdate the replay time)
    assert "diff not available" in _hist(data, "pr", "12", as_of="2026-05-05T00:00:00Z")
    assert "fix #10" not in _hist(data, "commits", as_of="2026-05-05T00:00:00Z")
    ci = _hist(data, "ci", "12", as_of="2026-05-04T12:10:00Z",
               extra={"OSPM_HIST_PR": "12", "OSPM_HIST_PR_SHA": data["pr_sha"]})
    assert "job 1000 test (py312): FAILED" in ci and "hist log 1000" in ci
    assert "1001" not in ci and "success" not in ci  # the passing re-run happened later
    later = _hist(data, "ci", "12", as_of="2026-05-04T14:00:00Z",
                  extra={"OSPM_HIST_PR": "12", "OSPM_HIST_PR_SHA": data["pr_sha"]})
    assert "attempt 2" in later and "job 1001 test (py312): success" in later
    assert "not available" in _hist(data, "log", "1000", as_of="2026-05-04T12:01:00Z")  # job still running
    assert "FAILED testing/test_a.py::test_x" in _hist(data, "log", "1000", "--grep", "FAILED", as_of="2026-05-05T00:00:00Z")


def test_hist_reviews_respects_time_paths_and_authors(tmp_path, monkeypatch, capsys):
    from ospm.replay import hist

    db = tmp_path / "history.db"
    s = HistoryStore(db)
    item = dict(is_pr=1, title="t", body="", author="alice", author_association="NONE", state="open", state_reason=None,
                reactions=0, labels_json="[]", in_window=1, updated_at=None, created_at="2026-05-01T00:00:00Z",
                closed_at=None)
    s.upsert("items", [{**item, "number": 20}])
    s.upsert("reviews", [
        {"id": 1, "number": 20, "author": "bob", "author_association": "MEMBER", "state": "COMMENTED", "body": "",
         "submitted_at": "2026-05-10T10:00:00Z", "commit_id": "x"},
        {"id": 2, "number": 20, "author": "carol", "author_association": "MEMBER", "state": "COMMENTED", "body": "",
         "submitted_at": "2026-05-20T10:00:00Z", "commit_id": "x"}])

    def rc(i, author, assoc, path, created, body, review=None, reply=None, line=5, original=5):
        return {"id": i, "number": 20, "review_id": review, "author": author, "author_association": assoc, "path": path,
                "line": line, "original_line": original, "side": "RIGHT", "diff_hunk": "", "body": body,
                "created_at": created, "commit_id": "x", "in_reply_to": reply}

    s.upsert("review_comments", [
        rc(1, "bob", "MEMBER", "src/a/x.py", "2026-05-10T09:00:00Z", "use a lock", review=1),
        rc(2, "bob", "MEMBER", "src/a/y.py", "2026-05-11T09:00:00Z", "add a test", line=None, original=7),
        rc(3, "carol", "MEMBER", "src/a/x.py", "2026-05-12T09:00:00Z", "carol early", review=2),  # review submitted later
        rc(4, "dave", "MEMBER", "src/a/x.py", "2026-05-25T09:00:00Z", "dave late"),  # created after as_of
        rc(5, "pytorch-bot[bot]", "MEMBER", "src/a/x.py", "2026-05-13T09:00:00Z", "bot says"),
        rc(6, "alice", "OWNER", "src/a/x.py", "2026-05-14T09:00:00Z", "own note"),  # the PR author
        rc(7, "bob", "MEMBER", "src/b/z.py", "2026-05-15T09:00:00Z", "zed review"),
        rc(8, "bob", "MEMBER", "docs/index.rst", "2026-05-16T09:00:00Z", "docs review"),
        rc(9, "bob", "MEMBER", "src/a/x.py", "2026-05-11T12:00:00Z", "reply thread", reply=1),
        rc(11, "bob", "MEMBER", "src/a/x.py", "2026-05-17T09:00:00Z", "rename it")])
    s.close()
    monkeypatch.setenv("OSPM_HIST_DB", str(db))
    monkeypatch.setenv("OSPM_HIST_ASOF", "2026-05-18T00:00:00Z")

    def reviews(*args):
        assert hist.main(["reviews", *args]) == 0
        return capsys.readouterr().out

    out = reviews("--path", "src/a/x.py")
    assert "2 earlier maintainer review comments on src/a/x.py" in out
    assert "[2026-05-17] #20 @bob on src/a/x.py:5" in out
    assert out.index("rename it") < out.index("use a lock")
    for hidden in ("carol early", "dave late", "bot says", "own note", "reply thread", "zed review"):
        assert hidden not in out
    out = reviews("--path", "src/a/", "-n", "2")
    assert "rename it" in out and "add a test" in out and "use a lock" not in out
    assert "src/a/y.py:7" in out and "docs review" not in out and "zed review" not in out
    assert "zed review" in reviews("--path", "src/*/z.py")
    assert "docs review" in reviews("--path", "docs/", "src/nothing.py")
    assert "no earlier maintainer review comments on src/a/" in reviews("--path", "src/a/", "--author", "alice")
    assert "ren\n...[6 more characters]" in reviews("--path", "src/a/x.py", "--max-chars", "3")


def test_workspace_cannot_see_future_commits(data):
    repo = ReplayRepo(data["dir"])
    with repo.worktree(data["pr_sha"], "2026-05-05T00:00:00Z") as ws:
        assert "eval('2')" in (ws / "src" / "_pytest" / "fixtures.py").read_text()
        log = subprocess.run(["git", "log", "--all", "--format=%s"], cwd=ws, capture_output=True, text=True).stdout
        assert "fix #10 (future)" not in log and "merge feature" not in log and "add new()" in log
        main = subprocess.run(["git", "rev-parse", "main"], cwd=ws, capture_output=True, text=True).stdout.strip()
        assert main == data["c1"]
        assert (ws / "src" / "_pytest" / "_version.py").exists()
        env = repo.environment(ws, "2026-05-05T00:00:00Z", pr=12, pr_sha=data["pr_sha"], base=data["c1"])
        assert "add new()" in env.execute("hist pr 12").output
        env.cleanup()
    assert not ws.exists()
    assert repo.base_at(data["pr_sha"], "2026-05-05T00:00:00Z") == data["c1"]


def test_workspace_keeps_no_refs_but_main_even_with_tags_and_symrefs(data):
    """Every mirror ref (branches, tags, PR heads, symrefs like origin/HEAD) may point into the future; all are
    deleted in one transaction, which a symref or a Windows line ending would otherwise abort as a whole."""
    git_dir = str(ReplayRepo(data["dir"]).git_dir)
    future = subprocess.run(["git", "--git-dir", git_dir, "rev-parse", "refs/heads/main"], capture_output=True,
                            text=True).stdout.strip()
    for args in (["tag", "v9-future", future], ["update-ref", "refs/remotes/origin/main", future],
                 ["symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"]):
        subprocess.run(["git", "--git-dir", git_dir, *args], check=True, capture_output=True)
    try:
        with ReplayRepo(data["dir"]).worktree(data["pr_sha"], "2026-05-05T00:00:00Z") as ws:
            refs = subprocess.run(["git", "for-each-ref", "--format=%(refname)"], cwd=ws, capture_output=True,
                                  text=True).stdout.split()
            assert refs == ["refs/heads/main"]
    finally:
        for args in (["symbolic-ref", "--delete", "refs/remotes/origin/HEAD"],
                     ["update-ref", "-d", "refs/remotes/origin/main"], ["tag", "-d", "v9-future"]):
            subprocess.run(["git", "--git-dir", git_dir, *args], capture_output=True)


def test_task_mining_and_ground_truth(data):
    repo = ReplayRepo(data["dir"])
    tri = {t.id: t for t in triage_tasks(repo)}
    t10, t11 = tri["triage:10"], tri["triage:11"]
    assert t10.truth["kind"] == "bug" and t10.truth["decision"] == "accept"
    assert t10.truth["components"] == ["topic: fixtures"] and t10.truth["fragment"] == "bug"
    assert t10.as_of == "2026-05-01T09:00:00Z" and t10.sha == data["c1"]
    assert t11.truth["decision"] == "duplicate" and t11.truth["duplicate_of"] == 10
    (rv,) = review_tasks(repo)
    assert rv.truth["verdict"] == "request_changes" and rv.truth["comments"][0]["line"] == 5
    assert rv.as_of < "2026-05-06T09:00:00Z" and rv.meta["base"] == data["c1"]
    (ci,) = ci_tasks(repo)
    assert ci.truth["flaky"] is True and ci.truth["failing_tests"] == ["testing/test_a.py::test_x"]
    assert ci.truth["category"] == "test_failure" and ci.pr == 12
    assert len(build_tasks(repo, kinds=("triage", "review", "ci"))) == 4
    assert len(build_tasks(repo)) == 5  # + one review-concern task (PR 12 has a maintainer inline comment)


def test_scoring(data):
    repo = ReplayRepo(data["dir"])
    tasks = {t.id: t for t in build_tasks(repo)}
    t10 = tasks["triage:10"]
    perfect = {"kind": "bug", "decision": "accept", "components": ["topic: fixtures"]}
    assert score_triage(t10, perfect)[0] == 1.0
    s, fb = score_triage(t10, {"kind": "feature", "decision": "accept", "components": []})
    assert 0 < s < 1 and "kind" in fb and "maintainers" in fb
    assert score_triage(tasks["triage:11"], {"kind": "bug", "decision": "duplicate", "duplicate_of": 99})[0] < 1
    rv = next(t for t in tasks.values() if t.kind == "review")
    good = {"verdict": "request_changes", "comments": [{"path": "src/_pytest/fixtures.py", "line": 6, "body": "x"}]}
    assert score_review(rv, good)[0] == pytest.approx((1 + 1 + 0.5) / 2.5)
    assert score_review(rv, {"verdict": "approve", "comments": []})[0] == 0.0
    ci = next(t for t in tasks.values() if t.kind == "ci")
    assert score_ci(ci, {"flaky": True, "category": "test_failure",
                         "failing_tests": ["testing/test_a.py::test_x[param]"]})[0] == 1.0
    assert score_ci(ci, None)[0] == 0.0
    orch = {"event_kind": "ci_failure", "ci": [{"job_id": 5, "flaky": False}, {"job_id": 1000, "flaky": True}]}
    assert task_output(ci, orch, "orchestrator")["job_id"] == 1000


def test_trim_log_and_mirror_pagination(monkeypatch):
    assert trim_log("x" * 10) == "x" * 10
    long = "a" * 100_000 + "b" * 300_000
    t = trim_log(long)
    assert t.startswith("a") and t.endswith("b") and "trimmed by ospm" in t and len(t) < 210_000
    from ospm.replay import mirror as m

    pages = {
        "https://api.github.com/x?per_page=100": ([1, 2], '<https://api.github.com/x?page=2>; rel="next"'),
        "https://api.github.com/x?page=2": ([3], ""),
    }

    class Resp(io.BytesIO):
        def __init__(self, body, link):
            super().__init__(body)
            self.headers = {"Link": link}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Opener:
        def open(self, req, timeout=None):
            body, link = pages[req.full_url]
            return Resp(json.dumps(body).encode(), link)

    gh = m.GH("tok")
    gh.opener = Opener()
    assert list(gh.paginate("/x")) == [1, 2, 3] and gh.calls == 2


def test_rmtree_removes_read_only_files(tmp_path):
    import stat

    from ospm.replay.workspace import _rmtree

    d = tmp_path / "wt" / ".git" / "objects" / "pack"
    d.mkdir(parents=True)
    f = d / "pack-1.pack"
    f.write_bytes(b"x")
    f.chmod(stat.S_IREAD)
    _rmtree(tmp_path / "wt")
    assert not (tmp_path / "wt").exists()
    _rmtree(tmp_path / "missing")  # no error for a path that does not exist


def test_ghstack_base_is_the_base_commit_at_review_time(tmp_path):
    """ghstack heads: [INITIAL] on the original base, a single-parent update keeps the base, a merge-shaped update
    moves it to its second parent. Non-ghstack commits have no ghstack base."""
    from ospm.replay.ghstack import base_of, is_ghstack, target_branch

    _git(tmp_path, "init", "-q")
    tree = _git(tmp_path, "write-tree")

    def commit(msg, *parents):
        return _git(tmp_path, "commit-tree", tree, *[a for p in parents for a in ("-p", p)], "-m", msg)

    main = commit("Fix a bug (#1)")
    b1 = commit("Update (base update)\n\n[ghstack-poisoned]", main)
    h1 = commit("[INITIAL] Update\n\n[ghstack-poisoned]", b1)
    h2 = commit("Update\n\n[ghstack-poisoned]", h1)  # older ghstack: no [UPDATE] prefix
    b2 = commit("Update (base update)\n\n[ghstack-poisoned]", b1)
    h3 = commit("[UPDATE] Update\n\n[ghstack-poisoned]", h2, b2)
    h4 = commit("[UPDATE] Update\n\n[ghstack-poisoned]", h3)

    def git(*args):
        return _git(tmp_path, *args)

    assert [base_of(git, h) for h in (h1, h2, h3, h4)] == [b1, b1, b2, b2]
    assert base_of(git, main) == "" and base_of(git, b1) == ""
    assert is_ghstack("gh/alice/12/base") and not is_ghstack("main") and not is_ghstack("gh-pages")
    assert target_branch("gh/alice/12/base") == "main" and target_branch("release/2.9") == "release/2.9"
    assert target_branch(None) == "main"


def test_mirror_counts_bot_landed_prs_as_merged_and_pytorch_bots_as_bots():
    from ospm.replay.mirror import merged_at
    from ospm.replay.store import is_bot

    assert merged_at({"merged_at": "2026-09-01T00:00:00Z", "closed_at": "2026-09-02T00:00:00Z"}) == "2026-09-01T00:00:00Z"
    landed = {"merged_at": None, "closed_at": "2026-09-02T00:00:00Z", "labels": [{"name": "Merged"}]}
    assert merged_at(landed) == "2026-09-02T00:00:00Z"
    assert merged_at({"merged_at": None, "closed_at": "2026-09-02T00:00:00Z", "labels": [{"name": "triaged"}]}) is None
    assert all(is_bot(b) for b in ("pytorchmergebot", "facebook-github-bot", "pytorch-bot[bot]", "Copilot"))
    assert not is_bot("ezyang")
