"""Data-feasibility audit: chain detection and SZZ linking on a synthetic repository."""

import json
import os
import subprocess

from ospm.replay.audit import RepoAudit
from ospm.replay.store import HistoryStore


def _git(cwd, *args, date=None):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@x")
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True).stdout.strip()


def test_szz_links_regression_to_the_pr_that_introduced_it(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    (src / "lib.py").write_text("def f(x):\n    if x is None:\n        return 0\n    return x\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "initial", date="2026-01-01T00:00:00Z")
    _git(src, "checkout", "-qb", "feature")
    (src / "lib.py").write_text("def f(x):\n    return x + 1\n")  # PR #5 drops the None handling
    _git(src, "commit", "-qam", "simplify f", date="2026-02-01T00:00:00Z")
    _git(src, "checkout", "-q", "main")
    _git(src, "merge", "-q", "--no-ff", "feature", "-m", "Merge pull request #5 from x/feature",
         date="2026-02-02T00:00:00Z")
    (src / "lib.py").write_text("def f(x):\n    return 0 if x is None else x + 1\n")  # fix PR #7
    (src / "testing").mkdir()
    (src / "testing" / "test_lib.py").write_text("def test_none():\n    assert f(None) == 0\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "Fix regression with None (#7)", date="2026-05-10T00:00:00Z")
    data = tmp_path / "data"
    data.mkdir()
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(data / "repo.git")], check=True)

    s = HistoryStore(data / "history.db")
    base = dict(author_association="NONE", state="closed", state_reason="completed", closed_at=None, updated_at=None,
                labels_json="[]", reactions=0)
    s.upsert("items", [
        {**base, "number": 5, "is_pr": 1, "title": "simplify f", "body": "cleanup", "author": "a",
         "created_at": "2026-01-30T00:00:00Z", "in_window": 0},
        {**base, "number": 7, "is_pr": 1, "title": "Fix regression with None", "body": "Fixes #8",
         "author": "b", "created_at": "2026-05-09T00:00:00Z", "in_window": 1},
        {**base, "number": 8, "is_pr": 0, "title": "f(None) crashes since 2.1",
         "body": "This worked in 2.0, regression after upgrading to 2.1", "author": "c",
         "created_at": "2026-05-01T00:00:00Z", "in_window": 1},
    ])
    s.upsert("pulls", [{"number": 7, "head_sha": "", "head_ref": "fix", "head_repo": "", "base_ref": "main",
                        "base_sha": "", "merged_at": "2026-05-10T00:00:00Z", "merge_commit_sha": None, "draft": 0,
                        "have_head": 0}])
    s.set_meta("repo", "demo/demo")
    s.set_meta("since", "2026-04-01T00:00:00Z")
    s.close()

    audit = RepoAudit(data)
    assert set(audit.commit_prs().values()) == {5, 7}
    (reg,) = audit.regressions()
    assert reg.linked and reg.earlier["number"] == 5 and "SZZ" in reg.notes
    assert reg.later_signal["fix_prs"] == [7] and reg.delay_days and reg.delay_days > 80
    summary, chains = audit.run()
    assert summary["chains"]["regression"]["linked_to_earlier_decision"] == 1
    dossier = audit.dossier(reg)
    assert "PR #5" in dossier and "return x + 1" in dossier and json.dumps(reg.reconstructable) in dossier
