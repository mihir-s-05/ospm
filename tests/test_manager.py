"""ProjectManager orchestration, feedback-driven learning, promotion gate, GitHub integration and CLI."""

import io
import json
import urllib.error
import urllib.request

import pytest

from ospm.benchmarks import task_from_memory
from ospm.benchmarks.ci_data import CI_LOGS
from ospm.benchmarks.pr_data import PRS
from ospm.cli import main
from ospm.evolution import GEPAConfig
from ospm.genome import Rule
from ospm.integrations import GitHubClient, GitHubError, handle_event, labels_to_triage, parse_command, parse_ref
from ospm.integrations.github import _StripAuthOnRedirect
from ospm.manager import ProjectManager, pr_verdict
from ospm.models import CIRun, Decision, Issue, PullRequest


@pytest.fixture
def pm(project, memory):
    return ProjectManager(project, memory)


def _pr(pid, ci=()):
    d = next(p for p in PRS if p["id"] == pid)
    logs = {c["id"]: c for c in CI_LOGS}
    return PullRequest(number=d["number"], title=d["title"], diff=d["diff"],
                       ci_runs=[CIRun(logs[c]["name"], logs[c]["conclusion"], logs[c]["log"],
                                      run_id=f"{pid}/{logs[c]['name']}") for c in ci])


# ---------------------------------------------------------------------------------------------- pipeline
def test_triage_records_episode_and_tracks_issue(pm, memory):
    dec = pm.triage_issue(Issue(1, "App crashes on build", "Traceback: KeyError in build"))
    ep = memory.latest_episode("triage", "issue-1")
    assert ep["decision"]["labels"] == dec.labels and ep["item"]["type"] == "Issue"
    assert "issue-1" in pm.tracker.by_id
    # a fresh manager rebuilds the duplicate tracker from memory
    assert "issue-1" in ProjectManager(pm.project, memory).tracker.by_id


def test_feedback_teaches_memory_and_credits_rules(pm, memory):
    issue = Issue(1, "Sitemap includes drafts", "sitemap.xml lists draft pages; version 0.10, steps: build")
    pm.triage_issue(issue)
    out = pm.feedback("triage", "issue-1", {"kind": "bug", "decision": "accept", "priority": "P1"})
    assert out["labels"]["priority"] == "P1"
    case = memory.cases("triage")[0]
    assert case["id"] == "issue-1" and case["labels"]["priority"] == "P1" and case["item"]["type"] == "Issue"
    # the very next similar decision retrieves the confirmed case
    dec = pm.triage_issue(Issue(2, "Draft pages listed in sitemap", "sitemap.xml lists draft pages"))
    assert dec.neighbors and dec.neighbors[0][0] == "issue-1"
    with pytest.raises(ValueError):
        pm.feedback("triage", "issue-1", {"kind": "nonsense"})
    with pytest.raises(ValueError):
        pm.feedback("triage", "issue-1", {"colour": "red"})
    with pytest.raises(KeyError):
        pm.feedback("triage", "issue-404", {"kind": "bug"})


def test_rule_stats_after_feedback(pm, memory):
    pm.triage_issue(Issue(3, "Crash with traceback", "It crashes, traceback attached, version 1.0, steps: run"))
    pm.feedback("triage", "issue-3", {"kind": "question"})
    stats = memory.rule_stats("triage")
    assert stats["bug-words"] == (1, 0)


def test_pr_review_verdicts(pm, memory):
    assert pm.review_pr(_pr("pr-8")).verdict == "security_review_required"
    # the seed CI genome knows nothing about flaky infrastructure; deploy one that does
    g = pm.genome("ci_triage").clone()
    g.add_rule(Rule(id="net", field="category", value="infra_flaky", any_of=["econnreset"], weight=2.0))
    memory.deploy(g)
    pm.reload()
    assert pm.review_pr(_pr("pr-1", ["ci-37"])).verdict == "ci_retry"
    assert pm.review_pr(_pr("pr-1", ["ci-07"])).verdict == "ci_failing"
    assert pm.review_pr(_pr("pr-3")).verdict == "changes_requested"
    assert pm.review_pr(_pr("pr-34")).verdict == "ready_for_review"
    report = pm.review_pr(_pr("pr-1", ["ci-07", "ci-37"]))
    assert report.to_dict()["ci"][0]["category"] == "lint"


def test_verdict_nits_only():
    review = Decision("review", "pr-1", {"findings": ["wildcard_import"]},
                      extras={"comments": [{"severity": "minor"}]})
    assert pr_verdict([], review, Decision("security", "pr-1", {"findings": []}))[0] == "approve_with_nits"


def test_backlog_uses_maintainer_corrections(pm):
    issues = [Issue(1, "Crash when building site", "Traceback: KeyError. version 1.0. Steps: build."),
              Issue(2, "Crash when building the site", "Traceback KeyError in build, version 1.0, steps: build")]
    triaged = {i.id: pm.triage_issue(i) for i in issues}
    pm.feedback("triage", "issue-2", {"decision": "duplicate", "priority": "P3"})
    items, _ = pm.groom_backlog(issues, triaged=triaged)
    assert [it.issue.number for it in items] == [1]


def test_evolve_promotes_and_then_continues_from_deployed(pm, memory):
    out = pm.evolve("ci_triage", config=GEPAConfig(max_iterations=15, seed=0))
    assert out["promoted"] and memory.deployed_id("ci_triage") == out["best"]
    assert memory.promotions("ci_triage")[0]["metrics"]["val"] == out["summary"]["best_val"]
    assert pm.agent("ci_triage").genome.id == out["best"]
    again = pm.evolve("ci_triage", config=GEPAConfig(max_iterations=3, seed=5), min_delta=0.5)
    assert again["incumbent"] == out["best"] and not again["promoted"]
    assert memory.deployed_id("ci_triage") == out["best"]


def test_feedback_cases_become_training_data(pm, memory, project):
    for i in range(6):
        run = CIRun(f"job{i}", "failure", f"Error: npm ERR! 503 Service Unavailable registry {i}", run_id=f"r{i}")
        pm.triage_ci(run)
        pm.feedback("ci_triage", f"r{i}", {"category": "infra_flaky"})
    task = task_from_memory("ci_triage", memory, project, include_benchmark=True)
    fresh = [e for e in task.examples if e.id.startswith("r")]
    assert len(fresh) == 6 and all(isinstance(e.item, CIRun) for e in fresh)
    assert {task.split_of[e.id] for e in fresh} == {"train", "val", "test"}
    only = task_from_memory("ci_triage", memory, project, include_benchmark=False)
    assert len(only.examples) == 6


# ---------------------------------------------------------------------------------------------- integrations
def test_parsers():
    assert parse_ref("octo/repo#12") == ("octo/repo", 12)
    assert parse_ref("octo/re.po") == ("octo/re.po", None)
    with pytest.raises(ValueError):
        parse_ref("not a ref")
    assert labels_to_triage(["bug", "priority: P1"]) == {"kind": "bug", "priority": "P1", "decision": "accept"}
    assert labels_to_triage(["duplicate"]) == {"decision": "duplicate", "priority": "P3"}
    assert labels_to_triage(["P0"]) == {"priority": "P0", "decision": "accept"}
    assert labels_to_triage(["good first issue"]) == {}
    assert parse_command("thanks!\n/ospm triage kind=docs priority=P3") == ("triage", {"kind": "docs", "priority": "P3"}, None)
    assert parse_command('/ospm ci_triage category=dependency job="tests (3.12)"') == (
        "ci_triage", {"category": "dependency"}, "tests (3.12)")
    assert parse_command("/ospm security findings=sql_injection,code_eval")[1] == {
        "findings": ["sql_injection", "code_eval"]}
    assert parse_command("/ospm review findings=none")[1] == {"findings": []}
    assert parse_command("/ospm unknown x=y") is None and parse_command("no command") is None


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Opener:
    def __init__(self, routes):
        self.routes, self.requests = routes, []

    def open(self, req, timeout=None):
        self.requests.append(req)
        for key, value in self.routes.items():
            if key in req.full_url:
                if isinstance(value, Exception):
                    raise value
                return _Resp(value if isinstance(value, bytes) else json.dumps(value).encode())
        raise AssertionError(f"unexpected URL {req.full_url}")


ISSUE_JSON = {"number": 5, "title": "Crash", "body": "Traceback", "user": {"login": "u"}, "labels": [{"name": "bug"}],
              "reactions": {"total_count": 3}, "created_at": "2026-01-01T00:00:00Z", "state": "open",
              "html_url": "https://github.com/o/r/issues/5"}


def test_github_client_reads_and_dry_run_writes():
    opener = _Opener({
        "/issues/5": ISSUE_JSON,
        "/issues?": [ISSUE_JSON, {**ISSUE_JSON, "number": 6, "pull_request": {}}],
    })
    gh = GitHubClient("o/r", token="t", opener=opener)
    issue = gh.get_issue(5)
    assert (issue.number, issue.labels, issue.reactions, issue.author) == (5, ["bug"], 3, "u")
    assert [i.number for i in gh.list_issues()] == [5]  # pull requests are skipped
    assert opener.requests[0].get_header("Authorization") == "Bearer t"
    gh.comment(5, "hello")
    gh.add_labels(5, ["bug"])
    gh.add_labels(5, [])
    assert [a["kind"] for a in gh.actions] == ["comment", "labels"] and not any(a["performed"] for a in gh.actions)
    assert len(opener.requests) == 2  # dry-run made no write requests


def test_github_client_pull_with_ci_logs_and_errors():
    check_runs = {"check_runs": [
        {"id": 77, "name": "tests", "conclusion": "failure", "html_url": "h", "app": {"slug": "github-actions"},
         "output": {"title": "t", "summary": "s", "text": None}},
        {"id": 78, "name": "lint", "conclusion": "success", "html_url": "h2", "app": {"slug": "github-actions"}},
    ]}
    opener = _Opener({
        "/actions/jobs/77/logs": b"FAILED tests/test_a.py::test_x - AssertionError",
        "/check-runs": check_runs,
        "/pulls/9": {"number": 9, "title": "feat: x", "body": "", "user": {"login": "c"}, "labels": [],
                     "head": {"sha": "abc"}, "merged_at": None},
    })
    # the diff request also hits /pulls/9; serve JSON-or-diff based on Accept header
    orig = opener.open

    def open_(req, timeout=None):
        if req.get_header("Accept") == "application/vnd.github.diff":
            opener.requests.append(req)
            return _Resp(b"diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -0,0 +1,1 @@\n+print(1)\n")
        return orig(req, timeout)
    opener.open = open_
    pr = GitHubClient("o/r", token="t", opener=opener).get_pull(9)
    assert pr.files == ["x"] and len(pr.ci_runs) == 2
    failed = [r for r in pr.ci_runs if r.failed]
    assert failed[0].run_id == "pr-9/tests" and "AssertionError" in failed[0].log
    err = urllib.error.HTTPError("u", 404, "nf", {}, io.BytesIO(b'{"message":"Not Found"}'))
    with pytest.raises(GitHubError, match="404"):
        GitHubClient("o/r", opener=_Opener({"/issues/1": err})).get_issue(1)


def test_redirect_handler_strips_auth_across_hosts():
    h = _StripAuthOnRedirect()
    req = urllib.request.Request("https://api.github.com/x", headers={"Authorization": "Bearer t"})
    new = h.redirect_request(req, None, 302, "Found", {}, "https://blob.example.com/log")
    assert new.get_header("Authorization") is None
    same = h.redirect_request(req, None, 302, "Found", {}, "https://api.github.com/y")
    assert same.get_header("Authorization") == "Bearer t"


def test_handle_event_flows(pm, memory):
    gh = GitHubClient("o/r", opener=_Opener({}))
    payload = {"action": "opened", "issue": {**ISSUE_JSON, "labels": []}, "sender": {"login": "rando"}}
    res = handle_event(pm, gh, "issues", payload)
    assert res["handled"] == "triage" and gh.actions[0]["kind"] == "comment"
    # maintainer labels -> feedback; bots and non-maintainers are ignored
    labeled = {"action": "labeled", "issue": {**ISSUE_JSON, "labels": [{"name": "documentation"}]},
               "label": {"name": "documentation"}, "sender": {"login": "ana"}}
    res = handle_event(pm, gh, "issues", labeled)
    assert res["handled"] == "feedback" and res["labels"]["kind"] == "docs"
    assert handle_event(pm, gh, "issues", {**labeled, "sender": {"login": "github-actions[bot]"}})["handled"] == "ignored"
    # slash command from a maintainer
    cmd = {"action": "created", "issue": {"number": 5}, "comment": {"body": "/ospm triage priority=P0"},
           "sender": {"login": "bo"}}
    assert handle_event(pm, gh, "issue_comment", cmd)["labels"]["priority"] == "P0"
    bad = {**cmd, "comment": {"body": "/ospm triage kind=banana"}}
    assert handle_event(pm, gh, "issue_comment", bad)["handled"] == "error"
    # pull request -> review; CI feedback addresses the job by run id
    pr = _pr("pr-9", ["ci-27"])
    gh.get_pull = lambda n: pr
    res = handle_event(pm, gh, "pull_request", {"action": "opened", "pull_request": {"number": 9}})
    assert res["verdict"] in ("ci_failing", "security_review_required")
    ci_cmd = {"action": "created", "issue": {"number": 9, "pull_request": {}},
              "comment": {"body": "/ospm ci_triage category=dependency"}, "sender": {"login": "chen"}}
    assert handle_event(pm, gh, "issue_comment", ci_cmd)["item"] == "pr-9/deps"
    assert handle_event(pm, gh, "workflow_run", {"action": "completed", "workflow_run": {"conclusion": "success"}})[
        "handled"] == "ignored"


# ---------------------------------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys):
    db = str(tmp_path / "m.db")
    base = ["--db", db]
    assert main(base + ["triage", "--title", "Crash on build", "--body", "Traceback KeyError", "--number", "7"]) == 0
    assert "Triage details" in capsys.readouterr().out
    assert main(base + ["feedback", "--agent", "triage", "--item", "issue-7", "--set", "kind=bug", "priority=P1"]) == 0
    log = tmp_path / "ci.log"
    log.write_text("npm ERR! code ERESOLVE\nnpm ERR! ERESOLVE unable to resolve dependency tree", encoding="utf-8")
    assert main(base + ["ci", "--log", str(log)]) == 0
    assert "root cause" in capsys.readouterr().out
    diff = tmp_path / "x.diff"
    diff.write_text(next(p for p in PRS if p["id"] == "pr-8")["diff"], encoding="utf-8")
    assert main(base + ["review", "--diff", str(diff)]) == 0
    assert "Security review required" in capsys.readouterr().out
    prs = tmp_path / "prs.json"
    prs.write_text(json.dumps([{"number": 1, "title": "feat: x"}, {"number": 2, "title": "fix: y"}]), encoding="utf-8")
    assert main(base + ["release", "--version", "1.0.0", "--prs", str(prs)]) == 0
    assert "1.1.0" in capsys.readouterr().out
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps([{"number": 1, "title": "Crash", "body": "Traceback, version 1.0, steps: run"}]),
                      encoding="utf-8")
    assert main(base + ["backlog", "--issues", str(issues)]) == 0
    assert "#1 Crash" in capsys.readouterr().out
    assert main(base + ["evolve", "--agent", "ci_triage", "--iterations", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["agent"] == "ci_triage"
    assert main(base + ["eval", "--agent", "ci_triage", "--split", "all"]) == 0
    assert set(json.loads(capsys.readouterr().out)["ci_triage"]) == {"genome", "train", "val", "test"}
    assert main(base + ["memory", "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["episodes"] >= 3
    assert main(base + ["memory", "genome", "ci_triage"]) == 0
    assert "## Playbook" in capsys.readouterr().out
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"action": "opened", "issue": ISSUE_JSON, "repository": {"full_name": "o/r"},
                                 "sender": {"login": "x"}}), encoding="utf-8")
    assert main(base + ["gh-event", "--event-name", "issues", "--event-path", str(event)]) == 0
    assert json.loads(capsys.readouterr().out)["handled"] == "triage"


def test_verdict_reason_for_design_comments_only():
    review = Decision("review", "pr-1", {"findings": []},
                      extras={"comments": [{"finding": "other", "severity": "major", "path": "a", "line": 1}]})
    verdict, reasons = pr_verdict([], review, Decision("security", "pr-1", {"findings": []}))
    assert verdict == "changes_requested" and reasons == ["1 major review comment(s) on correctness/design"]


def test_llm_duplicate_other_comments_are_dropped(project):
    from ospm.agents import ReviewAgent
    from ospm.llm.fake import FakeLLM

    d = next(p for p in PRS if p["id"] == "pr-3")
    comments = [{"finding": "bare_except", "severity": "major", "path": "src/inkwell/plugins/loader.py", "line": 6,
                 "message": "m"},
                {"finding": "other", "severity": "minor", "path": "src/inkwell/plugins/loader.py", "line": 6,
                 "message": "same thing"},
                {"finding": "other", "severity": "major", "path": "tests/test_loader.py", "line": 2, "message": "x"}]
    llm = FakeLLM(lambda s, u, sch: {"findings": ["bare_except"], "comments": comments, "summary": "",
                                     "rationale": "r"})
    dec = ReviewAgent(project=project, llm=llm).decide(PullRequest(3, d["title"], diff=d["diff"]))
    assert [(c["finding"], c["line"]) for c in dec.extras["comments"]] == [("bare_except", 6), ("other", 2)]
