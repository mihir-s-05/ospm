"""Agents (heuristic + LLM policy), planning, metrics and benchmark data integrity."""

import re

import pytest

from ospm.agents import AGENTS, CITriageAgent, ReviewAgent, SecurityAgent, TriageAgent
from ospm.agents.base import Case, CaseBase
from ospm.agents.ci_triage import failing_tests, first_error, salient_lines
from ospm.agents.planning import BacklogPlanner, bump_version, plan_release
from ospm.agents.triage import issue_case
from ospm.benchmarks import SPLITS, get_task
from ospm.benchmarks.issues_data import ISSUES
from ospm.benchmarks.pr_data import PRS
from ospm.evolution.metrics import evaluate, instance_score, multi_label_metrics, single_label_metrics
from ospm.llm import UsageMeter, validate
from ospm.llm.fake import FakeLLM
from ospm.models import CIRun, Decision, Issue, PullRequest, parse_diff


# ---------------------------------------------------------------------------------------------- triage
def test_triage_spam_and_scope(project):
    agent = TriageAgent(project=project)
    spam = agent.decide(Issue(1, "Buy now cheap", "buy now https://a.x https://b.x https://c.x"))
    assert spam.labels["decision"] == "spam"
    oos = agent.decide(Issue(2, "WYSIWYG editor please", "Please add a WYSIWYG editor in the browser."))
    assert oos.labels["decision"] == "out-of-scope"
    assert "wontfix" in oos.extras["suggested_labels"]
    vague = agent.decide(Issue(3, "broken", "it crashes"))
    assert vague.labels["decision"] == "needs-info"
    assert vague.extras["missing_info"]


def test_triage_duplicate_detection_only_points_to_earlier_issues(project):
    first = Issue(10, "Live reload broken when editing a partial template",
                  "Editing partial templates does not trigger live reload in serve.")
    later = Issue(20, "Live reload broken when I edit partial templates",
                  "When I edit partial templates the dev server does not live reload.")
    tracker = CaseBase([issue_case(first), issue_case(later)])
    agent = TriageAgent(project=project, tracker=tracker)
    dec = agent.decide(later)
    assert dec.labels["decision"] == "duplicate" and dec.extras["duplicate_of"] == "issue-10"
    assert agent.find_duplicate(first) is None  # the later issue is never the original


def test_knn_memory_influences_decisions(project):
    text_case = Case("issue-900", "Sitemap ignores draft flag\nsitemap.xml lists draft pages", {
        "kind": "bug", "decision": "accept", "priority": "P1"})
    agent = TriageAgent(project=project, cases=CaseBase([text_case]))
    dec = agent.decide(Issue(901, "Draft pages appear in sitemap.xml", "The sitemap lists draft pages."))
    assert dec.neighbors and dec.neighbors[0][0] == "issue-900"
    assert dec.labels["priority"] == "P1"
    # leave-one-out exclusion
    assert agent.decide(Issue(900, "Sitemap ignores draft flag", "sitemap.xml lists draft pages"),
                        exclude={"issue-900"}).neighbors == []


def test_output_schemas_are_strict_mode_compatible(project):
    """OpenAI strict mode needs every property required and additionalProperties false on all objects."""
    def check(s):
        if s.get("type") == "object":
            assert s.get("additionalProperties") is False
            assert set(s["required"]) == set(s["properties"])
            for sub in s["properties"].values():
                check(sub)
        if s.get("type") == "array":
            check(s["items"])
    for cls in AGENTS.values():
        check(cls(project=project).output_schema())


def _llm_answer(agent_cls, labels, **extra):
    def responder(system, user, schema):
        out = {}
        for name, prop in schema["properties"].items():
            if name in labels:
                out[name] = labels[name]
            elif name in extra:
                out[name] = extra[name]
            elif prop["type"] == "array":
                out[name] = []
            elif prop["type"] == "string":
                out[name] = prop.get("enum", ["x"])[0] if "enum" in prop else "ok"
            else:
                out[name] = 0
        assert validate(out, schema) == []
        return out
    return responder


def test_llm_policy_overrides_labels_and_keeps_heuristic_extras(project):
    llm = FakeLLM(_llm_answer(TriageAgent, {"kind": "feature", "decision": "accept", "priority": "P1"},
                              rationale="clear feature request"))
    agent = TriageAgent(project=project, llm=llm)
    dec = agent.decide(Issue(5, "Add support for Mermaid", "Please add mermaid diagrams to code fences."))
    assert dec.used_llm and dec.labels == {"kind": "feature", "decision": "accept", "priority": "P1"}
    assert dec.rationale == "clear feature request"
    assert dec.extras["suggested_labels"] == ["enhancement", "priority: P1"]
    req = llm.requests[0]
    assert "Playbook" in req["system"] and "inkwell" in req["system"]
    assert "Heuristic pre-assessment" in req["user"] and "Mermaid" in req["user"]


def test_llm_failure_falls_back_to_heuristics(project):
    llm = FakeLLM(lambda s, u, sch: {"kind": "nope"}, meter=UsageMeter(budget_usd=None))
    dec = TriageAgent(project=project, llm=llm).decide(Issue(6, "App crashes", "Traceback ... error"))
    assert not dec.used_llm and "llm_error" in dec.extras and dec.labels["kind"] == "bug"
    broke = FakeLLM(lambda s, u, sch: {}, meter=UsageMeter(budget_usd=0.0))
    dec = TriageAgent(project=project, llm=broke).decide(Issue(7, "x", "y"))
    assert "budget" in dec.extras["llm_error"]


def test_genome_agent_mismatch_rejected(project):
    with pytest.raises(ValueError):
        TriageAgent(CITriageAgent.seed_genome(project), project=project)


# ---------------------------------------------------------------------------------------------- CI
PYTEST_LOG = """2026-01-01T00:00:00Z collected 3 items
2026-01-01T00:00:01Z FAILED tests/test_a.py::test_one - AssertionError: assert 1 == 2
2026-01-01T00:00:02Z ##[error]Process completed with exit code 1."""


def test_ci_extraction_helpers():
    lines = salient_lines(PYTEST_LOG)
    assert all(not re.match(r"^\d{4}-", l) for l in lines)  # timestamps stripped
    assert failing_tests(PYTEST_LOG) == ["tests/test_a.py::test_one"]
    assert first_error(lines).startswith("FAILED tests/test_a.py::test_one")
    assert failing_tests("--- FAIL: TestX (0.0s)\n  ● Suite › case") == ["TestX", "Suite › case"]


def test_ci_agent_outputs(project):
    agent = CITriageAgent(project=project)
    dec = agent.decide(CIRun("tests", "failure", PYTEST_LOG, run_id="pr-1/tests"))
    assert dec.item_id == "pr-1/tests"
    assert dec.labels["category"] == "test_failure"
    assert "pytest tests/test_a.py::test_one" in dec.extras["suggested_fix"]
    assert dec.extras["retry_recommended"] is False
    assert agent.decide(CIRun("t", "timed_out", "still running...")).labels["category"] == "timeout"


# ---------------------------------------------------------------------------------------------- diffs
def _pr(pid):
    d = next(p for p in PRS if p["id"] == pid)
    return PullRequest(number=d["number"], title=d["title"], diff=d["diff"])


def test_review_agent_locates_findings(project):
    dec = ReviewAgent(project=project).decide(_pr("pr-24"))
    assert set(dec.labels["findings"]) >= {"breakpoint_left", "debug_print"}
    locs = {(c["finding"], c["path"], c["line"]) for c in dec.extras["comments"]}
    assert ("debug_print", "src/inkwell/markdown/footnotes.py", 8) in locs


def test_review_print_in_cli_and_missing_tests(project):
    agent = ReviewAgent(project=project)
    assert "missing_tests" in agent.decide(_pr("pr-4")).labels["findings"]
    assert "missing_tests" not in agent.decide(_pr("pr-1")).labels["findings"]


def test_security_agent_seed_rules(project):
    agent = SecurityAgent(project=project)
    assert set(agent.decide(_pr("pr-8")).labels["findings"]) == {"hardcoded_secret", "insecure_tls"}
    assert agent.decide(_pr("pr-20")).labels["findings"] == []  # secrets read from the environment are fine


# ---------------------------------------------------------------------------------------------- planning
def test_semver_and_release_plan():
    assert bump_version("v0.10.1", "major") == "v0.11.0"  # pre-1.0: breaking -> minor
    assert bump_version("1.4.2", "major") == "2.0.0"
    assert bump_version("1.4.2", "minor") == "1.5.0"
    assert bump_version("1.4.2", "patch") == "1.4.3"
    assert bump_version("1.4.2", "none") == "1.4.2"
    with pytest.raises(ValueError):
        bump_version("latest", "patch")
    plan = plan_release("1.2.3", [PullRequest(1, "fix: a"), PullRequest(2, "feat(cli)!: drop py3.9"),
                                  PullRequest(3, "chore: deps")])
    assert plan.bump == "major" and plan.next_version == "2.0.0"
    assert plan.breaking == ["drop py3.9 (#2)"] and "Internal" in plan.sections
    assert plan_release("1.2.3", [PullRequest(4, "chore: x")]).bump == "none"
    assert plan_release("1.2.3", [PullRequest(5, "docs: x")]).bump == "patch"


def test_backlog_ranking_and_milestones(project):
    def dec(i, kind, decision, pri, comp="unknown"):
        return Decision("triage", f"issue-{i}", {"kind": kind, "decision": decision, "priority": pri},
                        extras={"component": comp})
    issues = [Issue(i, f"t{i}", "", reactions=i) for i in range(1, 6)]
    decs = [dec(1, "bug", "accept", "P2"), dec(2, "bug", "accept", "P0"), dec(3, "feature", "spam", "P3"),
            dec(4, "feature", "accept", "P1", "i18n"), dec(5, "bug", "needs-info", "P1")]
    planner = BacklogPlanner(project)
    items = planner.groom(list(zip(issues, decs)))
    assert [it.issue.number for it in items][:2] == [2, 4]
    assert 3 not in [it.issue.number for it in items]
    assert items[1].assignees == ["chen"]
    plan = planner.plan_milestones(items, ["m1", "m2"], capacity=6)
    # P0 (#2, 2 pts) always lands first; the 5-pt feature no longer fits m1 (2+5 > 6) so it moves to m2,
    # and the next 2-pt bug fills m1; the needs-info issue is not ready and is not scheduled
    assert [i.issue.number for i in plan["m1"]] == [2, 1]
    assert [i.issue.number for i in plan["m2"]] == [4]


# ---------------------------------------------------------------------------------------------- metrics
def test_metrics():
    spec = TriageAgent.fields
    gold = {"kind": None, "decision": "spam", "priority": "P3"}
    assert instance_score(spec, {"kind": "bug", "decision": "spam", "priority": "P3"}, gold) == 1.0  # kind not scored
    assert instance_score(spec, {"kind": "bug", "decision": "accept", "priority": "P3"}, gold) == pytest.approx(1 / 2.5)
    m = single_label_metrics(("a", "b", "c"), [("a", "a"), ("b", "a"), ("b", "b")])
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert set(m["per_label"]) == {"a", "b"}  # "c" never appears
    ml = multi_label_metrics([(["x"], ["x", "y"]), ([], [])])
    assert ml["micro_precision"] == 1.0 and ml["micro_recall"] == 0.5 and ml["exact_match"] == 0.5
    rev = ReviewAgent.fields
    assert instance_score(rev, {"findings": []}, {"findings": []}) == 1.0
    assert instance_score(rev, {"findings": ["a"]}, {"findings": ["a", "b"]}) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------------------------- benchmark integrity
def test_issue_dataset_integrity():
    numbers = [d["number"] for d in ISSUES]
    assert len(numbers) == len(set(numbers)) == 96
    for d in ISSUES:
        lab = d["labels"]
        assert lab["decision"] in TriageAgent.field("decision").labels
        assert lab["priority"] in TriageAgent.field("priority").labels
        assert (lab["kind"] is None) == (lab["decision"] == "spam")
        if lab["kind"] is not None:
            assert lab["kind"] in TriageAgent.field("kind").labels
        if lab["decision"] != "accept":
            assert lab["priority"] == "P3", d["number"]
        if lab["decision"] == "duplicate":
            assert d["duplicate_of"] in numbers and d["duplicate_of"] < d["number"]
        else:
            assert d["duplicate_of"] is None


def test_pr_dataset_labels_follow_the_definitions(project):
    """missing_tests and the src/ scoping are mechanical; check every PR against the written definition."""
    for d in PRS:
        files = parse_diff(d["diff"])
        assert files and all(f.added or f.removed for f in files), d["id"]
        lib_added = sum(len(f.added) for f in files if project.is_library_path(f.path) and f.path.endswith(".py"))
        tests_changed = any(project.is_test_path(f.path) for f in files)
        assert ("missing_tests" in d["review"]) == (lib_added >= 3 and not tests_changed), d["id"]
        assert set(d["review"]) <= set(ReviewAgent.field("findings").labels)
        assert set(d["security"]) <= set(SecurityAgent.field("findings").labels)


@pytest.mark.parametrize("name", list(AGENTS))
def test_splits_are_disjoint_complete_and_stratified(name):
    task = get_task(name)
    assert set(task.split_of.values()) == set(SPLITS)
    assert sorted(task.split_of) == sorted(e.id for e in task.examples)
    sizes = {s: len(task.subset(s)) for s in SPLITS}
    assert sizes["train"] > sizes["val"] > 0 and sizes["test"] > 0
    assert len({e.id for e in task.examples}) == len(task.examples)


@pytest.mark.parametrize("name", list(AGENTS))
def test_seed_agents_score_reasonably(name):
    task = get_task(name)
    res = evaluate(task.make_agent(), task.subset("val"))
    assert 0.4 <= res.score <= 1.0


def test_llm_prompt_withholds_relational_labels(project):
    """A neighbour that duplicated *this* issue must not teach the LLM that this issue is a duplicate."""
    dup = Case("issue-111", "Live reload broken when editing partial templates",
               {"kind": "bug", "decision": "duplicate", "priority": "P3"})
    llm = FakeLLM(_llm_answer(TriageAgent, {"kind": "bug", "decision": "accept", "priority": "P1"}))
    agent = TriageAgent(project=project, llm=llm, cases=CaseBase([dup]))
    agent.decide(Issue(101, "Live reload broken when editing partial templates", "serve ignores partials"))
    user = llm.requests[0]["user"]
    assert "Live reload broken" in user and '"kind": "bug"' in user
    assert "duplicate" not in user.split("## Heuristic pre-assessment")[0]


@pytest.mark.parametrize("cls", [ReviewAgent, SecurityAgent, CITriageAgent, TriageAgent])
def test_llm_prompt_with_retrieved_neighbours_for_every_agent(cls, project):
    """Exercise the full LLM prompt path (exemplars + neighbours) for single- and multi-label agents."""
    task = get_task(cls.name)
    llm = FakeLLM(_llm_answer(cls, {}))
    agent = task.make_agent(llm=llm)
    agent.genome.exemplars = [e.id for e in task.subset("train")[:2]]
    agent.genome.params["min_sim"] = 0.0  # always retrieve neighbours
    for ex in task.subset("val")[:3]:
        dec = agent.decide(ex.item, exclude={ex.id})
        assert dec.used_llm, dec.extras.get("llm_error")
    assert "Similar past cases" in llm.requests[0]["user"]
    assert "Reference examples" in llm.requests[0]["user"]
