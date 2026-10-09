"""Benchmark tasks: labelled data + stratified train/val/test splits for every evolvable agent.

Splits follow GEPA's protocol:
* train – feedback set: minibatches are drawn from it for reflective mutation, and it seeds the case memory;
* val   – Pareto set: candidates are tracked per-instance on it and the final candidate is selected on it;
* test  – held out; reported only, never used for any decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents import AGENTS, Agent, Case, CaseBase
from ..agents.triage import TriageAgent, issue_case
from ..genome import Genome
from ..llm import LLM
from ..models import CIRun, Issue, PullRequest
from ..project import ProjectContext

INKWELL_PROJECT = Path(__file__).with_name("inkwell.toml")
SPLITS = ("train", "val", "test")


@dataclass
class Example:
    id: str
    item: Any
    labels: dict[str, Any]  # gold labels; a None value means "not applicable, do not score"


@dataclass
class Task:
    name: str
    examples: list[Example]
    project: ProjectContext
    split_of: dict[str, str] = field(default_factory=dict)
    extra_cases: list[Case] = field(default_factory=list)  # e.g. maintainer feedback stored in memory

    @property
    def agent_cls(self) -> type[Agent]:
        return AGENTS[self.name]

    def subset(self, split: str) -> list[Example]:
        return [e for e in self.examples if self.split_of[e.id] == split]

    def case_base(self, agent: Agent, splits: tuple[str, ...] = ("train",)) -> CaseBase:
        cases = [Case(e.id, agent.item_text(e.item), dict(e.labels)) for e in self.examples if self.split_of[e.id] in splits]
        return CaseBase(cases + list(self.extra_cases))

    def make_agent(self, genome: Genome | None = None, llm: LLM | None = None, effort: str | None = "medium") -> Agent:
        cls = self.agent_cls
        kwargs: dict[str, Any] = {"llm": llm, "project": self.project, "effort": effort}
        if cls is TriageAgent:
            kwargs["tracker"] = CaseBase(issue_case(e.item) for e in self.examples)
        agent = cls(genome or cls.seed_genome(self.project), **kwargs)
        agent.cases = self.case_base(agent)
        return agent


def stratified_split(examples: list[Example], key) -> dict[str, str]:
    """Deterministic ~60/20/20 split, stratified by `key(example)`.

    Examples are ordered by (stratum, id) and dealt round-robin *without* resetting between strata, so even
    rare strata (single examples) are spread across splits instead of all landing in train.
    """
    pattern = ("train", "val", "train", "test", "train")
    ordered = sorted(examples, key=lambda e: (str(key(e)), e.id))
    return {e.id: pattern[i % len(pattern)] for i, e in enumerate(ordered)}


def load_project() -> ProjectContext:
    return ProjectContext.load(INKWELL_PROJECT)


def triage_task(project: ProjectContext | None = None) -> Task:
    from .issues_data import ISSUES

    exs = [
        Example(
            id=f"issue-{d['number']}",
            item=Issue(number=d["number"], title=d["title"], body=d["body"], reactions=d["reactions"],
                       created_at="2026-01-01T00:00:00Z"),
            labels=dict(d["labels"]),
        )
        for d in ISSUES
    ]
    return Task("triage", exs, project or load_project(), stratified_split(exs, lambda e: e.labels["decision"]))


def ci_task(project: ProjectContext | None = None) -> Task:
    from .ci_data import CI_LOGS

    exs = [
        Example(id=d["id"], item=CIRun(name=d["name"], conclusion=d["conclusion"], log=d["log"], url=d["id"]),
                labels={"category": d["category"]})
        for d in CI_LOGS
    ]
    return Task("ci_triage", exs, project or load_project(), stratified_split(exs, lambda e: e.labels["category"]))


def _pr_task(name: str, project: ProjectContext | None) -> Task:
    from .pr_data import PRS

    exs = [
        Example(id=d["id"], item=PullRequest(number=d["number"], title=d["title"], body=d["body"], diff=d["diff"]),
                labels={"findings": list(d[name])})
        for d in PRS
    ]
    key = lambda e: e.labels["findings"][0] if e.labels["findings"] else "none"  # noqa: E731
    return Task(name, exs, project or load_project(), stratified_split(exs, key))


def review_task(project: ProjectContext | None = None) -> Task:
    return _pr_task("review", project)


def security_task(project: ProjectContext | None = None) -> Task:
    return _pr_task("security", project)


TASKS = {"triage": triage_task, "ci_triage": ci_task, "review": review_task, "security": security_task}


def task_from_memory(name: str, memory, project: ProjectContext, include_benchmark: bool = True) -> Task:
    """Training task = (optionally) the built-in benchmark + labelled cases accumulated from maintainer feedback.

    Feedback cases get their own stratified split, so new experience also reaches the val (selection) and test
    (reporting) sets instead of only the training set. Feedback overrides benchmark labels for the same id.
    """
    from ..models import item_from_dict

    base = get_task(name, project) if include_benchmark else Task(name, [], project)
    by_id = {e.id: e for e in base.examples}
    fresh: list[Example] = []
    for c in memory.cases(name):
        if c["source"] == "benchmark" or not c.get("item"):
            continue
        ex = Example(id=c["id"], item=item_from_dict(c["item"]), labels=dict(c["labels"]))
        if ex.id in by_id:
            # fields the benchmark marks "not applicable" (None) stay unscored
            old = by_id[ex.id].labels
            by_id[ex.id].labels = {k: (None if old.get(k, "") is None else v) for k, v in ex.labels.items()}
        else:
            fresh.append(ex)
    split = dict(base.split_of)
    if fresh:
        first = AGENTS[name].fields[0].name
        split.update(stratified_split(fresh, lambda e: e.labels.get(first)))
    return Task(name, base.examples + fresh, project, split)


def get_task(name: str, project: ProjectContext | None = None) -> Task:
    try:
        return TASKS[name](project)
    except KeyError:
        raise KeyError(f"unknown task {name!r}; choose from {', '.join(TASKS)}") from None
