"""Run replay tasks through the agent team and score them against what actually happened."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable

from ..evolution.team_gepa import EpisodeOutcome
from ..harness.models import ChatModel
from ..learning.skills import skills_hash
from ..team.roles import TeamGenome
from ..team.runner import Episode, run_episode
from .tasks import ReplayTask, score_task
from .workspace import ReplayRepo

PYTEST_CONTEXT = """## Project: pytest (pytest-dev/pytest)
The Python testing framework. Layout: src/_pytest/ (core + built-in plugins: fixtures.py, python.py
(collection of test modules/functions), main.py, runner.py, reports.py, terminal.py, assertion/ (assertion
rewriting), config/, mark/, capture.py, logging.py, monkeypatch.py, tmpdir.py, unittest.py, doctest.py,
junitxml.py, cacheprovider.py, warnings.py, raises.py, approx in python_api.py, ...), src/pytest/ (public API),
testing/ (the test-suite, uses the `pytester` fixture), doc/en/ (documentation), changelog/ (towncrier fragments
`<issue>.<type>.rst`, type in breaking|deprecation|feature|improvement|bugfix|doc|packaging|contrib|misc - every
user-facing change needs one), AUTHORS.
In scope: the test runner, collection, fixtures, assertion rewriting, marks/parametrize, reporting, built-in
plugins, configuration, typing of the public API. Out of scope: bugs that belong to third-party plugins
(pytest-xdist, pytest-asyncio, pytest-cov, ...) unless pytest causes them, IDE integration, general Python help.
Component labels: topic: approx|collection|config|fixtures|marks|parametrize|reporting|rewrite|selection|
tracebacks|typing|website; plugin: cache|capture|debugging|doctests|junitxml|logging|monkeypatch|pytester|
stepwise|subtests|tmpdir|unittest|warnings|xdist; platform: windows|mac|linux|cygwin|mingw|wasm/wasi.
Maintainer practice: ask for a minimal reproducer ("status: needs information") when one is missing; close
duplicates with a link to the original; regressions are high priority."""


def project_context(repo: ReplayRepo) -> tuple[str, str]:
    """(short name, context text). A `project_context.md` in the mirror's data directory takes precedence."""
    name = repo.store.meta("repo", "project")
    custom = repo.data / "project_context.md"
    if custom.exists():
        return name.split("/")[-1], custom.read_text(encoding="utf-8")
    if name == "pytest-dev/pytest":
        return "pytest", PYTEST_CONTEXT
    labels = [r["label"] for r in repo.store.q("SELECT DISTINCT label FROM issue_events WHERE event='labeled'")]
    return name, f"## Project: {name}\nLabels in use: {', '.join(sorted(l for l in labels if l))}"


def trace_name(task_id: str, genome_id: str, entry_role: str, extra_context: str = "", skills=(),
               role_context: dict[str, str] | None = None) -> str:
    extra_context += "".join(f"\n[{r}]{t}" for r, t in sorted((role_context or {}).items()))
    ctx = hashlib.sha256(extra_context.encode()).hexdigest()[:6] if extra_context else "plain"
    if skills:
        ctx += "-s" + skills_hash(list(skills))
    return f"{task_id.replace(':', '_').replace('@', '_')}__{genome_id}__{entry_role}__{ctx}.json.gz"


def run_task(repo: ReplayRepo, genome: TeamGenome, model: ChatModel, task: ReplayTask, *, entry_role: str,
             log: Callable[[str], None] | None = None, trace_dir: str | Path | None = None,
             extra_context: str = "", skills=(), role_context: dict[str, str] | None = None
             ) -> tuple[Episode, float, str]:
    """`extra_context` is appended to the event shown to the entry agent (e.g. learned interventions);
    `role_context[role]` to the task of every specialist of that role it spawns."""
    if entry_role == "specialist":  # no orchestrator: the task goes straight to its specialist
        entry_role = task.entry_specialist
    project, context = project_context(repo)
    with repo.worktree(task.sha, task.as_of, base=task.meta.get("base", "")) as ws:
        def env_factory():
            return repo.environment(ws, task.as_of, pr=task.pr, pr_sha=task.pr_sha, sha=task.sha,
                                    base=task.meta.get("base", ""), skills=skills)
        event = task.event + extra_context
        variables = {"event": event, "as_of": task.as_of, "task": f"(event time {task.as_of})\n{event}"}
        ep = run_episode(genome, model, env_factory, task_id=task.id, entry_role=entry_role, variables=variables,
                         project=project, project_context=context, log=log, role_context=role_context)
    score, feedback = score_task(task, ep.output, entry_role)
    if trace_dir:
        d = Path(trace_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = trace_name(task.id, genome.id, entry_role, extra_context, skills, role_context)
        with gzip.open(d / name, "wt", encoding="utf-8") as fh:
            json.dump({"task": task.to_dict(), "genome": genome.id, "entry_role": entry_role, "score": score,
                       "feedback": feedback, "output": ep.output, "exit_status": ep.exit_status,
                       "cost": ep.cost, "trace": ep.trace.to_dict()}, fh)
    return ep, score, feedback


def episode_fn(repo: ReplayRepo, model: ChatModel, *, entry_role: str, log: Callable[[str], None] | None = None,
               trace_dir: str | Path | None = None):
    """Adapter for Team-GEPA: (genome, task) -> EpisodeOutcome."""

    def fn(genome: TeamGenome, task: ReplayTask) -> EpisodeOutcome:
        ep, score, feedback = run_task(repo, genome, model, task, entry_role=entry_role, log=log, trace_dir=trace_dir)
        return EpisodeOutcome(task.id, score, feedback, round(ep.cost, 6), ep.role_hashes,
                              ep.trace.render(max_obs=500)[-12000:], ep.exit_status)

    return fn
