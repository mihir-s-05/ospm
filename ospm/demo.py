"""End-to-end walkthrough on the bundled 'inkwell' project: the SDLC pipeline plus both learning loops.

1. Online learning: issues arrive one at a time; after each decision a maintainer supplies the correct labels.
   The case memory grows and later decisions benefit (no retraining, no evolution yet).
2. Evolution: GEPA evolves the triage and CI agents; the promotion gate decides what goes live.
3. Pull requests: CI failure triage + convention review + security review -> merge-readiness verdicts.
4. Planning: backlog grooming, milestone packing and a release proposal.
"""

from __future__ import annotations

from .benchmarks import get_task
from .benchmarks.ci_data import CI_LOGS
from .benchmarks.issues_data import ISSUES
from .benchmarks.pr_data import PRS
from .evolution import GEPAConfig
from .evolution.metrics import instance_score
from .llm import LLM
from .manager import ProjectManager
from .models import CIRun, Issue, PullRequest
from .render import backlog_table, evolution_report, pr_comment, release_notes


def _h(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def run_demo(pm: ProjectManager, iterations: int = 25, reflection_llm: LLM | None = None) -> dict:
    out: dict = {}
    triage_fields = pm.agent("triage").fields

    # ------------------------------------------------------------------ 1. online learning
    _h("1. Online learning from maintainer feedback (issue stream, memory starts empty)")
    stream = sorted(ISSUES, key=lambda d: d["number"])
    scores: list[float] = []
    triaged: dict[str, object] = {}
    issues: list[Issue] = []
    for d in stream:
        issue = Issue(number=d["number"], title=d["title"], body=d["body"], reactions=d["reactions"],
                      created_at="2026-01-01T00:00:00Z")
        issues.append(issue)
        dec = pm.triage_issue(issue)
        triaged[issue.id] = dec
        scores.append(instance_score(triage_fields, dec.labels, d["labels"]))
        gold = {k: v for k, v in d["labels"].items() if v is not None}
        pm.feedback("triage", issue.id, gold, source="demo-maintainer")
    third = len(scores) // 3
    windows = [scores[:third], scores[third:2 * third], scores[2 * third:]]
    for i, w in enumerate(windows, 1):
        print(f"  issues {1 + (i - 1) * third:>3}-{min(len(scores), i * third) if i < 3 else len(scores):>3}: "
              f"mean triage score {sum(w) / len(w):.3f}  (memory size before window: {(i - 1) * third})")
    out["online_windows"] = [round(sum(w) / len(w), 4) for w in windows]

    # ------------------------------------------------------------------ 2. evolution
    _h("2. GEPA evolution with a promotion gate")
    for name in ("triage", "ci_triage"):
        outcome = pm.evolve(name, config=GEPAConfig(max_iterations=iterations, seed=0), reflection_llm=reflection_llm)
        outcome.pop("result")
        print(evolution_report(outcome))
        print()
        out[f"evolve_{name}"] = {"promoted": outcome["promoted"], "val": outcome["summary"]["best_val"],
                                 "test": outcome["summary"]["test"]}

    # ------------------------------------------------------------------ 3. pull requests
    _h("3. Pull requests: CI triage + review + security")
    ci = {d["id"]: d for d in CI_LOGS}
    prs = {d["id"]: d for d in PRS}
    scenario = [("pr-3", ["ci-12"]), ("pr-9", []), ("pr-19", ["ci-38"]), ("pr-34", [])]
    verdicts = {}
    for pid, logs in scenario:
        d = prs[pid]
        pr = PullRequest(number=d["number"], title=d["title"], body=d["body"], diff=d["diff"],
                         ci_runs=[CIRun(name=ci[c]["name"], conclusion=ci[c]["conclusion"], log=ci[c]["log"],
                                        run_id=f"{pid}/{ci[c]['name']}") for c in logs])
        report = pm.review_pr(pr)
        verdicts[pid] = report.verdict
        print(f"\n--- PR #{d['number']}: {d['title']}\n")
        print(pr_comment(report))
    out["verdicts"] = verdicts

    # ------------------------------------------------------------------ 4. planning
    _h("4. Backlog grooming and milestone plan")
    items, plan = pm.groom_backlog(issues, triaged=triaged, milestones=["v0.11", "v0.12", "v1.0"])
    print(backlog_table(items, plan, limit=15))
    out["backlog_size"] = len(items)

    _h("5. Release proposal")
    merged = [PullRequest(number=d["number"], title=d["title"], body=d["body"]) for d in PRS[:12]]
    rel = pm.plan_release("v0.10.1", merged)
    print(release_notes(rel))
    out["release"] = rel.next_version

    _h("Memory")
    stats = pm.memory.stats()
    print({k: v for k, v in stats.items()})
    out["memory"] = stats
    return out


def benchmark_overview() -> dict[str, int]:
    return {name: len(get_task(name).examples) for name in ("triage", "ci_triage", "review", "security")}
