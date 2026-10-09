"""The project-manager orchestrator: an AI-driven SDLC pipeline of specialised, self-evolving agents.

    new issue ──► TriageAgent ──► backlog (BacklogPlanner) ──► milestones ──► release (plan_release)
    pull request ──► CITriageAgent (per failed job) ─┐
                 ──► ReviewAgent (conventions)  ─────┼──► merge-readiness verdict + comments
                 ──► SecurityAgent              ─────┘
    maintainer feedback ──► episodic memory (cases) + rule credit ──► GEPA evolution ──► promotion gate

Every agent decision is stored as an episode. When a maintainer corrects or confirms it, the labelled case joins
the agent's retrieval memory immediately (non-parametric learning, no retraining) and becomes training data for
the next evolution run, which may rewrite the agent's instructions, playbook and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .agents import AGENTS, Agent, BacklogPlanner, Case, CaseBase, plan_release
from .agents.planning import BacklogItem, ReleasePlan
from .agents.triage import TriageAgent, issue_case
from .benchmarks import get_task, task_from_memory
from .evolution import GEPA, GEPAConfig, GEPAResult
from .genome import Genome
from .llm import LLM
from .memory import Memory
from .models import CIRun, Decision, Issue, PullRequest, item_to_dict
from .project import ProjectContext

VERDICTS = ("ready_for_review", "approve_with_nits", "changes_requested", "security_review_required",
            "ci_failing", "ci_retry")


@dataclass
class PRReport:
    pr: PullRequest
    ci: list[tuple[CIRun, Decision]]
    review: Decision
    security: Decision
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pr": self.pr.number,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "ci": [{"job": run.name, **d.labels, **{k: d.extras.get(k) for k in ("root_cause", "suggested_fix",
                                                                                   "failing_tests", "retry_recommended")}}
                   for run, d in self.ci],
            "review": {"findings": self.review.labels["findings"], "comments": self.review.extras.get("comments", [])},
            "security": {"findings": self.security.labels["findings"],
                         "comments": self.security.extras.get("comments", [])},
        }


def pr_verdict(ci: list[tuple[CIRun, Decision]], review: Decision, security: Decision) -> tuple[str, list[str]]:
    reasons: list[str] = []
    sec = security.labels["findings"]
    if sec:
        reasons.append("security findings: " + ", ".join(sec))
        return "security_review_required", reasons
    if ci:
        cats = [d.labels["category"] for _, d in ci]
        reasons.append("failed CI jobs: " + ", ".join(f"{r.name} ({c})" for (r, _), c in zip(ci, cats)))
        if all(c == "infra_flaky" for c in cats):
            return "ci_retry", reasons
        return "ci_failing", reasons
    comments = review.extras.get("comments", [])
    serious = [c for c in comments if c.get("severity", "minor") in ("blocker", "major")]
    if serious:
        found = review.labels["findings"]
        other = sum(1 for c in serious if c.get("finding") == "other")
        parts = (["convention findings: " + ", ".join(found)] if found else [])
        if other:
            parts.append(f"{other} major review comment(s) on correctness/design")
        reasons.append("; ".join(parts))
        return "changes_requested", reasons
    if review.labels["findings"]:
        reasons.append("minor review findings: " + ", ".join(review.labels["findings"]))
        return "approve_with_nits", reasons
    reasons.append("automated checks found no problems; needs a human review of the design and business logic")
    return "ready_for_review", reasons


class ProjectManager:
    def __init__(
        self,
        project: ProjectContext,
        memory: Memory,
        *,
        llm: LLM | None = None,
        effort: str | None = "medium",
        log: Callable[[str], None] | None = None,
    ):
        self.project = project
        self.memory = memory
        self.llm = llm
        self.effort = effort
        self.log = log or (lambda msg: None)
        self._agents: dict[str, Agent] = {}
        self.tracker = CaseBase()
        for ep in memory.episodes("triage", limit=5000):
            if ep.get("item") and ep["item"].get("type") == "Issue":
                d = ep["item"]["data"]
                self.tracker.add(issue_case(Issue(**d)))

    # ------------------------------------------------------------------ agents
    def genome(self, name: str) -> Genome:
        return self.memory.deployed(name) or AGENTS[name].seed_genome(self.project)

    def agent(self, name: str) -> Agent:
        if name not in self._agents:
            cls = AGENTS[name]
            kwargs: dict[str, Any] = {"llm": self.llm, "project": self.project, "effort": self.effort}
            if cls is TriageAgent:
                kwargs["tracker"] = self.tracker
            agent = cls(self.genome(name), **kwargs)
            agent.cases = CaseBase(Case(c["id"], c["text"], c["labels"]) for c in self.memory.cases(name))
            self._agents[name] = agent
        return self._agents[name]

    def reload(self) -> None:
        """Drop cached agents (after a promotion or new feedback) so they pick up the latest memory."""
        self._agents.clear()

    def _decide(self, name: str, item: Any) -> Decision:
        agent = self.agent(name)
        dec = agent.decide(item, exclude={agent.item_id(item)})
        self.memory.record_episode(name, dec.item_id, agent.item_text(item), dec.to_dict(), dec.genome_id,
                                   item_to_dict(item))
        return dec

    # ------------------------------------------------------------------ SDLC steps
    def triage_issue(self, issue: Issue, known_issues: list[Issue] = ()) -> Decision:
        for other in known_issues:
            if other.id != issue.id and other.id not in self.tracker.by_id:
                self.tracker.add(issue_case(other))
        dec = self._decide("triage", issue)
        if issue.id not in self.tracker.by_id:
            self.tracker.add(issue_case(issue))
        return dec

    def triage_ci(self, run: CIRun) -> Decision:
        return self._decide("ci_triage", run)

    def review_pr(self, pr: PullRequest) -> PRReport:
        ci = [(run, self.triage_ci(run)) for run in pr.ci_runs if run.failed]
        review = self._decide("review", pr)
        security = self._decide("security", pr)
        verdict, reasons = pr_verdict(ci, review, security)
        return PRReport(pr, ci, review, security, verdict, reasons)

    def groom_backlog(self, issues: list[Issue], triaged: dict[str, Decision] | None = None,
                      milestones: list[str] | None = None) -> tuple[list[BacklogItem], dict[str, list[BacklogItem]]]:
        triaged = dict(triaged or {})
        pairs = []
        for issue in issues:
            dec = triaged.get(issue.id) or self.triage_issue(issue, issues)
            pairs.append((issue, self.effective(dec)))
        planner = BacklogPlanner(self.project)
        items = planner.groom(pairs)
        return items, planner.plan_milestones(items, milestones)

    def effective(self, dec: Decision) -> Decision:
        """The decision with any maintainer corrections applied (feedback outranks the agent)."""
        ep = self.memory.latest_episode(dec.agent, dec.item_id)
        if not ep or not ep.get("outcome"):
            return dec
        fixed = Decision(**{**dec.to_dict(), "labels": {**dec.labels, **ep["outcome"]["labels"]}})
        fixed.neighbors = [tuple(n) for n in fixed.neighbors]
        return fixed

    def plan_release(self, current_version: str, merged: list[PullRequest]) -> ReleasePlan:
        return plan_release(current_version, merged)

    # ------------------------------------------------------------------ learning from feedback
    def feedback(self, agent_name: str, item_id: str, labels: dict[str, Any], source: str = "feedback") -> dict[str, Any]:
        """Record the correct labels for a past decision.

        Effects: the episode gets an outcome; rules that fired get credit/blame; the labelled case is added to
        the agent's retrieval memory (used from the very next decision) and to future evolution training data.
        """
        cls = AGENTS[agent_name]
        valid = {f.name: f for f in cls.fields}
        for k, v in labels.items():
            if k not in valid:
                raise ValueError(f"{agent_name} has no field {k!r} (fields: {', '.join(valid)})")
            allowed = valid[k].labels
            vals = v if isinstance(v, list) else [v]
            bad = [x for x in vals if x not in allowed]
            if bad:
                raise ValueError(f"invalid {k} value(s) {bad}; allowed: {', '.join(allowed)}")
            if valid[k].multi and not isinstance(v, list):
                labels[k] = [v]
        ep = self.memory.latest_episode(agent_name, item_id)
        if ep is None:
            raise KeyError(f"no recorded {agent_name} decision for {item_id!r}")
        predicted = ep["decision"]["labels"]
        full = {**predicted, **labels}
        correct = {k: predicted.get(k) == v if not isinstance(v, list) else sorted(predicted.get(k) or []) == sorted(v)
                   for k, v in labels.items()}
        self.memory.set_outcome(ep["id"], {"labels": full, "correct": correct, "source": source})
        genome = self.memory.load_genome(ep["genome_id"]) if ep["genome_id"] else None
        rules = {r.id: r for r in (genome.rules if genome else self.genome(agent_name).rules)}
        for rid in ep["decision"].get("fired_rules", []):
            r = rules.get(rid)
            if r is None or r.field not in full:
                continue
            gold = full[r.field]
            self.memory.update_rule_stats(agent_name, rid, r.value in gold if isinstance(gold, list) else r.value == gold)
        self.memory.upsert_case(agent_name, item_id, ep["item_text"], full, source, ep.get("item"))
        self._agents.pop(agent_name, None)
        return {"episode": ep["id"], "labels": full, "was_correct": correct}

    def bootstrap_benchmark_memory(self, names: tuple[str, ...] = tuple(AGENTS)) -> dict[str, int]:
        """Load the built-in benchmark's *train* split into case memory (useful for the inkwell demo)."""
        counts = {}
        for name in names:
            task = get_task(name, self.project)
            proto = task.make_agent()
            n = 0
            for ex in task.subset("train"):
                self.memory.upsert_case(name, ex.id, proto.item_text(ex.item), dict(ex.labels), "benchmark",
                                        item_to_dict(ex.item))
                n += 1
            counts[name] = n
        self.reload()
        return counts

    # ------------------------------------------------------------------ self-evolution
    def evolve(
        self,
        name: str,
        *,
        config: GEPAConfig | None = None,
        reflection_llm: LLM | None = None,
        use_policy_llm: bool = False,
        include_benchmark: bool = True,
        promote: bool = True,
        min_delta: float = 0.01,
    ) -> dict[str, Any]:
        """Run GEPA from the currently deployed genome and promote the winner if it clears the gate.

        Promotion gate: mean Pareto-set (val) score must improve by >= min_delta and, when measured, the train
        score must not regress. The held-out test score is reported but never used for the decision.
        """
        task = task_from_memory(name, self.memory, self.project, include_benchmark)
        incumbent = self.genome(name)
        gepa = GEPA(task, incumbent, policy_llm=self.llm if use_policy_llm else None, reflection_llm=reflection_llm,
                    config=config, memory=self.memory, log=self.log)
        result: GEPAResult = gepa.run()
        best, seed = result.best, result.seed
        reasons = []
        ok = best.id != seed.id
        if not ok:
            reasons.append("no candidate beat the incumbent")
        elif best.mean < seed.mean + min_delta:
            ok = False
            reasons.append(f"val gain {best.mean - seed.mean:+.4f} below min_delta {min_delta}")
        elif best.train is not None and seed.train is not None and best.train < seed.train:
            ok = False
            reasons.append("train score regressed")
        metrics = {"run_id": result.run_id, "val": round(best.mean, 4), "previous_val": round(seed.mean, 4),
                   "train": best.train, "test": result.test, "operators": list(gepa.operators)}
        promoted = False
        if ok and promote:
            self.memory.deploy(best.genome, metrics)
            self._agents.pop(name, None)
            promoted = True
            reasons.append(f"val {seed.mean:.4f} -> {best.mean:.4f}")
        elif ok:
            reasons.append("eligible, but promotion disabled")
        return {"agent": name, "promoted": promoted, "reasons": reasons, "incumbent": seed.id, "best": best.id,
                "summary": result.summary(), "result": result}
