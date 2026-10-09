"""Team-GEPA: reflective evolution of a hierarchical multi-agent system on replayed repository history.

GEPA (Agrawal et al., 2025) evolves the prompts of a compound AI system with natural-language reflection,
per-instance Pareto selection and system-aware merge. Team-GEPA adapts it to a *team of tool-using agents*
(an orchestrator that spawns specialists, each a mini-swe-agent):

1. Blame-directed module selection. GEPA picks the module to mutate round-robin. Here the reflector reads the
   hierarchical trajectory (orchestrator steps with the specialists' sub-trajectories nested inside) together
   with replay-grounded feedback, and must name the role responsible for the failure before proposing an edit
   to that role only. (`blame=False` restores round-robin over the roles that ran - the vanilla ablation.)
2. Invocation-aware episode caching. An episode's outcome depends only on the genomes of the roles that
   actually ran in it (a role that was never delegated to cannot influence it: role descriptions live in the
   orchestrator's genome). Cache entries are keyed by (task, {role: genome hash} of the roles that ran), so a
   mutated specialist reuses every cached episode that never invoked it - often most of a minibatch.
3. Cost-aware Pareto frontier. A candidate is dominated only if another is at least as good on every task
   *and* no more expensive on average, so cheaper teams with equal quality survive selection.
4. Replay-grounded feedback. mu_f is written from what actually happened later in the repository (the labels
   maintainers applied, the review comments they left, whether the CI job passed on re-run).
5. Placeholder-safe rewriting. Rewritten role prompts are repaired so tool documentation, the output
   contract and the playbook slot can never be lost.
"""

from __future__ import annotations

import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..llm import LLM, BudgetExceeded, LLMError
from ..team.roles import ROLES, TeamGenome

EpisodeFn = Callable[[TeamGenome, Any], "EpisodeOutcome"]  # (genome, task) -> outcome


@dataclass
class EpisodeOutcome:
    task_id: str
    score: float
    feedback: str
    cost: float
    roles_used: dict[str, str]  # role -> genome hash of the roles that ran
    trace_text: str = ""
    exit_status: str = ""


class EpisodeCache:
    """(task, hashes of the roles that ran) -> outcome. Valid for any genome agreeing on those roles."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._by_task: dict[str, list[EpisodeOutcome]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._by_task.setdefault(d["task_id"], []).append(EpisodeOutcome(**d))

    def get(self, task_id: str, genome: TeamGenome) -> EpisodeOutcome | None:
        hashes = genome.role_hashes()
        with self._lock:
            for o in self._by_task.get(task_id, []):
                if all(hashes.get(r) == h for r, h in o.roles_used.items()):
                    self.hits += 1
                    return o
            self.misses += 1
        return None

    def put(self, o: EpisodeOutcome) -> None:
        with self._lock:
            self._by_task.setdefault(o.task_id, []).append(o)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(o.__dict__) + "\n")


@dataclass
class TeamGEPAConfig:
    iterations: int = 8
    minibatch: int = 3
    workers: int = 3
    seed: int = 0
    merge_prob: float = 0.15
    max_merges: int = 2
    blame: bool = True  # False -> vanilla GEPA round-robin over the roles that ran
    reflection_effort: str | None = "medium"
    max_playbook: int = 12
    evaluate_test: bool = True


@dataclass
class TeamCandidate:
    genome: TeamGenome
    scores: dict[str, float]
    costs: dict[str, float]
    iteration: int
    operator: str
    blamed: str = ""

    @property
    def id(self) -> str:
        return self.genome.id

    @property
    def mean(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0

    @property
    def mean_cost(self) -> float:
        return sum(self.costs.values()) / len(self.costs) if self.costs else 0.0


REFLECT_SYSTEM = """You improve a team of AI agents that maintain an open-source repository. The team's behaviour is
fully determined by each role's system prompt, a playbook of lessons, and a step limit; model weights never change.
Roles: orchestrator (coordinates, delegates to specialists, writes the final answer), triage, ci_triage, review,
security. You receive the current configuration of the roles involved, then episodes replayed from the
repository's real history: the event, the team's hierarchical trajectory, the score, and feedback describing what
actually happened later (ground truth).

1. Diagnose the systematic failure and decide which ONE role is responsible (blame). Consider coordination
   failures (orchestrator briefed badly, ignored or failed to verify a specialist, wrong delegation) separately
   from specialist failures (wrong investigation, wrong judgement, wrong output).
2. Propose an edit to that role only: playbook lessons to add/remove (concise, general, actionable - never
   mention specific issue/PR numbers or memorise answers) and/or a full rewrite of its system prompt.
   A rewritten prompt MUST keep the placeholders {project}, {project_context}, {tools}, {playbook} and the
   output-contract placeholder(s) it had. Return "" to keep the prompt.
3. Optionally adjust its step limit (0 = keep) if episodes ran out of steps or wasted many.
Return JSON only."""


def _reflect_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "blamed_role": {"type": "string", "enum": list(ROLES)},
            "diagnosis": {"type": "string"},
            "playbook_add": {"type": "array", "items": {"type": "string"}},
            "playbook_remove": {"type": "array", "items": {"type": "integer"}},
            "system_template": {"type": "string"},
            "step_limit": {"type": "integer"},
        },
        "required": ["blamed_role", "diagnosis", "playbook_add", "playbook_remove", "system_template", "step_limit"],
        "additionalProperties": False,
    }


REQUIRED_PLACEHOLDERS = {
    "orchestrator": ("{project}", "{project_context}", "{tools}", "{contracts}", "{playbook}"),
    "triage": ("{project}", "{project_context}", "{tools}", "{triage_contract}", "{playbook}"),
    "ci_triage": ("{project}", "{project_context}", "{tools}", "{ci_contract}", "{playbook}"),
    "review": ("{project}", "{project_context}", "{tools}", "{review_contract}", "{playbook}"),
    "security": ("{project}", "{project_context}", "{tools}", "{security_contract}", "{playbook}"),
}


def repair_template(role: str, template: str) -> str:
    """Re-append any required placeholder an LLM rewrite dropped (tools, contract, playbook...)."""
    missing = [p for p in REQUIRED_PLACEHOLDERS[role] if p not in template]
    if not missing:
        return template
    extra = {"{project}": "Project: {project}", "{project_context}": "{project_context}", "{tools}": "{tools}",
             "{playbook}": "{playbook}"}
    tail = "\n\n".join(extra.get(p, "## Output contract\n" + p) for p in missing)
    return template.rstrip() + "\n\n" + tail


class TeamGEPA:
    def __init__(self, seed: TeamGenome, train: list, val: list, test: list, run_episode: EpisodeFn,
                 reflection_llm: LLM, *, entry_role: str = "orchestrator", config: TeamGEPAConfig | None = None,
                 cache: EpisodeCache | None = None, budget_exhausted: Callable[[], bool] = lambda: False,
                 log: Callable[[str], None] = print, out_dir: str | Path | None = None):
        if not train or not val:
            raise ValueError("Team-GEPA needs train and val tasks")
        self.seed = seed
        self.train, self.val, self.test = train, val, test
        self.run_fn = run_episode
        self.reflection_llm = reflection_llm
        self.entry_role = entry_role
        self.cfg = config or TeamGEPAConfig()
        self.cache = cache or EpisodeCache()
        self.budget_exhausted = budget_exhausted
        self.log = log
        self.rng = random.Random(self.cfg.seed)
        self.out_dir = Path(out_dir) if out_dir else None
        self._batch: list = []
        self._rr: dict[str, int] = {}
        self.genomes: dict[str, TeamGenome] = {seed.id: seed}
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ evaluation
    def episode(self, genome: TeamGenome, task) -> EpisodeOutcome:
        hit = self.cache.get(task.id, genome)
        if hit is not None:
            return hit
        out = self.run_fn(genome, task)
        self.cache.put(out)
        return out

    def evaluate(self, genome: TeamGenome, tasks: list) -> list[EpisodeOutcome]:
        if self.cfg.workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(self.cfg.workers) as pool:
                return list(pool.map(lambda t: self.episode(genome, t), tasks))
        return [self.episode(genome, t) for t in tasks]

    def _minibatch(self) -> list:
        n = min(self.cfg.minibatch, len(self.train))
        if len(self._batch) < n:
            order = list(self.train)
            self.rng.shuffle(order)
            self._batch.extend(order)
        batch, self._batch = self._batch[:n], self._batch[n:]
        return batch

    # ------------------------------------------------------------------ selection
    @staticmethod
    def dominates(a: TeamCandidate, b: TeamCandidate) -> bool:
        ge = all(a.scores[k] >= b.scores[k] for k in a.scores) and a.mean_cost <= b.mean_cost
        gt = any(a.scores[k] > b.scores[k] for k in a.scores) or a.mean_cost < b.mean_cost
        return ge and gt

    def front(self, cands: list[TeamCandidate]) -> dict[str, int]:
        nondom = [c for c in cands if not any(self.dominates(o, c) for o in cands if o is not c)]
        freq: dict[str, int] = {c.id: 0 for c in nondom}
        for k in cands[0].scores:
            best = max(c.scores[k] for c in cands)
            for c in nondom:
                if c.scores[k] >= best - 1e-9:
                    freq[c.id] += 1
        return {k: max(v, 1) for k, v in freq.items()}  # non-dominated on cost alone still gets a chance

    def select(self, cands: list[TeamCandidate]) -> TeamCandidate:
        freq = self.front(cands)
        by_id = {c.id: c for c in cands}
        ids = sorted(freq)
        return by_id[self.rng.choices(ids, weights=[freq[i] for i in ids], k=1)[0]]

    # ------------------------------------------------------------------ mutation
    def _config_text(self, genome: TeamGenome, roles: set[str]) -> str:
        parts = []
        for r in ROLES:
            if r not in roles:
                continue
            g = genome.roles[r]
            pb = "\n".join(f"  [{i}] {p}" for i, p in enumerate(g.playbook)) or "  (empty)"
            parts.append(f"### role: {r} (step_limit {g.step_limit})\nsystem prompt template:\n{g.system_template}\n"
                         f"playbook:\n{pb}")
        return "\n\n".join(parts)

    def reflect(self, parent: TeamGenome, outcomes: list[EpisodeOutcome], tasks: list) -> tuple[TeamGenome | None, str]:
        involved = set().union(*(set(o.roles_used) for o in outcomes)) or {self.entry_role}
        allowed = sorted(involved)
        if not self.cfg.blame:  # vanilla GEPA: round-robin over the roles that ran
            k = self._rr.get(parent.id, 0)
            self._rr[parent.id] = k + 1
            allowed = [allowed[k % len(allowed)]]
        blocks = []
        for o, t in zip(outcomes, tasks):
            blocks.append(f"## Episode {t.id} - score {o.score:.2f} - cost ${o.cost:.3f} - {o.exit_status}\n"
                          f"Event: {t.event}\nFeedback (what actually happened): {o.feedback}\n"
                          f"Trajectory:\n{o.trace_text[-9000:]}")
        user = (f"# Current configuration of the roles involved\n{self._config_text(parent, involved)}\n\n"
                f"# You may only blame and edit: {', '.join(allowed)}\n\n# Episodes\n" + "\n\n".join(blocks))
        try:
            d = self.reflection_llm.complete_json(REFLECT_SYSTEM, user, _reflect_schema(), name="team_reflection",
                                                  effort=self.cfg.reflection_effort)
        except (LLMError, BudgetExceeded) as e:
            return None, f"reflection failed: {e}"
        role = d["blamed_role"] if d["blamed_role"] in allowed else allowed[0]
        child = parent.clone()
        g = child.roles[role]
        drop = {i for i in d.get("playbook_remove") or [] if isinstance(i, int)}
        g.playbook = [p for i, p in enumerate(g.playbook) if i not in drop]
        for p in d.get("playbook_add") or []:
            p = str(p).strip()
            if p and p not in g.playbook:
                g.playbook.append(p[:500])
        g.playbook = g.playbook[-self.cfg.max_playbook:]
        tpl = (d.get("system_template") or "").strip()
        if tpl:
            g.system_template = repair_template(role, tpl)[:12000]
        if isinstance(d.get("step_limit"), int) and 5 <= d["step_limit"] <= 60:
            g.step_limit = d["step_limit"]
        child.origin = f"reflect:{role}"
        child.notes = f"blamed {role}: {d.get('diagnosis', '')[:300]}"
        if child.id == parent.id:
            return None, "reflection produced no change"
        return child, role

    def merge(self, a: TeamCandidate, b: TeamCandidate) -> TeamGenome | None:
        """Role-wise system-aware merge: per role take the version that changed from the common ancestor."""
        anc_id = next((p for p in a.genome.parents if p in b.genome.parents), None)
        anc = self.genomes.get(anc_id) if anc_id else None
        if anc is None:
            return None
        child = a.genome.clone()
        child.parents = [a.id, b.id]
        picks = []
        for r in ROLES:
            ha, hb, h0 = (x.roles[r].content_hash() for x in (a.genome, b.genome, anc))
            if ha == h0 and hb != h0:
                child.roles[r] = b.genome.roles[r]
                picks.append(f"{r}<-b")
        if not picks or child.id in (a.id, b.id):
            return None
        child.origin, child.notes = "merge", ", ".join(picks)
        return child

    # ------------------------------------------------------------------ main loop
    def _cand(self, genome: TeamGenome, outcomes: list[EpisodeOutcome], it: int, op: str, blamed: str = "") -> TeamCandidate:
        return TeamCandidate(genome, {o.task_id: o.score for o in outcomes}, {o.task_id: o.cost for o in outcomes},
                             it, op, blamed)

    def run(self) -> dict[str, Any]:
        t0 = time.time()
        run_id = uuid.uuid4().hex[:10]
        self.log(f"[team-gepa] run {run_id}: train={len(self.train)} val={len(self.val)} test={len(self.test)} "
                 f"entry={self.entry_role} blame={self.cfg.blame}")
        seed = self._cand(self.seed, self.evaluate(self.seed, self.val), 0, "seed")
        cands = [seed]
        self.log(f"[team-gepa] seed val={seed.mean:.3f} cost/episode=${seed.mean_cost:.3f}")
        stop = "iterations"
        merges = 0
        for it in range(1, self.cfg.iterations + 1):
            if self.budget_exhausted():
                stop = "budget"
                break
            rec: dict[str, Any] = {"iter": it}
            pair = None
            if len(cands) > 2 and merges < self.cfg.max_merges and self.rng.random() < self.cfg.merge_prob:
                a, b = self.rng.sample(cands[1:], 2)
                child = self.merge(a, b)
                if child is not None and child.id not in self.genomes:
                    pair, merges = (a, b), merges + 1
            if pair is not None:
                batch = self._minibatch()
                pa, pb = (self.evaluate(x.genome, batch) for x in pair)
                parent_sum = max(sum(o.score for o in pa), sum(o.score for o in pb))
                op, blamed, parent = "merge", "", pair[0]
            else:
                parent = self.select(cands)
                batch = self._minibatch()
                p_out = self.evaluate(parent.genome, batch)
                parent_sum = sum(o.score for o in p_out)
                if parent_sum >= len(batch) - 1e-9:
                    rec.update(op="reflect", status="skipped (perfect minibatch)")
                    self.history.append(rec)
                    self.log(f"[team-gepa] it={it} skipped: perfect minibatch")
                    continue
                child, blamed = self.reflect(parent.genome, p_out, batch)
                op = "reflect"
                if child is None or child.id in self.genomes:
                    rec.update(op=op, status=f"skipped ({blamed if child is None else 'duplicate genome'})")
                    self.history.append(rec)
                    self.log(f"[team-gepa] it={it} {rec['status']}")
                    continue
            self.genomes[child.id] = child
            c_out = self.evaluate(child, batch)
            child_sum = sum(o.score for o in c_out)
            accept = child_sum > parent_sum + 1e-9 if op == "reflect" else child_sum >= parent_sum
            rec.update(op=op, blamed=blamed, parent=parent.id, child=child.id, mb_parent=round(parent_sum, 3),
                       mb_child=round(child_sum, 3), notes=child.notes)
            if accept:
                cand = self._cand(child, self.evaluate(child, self.val), it, op, blamed)
                cands.append(cand)
                rec.update(status="accepted", val=round(cand.mean, 4), cost=round(cand.mean_cost, 4))
                self.log(f"[team-gepa] it={it} {op}({blamed}) ACCEPT minibatch {parent_sum:.2f}->{child_sum:.2f} "
                         f"val={cand.mean:.3f} ${cand.mean_cost:.3f}/ep | {child.notes[:120]}")
            else:
                rec["status"] = "rejected"
                self.log(f"[team-gepa] it={it} {op}({blamed}) reject minibatch {parent_sum:.2f}->{child_sum:.2f}")
            self.history.append(rec)
            if self.out_dir:
                self._save(run_id, cands)

        best = max(cands, key=lambda c: (round(c.mean, 9), -c.mean_cost, -c.iteration))
        test = {}
        if self.cfg.evaluate_test and self.test and not self.budget_exhausted():
            s_test = self.evaluate(self.seed, self.test)
            b_test = s_test if best is seed else self.evaluate(best.genome, self.test)
            test = {"seed": round(sum(o.score for o in s_test) / len(s_test), 4),
                    "best": round(sum(o.score for o in b_test) / len(b_test), 4)}
        result = {
            "run_id": run_id, "stop": stop, "elapsed_s": round(time.time() - t0, 1),
            "seed": {"id": seed.id, "val": round(seed.mean, 4), "cost": round(seed.mean_cost, 4)},
            "best": {"id": best.id, "val": round(best.mean, 4), "cost": round(best.mean_cost, 4),
                     "origin": best.genome.origin, "notes": best.genome.notes},
            "test": test, "candidates": [{"id": c.id, "op": c.operator, "blamed": c.blamed, "iter": c.iteration,
                                          "val": round(c.mean, 4), "cost": round(c.mean_cost, 4)} for c in cands],
            "history": self.history, "cache": {"hits": self.cache.hits, "misses": self.cache.misses},
        }
        self.log(f"[team-gepa] done ({stop}): val {seed.mean:.3f}->{best.mean:.3f}"
                 + (f", test {test['seed']:.3f}->{test['best']:.3f}" if test else "")
                 + f"; cache hits {self.cache.hits}/{self.cache.hits + self.cache.misses}")
        self.best_genome = best.genome
        if self.out_dir:
            self._save(run_id, cands, result)
        return result

    def _save(self, run_id: str, cands: list[TeamCandidate], result: dict | None = None) -> None:
        assert self.out_dir is not None
        d = self.out_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        for c in cands:
            p = d / f"genome-{c.id}.json"
            if not p.exists():
                p.write_text(json.dumps(c.genome.to_dict(), indent=1), encoding="utf-8")
        if result is not None:
            (d / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
