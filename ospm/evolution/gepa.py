"""GEPA-style non-parametric evolution of agent genomes.

Algorithm (after Agrawal et al., 2025, "GEPA: Reflective Prompt Evolution Can Outperform RL"):

    candidates <- [seed];  S[seed] <- per-instance scores on the Pareto set (val split)
    repeat until the iteration / metric-call / LLM budget is spent:
        with probability p_merge (and a mergeable pair exists):
            child <- system-aware merge of two Pareto candidates sharing an ancestor
        else:
            parent <- sample from the per-instance Pareto frontier, weighted by how many
                      instances it is (one of) the best on, after removing dominated candidates
            module <- next module of `parent` (round-robin: reflection, induction, params, exemplars)
            M      <- next minibatch of the train split
            traces <- run parent on M, attach feedback (what was wrong and why)
            child  <- reflective mutation of `module` given the traces
        if child improves on the parent(s) on the minibatch:      # cheap gate (strict for mutations)
            evaluate child on the full Pareto set and add it to the candidate pool
    return the candidate with the best mean Pareto-set score (ties -> fewer rules)

Speed-ups in the spirit of FlashEvolve: content-addressed genomes + a (genome, example) decision cache so no
genome is ever re-run on the same example, an LLM response cache in persistent memory, parallel evaluation
when an LLM is in the loop, and the minibatch gate that rejects most bad children before full evaluation.
"""

from __future__ import annotations

import copy
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agents.base import Agent
from ..benchmarks import Task
from ..genome import Genome
from ..llm import LLM
from ..memory import Memory
from ..models import Decision
from .metrics import EvalResult, evaluate
from .operators import LLMReflector, RuleInducer, Trace, make_trace, merge, mutate_exemplars, mutate_params


class EvaluationAborted(RuntimeError):
    """The policy LLM failed during an evaluation, so the scores would mix LLM and fallback decisions."""


@dataclass
class GEPAConfig:
    max_iterations: int = 40
    max_metric_calls: int | None = None  # budget of agent invocations (cache hits are free)
    minibatch_size: int = 8
    merge_prob: float = 0.2
    max_merges: int = 5
    seed: int = 0
    workers: int = 8
    reflection_effort: str | None = "high"
    policy_effort: str | None = "medium"
    induction_min_pos: int = 3
    induction_min_lift: float = 1.5
    # staged evaluation: a child that passes the minibatch gate must also not regress on the full train split
    # before it is scored on the Pareto set. None -> on when the policy is heuristic (train eval is free).
    train_gate: bool | None = None
    operators: tuple[str, ...] | None = None  # None -> chosen from the available LLMs


@dataclass
class Candidate:
    genome: Genome
    val: dict[str, float]
    operator: str
    iteration: int
    parents: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.genome.id

    train: float | None = None

    @property
    def mean(self) -> float:
        return sum(self.val.values()) / len(self.val) if self.val else 0.0


@dataclass
class GEPAResult:
    run_id: str
    task: str
    seed: Candidate
    best: Candidate
    candidates: list[Candidate]
    history: list[dict[str, Any]]
    metric_calls: int
    stop_reason: str
    elapsed_s: float
    test: dict[str, Any] = field(default_factory=dict)
    llm_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def improved(self) -> bool:
        return self.best.mean > self.seed.mean + 1e-9

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "stop_reason": self.stop_reason,
            "iterations": len(self.history),
            "metric_calls": self.metric_calls,
            "candidates": len(self.candidates),
            "accepted": [{"id": c.id, "op": c.operator, "iter": c.iteration, "val": round(c.mean, 4),
                          "train": None if c.train is None else round(c.train, 4),
                          "rules": len(c.genome.rules)} for c in self.candidates],
            "seed_val": round(self.seed.mean, 4),
            "best_val": round(self.best.mean, 4),
            "best_id": self.best.id,
            "test": self.test,
            "elapsed_s": round(self.elapsed_s, 1),
            "llm_usage": self.llm_usage,
        }


class GEPA:
    def __init__(
        self,
        task: Task,
        seed_genome: Genome | None = None,
        *,
        policy_llm: LLM | None = None,
        reflection_llm: LLM | None = None,
        config: GEPAConfig | None = None,
        memory: Memory | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.task = task
        self.cfg = config or GEPAConfig()
        self.rng = random.Random(self.cfg.seed)
        self.policy_llm = policy_llm
        self.reflection_llm = reflection_llm
        self.memory = memory
        self.log = log or (lambda msg: None)
        self.train = task.subset("train")
        self.val = task.subset("val")
        self.test = task.subset("test")
        if not self.train or not self.val:
            raise ValueError(f"task {task.name!r} needs labelled examples in both train and val splits "
                             f"(have {len(self.train)} train / {len(self.val)} val; ~5+ feedback cases are enough)")
        self._proto = task.make_agent(seed_genome, llm=policy_llm, effort=self.cfg.policy_effort)
        self.seed_genome = self._proto.genome
        self.cache: dict[tuple[str, str], Decision] = {}
        self.parents_of: dict[str, list[str]] = {self.seed_genome.id: []}
        self.genomes: dict[str, Genome] = {self.seed_genome.id: self.seed_genome}
        self._next_op: dict[str, int] = {}
        self._batch_order: list = []
        self._merge_pairs: set[tuple[str, str]] = set()
        self.inducer = RuleInducer(
            task.agent_cls,
            {e.id: self._proto.item_text(e.item) for e in self.train},
            {e.id: dict(e.labels) for e in self.train},
            min_pos=self.cfg.induction_min_pos,
            min_lift=self.cfg.induction_min_lift,
        )
        self.reflector = LLMReflector(reflection_llm, task.agent_cls, self.cfg.reflection_effort) if reflection_llm else None
        if self.cfg.operators:
            self.operators = tuple(self.cfg.operators)
        else:
            ops = ["reflection"] if self.reflector else []
            ops += ["induction", "params"]
            if policy_llm is not None:
                ops.append("exemplars")
            self.operators = tuple(ops)
        self.train_gate = self.cfg.train_gate if self.cfg.train_gate is not None else policy_llm is None
        unknown = set(self.operators) - {"reflection", "induction", "params", "exemplars"}
        if unknown or ("reflection" in self.operators and not self.reflector):
            raise ValueError(f"invalid operators {self.operators} (reflection requires a reflection LLM)")

    # ------------------------------------------------------------------ evaluation
    def agent(self, genome: Genome) -> Agent:
        a = copy.copy(self._proto)
        a.genome = genome
        return a

    def evaluate(self, genome: Genome, examples: list) -> EvalResult:
        res = evaluate(self.agent(genome), examples, workers=self.cfg.workers, cache=self.cache)
        if self.policy_llm is not None and res.llm_errors:
            err = next(d.extras["llm_error"] for d in res.predictions.values() if "llm_error" in d.extras)
            raise EvaluationAborted(f"{res.llm_errors} policy-LLM failure(s), e.g. {err[:200]}")
        return res

    @property
    def metric_calls(self) -> int:
        return len(self.cache)

    def _budget_left(self) -> bool:
        return self.cfg.max_metric_calls is None or self.metric_calls < self.cfg.max_metric_calls

    def _llm_exhausted(self) -> bool:
        for llm in (self.policy_llm, self.reflection_llm):
            if llm is not None and llm.meter.budget_usd is not None and llm.meter.cost_usd >= llm.meter.budget_usd:
                return True
        return False

    def _minibatch(self) -> list:
        n = min(self.cfg.minibatch_size, len(self.train))
        if len(self._batch_order) < n:
            order = list(self.train)
            self.rng.shuffle(order)
            self._batch_order.extend(order)
        batch, self._batch_order = self._batch_order[:n], self._batch_order[n:]
        return batch

    # ------------------------------------------------------------------ Pareto selection
    @staticmethod
    def _dominates(a: Candidate, b: Candidate) -> bool:
        keys = a.val.keys()
        return all(a.val[k] >= b.val[k] for k in keys) and any(a.val[k] > b.val[k] for k in keys)

    def pareto_front(self, cands: list[Candidate]) -> dict[str, int]:
        """Non-dominated candidates -> number of val instances on which they achieve the best score."""
        nondom = [c for c in cands if not any(self._dominates(o, c) for o in cands if o is not c)]
        freq: Counter = Counter()
        for j in self.val:
            best = max(c.val[j.id] for c in cands)
            for c in nondom:
                if c.val[j.id] >= best - 1e-9:
                    freq[c.id] += 1
        return dict(freq)

    def select_parent(self, cands: list[Candidate]) -> Candidate:
        freq = self.pareto_front(cands)
        if not freq:
            return max(cands, key=lambda c: c.mean)
        by_id = {c.id: c for c in cands}
        ids = sorted(freq)
        return by_id[self.rng.choices(ids, weights=[freq[i] for i in ids], k=1)[0]]

    # ------------------------------------------------------------------ lineage helpers
    def ancestors(self, gid: str) -> list[str]:
        """All ancestors of `gid`, nearest first."""
        out: list[str] = []
        frontier = list(self.parents_of.get(gid, []))
        while frontier:
            nxt = []
            for p in frontier:
                if p not in out:
                    out.append(p)
                    nxt.extend(self.parents_of.get(p, []))
            frontier = nxt
        return out

    def _merge_candidates(self, cands: list[Candidate]) -> tuple[Candidate, Candidate, Genome] | None:
        front = set(self.pareto_front(cands))
        pool = [c for c in cands if c.id in front]
        pairs = []
        for i, a in enumerate(pool):
            anc_a = self.ancestors(a.id)
            for b in pool[i + 1:]:
                key = tuple(sorted((a.id, b.id)))
                if key in self._merge_pairs:
                    continue
                anc_b = self.ancestors(b.id)
                if a.id in anc_b or b.id in anc_a:
                    continue
                common = [x for x in anc_a if x in anc_b]
                if common:
                    pairs.append((a, b, self.genomes[common[0]]))
        if not pairs:
            return None
        return self.rng.choice(pairs)

    # ------------------------------------------------------------------ main loop
    def run(self) -> GEPAResult:
        t0 = time.time()
        run_id = uuid.uuid4().hex[:12]
        meters_at_start = self._meter_snapshots()
        if self.memory:
            self.memory.start_run(run_id, self.task.name, {**self.cfg.__dict__, "operators": self.operators,
                                                           "seed_genome": self.seed_genome.id})
            self.memory.save_genome(self.seed_genome)
        seed_eval = self.evaluate(self.seed_genome, self.val)  # an abort here propagates: nothing to compare to
        seed = Candidate(self.seed_genome, seed_eval.per_instance, "seed", 0)
        if self.train_gate:
            seed.train = self.evaluate(self.seed_genome, self.train).score
        cands = [seed]
        history: list[dict[str, Any]] = []
        stop = "max_iterations"
        self.log(f"[gepa] task={self.task.name} operators={','.join(self.operators)} "
                 f"train={len(self.train)} val={len(self.val)} seed val={seed.mean:.4f}")
        if self.memory:
            self.memory.log_lineage(run_id, self.seed_genome, "seed", None, seed.mean, "accepted")

        for it in range(1, self.cfg.max_iterations + 1):
            if not self._budget_left():
                stop = "max_metric_calls"
                break
            if self._llm_exhausted():
                stop = "llm_budget"
                break
            rec: dict[str, Any] = {"iter": it}
            try:
                self._iteration(it, rec, cands, run_id)
            except EvaluationAborted as e:
                rec["status"] = "aborted"
                history.append(rec)
                stop = f"aborted ({e})"
                self.log(f"[gepa] it={it:>3} {stop}")
                break
            history.append(rec)

        # best on the Pareto (val) set; ties -> better train score, then fewer rules, then earlier
        best = max(cands, key=lambda c: (round(c.mean, 9), round(c.train or 0.0, 9), -len(c.genome.rules),
                                         -c.iteration))
        test: dict[str, Any] = {}
        if self.test:
            try:
                seed_test = self.evaluate(self.seed_genome, self.test)
                best_test = seed_test if best is seed else self.evaluate(best.genome, self.test)
                test = {"seed": round(seed_test.score, 4), "best": round(best_test.score, 4)}
            except EvaluationAborted as e:
                test = {"error": str(e)}
        usage = self._usage_since(meters_at_start)
        result = GEPAResult(run_id, self.task.name, seed, best, cands, history, self.metric_calls, stop,
                            time.time() - t0, test, usage)
        self.log(f"[gepa] done: {stop}; best {best.id} val {seed.mean:.4f}->{best.mean:.4f}"
                 + (f"; test {test['seed']:.4f}->{test['best']:.4f}" if "seed" in test else ""))
        if self.memory:
            self.memory.finish_run(run_id, result.summary())
        return result

    def _iteration(self, it: int, rec: dict[str, Any], cands: list[Candidate], run_id: str) -> None:
        """One GEPA step (merge or reflective mutation). Fills `rec`; may append to `cands`."""
        mergeable = (len(self._merge_pairs) < self.cfg.max_merges and len(cands) >= 2
                     and self.rng.random() < self.cfg.merge_prob)
        pair = self._merge_candidates(cands) if mergeable else None
        if pair is not None:
            a, b, ancestor = pair
            self._merge_pairs.add(tuple(sorted((a.id, b.id))))
            child = merge(a.genome, b.genome, ancestor, a.mean >= b.mean)
            rec.update(op="merge", parent=f"{a.id}+{b.id}")
            if child is None or child.id in self.genomes:
                rec["status"] = "skipped (no new genome)"
                self.log(f"[gepa] it={it:>3} merge      {rec['status']}")
                return
            batch = self._minibatch()
            pa, pb, pc = (self.evaluate(g, batch) for g in (a.genome, b.genome, child))
            parent_sum = max(sum(pa.per_instance.values()), sum(pb.per_instance.values()))
            child_sum = sum(pc.per_instance.values())
            accept = child_sum >= parent_sum  # GEPA merge: at least as good as the better parent
        else:
            parent = self.select_parent(cands)
            op = self.operators[self._next_op.get(parent.id, 0) % len(self.operators)]
            self._next_op[parent.id] = self._next_op.get(parent.id, 0) + 1
            batch = self._minibatch()
            p_eval = self.evaluate(parent.genome, batch)
            parent_sum = sum(p_eval.per_instance.values())
            rec.update(op=op, parent=parent.id)
            if parent_sum >= len(batch) - 1e-9 and op != "params":
                rec["status"] = "skipped (perfect minibatch)"
                self.log(f"[gepa] it={it:>3} {op:<10} {rec['status']}")
                return
            traces = [make_trace(self.agent(parent.genome), ex, p_eval.predictions[ex.id],
                                 p_eval.per_instance[ex.id]) for ex in batch]
            child = self._mutate(op, parent.genome, traces)
            if child is None or child.id in self.genomes:
                reason = "no new genome"
                if op == "reflection" and self.reflector and self.reflector.last_error:
                    reason = f"reflection failed: {self.reflector.last_error[:160]}"
                rec["status"] = f"skipped ({reason})"
                self.log(f"[gepa] it={it:>3} {op:<10} {rec['status']}")
                return
            c_eval = self.evaluate(child, batch)
            child_sum = sum(c_eval.per_instance.values())
            accept = child_sum > parent_sum + 1e-9  # GEPA: strict improvement on the minibatch
        child.generation = max(child.generation, 1)
        self.genomes[child.id] = child
        self.parents_of[child.id] = list(child.parents)
        rec.update(child=child.id, mb_parent=round(parent_sum, 3), mb_child=round(child_sum, 3), notes=child.notes)
        train_score = None
        if accept and self.train_gate:
            parent_train = max(c.train for c in cands if c.id in child.parents and c.train is not None)
            train_score = self.evaluate(child, self.train).score
            rec.update(train_parent=round(parent_train, 4), train_child=round(train_score, 4))
            if train_score < parent_train - 1e-9:
                accept = False
                rec["gate"] = "train"
        if accept:
            full = self.evaluate(child, self.val)
            cand = Candidate(child, full.per_instance, rec["op"], it, list(child.parents), train_score)
            cands.append(cand)
            rec.update(status="accepted", val=round(cand.mean, 4))
            self.log(f"[gepa] it={it:>3} {rec['op']:<10} ACCEPT {child.id} minibatch {parent_sum:.2f}->"
                     f"{child_sum:.2f} val={cand.mean:.4f} | {child.notes[:110]}")
            if self.memory:
                self.memory.save_genome(child)
        else:
            rec["status"] = "rejected" + (" (train regression)" if rec.get("gate") else "")
            self.log(f"[gepa] it={it:>3} {rec['op']:<10} {rec['status']} minibatch {parent_sum:.2f}->{child_sum:.2f}"
                     + (f" train {rec['train_parent']:.3f}->{rec['train_child']:.3f}" if rec.get("gate") else ""))
        if self.memory:
            self.memory.log_lineage(run_id, child, rec["op"], child_sum, rec.get("val"), rec["status"])

    def _llm_roles(self) -> dict[int, tuple[str, LLM]]:
        """Distinct meters -> (role label, llm). Policy and reflection often share one meter."""
        roles: dict[int, tuple[str, LLM]] = {}
        for role, llm in (("policy", self.policy_llm), ("reflection", self.reflection_llm)):
            if llm is None:
                continue
            key = id(llm.meter)
            roles[key] = (f"{roles[key][0]}+{role}", roles[key][1]) if key in roles else (role, llm)
        return roles

    def _meter_snapshots(self) -> dict[int, dict[str, Any]]:
        return {key: llm.meter.snapshot() for key, (_, llm) in self._llm_roles().items()}

    def _usage_since(self, start: dict[int, dict[str, Any]]) -> dict[str, Any]:
        """Per-run LLM usage (deltas of the shared meters)."""
        out: dict[str, Any] = {}
        for key, (role, llm) in self._llm_roles().items():
            now, before = llm.meter.snapshot(), start.get(key, {})
            delta = {k: round(now[k] - before.get(k, 0), 6) for k in
                     ("calls", "cache_hits", "errors", "input_tokens", "output_tokens", "cost_usd")}
            out[role] = {"model": llm.name, **delta}
        return out

    def _mutate(self, op: str, genome: Genome, traces: list[Trace]) -> Genome | None:
        if op == "reflection":
            assert self.reflector is not None
            return self.reflector.mutate(genome, traces)
        if op == "induction":
            return self.inducer.mutate(genome, traces, self.rng)
        if op == "params":
            return mutate_params(genome, self.rng)
        if op == "exemplars":
            return mutate_exemplars(genome, traces, {e.id: dict(e.labels) for e in self.train}, self.rng)
        raise ValueError(op)
