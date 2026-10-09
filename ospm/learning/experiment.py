"""Experiment: does learning validated, scoped interventions from earlier review corrections help on later PRs?

    python -m ospm.learning.experiment --data data/pytest --llm openai --train 40 --test 24 --budget 2

Learns chronologically on the newest `--train` concern tasks of the train split, then evaluates on the first
`--test` tasks of the (later) test split, for systems that see identical tasks (`--systems`):
  none          - the fixed review agent
  memory        - + ACE-style raw lesson memory built from the same training concerns
  interventions - + validated, scoped interventions
  tool          - + documentation of `hist reviews` (earlier maintainer comments on a file)
  skills        - + `tool` and a skill library evolved by post-mortem reflection (ospm/learning/postmortem.py)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..harness.models import make_chat_model
from ..llm import UsageMeter, make_llm
from ..replay.concerns import concern_metrics
from ..replay.engine import run_task, trace_name
from ..replay.semantic import Matcher, semantic_metrics, semantic_score
from ..replay.tasks import ReplayTask, load_tasks, task_output
from ..replay.workspace import ReplayRepo
from ..team.roles import seed_team
from .interventions import InterventionLearner, LearnConfig, save
from .postmortem import SkillLearner
from .skills import load as load_skills
from .skills import loaded_skills, skills_hash
from .skills import save as save_skills

SYSTEMS = ("none", "memory", "interventions", "tool", "skills")
# part of every cached episode's key: bump it when episode behaviour changes without a genome change
# (h3: agents get a warning before the step limit and one final turn to submit, in which plain text counts)
EPISODE_VERSION = "h3"
RATE_LIMIT_RETRIES = 3  # whole-episode reruns after an episode fails on a 429


class JsonlCache:
    """On-disk ResponseCache for the reflector/proposer LLM, so a resumed run replays the same proposals (and
    therefore hits the episode cache for their validation replays) instead of paying for new ones."""

    def __init__(self, path: Path):
        self.path, self.data, self.lock = path, {}, threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.data[d["k"]] = d["v"]

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as fh:
            self.data[key] = value
            fh.write(json.dumps({"k": key, "v": value}) + "\n")


class BudgetStop(Exception):
    pass


def make_runner(repo: ReplayRepo, model, meter: UsageMeter, cache_path: Path, trace_dir: Path,
                entry: str = "review", log=None, score_fn=None, genome=None):
    """run(task, ctx, skills=(), role_ctx=(), genome=None) -> (score, output), cached across processes.

    `genome` defaults to the runner's (the seed team). run.trace_of(...) gives an episode's trace (kept in memory
    for new episodes, read back from the trace file for cached ones) and run.cost_of(...) its USD cost (None for
    episodes cached before costs were recorded). `score_fn(task, output)` re-scores every episode, cached ones
    included (e.g. semantic concern matching)."""
    default = genome or seed_team()
    cache: dict[str, Any] = {}
    costs: dict[str, float] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cache[d["key"]] = (d["score"], d["output"])
                if "cost" in d:
                    costs[d["key"]] = d["cost"]
    traces: dict[str, dict] = {}
    locks: dict[str, threading.Lock] = {}
    guard = threading.Lock()

    def key_of(task: ReplayTask, ctx: str, skills: tuple, role_ctx: tuple, g) -> str:
        # invocation-aware (as in Team-GEPA): a specialist's episode depends only on that role's genome
        who = f"{entry}:{g.roles[entry].content_hash()}" if entry in g.roles and entry != "orchestrator" else g.id
        key = f"{task.id}|{who}|{EPISODE_VERSION}|{hashlib.sha256(ctx.encode()).hexdigest()[:12]}"
        if skills:  # keys of skill-free, single-agent episodes are unchanged
            key += f"|{skills_hash(list(skills))}"
        if role_ctx:
            key += "|r" + hashlib.sha256(json.dumps(sorted(role_ctx)).encode()).hexdigest()[:12]
        return key

    def run(task: ReplayTask, ctx: str = "", skills: tuple = (), role_ctx: tuple = (), genome=None):
        """role_ctx: ((role, text), ...) appended to the task of every specialist of that role the entry spawns."""
        g = genome or default
        key = key_of(task, ctx, skills, role_ctx, g)
        # identical (task, context) pairs requested concurrently - e.g. two systems that inject nothing - must
        # share one episode; separate runs would differ by sampling noise and show up as a spurious effect
        with guard:
            lock = locks.setdefault(key, threading.Lock())
        with lock:
            score, out = _run(task, ctx, skills, role_ctx, g, key)
        return (score_fn(task, out), out) if score_fn else (score, out)

    def _run(task: ReplayTask, ctx: str, skills: tuple, role_ctx: tuple, g, key: str):
        if key in cache:
            return cache[key]
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            if meter.budget_usd is not None and meter.cost_usd >= 0.95 * meter.budget_usd:
                raise BudgetStop(f"budget ${meter.budget_usd} reached")
            ep, score, _ = run_task(repo, g, model, task, entry_role=entry, extra_context=ctx,
                                    trace_dir=trace_dir, log=log, skills=list(skills),
                                    role_context=dict(role_ctx) or None)
            status = ep.exit_status
            if not (status.startswith("ModelError") and ("429" in status or "RateLimit" in status)) \
                    or attempt == RATE_LIMIT_RETRIES:
                break
            # a sustained tokens-per-minute burst outlasts the SDK's retries: wait it out, rerun the episode
            if log:
                log(f"[exp] rate-limited on {task.id}; retrying the episode in {60 * (attempt + 1)}s")
            time.sleep(60 * (attempt + 1))
        out = task_output(task, ep.output, entry)
        if ep.exit_status.startswith("ModelError"):
            raise BudgetStop(f"model error: {ep.exit_status}")  # never record a failed call as a score
        cache[key] = (score, out)
        costs[key] = round(ep.cost, 6)
        traces[key] = ep.trace.to_dict()
        with guard, cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "score": score, "output": out, "cost": costs[key]}) + "\n")
        return cache[key]

    def trace_of(task: ReplayTask, ctx: str = "", skills: tuple = (), role_ctx: tuple = (),
                 genome=None) -> dict | None:
        g = genome or default
        key = key_of(task, ctx, skills, role_ctx, g)
        if key not in traces:
            path = trace_dir / trace_name(task.id, g.id, entry, ctx, skills, dict(role_ctx) or None)
            if not path.exists():
                return None
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                traces[key] = json.load(fh)["trace"]
        return traces[key]

    def cost_of(task: ReplayTask, ctx: str = "", skills: tuple = (), role_ctx: tuple = (),
                genome=None) -> float | None:
        return costs.get(key_of(task, ctx, skills, role_ctx, genome or default))

    run.trace_of, run.cost_of = trace_of, cost_of
    return run


def pick(tasks: list[ReplayTask], n: int, newest: bool) -> list[ReplayTask]:
    """Balanced positives/negatives, chronological; `newest` takes the most recent ones (closest to test)."""
    pos = sorted((t for t in tasks if t.truth["positive"]), key=lambda t: t.as_of, reverse=newest)
    neg = sorted((t for t in tasks if not t.truth["positive"]), key=lambda t: t.as_of, reverse=newest)
    out = pos[: (n + 1) // 2] + neg[: n // 2]
    return sorted(out, key=lambda t: (t.as_of, t.id))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"n": len(rows), "mean_score": mean([r["score"] for r in rows]),
            "pos_recall": mean([r["recall"] or 0.0 for r in pos]), "pos_n": len(pos),
            "pos_mean_comments": mean([r["n_comments"] for r in pos]),
            "neg_silent_rate": mean([1.0 if r["n_comments"] == 0 else 0.0 for r in neg]), "neg_n": len(neg),
            "neg_mean_comments": mean([r["n_comments"] for r in neg]),
            **({"pos_sem_recall": mean([r["sem_recall"] for r in pos]),
                "pos_soft_recall": mean([r["soft_recall"] for r in pos])} if pos and "sem_recall" in pos[0] else {})}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--llm", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--test", type=int, default=24)
    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="episodes run in parallel")
    p.add_argument("--systems", default="none,memory,interventions", help=f"comma-separated subset of {SYSTEMS}")
    p.add_argument("--match", default="semantic", choices=("semantic", "line"),
                   help="how a comment matches a maintainer concern: Jev judge (ospm/replay/semantic.py) or ±5 lines")
    p.add_argument("--entry", default="review", choices=("review", "orchestrator"),
                   help="who gets the task: a single reviewer, or the orchestrator with its team")
    p.add_argument("--library", default=None,
                   help="evaluate this saved skill library (a *_skills.json) instead of learning one")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    systems = [s for s in SYSTEMS if s in a.systems.split(",")]
    if "none" not in systems or len(systems) < 2:
        p.error("--systems needs 'none' plus at least one other system")
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    data = Path(a.data)
    repo = ReplayRepo(data)
    tasks = [t for t in load_tasks(data / "tasks.jsonl") if t.kind == "concern"]
    train = pick([t for t in tasks if t.split == "train"], a.train, newest=True)
    test = pick([t for t in tasks if t.split == "test"], a.test, newest=False)
    log(f"[exp] {repo.store.meta('repo')}: train {len(train)} ({sum(t.truth['positive'] for t in train)} pos), "
        f"test {len(test)} ({sum(t.truth['positive'] for t in test)} pos), systems {','.join(systems)}, "
        f"entry {a.entry}")
    meter = UsageMeter(a.budget)
    model = make_chat_model(a.llm, a.model, meter=meter)
    reflect = make_llm(a.llm, a.model, meter=meter, cache=JsonlCache(data / "reflect_cache.jsonl"))
    matcher = Matcher(data / "semantic_cache.jsonl") if a.match == "semantic" else None
    score_fn = (lambda t, out: semantic_score(matcher, t, out)) if matcher else None
    run = make_runner(repo, model, meter, data / "learning_cache.jsonl", data / "traces", entry=a.entry,
                      score_fn=score_fn)
    # candidates are validated on earlier history too (never on the test split; outcomes must be known by then)
    past = [t for t in tasks if t.split != "test"]
    cfg = LearnConfig(batch=a.batch, workers=a.workers)
    learner = InterventionLearner(repo, run, reflect, cfg, log=log, past=past)
    skiller = SkillLearner(repo, run, reflect, replace(cfg, max_candidates_per_step=4), log=log, past=past,
                           trace_of=run.trace_of, entry=a.entry)
    learner.matcher = skiller.matcher = matcher
    t0 = time.time()
    stopped = None
    try:
        if "interventions" in systems:
            learner.learn(train)  # also fills the memory baseline
        else:
            learner.remember(train)
        if "skills" in systems and a.library:
            skiller.skills = load_skills(Path(a.library))
        elif "skills" in systems:
            skiller.learn(train)
    except BudgetStop as e:
        stopped = f"during learning: {e}"

    def serve(t: ReplayTask, system: str) -> tuple[str, tuple, tuple]:  # (ctx, skills, role_ctx)
        return skiller.serve(t, system) if system in ("tool", "skills") else (learner.serve_context(t, system), (), ())

    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in systems}
    loaded: list[int] = []
    if stopped is None:
        jobs = [(t, system, *serve(t, system)) for t in test for system in systems]
        try:
            outs = learner._run_many([(t, *job) for t, _, *job in jobs])
            for (t, system, *job), (score, out) in zip(jobs, outs):
                rows[system].append({"task": t.id, "score": score, **concern_metrics(t, out),
                                     **(semantic_metrics(matcher, t, out) if matcher else {})})
                if system == "skills" and job[1]:
                    loaded.append(len(loaded_skills(run.trace_of(t, *job) or {})))
        except BudgetStop as e:
            stopped = f"during evaluation: {e}"
    n_done = min(len(v) for v in rows.values())
    rows = {s: v[:n_done] for s, v in rows.items()}  # compare systems on exactly the same tasks
    paired = {}
    for s in systems[1:]:
        wins = sum(1 for x, y in zip(rows[s], rows["none"]) if x["score"] > y["score"] + 1e-9)
        losses = sum(1 for x, y in zip(rows[s], rows["none"]) if x["score"] < y["score"] - 1e-9)
        paired[s] = {"wins": wins, "losses": losses, "ties": n_done - wins - losses}
    evaluated = test[:n_done]
    # share of test PRs where the system injected anything (skills: at least one skill offered)
    coverage = {s: round(sum(1 for t in evaluated if (serve(t, s)[1] if s == "skills" else serve(t, s)[0]))
                         / max(1, n_done), 3) for s in systems[1:]}
    result = {
        "repo": repo.store.meta("repo"), "entry": a.entry, "systems": systems, "train_tasks": [t.id for t in train],
        "test_tasks_evaluated": n_done, "coverage": coverage,
        "stopped": stopped, "elapsed_s": round(time.time() - t0, 1), "llm": meter.snapshot(), "match": a.match,
        "interventions": {"proposed": len(learner.interventions), "active": len(learner.active()),
                          "rejected": sum(1 for iv in learner.interventions if iv.status == "rejected")},
        "skills": {"edits": len(skiller.edits), "adopted": sum(1 for e in skiller.edits if e["status"] == "adopted"),
                   "library": len(skiller.skills),
                   "test_prs_offered": sum(1 for t in evaluated if serve(t, "skills")[1]),
                   "test_prs_loading_any": sum(1 for n in loaded if n)},
        "summary": {s: summarise(v) for s, v in rows.items()}, "paired_vs_none": paired, "rows": rows,
    }
    out_path = Path(a.out or f"data/results/learning_{data.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if "interventions" in systems:
        save(learner.interventions, out_path.with_name(out_path.stem + "_interventions.json"))
    if "skills" in systems:
        save_skills(skiller.skills, out_path.with_name(out_path.stem + "_skills.json"))
        out_path.with_name(out_path.stem + "_skill_edits.json").write_text(json.dumps(skiller.edits, indent=1),
                                                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
