"""CLI commands for history mirroring, replay tasks and the agent team (registered by ospm.cli)."""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any

from ..llm import UsageMeter, make_llm


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _meter(args) -> UsageMeter:
    import os

    return UsageMeter(args.budget if args.budget is not None else float(os.environ.get("OSPM_BUDGET_USD", 2.0)))


def _chat(args, meter: UsageMeter):
    from ..harness.models import make_chat_model

    provider = args.llm or "openai"
    if provider in ("none", "off"):
        raise SystemExit("the agent team needs an LLM: pass --llm openai|anthropic")
    return make_chat_model(provider, args.model, meter=meter)


def _repo(args):
    from .workspace import ReplayRepo

    return ReplayRepo(args.data)


def _tasks(args, repo) -> list:
    from .tasks import build_tasks, load_tasks, save_tasks

    path = Path(args.data) / "tasks.jsonl"
    if path.exists() and not getattr(args, "rebuild", False):
        tasks = load_tasks(path)
    else:
        tasks = build_tasks(repo)
        save_tasks(tasks, path)
    kinds = set((getattr(args, "kinds", None) or "triage,review,ci,concern").split(","))
    return [t for t in tasks if t.kind in kinds]


def _genome(args):
    from ..team.roles import TeamGenome, seed_team

    if getattr(args, "genome", None):
        return TeamGenome.from_dict(json.loads(Path(args.genome).read_text(encoding="utf-8")))
    return seed_team()


STRATUM = {"triage": lambda t: t.truth.get("decision"), "review": lambda t: t.truth.get("verdict"),
           "ci": lambda t: t.truth.get("flaky"), "concern": lambda t: t.truth.get("positive")}


def _pick(tasks: list, split: str, n: int, seed: int) -> list:
    """Deterministic sample of a split, balanced over task kinds and, within a kind, over outcomes."""
    rng = random.Random(seed)
    groups: dict[tuple, list] = {}
    for t in sorted((t for t in tasks if t.split == split), key=lambda t: t.id):
        groups.setdefault((t.kind, str(STRATUM[t.kind](t))), []).append(t)
    for v in groups.values():
        rng.shuffle(v)
    kinds = sorted({k for k, _ in groups})
    out: list = []
    while len(out) < n and any(groups.values()):
        for kind in kinds:
            strata = sorted(k for k in groups if k[0] == kind and groups[k])
            if strata and len(out) < n:
                key = strata[len([o for o in out if o.kind == kind]) % len(strata)]
                out.append(groups[key].pop())
    return out


# ---------------------------------------------------------------------------------------------- commands
def cmd_mirror(ctx) -> int:
    from .mirror import mirror

    a = ctx.args
    stats = mirror(a.repo, a.out, a.since, max_logs=a.max_logs, blobless=a.blobless, incremental=a.incremental,
                   review_scope=a.review_scope, log=_log)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_archive(ctx) -> int:
    """Incrementally update every mirror under --data-root (run weekly so CI logs are archived before expiry)."""
    from .mirror import mirror
    from .store import HistoryStore

    a = ctx.args
    results, failed = {}, 0
    for d in sorted(Path(a.data_root).iterdir()):
        if not (d / "history.db").exists():
            continue
        store = HistoryStore(d / "history.db", readonly=True)
        repo, blobless = store.meta("repo"), store.meta("blobless", False)
        store.close()
        if not repo:
            continue
        try:
            results[d.name] = mirror(repo, d, None, max_logs=a.max_logs, blobless=blobless, incremental=True, log=_log)
        except Exception as e:  # keep archiving the other repositories
            failed += 1
            results[d.name] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"[archive] {d.name} failed: {e}")
    log_path = Path(a.data_root) / "archive.log"
    with log_path.open("a", encoding="utf-8") as fh:
        from datetime import datetime, timezone

        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": results}) + "\n")
    print(json.dumps(results, indent=2))
    return 1 if failed else 0


def cmd_replay(ctx) -> int:
    a = ctx.args
    repo = _repo(a)
    if a.action == "setup-ci":
        print(repo.ensure_venv(log=_log))
    elif a.action == "tasks":
        a.rebuild = True
        tasks = _tasks(a, repo)
        from collections import Counter

        print(json.dumps({"total": len(tasks), "by_kind": Counter(t.kind for t in tasks),
                          "by_split": Counter(f"{t.kind}/{t.split}" for t in tasks)}, indent=2, default=dict))
    elif a.action == "stats":
        from .workspace import disk_usage

        stats = repo.store.stats()
        stats["disk_mb"] = {p.name: round(disk_usage(p) / 2**20, 1) for p in Path(a.data).iterdir()
                            if p.is_dir()} | {"history.db": round(repo.db_path.stat().st_size / 2**20, 1)}
        print(json.dumps(stats, indent=2))
    elif a.action == "clean":
        repo.clean()
        if a.traces:
            import shutil

            shutil.rmtree(Path(a.data) / "traces", ignore_errors=True)
        print("cleaned")
    return 0


def cmd_team(ctx) -> int:
    from ..evolution.team_gepa import EpisodeCache, TeamGEPA, TeamGEPAConfig
    from .engine import episode_fn, run_task

    a = ctx.args
    repo = _repo(a)
    meter = _meter(a)
    model = _chat(a, meter)
    genome = _genome(a)
    trace_dir = Path(a.data) / "traces"
    if a.action == "run":
        task = next((t for t in _tasks(a, repo) if t.id == a.task), None)
        if task is None:
            raise SystemExit(f"unknown task {a.task!r} (see `ospm replay tasks`)")
        ep, score, feedback = run_task(repo, genome, model, task, entry_role=a.entry, log=_log, trace_dir=trace_dir)
        print(json.dumps({"task": task.id, "entry": a.entry, "score": round(score, 4), "feedback": feedback,
                          "exit_status": ep.exit_status, "roles_used": sorted(ep.roles_used),
                          "steps": ep.steps, "cost_usd": round(ep.cost, 4), "output": ep.output}, indent=2))
    elif a.action == "eval":
        tasks = _pick(_tasks(a, repo), a.split, a.limit, a.seed)
        fn = episode_fn(repo, model, entry_role=a.entry, log=_log if a.verbose else None, trace_dir=trace_dir)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(a.workers) as pool:
            outs = list(pool.map(lambda t: fn(genome, t), tasks))
        rows = [{"task": o.task_id, "score": round(o.score, 3), "cost": o.cost, "roles": sorted(o.roles_used),
                 "exit": o.exit_status} for o in outs]
        mean = sum(o.score for o in outs) / len(outs) if outs else 0.0
        print(json.dumps({"entry": a.entry, "split": a.split, "n": len(outs), "mean_score": round(mean, 4),
                          "mean_cost": round(sum(o.cost for o in outs) / max(1, len(outs)), 4), "tasks": rows},
                         indent=2))
    elif a.action == "evolve":
        tasks = _tasks(a, repo)
        train = _pick(tasks, "train", a.train, a.seed)
        val = _pick(tasks, "val", a.val, a.seed)
        test = _pick(tasks, "test", a.test, a.seed)
        reflection = make_llm(a.llm or "openai", a.model, meter=meter)
        cache = EpisodeCache(Path(a.data) / "episode_cache" / f"{a.entry}.jsonl")
        engine = TeamGEPA(genome, train, val, test,
                          episode_fn(repo, model, entry_role=a.entry, log=_log if a.verbose else None,
                                     trace_dir=trace_dir),
                          reflection, entry_role=a.entry,
                          config=TeamGEPAConfig(iterations=a.iterations, minibatch=a.minibatch, workers=a.workers,
                                                seed=a.seed, blame=not a.no_blame, evaluate_test=not a.no_test),
                          cache=cache, budget_exhausted=lambda: meter.budget_usd is not None and
                          meter.cost_usd >= 0.9 * meter.budget_usd, log=_log, out_dir=Path(a.data) / "evolution")
        result = engine.run()
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    s = meter.snapshot()
    _log(f"[llm] {s['calls']} calls, {s['input_tokens']} in / {s['output_tokens']} out tokens, "
         f"${s['cost_usd']:.4f} of ${s['budget_usd']:.2f}")
    return 0


def cmd_trace(ctx) -> int:
    from ..harness.agent import Step, Trace

    def build(d: dict[str, Any]) -> Trace:
        t = Trace(d["agent_id"], d["role"], d["task"], d.get("parent", ""), [Step(**s) for s in d["steps"]],
                  [build(c) for c in d["children"]], d["submission"], d["exit_status"], d["cost"], d["compactions"])
        return t

    with gzip.open(ctx.args.file, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"task {data['task']['id']} | entry {data['entry_role']} | score {data['score']:.3f} | "
          f"cost ${data['cost']:.4f} | {data['exit_status']}\nfeedback: {data['feedback']}\n")
    print(build(data["trace"]).render(max_obs=ctx.args.max_obs))
    return 0


def register(sub, add_parser) -> None:
    s = add_parser("mirror", help="mirror a GitHub repository's history for replay")
    s.add_argument("repo", help="owner/name")
    s.add_argument("--since", help="start of the window (YYYY-MM-DD); not needed with --incremental")
    s.add_argument("--out", required=True, help="data directory, e.g. data/pytest")
    s.add_argument("--max-logs", type=int, default=800)
    s.add_argument("--blobless", action="store_true", help="partial clone without file contents (small on disk)")
    s.add_argument("--incremental", action="store_true", help="update an existing mirror with what changed since")
    s.add_argument("--review-scope", default="all", choices=["all", "merged_or_commented"],
                   help="which PRs get per-PR review fetches (merged_or_commented saves API calls on large repos)")
    s.set_defaults(fn=cmd_mirror)

    s = add_parser("archive", help="incrementally update every mirror under a data root (schedule weekly)")
    s.add_argument("--data-root", default="data")
    s.add_argument("--max-logs", type=int, default=800)
    s.set_defaults(fn=cmd_archive)

    s = add_parser("replay", help="replay environment: local CI setup, task mining, stats, cleanup")
    s.add_argument("action", choices=["setup-ci", "tasks", "stats", "clean"])
    s.add_argument("--data", required=True)
    s.add_argument("--kinds", default="triage,review,ci,concern")
    s.add_argument("--traces", action="store_true", help="clean: also delete saved traces")
    s.set_defaults(fn=cmd_replay)

    s = add_parser("team", help="run, evaluate or evolve (Team-GEPA) the multi-agent team on replayed history")
    s.add_argument("action", choices=["run", "eval", "evolve"])
    s.add_argument("--data", required=True)
    s.add_argument("--task", help="run: task id (see `ospm replay tasks`)")
    s.add_argument("--entry", default="orchestrator",
                   choices=["orchestrator", "specialist", "triage", "ci_triage", "review", "security"],
                   help="orchestrator = full team; specialist = each task goes directly to its specialist")
    s.add_argument("--genome", help="team genome JSON (default: seed team)")
    s.add_argument("--kinds", default="triage,review,ci,concern")
    s.add_argument("--split", default="val", choices=["train", "val", "test"])
    s.add_argument("--limit", type=int, default=6)
    s.add_argument("--train", type=int, default=9)
    s.add_argument("--val", type=int, default=6)
    s.add_argument("--test", type=int, default=6)
    s.add_argument("--iterations", type=int, default=6)
    s.add_argument("--minibatch", type=int, default=3)
    s.add_argument("--workers", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-blame", action="store_true", help="vanilla GEPA module selection (ablation)")
    s.add_argument("--no-test", action="store_true")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_team)

    s = add_parser("trace", help="pretty-print a saved episode trace (.json.gz)")
    s.add_argument("file")
    s.add_argument("--max-obs", type=int, default=800)
    s.set_defaults(fn=cmd_trace)
