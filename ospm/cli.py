"""Command-line interface: `ospm <command> ...` (run `ospm -h`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .agents import AGENTS
from .benchmarks import INKWELL_PROJECT, task_from_memory
from .evolution import GEPAConfig, evaluate
from .llm import LLMError, UsageMeter, make_llm
from .manager import ProjectManager
from .memory import Memory
from .models import CIRun, Issue, PullRequest
from .project import ProjectContext
from .render import backlog_table, evolution_report, issue_comment, pr_comment, release_notes


def _print(obj: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str))


class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project = ProjectContext.load(args.project)
        self.memory = Memory(args.db)
        self.meter = UsageMeter(args.budget if args.budget is not None else float(os.environ.get("OSPM_BUDGET_USD", 2.0)))
        self.llm = make_llm(args.llm, args.model, cache=self.memory, meter=self.meter)
        self.pm = ProjectManager(self.project, self.memory, llm=self.llm, effort=args.effort,
                                 log=lambda m: print(m, file=sys.stderr))
        # the bundled benchmark only describes the example project; real projects learn from their own feedback
        self.use_benchmark = self.project.name == "inkwell" and not getattr(args, "no_benchmark", False)

    def usage_note(self) -> None:
        if self.meter.calls or self.meter.cache_hits:
            s = self.meter.snapshot()
            print(f"[llm] {s['calls']} calls, {s['cache_hits']} cache hits, {s['input_tokens']} in / "
                  f"{s['output_tokens']} out tokens, ${s['cost_usd']:.4f} of ${s['budget_usd']:.2f} budget",
                  file=sys.stderr)


def _gh(args, repo: str):
    from .integrations import GitHubClient

    return GitHubClient(repo, dry_run=not getattr(args, "apply", False))


def _report_actions(gh) -> None:
    for a in gh.actions:
        state = "performed" if a["performed"] else "dry-run (use --apply to perform)"
        print(f"[github] {a['kind']} {a['path']}: {state}", file=sys.stderr)


# ---------------------------------------------------------------------------------------------- commands
def cmd_triage(ctx: Ctx) -> int:
    a = ctx.args
    gh = None
    known: list[Issue] = []
    if a.github:
        from .integrations import parse_ref

        repo, number = parse_ref(a.github)
        if number is None:
            raise SystemExit("--github needs owner/repo#number")
        gh = _gh(a, repo)
        issue = gh.get_issue(number)
        known = gh.list_issues(limit=a.known_limit) if a.known_limit else []
    elif a.file:
        issue = Issue(**json.loads(Path(a.file).read_text(encoding="utf-8")))
    else:
        if not a.title:
            raise SystemExit("give --title/--body, --file or --github")
        issue = Issue(number=a.number, title=a.title, body=a.body or "")
    dec = ctx.pm.triage_issue(issue, known)
    if gh is not None:
        gh.comment(issue.number, issue_comment(issue, dec))
        gh.add_labels(issue.number, dec.extras.get("suggested_labels", []))
        _report_actions(gh)
    _print(dec.to_dict() if a.json else issue_comment(issue, dec), a.json)
    return 0


def cmd_ci(ctx: Ctx) -> int:
    a = ctx.args
    run = CIRun(name=a.name, conclusion=a.conclusion, log=Path(a.log).read_text(encoding="utf-8", errors="replace"),
                run_id=f"cli/{Path(a.log).name}")
    dec = ctx.pm.triage_ci(run)
    if a.json:
        _print(dec.to_dict(), True)
    else:
        e = dec.extras
        print(f"category:      {dec.labels['category']}  (confidence {dec.confidence.get('category', 0):.2f}"
              f"{', LLM' if dec.used_llm else ''})")
        print(f"root cause:    {e.get('root_cause')}")
        if e.get("failing_tests"):
            print(f"failing tests: {', '.join(e['failing_tests'])}")
        print(f"suggested fix: {e.get('suggested_fix')}")
        print(f"retry:         {'yes' if e.get('retry_recommended') else 'no'}")
        if dec.rationale:
            print(f"rationale:     {dec.rationale}")
    return 0


def cmd_review(ctx: Ctx) -> int:
    a = ctx.args
    gh = None
    if a.github:
        from .integrations import parse_ref

        repo, number = parse_ref(a.github)
        if number is None:
            raise SystemExit("--github needs owner/repo#number")
        gh = _gh(a, repo)
        pr = gh.get_pull(number)
    elif a.diff:
        pr = PullRequest(number=a.number, title=a.title or Path(a.diff).name,
                         diff=Path(a.diff).read_text(encoding="utf-8", errors="replace"))
    else:
        raise SystemExit("give --diff or --github")
    report = ctx.pm.review_pr(pr)
    if gh is not None:
        gh.comment(pr.number, pr_comment(report))
        _report_actions(gh)
    _print(report.to_dict() if a.json else pr_comment(report), a.json)
    return 0


def cmd_backlog(ctx: Ctx) -> int:
    a = ctx.args
    if a.github:
        from .integrations import parse_ref

        repo, _ = parse_ref(a.github)
        issues = _gh(a, repo).list_issues(limit=a.limit)
    elif a.issues:
        issues = [Issue(**d) for d in json.loads(Path(a.issues).read_text(encoding="utf-8"))]
    else:
        raise SystemExit("give --github owner/repo or --issues FILE")
    milestones = [m.strip() for m in a.milestones.split(",")] if a.milestones else None
    items, plan = ctx.pm.groom_backlog(issues, milestones=milestones)
    _print([i.to_dict() for i in items] if a.json else backlog_table(items, plan, a.limit), a.json)
    return 0


def cmd_release(ctx: Ctx) -> int:
    a = ctx.args
    if a.github:
        from .integrations import parse_ref

        repo, _ = parse_ref(a.github)
        gh = _gh(a, repo)
        latest = gh.latest_release()
        version = a.version or (latest[0] if latest else "0.0.0")
        since = a.since or (latest[1] if latest else None)
        prs = gh.list_merged_pulls(limit=a.limit, since=since)
    elif a.prs:
        version = a.version or "0.0.0"
        prs = [PullRequest(**d) for d in json.loads(Path(a.prs).read_text(encoding="utf-8"))]
    else:
        raise SystemExit("give --github owner/repo or --prs FILE")
    plan = ctx.pm.plan_release(version, prs)
    _print({"current": plan.current_version, "next": plan.next_version, "bump": plan.bump,
            "sections": plan.sections, "breaking": plan.breaking} if a.json else release_notes(plan), a.json)
    return 0


def _agent_names(name: str) -> list[str]:
    if name == "all":
        return list(AGENTS)
    if name not in AGENTS:
        raise SystemExit(f"unknown agent {name!r}; choose from all, {', '.join(AGENTS)}")
    return [name]


def cmd_eval(ctx: Ctx) -> int:
    a = ctx.args
    out = {}
    for name in _agent_names(a.agent):
        task = task_from_memory(name, ctx.memory, ctx.project, include_benchmark=ctx.use_benchmark)
        if not task.examples:
            out[name] = {"skipped": "no labelled examples yet (give feedback first)"}
            continue
        genome = AGENTS[name].seed_genome(ctx.project) if a.genome == "seed" else ctx.pm.genome(name)
        agent = task.make_agent(genome, llm=ctx.llm, effort=a.effort)
        splits = ("train", "val", "test") if a.split == "all" else (a.split,)
        out[name] = {"genome": genome.id}
        for split in splits:
            res = evaluate(agent, task.subset(split), workers=a.workers)
            out[name][split] = res.summary()
            if a.verbose:
                out[name][split]["details"] = res.field_metrics
    _print(out, True)
    return 0


def cmd_evolve(ctx: Ctx) -> int:
    a = ctx.args
    reflection_provider = a.reflection if a.reflection is not None else (a.llm or os.environ.get("OSPM_LLM") or "none")
    reflection = make_llm(reflection_provider, a.reflection_model or a.model, cache=ctx.memory, meter=ctx.meter)
    if a.policy_llm and ctx.llm is None:
        raise SystemExit("--policy-llm needs --llm openai|anthropic")
    results = []
    for name in _agent_names(a.agent):
        cfg = GEPAConfig(max_iterations=a.iterations, max_metric_calls=a.max_calls, minibatch_size=a.minibatch,
                         seed=a.seed, workers=a.workers, policy_effort=a.effort)
        try:
            outcome = ctx.pm.evolve(name, config=cfg, reflection_llm=reflection, use_policy_llm=a.policy_llm,
                                    include_benchmark=ctx.use_benchmark, promote=not a.no_promote,
                                    min_delta=a.min_delta)
        except ValueError as e:  # not enough labelled experience for train/val splits yet
            results.append({"agent": name, "skipped": str(e)})
            if not a.json:
                print(f"## {name}: skipped - {e}\n")
            continue
        outcome.pop("result")
        results.append(outcome)
        if not a.json:
            print(evolution_report(outcome))
            print()
    if a.json:
        _print(results, True)
    return 0


def cmd_feedback(ctx: Ctx) -> int:
    a = ctx.args
    labels: dict[str, Any] = {}
    multi = {f.name for f in AGENTS[a.agent].fields if f.multi}
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"expected field=value, got {kv!r}")
        k, v = kv.split("=", 1)
        labels[k] = [x for x in v.split(",") if x and x != "none"] if k in multi else v
    _print(ctx.pm.feedback(a.agent, a.item, labels), True)
    return 0


def cmd_memory(ctx: Ctx) -> int:
    a = ctx.args
    m = ctx.memory
    if a.action == "stats":
        _print(m.stats(), True)
    elif a.action == "genome":
        g = ctx.pm.genome(a.target or "triage")
        print(g.to_json() if a.json else f"{g.agent} genome {g.id} (origin {g.origin}, gen {g.generation})\n\n"
              f"## Instructions\n{g.instructions}\n\n## Params\n{json.dumps(g.params)}\n\n## Playbook\n"
              f"{g.render_playbook()}\n\n## Exemplars\n{', '.join(g.exemplars) or '(none)'}")
    elif a.action == "promotions":
        _print(m.promotions(a.target), True)
    elif a.action == "runs":
        _print(m.runs(a.target), True)
    elif a.action == "lineage":
        if not a.target:
            raise SystemExit("memory lineage RUN_ID")
        _print(m.lineage(a.target), True)
    elif a.action == "rules":
        _print(m.rule_stats(a.target or "triage"), True)
    elif a.action == "bootstrap":
        _print({"loaded_benchmark_train_cases": ctx.pm.bootstrap_benchmark_memory()}, True)
    return 0


def cmd_gh_event(ctx: Ctx) -> int:
    from .integrations import handle_event

    a = ctx.args
    payload = json.loads(Path(a.event_path).read_text(encoding="utf-8"))
    repo = a.repo or (payload.get("repository") or {}).get("full_name") or os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise SystemExit("cannot determine repository (use --repo)")
    gh = _gh(a, repo)
    result = handle_event(ctx.pm, gh, a.event_name, payload)
    _report_actions(gh)
    _print(result, True)
    return 0


def cmd_demo(ctx: Ctx) -> int:
    from .demo import run_demo

    run_demo(ctx.pm, iterations=ctx.args.iterations, reflection_llm=ctx.llm)
    return 0


# ---------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    def add_globals(parser: argparse.ArgumentParser, defaults: bool) -> None:
        # Global options work before *or* after the subcommand; subparsers use SUPPRESS so they only
        # override the top-level value when actually given.
        d = (lambda v: v) if defaults else (lambda v: argparse.SUPPRESS)
        parser.add_argument("--db", default=d(os.environ.get("OSPM_DB", ".ospm/memory.db")), help="SQLite memory file")
        parser.add_argument("--project", default=d(os.environ.get("OSPM_PROJECT", str(INKWELL_PROJECT))),
                            help="project context TOML (default: the bundled 'inkwell' example)")
        parser.add_argument("--llm", default=d(None), help="none | openai | anthropic | auto (default: $OSPM_LLM or none)")
        parser.add_argument("--model", default=d(None), help="model id (default: gpt-6-luna / claude-opus-5-5)")
        parser.add_argument("--budget", type=float, default=d(None),
                            help="hard LLM spend cap in USD for this process (default 2)")
        parser.add_argument("--effort", default=d("medium"), choices=["low", "medium", "high"],
                            help="LLM reasoning effort")
        parser.add_argument("--json", action="store_true", default=d(False), help="machine-readable output")

    p = argparse.ArgumentParser(prog="ospm", description="Self-evolving multi-agent project manager for OSS repos.")
    add_globals(p, defaults=True)
    common = argparse.ArgumentParser(add_help=False)
    add_globals(common, defaults=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser

    def add_parser(name: str, **kw):
        return _add(name, parents=[common], **kw)

    sub.add_parser = add_parser  # type: ignore[method-assign]

    s = sub.add_parser("triage", help="triage one issue")
    s.add_argument("--title")
    s.add_argument("--body")
    s.add_argument("--number", default="new")
    s.add_argument("--file", help="JSON file with Issue fields")
    s.add_argument("--github", help="owner/repo#number")
    s.add_argument("--known-limit", type=int, default=100, help="open issues to load for duplicate detection")
    s.add_argument("--apply", action="store_true", help="post the comment/labels to GitHub")
    s.set_defaults(fn=cmd_triage)

    s = sub.add_parser("ci", help="diagnose a failed CI log")
    s.add_argument("--log", required=True)
    s.add_argument("--name", default="ci")
    s.add_argument("--conclusion", default="failure")
    s.set_defaults(fn=cmd_ci)

    s = sub.add_parser("review", help="review a pull request (CI triage + conventions + security)")
    s.add_argument("--diff")
    s.add_argument("--title")
    s.add_argument("--number", default="local")
    s.add_argument("--github", help="owner/repo#number")
    s.add_argument("--apply", action="store_true")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("backlog", help="groom and rank the backlog, plan milestones")
    s.add_argument("--github", help="owner/repo")
    s.add_argument("--issues", help="JSON list of issues")
    s.add_argument("--limit", type=int, default=30)
    s.add_argument("--milestones", help="comma-separated milestone names")
    s.set_defaults(fn=cmd_backlog)

    s = sub.add_parser("release", help="propose the next version and release notes")
    s.add_argument("--github", help="owner/repo")
    s.add_argument("--prs", help="JSON list of merged PRs")
    s.add_argument("--version", help="current version (default: latest GitHub release)")
    s.add_argument("--since", help="ISO date: only PRs merged after it")
    s.add_argument("--limit", type=int, default=100)
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("eval", help="evaluate agents on the benchmark (+ accumulated feedback)")
    s.add_argument("--agent", default="all")
    s.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    s.add_argument("--genome", default="deployed", choices=["deployed", "seed"])
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--no-benchmark", action="store_true",
                   help="only use feedback cases from memory (automatic for projects other than 'inkwell')")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("evolve", help="GEPA evolution of an agent's genome, with a promotion gate")
    s.add_argument("--agent", default="all")
    s.add_argument("--iterations", type=int, default=30)
    s.add_argument("--max-calls", type=int, default=None, help="budget of agent invocations")
    s.add_argument("--minibatch", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--reflection", default=None,
                   help="LLM provider for reflective mutation (default: same as --llm; 'none' = offline induction)")
    s.add_argument("--reflection-model", default=None)
    s.add_argument("--policy-llm", action="store_true",
                   help="evaluate candidates with the LLM policy (costly) instead of the heuristic policy")
    s.add_argument("--no-promote", action="store_true")
    s.add_argument("--min-delta", type=float, default=0.01)
    s.add_argument("--no-benchmark", action="store_true")
    s.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("feedback", help="tell an agent the correct answer for a past decision")
    s.add_argument("--agent", required=True, choices=list(AGENTS))
    s.add_argument("--item", required=True, help="item id, e.g. issue-12 or pr-7")
    s.add_argument("--set", nargs="+", required=True, metavar="FIELD=VALUE")
    s.set_defaults(fn=cmd_feedback)

    s = sub.add_parser("memory", help="inspect or bootstrap the persistent memory")
    s.add_argument("action", choices=["stats", "genome", "promotions", "runs", "lineage", "rules", "bootstrap"])
    s.add_argument("target", nargs="?")
    s.set_defaults(fn=cmd_memory)

    s = sub.add_parser("gh-event", help="handle a GitHub Actions event (see .github/workflows/ospm.yml)")
    s.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME"))
    s.add_argument("--event-path", default=os.environ.get("GITHUB_EVENT_PATH"))
    s.add_argument("--repo")
    s.add_argument("--apply", action="store_true", help="actually post comments/labels")
    s.set_defaults(fn=cmd_gh_event)

    from .replay.commands import register as register_replay

    register_replay(sub, sub.add_parser)

    s = sub.add_parser("demo", help="end-to-end walkthrough on the bundled 'inkwell' project")
    s.add_argument("--iterations", type=int, default=25)
    s.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; output contains → ✓ etc.
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    args = build_parser().parse_args(argv)
    if args.cmd == "demo" and args.db == os.environ.get("OSPM_DB", ".ospm/memory.db"):
        args.db = str(Path(tempfile.mkdtemp(prefix="ospm-demo-")) / "memory.db")
        print(f"[demo] using a fresh memory at {args.db}", file=sys.stderr)
    if args.cmd == "gh-event" and not (args.event_name and args.event_path):
        raise SystemExit("gh-event needs --event-name and --event-path (or GITHUB_EVENT_NAME/GITHUB_EVENT_PATH)")
    ctx = Ctx(args)
    try:
        return args.fn(ctx)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        ctx.usage_note()
        ctx.memory.close()


if __name__ == "__main__":
    sys.exit(main())
