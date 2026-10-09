"""Evolving a review team: the agent-type library, each type, and the team's coordination.

GEPA-style reflective mutation over a TeamGenome whose specialists form a library of agent types
(ospm/team/roles.py). After each chronological batch of training PRs, a reflector reads per-agent credit (which
agent found which maintainer concern, whose comments were false alarms, what each spent) and the coordination of
the episodes, and proposes edits, each at ONE level:

* agent   - one type's rules (`add_rule` / `edit_rule` / `remove_rule`) or its description (when it is spawned);
* library - `add_type` (a new specialist for concerns nobody catches) or `remove_type` (no contribution);
* team    - the orchestrator's own rules (which types to compose, what a brief contains, no duplicate work, ...).

An edit is kept only if replaying other earlier PRs (source PRs excluded; outcomes known by then) shows it helps:
quality (semantic concern score; clean PRs score on silence) must rise without the cost growing more than
`max_cost_ratio`, or stay level while the cost falls by `min_saving`. Cost is measured per episode, so a
cost-saving edit (dropping a type that never contributes, stopping duplicate reading) is visible on a few replays.

    python -m ospm.learning.team_evolution --data data/sglang --train 64 --test 48 --budget 9
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from ..harness.models import make_chat_model
from ..llm import LLM, BudgetExceeded, LLMError, UsageMeter, make_llm
from ..replay.semantic import THRESHOLD, Matcher, semantic_metrics, semantic_score
from ..replay.tasks import ReplayTask, load_tasks
from ..replay.workspace import ReplayRepo
from ..team.roles import TeamGenome, parse_json, seed_library, specialist
from .experiment import BudgetStop, JsonlCache, make_runner, pick, summarise
from .interventions import pr_key, resolve_sources

OPS = ("add_rule", "edit_rule", "remove_rule", "set_description", "add_type", "remove_type")
LEVEL = {"add_type": "library", "remove_type": "library"}

REFLECT_SYSTEM = """You improve a TEAM of code-review agents for one repository. An orchestrator receives each pull
request, spawns specialists from a LIBRARY of agent types (each has a description the orchestrator reads, its own
procedure, and a list of rules), reads their final messages and writes the final review. The goal is the comments
the maintainers actually made (and silence on changes they approved as-is), at the lowest cost.

You get the current library, the orchestrator's rules, per-type credit over recent PRs (how often each type was
spawned, what it cost, how many maintainer concerns it found - and how many ONLY it found -, how many comments it
made on clean PRs), the team's cost and duplicate work, and post-mortems of PRs where the team missed concerns or
raised false alarms.

Propose edits; each edit changes ONE thing at ONE level:
- team (target "orchestrator"): add_rule / edit_rule / remove_rule on the orchestrator's coordination rules - which
  types to spawn for which changes, what a brief must contain, not re-reading what a specialist covers, verifying
  a finding before adding it, when to follow up on a report.
- agent (target = a type): add_rule / edit_rule / remove_rule on that type's procedure rules, or set_description
  (changes when the orchestrator spawns it).
- library: add_type (name, description, focus = the new type's instructions, contract "review" for reviewers or
  "notes" for researchers) when a kind of concern is systematically missed; remove_type for a type whose credit
  shows no contribution for its cost.
Prefer cost cuts that lose nothing (types that never contribute, duplicate work) and coverage for kinds of concerns
nobody catches. Rules must be general - never specific to one pull request. Return JSON only."""


def _schema() -> dict:
    edit = {"type": "object", "properties": {
        "op": {"type": "string", "enum": list(OPS)}, "target": {"type": "string"},
        "index": {"type": "integer", "description": "rule index for edit_rule/remove_rule, else -1"},
        "text": {"type": "string", "description": "rule text, description (set_description) or focus (add_type)"},
        "description": {"type": "string", "description": "add_type only, else empty"},
        "contract": {"type": "string", "enum": ["review", "notes", ""]},
        "rationale": {"type": "string"}, "source_tasks": {"type": "array", "items": {"type": "string"}}},
        "required": ["op", "target", "index", "text", "description", "contract", "rationale", "source_tasks"],
        "additionalProperties": False}
    return {"type": "object", "properties": {"edits": {"type": "array", "items": edit}}, "required": ["edits"],
            "additionalProperties": False}


def level_of(e: dict) -> str:
    return LEVEL.get(e["op"]) or ("team" if e.get("target") == "orchestrator" else "agent")


def apply_edit(genome: TeamGenome, e: dict[str, Any]) -> TeamGenome | None:
    """The edited team, or None for an invalid or no-op edit. The input genome is never mutated."""
    g = genome.clone()
    op, target, text = e["op"], (e.get("target") or "").strip(), (e.get("text") or "").strip()[:1500]
    if op == "add_type":
        name = "".join(ch if ch.isalnum() else "_" for ch in target.lower()).strip("_")[:32]
        desc = (e.get("description") or "").strip()[:400]
        if not name or name in g.roles or not text or not desc:
            return None
        g.roles[name] = specialist(name, text, contract=e.get("contract") or "review", description=desc)
    elif op == "remove_type":
        reviewers = [r for r in g.specialists() if (g.roles[r].contract or r) == "review"]
        if target not in g.specialists() or (target in reviewers and len(reviewers) <= 1):
            return None  # never remove the last reviewer
        del g.roles[target]
    else:
        if target not in g.roles:
            return None
        role = g.roles[target]
        rules = list(role.playbook)
        i = e.get("index", -1)
        if op == "add_rule" and text and text not in rules:
            rules.append(text)
        elif op == "edit_rule" and 0 <= i < len(rules) and text and text != rules[i]:
            rules[i] = text
        elif op == "remove_rule" and 0 <= i < len(rules):
            del rules[i]
        elif op == "set_description" and target != "orchestrator" and text and text != role.description:
            g.roles[target] = replace(role, description=text[:400])
            return _tag(g, e)
        else:
            return None
        g.roles[target] = replace(role, playbook=rules[:12])
    return _tag(g, e)


def _tag(g: TeamGenome, e: dict) -> TeamGenome:
    g.origin, g.notes = level_of(e), f"{e['op']} {e.get('target')}: {(e.get('text') or '')[:120]}"
    return g


# ---------------------------------------------------------------------------------------------- credit
def _comments(submission: str) -> list[dict]:
    d = parse_json(submission or "") or {}
    return [c for c in d.get("comments") or [] if isinstance(c, dict) and c.get("body")]


def episode_credit(trace: dict | None, task: ReplayTask, matcher: Matcher) -> dict[str, Any]:
    """Who did what in one team episode: per specialist the concerns it found (semantic matching), its comments,
    cost and steps; for the orchestrator its own cost, bash steps and coordination calls."""
    if not trace:
        return {"agents": [], "orchestrator": {}}
    gold = [c for c in task.truth["concerns"] if not c["nit"]] if task.truth["positive"] else []
    agents = []
    for ch in trace.get("children", []):
        mine = _comments(ch.get("submission", ""))
        found = [i for i, p in enumerate(matcher.concern_scores(gold, {"comments": mine})) if p >= THRESHOLD] \
            if gold else []
        agents.append({"type": ch.get("role"), "cost": ch.get("cost", 0.0), "steps": len(ch.get("steps", [])),
                       "comments": len(mine), "found": found, "exit": ch.get("exit_status")})
    calls = [c["name"] for s in trace.get("steps", []) for c in s.get("calls", [])]
    orch = {"cost": trace.get("cost", 0.0), "bash_steps": calls.count("bash"),
            "coordination": {k: calls.count(k) for k in ("spawn_agent", "send_message", "wait_for_agents",
                                                         "close_agent") if calls.count(k)}}
    return {"agents": agents, "orchestrator": orch}


def credit_table(credits: list[tuple[ReplayTask, dict]]) -> dict[str, dict[str, Any]]:
    """Per agent type over many episodes: spawns, cost, concerns found (and found by it alone), clean-PR comments."""
    table: dict[str, dict[str, Any]] = {}
    for task, cr in credits:
        by_type: dict[str, set] = {}
        for a in cr["agents"]:
            by_type.setdefault(a["type"], set()).update(a["found"])
        for a in cr["agents"]:
            t = table.setdefault(a["type"], {"spawned": 0, "cost": 0.0, "steps": 0, "found": 0, "found_alone": 0,
                                             "clean_pr_comments": 0})
            t["spawned"] += 1
            t["cost"] += a["cost"]
            t["steps"] += a["steps"]
            others = set().union(*[v for k, v in by_type.items() if k != a["type"]]) if len(by_type) > 1 else set()
            t["found"] += len(a["found"])
            t["found_alone"] += len(set(a["found"]) - others)
            if not task.truth["positive"]:
                t["clean_pr_comments"] += a["comments"]
    for t in table.values():
        t["mean_cost"] = round(t.pop("cost") / t["spawned"], 4)
        t["mean_steps"] = round(t.pop("steps") / t["spawned"], 1)
    return table


# ---------------------------------------------------------------------------------------------- learner
@dataclass
class TeamEvoConfig:
    batch: int = 8
    max_edits_per_step: int = 3
    validate_pos: int = 3  # replays per edit: PRs with maintainer concerns (distinct PRs) ...
    validate_neg: int = 3  # ... and clean PRs
    min_gain: float = 0.01  # quality gain for an edit that does not cut cost
    max_cost_ratio: float = 1.3  # ... and the most extra cost such an edit may bring
    min_saving: float = 0.15  # cost cut for an edit that keeps quality level
    validation_pool: int = 150
    workers: int = 4


class TeamEvolver:
    """`run(task, genome=...) -> (score, output)` with run.trace_of / run.cost_of (experiment.make_runner, entry
    "orchestrator"); `past`: earlier tasks edits may be validated on (outcome known at the current step)."""

    def __init__(self, repo: ReplayRepo, run: Callable, llm: LLM, matcher: Matcher, genome: TeamGenome,
                 config: TeamEvoConfig | None = None, past: list[ReplayTask] | None = None,
                 log: Callable[[str], None] = print):
        self.repo, self.run, self.llm, self.matcher = repo, run, llm, matcher
        self.genome = genome
        self.cfg = config or TeamEvoConfig()
        self.past = sorted(past or [], key=lambda t: (t.as_of, t.id), reverse=True)
        self.log = log
        self.seen: list[ReplayTask] = []
        self.now = ""
        self.edits: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.credits: list[tuple[ReplayTask, dict]] = []

    # outcome-known rule and validation pool: as in interventions.InterventionLearner
    def known(self, task: ReplayTask) -> bool:
        if task.truth["positive"] or self.repo is None:
            return task.as_of <= self.now
        merged = ((self.repo.store.pull(task.pr) or {}).get("merged_at") or "") if task.pr is not None else ""
        return bool(merged) and merged <= self.now

    def validation_tasks(self, source_prs: set) -> list[ReplayTask]:
        ids = {t.id for t in self.seen}
        older = [t for t in self.past if t.id not in ids and t.as_of <= self.now][: self.cfg.validation_pool]
        pool = [t for t in self.seen[::-1] + older if self.known(t) and pr_key(t.id, t.pr) not in source_prs]
        pos, prs = [], set()
        for t in pool:
            if t.truth["positive"] and len(pos) < self.cfg.validate_pos and pr_key(t.id, t.pr) not in prs:
                pos.append(t)
                prs.add(pr_key(t.id, t.pr))
        neg = [t for t in pool if not t.truth["positive"]][: self.cfg.validate_neg]
        return pos + neg

    def _run_many(self, jobs: list[tuple[ReplayTask, TeamGenome]]) -> list[tuple[float, dict | None]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max(1, self.cfg.workers)) as pool:
            return list(pool.map(lambda j: self.run(j[0], genome=j[1]), jobs))

    def cost(self, task: ReplayTask, genome: TeamGenome) -> float:
        c = self.run.cost_of(task, genome=genome)
        if c is None:  # an episode cached before costs were recorded: read it off the trace
            tr = self.run.trace_of(task, genome=genome) or {}
            c = tr.get("cost", 0.0) + sum(ch.get("cost", 0.0) for ch in tr.get("children", []))
        return c

    # ------------------------------------------------------------------ post-mortem + proposals
    def postmortem(self, task: ReplayTask, out: dict | None, cr: dict) -> dict | None:
        mine = [c for c in (out or {}).get("comments") or [] if isinstance(c, dict)]
        sem = semantic_metrics(self.matcher, task, out)
        if task.truth["positive"] and sem["sem_recall"] == 1.0:
            return None
        if not task.truth["positive"] and not mine:
            return None
        gold = [c for c in task.truth["concerns"] if not c["nit"]] if task.truth["positive"] else []
        return {"task": task.id, "positive": task.truth["positive"],
                "concerns": [{"i": i, "path": c["path"], "line": c["line"], "type": (c.get("llm_labels") or {}).get(
                    "primary"), "body": c["body"][:300]} for i, c in enumerate(gold)],
                "final_comments": [{k: str(m.get(k))[:200] for k in ("path", "line", "body")} for m in mine][:5],
                "agents": cr["agents"], "orchestrator": cr["orchestrator"]}

    def propose(self, items: list[dict], table: dict) -> list[dict]:
        lib = []
        for name in self.genome.specialists():
            r = self.genome.roles[name]
            rules = "".join(f"\n    [{i}] {p}" for i, p in enumerate(r.playbook))
            lib.append(f"- {name} (contract {r.contract or name}): {r.description}{rules}")
        orch = "".join(f"\n  [{i}] {p}" for i, p in enumerate(self.genome.roles["orchestrator"].playbook)) or " (none)"
        user = ("# Library\n" + "\n".join(lib) + f"\n\n# Orchestrator rules{orch}\n\n# Credit per type (this and "
                f"earlier batches)\n{json.dumps(table, indent=1)}\n\n# Post-mortems\n"
                + "\n\n".join(json.dumps(it) for it in items[:10])
                + f"\n\nPropose at most {self.cfg.max_edits_per_step} edits.")
        try:
            d = self.llm.complete_json(REFLECT_SYSTEM, user, _schema(), name="team_edits", effort="medium")
        except (LLMError, BudgetExceeded):
            return []
        known = [it["task"] for it in items]
        edits = d["edits"][: self.cfg.max_edits_per_step]
        for e in edits:
            e["source_tasks"] = resolve_sources(list(e.get("source_tasks") or []), known)[:8]
        return edits

    # ------------------------------------------------------------------ acceptance gate
    def validate(self, e: dict) -> bool:
        cand = apply_edit(self.genome, e)
        record = {"edit": e, "level": level_of(e), "status": "rejected"}
        self.edits.append(record)
        if cand is None:
            record["validation"] = {"reason": "invalid or no-op edit"}
            return False
        by_id = {t.id: t for t in self.seen + self.past}
        source = {pr_key(i, by_id[i].pr if i in by_id else None) for i in e.get("source_tasks") or []}
        tasks = self.validation_tasks(source)
        if not tasks:
            record["validation"] = {"reason": "no earlier PRs with known outcomes"}
            return False
        res = self._run_many([(t, self.genome) for t in tasks] + [(t, cand) for t in tasks])
        q0 = sum(r[0] for r in res[: len(tasks)]) / len(tasks)
        q1 = sum(r[0] for r in res[len(tasks):]) / len(tasks)
        c0 = sum(self.cost(t, self.genome) for t in tasks) / len(tasks)
        c1 = sum(self.cost(t, cand) for t in tasks) / len(tasks)
        ratio = c1 / c0 if c0 else 1.0
        gain_ok = q1 - q0 >= self.cfg.min_gain and ratio <= self.cfg.max_cost_ratio
        saving_ok = q1 >= q0 and ratio <= 1 - self.cfg.min_saving
        record["validation"] = {"tasks": [t.id for t in tasks], "quality_before": round(q0, 4),
                                "quality_after": round(q1, 4), "cost_before": round(c0, 4),
                                "cost_after": round(c1, 4), "cost_ratio": round(ratio, 3),
                                "accepted_for": "quality" if gain_ok else "cost" if saving_ok else None}
        if gain_ok or saving_ok:
            self.genome = cand
            record["status"] = "adopted"
            return True
        return False

    # ------------------------------------------------------------------ loop
    def learn(self, train: list[ReplayTask]) -> dict[str, Any]:
        ordered = sorted(train, key=lambda t: (t.as_of, t.id))
        pending: list[dict] = []
        for i in range(0, len(ordered), self.cfg.batch):
            batch = ordered[i:i + self.cfg.batch]
            outs = self._run_many([(t, self.genome) for t in batch])
            items = []
            for t, (score, out) in zip(batch, outs):
                cr = episode_credit(self.run.trace_of(t, genome=self.genome), t, self.matcher)
                self.credits.append((t, cr))
                self.history.append({"task": t.id, "score": score, "genome": self.genome.id,
                                     "cost": self.cost(t, self.genome)})
                if (pm := self.postmortem(t, out, cr)) is not None:
                    items.append(pm)
            self.seen.extend(batch)
            self.now = max(self.now, batch[-1].as_of)
            pending = (pending + items)[-12:]
            edits = self.propose(pending, credit_table(self.credits[-32:]))
            adopted = 0
            for e in edits:
                if self.validate(e):
                    adopted += 1
                    done = set(e.get("source_tasks") or [])
                    pending = [it for it in pending if it["task"] not in done]
            self.log(f"[team] step {i // self.cfg.batch + 1}: {len(batch)} PRs, {len(items)} post-mortems, "
                     f"{len(edits)} edits ({', '.join(level_of(e) + ':' + e['op'] for e in edits)}), {adopted} "
                     f"adopted; types {self.genome.specialists()}, orchestrator rules "
                     f"{len(self.genome.roles['orchestrator'].playbook)}")
        return {"genome": self.genome.to_dict(), "edits": self.edits, "history": self.history}


# ---------------------------------------------------------------------------------------------- experiment
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--llm", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--train", type=int, default=64)
    p.add_argument("--test", type=int, default=48)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=float, default=9.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    data = Path(a.data)
    repo = ReplayRepo(data)
    tasks = [t for t in load_tasks(data / "tasks.jsonl") if t.kind == "concern"]
    train = pick([t for t in tasks if t.split == "train"], a.train, newest=True)
    test = pick([t for t in tasks if t.split == "test"], a.test, newest=False)
    log(f"[team] {repo.store.meta('repo')}: train {len(train)}, test {len(test)}")
    meter = UsageMeter(a.budget)
    model = make_chat_model(a.llm, a.model, meter=meter)
    reflect = make_llm(a.llm, a.model, meter=meter, cache=JsonlCache(data / "reflect_cache.jsonl"))
    matcher = Matcher(data / "semantic_cache.jsonl")
    seed = seed_library()
    run = make_runner(repo, model, meter, data / "learning_cache.jsonl", data / "traces", entry="orchestrator",
                      score_fn=lambda t, out: semantic_score(matcher, t, out), genome=seed)
    evo = TeamEvolver(repo, run, reflect, matcher, seed, TeamEvoConfig(batch=a.batch, workers=a.workers),
                      past=[t for t in tasks if t.split != "test"], log=log)
    t0, stopped = time.time(), None
    try:
        evo.learn(train)
    except BudgetStop as e:
        stopped = f"during learning: {e}"
    systems = {"seed_library": seed, "evolved": evo.genome}
    rows: dict[str, list[dict]] = {s: [] for s in systems}
    test_credits: dict[str, list] = {s: [] for s in systems}
    if stopped is None:
        try:
            jobs = [(t, s) for t in test for s in systems]
            outs = evo._run_many([(t, systems[s]) for t, s in jobs])
            for (t, s), (score, out) in zip(jobs, outs):
                from ..replay.concerns import concern_metrics

                rows[s].append({"task": t.id, "score": score, "cost": evo.cost(t, systems[s]),
                                **concern_metrics(t, out), **semantic_metrics(matcher, t, out)})
                test_credits[s].append((t, episode_credit(run.trace_of(t, genome=systems[s]), t, matcher)))
        except BudgetStop as e:
            stopped = f"during evaluation: {e}"
    n = min(len(v) for v in rows.values())
    rows = {s: v[:n] for s, v in rows.items()}
    wins = sum(1 for x, y in zip(rows["evolved"], rows["seed_library"]) if x["score"] > y["score"] + 1e-9)
    losses = sum(1 for x, y in zip(rows["evolved"], rows["seed_library"]) if x["score"] < y["score"] - 1e-9)
    result = {"repo": repo.store.meta("repo"), "train_tasks": [t.id for t in train], "test_tasks_evaluated": n,
              "stopped": stopped, "elapsed_s": round(time.time() - t0, 1), "llm": meter.snapshot(),
              "edits": {"proposed": len(evo.edits), "adopted": sum(e["status"] == "adopted" for e in evo.edits),
                        "by_level": {lv: sum(e["level"] == lv for e in evo.edits) for lv in ("agent", "library",
                                                                                            "team")}},
              "summary": {s: {**summarise(v), "mean_cost": round(sum(r["cost"] or 0 for r in v) / max(1, len(v)), 4)}
                          for s, v in rows.items()},
              "paired_evolved_vs_seed": {"wins": wins, "losses": losses, "ties": n - wins - losses},
              "test_credit": {s: credit_table(c[:n]) for s, c in test_credits.items()},
              "genomes": {s: g.to_dict() for s, g in systems.items()}, "rows": rows}
    out_path = Path(a.out or f"data/results/team_evolution_{data.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    out_path.with_name(out_path.stem + "_edits.json").write_text(json.dumps(evo.edits, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "genomes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
