"""Post-mortem skill evolution: reflect on what the reviewer *did* (its trace) next to what maintainers said, edit
the skill library, and keep an edit only if replaying other earlier PRs shows it helps.

GEPA-style reflective mutation, with skills as the modules:
* reflection reads the trace digest (commands, files opened, skills offered and loaded, steps used) together with
  the maintainer concerns it missed or the comments it should not have made;
* blame picks the part to change - a skill's description (it was not loaded when it should have been), its body
  (loaded, but the procedure did not lead to the concern), a new skill (no procedure for that area) or a deletion
  (it causes false alarms);
* the acceptance gate replays *other* earlier PRs where the edited skill is offered, with the old vs. the new
  library: recall on PRs with maintainer concerns must rise and clean PRs must get no extra comments.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from typing import Any, Callable

from ..llm import BudgetExceeded, LLMError
from ..replay.concerns import concern_metrics
from ..replay.tasks import ReplayTask
from .interventions import InterventionLearner, pr_key, resolve_sources
from .skills import TOOLS_NOTE, Skill, index_block, loaded_skills, offered, rubric_block, slug

TraceLookup = Callable[[ReplayTask, str, tuple], "dict | None"]  # (task, ctx, skills) -> trace dict of that episode

POSTMORTEM_SYSTEM = """You run post-mortems on a code-review agent for one repository and maintain its library of
SKILLS: short procedures for reviewing a code area, which the agent can load on demand (`skill <name>`).

For each reviewed pull request you get what the maintainers actually raised (or that they approved it without
comments), the agent's comments, and a digest of how it worked: the commands it ran, whether it opened the files the
maintainers commented on, which skills were offered to it and which it loaded, and how many steps it used.

Diagnose WHY the agent missed each concern or raised a false alarm, then propose skill edits:
- `create`: no skill covers this area of the code. Write a procedure for reviewing changes there.
- `edit` part `description`: a relevant skill was offered but not loaded - make its description say when to use it.
- `edit` part `body`: a skill was loaded but its procedure did not lead to the concern - fix the procedure.
- `delete`: a skill misleads the agent (e.g. causes comments on changes maintainers approved as-is).

A skill is a PROCEDURE for an area, not a fact about one PR: what to read beyond the diff (callers, registries,
configs, tests, docs), what invariants to check, which questions maintainers of this area tend to ask (look them up
with `hist reviews --path <file>` - list that step in the procedure where useful), and when NOT to comment. It is
validated on other pull requests touching the same area, so procedures fitted to one PR's constants will fail.
`paths` are fnmatch globs over changed files that scope where the skill is offered ([] = everywhere, rarely right).
Keep descriptions under 300 characters and bodies under 1500. Prefer editing an existing skill to adding a near
duplicate. Propose nothing for concerns that are one-off judgement calls. Return JSON only."""


def _schema(roles: tuple[str, ...] = ("review",)) -> dict:
    edit = {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["create", "edit", "delete"]},
        "role": {"type": "string", "enum": list(roles)},
        "part": {"type": "string", "enum": ["description", "body", "both"]},
        "name": {"type": "string"}, "description": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}}, "body": {"type": "string"},
        "diagnosis": {"type": "string"}, "source_tasks": {"type": "array", "items": {"type": "string"}}},
        "required": ["op", "role", "part", "name", "description", "paths", "body", "diagnosis", "source_tasks"],
        "additionalProperties": False}
    return {"type": "object", "properties": {"edits": {"type": "array", "items": edit}}, "required": ["edits"],
            "additionalProperties": False}


def _commands(trace: dict) -> list[str]:
    out = []
    for s in trace.get("steps", []):
        for c in s.get("calls", []):
            a = c.get("arguments")
            if isinstance(a, dict) and a.get("command"):
                out.append(str(a["command"]))
    for child in trace.get("children", []):
        out.extend(_commands(child))
    return out


def trace_digest(trace: dict | None, paths: list[str], max_commands: int = 40) -> dict[str, Any]:
    """What the agent did, compactly. `opened[path]` is whether any command named the file (a full `git diff` shows
    every changed file, so this tells whether the agent looked at it *specifically*)."""
    if not trace:
        return {"steps": None, "loaded": [], "opened": {}, "commands": []}
    cmds = _commands(trace)
    text = "\n".join(cmds)
    return {"steps": len(trace.get("steps", [])), "loaded": loaded_skills(trace),
            "opened": {p: p in text for p in paths},
            "commands": [re.sub(r"\s+", " ", c)[:200] for c in cmds[:max_commands]]}


def team_digest(trace: dict, max_chars: int = 400) -> dict[str, Any]:
    """Coordination as the reflector should see it: the orchestrator's briefs and messages (its non-bash tool calls)
    and, per specialist, what it loaded and the final report its parent received."""
    calls = [f"{c['name']}({json.dumps(c.get('arguments'))[:max_chars]})" for s in trace.get("steps", [])
             for c in s.get("calls", []) if c.get("name") != "bash"]
    agents = [{"agent": c.get("agent_id"), "role": c.get("role"), "steps": len(c.get("steps", [])),
               "exit": c.get("exit_status"), "loaded": loaded_skills(c),
               "final_message": (c.get("submission") or "")[:max_chars * 2]} for c in trace.get("children", [])]
    return {"coordination": calls, "specialists": agents}


def apply_edit(skills: list[Skill], e: dict[str, Any]) -> tuple[list[Skill], Skill | None]:
    """The library after the edit, and the skill whose effect has to be validated (the new version, or the deleted
    one). Edits never mutate: a changed skill is a new object with a higher version."""
    try:
        name = slug(e.get("name") or "")
    except ValueError:
        return skills, None
    old = next((s for s in skills if s.name == name and s.status == "active"), None)
    rest = [s for s in skills if s is not old]
    if e["op"] == "delete":
        return (rest, old) if old else (skills, None)
    paths = [p for p in e.get("paths") or [] if p][:8]
    desc, body = (e.get("description") or "").strip()[:300], (e.get("body") or "").strip()[:1500]
    if old is None:  # create (an edit of a missing skill is a create)
        if not (name and desc and body):
            return skills, None
        new = Skill(name, desc, body, paths=paths, role=e.get("role") or "review",
                    provenance=list(e.get("source_tasks") or []))
    else:
        part = e.get("part") or "both"
        new = replace(old, description=desc if part in ("description", "both") and desc else old.description,
                      body=body if part in ("body", "both") and body else old.body, paths=paths or old.paths,
                      provenance=sorted(set(old.provenance) | set(e.get("source_tasks") or [])),
                      version=old.version + 1, validation={})
        if (new.description, new.body, new.paths) == (old.description, old.body, old.paths):
            return skills, None
    return rest + [new], new


SPECIALIST_ROLES = ("review", "security")
TEAM_NOTE = """
The agent is a TEAM: an orchestrator that spawns specialists (review, security) with written briefs, reads their
final messages (each with a "report") and writes the final review. Blame the right role: a coordination failure
(no or wrong specialist, a brief that omitted what mattered, a report's unverified items ignored, a correct finding
dropped from the final answer) is fixed by an `orchestrator` skill - a procedure for coordinating reviews of an
area; a specialist that looked in the wrong place is fixed by a skill of its role."""


class SkillLearner(InterventionLearner):
    """Evolves a skill library over the training PRs in chronological batches: serve -> post-mortem -> validated
    edits. Reuses the intervention learner's PR views, outcome-known rule and validation pool.

    `entry` is the agent the task goes to: "review" (a single reviewer) or "orchestrator" (the team; skills of
    every role are evolved, and each spawned specialist gets the index of its role's skills)."""

    system = "skills"  # how the library is served while learning (see serve)

    def __init__(self, *args, trace_of: TraceLookup, entry: str = "review", **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_of = trace_of
        self.entry = entry
        self.roles = (entry,) if entry != "orchestrator" else ("orchestrator", *SPECIALIST_ROLES)
        self.skills: list[Skill] = []
        self.edits: list[dict[str, Any]] = []
        self.sample = 0  # which independent run of the training episodes this epoch sees (0: the cached first)
        self.spent: Callable[[], bool] = lambda: False  # true once learning must stop (the library so far is kept)

    # ------------------------------------------------------------------ serving
    def serve(self, task: ReplayTask, system: str, skills: list[Skill] | None = None) -> tuple[str, tuple, tuple]:
        """(extra context for the entry agent, offered skills, ((role, context), ...) for spawned specialists).
        `tool` documents `hist reviews` only; `skills` adds the index of the skills offered for this PR's files;
        `rubric` inlines those skills' bodies (a mined rubric, `ospm/learning/rubric.py`) instead of offering them."""
        if system not in ("tool", "skills", "rubric"):
            return self.serve_context(task, system), (), ()
        lib = (self.skills if skills is None else skills) if system in ("skills", "rubric") else []
        files = self.view(task).files
        mine = offered(lib, files, self.entry)
        ctx = TOOLS_NOTE + (rubric_block(mine) if system == "rubric" else index_block(mine))
        if self.entry != "orchestrator":
            return ctx, tuple(mine), ()
        per_role = {r: offered(lib, files, r) for r in SPECIALIST_ROLES}
        role_ctx = tuple((r, "\n\n" + TOOLS_NOTE + index_block(o)) for r, o in per_role.items())
        allowed = list({s.name: s for s in mine + [s for o in per_role.values() for s in o]}.values())
        return "\n\n" + ctx, tuple(allowed), role_ctx

    # ------------------------------------------------------------------ post-mortem
    def postmortem_items(self, batch: list[ReplayTask], outs: list[tuple[float, dict | None]]) -> list[dict]:
        items = []
        for t, (_, out) in zip(batch, outs):
            mine = [c for c in (out or {}).get("comments") or [] if isinstance(c, dict)]
            if t.truth["positive"]:
                missed = self.missed(t, out)
                if not missed:
                    continue
            else:
                missed = []
                if not mine:
                    continue  # silent on a clean PR: nothing to learn
            job = self.serve(t, self.system)
            trace = self.trace_of(t, *job, **({"sample": self.sample} if self.sample else {}))
            digest = trace_digest(trace, sorted({c["path"] for c in missed}))
            if self.entry == "orchestrator" and trace:
                digest["team"] = team_digest(trace)
            items.append({"task": t.id, "positive": t.truth["positive"], "files": self.view(t).files[:25],
                          "offered": [s.name for s in job[1]], "missed": missed, "agent_comments": mine,
                          "digest": digest})
        return items

    def propose(self, items: list[dict]) -> list[dict]:
        if not items:
            return []
        lib = "\n\n".join(f"### {s.name} (role: {s.role}; paths: {', '.join(s.paths) or 'everywhere'})\ndescription: "
                          f"{s.description}\n{s.body}" for s in self.skills if s.status == "active") or "(empty)"
        blocks = []
        for it in items[:12]:
            d = it["digest"]
            lines = [f"## task {it['task']} - " + ("maintainers raised concerns" if it["positive"] else
                                                   "maintainers approved it without comments"),
                     f"changed files: {', '.join(it['files'])}",
                     f"skills offered: {', '.join(it['offered']) or 'none'}; loaded: {', '.join(d['loaded']) or 'none'}"
                     f"; steps used: {d['steps']}"]
            for c in it["missed"]:
                lines.append(f"- missed {c['path']}:{c['line']}: {c['body'][:400]} (agents opened this file: "
                             f"{'yes' if d['opened'].get(c['path']) else 'no'})")
            lines.append("final comments: " + (json.dumps([{k: m.get(k) for k in ("path", "line", "body")}
                                                           for m in it["agent_comments"]])[:800] or "none"))
            if "team" in d:
                lines.append("orchestrator actions:\n" + "\n".join(f"  {c}" for c in d["team"]["coordination"]))
                lines.append("specialists: " + json.dumps(d["team"]["specialists"])[:2000])
            lines.append("commands:\n" + "\n".join(f"  $ {c}" for c in d["commands"]))
            blocks.append("\n".join(lines))
        user = (f"# Current skill library\n{lib}\n\n# Post-mortems\n" + "\n\n".join(blocks)
                + f"\n\nPropose at most {self.cfg.max_candidates_per_step} skill edits.")
        system = POSTMORTEM_SYSTEM + (TEAM_NOTE if self.entry == "orchestrator" else "")
        try:
            d = self.llm.complete_json(system, user, _schema(self.roles), name="skill_postmortem", effort="medium")
        except (LLMError, BudgetExceeded):
            return []
        known = [it["task"] for it in items]
        edits = d["edits"][: self.cfg.max_candidates_per_step]
        for e in edits:
            e["source_tasks"] = resolve_sources(list(e.get("source_tasks") or []), known)[:8]
        return edits

    # ------------------------------------------------------------------ acceptance gate
    def validate_edit(self, e: dict[str, Any]) -> bool:
        new_lib, subject = apply_edit(self.skills, e)
        record = {"edit": {k: e.get(k) for k in ("op", "role", "part", "name", "description", "paths", "body",
                                                 "diagnosis", "source_tasks")}, "status": "rejected"}
        self.edits.append(record)
        if subject is None:
            record["validation"] = {"reason": "no-op or invalid edit"}
            return False
        lib_with_subject = new_lib if e["op"] != "delete" else self.skills
        source = self.source_prs(list(e.get("source_tasks") or []))
        past = [t for t in self.validation_candidates() if pr_key(t.id, t.pr) not in source]
        touched = [t for t in past if subject in offered(lib_with_subject, self.view(t).files, subject.role)]
        n_pos = max(1, self.cfg.validate_max_tasks // 2)
        pos, prs = [], set()
        for t in touched:
            if t.truth["positive"] and len(pos) < n_pos and pr_key(t.id, t.pr) not in prs:
                pos.append(t)
                prs.add(pr_key(t.id, t.pr))
        if not pos and e["op"] != "delete":
            record["validation"] = {"reason": "no independent support: offered on no other past PR with a "
                                              "maintainer concern"}
            return False
        neg = [t for t in touched if not t.truth["positive"]][: self.cfg.validate_max_tasks - len(pos)]
        tasks = pos + neg
        if not tasks:  # a deletion of a skill never offered on past PRs: nothing to lose
            self.skills = new_lib
            record.update(status="adopted", validation={"reason": "never offered on past PRs"})
            return True
        jobs = [(t, *self.serve(t, "skills", self.skills)) for t in tasks]
        jobs += [(t, *self.serve(t, "skills", new_lib)) for t in tasks]
        results = self._run_many(jobs)
        before, after = [r[0] for r in results[: len(tasks)]], [r[0] for r in results[len(tasks):]]
        n_comments = [len((r[1] or {}).get("comments") or []) for r in results]
        k = len(pos)
        pos_gain = (sum(after[:k]) - sum(before[:k])) / k if k else 0.0
        fp_inc = (sum(n_comments[len(tasks) + k:]) - sum(n_comments[k:len(tasks)])) / len(neg) if neg else 0.0
        loaded = sum(1 for job in jobs[len(tasks):] if subject.name in loaded_skills(self.trace_of(*job) or {}))
        if e["op"] == "delete":
            ok = pos_gain >= 0 and fp_inc <= 0
        else:
            ok = pos_gain >= self.cfg.min_gain and fp_inc <= self.cfg.max_fp_increase
        record["validation"] = {"tasks": [t.id for t in tasks], "pos_gain": round(pos_gain, 4),
                                "fp_increase": round(fp_inc, 3), "n_pos": k, "n_neg": len(neg),
                                "loaded": f"{loaded}/{len(tasks)}" if e["op"] != "delete" else None}
        if ok:
            self.skills = new_lib
            record["status"] = "adopted"
        return ok

    # ------------------------------------------------------------------ loop
    def learn(self, train: list[ReplayTask], epochs: int = 1) -> dict[str, Any]:
        """Chronological batches of `train`, `epochs` times. Epoch n > 1 reruns the training PRs as fresh samples
        (the library has changed, and a rerun of an unchanged one would otherwise replay the cached outcome)."""
        ordered = sorted(train, key=lambda t: (t.as_of, t.id))
        pending: list[dict] = []  # post-mortems not yet addressed by an adopted edit (rolling pool)
        for epoch in range(epochs):
            self.sample = epoch
            for i in range(0, len(ordered), self.cfg.batch):
                if self.spent():
                    self.log(f"[{self.system}] learning budget reached: stopping in epoch {epoch + 1}")
                    self.sample = 0
                    return {"skills": [asdict(s) for s in self.skills], "edits": self.edits, "history": self.history}
                pending = self._step(ordered[i:i + self.cfg.batch], pending,
                                     f"step {i // self.cfg.batch + 1}" if epochs == 1 else
                                     f"epoch {epoch + 1} step {i // self.cfg.batch + 1}")
        self.sample = 0
        return {"skills": [asdict(s) for s in self.skills], "edits": self.edits, "history": self.history}

    def _step(self, batch: list[ReplayTask], pending: list[dict], label: str) -> list[dict]:
        outs = self._run_many([(t, *self.serve(t, self.system), *((None, self.sample) if self.sample else ()))
                               for t in batch])
        for t, (score, out) in zip(batch, outs):
            self.history.append({"task": t.id, "score": score, **concern_metrics(t, out),
                                 "skills": len(self.skills), "epoch": self.sample + 1})
        if not self.sample:
            self.seen.extend(batch)
            self.now = max(self.now, batch[-1].as_of)
        items = self.postmortem_items(batch, outs)
        pending = (pending + items)[-self.cfg.miss_pool:]
        edits = self.propose(pending)
        adopted = 0
        for e in edits:
            if self.validate_edit(e):
                adopted += 1
                done = set(e.get("source_tasks") or [])
                pending = [it for it in pending if it["task"] not in done]
        self.log(f"[{self.system}] {label}: {len(batch)} tasks, {len(items)} post-mortems, "
                 f"{len(edits)} edits, {adopted} adopted, {len(self.skills)} skills")
        return pending
