"""A review rubric mined from a repository's own review history.

Post-mortem skills learn from one missed concern at a time, so they write procedures for the area of that PR and
never see a rule that maintainers apply across areas. This module starts from the rules themselves:

1. pairs of earlier maintainer concerns (before the test period, from different PRs) that apply the same
   project-specific rule are found with the recurrence judges (`ospm/replay/recurrence.py`);
2. linked concerns are clustered, and clusters seen on at least `min_prs` PRs are kept;
3. an LLM writes each cluster as a checkable rubric item, grouped into areas with file globs;
4. each area becomes a skill (`rubric-<area>`): offered on matching PRs (system `skills`), or inlined into the
   reviewer's context (system `rubric` in `ospm/learning/experiment.py`);
5. post-mortems of the training PRs then add, edit or remove one item at a time, each validated on other earlier
   PRs (`RubricLearner`, system `evolved_rubric`);
6. for the team, each area can instead be its own specialist (`rubric_team`, `ospm/learning/team_evolution.py`).

    python -m ospm.learning.rubric --data data/home-assistant --out data/results/rubric_ha
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable

from ..llm import BudgetExceeded, LLMError, make_llm
from ..llm.base import UsageMeter
from ..replay.concerns import _near
from ..replay.recurrence import LOOSE, candidates, concerns_of, judges
from ..replay.semantic import THRESHOLD, Matcher
from ..replay.tasks import ReplayTask, load_tasks
from ..team.roles import TeamGenome, specialist
from ..text import TfidfIndex
from .interventions import pr_key, resolve_sources
from .postmortem import SkillLearner
from .skills import Skill, offered, save, slug

SYSTEM = """You turn recurring maintainer review comments of one open-source project into a review rubric.

Each cluster below groups review comments that maintainers made on DIFFERENT pull requests and that apply the same
project-specific rule. Write the rubric a new reviewer of this project should check every pull request against.

- One item per reusable rule. Merge clusters that state the same rule; split a cluster that mixes rules; drop a
  cluster whose comments do not share a rule that applies to future pull requests, or that is general advice any
  experienced reviewer would give ("add a test", "handle errors").
- `check` is a concrete, verifiable criterion phrased as what to look for in the diff (files, names, keys, APIs,
  patterns), and what the maintainer asks for when it is violated. Name the project's own helpers, files and keys.
- `applies_when` says which changes the item is relevant to, so a reviewer can skip it otherwise.
- Group items into a few areas (e.g. config flows, translations/strings, tests, entities, manifests and quality
  scale). Give each area fnmatch globs over changed file paths ("homeassistant/components/*/strings.json",
  "tests/components/*/*.py"); an item may sit in only one area.
- Cite the clusters each item comes from."""


def _schema() -> dict:
    item = {"type": "object", "properties": {
        "rule": {"type": "string"}, "applies_when": {"type": "string"}, "check": {"type": "string"},
        "example": {"type": "string"}, "clusters": {"type": "array", "items": {"type": "integer"}}},
        "required": ["rule", "applies_when", "check", "example", "clusters"], "additionalProperties": False}
    area = {"type": "object", "properties": {
        "name": {"type": "string"}, "description": {"type": "string"},
        "paths": {"type": "array", "items": {"type": "string"}}, "items": {"type": "array", "items": item}},
        "required": ["name", "description", "paths", "items"], "additionalProperties": False}
    return {"type": "object", "properties": {"areas": {"type": "array", "items": area}}, "required": ["areas"],
            "additionalProperties": False}


def test_start(data: Path) -> str:
    return min(t.as_of for t in load_tasks(data / "tasks.jsonl") if t.kind == "concern" and t.split == "test")


def _components(links: list[tuple[int, int]]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in links:
        parent[root(i)] = root(j)
    groups = defaultdict(set)
    for i, j in links:
        groups[root(i)] |= {i, j}
    return list(groups.values())


def clusters(data: Path, sample: int = 1500, k: int = 8, min_prs: int = 3, max_prs: int = 40,
             seed: int = 0) -> list[list[dict]]:
    """Concerns from before the test period, grouped by shared project-specific rule; largest first.

    Groups are connected components of judged links. Loose links chain unrelated rules into one giant component,
    so a component on more than `max_prs` PRs is split again using only its stronger links."""
    cutoff = test_start(data)
    cs = sorted((c for c in concerns_of(data) if c["as_of"] < cutoff), key=lambda c: c["as_of"])
    targets = list(range(len(cs)))
    random.Random(seed).shuffle(targets)
    rule, spec = judges(data)
    pairs = candidates(cs, targets[:sample], k, seed)
    links = [(ij, p) for ij, p in zip(pairs, rule.probabilities([("", cs[j], cs[i]) for i, j in pairs]))
             if p >= LOOSE]
    links = [(ij, p) for (ij, p), q in zip(links, spec.probabilities([("", cs[j], cs[i]) for (i, j), _ in links]))
             if q >= 0.5]
    prs = lambda g: len({cs[i]["pr"] for i in g})  # noqa: E731
    out, todo = [], [(set(range(len(cs))), LOOSE)]
    while todo:
        members, cut = todo.pop()
        for g in _components([ij for ij, p in links if p >= cut and ij[0] in members and ij[1] in members]):
            stronger = [c for c in (0.8, 0.9, 0.95) if c > cut]
            if prs(g) > max_prs and stronger:
                todo.append((g, stronger[0]))
            elif prs(g) >= min_prs:
                out.append([cs[i] for i in sorted(g)])
    return sorted(out, key=lambda g: -len({c["pr"] for c in g}))


def write(llm, groups: list[list[dict]], per_cluster: int = 6) -> dict:
    blocks = []
    for n, g in enumerate(groups):
        prs = sorted({c["pr"] for c in g})
        shown = random.Random(n).sample(g, min(per_cluster, len(g)))
        lines = [f"- PR #{c['pr']} {c['path']}: {' '.join(c['body'].split())[:400]}" for c in shown]
        blocks.append(f"### Cluster {n} ({len(g)} comments on {len(prs)} PRs)\n" + "\n".join(lines))
    return llm.complete_json(SYSTEM, "\n\n".join(blocks), _schema(), name="review_rubric", effort="high",
                             max_tokens=32000)


HEADER = ("Check the change against each item that applies; for each one it violates, raise a review comment on the "
          "offending line asking for what the item says (maintainers of this project ask for these repeatedly). Skip "
          "items that do not apply.")
ITEM_KEYS = ("rule", "applies_when", "check", "example")


def render(items: list[dict]) -> str:
    lines = [HEADER, ""]
    for n, it in enumerate(items, 1):
        lines += [f"{n}. **{it['rule']}**", f"   - Applies when: {it['applies_when']}", f"   - Check: {it['check']}",
                  f"   - Maintainers have written: {it['example']}"]
    return "\n".join(lines)


def area_name(name: str) -> str:
    s = slug(name)
    return slug(s if s.startswith("rubric-") else f"rubric-{s}")


def area_skill(name: str, description: str, paths: list[str], items: list[dict], provenance=(),
               source: str = "mined rubric") -> Skill:
    return Skill(name=area_name(name), description=description, body=render(items), paths=list(paths),
                 provenance=sorted(set(provenance)), validation={"source": source, "items": len(items)},
                 items=[dict(it) for it in items])


def to_skills(rubric: dict, groups: list[list[dict]]) -> list[Skill]:
    skills = []
    for area in rubric["areas"]:
        if not area["items"]:
            continue
        prov = [f"pr:{c['pr']}" for it in area["items"] for k in it["clusters"] if 0 <= k < len(groups)
                for c in groups[k]]
        skills.append(area_skill(area["name"], area["description"], area["paths"], area["items"], prov))
    return skills


# ---------------------------------------------------------------------------------------------- post-mortem edits
POSTMORTEM_SYSTEM = """You run post-mortems on a code-review agent for one repository and maintain its REVIEW RUBRIC:
checklist items, grouped into areas with file globs, that the agent gets in its context on every pull request
touching an area. An item states a project-specific rule that maintainers apply on many pull requests: `rule`,
`applies_when` (which changes it is relevant to), `check` (what to look for in the diff - the project's own helpers,
files, keys and patterns - and what to ask for when it is violated) and `example` (a maintainer's words).

For each reviewed pull request you get what the maintainers raised (or that they approved it as-is), the agent's
comments, the rubric areas it was given, and for each missed concern the most similar comments maintainers made on
EARLIER pull requests. Diagnose each miss or false alarm, then propose rubric edits:
- `add_item`: the concern applies a rule that recurs - earlier comments show maintainers asking the same of other
  pull requests - and no item states it. Put it in the area whose globs match the files (`area` = that area's name),
  or start an area (`area` = a new name, with `area_description` and `area_paths`, fnmatch globs over file paths).
- `edit_item`: an item covers the concern but the agent did not raise it - make its `check` or `applies_when` concrete
  enough to recognise the violation in a diff; or an item caused a false alarm - narrow it. Give the full new item.
- `remove_item`: an item misleads the agent and cannot be fixed by narrowing it.
Never add items for one-off judgement calls, for concerns that no earlier comment repeats, or for general advice any
experienced reviewer gives ("add a test", "handle errors"). `index` is the item's number in its area for edit_item
and remove_item, else -1. Each edit is validated on other earlier pull requests. Return JSON only."""


def _edit_schema() -> dict:
    e = {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["add_item", "edit_item", "remove_item"]}, "area": {"type": "string"},
        "index": {"type": "integer"}, **{k: {"type": "string"} for k in ITEM_KEYS},
        "area_description": {"type": "string"}, "area_paths": {"type": "array", "items": {"type": "string"}},
        "diagnosis": {"type": "string"}, "source_tasks": {"type": "array", "items": {"type": "string"}}},
        "required": ["op", "area", "index", *ITEM_KEYS, "area_description", "area_paths", "diagnosis",
                     "source_tasks"], "additionalProperties": False}
    return {"type": "object", "properties": {"edits": {"type": "array", "items": e}}, "required": ["edits"],
            "additionalProperties": False}


def apply_item_edit(skills: list[Skill], e: dict) -> tuple[list[Skill], Skill | None, dict | None]:
    """(library after the edit, the area to validate - its new version, or the old one when its last item is
    removed -, the item added, edited or removed). Invalid or no-op edits give (skills, None, None)."""
    try:
        name = area_name(e.get("area") or "")
    except ValueError:
        return skills, None, None
    old = next((s for s in skills if s.name == name and s.status == "active"), None)
    rest = [s for s in skills if s is not old]
    item = {k: " ".join(str(e.get(k) or "").split())[:600] for k in ITEM_KEYS}
    items = list(old.items) if old else []
    i = int(e.get("index") or 0) - 1
    if e["op"] == "add_item":
        if not all(item.values()) or len(items) >= 20:
            return skills, None, None
        if old is None:
            paths = [p for p in e.get("area_paths") or [] if p][:8]
            desc = (e.get("area_description") or "").strip()[:300]
            if not (paths and desc):
                return skills, None, None
            new = area_skill(name, desc, paths, [item], e.get("source_tasks") or [], source="post-mortem")
            return rest + [new], new, item
        items.append(item)
    elif old is None or not 0 <= i < len(items):
        return skills, None, None
    elif e["op"] == "edit_item":
        item = {k: item[k] or items[i][k] for k in ITEM_KEYS}
        if all(item[k] == items[i][k] for k in ITEM_KEYS):
            return skills, None, None
        items[i] = {**items[i], **item}
    else:  # remove_item
        item = items.pop(i)
        if not items:
            return rest, old, item
    new = replace(old, items=items, body=render(items), version=old.version + 1,
                  provenance=sorted(set(old.provenance) | set(e.get("source_tasks") or [])),
                  validation={"source": old.validation.get("source", "post-mortem"), "items": len(items)})
    return rest + [new], new, item


class ItemJudge(Matcher):
    """P(a maintainer's review comment applies a rubric item), cached like Matcher: the past PRs an item would have
    caught, and the concerns on them it has to raise."""

    QUESTION = ("Is the maintainer's review comment an application of this review rule: would a reviewer checking the "
                "pull request against the rule have raised the comment's point?")
    YES = "Applies the rule: the comment asks for what the rule requires, or objects to what the rule forbids."
    NO = ("Does not: a different problem or request, or only the same broad topic, so that following the rule would "
          "not lead a reviewer to the comment's point.")

    def _state(self, path: str, concern: dict, item: dict) -> dict:
        return {"review_rule": str(item.get("body", ""))[:1500],
                "maintainer_comment": {"file": path, "text": str(concern.get("body", ""))[:1500]}}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


class RubricLearner(SkillLearner):
    """Evolves a review rubric (areas of items, served inlined: system `rubric`) by post-mortems of the training PRs,
    one item at a time.

    An added or edited item is validated on other earlier PRs where its area is offered: positives are PRs whose
    maintainer concerns apply the item (`judge`, an ItemJudge; else the most TF-IDF-similar concern of each PR), and
    at least `min_support` of them must exist before anything is replayed; negatives are clean PRs. Each PR is run
    `samples` times with and without the edit, and the edit must raise the target concerns more often (by at least
    `min_hit_gain` hits over all runs) without more comments on the clean PRs. Rejected edits are shown to later
    post-mortems, and near-copies of them are rejected without a replay."""

    system = "rubric"

    def __init__(self, *args, min_sim: float = 0.12, judge: Matcher | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_sim = min_sim
        self.judge = judge
        self.rejected: list[dict] = []
        self._index: TfidfIndex | None = None

    def similar(self, text: str, k: int = 200, min_sim: float | None = None) -> list[tuple[ReplayTask, dict, float]]:
        """Maintainer concerns of earlier tasks (training and history), most similar to `text` first."""
        if self._index is None:
            self._index = TfidfIndex()
            for t in {t.id: t for t in self.past + self.seen}.values():
                if t.truth["positive"]:
                    for n, c in enumerate(t.truth["concerns"]):
                        if not c["nit"]:
                            self._index.add(f"{t.id}#{n}", f"{c['path']} {c['body']}", (t, c))
        hits = self._index.query(text, k=k, min_sim=self.min_sim if min_sim is None else min_sim)
        return [(*payload, sim) for _, sim, payload in hits]

    def support(self, item: dict, eligible: Callable[[ReplayTask], bool], exclude: set, n: int) -> list[tuple]:
        """Up to `n` (task, concerns) of distinct other `eligible` past PRs: their concerns that apply `item`. All of
        history is searched (not only the validation pool): the PRs a rule recurs on are spread over years."""
        text = (f"{item['rule']} Applies when: {item['applies_when']} Check: {item['check']} "
                f"Maintainers have written: {item['example']}")
        groups: dict[str, tuple[ReplayTask, list[dict]]] = {}
        task_of_pr: dict = {}
        for u, c, _ in self.similar(text, k=400, min_sim=self.min_sim / 3 if self.judge else None):
            key = pr_key(u.id, u.pr)
            if key in exclude or task_of_pr.get(key, u.id) != u.id:
                continue
            if u.id not in groups and (len(groups) >= 4 * n or not eligible(u)):
                continue
            task_of_pr[key] = u.id
            groups.setdefault(u.id, (u, []))[1].append(c)
        if self.judge is None:
            return [(u, cs[:1]) for u, cs in groups.values()][:n]
        probs = iter(self.judge.probabilities([(c["path"], c, {"body": text}) for _, cs in groups.values()
                                                for c in cs]))
        out = [(u, hit) for u, cs in groups.values() if (hit := [c for c in cs if next(probs) >= THRESHOLD])]
        return out[:n]

    def hits(self, targets: list[dict], out: dict | None) -> float:
        """Share of `targets` (maintainer concerns) the output raises."""
        if self.matcher is not None:
            return sum(p >= THRESHOLD for p in self.matcher.concern_scores(targets, out)) / len(targets)
        mine = [m for m in (out or {}).get("comments") or [] if isinstance(m, dict)]
        return sum(any(_near(m, c) for m in mine) for c in targets) / len(targets)

    def rejected_like(self, area: str, item: dict) -> dict | None:
        """An earlier rejected edit of the same area stating (nearly) the same rule."""
        w = _words(item.get("rule") or "")
        for r in self.rejected:
            v = _words(r["rule"])
            if r["area"] == area and w and len(w & v) / len(w | v) >= 0.5:
                return r
        return None

    def postmortem_items(self, batch: list[ReplayTask], outs: list[tuple[float, dict | None]]) -> list[dict]:
        items = super().postmortem_items(batch, outs)
        by_id = {t.id: t for t in batch}
        for it in items:
            t = by_id[it["task"]]
            for c in it["missed"]:
                c["earlier"] = [f"PR #{u.pr} {d['path']}: {' '.join(d['body'].split())[:200]}"
                                for u, d, _ in self.similar(f"{c['path']} {c['body']}", k=40)
                                if u.pr != t.pr and u.as_of < t.as_of][:3]
        return items

    def propose(self, items: list[dict]) -> list[dict]:
        if not items:
            return []
        areas = [f"### {s.name} (paths: {', '.join(s.paths)})\n{s.description}\n" + "\n".join(
            f"{n}. {it['rule']} | applies when: {it['applies_when']} | check: {it['check']}"
            for n, it in enumerate(s.items, 1)) for s in self.skills if s.status == "active"] or ["(empty)"]
        blocks = []
        for it in items[:12]:
            lines = [f"## task {it['task']} - " + ("maintainers raised concerns" if it["positive"] else
                                                   "maintainers approved it without comments"),
                     f"changed files: {', '.join(it['files'])}",
                     f"rubric areas given: {', '.join(it['offered']) or 'none'}"]
            for c in it["missed"]:
                lines.append(f"- missed {c['path']}:{c['line']}: {' '.join(c['body'].split())[:400]}")
                lines += [f"    earlier: {x}" for x in c.get("earlier") or []] or ["    earlier: (no similar comment)"]
            lines.append("agent comments: " + (json.dumps([{k: m.get(k) for k in ("path", "line", "body")}
                                                            for m in it["agent_comments"]])[:800] or "none"))
            blocks.append("\n".join(lines))
        rejected = ("\n\n# Edits already rejected\nEach was validated on other earlier pull requests and did not help."
                    " Do not propose them again; a similar rule only with a concretely different check.\n"
                    + "\n".join(f"- {r['op']} {r['area']}: {r['rule']} ({r['why']})" for r in self.rejected[-15:])
                    if self.rejected else "")
        user = ("# Current rubric\n" + "\n\n".join(areas) + "\n\n# Post-mortems\n" + "\n\n".join(blocks) + rejected
                + f"\n\nPropose at most {self.cfg.max_candidates_per_step} rubric edits.")
        try:
            d = self.llm.complete_json(POSTMORTEM_SYSTEM, user, _edit_schema(), name="rubric_postmortem",
                                       effort="medium")
        except (LLMError, BudgetExceeded):
            return []
        known = [it["task"] for it in items]
        edits = d["edits"][: self.cfg.max_candidates_per_step]
        for e in edits:
            e["source_tasks"] = resolve_sources(list(e.get("source_tasks") or []), known)[:8]
        return edits

    def _reject(self, record: dict, area: str, e: dict, item: dict, why: str, **validation) -> bool:
        record["validation"] = {"reason": why, **validation}
        self.rejected.append({"area": area, "op": e["op"], "rule": item.get("rule") or "", "why": why})
        return False

    def validate_edit(self, e: dict) -> bool:
        new_lib, subject, item = apply_item_edit(self.skills, e)
        record = {"edit": dict(e), "status": "rejected"}
        self.edits.append(record)
        if subject is None:
            record["validation"] = {"reason": "no-op or invalid edit"}
            return False
        removal = e["op"] == "remove_item"
        if not removal and (dup := self.rejected_like(subject.name, item)):
            record["validation"] = {"reason": "repeats a rejected edit", "rejected_rule": dup["rule"]}
            return False
        source = self.source_prs(list(e.get("source_tasks") or []))
        lib = self.skills if removal else new_lib
        touched = {t.id: t for t in self.validation_candidates() if pr_key(t.id, t.pr) not in source
                   and subject in offered(lib, self.view(t).files, subject.role)}
        n_pos = max(1, self.cfg.validate_max_tasks // 2)
        if removal:  # removing an item must not lose any concern it helped raise
            pos, prs = [], set()
            for t in touched.values():
                if t.truth["positive"] and len(pos) < n_pos and pr_key(t.id, t.pr) not in prs:
                    pos.append((t, [c for c in t.truth["concerns"] if not c["nit"]] or t.truth["concerns"]))
                    prs.add(pr_key(t.id, t.pr))
        else:
            pos = self.support(item, lambda t: t.as_of <= self.now and subject in offered(
                lib, self.view(t).files, subject.role), source, n_pos)
            if len(pos) < max(1, self.cfg.min_support):
                return self._reject(record, subject.name, e, item, f"not recurring: it applies to maintainer concerns "
                                    f"on {len(pos)} other past PR(s) in the area", n_support=len(pos))
        neg = [t for t in touched.values() if not t.truth["positive"]][: self.cfg.validate_max_tasks - len(pos)]
        tasks = [t for t, _ in pos] + neg
        if not tasks:
            self.skills = new_lib
            record.update(status="adopted", validation={"reason": "area never offered on past PRs"})
            return True
        k = max(1, self.cfg.samples)
        results = self._run_many([(t, *self.serve(t, "rubric", L), *((None, s) if s else ()))
                                  for L in (self.skills, new_lib) for s in range(k) for t in tasks])
        runs = [[results[(a * k + s) * len(tasks):(a * k + s + 1) * len(tasks)] for s in range(k)] for a in (0, 1)]
        hits = [sum(self.hits(cs, r[i][1]) for r in arm for i, (_, cs) in enumerate(pos)) for arm in runs]
        fps = [sum(len((r[len(pos) + j][1] or {}).get("comments") or []) for r in arm for j in range(len(neg)))
               / (k * len(neg)) if neg else 0.0 for arm in runs]
        gain, fp_inc = hits[1] - hits[0], fps[1] - fps[0]
        validation = {"tasks": [t.id for t in tasks], "n_pos": len(pos), "n_neg": len(neg), "samples": k,
                      "hits_before": round(hits[0], 2), "hits_after": round(hits[1], 2), "trials": k * len(pos),
                      "fp_before": round(fps[0], 3), "fp_after": round(fps[1], 3)}
        if removal:
            ok = gain >= 0 and fp_inc <= 0
        else:
            ok = gain > 0 and gain >= self.cfg.min_hit_gain and fp_inc <= self.cfg.max_fp_increase
        if not ok:
            return self._reject(record, subject.name, e, item, (
                f"target concerns raised {hits[0]:.1f} -> {hits[1]:.1f} times in {k * len(pos)} runs on {len(pos)} "
                f"PRs they apply to; comments per clean-PR run {fps[0]:.2f} -> {fps[1]:.2f}"), **validation)
        record.update(status="adopted", validation=validation)
        self.skills = new_lib
        return True


# ---------------------------------------------------------------------------------------------- team
def rubric_team(team: TeamGenome, skills: list[Skill]) -> TeamGenome:
    """`team` plus one specialist per rubric area, checking a pull request against that area's items only."""
    g = team.clone()
    for s in skills:
        if s.status != "active":
            continue
        name = s.name.replace("-", "_")[:32]
        focus = (f"Review the pull request at the commit in your workspace ONLY against this project's review rubric "
                 f"for one area ({s.description}). Read the diff and the code it touches; comment only where the "
                 f"change violates an item that applies, on the offending file and line, and approve otherwise.\n\n"
                 f"{s.body}")
        g.roles[name] = specialist(name, focus, description=(
            f"Checks a pull request against the project's review rubric for: {s.description} Spawn it when the "
            f"change touches {', '.join(s.paths)}.")[:400])
    g.origin, g.notes = "rubric", f"rubric specialists: {', '.join(r for r in g.roles if r.startswith('rubric_'))}"
    return g


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--llm", default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sample", type=int, default=1500, help="history concerns whose earlier matches are judged")
    ap.add_argument("--min-prs", type=int, default=3)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="output prefix: <out>.json (rubric) and <out>_skills.json")
    a = ap.parse_args(argv)
    data = Path(a.data)
    groups = clusters(data, a.sample, min_prs=a.min_prs)
    print(f"[rubric] {len(groups)} clusters on >= {a.min_prs} PRs (sizes "
          f"{[len({c['pr'] for c in g}) for g in groups[:20]]}...)", flush=True)
    meter = UsageMeter(a.budget)
    rubric = write(make_llm(a.llm, a.model, meter=meter), groups)
    skills = to_skills(rubric, groups)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps({"test_start": test_start(data), "rubric": rubric, "clusters": [
        [{k: c[k] for k in ("pr", "path", "body", "as_of")} for c in g] for g in groups]}, indent=1),
        encoding="utf-8")
    save(skills, f"{a.out}_skills.json")
    print(f"[rubric] {sum(len(s.body.splitlines()) for s in skills)} lines, {len(skills)} areas, "
          f"{sum(s.validation['items'] for s in skills)} items, ${meter.cost_usd:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
