"""Learning *where to intervene*: validated, scoped interventions induced from maintainer review corrections.

An Intervention is a small structured artifact - (trigger, check, action, stage, scope) - rather than a free-text
memory. It is induced from review concerns the agent missed on *earlier* PRs, validated by replaying the PRs its
trigger applies to (it must recover concerns without adding false alarms on PRs maintainers approved as-is), and
only then adopted. At serving time only interventions whose trigger matches the PR are put in front of the
reviewer.

Baselines that share the same data:
* `none`   - the fixed review agent;
* `memory` - an ACE-style raw lesson memory: every missed concern is stored verbatim and the most similar
             lessons are retrieved for each PR, with no structure, scoping or validation.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..llm import LLM, BudgetExceeded, LLMError
from ..replay.concerns import concern_metrics
from ..replay.tasks import ReplayTask
from ..replay.workspace import ReplayRepo
from ..text import TfidfIndex

STAGES = ("pr_creation", "review")


@dataclass
class Intervention:
    id: str
    trigger: str  # natural-language applicability condition
    paths: list[str]  # glob patterns over changed files ([] = any file)
    keywords: list[str]  # tokens in the added diff lines that indicate applicability ([] = no keyword filter)
    check: str  # what to verify / what evidence to gather
    action: str  # what to tell the author when the check fails
    stage: str = "review"
    provenance: list[str] = field(default_factory=list)  # task ids it was induced from
    status: str = "candidate"  # candidate | active | rejected
    validation: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        scope = f" (files: {', '.join(self.paths)})" if self.paths else ""
        return f"- [{self.id}] WHEN {self.trigger}{scope}: CHECK {self.check}; IF NOT, ASK: {self.action}"


class PRView:
    """What an intervention trigger can look at: changed files and added lines (available at the event time).

    The added lines are read lazily: in a blobless mirror they cost a fetch of the changed files' contents, and a
    trigger whose path scope does not match never needs them.
    """

    def __init__(self, files: list[str], added: str = "", loader: Callable[[], str] | None = None):
        self.files = files
        self._added = added
        self._loader = loader

    @property
    def added(self) -> str:
        if self._loader is not None:
            self._added, self._loader = self._loader()[:200_000], None
        return self._added

    @classmethod
    def of(cls, repo: ReplayRepo, task: ReplayTask) -> "PRView":
        base = task.meta.get("base") or ""
        git = ["git", "--git-dir", str(repo.git_dir)]
        # tree-only: no rename detection, which would compare (and so fetch) file contents
        files = subprocess.run([*git, "diff", "--no-renames", "--name-only", f"{base}..{task.sha}"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "GIT_NO_LAZY_FETCH": "1"}).stdout.split()

        def added() -> str:
            diff = subprocess.run([*git, "diff", "--no-renames", "-U0", f"{base}..{task.sha}"], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace").stdout
            return "\n".join(l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))

        return cls(files, loader=added)


_PR_ID = re.compile(r"(?:^|:)(\d+)@")


def pr_key(task_id: str, pr: int | None = None) -> int | str:
    """The pull request a task id (or a proposer's rendering of one) belongs to; the id itself when unknown."""
    if pr is not None:
        return pr
    m = _PR_ID.search(task_id)
    return int(m.group(1)) if m else task_id


def resolve_sources(sources: list[str], known: list[str]) -> list[str]:
    """Map the proposer's source ids onto real task ids (models tend to drop prefixes such as 'concern:')."""
    out: list[str] = []
    for s in sources:
        s = s.strip()
        hits = [k for k in known if s == k] or [k for k in known if s and s in k]
        out.extend(h for h in (hits or [s]) if h not in out)
    return out


def applies(iv: Intervention, view: PRView) -> bool:
    if iv.paths and not any(fnmatch.fnmatch(f, pat) for f in view.files for pat in iv.paths):
        return False
    if iv.keywords and not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])", view.added)
                               for k in iv.keywords):
        return False
    return True


def select(interventions: list[Intervention], view: PRView, k: int = 5) -> list[Intervention]:
    hits = [iv for iv in interventions if iv.status == "active" and applies(iv, view)]
    hits.sort(key=lambda iv: (-(iv.validation.get("gain", 0.0)), iv.id))
    return hits[:k]


def context_block(selected: list[Intervention]) -> str:
    if not selected:
        return ""
    return ("\n\nProject-specific checks learned from earlier reviews in this repository (each was validated on "
            "past pull requests). Apply a check only if its condition really holds for this change; do not "
            "comment just because a check exists:\n" + "\n".join(iv.render() for iv in selected))


# ---------------------------------------------------------------------------------------------- induction
PROPOSE_SYSTEM = """You turn maintainer code-review corrections into reusable, *scoped* review interventions for one
repository. You see review concerns that a reviewing agent missed (the maintainer raised them, the agent did
not), with the changed files and the agent's own comments. Propose interventions that would make a reviewer raise
such concerns on FUTURE pull requests where the same condition holds - and stay silent elsewhere.

Rules:
- Generalise from the concern to a condition about the change (which files/patterns), not to this PR.
- `paths`: glob patterns (fnmatch, e.g. "vllm/entrypoints/*.py", "tests/*") narrowing where it applies; [] only
  if the concern is genuinely repository-wide. `keywords`: identifiers/tokens that appear in ADDED lines when the
  condition holds (e.g. "torch.compile", "deprecated"); [] if none fits. Over-broad triggers will be rejected.
- `check`: what to verify in the diff/code; `action`: the question or request to put to the author.
- Prefer RECURRING patterns: an intervention supported by missed concerns from two or more different PRs is
  far more valuable than one fitted to a single PR. It will be validated only on OTHER pull requests than the
  ones it came from, so triggers naming one PR's specific constants or files will fail.
- Skip concerns that are one-off judgement calls, pure taste, or depend on information not in the change; it is
  fine to return fewer interventions than concerns (or none).
Return JSON only."""


def _propose_schema() -> dict:
    iv = {"type": "object", "properties": {
        "trigger": {"type": "string"}, "paths": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}}, "check": {"type": "string"},
        "action": {"type": "string"}, "stage": {"type": "string", "enum": list(STAGES)},
        "source_tasks": {"type": "array", "items": {"type": "string"}}},
        "required": ["trigger", "paths", "keywords", "check", "action", "stage", "source_tasks"],
        "additionalProperties": False}
    return {"type": "object", "properties": {"interventions": {"type": "array", "items": iv}},
            "required": ["interventions"], "additionalProperties": False}


def propose(llm: LLM, misses: list[dict[str, Any]], existing: list[Intervention], max_new: int = 6,
            effort: str | None = "medium") -> list[Intervention]:
    """misses: [{"task": id, "files": [...], "concern": {...}, "agent_comments": [...]}]"""
    if not misses:
        return []
    have = "\n".join(iv.render() for iv in existing if iv.status == "active") or "(none yet)"
    blocks = []
    for m in misses[:16]:
        c = m["concern"]
        blocks.append(f"## task {m['task']}\nchanged files: {', '.join(m['files'][:25])}\nmaintainer concern at "
                      f"{c['path']}:{c['line']}: {c['body'][:500]}\nagent's comments: "
                      f"{json.dumps(m['agent_comments'])[:600]}")
    user = (f"# Interventions already active\n{have}\n\n# Missed concerns\n" + "\n\n".join(blocks)
            + f"\n\nPropose at most {max_new} NEW interventions (do not duplicate active ones).")
    try:
        d = llm.complete_json(PROPOSE_SYSTEM, user, _propose_schema(), name="propose_interventions", effort=effort)
    except (LLMError, BudgetExceeded):
        return []
    known = [m["task"] for m in misses]
    out = []
    for spec in d["interventions"][:max_new]:
        key = json.dumps([spec["trigger"], sorted(spec["paths"]), sorted(spec["keywords"])], sort_keys=True)
        iid = "iv-" + hashlib.sha256(key.encode()).hexdigest()[:8]
        if any(iv.id == iid for iv in existing):
            continue
        out.append(Intervention(iid, spec["trigger"][:300], [p for p in spec["paths"] if p][:8],
                                [k for k in spec["keywords"] if k.strip()][:8], spec["check"][:400],
                                spec["action"][:400], spec["stage"],
                                resolve_sources(list(spec["source_tasks"]), known)[:8]))
    return out


# ---------------------------------------------------------------------------------------------- memory baseline
class LessonMemory:
    """ACE-style baseline: raw lessons ('on <path> a maintainer asked: ...') retrieved by similarity."""

    def __init__(self):
        self.index = TfidfIndex()
        self.n = 0

    def add(self, task_id: str, concern: dict) -> None:
        text = f"{concern['path']} {concern['body']}"
        self.index.add(f"{task_id}#{self.n}", text, f"On {concern['path']} a maintainer asked: {concern['body'][:300]}")
        self.n += 1

    def context(self, view: PRView, k: int = 5) -> str:
        if not self.n:
            return ""
        hits = self.index.query(" ".join(view.files) + "\n" + view.added[:20000], k=k, min_sim=0.05)
        if not hits:
            return ""
        return ("\n\nLessons from earlier reviews in this repository (may or may not apply):\n"
                + "\n".join(f"- {payload}" for _, _, payload in hits))


# ---------------------------------------------------------------------------------------------- learning loop
@dataclass
class LearnConfig:
    batch: int = 8  # train tasks per learning step (chronological)
    max_candidates_per_step: int = 6
    validate_max_tasks: int = 6  # replays per candidate (positives from distinct PRs + negatives)
    min_gain: float = 0.01  # mean score gain required on the held-out *positive* replays
    max_fp_increase: float = 0.0  # extra comments allowed on negatives (mean)
    max_negative_fire_rate: float = 0.5  # reject triggers that fire on more than this share of clean past PRs
    validation_pool: int = 150  # most recent earlier tasks (beyond those learned on) a candidate may be checked on
    miss_pool: int = 24  # rolling pool of unexplained misses shown to the proposer (patterns across time)
    workers: int = 4
    # rubric items (rubric.RubricLearner): runs per validation PR and arm, other past PRs whose concerns must apply
    # the item before it is replayed at all, and the extra target-concern hits (summed over runs) it must earn
    samples: int = 1
    min_support: int = 1
    min_hit_gain: float = 0.0


EpisodeRunner = Callable[[ReplayTask, str], tuple[float, dict | None]]  # (task, extra_context) -> (score, output)


class InterventionLearner:
    """`past`: earlier tasks of the same repository that candidates may be validated on (only those whose outcome is
    already known at the current learning step are used); the tasks learned from are always eligible."""

    def __init__(self, repo: ReplayRepo, run: EpisodeRunner, llm: LLM, config: LearnConfig | None = None,
                 log: Callable[[str], None] = print, past: list[ReplayTask] | None = None):
        self.repo = repo
        self.run = run
        self.llm = llm
        self.cfg = config or LearnConfig()
        self.log = log
        self.interventions: list[Intervention] = []
        self.memory = LessonMemory()
        self._views: dict[str, PRView] = {}
        self.seen: list[ReplayTask] = []
        self.past = sorted(past or [], key=lambda t: (t.as_of, t.id), reverse=True)
        self.now = ""  # event time of the latest task learned from
        self.history: list[dict[str, Any]] = []
        self.matcher = None  # semantic concern matcher (ospm.replay.semantic.Matcher); None: the ±5-line rule

    def view(self, task: ReplayTask) -> PRView:
        if task.id not in self._views:
            self._views[task.id] = PRView.of(self.repo, task)
        return self._views[task.id]

    def active(self) -> list[Intervention]:
        return [iv for iv in self.interventions if iv.status == "active"]

    def serve_context(self, task: ReplayTask, system: str) -> str:
        if system == "interventions":
            return context_block(select(self.interventions, self.view(task)))
        if system == "memory":
            return self.memory.context(self.view(task))
        return ""

    def missed(self, task: ReplayTask, out: dict | None) -> list[dict]:
        """The maintainer's non-nit concerns the output did not raise."""
        if self.matcher is not None:
            from ..replay.semantic import missed_concerns

            return missed_concerns(self.matcher, task, out)
        mine = [c for c in (out or {}).get("comments") or [] if isinstance(c, dict)]
        return [c for c in task.truth["concerns"] if not c["nit"] and not any(
            m.get("path") == c["path"] and abs(int(m.get("line") or 0) - int(c["line"] or 0)) <= 5 for m in mine)]

    def known(self, task: ReplayTask) -> bool:
        """The task's label is known at the current step: a concern once raised, a clean PR only once merged."""
        if task.truth["positive"] or self.repo is None:
            return task.as_of <= self.now
        merged = ((self.repo.store.pull(task.pr) or {}).get("merged_at") or "") if task.pr is not None else ""
        return bool(merged) and merged <= self.now

    def source_prs(self, task_ids: list[str]) -> set:
        """PR numbers of the tasks a candidate was induced from (looked up on the task, else parsed from the id)."""
        by_id = {t.id: t for t in self.seen + self.past}
        return {pr_key(i, by_id[i].pr if i in by_id else None) for i in task_ids}

    def validation_candidates(self) -> list[ReplayTask]:
        """Earlier tasks with known outcomes, most recent first: those learned from, then older history."""
        ids = {t.id for t in self.seen}
        older = [t for t in self.past if t.id not in ids and t.as_of <= self.now][: self.cfg.validation_pool]
        return [t for t in self.seen[::-1] + older if self.known(t)]

    def validate(self, cand: Intervention) -> bool:
        """Replay *other* past PRs where the candidate's trigger applies, with vs without it.

        Every PR the candidate was induced from is excluded (all of its review rounds): replaying a source PR only
        shows that the lesson fits the example it was written from, not that it generalises. Positives come from
        distinct PRs, and adoption requires a gain on those positives - a quieter run on a clean PR alone is noise.
        """
        source = self.source_prs(cand.provenance)
        past = [t for t in self.validation_candidates() if pr_key(t.id, t.pr) not in source]
        past_neg = [t for t in past if not t.truth["positive"]]
        fire_rate = (sum(1 for t in past_neg if applies(cand, self.view(t))) / len(past_neg)) if past_neg else 0.0
        if fire_rate > self.cfg.max_negative_fire_rate:
            cand.status, cand.validation = "rejected", {"reason": "trigger too broad",
                                                        "negative_fire_rate": round(fire_rate, 2)}
            return False
        n_pos = max(1, self.cfg.validate_max_tasks // 2)
        pos: list[ReplayTask] = []
        prs: set = set()
        for t in past:
            if len(pos) < n_pos and t.truth["positive"] and pr_key(t.id, t.pr) not in prs and applies(cand, self.view(t)):
                pos.append(t)
                prs.add(pr_key(t.id, t.pr))
        if not pos:
            cand.status, cand.validation = "rejected", {"reason": "no independent support: the trigger matches no "
                                                                  "other past PR with a maintainer concern"}
            return False
        neg = [t for t in past_neg if applies(cand, self.view(t))][: self.cfg.validate_max_tasks - len(pos)]
        tasks = pos + neg
        base_ctx = {t.id: context_block(select(self.interventions, self.view(t))) for t in tasks}
        with_ctx = {t.id: context_block(select(self.interventions + [_activated(cand)], self.view(t))) for t in tasks}
        jobs = [(t, base_ctx[t.id]) for t in tasks] + [(t, with_ctx[t.id]) for t in tasks]
        results = self._run_many(jobs)
        before, after, fp_before, fp_after = [], [], [], []
        for i, t in enumerate(tasks):
            (s0, o0), (s1, o1) = results[i], results[i + len(tasks)]
            before.append(s0)
            after.append(s1)
            if not t.truth["positive"]:
                fp_before.append(len((o0 or {}).get("comments") or []))
                fp_after.append(len((o1 or {}).get("comments") or []))
        gain = (sum(after) - sum(before)) / len(tasks)
        pos_gain = (sum(after[: len(pos)]) - sum(before[: len(pos)])) / len(pos)
        fp_inc = ((sum(fp_after) - sum(fp_before)) / len(fp_after)) if fp_after else 0.0
        ok = pos_gain >= self.cfg.min_gain and fp_inc <= self.cfg.max_fp_increase
        cand.validation = {"tasks": [t.id for t in tasks], "gain": round(gain, 4), "pos_gain": round(pos_gain, 4),
                           "fp_increase": round(fp_inc, 3), "n_pos": len(pos), "n_neg": len(neg),
                           "negative_fire_rate": round(fire_rate, 2)}
        cand.status = "active" if ok else "rejected"
        return ok

    def _run_many(self, jobs: list[tuple]) -> list[tuple[float, dict | None]]:
        """jobs: (task, extra_context[, skills]) tuples, run with up to `workers` in parallel."""
        if self.cfg.workers <= 1 or len(jobs) <= 1:
            return [self.run(*j) for j in jobs]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.cfg.workers) as pool:
            return list(pool.map(lambda j: self.run(*j), jobs))

    def remember(self, train: list[ReplayTask]) -> None:
        """Fill the memory baseline exactly as `learn` does, without running anything."""
        for t in sorted(train, key=lambda t: (t.as_of, t.id)):
            if t.truth["positive"]:
                for c in t.truth["concerns"]:
                    self.memory.add(t.id, c)

    def learn(self, train: list[ReplayTask]) -> dict[str, Any]:
        ordered = sorted(train, key=lambda t: (t.as_of, t.id))
        pending: list[dict[str, Any]] = []  # misses not yet explained by an active intervention (rolling pool)
        for i in range(0, len(ordered), self.cfg.batch):
            batch = ordered[i:i + self.cfg.batch]
            misses = []
            outs = self._run_many([(t, context_block(select(self.interventions, self.view(t)))) for t in batch])
            for t, (score, out) in zip(batch, outs):
                m = concern_metrics(t, out)
                if t.truth["positive"]:
                    mine = [c for c in (out or {}).get("comments") or [] if isinstance(c, dict)]
                    for c in self.missed(t, out):
                        misses.append({"task": t.id, "files": self.view(t).files, "concern": c,
                                       "agent_comments": [{"path": x.get("path"), "line": x.get("line"),
                                                           "body": str(x.get("body", ""))[:150]} for x in mine]})
                    for c in t.truth["concerns"]:
                        self.memory.add(t.id, c)  # the memory baseline stores every concern, unvalidated
                self.history.append({"task": t.id, "score": score, **m, "active": len(self.active())})
            self.seen.extend(batch)
            self.now = max(self.now, batch[-1].as_of)
            pending = (pending + misses)[-self.cfg.miss_pool:]
            cands = propose(self.llm, pending, self.interventions, self.cfg.max_candidates_per_step)
            adopted = 0
            for c in cands:
                self.interventions.append(c)
                if self.validate(c):
                    adopted += 1
                    covered = set(c.provenance)
                    pending = [m for m in pending if m["task"] not in covered]
            self.log(f"[learn] step {i // self.cfg.batch + 1}: {len(batch)} tasks, {len(misses)} missed concerns, "
                     f"{len(cands)} candidates, {adopted} adopted, {len(self.active())} active")
        return {"interventions": [asdict(iv) for iv in self.interventions], "history": self.history}


def _activated(iv: Intervention) -> Intervention:
    copy = Intervention(**{**asdict(iv), "status": "active"})
    copy.validation = {"gain": 0.0}
    return copy


def save(interventions: list[Intervention], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(iv) for iv in interventions], indent=1), encoding="utf-8")


def load(path: str | Path) -> list[Intervention]:
    return [Intervention(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
