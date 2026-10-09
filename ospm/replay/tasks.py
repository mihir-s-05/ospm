"""Replay tasks mined from repository history, with ground truth taken from what actually happened later.

Ground truth is never shown to agents (the replay clock stops at the event time); it is used only to score
episodes and to write GEPA feedback ("the maintainer labelled this `type: docs` and closed it as a duplicate of
#123", "the reviewer asked for changes at src/_pytest/x.py:120: ...").
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .ghstack import target_branch
from .store import HistoryStore, is_bot
from .workspace import ReplayRepo

LABEL_WINDOW_DAYS = 60
KIND_LABELS = [("type: regression", "bug"), ("type: bug", "bug"), ("type: docs", "docs"),
               ("type: question", "question"), ("type: enhancement", "feature"), ("type: proposal", "feature"),
               ("type: feature-branch", "feature"), ("type: performance", "feature"),
               ("type: deprecation", "other"), ("type: removal", "other"), ("type: refactoring", "other"),
               ("type: infrastructure", "other"), ("type: selftests", "other"),
               ("type: backward compatibility", "other")]
COMPONENT_PREFIXES = ("topic: ", "plugin: ", "platform: ")
# towncrier changelog fragments (changelog/<issue>.<type>.rst) record how the fix of an issue was classified
FRAGMENT_KIND = {"bugfix": "bug", "feature": "feature", "improvement": "feature", "doc": "docs",
                 "deprecation": "other", "breaking": "other", "removal": "other", "misc": "other",
                 "contrib": "other", "packaging": "other", "vendored": "other", "trivial": "other"}
_DUP_RE = re.compile(r"(?:duplicate|dupe|dup)\s+(?:of\s+)?#(\d+)", re.IGNORECASE)
_FAILED_TEST_RE = re.compile(r"^(?:\S+\s+)?FAILED\s+(\S+::\S+)", re.MULTILINE)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _fmt(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _minus(ts: str, seconds: int = 1) -> str:
    return _fmt(_ts(ts) - timedelta(seconds=seconds))


@dataclass
class ReplayTask:
    id: str
    kind: str  # triage | review | ci
    as_of: str
    sha: str
    event: str  # what the agent is told
    truth: dict[str, Any]
    pr: int | None = None
    pr_sha: str = ""
    split: str = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_specialist(self) -> str:
        return {"triage": "triage", "review": "review", "ci": "ci_triage", "concern": "review"}[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _split(task_id: str) -> str:
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 4
    return ("train", "train", "val", "test")[h]


# ---------------------------------------------------------------------------------------------- triage
def changelog_fragments(repo: ReplayRepo, branch: str = "main") -> dict[int, str]:
    """issue number -> kind, from changelog fragments added on the main branch (any time up to the mirror)."""
    import subprocess

    out = subprocess.run(["git", "--git-dir", str(repo.git_dir), "log", "--diff-filter=A", "--name-only",
                          "--format=", f"refs/heads/{branch}", "--", "changelog/"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    frags: dict[int, str] = {}
    for line in out.splitlines():
        m = re.match(r"changelog/(\d+)\.([a-z]+)", line.strip())
        if m and m.group(2) in FRAGMENT_KIND:
            frags.setdefault(int(m.group(1)), FRAGMENT_KIND[m.group(2)])
    return frags


def triage_truth(store: HistoryStore, item: dict[str, Any], fragments: dict[int, str] | None = None
                 ) -> dict[str, Any] | None:
    n, author = item["number"], item["author"]
    horizon = _fmt(_ts(item["created_at"]) + timedelta(days=LABEL_WINDOW_DAYS))
    labels: list[str] = []
    closed_reason = None
    for e in store.q("SELECT * FROM issue_events WHERE number=? AND created_at<=? ORDER BY created_at, id",
                     (n, horizon)):
        if e["event"] in ("labeled", "unlabeled") and (is_bot(e["actor"]) or e["actor"] == author):
            continue
        if e["event"] == "labeled" and e["label"] not in labels:
            labels.append(e["label"])
        elif e["event"] == "unlabeled" and e["label"] in labels:
            labels.remove(e["label"])
        elif e["event"] == "closed":
            closed_reason = json.loads(e["extra_json"] or "{}").get("state_reason") or "completed"
        elif e["event"] == "reopened":
            closed_reason = None
    if closed_reason is None and item["closed_at"] and item["closed_at"] <= horizon:
        closed_reason = item["state_reason"] or "completed"
    kind = next((k for lab, k in KIND_LABELS if lab in labels), None)
    fragment = (fragments or {}).get(n)
    if kind is None and fragment:
        kind = fragment
    dup_of = None
    for c in store.q("SELECT body, author, author_association FROM comments WHERE number=? AND created_at<=? "
                     "ORDER BY created_at", (n, horizon)):
        if c["author"] != author and not is_bot(c["author"]):
            m = _DUP_RE.search(c["body"] or "")
            if m and int(m.group(1)) != n:
                dup_of = int(m.group(1))
                break
    if "closed as duplicate" in labels or closed_reason == "duplicate":
        decision = "duplicate"
    elif "status: needs information" in labels:
        decision = "needs-info"
    elif "invalid" in labels:
        decision = "invalid"
    elif closed_reason == "not_planned":
        decision = "declined"
    elif fragment or kind is not None:
        decision = "accept"  # a maintainer classified it, or a fix/feature for it was merged
    else:
        decision = None
    if kind is None and decision is None:
        return None
    return {"kind": kind, "decision": decision, "duplicate_of": dup_of if decision == "duplicate" else None,
            "components": sorted(l for l in labels if l.startswith(COMPONENT_PREFIXES)), "labels": labels,
            "fragment": fragment, "closed_reason": closed_reason}


def triage_tasks(repo: ReplayRepo) -> list[ReplayTask]:
    s = repo.store
    frags = changelog_fragments(repo)
    out = []
    for r in s.q("SELECT * FROM items WHERE in_window=1 AND is_pr=0 ORDER BY number"):
        item = dict(r)
        if is_bot(item["author"]):
            continue
        truth = triage_truth(s, item, frags)
        if truth is None:
            continue
        as_of = item["created_at"]
        tid = f"triage:{item['number']}"
        event = (f"Issue #{item['number']} was just opened by @{item['author']}: \"{item['title']}\". "
                 f"Triage it (read it with `hist issue {item['number']}`).")
        out.append(ReplayTask(tid, "triage", as_of, repo.commit_before(as_of), event, truth, split=_split(tid),
                              meta={"title": item["title"]}))
    return out


def score_triage(task: ReplayTask, out: dict | None) -> tuple[float, str]:
    t = task.truth
    if not out:
        return 0.0, "No valid triage output was produced."
    num = den = 0.0
    notes = []
    if t["kind"]:
        ok = out.get("kind") == t["kind"]
        num, den = num + ok, den + 1
        if not ok:
            notes.append(f"kind: you said {out.get('kind')!r}; the maintainers classified it as {t['kind']!r}")
    if t["decision"]:
        ok = out.get("decision") == t["decision"]
        if ok and t["decision"] == "duplicate" and t["duplicate_of"]:
            ok = out.get("duplicate_of") == t["duplicate_of"]
            if not ok:
                notes.append(f"duplicate_of: you said {out.get('duplicate_of')}, actually #{t['duplicate_of']}")
        num, den = num + 1.5 * ok, den + 1.5
        if out.get("decision") != t["decision"]:
            notes.append(f"decision: you said {out.get('decision')!r}; what actually happened: {t['decision']!r}"
                         + (f" (duplicate of #{t['duplicate_of']})" if t["duplicate_of"] else ""))
    if t["components"]:
        pred = set(out.get("components") or [])
        gold = set(t["components"])
        j = len(pred & gold) / len(pred | gold) if pred | gold else 1.0
        num, den = num + 0.5 * j, den + 0.5
        if j < 1:
            notes.append(f"components: you said {sorted(pred)}; maintainers applied {sorted(gold)}")
    notes.append(f"what happened: maintainer labels {t['labels']}, closed as {t.get('closed_reason')!r}"
                 + (f", fixed with a '{t['fragment']}' changelog entry" if t.get("fragment") else ""))
    return (num / den if den else 0.0), "; ".join(notes)


# ---------------------------------------------------------------------------------------------- review
def review_tasks(repo: ReplayRepo) -> list[ReplayTask]:
    s = repo.store
    out = []
    for r in s.q("SELECT i.*, p.base_ref FROM items i JOIN pulls p ON p.number=i.number WHERE i.in_window=1 "
                 "AND i.is_pr=1 ORDER BY i.number"):
        pr = dict(r)
        if is_bot(pr["author"]):
            continue
        reviews = [rv for rv in s.q("SELECT * FROM reviews WHERE number=? ORDER BY submitted_at", (pr["number"],))
                   if rv["author"] != pr["author"] and s.is_maintainer(rv["author"], rv["author_association"])
                   and rv["commit_id"]]
        if not reviews:
            continue
        first = reviews[0]
        sha = first["commit_id"]
        if not repo.has_commit(sha):
            continue
        state = {"APPROVED": "approve", "CHANGES_REQUESTED": "request_changes"}.get(first["state"], "comment")
        inline = [dict(c) for c in s.q(
            "SELECT * FROM review_comments WHERE number=? AND review_id=? AND in_reply_to IS NULL", (pr["number"], first["id"]))]
        if state == "comment" and not inline and not (first["body"] or "").strip():
            continue  # an empty "commented" review carries no signal
        as_of = _minus(first["submitted_at"])
        if repo.committed_after(sha, as_of):
            continue
        base = repo.base_at(sha, as_of, pr["base_ref"] or "main")
        if not base:
            continue
        tid = f"review:{pr['number']}@{sha[:8]}"
        event = (f"Pull request #{pr['number']} \"{pr['title']}\" by @{pr['author']} is ready for review. Your "
                 f"workspace is checked out at its head commit {sha[:10]}; it targets {target_branch(pr['base_ref'])}. The change "
                 f"under review is `git diff {base[:12]}..HEAD` (description and discussion: `hist pr {pr['number']}`).")
        truth = {"verdict": state, "reviewer": first["author"], "review_body": (first["body"] or "")[:1500],
                 "comments": [{"path": c["path"], "line": c["line"] or c["original_line"], "body": c["body"][:600]}
                              for c in inline],
                 "merged": bool(s.pull(pr["number"]) and s.pull(pr["number"])["merged_at"])}
        out.append(ReplayTask(tid, "review", as_of, sha, event, truth, pr=pr["number"], pr_sha=sha,
                              split=_split(tid), meta={"title": pr["title"], "base": base}))
    return out


def _near(a: dict, b: dict, tol: int = 5) -> bool:
    try:
        return a.get("path") == b.get("path") and abs(int(a.get("line") or 0) - int(b.get("line") or 0)) <= tol
    except (TypeError, ValueError):
        return False


def score_review(task: ReplayTask, out: dict | None) -> tuple[float, str]:
    t = task.truth
    if not out:
        return 0.0, "No valid review output was produced."
    order = {"approve": 0, "comment": 1, "request_changes": 2}
    v = out.get("verdict")
    verdict_score = 1.0 if v == t["verdict"] else 0.5 if abs(order.get(v, 9) - order[t["verdict"]]) == 1 else 0.0
    mine = [c for c in out.get("comments") or [] if isinstance(c, dict)]
    gold = t["comments"]
    notes = [f"verdict: you said {v!r}; the maintainer ({t['reviewer']}) said {t['verdict']!r}"]
    if gold:
        hit = [g for g in gold if any(_near(m, g) for m in mine)]
        recall = len(hit) / len(gold)
        precision = (sum(1 for m in mine if any(_near(m, g) for g in gold)) / len(mine)) if mine else 0.0
        score = (verdict_score + recall + 0.5 * precision) / 2.5
        missed = [g for g in gold if g not in hit][:6]
        if missed:
            notes.append("the maintainer commented where you did not: " + " | ".join(
                f"{g['path']}:{g['line']}: {g['body'][:200]}" for g in missed))
    else:
        noise = 1.0 if len(mine) <= 2 else 2 / len(mine)
        score = (verdict_score + 0.5 * noise) / 1.5
        if t["review_body"]:
            notes.append(f"the maintainer's review said: {t['review_body'][:400]}")
    return score, "; ".join(notes)


# ---------------------------------------------------------------------------------------------- CI
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ?", re.MULTILINE)
_GATE_STEPS = ("decide whether the needed jobs succeeded",)


def clean_log(text: str) -> str:
    return _TS_RE.sub("", ANSI_RE.sub("", text))


def failing_tests_in(log: str) -> list[str]:
    return sorted(set(_FAILED_TEST_RE.findall(log)))


def _ci_category(job_name: str, step: str, log: str, conclusion: str) -> str:
    """Category from the log content first (it says what actually broke), then from the step/job names."""
    if conclusion == "timed_out" or "exceeded the maximum execution time" in log:
        return "timeout"
    if re.search(r"^FAILED \S+::|^=+ .*\d+ failed|ERROR collecting|errors? during collection", log, re.MULTILINE):
        return "test_failure"
    if re.search(r"Failed to build|ResolutionImpossible|No matching distribution|subprocess-exited-with-error", log):
        return "dependency"
    if re.search(r"ECONNRESET|Connection reset|503 Service|lost communication|Could not resolve host", log):
        return "infra"
    text = f"{step} {job_name}".lower()
    for keys, cat in ((("lint", "pre-commit", "ruff", "flake8"), "lint"), (("mypy", "pyright", "typing"), "type_check"),
                      (("doc",), "docs"), (("package", "wheel", "sdist", "dist"), "build"), (("test",), "test_failure")):
        if any(k in text for k in keys):
            return cat
    return "build"


def ci_tasks(repo: ReplayRepo, co_failure_hours: int = 36) -> list[ReplayTask]:
    s = repo.store
    rows = [dict(j) for j in s.q(
        "SELECT j.*, r.head_sha, r.name AS run_name, r.pr_numbers_json, r.head_branch, r.event, "
        "r.created_at AS run_created FROM jobs j JOIN runs r ON r.id=j.run_id AND r.attempt>=j.attempt "
        "WHERE j.conclusion IN ('failure','timed_out') AND j.id IN (SELECT job_id FROM job_logs) "
        "GROUP BY j.id")]
    info = {}
    for j in rows:
        steps = json.loads(j["steps_json"] or "[]")
        failed_step = next((st["name"] for st in steps if st.get("conclusion") == "failure"), "")
        log = clean_log(s.job_log(j["id"]) or "")
        info[j["id"]] = {"tests": set(failing_tests_in(log)), "step": failed_step, "log": log,
                         "cat": _ci_category(j["name"], failed_step, log, j["conclusion"])}
    out = []
    for j in rows:
        meta = info[j["id"]]
        if meta["step"].lower().startswith(_GATE_STEPS) or j["name"] == "check":
            continue  # aggregate gate job: fails only because other jobs failed
        if j["event"] == "push":
            continue  # tasks are about pull requests
        prs = json.loads(j["pr_numbers_json"] or "[]")
        pr = prs[0] if prs else None
        if pr is None:
            hit = s.q("SELECT number FROM pulls WHERE head_ref=? ORDER BY number DESC LIMIT 1", (j["head_branch"],))
            pr = hit[0]["number"] if hit else None
        if pr is None or not repo.has_commit(j["head_sha"]):
            continue
        t0 = _ts(j["run_created"])
        lo, hi = _fmt(t0 - timedelta(hours=co_failure_hours)), _fmt(t0 + timedelta(hours=co_failure_hours))
        # (a) the same job later passed on the same commit
        later_pass = s.q("SELECT 1 FROM jobs WHERE run_id=? AND attempt>? AND name=? AND conclusion='success'",
                         (j["run_id"], j["attempt"], j["name"])) or s.q(
            "SELECT 1 FROM runs WHERE head_sha=? AND name=? AND created_at>? AND conclusion='success'",
            (j["head_sha"], j["run_name"], j["run_created"]))
        # (b) the same failure happened around the same time on other branches (other PRs or main)
        co = 0
        for o in rows:
            if o["id"] == j["id"] or o["head_branch"] == j["head_branch"] or not (lo <= o["run_created"] <= hi):
                continue
            om = info[o["id"]]
            if meta["tests"] and om["tests"] & meta["tests"]:
                co += 1
            elif not meta["tests"] and o["name"] == j["name"] and om["cat"] == meta["cat"]:
                co += 1
        newer_commit = s.q("SELECT 1 FROM runs WHERE head_branch=? AND name=? AND created_at>? AND head_sha<>?",
                           (j["head_branch"], j["run_name"], j["run_created"], j["head_sha"]))
        if later_pass or co:
            flaky = True  # not caused by this change: transient, environmental, or already broken elsewhere
        elif newer_commit:
            flaky = False
        else:
            continue  # outcome unknown
        truth = {"flaky": flaky, "category": meta["cat"], "failing_tests": sorted(meta["tests"])[:50],
                 "job_name": j["name"], "failed_step": meta["step"],
                 "evidence": "passed on re-run" if later_pass else (f"same failure in {co} other branch(es)" if co
                                                                    else "fixed by a new commit")}
        as_of = j["completed_at"] or j["started_at"] or j["run_created"]
        tid = f"ci:{j['id']}"
        event = (f"CI job {j['id']} (\"{j['name']}\", workflow \"{j['run_name']}\") failed on pull request #{pr} at "
                 f"commit {j['head_sha'][:10]} (your workspace). Diagnose it: `hist log {j['id']}`, `hist ci {pr}`.")
        out.append(ReplayTask(tid, "ci", as_of, j["head_sha"], event, truth, pr=pr, pr_sha=j["head_sha"],
                              split=_split(tid), meta={"job_id": j["id"]}))
    return out


def score_ci(task: ReplayTask, out: dict | None) -> tuple[float, str]:
    t = task.truth
    if not out:
        return 0.0, "No valid CI diagnosis was produced."
    flaky_ok = out.get("flaky") == t["flaky"]
    cat_ok = out.get("category") == t["category"]
    gold, pred = set(t["failing_tests"]), set(out.get("failing_tests") or [])
    norm = lambda xs: {x.split("[")[0] for x in xs}  # noqa: E731 - ignore parametrisation ids
    if gold:
        g, p = norm(gold), norm(pred)
        tests_f1 = 2 * len(g & p) / (len(g) + len(p)) if g or p else 1.0
        score = (1.5 * flaky_ok + 0.5 * cat_ok + tests_f1) / 3.0
    else:
        tests_f1 = None
        score = (1.5 * flaky_ok + 0.5 * cat_ok) / 2.0
    what = (f"it was not caused by the change ({t.get('evidence')})" if t["flaky"]
            else "it was caused by the change (fixed by a new commit)")
    notes = [f"flaky: you said {out.get('flaky')}; in reality {what}",
             f"category: you said {out.get('category')!r}; expected {t['category']!r} (job {t['job_name']}, "
             f"failed step {t['failed_step']!r})"]
    if tests_f1 is not None and tests_f1 < 1:
        notes.append(f"failing tests in the log: {sorted(gold)[:8]}; you listed {sorted(pred)[:8]}")
    return score, "; ".join(notes)


def _score_concerns(task, out):
    from .concerns import score_concerns

    return score_concerns(task, out)


SCORERS = {"triage": score_triage, "review": score_review, "ci": score_ci, "concern": _score_concerns}
OUTPUT_KEY = {"triage": "triage", "review": "review", "ci": "ci", "concern": "review"}


def task_output(task: ReplayTask, output: dict | None, entry_role: str) -> dict | None:
    """Pick the part of an episode's output that answers this task."""
    if output is None:
        return None
    if entry_role != "orchestrator":
        return output
    part = output.get(OUTPUT_KEY[task.kind])
    if task.kind == "ci":
        if isinstance(part, list):
            job = task.meta.get("job_id")
            match = [c for c in part if isinstance(c, dict) and c.get("job_id") == job]
            return (match or [c for c in part if isinstance(c, dict)] or [None])[0]
        return part if isinstance(part, dict) else None
    return part if isinstance(part, dict) else None


def score_task(task: ReplayTask, output: dict | None, entry_role: str) -> tuple[float, str]:
    return SCORERS[task.kind](task, task_output(task, output, entry_role))


def build_tasks(repo: ReplayRepo, kinds: tuple[str, ...] = ("triage", "review", "ci", "concern")) -> list[ReplayTask]:
    from .concerns import concern_tasks

    fns = {"triage": triage_tasks, "review": review_tasks, "ci": ci_tasks, "concern": concern_tasks}
    out: list[ReplayTask] = []
    for k in kinds:
        out.extend(fns[k](repo))
    return out


def save_tasks(tasks: list[ReplayTask], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t.to_dict()) + "\n")


def load_tasks(path) -> list[ReplayTask]:
    with open(path, encoding="utf-8") as fh:
        return [ReplayTask(**json.loads(line)) for line in fh if line.strip()]
