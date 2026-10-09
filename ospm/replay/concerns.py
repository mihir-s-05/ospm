"""Review-concern tasks: can an agent raise, at PR time, the concerns a maintainer later raised in review?

The data audit (docs/DATA_AUDIT.md) found maintainer review corrections to be the densest cross-stage signal:
frequent, immediate, and the stage where nearly all detectable problems had evidence. Each task is one
maintainer review round:

* positive task: a maintainer left inline comments (concerns) on commit C. The agent reviews C with only the
  history available just before the review, and is scored on how many concerns it raised (same file, nearby
  line), under a comment budget so that spraying comments does not pay;
* negative task: a PR that maintainers approved without any inline comment. The agent should stay (nearly)
  silent, which measures false positives.

Splits are chronological (train = oldest), so evaluation is on later episodes than anything learned from.
"""

from __future__ import annotations

import re
from typing import Any

from .ghstack import target_branch
from .store import is_bot
from .tasks import ReplayTask, _minus
from .workspace import ReplayRepo

THEMES = {
    "tests": r"\btests?\b|coverage|regression test|pytest\.mark|assert",
    "docs": r"\bdocs?\b|documentation|docstring|readme|comment (here|this)",
    "changelog": r"changelog|news fragment|release note|towncrier",
    "compat": r"backward|backwards|compat|deprecat|breaking|public api|semver|old(er)? version",
    "typing": r"\btyp(e|ing)\b|annotation|mypy|pyright|Optional\[|\bcast\(",
    "correctness": r"\bbug\b|wrong|incorrect|crash|race|leak|off[- ]by|edge case|corner case|what (happens|if)|"
                   r"\bnone\b|empty|overflow|exception|error handling",
    "performance": r"perf(ormance)?|\bslow\b|overhead|complexity|allocation|cache|memory",
    "design": r"\bdesign\b|abstraction|refactor|instead of|why not|should (we|this|it)|move (this|it)|api",
    "naming_style": r"\bnaming\b|\brename\b|\bnit\b|style|readab|simplif|typo|unused",
}
TRIVIAL = re.compile(r"^\s*(lgtm|thanks?( you)?|nice|\+1|looks good|ok|done)\W*$", re.IGNORECASE)
COMMENT_BUDGET = 3


def themes_of(text: str) -> list[str]:
    hits = [t for t, pat in THEMES.items() if re.search(pat, text or "", re.IGNORECASE)]
    return hits or ["other"]


def is_nit(text: str) -> bool:
    return bool(re.match(r"\s*(nit|minor|optional|small)\b", text or "", re.IGNORECASE))


def concern_tasks(repo: ReplayRepo, include_negatives: bool = True) -> list[ReplayTask]:
    s = repo.store
    out: list[ReplayTask] = []
    prs = [dict(r) for r in s.q("SELECT i.*, p.base_ref FROM items i JOIN pulls p ON p.number=i.number "
                                "WHERE i.in_window=1 AND i.is_pr=1 ORDER BY i.number")]
    for pr in prs:
        if is_bot(pr["author"]):
            continue
        reviews = [dict(rv) for rv in s.q("SELECT * FROM reviews WHERE number=? ORDER BY submitted_at", (pr["number"],))
                   if rv["author"] != pr["author"] and s.is_maintainer(rv["author"], rv["author_association"])
                   and rv["commit_id"]]
        any_inline = False
        for rv in reviews:
            inline = [dict(c) for c in s.q("SELECT * FROM review_comments WHERE number=? AND review_id=? AND "
                                           "in_reply_to IS NULL ORDER BY created_at", (pr["number"], rv["id"]))]
            inline = [c for c in inline if not TRIVIAL.match(c["body"] or "")]
            if not inline:
                continue
            any_inline = True
            task = _make(repo, pr, rv, inline, positive=True)
            if task:
                out.append(task)
        if include_negatives and not any_inline and reviews and pr.get("state") == "closed":
            approved = next((rv for rv in reviews if rv["state"] == "APPROVED"), None)
            merged = (s.pull(pr["number"]) or {}).get("merged_at")
            # bots (e.g. AI review assistants) comment on most PRs in large projects; only maintainers count
            has_any_inline = any(
                c["author"] != pr["author"] and s.is_maintainer(c["author"], c["author_association"])
                for c in s.q("SELECT author, author_association FROM review_comments WHERE number=?", (pr["number"],)))
            if approved and merged and not has_any_inline:
                task = _make(repo, pr, approved, [], positive=False)
                if task:
                    out.append(task)
    return chronological_split(out)


def _make(repo: ReplayRepo, pr: dict, rv: dict, inline: list[dict], positive: bool) -> ReplayTask | None:
    sha = rv["commit_id"]
    if not repo.has_commit(sha):
        return None
    as_of = _minus(rv["submitted_at"])
    if repo.committed_after(sha, as_of):
        return None
    base = repo.base_at(sha, as_of, pr["base_ref"] or "main")
    if not base:
        return None
    concerns = [{"path": c["path"], "line": c["line"] or c["original_line"], "body": (c["body"] or "")[:600],
                 "themes": themes_of(c["body"]), "nit": is_nit(c["body"])} for c in inline]
    tid = f"concern:{pr['number']}@{sha[:8]}:{rv['id']}" if positive else f"concern-neg:{pr['number']}@{sha[:8]}"
    event = (f"Pull request #{pr['number']} \"{pr['title']}\" by @{pr['author']} is ready for review. Your workspace "
             f"is checked out at commit {sha[:10]}; it targets {target_branch(pr['base_ref'])}. The change under review is "
             f"`git diff {base[:12]}..HEAD` (description and discussion: `hist pr {pr['number']}`). Raise only "
             f"concerns a maintainer would actually ask the author to address, each anchored to a file and line; "
             f"an empty list is the right answer when the change is ready.")
    truth = {"positive": positive, "reviewer": rv["author"], "review_state": rv["state"],
             "review_body": (rv["body"] or "")[:1200], "concerns": concerns}
    return ReplayTask(tid, "concern", as_of, sha, event, truth, pr=pr["number"], pr_sha=sha,
                      meta={"title": pr["title"], "base": base})


def chronological_split(tasks: list[ReplayTask], fractions: tuple[float, float] = (0.6, 0.2)) -> list[ReplayTask]:
    """Oldest 60% -> train, next 20% -> val, newest 20% -> test (by event time)."""
    ordered = sorted(tasks, key=lambda t: (t.as_of, t.id))
    n = len(ordered)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    for i, t in enumerate(ordered):
        t.split = "train" if i < a else "val" if i < b else "test"
    return ordered


def _near(a: dict, b: dict, tol: int = 5) -> bool:
    if a.get("path") != b.get("path"):
        return False
    try:
        return abs(int(a.get("line") or 0) - int(b.get("line") or 0)) <= tol
    except (TypeError, ValueError):
        return False


def concern_metrics(task: ReplayTask, out: dict | None, budget: int = COMMENT_BUDGET) -> dict[str, Any]:
    mine = [c for c in ((out or {}).get("comments") or []) if isinstance(c, dict)]
    gold = [c for c in task.truth["concerns"] if not c["nit"]] or task.truth["concerns"]
    matched = [g for g in gold if any(_near(m, g) for m in mine)]
    useful = [m for m in mine if any(_near(m, g) for g in task.truth["concerns"])]
    return {"positive": task.truth["positive"], "n_comments": len(mine), "n_gold": len(gold),
            "matched": len(matched), "recall": (len(matched) / len(gold)) if gold else None,
            "located_precision": (len(useful) / len(mine)) if mine else None,
            "matched_themes": sorted({t for g in matched for t in g["themes"]}),
            "missed_themes": sorted({t for g in gold if g not in matched for t in g["themes"]})}


def score_concerns(task: ReplayTask, out: dict | None, budget: int = COMMENT_BUDGET) -> tuple[float, str]:
    if out is None:
        return 0.0, "No valid review output was produced."
    m = concern_metrics(task, out, budget)
    over = max(0, m["n_comments"] - budget)
    if not task.truth["positive"]:
        score = 1.0 if m["n_comments"] == 0 else max(0.0, 1.0 - m["n_comments"] / (budget + 1))
        return score, (f"the maintainer approved this change without inline comments; you left {m['n_comments']} "
                       f"comment(s)" + (" (good)" if m["n_comments"] == 0 else ""))
    recall = m["recall"] or 0.0
    score = recall * (budget / (budget + over))
    gold = task.truth["concerns"]
    missed = [g for g in gold if not any(_near(c, g) for c in (out.get("comments") or []) if isinstance(c, dict))]
    fb = [f"you matched {m['matched']}/{m['n_gold']} maintainer concerns with {m['n_comments']} comments "
          f"(budget {budget})"]
    if missed:
        fb.append("the maintainer raised: " + " | ".join(f"{g['path']}:{g['line']} [{','.join(g['themes'])}] "
                                                         f"{g['body'][:220]}" for g in missed[:5]))
    return score, "; ".join(fb)
