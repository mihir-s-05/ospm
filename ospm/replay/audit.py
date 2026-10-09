"""Data-feasibility audit for cross-stage learning: does the repository history contain enough *chains* where
a later signal (regression report, revert, follow-up fix, broken main, recurring review correction) can be
traced back to an earlier decision (a merged PR, an accepted issue) - and can those episodes be reconstructed?

    python -m ospm.replay.audit data/pytest data/pip data/pydantic --out data/audit --sample 15

Produces, per repository: summary.json (counts, link rates, reconstructability, delays), chains.jsonl, and
markdown dossiers for a stratified sample of chains (for human / LLM adjudication).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .store import HistoryStore, is_bot
from .tasks import clean_log, failing_tests_in

REGRESSION_RE = re.compile(
    r"\bregress(ion|ed|es)?\b|worked (fine |correctly |as expected )?(in|with|before|on|until) v?\d+\.\d+|"
    r"\bbroke(n)? (in|with|since|after|by) v?\d|\bsince (version |upgrading to )?v?\d+\.\d+(\.\d+)?|"
    r"after upgrad(e|ing) (to )?v?\d", re.IGNORECASE)
CAUSE_RE = re.compile(
    r"(?:caused|introduced|broken|broke|regressed)\s+(?:by|in|with)\s+(?:PR\s*)?#(\d+)|"
    r"regression\s+(?:from|in|of|introduced in|caused by)\s+(?:PR\s*)?#(\d+)|"
    r"(?:since|after|due to)\s+(?:PR\s*)?#(\d+)|(?:git )?bisect\w*\s+(?:\w+\s+){0,6}#(\d+)", re.IGNORECASE)
FOLLOWUP_RE = re.compile(
    r"(?:follow[- ]?up|fixup|fix-up|amend\w*|regression)\s+(?:to|for|from|of|in|on)\s+(?:PR\s*)?#(\d+)|"
    r"(?:broke|broken|introduced|caused)\s+(?:in|by)\s+(?:PR\s*)?#(\d+)|(?:reverts?|partially reverts?)\s+#(\d+)",
    re.IGNORECASE)
CLOSES_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+#(\d+)", re.IGNORECASE)
REVIEW_THEMES = {
    "changelog": r"changelog|news fragment|changes/|towncrier|release note",
    "tests": r"\btests?\b.*\b(add|missing|cover|please|should)|\b(add|missing|write|cover)\w*\b.*\btests?\b|regression test",
    "docs": r"\bdocs?\b|documentation|docstring|\breadme\b",
    "typing": r"\btyp(e|ing) (hint|annotation|check)|\bmypy\b|\bpyright\b|annotation",
    "compat": r"backward|backwards|compat|deprecat|breaking change|public api|semver|old(er)? version",
    "edge_case": r"edge case|corner case|what (happens|if)|\bnone\b.*\bcase|empty|unicode|windows",
    "naming_style": r"\bnaming\b|\brename\b|\bnit\b|style|readab|simplif",
    "performance": r"perf(ormance)?|\bslow\b|overhead|complexity|allocation",
}


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _days(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return round((_ts(b) - _ts(a)).total_seconds() / 86400, 1)


@dataclass
class Chain:
    id: str
    kind: str  # regression | revert | followup | main_break | review_theme | post_release
    later_signal: dict[str, Any]  # what revealed the problem (issue / revert commit / follow-up PR / failed run)
    earlier: dict[str, Any] = field(default_factory=dict)  # the earlier decision (PR / issue) if identified
    linked: bool = False
    delay_days: float | None = None
    reconstructable: dict[str, bool] = field(default_factory=dict)
    notes: str = ""


class RepoAudit:
    def __init__(self, data_dir: str | Path):
        self.data = Path(data_dir).resolve()
        self.git_dir = self.data / "repo.git"
        self.s = HistoryStore(self.data / "history.db", readonly=True)
        self.repo = self.s.meta("repo", self.data.name)
        self.since = self.s.meta("since", "")
        self.is_pr = {r["number"]: bool(r["is_pr"]) for r in self.s.q("SELECT number, is_pr FROM items")}
        self.default_branch = "main" if self._git("rev-parse", "--verify", "-q", "refs/heads/main") else "master"
        self._commit_pr: dict[str, int] | None = None

    # ------------------------------------------------------------------ helpers
    def _git(self, *args: str) -> str:
        out = subprocess.run(["git", "--git-dir", str(self.git_dir), *args], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        return out.stdout.strip() if out.returncode == 0 else ""

    def item(self, n: int) -> dict[str, Any] | None:
        rows = self.s.q("SELECT * FROM items WHERE number=?", (n,))
        return dict(rows[0]) if rows else None

    def pr_meta(self, n: int) -> dict[str, Any] | None:
        it = self.item(n)
        if not it or not it["is_pr"]:
            return None
        pr = self.s.pull(n) or {}
        merge_sha = pr.get("merge_commit_sha") if pr.get("merged_at") else None
        if not merge_sha:
            merge_sha = next((sha for sha, num in self.commit_prs().items() if num == n), None)
        merged_at = pr.get("merged_at")
        if not merged_at and merge_sha:  # landed on the default branch: the landing commit dates the merge
            merged_at = self._git("show", "-s", "--format=%cI", merge_sha) or None
            if merged_at:
                merged_at = datetime.fromisoformat(merged_at.replace("Z", "+00:00")).astimezone(
                    timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"number": n, "title": it["title"], "author": it["author"], "created_at": it["created_at"],
                "merged_at": merged_at, "merge_sha": merge_sha,
                "in_window": bool(it["in_window"]), "head_sha": pr.get("head_sha")}

    def commit_prs(self) -> dict[str, int]:
        """Commits on the default branch (first-parent) -> PR number, from merge/squash commit subjects."""
        if self._commit_pr is None:
            out = self._git("log", "--first-parent", "--format=%H%x09%s", f"refs/heads/{self.default_branch}")
            m: dict[str, int] = {}
            for line in out.splitlines():
                sha, _, subj = line.partition("\t")
                hit = re.search(r"Merge pull request #(\d+)|\(#(\d+)\)\s*$", subj)
                if hit:
                    m[sha] = int(hit.group(1) or hit.group(2))
            self._commit_pr = m
        return self._commit_pr

    def first_parent(self) -> list[str]:
        if not hasattr(self, "_fp"):
            self._fp = list(reversed(self._git("log", "--first-parent", "--format=%H",
                                               f"refs/heads/{self.default_branch}").split()))  # oldest first
        return self._fp

    def landing_commit(self, sha: str) -> str | None:
        """First commit on the default branch's first-parent chain that contains `sha` (binary search)."""
        fp = self.first_parent()
        lo, hi = 0, len(fp) - 1
        if hi < 0 or subprocess.run(["git", "--git-dir", str(self.git_dir), "merge-base", "--is-ancestor", sha, fp[hi]],
                                    capture_output=True).returncode != 0:
            return None
        while lo < hi:
            mid = (lo + hi) // 2
            ok = subprocess.run(["git", "--git-dir", str(self.git_dir), "merge-base", "--is-ancestor", sha, fp[mid]],
                                capture_output=True).returncode == 0
            if ok:
                hi = mid
            else:
                lo = mid + 1
        return fp[lo]

    def szz(self, fix_sha: str, before: str, max_files: int = 6) -> list[tuple[int, int]]:
        """SZZ: blame the lines a fix removed/changed (non-test, non-doc files) in the fix's parent; map the
        introducing commits to the PRs that landed them. Returns [(pr, n_lines)] most-blamed first."""
        parent = self._git("rev-parse", f"{fix_sha}^1")
        if not parent:
            return []
        files = [f for f in self._git("diff", "--name-only", parent, fix_sha).splitlines()
                 if f.endswith(".py") and not re.search(r"(^|/)(tests?|testing|docs?|changelog|changes)/|test_", f)]
        counts: Counter = Counter()
        for f in files[:max_files]:
            hunks = re.findall(r"^@@ -(\d+)(?:,(\d+))? ", self._git("diff", "-U0", parent, fix_sha, "--", f), re.M)
            for start, n in hunks:
                start, n = int(start), int(n or 1)
                if n == 0:
                    continue  # pure insertion: nothing to blame
                blame = self._git("blame", "--porcelain", "-L", f"{start},{start + n - 1}", parent, "--", f)
                for sha in re.findall(r"^([0-9a-f]{40}) \d+ \d+", blame, re.M):
                    counts[sha] += 1
        prs: Counter = Counter()
        cp = self.commit_prs()
        for sha, n in counts.items():
            land = self.landing_commit(sha)
            pr = cp.get(land) if land else None
            if pr:
                prs[pr] += n
        return [(pr, n) for pr, n in prs.most_common()
                if (self.pr_meta(pr) or {}).get("merged_at") and self.pr_meta(pr)["merged_at"] < before]

    def text_of(self, n: int, maintainers_only: bool = False) -> str:
        it = self.item(n) or {}
        parts = [it.get("title") or "", it.get("body") or ""]
        for c in self.s.q("SELECT author, author_association, body FROM comments WHERE number=?", (n,)):
            if not maintainers_only or self.s.is_maintainer(c["author"], c["author_association"]):
                parts.append(c["body"] or "")
        return "\n".join(parts)

    def reconstructable(self, pr: dict[str, Any] | None) -> dict[str, bool]:
        if not pr:
            return {}
        n = pr["number"]
        runs = self.s.q("SELECT id FROM runs WHERE head_sha=?", (pr.get("head_sha") or "",)) if pr.get("head_sha") else []
        return {
            "pr_text": bool(self.item(n) and (self.item(n)["body"] or "").strip()),
            "full_detail": pr["in_window"],
            "reviews": bool(self.s.q("SELECT 1 FROM reviews WHERE number=?", (n,))),
            "diff": bool(pr.get("merge_sha")),
            "ci_runs": bool(runs),
            "ci_logs": bool(runs and self.s.q("SELECT 1 FROM jobs j JOIN job_logs l ON l.job_id=j.id "
                                              "WHERE j.run_id IN (%s)" % ",".join(str(r["id"]) for r in runs))),
        }

    # ------------------------------------------------------------------ chain detectors
    def regressions(self) -> list[Chain]:
        out = []
        for r in self.s.q("SELECT number, title, body, created_at, author FROM items WHERE in_window=1 AND is_pr=0"):
            if is_bot(r["author"]):
                continue
            labels = [e["label"] for e in self.s.q("SELECT label FROM issue_events WHERE number=? AND event='labeled'",
                                                    (r["number"],))]
            text = self.text_of(r["number"])
            if not (REGRESSION_RE.search(f"{r['title']}\n{r['body'] or ''}") or any("regress" in (l or "") for l in labels)):
                continue
            causes = []
            for m in CAUSE_RE.finditer(text):
                n = int(next(g for g in m.groups() if g))
                if self.is_pr.get(n) and n < r["number"]:
                    causes.append(n)
            # fix PR = a later PR that closes this issue
            fixers = [p["number"] for p in self.s.q("SELECT number, title, body FROM items WHERE is_pr=1 AND in_window=1")
                      if any(int(x) == r["number"] for x in CLOSES_RE.findall(f"{p['title']}\n{p['body'] or ''}"))]
            cause, how = (causes[0], "explicit reference") if causes else (None, "")
            if cause is None:
                for fx in fixers:
                    meta = self.pr_meta(fx)
                    if meta and meta.get("merge_sha"):
                        blamed = self.szz(meta["merge_sha"], r["created_at"])
                        if blamed:
                            cause, how = blamed[0][0], f"SZZ via fix PR #{fx} ({blamed[0][1]} blamed lines)"
                            break
            pr = self.pr_meta(cause) if cause else None
            c = Chain(f"regression:{r['number']}", "regression",
                      {"type": "issue", "number": r["number"], "title": r["title"], "at": r["created_at"],
                       "labels": labels, "fix_prs": fixers}, pr or {}, bool(pr),
                      _days(pr and pr["merged_at"], r["created_at"]), self.reconstructable(pr),
                      notes=how or (f"fix PRs {fixers}, no blame-able lines" if fixers else "no fix PR / causing PR"))
            out.append(c)
        return out

    def regression_fix_prs(self, seen_issues: set[int]) -> list[Chain]:
        """Merged PRs that say they fix a regression (title/body), when no regression issue already covers them."""
        out = []
        for r in self.s.q("SELECT number, title, body, created_at, author FROM items WHERE is_pr=1 AND in_window=1"):
            text = f"{r['title']}\n{r['body'] or ''}"
            if is_bot(r["author"]) or not re.search(r"\bregress", text, re.IGNORECASE):
                continue
            if any(int(x) in seen_issues for x in CLOSES_RE.findall(text)):
                continue
            meta = self.pr_meta(r["number"])
            if not meta or not meta.get("merge_sha") or not (self.s.pull(r["number"]) or {}).get("merged_at"):
                continue
            explicit = [int(next(g for g in m.groups() if g)) for m in CAUSE_RE.finditer(text)]
            explicit = [n for n in explicit if self.is_pr.get(n) and n < r["number"]]
            cause, how = (explicit[0], "explicit reference") if explicit else (None, "")
            if cause is None:
                blamed = self.szz(meta["merge_sha"], r["created_at"])
                if blamed:
                    cause, how = blamed[0][0], f"SZZ ({blamed[0][1]} blamed lines)"
            pr = self.pr_meta(cause) if cause else None
            out.append(Chain(f"regression_fix:{r['number']}", "regression",
                             {"type": "pr", "number": r["number"], "title": r["title"], "at": r["created_at"]},
                             pr or {}, bool(pr), _days(pr and pr["merged_at"], r["created_at"]),
                             self.reconstructable(pr), notes=how or "no causing PR found"))
        return out

    def reverts(self) -> list[Chain]:
        out = []
        log = self._git("log", "--first-parent", f"--since={self.since}", "--format=%H%x09%cI%x09%s%x09%b%x1e",
                        f"refs/heads/{self.default_branch}")
        for rec in log.split("\x1e"):
            parts = rec.strip().split("\t")
            if len(parts) < 3 or not re.match(r"^\s*Revert\b", parts[2]):
                continue
            sha, at, subj = parts[0], parts[1], parts[2]
            body = parts[3] if len(parts) > 3 else ""
            nums = [int(x) for x in re.findall(r"#(\d+)", subj + " " + body) if self.is_pr.get(int(x))]
            reverting_pr = self.commit_prs().get(sha)
            target = next((n for n in nums if n != reverting_pr), None)
            if target is None:
                m = re.search(r"This reverts commit ([0-9a-f]{7,40})", body)
                if m:
                    target = self.commit_prs().get(self._git("rev-parse", m.group(1)))
            pr = self.pr_meta(target) if target else None
            out.append(Chain(f"revert:{sha[:10]}", "revert", {"type": "revert", "sha": sha, "subject": subj, "at": at,
                                                              "pr": reverting_pr}, pr or {}, bool(pr),
                             _days(pr and pr["merged_at"], at), self.reconstructable(pr)))
        return out

    def followups(self) -> list[Chain]:
        out = []
        for r in self.s.q("SELECT number, title, body, created_at, author FROM items WHERE is_pr=1 AND in_window=1"):
            if is_bot(r["author"]):
                continue
            text = f"{r['title']}\n{r['body'] or ''}"
            if re.match(r"^\s*Revert\b", r["title"]):
                continue  # counted as reverts
            targets = []
            for m in FOLLOWUP_RE.finditer(text):
                n = int(next(g for g in m.groups() if g))
                if self.is_pr.get(n) and n < r["number"]:
                    targets.append(n)
            if not targets:
                continue
            pr = self.pr_meta(targets[0])
            if not pr or not pr["merged_at"] or pr["merged_at"] > r["created_at"]:
                continue  # follow-ups to unmerged PRs are not downstream signals
            out.append(Chain(f"followup:{r['number']}", "followup",
                             {"type": "pr", "number": r["number"], "title": r["title"], "at": r["created_at"]},
                             pr, True, _days(pr["merged_at"], r["created_at"]), self.reconstructable(pr)))
        return out

    def main_breaks(self) -> list[Chain]:
        out = []
        runs = [dict(r) for r in self.s.q(
            "SELECT * FROM runs WHERE event='push' AND head_branch=? ORDER BY created_at", (self.default_branch,))]
        by_wf: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            by_wf[r["name"]].append(r)
        cp = self.commit_prs()
        for wf, rs in by_wf.items():
            for prev, cur in zip(rs, rs[1:]):
                if prev["conclusion"] == "success" and cur["conclusion"] in ("failure", "timed_out"):
                    n = cp.get(cur["head_sha"])
                    pr = self.pr_meta(n) if n else None
                    jobs = self.s.q("SELECT id, name FROM jobs WHERE run_id=? AND conclusion IN ('failure','timed_out')",
                                    (cur["id"],))
                    tests = set()
                    for j in jobs:
                        log = self.s.job_log(j["id"])
                        if log:
                            tests |= set(failing_tests_in(clean_log(log)))
                    # did the same jobs fail on other branches around the same time? then it is environmental
                    t0 = _ts(cur["created_at"])
                    lo = (t0 - timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    hi = (t0 + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    co = [j["name"] for j in jobs if self.s.q(
                        "SELECT 1 FROM jobs j JOIN runs r ON r.id=j.run_id WHERE j.name=? AND j.conclusion IN "
                        "('failure','timed_out') AND r.head_branch<>? AND r.created_at BETWEEN ? AND ? LIMIT 1",
                        (j["name"], self.default_branch, lo, hi))]
                    environmental = bool(jobs) and len(co) == len(jobs)
                    out.append(Chain(f"main_break:{cur['id']}", "main_break",
                                     {"type": "run", "run_id": cur["id"], "workflow": wf, "at": cur["created_at"],
                                      "failed_jobs": [j["name"] for j in jobs][:8], "failing_tests": sorted(tests)[:10],
                                      "co_failing_elsewhere": co[:8]},
                                     pr or {}, bool(pr) and not environmental,
                                     _days(pr and pr["merged_at"], cur["created_at"]), self.reconstructable(pr),
                                     notes="same jobs failed on other branches at the same time (environmental)"
                                     if environmental else ""))
        return out

    def review_themes(self) -> tuple[list[Chain], dict[str, Any]]:
        theme_prs: dict[str, set[int]] = defaultdict(set)
        examples: dict[str, list[str]] = defaultdict(list)
        rows = list(self.s.q("SELECT number, author, author_association, body FROM review_comments")) + list(
            self.s.q("SELECT number, author, author_association, body FROM reviews WHERE state='CHANGES_REQUESTED'"))
        for c in rows:
            if not self.s.is_maintainer(c["author"], c["author_association"]) or not c["body"]:
                continue
            pr_author = (self.item(c["number"]) or {}).get("author")
            if c["author"] == pr_author:
                continue
            for theme, pat in REVIEW_THEMES.items():
                if re.search(pat, c["body"], re.IGNORECASE):
                    theme_prs[theme].add(c["number"])
                    if len(examples[theme]) < 3:
                        examples[theme].append(c["body"][:200].replace("\n", " "))
        n_prs = len({c["number"] for c in rows})
        chains = [Chain(f"review_theme:{t}", "review_theme", {"type": "theme", "theme": t, "prs": sorted(prs)[:50],
                                                              "n_prs": len(prs), "examples": examples[t]},
                        linked=len(prs) >= 5) for t, prs in theme_prs.items()]
        return chains, {"prs_with_maintainer_review_text": n_prs,
                        "themes": {t: len(p) for t, p in sorted(theme_prs.items(), key=lambda kv: -len(kv[1]))}}

    def post_release(self, regressions: list[Chain], days: int = 21) -> list[Chain]:
        tags = self._git("for-each-ref", "--sort=creatordate", "--format=%(refname:short)%09%(creatordate:iso-strict)",
                         "refs/tags")
        releases = [(t.split("\t")[0], t.split("\t")[1]) for t in tags.splitlines() if "\t" in t
                    and t.split("\t")[1] >= self.since[:10] and re.search(r"\d+\.\d+", t)]
        out = []
        for tag, at in releases:
            ver = re.sub(r"^v", "", tag)
            hits = [c for c in regressions if c.later_signal.get("type") == "issue"
                    and 0 <= (_days(at, c.later_signal["at"]) or -1) <= days
                    and ver in self.text_of(c.later_signal["number"])]
            if hits:
                out.append(Chain(f"post_release:{tag}", "post_release",
                                 {"type": "release", "tag": tag, "at": at,
                                  "regression_issues": [c.later_signal["number"] for c in hits]}, linked=True))
        return out

    # ------------------------------------------------------------------ report
    def run(self) -> tuple[dict[str, Any], list[Chain]]:
        regs = self.regressions()
        regs += self.regression_fix_prs({c.later_signal["number"] for c in regs})
        chains = regs + self.reverts() + self.followups() + self.main_breaks()
        themes, theme_stats = self.review_themes()
        rel = self.post_release(regs)
        chains += themes + rel
        by_kind = defaultdict(list)
        for c in chains:
            by_kind[c.kind].append(c)
        kinds = {}
        for k, cs in by_kind.items():
            delays = [c.delay_days for c in cs if c.delay_days is not None]
            linked = [c for c in cs if c.linked]
            recon = Counter()
            for c in linked:
                for key, ok in c.reconstructable.items():
                    recon[key] += ok
            kinds[k] = {"count": len(cs), "linked_to_earlier_decision": len(linked),
                        "earlier_pr_in_window": sum(1 for c in linked if c.earlier.get("in_window")),
                        "median_delay_days": statistics.median(delays) if delays else None,
                        "reconstructable_among_linked": {k2: v for k2, v in recon.items()}}
        store = self.s.stats()
        summary = {
            "repo": self.repo, "window_start": self.since, "window_issues": store["window_issues"],
            "window_prs": store["window_prs"], "reviews": store["reviews"], "review_comments": store["review_comments"],
            "ci_runs": store["runs"], "failed_job_logs": store["job_logs"], "chains": kinds,
            "review_themes": theme_stats,
            "complete_chains": sum(1 for c in chains if c.linked and c.kind in ("regression", "revert", "followup",
                                                                                "main_break")
                                   and c.earlier.get("in_window") and c.reconstructable.get("diff")),
        }
        return summary, chains

    # ------------------------------------------------------------------ dossiers
    def dossier(self, c: Chain, max_diff: int = 5000) -> str:
        L = [f"# Episode {c.id} ({self.repo})", "", f"chain type: **{c.kind}**", ""]
        sig = c.later_signal
        L.append("## Later signal (what revealed the problem)")
        L.append("```json\n" + json.dumps(sig, indent=1)[:1500] + "\n```")
        if sig.get("type") in ("issue", "pr") and sig.get("number"):
            it = self.item(sig["number"]) or {}
            L += [f"### #{sig['number']}: {it.get('title')}", f"opened {it.get('created_at')} by {it.get('author')}",
                  "", (it.get("body") or "")[:3000], ""]
            for cm in self.s.q("SELECT author, author_association, created_at, body FROM comments WHERE number=? "
                               "ORDER BY created_at LIMIT 8", (sig["number"],)):
                L.append(f"- [{cm['created_at'][:10]}] {cm['author']} ({cm['author_association']}): "
                         f"{(cm['body'] or '')[:600].replace(chr(10), ' ')}")
        if c.earlier:
            e = c.earlier
            it = self.item(e["number"]) or {}
            L += ["", "## Earlier decision (the PR that was merged)",
                  f"### PR #{e['number']}: {e.get('title')}",
                  f"opened {e.get('created_at')} by {e.get('author')}; merged {e.get('merged_at')}; "
                  f"delay to later signal: {c.delay_days} days", "", (it.get("body") or "")[:2500], ""]
            for rv in self.s.q("SELECT author, author_association, state, submitted_at, body FROM reviews WHERE number=? "
                               "ORDER BY submitted_at", (e["number"],)):
                L.append(f"- review [{rv['submitted_at'][:10]}] {rv['author']} ({rv['author_association']}): "
                         f"{rv['state']} {(rv['body'] or '')[:400].replace(chr(10), ' ')}")
            for rc in self.s.q("SELECT author, path, line, body FROM review_comments WHERE number=? LIMIT 15", (e["number"],)):
                L.append(f"- inline {rc['author']} {rc['path']}:{rc['line']}: {(rc['body'] or '')[:300].replace(chr(10), ' ')}")
            if e.get("head_sha"):
                for r in self.s.q("SELECT name, conclusion FROM runs WHERE head_sha=?", (e["head_sha"],)):
                    L.append(f"- CI at PR head: {r['name']}: {r['conclusion']}")
            if e.get("merge_sha"):
                stat = self._git("show", "--stat=120", "--format=%H %s", "-m", "--first-parent", e["merge_sha"])
                diff = self._git("show", "--format=", "-m", "--first-parent", e["merge_sha"])
                L += ["", "### Diff summary", "```", stat[:3000], "```", "### Diff (truncated)", "```diff",
                      diff[:max_diff], "```"]
        L += ["", f"reconstructable: {json.dumps(c.reconstructable)}", f"notes: {c.notes}"]
        return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dirs", nargs="+")
    p.add_argument("--out", default="data/audit")
    p.add_argument("--sample", type=int, default=15, help="dossiers per repository")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enrich", action="store_true",
                   help="fetch reviews/comments of linked earlier PRs that lie outside the mirrored window")
    a = p.parse_args(argv)
    out = Path(a.out)
    overall = {}
    for d in a.data_dirs:
        au = RepoAudit(d)
        summary, chains = au.run()
        if a.enrich:
            old = sorted({c.earlier["number"] for c in chains if c.linked and c.earlier.get("number")
                          and not c.earlier.get("in_window")})
            if old:
                from .mirror import enrich_prs

                au.s.close()
                enrich_prs(d, old)
                au = RepoAudit(d)
                summary, chains = au.run()
            summary["enriched_prs_outside_window"] = len(old)
        name = Path(d).name
        od = out / name
        if (od / "dossiers").exists():  # never mix dossiers from different runs
            for old_file in (od / "dossiers").glob("*.md"):
                old_file.unlink()
        (od / "dossiers").mkdir(parents=True, exist_ok=True)
        (od / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        with (od / "chains.jsonl").open("w", encoding="utf-8") as fh:
            for c in chains:
                fh.write(json.dumps(asdict(c)) + "\n")
        # stratified sample of *linked* event chains for adjudication
        rng = random.Random(a.seed)
        pool = defaultdict(list)
        for c in chains:
            if c.kind in ("regression", "revert", "followup", "main_break") and (c.linked or c.kind == "regression"):
                pool[c.kind].append(c)
        for v in pool.values():
            rng.shuffle(v)
        picked: list[Chain] = []
        while len(picked) < a.sample and any(pool.values()):
            for k in sorted(pool):
                if pool[k] and len(picked) < a.sample:
                    picked.append(pool[k].pop())
        for c in picked:
            (od / "dossiers" / f"{c.id.replace(':', '_')}.md").write_text(au.dossier(c), encoding="utf-8")
        overall[name] = summary
        print(json.dumps({name: {k: summary[k] for k in ("window_issues", "window_prs", "complete_chains", "chains")}},
                         indent=1))
    (out / "overall.json").write_text(json.dumps(overall, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
