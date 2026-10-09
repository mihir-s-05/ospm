"""`hist` - read-only, time-travel view of the mirrored repository history for agents.

Runs inside the agent sandbox (exposed as the `hist` command). The history database and the replay time come
from OSPM_HIST_DB / OSPM_HIST_ASOF; nothing created after OSPM_HIST_ASOF is ever shown. For the pull request
under review, OSPM_HIST_PR / OSPM_HIST_PR_SHA pin the diff to the commit being reviewed.

    hist issue 123            hist search "fixture teardown order"     hist recent --days 30
    hist pr 456 [--diff]      hist ci 456                              hist log 987654 [--grep E] [--tail 80]
    hist labels               hist commits [--path src/_pytest/x.py] [-n 15]
    hist reviews --path src/_pytest/ [-n 20] [--author LOGIN] [--max-chars 500]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

from .ghstack import base_of as ghstack_base
from .ghstack import is_ghstack
from .store import HistoryStore, is_bot


def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        sys.exit(f"hist: {name} is not set (hist only works inside an ospm replay episode)")
    return v


def _short(text: str | None, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + f"\n...[{len(text) - n} more characters]"


def _git(args: list[str]) -> str:
    git_dir = os.environ.get("OSPM_HIST_GIT")
    if not git_dir:
        return ""
    out = subprocess.run(["git", "--git-dir", git_dir, *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    return out.stdout


def cmd_issue(s: HistoryStore, as_of: str, a) -> None:
    it = s.item(a.number, as_of)
    if not it:
        print(f"#{a.number} does not exist (as of {as_of})")
        return
    kind = "PR" if it["is_pr"] else "Issue"
    print(f"{kind} #{it['number']}: {it['title']}")
    print(f"author: {it['author']} ({it['author_association']}) | state: {it['state']} | created: {it['created_at']}")
    print(f"labels: {', '.join(it['labels']) or '(none)'} | reactions: {it['reactions']}")
    print("\n" + _short(it["body"], a.max_chars))
    comments = s.comments(a.number, as_of)
    if comments:
        print(f"\n--- {len(comments)} comment(s) ---")
        for c in comments[-a.comments:]:
            tag = " [bot]" if is_bot(c["author"]) else ""
            print(f"\n[{c['created_at']}] {c['author']} ({c['author_association']}){tag}:\n{_short(c['body'], 1500)}")


def cmd_search(s: HistoryStore, as_of: str, a) -> None:
    res = s.search_items(" ".join(a.query), as_of, a.limit, include_prs=a.prs)
    if not res:
        print("no matches")
    for r in res:
        print(f"#{r['number']} [{r['state']}] (sim {r['similarity']}) {r['title']}  labels: {', '.join(r['labels'])}")


def cmd_recent(s: HistoryStore, as_of: str, a) -> None:
    start = (datetime.fromisoformat(as_of.replace("Z", "+00:00")) - timedelta(days=a.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = s.q("SELECT number FROM items WHERE created_at>=? AND created_at<=? AND is_pr=? ORDER BY created_at DESC "
               "LIMIT ?", (start, as_of, int(a.prs), a.limit))
    for r in rows:
        it = s.item(r["number"], as_of)
        print(f"#{it['number']} [{it['state']}] {it['created_at'][:10]} {it['title']}  labels: {', '.join(it['labels'])}")


def _pr_sha(s: HistoryStore, number: int, as_of: str) -> str | None:
    if os.environ.get("OSPM_HIST_PR") == str(number) and os.environ.get("OSPM_HIST_PR_SHA"):
        return os.environ["OSPM_HIST_PR_SHA"]
    pr = s.pull(number)
    if pr and pr["merged_at"] and pr["merged_at"] <= as_of:
        return pr["head_sha"]
    return None  # the latest head may postdate as_of: do not leak it


def cmd_pr(s: HistoryStore, as_of: str, a) -> None:
    cmd_issue(s, as_of, a)
    pr = s.pull(a.number)
    if not pr:
        return
    merged = pr["merged_at"] if pr["merged_at"] and pr["merged_at"] <= as_of else None
    print(f"\nbase: {pr['base_ref']} | head: {pr['head_repo']}:{pr['head_ref']} | merged: {merged or 'no'}")
    reviews = [r for r in s.reviews(a.number, as_of)]
    for r in reviews:
        print(f"\n[review {r['submitted_at']}] {r['author']} ({r['author_association']}): {r['state']}"
              + (f"\n{_short(r['body'], 1200)}" if r["body"] else ""))
    for c in s.review_comments(a.number, as_of):
        print(f"\n[inline {c['created_at']}] {c['author']} on {c['path']}:{c['line'] or c['original_line']}:\n"
              f"{_short(c['body'], 800)}")
    sha = _pr_sha(s, a.number, as_of)
    if sha:
        base = os.environ.get("OSPM_HIST_PR_BASE", "") if os.environ.get("OSPM_HIST_PR") == str(a.number) else ""
        if not base and is_ghstack(pr["base_ref"]):
            base = ghstack_base(lambda *args: _git(list(args)).strip(), sha)
        elif not base:  # fork point from the target branch as it was at as_of (later merges excluded)
            main_then = _git(["rev-list", "-1", "--first-parent", f"--before={as_of}",
                              "refs/heads/" + (pr["base_ref"] or "main")]).strip()
            base = _git(["merge-base", sha, main_then]).strip() if main_then else ""
        if base:
            print("\nfiles changed:\n" + _git(["diff", "--stat=160", f"{base}..{sha}"]))
            if a.diff:
                print(_short(_git(["diff", f"{base}..{sha}", "--", *(a.path or [])]), a.max_chars))
    else:
        print("\n(diff not available as of this time; inspect the workspace with git instead)")


def cmd_ci(s: HistoryStore, as_of: str, a) -> None:
    """CI runs of a commit as they looked at `as_of`: later re-run attempts and unfinished jobs are hidden."""
    sha = _pr_sha(s, a.number, as_of) if a.number else os.environ.get("OSPM_HIST_SHA")
    if not sha:
        print("no CI information available")
        return
    runs = s.runs_for_sha(sha, as_of)
    if not runs:
        print(f"no recorded CI runs for {sha[:10]} as of {as_of}")
    for r in runs:
        jobs = [dict(j) for j in s.q("SELECT * FROM jobs WHERE run_id=? AND started_at<=? ORDER BY attempt, id",
                                     (r["id"], as_of))]
        if jobs:
            attempt = max(j["attempt"] for j in jobs)
            jobs = [j for j in jobs if j["attempt"] == attempt]
            for j in jobs:
                if not j["completed_at"] or j["completed_at"] > as_of:
                    j["conclusion"] = "in_progress"
            states = {j["conclusion"] for j in jobs}
            status = ("failure" if states & {"failure", "timed_out"} else
                      "in_progress" if "in_progress" in states else "success")
        else:
            attempt = 1
            done = r["updated_at"] and r["updated_at"] <= as_of and r["attempt"] == 1
            status = (r["conclusion"] or r["status"]) if done else "in_progress"
        print(f"run {r['id']} attempt {attempt} '{r['name']}' ({r['event']}): {status} (started {r['created_at']})")
        for j in jobs:
            mark = "FAILED" if j["conclusion"] in ("failure", "timed_out") else j["conclusion"]
            failed_steps = [st["name"] for st in json.loads(j["steps_json"] or "[]") if st.get("conclusion") == "failure"]
            extra = f" failed step: {', '.join(failed_steps)}" if failed_steps and mark == "FAILED" else ""
            log = " (log: hist log %d)" % j["id"] if mark == "FAILED" and s.has_log(j["id"]) else ""
            print(f"   job {j['id']} {j['name']}: {mark}{extra}{log}")


def cmd_log(s: HistoryStore, as_of: str, a) -> None:
    job = s.job(a.job_id)
    text = s.job_log(a.job_id)
    if job is None or text is None or not job["completed_at"] or job["completed_at"] > as_of:
        print("log not available (as of this time)")
        return
    from .tasks import clean_log

    lines = clean_log(text).splitlines()
    if a.grep:
        pat = re.compile(a.grep, re.IGNORECASE)
        hits = [i for i, l in enumerate(lines) if pat.search(l)]
        keep = sorted({j for i in hits for j in range(max(0, i - a.context), min(len(lines), i + a.context + 1))})
        print("\n".join(lines[i] for i in keep[: a.max_lines]) or "no matching lines")
        return
    sel = lines[: a.head] if a.head else lines[-a.tail:]
    print("\n".join(sel))


def cmd_labels(s: HistoryStore, as_of: str, a) -> None:
    rows = s.q("SELECT label, COUNT(*) AS n FROM issue_events WHERE event='labeled' AND created_at<=? "
               "GROUP BY label ORDER BY n DESC", (as_of,))
    for r in rows:
        print(f"{r['n']:>5}  {r['label']}")


def cmd_commits(s: HistoryStore, as_of: str, a) -> None:
    main_then = _git(["rev-list", "-1", "--first-parent", f"--before={as_of}", "refs/heads/main"]).strip()
    if not main_then:
        print("no commits")
        return
    args = ["log", f"-n{a.n}", "--format=%h %ad %an: %s", "--date=short", main_then]
    if a.path:
        args += ["--", *a.path]
    print(_git(args) or "no commits")


def _path_match(path: str, given: str) -> bool:
    if path == given:
        return True
    if given.endswith("/"):
        return path.startswith(given)
    return any(c in given for c in "*?[") and fnmatch.fnmatchcase(path, given)


def cmd_reviews(s: HistoryStore, as_of: str, a) -> None:
    """Top-level maintainer inline comments on the given paths, visible as of `as_of` (same rule as
    HistoryStore.review_comments), never by the PR's own author."""
    rows = s.q("SELECT c.*, (SELECT i.author FROM items i WHERE i.number=c.number) AS pr_author "
               "FROM review_comments c LEFT JOIN reviews r ON r.id=c.review_id "
               "WHERE c.in_reply_to IS NULL AND c.created_at<=? AND (r.id IS NULL OR r.submitted_at<=?) "
               "ORDER BY c.created_at DESC, c.id DESC", (as_of, as_of))
    out = []
    for r in rows:
        if not any(_path_match(r["path"], p) for p in a.path):
            continue
        own = (r["author"] or "").lower() == (r["pr_author"] or "").lower()
        if own or not s.is_maintainer(r["author"], r["author_association"]):
            continue
        if a.author and r["author"].lower() != a.author.lower():
            continue
        out.append(r)
        if len(out) >= a.n:
            break
    paths = ", ".join(a.path)
    if not out:
        print(f"no earlier maintainer review comments on {paths}")
        return
    print(f"{len(out)} earlier maintainer review comments on {paths}")
    for r in out:
        where = f"{r['path']}:{r['line'] or r['original_line']}"
        print(f"\n[{r['created_at'][:10]}] #{r['number']} @{r['author']} on {where}\n"
              f"{_short(r['body'], a.max_chars)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hist", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("issue")
    s.add_argument("number", type=int)
    s.add_argument("--comments", type=int, default=15)
    s.add_argument("--max-chars", type=int, default=6000)
    s.set_defaults(fn=cmd_issue)
    s = sub.add_parser("search")
    s.add_argument("query", nargs="+")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--prs", action="store_true")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("recent")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--limit", type=int, default=40)
    s.add_argument("--prs", action="store_true")
    s.set_defaults(fn=cmd_recent)
    s = sub.add_parser("pr")
    s.add_argument("number", type=int)
    s.add_argument("--diff", action="store_true")
    s.add_argument("--path", nargs="*")
    s.add_argument("--comments", type=int, default=15)
    s.add_argument("--max-chars", type=int, default=15000)
    s.set_defaults(fn=cmd_pr)
    s = sub.add_parser("ci")
    s.add_argument("number", type=int, nargs="?")
    s.set_defaults(fn=cmd_ci)
    s = sub.add_parser("log")
    s.add_argument("job_id", type=int)
    s.add_argument("--grep")
    s.add_argument("--context", type=int, default=2)
    s.add_argument("--tail", type=int, default=120)
    s.add_argument("--head", type=int, default=0)
    s.add_argument("--max-lines", type=int, default=200)
    s.set_defaults(fn=cmd_log)
    s = sub.add_parser("labels")
    s.set_defaults(fn=cmd_labels)
    s = sub.add_parser("commits")
    s.add_argument("--path", nargs="*")
    s.add_argument("-n", type=int, default=15)
    s.set_defaults(fn=cmd_commits)
    s = sub.add_parser("reviews")
    s.add_argument("--path", nargs="+", required=True)
    s.add_argument("-n", type=int, default=20)
    s.add_argument("--author")
    s.add_argument("--max-chars", type=int, default=500)
    s.set_defaults(fn=cmd_reviews)
    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    store = HistoryStore(_env("OSPM_HIST_DB"), readonly=True)
    try:
        a.fn(store, _env("OSPM_HIST_ASOF"), a)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
