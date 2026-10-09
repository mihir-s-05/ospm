"""Mirror a GitHub repository's history into a HistoryStore + a local bare git clone.

    ospm mirror pytest-dev/pytest --since 2026-04-01 --out data/pytest

Uses the authenticated GitHub REST API (token from GITHUB_TOKEN/GH_TOKEN or `gh auth token`), with bulk
repository-wide endpoints wherever possible to stay well inside the rate limit. Storage is kept small: a bare
clone, only the mirrored window's PR heads, CI logs only for failed jobs (trimmed + zlib-compressed), and older
issues with truncated bodies (they are only needed for duplicate search).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from ..integrations.github import _StripAuthOnRedirect
from .store import HistoryStore


def merged_at(pr: dict) -> str | None:
    """When the PR landed. PyTorch's mergebot lands the commit itself, then closes the PR and labels it "Merged",
    so GitHub never records a merge: the close time stands in for the landing time."""
    if pr.get("merged_at"):
        return pr["merged_at"]
    if any((lb.get("name") or "").lower() == "merged" for lb in pr.get("labels") or []):
        return pr.get("closed_at")
    return None


API = "https://api.github.com"
OLD_BODY_CHARS = 2000
LOG_RETENTION_DAYS = 88  # GitHub deletes Actions logs after 90 days


def github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        out = None
    if out is None or out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError("no GitHub token: set GITHUB_TOKEN or run `gh auth login`")
    return out.stdout.strip()


class GH:
    def __init__(self, token: str, log: Callable[[str], None] = print):
        self.token = token
        self.opener = urllib.request.build_opener(_StripAuthOnRedirect)
        self.log = log
        self.calls = 0

    def get(self, url: str, params: dict | None = None, raw: bool = False, accept: str = "application/vnd.github+json"):
        if not url.startswith("http"):
            url = API + url
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        for attempt in range(6):
            req = urllib.request.Request(url, headers={
                "Accept": accept, "Authorization": f"Bearer {self.token}", "User-Agent": "ospm-mirror",
                "X-GitHub-Api-Version": "2022-11-28"})
            try:
                with self.opener.open(req, timeout=60) as resp:
                    self.calls += 1
                    body = resp.read()
                    headers = dict(resp.headers)
                    if raw:
                        return body.decode("utf-8", "replace"), headers
                    return (json.loads(body) if body else None), headers
            except urllib.error.HTTPError as e:
                if e.code in (403, 429) and (e.headers.get("Retry-After") or e.headers.get("X-RateLimit-Remaining") == "0"):
                    wait = int(e.headers.get("Retry-After") or max(5, int(e.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time()))
                    self.log(f"[mirror] rate limited, sleeping {wait}s")
                    time.sleep(min(wait, 900))
                    continue
                if e.code in (404, 410, 451):
                    return None, {}
                if e.code >= 500 and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError):
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"GET {url} failed repeatedly")

    def paginate(self, path: str, params: dict | None = None, key: str | None = None) -> Iterator[dict]:
        url = path
        p = dict(params or {})
        p.setdefault("per_page", 100)
        first = True
        while url:
            data, headers = self.get(url, p if first else None)
            first = False
            if data is None:
                return
            items = data[key] if key else data
            yield from items
            m = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or "")
            url = m.group(1) if m else None


def _git(args: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=check)


def _login(d: dict | None) -> str:
    return (d or {}).get("login", "") or ""


def enrich_prs(out_dir: str | Path, numbers: list[int], *, workers: int = 4,
               log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> int:
    """Fetch full PR detail (body, head/merge sha, reviews, inline and issue comments) for specific PRs outside
    the mirrored window - e.g. the PRs that later turned out to cause regressions. CI logs of old PRs are not
    recoverable (GitHub deletes them after ~90 days)."""
    store = HistoryStore(Path(out_dir) / "history.db")
    repo = store.meta("repo")
    gh = GH(github_token(), log)

    def one(n: int) -> None:
        d, _ = gh.get(f"/repos/{repo}/pulls/{n}")
        if not d:
            return
        store.upsert("pulls", [{"number": n, "head_sha": d["head"]["sha"], "head_ref": d["head"].get("ref"),
                                "head_repo": (d["head"].get("repo") or {}).get("full_name"),
                                "base_ref": d["base"].get("ref"), "base_sha": d["base"].get("sha"),
                                "merged_at": merged_at(d), "merge_commit_sha": d.get("merge_commit_sha"),
                                "draft": int(bool(d.get("draft"))), "have_head": 0}])
        store.q("UPDATE items SET body=? WHERE number=?", (d.get("body") or "", n))
        rows = [{"id": r["id"], "number": n, "author": _login(r.get("user")),
                 "author_association": r.get("author_association", ""), "state": r.get("state"),
                 "body": r.get("body") or "", "submitted_at": r["submitted_at"], "commit_id": r.get("commit_id")}
                for r in gh.paginate(f"/repos/{repo}/pulls/{n}/reviews") if r.get("submitted_at")]
        store.upsert("reviews", rows)
        store.upsert("review_comments", [
            {"id": c["id"], "number": n, "review_id": c.get("pull_request_review_id"), "author": _login(c.get("user")),
             "author_association": c.get("author_association", ""), "path": c.get("path"), "line": c.get("line"),
             "original_line": c.get("original_line"), "side": c.get("side"), "diff_hunk": (c.get("diff_hunk") or "")[-1500:],
             "body": c.get("body") or "", "created_at": c["created_at"], "commit_id": c.get("commit_id"),
             "in_reply_to": c.get("in_reply_to_id")} for c in gh.paginate(f"/repos/{repo}/pulls/{n}/comments")])
        store.upsert("comments", [
            {"id": c["id"], "number": n, "author": _login(c.get("user")), "author_association": c.get("author_association", ""),
             "created_at": c["created_at"], "body": c.get("body") or ""}
            for c in gh.paginate(f"/repos/{repo}/issues/{n}/comments")])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, sorted(set(numbers))))
    with store._lock:
        store._conn.commit()
    store.close()
    log(f"[enrich] {len(set(numbers))} PRs, {gh.calls} API calls")
    return gh.calls


def mirror(repo: str, out_dir: str | Path, since: str | None, *, all_issue_titles: bool = True, max_logs: int = 800,
           workers: int = 6, blobless: bool = False, incremental: bool = False, overlap_days: int = 3,
           review_scope: str = "all",
           log: Callable[[str], None] = lambda m: print(m, file=sys.stderr)) -> dict[str, Any]:
    """Mirror a repository's history.

    `blobless=True` makes a partial clone (commits and trees only; file contents are fetched on demand), which is
    enough for history analysis and keeps large repositories small on disk.

    `incremental=True` updates an existing mirror: the original window start is kept, and only what changed since
    the previous mirror (minus `overlap_days`) is fetched - new and updated items, PRs, events, comments, reviews
    and CI runs, plus logs of newly failed jobs. Run it at least every ~80 days (weekly is sensible) so CI logs are
    archived before GitHub deletes them.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = HistoryStore(out / "history.db")
    gh = GH(github_token(), log)
    prev = store.meta("mirrored_at")
    if incremental and prev:
        since_ts = store.meta("since")  # the window start never moves
        fetch_from = (datetime.fromisoformat(prev.replace("Z", "+00:00")) - timedelta(days=overlap_days)
                      ).strftime("%Y-%m-%dT%H:%M:%SZ")
        log(f"[mirror] incremental update of {repo}: changes since {fetch_from} (window starts {since_ts})")
    else:
        if not since:
            raise ValueError("since is required for a first (non-incremental) mirror")
        incremental = False
        since_ts = since if "T" in since else since + "T00:00:00Z"
        fetch_from = since_ts
    t0 = time.time()

    # ---------------------------------------------------------------- git (bare clone, small)
    git_dir = out / "repo.git"
    if not git_dir.exists():
        log(f"[mirror] cloning {repo} (bare)")
        _git(["clone", "--bare", "--quiet", *(["--filter=blob:none"] if blobless else []),
              f"https://github.com/{repo}.git", str(git_dir)])
    else:
        _git(["fetch", "--quiet", "origin", "+refs/heads/*:refs/heads/*", "--prune"], cwd=str(git_dir))

    # ---------------------------------------------------------------- issues + PR stubs
    log("[mirror] issues/PRs")
    rows = []
    params = ({"state": "all", "since": fetch_from, "sort": "updated", "direction": "desc"} if incremental
              else {"state": "all", "sort": "created", "direction": "desc"})
    for d in gh.paginate(f"/repos/{repo}/issues", params):
        in_window = d["created_at"] >= since_ts
        if not in_window and not all_issue_titles and not incremental:
            break
        body = d.get("body") or ""
        rows.append({
            "number": d["number"], "is_pr": int("pull_request" in d), "title": d.get("title") or "",
            "body": body if in_window else body[:OLD_BODY_CHARS], "author": _login(d.get("user")),
            "author_association": d.get("author_association", ""), "state": d.get("state"),
            "state_reason": d.get("state_reason"), "created_at": d["created_at"], "closed_at": d.get("closed_at"),
            "updated_at": d.get("updated_at"), "labels_json": json.dumps([l["name"] for l in d.get("labels", [])]),
            "reactions": int((d.get("reactions") or {}).get("total_count", 0)), "in_window": int(in_window)})
        if len(rows) >= 500:
            store.upsert("items", rows)
            rows = []
    store.upsert("items", rows)
    window = [r["number"] for r in store.q("SELECT number FROM items WHERE in_window=1")]
    window_set = set(window)
    log(f"[mirror] {len(window)} items in window ({gh.calls} API calls so far)")

    # ---------------------------------------------------------------- pulls
    log("[mirror] pull requests")
    prs = []
    order = "updated" if incremental else "created"
    for d in gh.paginate(f"/repos/{repo}/pulls", {"state": "all", "sort": order, "direction": "desc"}):
        if d[f"{order}_at"] < fetch_from:
            break
        if d["created_at"] < since_ts:
            continue  # updated, but created before the window
        prs.append({"number": d["number"], "head_sha": d["head"]["sha"], "head_ref": d["head"].get("ref"),
                    "head_repo": ((d["head"].get("repo") or {}).get("full_name")), "base_ref": d["base"].get("ref"),
                    "base_sha": d["base"].get("sha"), "merged_at": merged_at(d),
                    "merge_commit_sha": d.get("merge_commit_sha"), "draft": int(bool(d.get("draft"))), "have_head": 0})
    store.upsert("pulls", prs)

    # ---------------------------------------------------------------- issue events (labels, closes, duplicates)
    log("[mirror] issue events")
    evs = []
    for e in gh.paginate(f"/repos/{repo}/issues/events"):
        if e["created_at"] < fetch_from:
            break
        num = (e.get("issue") or {}).get("number")
        if num not in window_set:
            continue
        evs.append({"id": e["id"], "number": num, "event": e["event"], "actor": _login(e.get("actor")),
                    "label": (e.get("label") or {}).get("name"), "created_at": e["created_at"],
                    "commit_id": e.get("commit_id"),
                    "extra_json": json.dumps({k: e[k] for k in ("state_reason",) if k in e})})
    store.upsert("issue_events", evs)

    # ---------------------------------------------------------------- comments (bulk)
    log("[mirror] issue comments")
    cms = []
    for c in gh.paginate(f"/repos/{repo}/issues/comments", {"since": fetch_from, "sort": "created", "direction": "asc"}):
        num = int(c["issue_url"].rsplit("/", 1)[1])
        if num not in window_set:
            continue
        cms.append({"id": c["id"], "number": num, "author": _login(c.get("user")),
                    "author_association": c.get("author_association", ""), "created_at": c["created_at"],
                    "body": c.get("body") or ""})
    store.upsert("comments", cms)

    log("[mirror] review comments")
    rcs = []
    for c in gh.paginate(f"/repos/{repo}/pulls/comments", {"since": fetch_from, "sort": "created", "direction": "asc"}):
        num = int(c["pull_request_url"].rsplit("/", 1)[1])
        if num not in window_set:
            continue
        rcs.append({"id": c["id"], "number": num, "review_id": c.get("pull_request_review_id"),
                    "author": _login(c.get("user")), "author_association": c.get("author_association", ""),
                    "path": c.get("path"), "line": c.get("line"), "original_line": c.get("original_line"),
                    "side": c.get("side"), "diff_hunk": (c.get("diff_hunk") or "")[-1500:], "body": c.get("body") or "",
                    "created_at": c["created_at"], "commit_id": c.get("commit_id"), "in_reply_to": c.get("in_reply_to_id")})
    store.upsert("review_comments", rcs)

    log("[mirror] reviews (per PR)")

    def fetch_reviews(n: int) -> list[dict]:
        out_rows = []
        for r in gh.paginate(f"/repos/{repo}/pulls/{n}/reviews"):
            if not r.get("submitted_at"):
                continue
            out_rows.append({"id": r["id"], "number": n, "author": _login(r.get("user")),
                             "author_association": r.get("author_association", ""), "state": r.get("state"),
                             "body": r.get("body") or "", "submitted_at": r["submitted_at"],
                             "commit_id": r.get("commit_id")})
        return out_rows

    review_numbers = [p["number"] for p in prs]
    if review_scope == "merged_or_commented":  # large repos: abandoned, never-discussed PRs have no reviews
        commented = {c["number"] for c in rcs}
        review_numbers = [p["number"] for p in prs if p["merged_at"] or p["number"] in commented]
    log(f"[mirror] fetching reviews for {len(review_numbers)} of {len(prs)} PRs")
    with ThreadPoolExecutor(workers) as pool:
        for chunk in pool.map(fetch_reviews, review_numbers):
            store.upsert("reviews", chunk)

    # ---------------------------------------------------------------- CI runs (weekly slices: API caps at 1000)
    log("[mirror] workflow runs")
    start = datetime.fromisoformat(fetch_from.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    runs: list[dict] = []
    cur = start
    while cur < now:
        nxt = min(cur + timedelta(days=7), now)
        rng = f"{cur:%Y-%m-%dT%H:%M:%SZ}..{nxt:%Y-%m-%dT%H:%M:%SZ}"
        for r in gh.paginate(f"/repos/{repo}/actions/runs", {"created": rng}, key="workflow_runs"):
            if r.get("event") not in ("pull_request", "push", "pull_request_target", "merge_group"):
                continue
            runs.append({"id": r["id"], "attempt": r.get("run_attempt", 1), "name": r.get("name"), "event": r["event"],
                         "head_sha": r["head_sha"], "head_branch": r.get("head_branch"),
                         "pr_numbers_json": json.dumps([p["number"] for p in r.get("pull_requests") or []]),
                         "status": r.get("status"), "conclusion": r.get("conclusion"), "created_at": r["created_at"],
                         "updated_at": r.get("updated_at"), "html_url": r.get("html_url")})
        cur = nxt
    store.upsert("runs", runs)
    log(f"[mirror] {len(runs)} runs ({gh.calls} API calls so far)")

    # jobs for failed runs and for every earlier attempt of re-run workflows (flakiness evidence)
    log_cutoff = (now - timedelta(days=LOG_RETENTION_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    targets = []
    for r in runs:
        attempts = range(1, r["attempt"] + 1) if r["attempt"] > 1 else (
            [1] if r["conclusion"] in ("failure", "timed_out") else [])
        for a in attempts:
            targets.append((r["id"], a, r["created_at"]))

    def fetch_jobs(t) -> list[dict]:
        run_id, attempt, _ = t
        out_rows = []
        for j in gh.paginate(f"/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs", key="jobs"):
            out_rows.append({"id": j["id"], "run_id": run_id, "attempt": attempt, "name": j.get("name"),
                             "conclusion": j.get("conclusion"), "started_at": j.get("started_at"),
                             "completed_at": j.get("completed_at"),
                             "steps_json": json.dumps([{"name": s.get("name"), "conclusion": s.get("conclusion")}
                                                       for s in j.get("steps") or []])})
        return out_rows

    all_jobs: list[dict] = []
    with ThreadPoolExecutor(workers) as pool:
        for chunk in pool.map(fetch_jobs, targets):
            store.upsert("jobs", chunk)
            all_jobs.extend(chunk)
    run_created = {(t[0], t[1]): t[2] for t in targets}
    failed = [j for j in all_jobs if j["conclusion"] in ("failure", "timed_out")
              and run_created.get((j["run_id"], j["attempt"]), "") >= log_cutoff and not store.has_log(j["id"])]
    failed = failed[:max_logs]
    log(f"[mirror] downloading {len(failed)} failed-job logs")

    def fetch_log(j: dict) -> None:
        text, _ = gh.get(f"/repos/{repo}/actions/jobs/{j['id']}/logs", raw=True)
        if text:
            store.put_log(j["id"], text)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(fetch_log, failed))

    # ---------------------------------------------------------------- PR heads into the local clone
    log("[mirror] fetching PR head commits")
    nums = [p["number"] for p in prs]
    for i in range(0, len(nums), 80):
        batch = nums[i:i + 80]
        _git(["fetch", "--quiet", "origin", *[f"+refs/pull/{n}/head:refs/pull/{n}/head" for n in batch]],
             cwd=str(git_dir), check=False)
    have = set(_git(["for-each-ref", "--format=%(refname)", "refs/pull/"], cwd=str(git_dir)).stdout.split())
    store.upsert("pulls", [{**p, "have_head": int(f"refs/pull/{p['number']}/head" in have)} for p in
                           [dict(r) for r in store.q("SELECT * FROM pulls")]])

    store.set_meta("repo", repo)
    store.set_meta("since", since_ts)
    if not incremental:
        store.set_meta("blobless", blobless)
    store.set_meta("mirrored_at", now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    stats = store.stats()
    stats["api_calls"] = gh.calls
    stats["elapsed_s"] = round(time.time() - t0, 1)
    store.close()
    log(f"[mirror] done: {stats}")
    return stats
