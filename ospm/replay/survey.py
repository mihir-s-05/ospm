"""Deterministic repository survey for choosing a replay target (identical metrics for every candidate).

    python -m ospm.replay.survey vllm-project/vllm sgl-project/sglang pytest-dev/pytest --since 2026-07-10 \
        --out data/repo_survey/survey.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from typing import Any

from ..text import TfidfIndex
from .mirror import GH, github_token

BOTS = {"gemini-code-assist", "coderabbitai", "copilot-pull-request-reviewer", "copilot", "mergify", "github-actions",
        "pre-commit-ci", "dependabot", "renovate", "codspeed-hq", "codecov", "sourcery-ai", "chatgpt-codex-connector",
        "cursor", "claude", "devin-ai-integration"}
MAINT = {"OWNER", "MEMBER", "COLLABORATOR"}
FOLLOWUP_RE = re.compile(r"follow[- ]?up (to|for|of|on) #\d+|broke(n)? (in|by) #\d+|"
                         r"regress\w* (from|in|caused by|introduced (in|by)) #\d+|(introduced|caused) (in|by) #\d+",
                         re.IGNORECASE)
FIX_RE = re.compile(r"^\s*(\[[^\]]*(bug)?fix[^\]]*\]|fix\b|fix\(|bugfix)", re.IGNORECASE)


def is_bot(login: str) -> bool:
    login = (login or "").lower()
    return login.endswith("[bot]") or login in BOTS


def landed(p: dict) -> bool:
    """Merged through GitHub, or landed by a bot that closes the PR and labels it (PyTorch's mergebot adds
    "Merged"; TensorFlow's Copybara imports changes the same way)."""
    return bool(p.get("merged_at")) or any((lb.get("name") or "").lower() == "merged" for lb in p.get("labels") or [])


def recurrence(comments: list[tuple[int, str]], thresholds: tuple[float, ...] = (0.3, 0.5)) -> dict[str, Any]:
    """How often a maintainer review comment closely resembles one left on a *different* PR (TF-IDF cosine).

    A proxy for whether review concerns recur - the property a learned, scoped review lesson depends on.
    """
    idx = TfidfIndex()
    for i, (pr, body) in enumerate(comments):
        idx.add(f"{pr}#{i}", body, pr)
    best = []
    for pr, body in comments:
        best.append(max((sim for _, sim, other in idx.query(body, k=10) if other != pr), default=0.0))
    n = max(1, len(best))
    return {"comments": len(comments), "mean_best_sim": round(sum(best) / n, 3),
            **{f"frac_sim_ge_{t}": round(sum(1 for b in best if b >= t) / n, 3) for t in thresholds}}


def survey(gh: GH, repo: str, since: str, sample: int = 60, ci_sample: int = 12, seed: int = 0) -> dict[str, Any]:
    calls0 = gh.calls
    basics, _ = gh.get(f"/repos/{repo}")
    prs = []
    for d in gh.paginate(f"/repos/{repo}/pulls", {"state": "all", "sort": "created", "direction": "desc"}):
        if d["created_at"] < since:
            break
        prs.append(d)
    issues = []
    for d in gh.paginate(f"/repos/{repo}/issues", {"state": "all", "sort": "created", "direction": "desc"}):
        if d["created_at"] < since:
            break
        if "pull_request" not in d:
            issues.append(d)
    human_prs = [p for p in prs if not is_bot(p["user"]["login"])]
    merged = [p for p in human_prs if landed(p)]
    closed = [p for p in human_prs if p.get("state") == "closed"]
    text = lambda p: f"{p.get('title') or ''}\n{p.get('body') or ''}"  # noqa: E731
    signals = {
        "revert_prs": sum(1 for p in prs if re.match(r"^\s*revert", p["title"] or "", re.IGNORECASE)),
        "followup_prs": sum(1 for p in prs if FOLLOWUP_RE.search(text(p))),
        "fix_prs": sum(1 for p in human_prs if FIX_RE.search(p["title"] or "")),
        "regression_mentions_prs": sum(1 for p in human_prs if re.search(r"\bregress", text(p), re.IGNORECASE)),
        "regression_issues": sum(1 for i in issues if re.search(r"\bregress", f"{i['title']}\n{i.get('body') or ''}",
                                                              re.IGNORECASE)),
    }
    rng = random.Random(seed)
    sampled = rng.sample(merged, min(sample, len(merged)))
    rev = Counter()
    per_pr_maint_inline = []
    maint_comments: list[tuple[int, str]] = []
    for p in sampled:
        n, author = p["number"], p["user"]["login"]
        reviews = list(gh.paginate(f"/repos/{repo}/pulls/{n}/reviews"))
        comments = list(gh.paginate(f"/repos/{repo}/pulls/{n}/comments"))
        maint_inline = 0
        for c in comments:
            login = (c.get("user") or {}).get("login", "")
            top = c.get("in_reply_to_id") is None
            if is_bot(login):
                rev["bot_inline"] += 1
            elif login != author and c.get("author_association") in MAINT and top:
                maint_inline += 1
                maint_comments.append((n, c.get("body") or ""))
            elif login != author and top:
                rev["other_human_inline"] += 1
        per_pr_maint_inline.append(maint_inline)
        states = {(r.get("state"), (r.get("user") or {}).get("login", "") != author and
                   r.get("author_association") in MAINT and not is_bot((r.get("user") or {}).get("login", "")))
                  for r in reviews}
        rev["approved"] += (("APPROVED", True) in states)
        rev["changes_requested"] += (("CHANGES_REQUESTED", True) in states)
        rev["bot_reviews"] += sum(1 for r in reviews if is_bot((r.get("user") or {}).get("login", "")))
    k = max(1, len(sampled))
    review = {"sampled_merged_prs": len(sampled),
              "frac_with_maintainer_inline": round(sum(1 for x in per_pr_maint_inline if x) / k, 3),
              "mean_maintainer_inline": round(sum(per_pr_maint_inline) / k, 2),
              "maintainer_inline_total": sum(per_pr_maint_inline),
              "frac_maintainer_approved": round(rev["approved"] / k, 3),
              "frac_changes_requested": round(rev["changes_requested"] / k, 3),
              "bot_inline_total": rev["bot_inline"], "bot_reviews_total": rev["bot_reviews"],
              "other_human_inline_total": rev["other_human_inline"],
              "recurrence": recurrence(maint_comments)}
    ci = Counter()
    hosts = Counter()
    for p in sampled[:ci_sample]:
        sha = p["head"]["sha"]
        cr, _ = gh.get(f"/repos/{repo}/commits/{sha}/check-runs", {"per_page": 100})
        apps = {(r.get("app") or {}).get("slug", "?") for r in (cr or {}).get("check_runs", [])}
        for a in apps:
            ci[f"app:{a}"] += 1
        ci["failed_check_heads"] += any(r.get("conclusion") in ("failure", "timed_out")
                                        for r in (cr or {}).get("check_runs", []))
        st, _ = gh.get(f"/repos/{repo}/commits/{sha}/status")
        for s in (st or {}).get("statuses", []):
            host = re.sub(r"^https?://([^/]+)/.*$", r"\1", s.get("target_url") or "")
            hosts[host or "(none)"] += 1
        ci["status_failed_heads"] += (st or {}).get("state") == "failure"
    return {
        "repo": repo, "stars": basics.get("stargazers_count"), "size_mb": round(basics.get("size", 0) / 1024),
        "window_start": since, "prs_opened": len(prs), "human_prs_opened": len(human_prs),
        "human_prs_merged": len(merged),
        "merged_via_github": sum(1 for p in merged if p.get("merged_at")),  # << merged: landed by a bot instead
        "human_prs_closed": len(closed), "issues_opened": len(issues), "signals": signals, "review": review,
        "ci": {"heads_sampled": min(ci_sample, len(sampled)), **dict(ci), "status_hosts": dict(hosts)},
        "api_calls": gh.calls - calls0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repos", nargs="+")
    p.add_argument("--since", required=True)
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    since = a.since if "T" in a.since else a.since + "T00:00:00Z"
    gh = GH(github_token(), lambda m: print(m, file=sys.stderr))
    out = {}
    for r in a.repos:
        print(f"[survey] {r}", file=sys.stderr, flush=True)
        out[r] = survey(gh, r, since, a.sample)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
