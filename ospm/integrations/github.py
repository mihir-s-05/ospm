"""GitHub REST integration (stdlib only). Writes are dry-run unless explicitly enabled."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from ..models import CIRun, Issue, PullRequest

API = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Log downloads redirect to blob storage, which rejects a forwarded GitHub Authorization header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


def parse_ref(ref: str) -> tuple[str, int | None]:
    """'owner/repo#12' -> ('owner/repo', 12); 'owner/repo' -> ('owner/repo', None)."""
    m = re.fullmatch(r"([\w.-]+/[\w.-]+)(?:#(\d+))?", ref.strip())
    if not m:
        raise ValueError(f"expected owner/repo or owner/repo#number, got {ref!r}")
    return m.group(1), int(m.group(2)) if m.group(2) else None


class GitHubClient:
    def __init__(self, repo: str, token: str | None = None, *, api: str = API, dry_run: bool = True,
                 timeout: float = 30.0, opener: Any = None):
        self.repo = repo
        self.token = token if token is not None else (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
        self.api = api.rstrip("/")
        self.dry_run = dry_run
        self.timeout = timeout
        self.opener = opener or urllib.request.build_opener(_StripAuthOnRedirect)
        self.actions: list[dict[str, Any]] = []  # write operations (performed or, in dry-run, planned)

    # ------------------------------------------------------------------ transport
    def _request(self, method: str, path: str, *, params: dict | None = None, body: Any = None,
                 accept: str = "application/vnd.github+json", raw: bool = False) -> Any:
        url = path if path.startswith("http") else f"{self.api}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", accept)
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "ospm")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with self.opener.open(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300] if hasattr(e, "read") else ""
            raise GitHubError(f"{method} {path} -> HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise GitHubError(f"{method} {path} failed: {e.reason}") from e
        if raw:
            return payload.decode("utf-8", errors="replace")
        return json.loads(payload) if payload else None

    # ------------------------------------------------------------------ reads
    @staticmethod
    def _issue(d: dict) -> Issue:
        return Issue(
            number=d["number"],
            title=d.get("title") or "",
            body=d.get("body") or "",
            author=(d.get("user") or {}).get("login", ""),
            labels=[l["name"] if isinstance(l, dict) else str(l) for l in d.get("labels", [])],
            reactions=int((d.get("reactions") or {}).get("total_count", 0)),
            created_at=d.get("created_at") or "",
            state=d.get("state") or "open",
            url=d.get("html_url") or "",
        )

    def get_issue(self, number: int) -> Issue:
        return self._issue(self._request("GET", f"/repos/{self.repo}/issues/{number}"))

    def list_issues(self, state: str = "open", limit: int = 100) -> list[Issue]:
        out: list[Issue] = []
        page = 1
        while len(out) < limit:
            batch = self._request("GET", f"/repos/{self.repo}/issues",
                                  params={"state": state, "per_page": 100, "page": page})
            if not batch:
                break
            out.extend(self._issue(d) for d in batch if "pull_request" not in d)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    def _ci_runs(self, number: int, sha: str, with_logs: bool = True) -> list[CIRun]:
        data = self._request("GET", f"/repos/{self.repo}/commits/{sha}/check-runs", params={"per_page": 100})
        runs = []
        for cr in data.get("check_runs", []):
            conclusion = cr.get("conclusion") or cr.get("status") or "unknown"
            log = ""
            if with_logs and conclusion in ("failure", "timed_out", "startup_failure"):
                log = self._job_log(cr)
            runs.append(CIRun(name=cr.get("name", "check"), conclusion=conclusion, log=log,
                              url=cr.get("html_url", ""), run_id=f"pr-{number}/{cr.get('name', 'check')}"))
        return runs

    def _job_log(self, check_run: dict) -> str:
        """Full job log for GitHub Actions checks (needs a token); else the check run's output text."""
        if self.token and (check_run.get("app") or {}).get("slug") == "github-actions":
            try:
                return self._request("GET", f"/repos/{self.repo}/actions/jobs/{check_run['id']}/logs", raw=True)
            except GitHubError:
                pass
        out = check_run.get("output") or {}
        return "\n".join(x for x in (out.get("title"), out.get("summary"), out.get("text")) if x)

    def get_pull(self, number: int, with_ci: bool = True) -> PullRequest:
        d = self._request("GET", f"/repos/{self.repo}/pulls/{number}")
        diff = self._request("GET", f"/repos/{self.repo}/pulls/{number}", accept="application/vnd.github.diff",
                             raw=True)
        pr = PullRequest(
            number=number, title=d.get("title") or "", body=d.get("body") or "",
            author=(d.get("user") or {}).get("login", ""), diff=diff,
            labels=[l["name"] for l in d.get("labels", [])], created_at=d.get("created_at") or "",
            updated_at=d.get("updated_at") or "", draft=bool(d.get("draft")), merged=bool(d.get("merged_at")),
            url=d.get("html_url") or "",
        )
        if with_ci and (d.get("head") or {}).get("sha"):
            pr.ci_runs = self._ci_runs(number, d["head"]["sha"])
        return pr

    def list_merged_pulls(self, limit: int = 100, since: str | None = None) -> list[PullRequest]:
        """Merged PRs (most recently updated first), optionally only those merged after ISO date `since`."""
        out: list[PullRequest] = []
        page = 1
        while len(out) < limit:
            batch = self._request("GET", f"/repos/{self.repo}/pulls",
                                  params={"state": "closed", "sort": "updated", "direction": "desc",
                                          "per_page": 100, "page": page})
            if not batch:
                break
            for d in batch:
                merged_at = d.get("merged_at")
                if not merged_at or (since and merged_at <= since):
                    continue
                out.append(PullRequest(number=d["number"], title=d.get("title") or "", body=d.get("body") or "",
                                       labels=[l["name"] for l in d.get("labels", [])], merged=True,
                                       url=d.get("html_url") or "", updated_at=merged_at))
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    def latest_release(self) -> tuple[str, str] | None:
        """(tag, published_at) of the latest release, or None."""
        try:
            d = self._request("GET", f"/repos/{self.repo}/releases/latest")
        except GitHubError:
            return None
        return d.get("tag_name", ""), d.get("published_at", "")

    # ------------------------------------------------------------------ writes (dry-run by default)
    def _write(self, kind: str, method: str, path: str, body: Any) -> Any:
        action = {"kind": kind, "method": method, "path": path, "body": body, "performed": not self.dry_run}
        self.actions.append(action)
        if self.dry_run:
            return None
        return self._request(method, path, body=body)

    def comment(self, number: int, body: str) -> Any:
        return self._write("comment", "POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body})

    def add_labels(self, number: int, labels: list[str]) -> Any:
        if not labels:
            return None
        return self._write("labels", "POST", f"/repos/{self.repo}/issues/{number}/labels", {"labels": labels})
