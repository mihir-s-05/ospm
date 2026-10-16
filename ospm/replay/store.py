"""History store: a local, replayable copy of a repository's issues, PRs, reviews, comments, label events and CI.

Everything is stored in one SQLite file. Every read API takes an ``as_of`` timestamp and never returns anything
that happened after it, so agents replaying the past cannot see the future (no label leakage, no later comments).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS items (            -- issues and pull requests
    number INTEGER PRIMARY KEY,
    is_pr INTEGER NOT NULL,
    title TEXT, body TEXT, author TEXT, author_association TEXT,
    state TEXT, state_reason TEXT,
    created_at TEXT NOT NULL, closed_at TEXT, updated_at TEXT,
    labels_json TEXT,                          -- labels as of mirror time (do not use for replay)
    reactions INTEGER DEFAULT 0,
    in_window INTEGER DEFAULT 0                -- created inside the mirrored window (full detail fetched)
);
CREATE INDEX IF NOT EXISTS ix_items_created ON items(created_at);
CREATE TABLE IF NOT EXISTS issue_events (     -- labeled / unlabeled / closed / reopened / marked_as_duplicate ...
    id INTEGER PRIMARY KEY, number INTEGER, event TEXT, actor TEXT, label TEXT, created_at TEXT,
    commit_id TEXT, extra_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_number ON issue_events(number, created_at);
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY, number INTEGER, author TEXT, author_association TEXT, created_at TEXT, body TEXT
);
CREATE INDEX IF NOT EXISTS ix_comments_number ON comments(number, created_at);
CREATE TABLE IF NOT EXISTS pulls (
    number INTEGER PRIMARY KEY, head_sha TEXT, head_ref TEXT, head_repo TEXT, base_ref TEXT, base_sha TEXT,
    merged_at TEXT, merge_commit_sha TEXT, draft INTEGER, have_head INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, number INTEGER, author TEXT, author_association TEXT, state TEXT, body TEXT,
    submitted_at TEXT, commit_id TEXT
);
CREATE INDEX IF NOT EXISTS ix_reviews_number ON reviews(number, submitted_at);
CREATE TABLE IF NOT EXISTS review_comments (
    id INTEGER PRIMARY KEY, number INTEGER, review_id INTEGER, author TEXT, author_association TEXT,
    path TEXT, line INTEGER, original_line INTEGER, side TEXT, diff_hunk TEXT, body TEXT, created_at TEXT,
    commit_id TEXT, in_reply_to INTEGER
);
CREATE INDEX IF NOT EXISTS ix_rc_number ON review_comments(number, created_at);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER, attempt INTEGER, name TEXT, event TEXT, head_sha TEXT, head_branch TEXT, pr_numbers_json TEXT,
    status TEXT, conclusion TEXT, created_at TEXT, updated_at TEXT, html_url TEXT,
    PRIMARY KEY (id, attempt)
);
CREATE INDEX IF NOT EXISTS ix_runs_sha ON runs(head_sha);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY, run_id INTEGER, attempt INTEGER, name TEXT, conclusion TEXT,
    started_at TEXT, completed_at TEXT, steps_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_run ON jobs(run_id, attempt);
CREATE TABLE IF NOT EXISTS job_logs (job_id INTEGER PRIMARY KEY, log_z BLOB, original_bytes INTEGER);
"""

LOG_HEAD = 40_000
LOG_TAIL = 160_000


def trim_log(text: str) -> str:
    """Keep the head and the (usually decisive) tail of very long CI logs."""
    if len(text) <= LOG_HEAD + LOG_TAIL:
        return text
    return text[:LOG_HEAD] + f"\n... [{len(text) - LOG_HEAD - LOG_TAIL} characters trimmed by ospm] ...\n" + text[-LOG_TAIL:]


# bot accounts without the "[bot]" suffix (PyTorch's land and sync bots, Rust's merge queue and triage bots are
# ordinary user accounts)
BOT_LOGINS = {"copilot", "pre-commit-ci", "github-actions", "dependabot", "pytorchmergebot", "pytorchbot",
              "pytorch-bot", "facebook-github-bot", "bors", "rustbot", "rust-timer", "rust-log-analyzer",
              "rust-highfive", "rfcbot"}


def is_bot(login: str | None) -> bool:
    login = (login or "").lower()
    return login.endswith("[bot]") or login in BOT_LOGINS


MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


class HistoryStore:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = str(path)
        self._lock = threading.RLock()
        if readonly:
            if not Path(self.path).exists():
                raise FileNotFoundError(self.path)
            uri = "file:" + Path(self.path).resolve().as_posix() + "?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ low level
    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def upsert(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0])
        sql = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        with self._lock:
            self._conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
            self._conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        self.upsert("meta", [{"key": key, "value": json.dumps(value)}])

    def meta(self, key: str, default: Any = None) -> Any:
        rows = self.q("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows else default

    def maintainers(self, min_approvals: int = 2) -> set[str]:
        """People whose review carries maintainer weight. GitHub's author_association misses trusted reviewers
        in large projects (e.g. vLLM committers often show as CONTRIBUTOR), so this also includes anyone who
        approved at least `min_approvals` PRs that were then merged."""
        if getattr(self, "_maintainers", None) is None:
            names = {r["author"] for r in self.q("SELECT DISTINCT author, author_association FROM reviews")
                     if r["author_association"] in MAINTAINER_ASSOCIATIONS and not is_bot(r["author"])}
            for r in self.q("SELECT r.author, COUNT(DISTINCT r.number) AS n FROM reviews r JOIN pulls p "
                            "ON p.number=r.number WHERE r.state='APPROVED' AND p.merged_at IS NOT NULL "
                            "GROUP BY r.author"):
                if r["n"] >= min_approvals and not is_bot(r["author"]):
                    names.add(r["author"])
            self._maintainers = names
        return self._maintainers

    def is_maintainer(self, login: str, association: str | None = None) -> bool:
        if is_bot(login):
            return False
        return (association in MAINTAINER_ASSOCIATIONS) or login in self.maintainers()

    def put_log(self, job_id: int, text: str) -> None:
        trimmed = trim_log(text)
        self.upsert("job_logs", [{"job_id": job_id, "log_z": zlib.compress(trimmed.encode("utf-8", "replace"), 9),
                                  "original_bytes": len(text)}])

    def has_log(self, job_id: int) -> bool:
        return bool(self.q("SELECT 1 FROM job_logs WHERE job_id=?", (job_id,)))

    # ------------------------------------------------------------------ as-of reads (no future leakage)
    def item(self, number: int, as_of: str) -> dict[str, Any] | None:
        rows = self.q("SELECT * FROM items WHERE number=? AND created_at<=?", (number, as_of))
        if not rows:
            return None
        d = dict(rows[0])
        d.pop("labels_json", None)  # labels at mirror time would leak the future; use labels_at()
        if d["closed_at"] and d["closed_at"] > as_of:
            d["state"], d["state_reason"], d["closed_at"] = "open", None, None
        d["labels"] = self.labels_at(number, as_of)
        return d

    def labels_at(self, number: int, as_of: str) -> list[str]:
        labels: list[str] = []
        for r in self.q("SELECT event, label FROM issue_events WHERE number=? AND created_at<=? "
                        "AND event IN ('labeled','unlabeled') ORDER BY created_at, id", (number, as_of)):
            if r["event"] == "labeled" and r["label"] not in labels:
                labels.append(r["label"])
            elif r["event"] == "unlabeled" and r["label"] in labels:
                labels.remove(r["label"])
        return labels

    def comments(self, number: int, as_of: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.q("SELECT * FROM comments WHERE number=? AND created_at<=? ORDER BY created_at",
                                        (number, as_of))]

    def reviews(self, number: int, as_of: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.q("SELECT * FROM reviews WHERE number=? AND submitted_at<=? "
                                        "ORDER BY submitted_at", (number, as_of))]

    def review_comments(self, number: int, as_of: str) -> list[dict[str, Any]]:
        """Inline comments visible at `as_of`. Comments drafted in a pending review are created *before* the
        review is submitted but are only published with it, so they follow their review's submission time."""
        return [dict(r) for r in self.q(
            "SELECT c.* FROM review_comments c LEFT JOIN reviews r ON r.id=c.review_id "
            "WHERE c.number=? AND c.created_at<=? AND (r.id IS NULL OR r.submitted_at<=?) ORDER BY c.created_at",
            (number, as_of, as_of))]

    def pull(self, number: int) -> dict[str, Any] | None:
        rows = self.q("SELECT * FROM pulls WHERE number=?", (number,))
        return dict(rows[0]) if rows else None

    def runs_for_sha(self, sha: str, as_of: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.q("SELECT * FROM runs WHERE head_sha=? AND created_at<=? ORDER BY created_at",
                                        (sha, as_of))]

    def jobs(self, run_id: int, attempt: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self.q("SELECT * FROM jobs WHERE run_id=? AND attempt=? ORDER BY id",
                                        (run_id, attempt))]

    def job(self, job_id: int) -> dict[str, Any] | None:
        rows = self.q("SELECT * FROM jobs WHERE id=?", (job_id,))
        return dict(rows[0]) if rows else None

    def job_log(self, job_id: int) -> str | None:
        rows = self.q("SELECT log_z FROM job_logs WHERE job_id=?", (job_id,))
        return zlib.decompress(rows[0]["log_z"]).decode("utf-8", "replace") if rows else None

    def search_items(self, query: str, as_of: str, limit: int = 10, include_prs: bool = False) -> list[dict[str, Any]]:
        """TF-IDF search over titles/bodies of items created before as_of."""
        from ..text import TfidfIndex

        sql = "SELECT number, title, body, created_at, is_pr FROM items WHERE created_at<=?"
        if not include_prs:
            sql += " AND is_pr=0"
        idx = TfidfIndex()
        rows = self.q(sql, (as_of,))
        for r in rows:
            idx.add(str(r["number"]), f"{r['title']}\n{(r['body'] or '')[:1500]}", dict(r))
        out = []
        for _, sim, payload in idx.query(query, k=limit):
            item = self.item(payload["number"], as_of)
            if item:
                out.append({"number": item["number"], "title": item["title"], "state": item["state"],
                            "labels": item["labels"], "created_at": item["created_at"], "similarity": round(sim, 3)})
        return out

    def stats(self) -> dict[str, Any]:
        out = {}
        for t in ("items", "issue_events", "comments", "pulls", "reviews", "review_comments", "runs", "jobs",
                  "job_logs"):
            out[t] = self.q(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
        out["window_issues"] = self.q("SELECT COUNT(*) AS n FROM items WHERE in_window=1 AND is_pr=0")[0]["n"]
        out["window_prs"] = self.q("SELECT COUNT(*) AS n FROM items WHERE in_window=1 AND is_pr=1")[0]["n"]
        out["prs_with_local_head"] = self.q("SELECT COUNT(*) AS n FROM pulls WHERE have_head=1")[0]["n"]
        out["log_bytes_compressed"] = self.q("SELECT COALESCE(SUM(LENGTH(log_z)),0) AS n FROM job_logs")[0]["n"]
        return out
