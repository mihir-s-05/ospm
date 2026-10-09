"""GitHub webhook / Actions event handling: run the right agents and learn from maintainer actions."""

from __future__ import annotations

import re
import shlex
from typing import Any

from ..agents import AGENTS
from ..agents.triage import KIND_LABEL
from ..manager import ProjectManager
from ..render import issue_comment, pr_comment
from .github import GitHubClient

_KIND_FROM_LABEL = {v: k for k, v in KIND_LABEL.items()} | {"feature": "feature", "docs": "docs", "enhancement": "feature"}
_DECISION_FROM_LABEL = {"wontfix": "out-of-scope", "out-of-scope": "out-of-scope", "duplicate": "duplicate",
                        "spam": "spam", "invalid": "spam", "needs-info": "needs-info", "needs more info": "needs-info",
                        "accepted": "accept", "confirmed": "accept", "triaged": "accept"}
_PRIORITY_RE = re.compile(r"^(?:priority[:/ -]*)?(p[0-3])$", re.IGNORECASE)
BOT_SUFFIX = "[bot]"


def labels_to_triage(labels: list[str]) -> dict[str, str]:
    """Map a maintainer's issue labels back to triage fields (only the fields the labels determine)."""
    out: dict[str, str] = {}
    for raw in labels:
        name = raw.strip().lower()
        if name in _KIND_FROM_LABEL:
            out["kind"] = _KIND_FROM_LABEL[name]
        if name in _DECISION_FROM_LABEL:
            out["decision"] = _DECISION_FROM_LABEL[name]
        m = _PRIORITY_RE.match(name)
        if m:
            out["priority"] = m.group(1).upper()
    if "priority" in out and "decision" not in out:
        out["decision"] = "accept"  # assigning a priority means the maintainer accepted the issue
    if out.get("decision") not in (None, "accept") and "priority" not in out:
        out["priority"] = "P3"  # labelling guide: anything not accepted is P3
    return out


def parse_command(body: str) -> tuple[str, dict[str, Any], str | None] | None:
    """`/ospm <agent> field=value[,value2] ... [job=<name>]` -> (agent, labels, job)."""
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("/ospm "):
            continue
        try:
            parts = shlex.split(line[len("/ospm "):])
        except ValueError:
            return None
        if not parts or parts[0] not in AGENTS:
            return None
        agent, labels, job = parts[0], {}, None
        multi = {f.name for f in AGENTS[agent].fields if f.multi}
        for p in parts[1:]:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k == "job":
                job = v
            elif k in multi:
                labels[k] = [x for x in v.split(",") if x and x != "none"]
            else:
                labels[k] = v
        return agent, labels, job
    return None


def _is_maintainer(pm: ProjectManager, login: str) -> bool:
    return bool(login) and not login.endswith(BOT_SUFFIX) and login in pm.project.maintainers


def handle_event(pm: ProjectManager, gh: GitHubClient, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one GitHub event. Returns a summary; writes go through `gh` (dry-run unless enabled)."""
    action = payload.get("action", "")
    sender = (payload.get("sender") or {}).get("login", "")

    if event == "issues" and action in ("opened", "reopened"):
        issue = gh._issue(payload["issue"])
        dec = pm.triage_issue(issue)
        gh.comment(issue.number, issue_comment(issue, dec))
        gh.add_labels(issue.number, dec.extras.get("suggested_labels", []))
        return {"handled": "triage", "issue": issue.number, "labels": dec.labels}

    if event == "issues" and action == "labeled" and _is_maintainer(pm, sender):
        issue = gh._issue(payload["issue"])
        added = (payload.get("label") or {}).get("name")
        # learn from the label the maintainer just added (other labels may have been ospm's own suggestions)
        labels = labels_to_triage([added] if added else issue.labels)
        if not labels:
            return {"handled": "ignored", "reason": "labels carry no triage signal"}
        try:
            fb = pm.feedback("triage", issue.id, labels, source=f"label:{sender}")
        except KeyError:
            return {"handled": "ignored", "reason": "issue was never triaged by ospm"}
        return {"handled": "feedback", "issue": issue.number, **fb}

    if event == "pull_request" and action in ("opened", "reopened", "synchronize", "ready_for_review"):
        number = payload["pull_request"]["number"]
        pr = gh.get_pull(number)
        report = pm.review_pr(pr)
        gh.comment(number, pr_comment(report))
        return {"handled": "pr_review", "pr": number, "verdict": report.verdict}

    if event == "workflow_run" and action == "completed":
        run = payload.get("workflow_run") or {}
        if run.get("conclusion") not in ("failure", "timed_out"):
            return {"handled": "ignored", "reason": f"workflow conclusion {run.get('conclusion')}"}
        results = []
        for pr_ref in run.get("pull_requests") or []:
            pr = gh.get_pull(pr_ref["number"])
            report = pm.review_pr(pr)
            gh.comment(pr.number, pr_comment(report))
            results.append({"pr": pr.number, "verdict": report.verdict})
        return {"handled": "ci_triage", "results": results}

    if event == "issue_comment" and action == "created" and _is_maintainer(pm, sender):
        cmd = parse_command((payload.get("comment") or {}).get("body", ""))
        if cmd is None:
            return {"handled": "ignored", "reason": "no /ospm command"}
        agent, labels, job = cmd
        number = payload["issue"]["number"]
        is_pr = "pull_request" in payload["issue"]
        if agent == "ci_triage":
            prefix = f"pr-{number}/"
            ids = sorted({e["item_id"] for e in pm.memory.episodes("ci_triage", limit=5000)
                          if e["item_id"].startswith(prefix) and (job is None or e["item_id"] == prefix + job)})
            if len(ids) != 1:
                return {"handled": "ignored", "reason": f"need exactly one matching CI job, found {ids}"}
            item_id = ids[0]
        else:
            item_id = f"{'pr' if is_pr else 'issue'}-{number}"
        try:
            fb = pm.feedback(agent, item_id, labels, source=f"command:{sender}")
        except (KeyError, ValueError) as e:
            gh.comment(number, f"ospm could not apply that feedback: {e}")
            return {"handled": "error", "error": str(e)}
        return {"handled": "feedback", "item": item_id, **fb}

    return {"handled": "ignored", "reason": f"unhandled event {event}/{action}"}
