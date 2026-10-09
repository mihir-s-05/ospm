"""Backlog grooming, milestone planning and release planning (deterministic, explainable)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..models import Decision, Issue, PullRequest
from ..project import ProjectContext

PRIORITY_POINTS = {"P0": 100.0, "P1": 60.0, "P2": 30.0, "P3": 10.0}
EFFORT_POINTS = {"bug": 2, "docs": 1, "question": 1, "feature": 5}


@dataclass
class BacklogItem:
    issue: Issue
    triage: Decision
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ready: bool = False
    effort: int = 1
    assignees: list[str] = field(default_factory=list)
    milestone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue.number,
            "title": self.issue.title,
            "kind": self.triage.labels["kind"],
            "priority": self.triage.labels["priority"],
            "score": round(self.score, 2),
            "ready": self.ready,
            "effort": self.effort,
            "assignees": self.assignees,
            "milestone": self.milestone,
            "reasons": self.reasons,
        }


def _age_days(created_at: str, now: datetime) -> float:
    if not created_at:
        return 0.0
    try:
        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 86400)


class BacklogPlanner:
    """Ranks accepted issues and packs them into capacity-bounded milestones."""

    def __init__(self, project: ProjectContext, now: datetime | None = None):
        self.project = project
        self.now = now or datetime.now(timezone.utc)

    def groom(self, triaged: list[tuple[Issue, Decision]]) -> list[BacklogItem]:
        items: list[BacklogItem] = []
        roadmap_text = " ".join(self.project.roadmap).lower()
        for issue, dec in triaged:
            if dec.labels["decision"] not in ("accept", "needs-info"):
                continue
            it = BacklogItem(issue=issue, triage=dec)
            pr = dec.labels["priority"]
            it.score = PRIORITY_POINTS[pr]
            it.reasons.append(f"priority {pr} (+{PRIORITY_POINTS[pr]:.0f})")
            if issue.reactions:
                bonus = 8 * math.log1p(issue.reactions)
                it.score += bonus
                it.reasons.append(f"{issue.reactions} reactions (+{bonus:.1f})")
            age = _age_days(issue.created_at, self.now)
            if age > 0:
                bonus = min(15.0, age / 7)
                it.score += bonus
                it.reasons.append(f"waiting {age:.0f} days (+{bonus:.1f})")
            comp = dec.extras.get("component", "unknown")
            if comp != "unknown" and comp.lower() in roadmap_text:
                it.score += 10
                it.reasons.append(f"component '{comp}' is on the roadmap (+10)")
            if dec.labels["kind"] == "bug" and "regression" in issue.text.lower():
                it.score += 10
                it.reasons.append("regression (+10)")
            it.ready = dec.labels["decision"] == "accept"
            if not it.ready:
                it.score *= 0.5
                it.reasons.append("not ready: waiting for reporter info (x0.5)")
            it.effort = EFFORT_POINTS.get(dec.labels["kind"], 2)
            it.assignees = self.project.owners_of(comp)
            items.append(it)
        items.sort(key=lambda x: (-x.score, str(x.issue.number)))
        return items

    def plan_milestones(self, items: list[BacklogItem], names: list[str] | None = None,
                        capacity: int | None = None) -> dict[str, list[BacklogItem]]:
        """Greedy: P0s always go in the first milestone; others fill remaining capacity in rank order."""
        capacity = capacity or self.project.milestone_capacity
        names = names or ["next", "next+1", "next+2"]
        plan: dict[str, list[BacklogItem]] = {n: [] for n in names}
        used = {n: 0 for n in names}
        for it in items:
            if not it.ready:
                continue
            if it.triage.labels["priority"] == "P0":
                target = names[0]
            else:
                target = next((n for n in names if used[n] + it.effort <= capacity), None)
            if target is None:
                continue
            plan[target].append(it)
            used[target] += it.effort
            it.milestone = target
        return plan


_CONVENTIONAL = re.compile(r"^(?P<type>\w+)(\((?P<scope>[^)]*)\))?(?P<bang>!)?:\s*(?P<desc>.+)$")
SECTION_FOR = {"feat": "Features", "fix": "Bug fixes", "perf": "Performance", "docs": "Documentation",
               "refactor": "Internal", "test": "Internal", "chore": "Internal", "ci": "Internal", "build": "Internal"}
LABEL_SECTION = {"enhancement": "Features", "feature": "Features", "bug": "Bug fixes",
                 "documentation": "Documentation", "security": "Security"}


@dataclass
class ReleasePlan:
    current_version: str
    next_version: str
    bump: str
    sections: dict[str, list[str]]
    breaking: list[str]

    def notes(self) -> str:
        out = [f"## {self.next_version}", ""]
        if self.breaking:
            out += ["### ⚠ Breaking changes", *[f"- {b}" for b in self.breaking], ""]
        for name in ("Security", "Features", "Bug fixes", "Performance", "Documentation", "Internal", "Other"):
            if self.sections.get(name):
                out += [f"### {name}", *[f"- {e}" for e in self.sections[name]], ""]
        return "\n".join(out).rstrip() + "\n"


def bump_version(version: str, bump: str) -> str:
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", version)
    if not m:
        raise ValueError(f"not a semantic version: {version!r}")
    major, minor, patch = (int(x) for x in m.groups())
    prefix = "v" if version.startswith("v") else ""
    if bump == "major":
        # pre-1.0 projects conventionally bump minor for breaking changes
        return f"{prefix}{major}.{minor + 1}.0" if major == 0 else f"{prefix}{major + 1}.0.0"
    if bump == "minor":
        return f"{prefix}{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{prefix}{major}.{minor}.{patch + 1}"
    return version


def plan_release(current_version: str, merged: list[PullRequest]) -> ReleasePlan:
    sections: dict[str, list[str]] = {}
    breaking: list[str] = []
    level = 0  # 0 none, 1 patch, 2 minor, 3 major
    for pr in merged:
        m = _CONVENTIONAL.match(pr.title.strip())
        labels = {l.lower() for l in pr.labels}
        desc = m.group("desc") if m else pr.title.strip()
        entry = f"{desc} (#{pr.number})"
        if (m and m.group("bang")) or "breaking change" in pr.body.lower() or "breaking" in labels:
            breaking.append(entry)
            level = max(level, 3)
        section = next((LABEL_SECTION[l] for l in labels if l in LABEL_SECTION), None)
        if section is None and m:
            section = SECTION_FOR.get(m.group("type").lower())
        section = section or "Other"
        sections.setdefault(section, []).append(entry)
        if section == "Features":
            level = max(level, 2)
        elif section != "Internal":  # internal-only changes do not warrant a release
            level = max(level, 1)
    bump = {0: "none", 1: "patch", 2: "minor", 3: "major"}[level]
    return ReleasePlan(current_version, bump_version(current_version, bump), bump, sections, breaking)
