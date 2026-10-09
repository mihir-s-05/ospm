"""Issue triage: is the report valid and in scope, what kind is it, how urgent, is it a duplicate?"""

from __future__ import annotations

import re
from typing import Any

from ..genome import Rule
from ..models import Issue
from ..project import ProjectContext
from ..text import truncate
from .base import Agent, Case, CaseBase, FieldSpec

KIND_LABEL = {"bug": "bug", "feature": "enhancement", "docs": "documentation", "question": "question"}
DECISION_LABEL = {"needs-info": "needs-info", "out-of-scope": "wontfix", "duplicate": "duplicate", "spam": "spam"}

_DETAIL_MARKERS = ("```", "traceback", "steps to reproduce", "steps:", "version", "expected", "actual", "repro", "error:")


class TriageAgent(Agent):
    name = "triage"
    title = "issue triage lead"
    fields = (
        FieldSpec("kind", ("bug", "feature", "docs", "question"), default="question",
                  description="bug = existing behaviour is wrong; feature = new capability; docs = documentation "
                              "change; question = usage/support request."),
        FieldSpec("decision", ("accept", "needs-info", "out-of-scope", "duplicate", "spam"), default="accept",
                  weight=1.5,
                  description="accept = valid, actionable and in scope; needs-info = plausibly valid but not "
                              "actionable without more details from the reporter; out-of-scope = valid request the "
                              "project will not do; duplicate = already tracked by an existing issue; spam = not a "
                              "genuine report."),
        FieldSpec("priority", ("P0", "P1", "P2", "P3"), default="P2",
                  description="P0 = critical (data loss, security, crash affecting most users, broken release); "
                              "P1 = major functionality broken without workaround or high-demand feature; "
                              "P2 = normal; P3 = minor/cosmetic, questions, and anything not accepted."),
    )
    default_params = {**Agent.default_params, "dup_threshold": 0.5}
    knn_ignore = {"decision": frozenset({"duplicate"})}
    seed_instructions = """
Triage each new issue the way an experienced maintainer would.
1. Decide whether it is a genuine report (not spam) and whether it falls within the project's stated scope.
2. Classify the kind of request.
3. If it is a bug report without enough information to reproduce (version, steps, error output), ask for it
   (needs-info) instead of accepting it.
4. Mark it a duplicate only when an existing issue clearly describes the same problem.
5. Assign priority by user impact and urgency; anything not accepted is P3.
"""
    seed_rules = (
        {"id": "bug-words", "field": "kind", "value": "bug",
         "any_of": ["crash", "crashes", "exception", "traceback", "broken", "regression", "bug"], "weight": 1.0},
        {"id": "feature-words", "field": "kind", "value": "feature",
         "any_of": ["feature request", "add support", "would be nice", "please add", "proposal"], "weight": 1.0},
        {"id": "docs-words", "field": "kind", "value": "docs",
         "any_of": ["documentation", "docs", "readme"], "weight": 1.0},
        {"id": "question-words", "field": "kind", "value": "question",
         "any_of": ["how do i", "how can i", "is it possible", "question"], "weight": 1.0},
        {"id": "spam-words", "field": "decision", "value": "spam",
         "any_of": ["buy now", "casino", "crypto", "seo services", "discount code"], "weight": 1.5},
        {"id": "p0-words", "field": "priority", "value": "P0",
         "any_of": ["data loss", "security vulnerability", "deletes"], "weight": 1.0},
        {"id": "p3-words", "field": "priority", "value": "P3",
         "any_of": ["typo", "cosmetic", "nice to have"], "weight": 0.8},
    )

    def __init__(self, *args, tracker: CaseBase | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # existing issues in the repository, for duplicate detection
        self.tracker = tracker or CaseBase()

    @classmethod
    def project_rules(cls, project: ProjectContext) -> list[Rule]:
        if not project.out_of_scope_keywords:
            return []
        return [
            Rule(id="out-of-scope-keywords", field="decision", value="out-of-scope",
                 any_of=list(project.out_of_scope_keywords), weight=1.2,
                 rationale="project scope statement", origin="seed")
        ]

    # ------------------------------------------------------------------ hooks
    def item_id(self, item: Issue) -> str:
        return item.id

    def item_text(self, item: Issue) -> str:
        return f"{item.title}\n{item.body}"

    def render_item(self, item: Issue) -> str:
        lines = [f"Issue #{item.number}: {item.title}"]
        if item.author:
            lines.append(f"Author: {item.author}")
        if item.labels:
            lines.append(f"Existing labels: {', '.join(item.labels)}")
        dup = self.find_duplicate(item)
        if dup:
            cid, sim, case = dup
            lines.append(f"Possible duplicate of {cid} (similarity {sim:.2f}): {case.text.splitlines()[0]}")
        lines.append("")
        lines.append(truncate(item.body or "(empty body)", 6000))
        return "\n".join(lines)

    def find_duplicate(self, item: Issue):
        """Most similar *earlier* issue in the tracker above the duplicate threshold, or None."""
        thr = self.p("dup_threshold")
        for cid, sim, case in self.tracker.neighbors(self.item_text(item), 5, {item.id}):
            if sim < thr:
                break
            other = case.labels.get("number")
            if isinstance(other, int) and isinstance(item.number, int) and other >= item.number:
                continue
            return cid, sim, case
        return None

    def base_scores(self, item: Issue, text: str) -> dict[str, dict[str, float]]:
        lower = text.lower()
        decision: dict[str, float] = {"accept": 0.3}
        kind: dict[str, float] = {}
        priority: dict[str, float] = {"P2": 0.3}

        if len(re.findall(r"https?://", text)) >= 3:
            decision["spam"] = 1.0
        has_detail = any(m in lower for m in _DETAIL_MARKERS)
        body_len = len(item.body.strip())
        if body_len < 60 and not has_detail:
            decision["needs-info"] = 0.8
        elif body_len < 160 and not has_detail:
            decision["needs-info"] = 0.3
        if item.title.strip().endswith("?"):
            kind["question"] = 0.6
        dup = self.find_duplicate(item)
        if dup:
            thr = self.p("dup_threshold")
            decision["duplicate"] = 1.0 + 2.0 * (dup[1] - thr) / max(1e-6, 1.0 - thr)
        if item.reactions >= 10:
            priority["P1"] = 0.5
        return {"decision": decision, "kind": kind, "priority": priority}

    def extras(self, item: Issue, text: str, labels: dict[str, Any], fired) -> dict[str, Any]:
        out: dict[str, Any] = {
            "summary": item.title.strip(),
            "component": self.project.guess_component(text),
        }
        lower = text.lower()
        missing = []
        if labels["kind"] == "bug":
            if "version" not in lower and not re.search(r"\b\d+\.\d+(\.\d+)?\b", lower):
                missing.append("the version you are running")
            if not any(m in lower for m in ("steps", "repro", "to reproduce", "run `", "```")):
                missing.append("minimal steps to reproduce")
            if not any(m in lower for m in ("traceback", "error", "exception", "log")):
                missing.append("the full error output / traceback")
            if not any(m in lower for m in ("expected", "should")):
                missing.append("expected vs. actual behaviour")
        out["missing_info"] = missing if labels["decision"] == "needs-info" else []
        dup = self.find_duplicate(item) if labels["decision"] == "duplicate" else None
        out["duplicate_of"] = dup[0] if dup else ""
        out["suggested_labels"] = self.suggested_labels(labels)
        out["reply"] = self.template_reply(labels, out)
        return out

    def extra_schema(self) -> dict[str, dict]:
        components = sorted({c.name for c in self.project.components} | {"unknown"})
        return {
            "summary": {"type": "string"},
            "component": {"type": "string", "enum": components},
            "missing_info": {"type": "array", "items": {"type": "string"}},
            "reply": {"type": "string"},
        }

    def merge_llm_extras(self, item, text, labels, fired, heuristic_extras, llm_extras):
        merged = super().merge_llm_extras(item, text, labels, fired, heuristic_extras, llm_extras)
        if labels["decision"] != "needs-info":
            merged["missing_info"] = []
        merged["suggested_labels"] = self.suggested_labels(labels)
        return merged

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def suggested_labels(labels: dict[str, Any]) -> list[str]:
        out = []
        if labels["decision"] in ("accept", "needs-info"):
            out.append(KIND_LABEL[labels["kind"]])
            if labels["decision"] == "accept":
                out.append(f"priority: {labels['priority']}")
        if labels["decision"] in DECISION_LABEL:
            out.append(DECISION_LABEL[labels["decision"]])
        return out

    def template_reply(self, labels: dict[str, Any], extras: dict[str, Any]) -> str:
        d = labels["decision"]
        if d == "accept":
            return (f"Thanks for the report! Triaged as **{labels['kind']}** with priority **{labels['priority']}** "
                    f"(component: {extras['component']}). It has been added to the backlog.")
        if d == "needs-info":
            asks = "\n".join(f"- {m}" for m in extras["missing_info"]) or "- more details about the problem"
            return f"Thanks for reporting! To investigate we need a bit more information:\n{asks}"
        if d == "duplicate":
            ref = extras.get("duplicate_of") or "an existing issue"
            return f"Thanks! This looks like a duplicate of {ref}; please follow and add details there."
        if d == "out-of-scope":
            return (f"Thanks for the suggestion. This is outside the scope of {self.project.name}, so we won't "
                    "pursue it here; a plugin or separate project may be a better fit.")
        return "Closing: this does not appear to be a genuine report."


def issue_case(issue: Issue) -> Case:
    """Tracker entry for duplicate detection."""
    return Case(id=issue.id, text=f"{issue.title}\n{issue.body}", labels={"number": issue.number, "state": issue.state})
