"""Markdown rendering for GitHub comments and CLI output."""

from __future__ import annotations

from typing import Any

from .agents.planning import BacklogItem, ReleasePlan
from .manager import PRReport
from .models import Decision, Issue

FOOTER = "\n\n<sub>🤖 ospm — automated triage; maintainers have the final say. Reply `/ospm <agent> field=value` to correct me.</sub>"

VERDICT_TEXT = {
    "ready_for_review": "✅ **Ready for human review**",
    "approve_with_nits": "🟢 **Looks good — minor nits**",
    "changes_requested": "🟠 **Changes requested**",
    "security_review_required": "🔴 **Security review required**",
    "ci_failing": "❌ **CI is failing**",
    "ci_retry": "🔁 **CI failed for infrastructure reasons — re-run recommended**",
}


def issue_comment(issue: Issue, dec: Decision) -> str:
    lab = dec.labels
    lines = [dec.extras.get("reply", "").strip(), "", "<details><summary>Triage details</summary>", ""]
    lines.append(f"- kind: `{lab['kind']}` · decision: `{lab['decision']}` · priority: `{lab['priority']}`")
    if dec.extras.get("component"):
        lines.append(f"- component: `{dec.extras['component']}`")
    if dec.extras.get("duplicate_of"):
        lines.append(f"- duplicate of: {dec.extras['duplicate_of']}")
    if dec.extras.get("suggested_labels"):
        lines.append("- labels: " + ", ".join(f"`{l}`" for l in dec.extras["suggested_labels"]))
    if dec.rationale:
        lines.append(f"- rationale: {dec.rationale}")
    lines += ["", "</details>"]
    return "\n".join(lines).strip() + FOOTER


def pr_comment(report: PRReport) -> str:
    out = ["## ospm pull-request report", "", VERDICT_TEXT[report.verdict], ""]
    out += [f"- {r}" for r in report.reasons]
    if report.ci:
        out += ["", "### CI failures"]
        for run, d in report.ci:
            out += ["", f"**{run.name}** — `{d.labels['category']}`"]
            if d.extras.get("root_cause"):
                out.append(f"- root cause: `{d.extras['root_cause'][:300]}`")
            if d.extras.get("failing_tests"):
                out.append("- failing tests: " + ", ".join(f"`{t}`" for t in d.extras["failing_tests"][:10]))
            if d.extras.get("suggested_fix"):
                out.append(f"- suggested fix: {d.extras['suggested_fix']}")
    for title, dec in (("Security", report.security), ("Code review", report.review)):
        comments = dec.extras.get("comments", [])
        if not comments:
            continue
        out += ["", f"### {title}"]
        for c in comments:
            loc = f"`{c['path']}:{c['line']}`" if c.get("path") else "(general)"
            out.append(f"- **{c.get('severity', 'minor')}** {loc} `{c.get('finding')}` — {c.get('message', '')}")
        if dec.extras.get("summary"):
            out += ["", dec.extras["summary"]]
    return "\n".join(out) + FOOTER


def backlog_table(items: list[BacklogItem], plan: dict[str, list[BacklogItem]] | None = None, limit: int = 30) -> str:
    out = ["| rank | issue | kind | pri | score | ready | milestone | owners |", "|---:|---|---|---|---:|:-:|---|---|"]
    for i, it in enumerate(items[:limit], 1):
        out.append(f"| {i} | #{it.issue.number} {it.issue.title[:60]} | {it.triage.labels['kind']} | "
                   f"{it.triage.labels['priority']} | {it.score:.1f} | {'✓' if it.ready else '…'} | "
                   f"{it.milestone or '—'} | {', '.join(it.assignees) or '—'} |")
    if plan:
        out += ["", "### Milestone plan"]
        for name, its in plan.items():
            pts = sum(i.effort for i in its)
            out.append(f"- **{name}** ({pts} pts): " + (", ".join(f"#{i.issue.number}" for i in its) or "—"))
    return "\n".join(out)


def release_notes(plan: ReleasePlan) -> str:
    return f"Proposed bump: **{plan.bump}** ({plan.current_version} → {plan.next_version})\n\n" + plan.notes()


def evolution_report(outcome: dict[str, Any]) -> str:
    s = outcome["summary"]
    out = [f"## Evolution run `{s['run_id']}` — agent `{outcome['agent']}`", "",
           f"- stop reason: {s['stop_reason']} after {s['iterations']} iterations, {s['metric_calls']} agent calls, "
           f"{s['elapsed_s']}s",
           f"- Pareto-set (val) score: {s['seed_val']:.4f} → {s['best_val']:.4f}"]
    if s.get("test"):
        out.append(f"- held-out test score: {s['test']['seed']:.4f} → {s['test']['best']:.4f}")
    out.append(f"- promoted: **{outcome['promoted']}** ({'; '.join(outcome['reasons'])})")
    for name, u in (s.get("llm_usage") or {}).items():
        out.append(f"- {name} LLM {u['model']}: {u['calls']} calls, {u['cache_hits']} cache hits, "
                   f"${u['cost_usd']:.4f}")
    out += ["", "| candidate | operator | iter | val | train | rules |", "|---|---|---:|---:|---:|---:|"]
    for c in s["accepted"]:
        mark = " ⭐" if c["id"] == s["best_id"] else ""
        train = "—" if c["train"] is None else f"{c['train']:.4f}"
        out.append(f"| `{c['id']}`{mark} | {c['op']} | {c['iter']} | {c['val']:.4f} | {train} | {c['rules']} |")
    return "\n".join(out)
