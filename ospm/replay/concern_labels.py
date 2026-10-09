"""LLM-based labelling of review concerns with a fixed taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..llm import BudgetExceeded, LLMError
from .tasks import ReplayTask, load_tasks, save_tasks

TAXONOMY: dict[str, str] = {
    "correctness_bug": "The code does the wrong thing or has a logic error.",
    "edge_case": "Unhandled input, boundary or unusual situation.",
    "error_handling": "Missing, wrong or overly broad handling of errors and exceptions.",
    "concurrency": "Threading, async, races, ordering, or shared-state problems.",
    "performance": "Speed, memory or algorithmic efficiency.",
    "api_compat": "Public API/CLI/config compatibility, deprecation, or breaking changes.",
    "test_gap": "Missing or weak tests.",
    "docs": "Documentation, docstrings, comments, or changelog.",
    "naming_style": "Naming, formatting, readability, and nits.",
    "design": "Structure, abstraction, where code belongs, or duplication.",
    "typing": "Type annotations and type-checking issues.",
    "dependency_build": "Dependencies, packaging, build, or CI configuration.",
    "security": "Security vulnerabilities or unsafe handling of untrusted data.",
    "clarification": "Reviewer asks a question or requests explanation without asserting a problem.",
    "other": "Anything that fits none of the above.",
}

SYSTEM = (
    "You label code-review comments (concerns) left by maintainers on pull requests.\n"
    "Taxonomy (label: definition):\n"
    + "\n".join(f"- {k}: {v}" for k, v in TAXONOMY.items())
    + "\n\nFor each concern give exactly one primary label and up to 2 secondary labels (distinct from the "
    "primary; may be empty). Also give:\n"
    "- actionable: true if the reviewer asks the author to change something, false otherwise.\n"
    "- needs_context: true if a reviewer could only raise this with knowledge beyond the diff itself (project "
    "history, other files, conventions not visible in the change), false if the diff alone suffices.\n"
    "Return one entry per concern, using the exact key given."
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["labels"],
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "primary", "secondary", "actionable", "needs_context"],
                "properties": {
                    "key": {"type": "string"},
                    "primary": {"type": "string", "enum": list(TAXONOMY)},
                    "secondary": {"type": "array", "items": {"type": "string", "enum": list(TAXONOMY)}},
                    "actionable": {"type": "boolean"},
                    "needs_context": {"type": "boolean"},
                },
            },
        }
    },
}


def _is_positive_concern(task: ReplayTask) -> bool:
    return task.kind == "concern" and bool(task.truth.get("positive"))


def _collect(tasks: list[ReplayTask]) -> list[tuple[str, dict]]:
    items = []
    for t in tasks:
        if _is_positive_concern(t):
            for i, c in enumerate(t.truth.get("concerns") or []):
                items.append((f"{t.id}#{i}", c))
    return items


def _user_prompt(batch: list[tuple[str, dict]]) -> str:
    """Concerns are listed under short opaque keys (c0, c1, ...): models reliably echo those, whereas long
    task ids (e.g. "concern:123@abc:456#0") came back mangled."""
    lines = ["Label these review concerns:", ""]
    for j, (_key, c) in enumerate(batch):
        body = (c.get("body") or "")[:600]
        lines.append(f"[c{j}] {c.get('path')}:{c.get('line')}\n{body}\n")
    return "\n".join(lines)


def _chunks(tasks: list[ReplayTask], batch_size: int) -> list[list[tuple[str, dict]]]:
    items = _collect(tasks)
    batch_size = max(1, batch_size)
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]


def _parse(batch: list[tuple[str, dict]], res: dict, out: dict[str, dict]) -> None:
    short = {f"c{j}": key for j, (key, _) in enumerate(batch)}
    for lab in res.get("labels", []):
        key = short.get(str(lab.get("key", "")).strip().strip("[]"))
        if key is None:
            continue
        primary = lab["primary"]
        sec: list[str] = []
        for s in lab.get("secondary") or []:
            if s != primary and s not in sec:
                sec.append(s)
        out[key] = {"primary": primary, "secondary": sec[:2], "actionable": bool(lab["actionable"]),
                    "needs_context": bool(lab["needs_context"])}


def label_concerns(tasks: list[ReplayTask], llm, batch_size: int = 25, effort: str = "low") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for batch in _chunks(tasks, batch_size):
        try:
            res = llm.complete_json(SYSTEM, _user_prompt(batch), SCHEMA, name="concern_labels", effort=effort)
        except (LLMError, BudgetExceeded):
            continue
        _parse(batch, res, out)
    return out


def label_concerns_batch(tasks: list[ReplayTask], model: str, meter, state_path: Path, batch_size: int = 25,
                         effort: str = "low", log=print) -> dict[str, dict]:
    """Same labels through the OpenAI Batch API (half price, asynchronous). Re-running resumes the same batch."""
    from ..llm.batch import complete_json_batch

    chunks = _chunks(tasks, batch_size)
    reqs = [(str(i), SYSTEM, _user_prompt(b)) for i, b in enumerate(chunks)]
    res = complete_json_batch(model, reqs, SCHEMA, meter=meter, name="concern_labels", effort=effort,
                              max_tokens=6000, state_path=state_path, log=log)
    out: dict[str, dict] = {}
    for i, b in enumerate(chunks):
        r = res.get(str(i))
        if isinstance(r, dict):
            _parse(b, r, out)
    return out


def apply_labels(tasks: list[ReplayTask], labels: dict[str, dict]) -> None:
    for t in tasks:
        if not _is_positive_concern(t):
            continue
        for i, c in enumerate(t.truth.get("concerns") or []):
            lab = labels.get(f"{t.id}#{i}")
            if lab is not None:
                c["llm_labels"] = dict(lab)


def summarize(labels: dict[str, dict], total: int, usage: dict) -> dict:
    counts: dict[str, int] = {}
    for lab in labels.values():
        counts[lab["primary"]] = counts.get(lab["primary"], 0) + 1
    n = len(labels)
    return {"labelled": n, "total": total, "primary_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "actionable_fraction": (sum(l["actionable"] for l in labels.values()) / n) if n else 0.0,
            "needs_context_fraction": (sum(l["needs_context"] for l in labels.values()) / n) if n else 0.0,
            "usage": usage}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ospm.replay.concern_labels")
    ap.add_argument("--data", required=True, help="directory containing tasks.jsonl")
    ap.add_argument("--llm", default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--batch", action="store_true", help="use the OpenAI Batch API (half price, asynchronous)")
    args = ap.parse_args(argv)
    from ..llm import DEFAULT_MODELS, UsageMeter, make_llm

    path = Path(args.data) / "tasks.jsonl"
    tasks = load_tasks(path)
    total = len(_collect(tasks))
    if args.batch:
        if args.llm != "openai":
            print("--batch requires --llm openai", file=sys.stderr)
            return 2
        meter = UsageMeter(args.budget)
        labels = label_concerns_batch(tasks, args.model or DEFAULT_MODELS["openai"], meter,
                                      Path(args.data) / "batches" / "concern_labels.json",
                                      log=lambda m: print(m, file=sys.stderr, flush=True))
        usage = meter.snapshot()
    else:
        llm = make_llm(args.llm, args.model, budget_usd=args.budget)
        if llm is None:
            print("no LLM configured (provider resolved to none)", file=sys.stderr)
            return 2
        labels = label_concerns(tasks, llm)
        usage = llm.meter.snapshot()
    apply_labels(tasks, labels)
    save_tasks(tasks, path)
    print(json.dumps(summarize(labels, total, usage), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
