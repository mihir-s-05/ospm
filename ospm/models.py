"""Domain objects shared by agents, the pipeline and integrations."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Issue:
    number: int | str
    title: str
    body: str = ""
    author: str = ""
    labels: list[str] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    reactions: int = 0
    created_at: str = ""  # ISO-8601
    state: str = "open"
    url: str = ""

    @property
    def id(self) -> str:
        return f"issue-{self.number}"

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}"


@dataclass
class CIRun:
    name: str
    conclusion: str  # success | failure | cancelled | timed_out | ...
    log: str = ""
    url: str = ""
    run_id: str = ""  # stable id, e.g. "pr-412/tests (3.12)"

    @property
    def failed(self) -> bool:
        return self.conclusion in {"failure", "timed_out", "startup_failure"}


@dataclass
class PullRequest:
    number: int | str
    title: str
    body: str = ""
    author: str = ""
    diff: str = ""
    labels: list[str] = field(default_factory=list)
    ci_runs: list[CIRun] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    draft: bool = False
    merged: bool = False
    url: str = ""

    @property
    def id(self) -> str:
        return f"pr-{self.number}"

    @property
    def files(self) -> list[str]:
        return [f.path for f in parse_diff(self.diff)]


@dataclass
class FileDiff:
    path: str
    added: list[tuple[int, str]] = field(default_factory=list)  # (new line number, text)
    removed: list[str] = field(default_factory=list)


_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(diff: str) -> list[FileDiff]:
    """Parse a unified diff into per-file added/removed lines (with new-file line numbers).

    Hunk line counts are honoured, so content lines that happen to start with "+++"/"---"
    (e.g. a removed SQL comment) are not mistaken for file headers.
    """
    files: list[FileDiff] = []
    cur: FileDiff | None = None
    new_line = 0
    old_left = new_left = 0  # lines remaining in the current hunk
    for raw in diff.splitlines():
        if old_left > 0 or new_left > 0:
            if cur is not None:
                if raw.startswith("+"):
                    cur.added.append((new_line, raw[1:]))
                    new_line += 1
                    new_left -= 1
                elif raw.startswith("-"):
                    cur.removed.append(raw[1:])
                    old_left -= 1
                elif raw.startswith("\\"):
                    pass  # "\ No newline at end of file"
                else:
                    new_line += 1
                    old_left -= 1
                    new_left -= 1
            continue
        if raw.startswith("diff --git "):
            parts = raw.split(" b/", 1)
            cur = FileDiff(path=parts[1] if len(parts) == 2 else raw[len("diff --git ") :])
            files.append(cur)
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            if cur is None:
                cur = FileDiff(path=target)
                files.append(cur)
            elif target != "/dev/null":
                cur.path = target
            continue
        m = _HUNK_RE.match(raw)
        if m:
            old_left = int(m.group(1)) if m.group(1) is not None else 1
            new_line = int(m.group(2))
            new_left = int(m.group(3)) if m.group(3) is not None else 1
            if cur is None:
                cur = FileDiff(path="<unknown>")
                files.append(cur)
    return files


@dataclass
class Decision:
    """The output of one agent invocation."""

    agent: str
    item_id: str
    labels: dict[str, Any]  # field -> label (str) or list of labels (multi-label fields)
    confidence: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    fired_rules: list[str] = field(default_factory=list)
    neighbors: list[tuple[str, float]] = field(default_factory=list)
    rationale: str = ""
    used_llm: bool = False
    genome_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ITEM_TYPES = {"Issue": Issue, "CIRun": CIRun, "PullRequest": PullRequest}


def item_to_dict(item: Any) -> dict[str, Any]:
    """Serialise an Issue / CIRun / PullRequest (with its type) for storage in memory."""
    name = type(item).__name__
    if name not in _ITEM_TYPES:
        raise TypeError(f"cannot serialise {name}")
    return {"type": name, "data": asdict(item)}


def item_from_dict(d: dict[str, Any]) -> Any:
    cls = _ITEM_TYPES[d["type"]]
    data = dict(d["data"])
    if cls is PullRequest:
        data["ci_runs"] = [CIRun(**r) for r in data.get("ci_runs", [])]
    return cls(**data)
