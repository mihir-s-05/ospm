"""Skills: short per-area procedure documents the replay agents load on demand with the sandbox command `skill <name>`.

A learner creates/edits them from earlier reviews; this module holds the data model, selection and trace parsing.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
STATUSES = ("candidate", "active", "rejected")

TOOLS_NOTE = ("`hist reviews --path <file-or-dir/> [-n N]` prints earlier maintainer inline review comments on those "
              "files (as of the review time): what reviewers have asked of changes in that area before.")

_SKILL_CALL = re.compile(r"(?:^|[;&|(\n])[ \t]*skill[ \t]+([a-z0-9][a-z0-9-]{0,47})(?![A-Za-z0-9_-])")


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48].strip("-")
    if not s:
        raise ValueError(f"cannot make a skill name from {text!r}")
    return s


@dataclass
class Skill:
    name: str
    description: str  # when to use it
    body: str  # the procedure (markdown)
    paths: list[str] = field(default_factory=list)  # fnmatch globs over changed files; [] = applies everywhere
    role: str = "review"
    provenance: list[str] = field(default_factory=list)
    status: str = "active"  # candidate | active | rejected
    validation: dict = field(default_factory=dict)
    version: int = 1

    def __post_init__(self) -> None:
        self.name = slug(self.name)
        if self.status not in STATUSES:
            raise ValueError(f"bad skill status {self.status!r}")


def offered(skills: list[Skill], files: list[str], role: str = "review", k: int = 8) -> list[Skill]:
    def hit(s: Skill) -> bool:
        return not s.paths or any(fnmatch.fnmatch(f, pat) for f in files for pat in s.paths)

    ok = [s for s in skills if s.status == "active" and s.role == role and hit(s)]
    ok.sort(key=lambda s: (not s.paths, s.name))
    return ok[:k]


def skills_hash(skills: list[Skill]) -> str:
    if not skills:
        return ""
    blob = json.dumps([[s.name, s.description, s.body] for s in skills], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:6]


def index_block(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [f"- {s.name}: {' '.join(s.description.split())}" for s in skills]
    return ("\n\n## Skills\nEach skill is a procedure for this repository, learned from earlier reviews. When the "
            "change touches its area, load it with `skill <name>` (`skill` alone lists them) and follow it.\n"
            + "\n".join(lines))


def loaded_skills(trace: dict) -> list[str]:
    """Skill names loaded in a trace dict (Trace.to_dict format), children included, in order of first load."""
    out: list[str] = []

    def walk(t: dict) -> None:
        for step in t.get("steps", []):
            for call in step.get("calls", []):
                cmd = (call.get("arguments") or {}).get("command")
                if isinstance(cmd, str):
                    out.extend(m.group(1) for m in _SKILL_CALL.finditer(cmd))
        for c in t.get("children", []):
            walk(c)

    walk(trace)
    return list(dict.fromkeys(out))


def save(skills: list[Skill], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(s) for s in skills], indent=1), encoding="utf-8")


def load(path: str | Path) -> list[Skill]:
    return [Skill(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
