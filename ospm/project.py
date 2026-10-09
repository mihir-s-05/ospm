"""Project context: scope, components, conventions, roadmap. Loaded from a TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .text import content_tokens


@dataclass
class Component:
    name: str
    keywords: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    name: str = "unnamed-project"
    description: str = ""
    in_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    out_of_scope_keywords: list[str] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    security_policy: str = ""
    roadmap: list[str] = field(default_factory=list)
    maintainers: list[str] = field(default_factory=list)
    library_paths: list[str] = field(default_factory=lambda: ["src/"])
    test_paths: list[str] = field(default_factory=lambda: ["tests/"])
    milestone_capacity: int = 20

    @classmethod
    def load(cls, path: str | Path) -> "ProjectContext":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectContext":
        proj = data.get("project", data)
        comps = [Component(**c) for c in data.get("components", [])]
        return cls(
            name=proj.get("name", "unnamed-project"),
            description=proj.get("description", ""),
            in_scope=list(proj.get("in_scope", [])),
            out_of_scope=list(proj.get("out_of_scope", [])),
            out_of_scope_keywords=list(proj.get("out_of_scope_keywords", [])),
            components=comps,
            conventions=list(proj.get("conventions", [])),
            security_policy=proj.get("security_policy", ""),
            roadmap=list(proj.get("roadmap", [])),
            maintainers=list(proj.get("maintainers", [])),
            library_paths=list(proj.get("library_paths", ["src/"])),
            test_paths=list(proj.get("test_paths", ["tests/"])),
            milestone_capacity=int(proj.get("milestone_capacity", 20)),
        )

    def render(self) -> str:
        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {x}" for x in items) if items else "- (not specified)"

        parts = [f"# Project: {self.name}", self.description.strip()]
        parts += ["## In scope", bullets(self.in_scope), "## Out of scope", bullets(self.out_of_scope)]
        if self.components:
            parts += ["## Components", bullets([f"{c.name}: {', '.join(c.keywords)}" for c in self.components])]
        if self.conventions:
            parts += ["## Coding conventions", bullets(self.conventions)]
        if self.roadmap:
            parts += ["## Roadmap focus", bullets(self.roadmap)]
        return "\n\n".join(p for p in parts if p)

    def guess_component(self, text: str) -> str:
        toks = set(content_tokens(text))
        best, best_score = "unknown", 0
        for c in self.components:
            score = sum(1 for k in c.keywords if set(content_tokens(k)) <= toks and content_tokens(k))
            if score > best_score:
                best, best_score = c.name, score
        return best

    def component_for_path(self, path: str) -> str:
        for c in self.components:
            if any(path.startswith(p) for p in c.paths):
                return c.name
        return "unknown"

    def owners_of(self, component: str) -> list[str]:
        for c in self.components:
            if c.name == component:
                return list(c.owners)
        return []

    def is_library_path(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.library_paths) and not self.is_test_path(path)

    def is_test_path(self, path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        return any(path.startswith(p) for p in self.test_paths) or name.startswith("test_") or name.endswith("_test.py")
