"""Evolvable agent configuration ("genome").

Non-parametric evolution never touches LLM weights. Everything an agent learns lives here:

* ``instructions`` – natural-language policy given to the LLM (rewritten by reflective mutation),
* ``rules``        – structured, executable playbook entries; they drive the offline heuristic policy
                     *and* are rendered into the LLM prompt, so one learned lesson serves both modes,
* ``params``       – numeric knobs (retrieval depth, score weights, thresholds),
* ``exemplars``    – ids of solved cases pinned as few-shot examples.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .text import normalize, phrase_in

_MAX_PATTERN_LEN = 300


@dataclass
class Rule:
    id: str
    field: str
    value: str
    any_of: list[str] = field(default_factory=list)  # token phrases; rule triggers if any is present
    all_of: list[str] = field(default_factory=list)  # token phrases; all must be present
    none_of: list[str] = field(default_factory=list)  # token phrases that veto the rule
    pattern: str = ""  # optional regex, matched case-insensitively against raw text
    weight: float = 1.0
    rationale: str = ""
    origin: str = "seed"  # seed | induced | llm | crossover | manual

    def signature(self) -> tuple:
        """Content identity (ignores id/weight/rationale/origin) for de-duplication."""
        return (
            self.field,
            self.value,
            tuple(sorted(p.lower() for p in self.any_of)),
            tuple(sorted(p.lower() for p in self.all_of)),
            tuple(sorted(p.lower() for p in self.none_of)),
            self.pattern,
        )

    def is_valid(self) -> bool:
        if not (self.any_of or self.all_of or self.pattern):
            return False
        if self.pattern:
            if len(self.pattern) > _MAX_PATTERN_LEN:
                return False
            try:
                re.compile(self.pattern)
            except re.error:
                return False
        return True

    def matches(self, raw_lower: str, norm: str) -> bool:
        """`raw_lower` is the lower-cased raw text, `norm` its ``text.normalize`` form."""
        if self.any_of and not any(phrase_in(p, norm) for p in self.any_of):
            return False
        if self.all_of and not all(phrase_in(p, norm) for p in self.all_of):
            return False
        if self.none_of and any(phrase_in(p, norm) for p in self.none_of):
            return False
        if self.pattern:
            try:
                if not re.search(self.pattern, raw_lower, flags=re.IGNORECASE | re.MULTILINE):
                    return False
            except re.error:
                return False
        return bool(self.any_of or self.all_of or self.pattern)

    def describe(self) -> str:
        cond: list[str] = []
        if self.any_of:
            cond.append("mentions any of " + ", ".join(repr(p) for p in self.any_of))
        if self.all_of:
            cond.append("mentions all of " + ", ".join(repr(p) for p in self.all_of))
        if self.pattern:
            cond.append(f"matches /{self.pattern}/")
        if self.none_of:
            cond.append("does NOT mention " + ", ".join(repr(p) for p in self.none_of))
        why = f" ({self.rationale})" if self.rationale else ""
        return f"[{self.id}] If the item {' and '.join(cond)} -> {self.field} = {self.value} (weight {self.weight:.2f}){why}"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Rule":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        for key in ("any_of", "all_of", "none_of"):
            known[key] = [str(x) for x in known.get(key, []) if str(x).strip()]
        if "weight" in known:
            known["weight"] = float(known["weight"])
        return cls(**known)


def match_rules(rules: list[Rule], text: str) -> list[Rule]:
    raw_lower = text.lower()
    norm = normalize(text)
    return [r for r in rules if r.matches(raw_lower, norm)]


@dataclass
class Genome:
    agent: str
    instructions: str
    rules: list[Rule] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
    exemplars: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    generation: int = 0
    origin: str = "seed"
    notes: str = ""

    @property
    def id(self) -> str:
        """Content hash. Two genomes with the same behaviour-relevant content share an id."""
        content = {
            "agent": self.agent,
            "instructions": self.instructions,
            "rules": sorted(
                [list(r.signature()) + [round(r.weight, 4)] for r in self.rules], key=lambda x: json.dumps(x)
            ),
            "params": {k: round(float(v), 6) for k, v in sorted(self.params.items())},
            "exemplars": sorted(self.exemplars),
        }
        blob = json.dumps(content, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def param(self, name: str, default: float) -> float:
        return float(self.params.get(name, default))

    def clone(self) -> "Genome":
        g = copy.deepcopy(self)
        g.parents = [self.id]
        g.generation = self.generation + 1
        return g

    def add_rule(self, rule: Rule) -> bool:
        """Add a rule unless an equivalent one exists. Returns True if added."""
        if not rule.is_valid():
            return False
        sig = rule.signature()
        if any(r.signature() == sig for r in self.rules):
            return False
        existing = {r.id for r in self.rules}
        base, n = rule.id or f"{rule.field}-{rule.value}", 1
        rid = base
        while rid in existing:
            n += 1
            rid = f"{base}-{n}"
        rule.id = rid
        self.rules.append(rule)
        return True

    def render_playbook(self, max_rules: int = 60) -> str:
        if not self.rules:
            return "(no playbook rules yet)"
        ranked = sorted(self.rules, key=lambda r: -r.weight)[:max_rules]
        return "\n".join("- " + r.describe() for r in ranked)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Genome":
        return cls(
            agent=d["agent"],
            instructions=d.get("instructions", ""),
            rules=[Rule.from_dict(r) for r in d.get("rules", [])],
            params={k: float(v) for k, v in d.get("params", {}).items()},
            exemplars=list(d.get("exemplars", [])),
            parents=list(d.get("parents", [])),
            generation=int(d.get("generation", 0)),
            origin=d.get("origin", "seed"),
            notes=d.get("notes", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))
