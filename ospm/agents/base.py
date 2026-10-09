"""Base class for evolvable agents.

An agent maps an item (issue, CI run, pull request) to labels for one or more *fields*.
Its policy is hybrid:

1. **Heuristic policy** (always available, free, deterministic):
   score(label) = rule_weight * sum(matching playbook rules)
                + knn_weight  * sum(similarity of retrieved past cases carrying that label)
                + base_weight * agent-specific structural signals
2. **LLM policy** (optional): the same playbook, retrieved cases, pinned exemplars and the heuristic
   pre-assessment are put in front of an LLM, which returns schema-constrained JSON.

Both policies are configured entirely by the agent's Genome, so evolution improves either mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable

from ..genome import Genome, Rule, match_rules
from ..llm import LLM, BudgetExceeded, LLMError
from ..models import Decision
from ..project import ProjectContext
from ..text import TfidfIndex, truncate


@dataclass(frozen=True)
class FieldSpec:
    name: str
    labels: tuple[str, ...]
    multi: bool = False
    default: str = ""
    weight: float = 1.0
    description: str = ""


@dataclass
class Case:
    id: str
    text: str
    labels: dict[str, Any]


class CaseBase:
    """Labelled experience available for retrieval (episodic memory)."""

    def __init__(self, cases: Iterable[Case] = ()):
        self.index = TfidfIndex()
        self.by_id: dict[str, Case] = {}
        for c in cases:
            self.add(c)

    def add(self, case: Case) -> None:
        if case.id in self.by_id:
            # replace: rebuild is simplest and keeps the index exact
            self.by_id[case.id] = case
            old = list(self.by_id.values())
            self.index = TfidfIndex()
            for c in old:
                self.index.add(c.id, c.text, c)
            return
        self.by_id[case.id] = case
        self.index.add(case.id, case.text, case)

    def __len__(self) -> int:
        return len(self.by_id)

    def neighbors(self, text: str, k: int, exclude: set[str] | frozenset[str] = frozenset(), min_sim: float = 0.0):
        return [(cid, sim, payload) for cid, sim, payload in self.index.query(text, k, set(exclude), min_sim)]


Scores = dict[str, dict[str, float]]


class Agent:
    name: ClassVar[str] = "base"
    title: ClassVar[str] = "agent"
    fields: ClassVar[tuple[FieldSpec, ...]] = ()
    default_params: ClassVar[dict[str, float]] = {
        "knn_k": 5,
        "knn_weight": 1.0,
        "rule_weight": 1.0,
        "base_weight": 1.0,
        "min_sim": 0.08,
        "multi_threshold": 0.9,
        "llm_neighbors": 4,
    }
    seed_instructions: ClassVar[str] = ""
    seed_rules: ClassVar[tuple[dict, ...]] = ()
    # labels that are relational rather than content-based; retrieved neighbours never vote for them
    knn_ignore: ClassVar[dict[str, frozenset[str]]] = {}

    def __init__(
        self,
        genome: Genome | None = None,
        *,
        llm: LLM | None = None,
        cases: CaseBase | None = None,
        project: ProjectContext | None = None,
        effort: str | None = "medium",
    ):
        self.project = project or ProjectContext()
        self.genome = genome or self.seed_genome(self.project)
        if self.genome.agent != self.name:
            raise ValueError(f"genome for {self.genome.agent!r} given to agent {self.name!r}")
        self.llm = llm
        self.cases = cases or CaseBase()
        self.effort = effort

    # ------------------------------------------------------------------ genome
    @classmethod
    def seed_genome(cls, project: ProjectContext | None = None) -> Genome:
        g = Genome(agent=cls.name, instructions=cls.seed_instructions.strip(), params=dict(cls.default_params))
        for spec in cls.seed_rules:
            g.add_rule(Rule.from_dict({**spec, "origin": "seed"}))
        for r in cls.project_rules(project or ProjectContext()):
            g.add_rule(r)
        return g

    @classmethod
    def project_rules(cls, project: ProjectContext) -> list[Rule]:
        """Rules derived from the project context (e.g. out-of-scope keywords). Override as needed."""
        return []

    @classmethod
    def field(cls, name: str) -> FieldSpec:
        for f in cls.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def p(self, name: str) -> float:
        return self.genome.param(name, float(self.default_params.get(name, 0.0)))

    # ------------------------------------------------------------------ subclass hooks
    def item_id(self, item: Any) -> str:
        return str(getattr(item, "id", id(item)))

    def item_text(self, item: Any) -> str:
        """Text used for rule matching and retrieval."""
        raise NotImplementedError

    def render_item(self, item: Any) -> str:
        """What the LLM sees about the item."""
        return truncate(self.item_text(item), 8000)

    def base_scores(self, item: Any, text: str) -> Scores:
        return {}

    def extras(self, item: Any, text: str, labels: dict[str, Any], fired: list[Rule]) -> dict[str, Any]:
        """Agent-specific outputs computed without an LLM (also used to backfill LLM output)."""
        return {}

    def extra_schema(self) -> dict[str, dict]:
        """Additional (unscored) properties requested from the LLM."""
        return {}

    # ------------------------------------------------------------------ heuristic policy
    def score(self, item: Any, text: str, exclude: frozenset[str] = frozenset()):
        scores: Scores = {f.name: {lab: 0.0 for lab in f.labels} for f in self.fields}
        fired = match_rules(self.genome.rules, text)
        rw = self.p("rule_weight")
        for r in fired:
            if r.field in scores and r.value in scores[r.field]:
                scores[r.field][r.value] += rw * r.weight

        neighbors = self.cases.neighbors(text, int(self.p("knn_k")), exclude, self.p("min_sim"))
        kw = self.p("knn_weight")
        for _cid, sim, case in neighbors:
            for f in self.fields:
                val = case.labels.get(f.name)
                vals = val if isinstance(val, list) else [val] if val is not None else []
                ignored = self.knn_ignore.get(f.name, frozenset())
                for v in vals:
                    if v in scores[f.name] and v not in ignored:
                        scores[f.name][v] += kw * sim

        bw = self.p("base_weight")
        for fname, lab_scores in self.base_scores(item, text).items():
            for lab, s in lab_scores.items():
                if fname in scores and lab in scores[fname]:
                    scores[fname][lab] += bw * s
        return scores, fired, neighbors

    def labels_from_scores(self, scores: Scores) -> tuple[dict[str, Any], dict[str, float]]:
        labels: dict[str, Any] = {}
        conf: dict[str, float] = {}
        thr = self.p("multi_threshold")
        for f in self.fields:
            s = scores[f.name]
            if f.multi:
                chosen = sorted(lab for lab, v in s.items() if v >= thr)
                labels[f.name] = chosen
                conf[f.name] = round(min(1.0, min((s[c] for c in chosen), default=0.0) / (2 * thr)) if chosen else 0.0, 3)
            else:
                total = sum(v for v in s.values() if v > 0)
                if total <= 0:
                    labels[f.name] = f.default or f.labels[0]
                    conf[f.name] = 0.0
                else:
                    # deterministic tie-break: field label order
                    best = max(f.labels, key=lambda lab: (s[lab], -f.labels.index(lab)))
                    labels[f.name] = best
                    conf[f.name] = round(s[best] / total, 3)
        return labels, conf

    # ------------------------------------------------------------------ LLM policy
    def output_schema(self) -> dict:
        props: dict[str, dict] = {}
        for f in self.fields:
            enum = {"type": "string", "enum": list(f.labels)}
            props[f.name] = {"type": "array", "items": enum} if f.multi else enum
        props.update(self.extra_schema())
        props["rationale"] = {"type": "string"}
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def system_prompt(self) -> str:
        field_docs = []
        for f in self.fields:
            kind = "zero or more of" if f.multi else "exactly one of"
            field_docs.append(f"- `{f.name}`: {kind} {', '.join(f.labels)}. {f.description}".rstrip())
        return "\n\n".join(
            [
                f"You are the {self.title} of an AI project-management team for the open-source project "
                f"'{self.project.name}'. You make the decision a careful, experienced maintainer would make.",
                self.project.render(),
                "## Your operating policy (evolves over time)\n" + (self.genome.instructions or "(none)"),
                "## Playbook\nRules learned from past mistakes and maintainer feedback. They are strong evidence, "
                "not absolute: override a rule when the item clearly contradicts it.\n" + self.genome.render_playbook(),
                "## Output fields\n" + "\n".join(field_docs)
                + "\n- `rationale`: one or two sentences justifying the decision.",
            ]
        )

    def user_prompt(self, item: Any, scores: Scores, fired: list[Rule], neighbors, exclude: frozenset[str]) -> str:
        parts: list[str] = []
        exemplars = [self.cases.by_id[e] for e in self.genome.exemplars if e in self.cases.by_id and e not in exclude]
        if exemplars:
            parts.append("## Reference examples (solved)")
            for c in exemplars:
                parts.append(f"<example labels='{self._shown_labels(c.labels)}'>\n{truncate(c.text, 800)}\n</example>")
        top = neighbors[: int(self.p("llm_neighbors"))]
        if top:
            parts.append("## Similar past cases from memory (with the decision that turned out correct)")
            for _cid, sim, c in top:
                parts.append(
                    f"<case similarity='{sim:.2f}' labels='{self._shown_labels(c.labels)}'>\n{truncate(c.text, 600)}\n</case>"
                )
        hint_lines = []
        for f in self.fields:
            ranked = sorted(scores[f.name].items(), key=lambda kv: -kv[1])
            shown = ", ".join(f"{lab}={v:.2f}" for lab, v in ranked if v > 0) or "no signal"
            hint_lines.append(f"- {f.name}: {shown}")
        if fired:
            hint_lines.append("- playbook rules that fired: " + ", ".join(r.id for r in fired))
        parts.append("## Heuristic pre-assessment (cheap signals; may be wrong)\n" + "\n".join(hint_lines))
        parts.append("## Item to decide\n" + self.render_item(item))
        return "\n\n".join(parts)

    def _shown_labels(self, labels: dict[str, Any]) -> str:
        """Labels of a past case as shown to the LLM. Relational labels (e.g. "duplicate") describe that case's
        relation to *another* issue, so they are withheld: a near-identical neighbour that was a duplicate of
        the current item would otherwise teach the LLM to mark the original as a duplicate."""
        shown: dict[str, Any] = {}
        names = {f.name for f in self.fields}
        for k, v in labels.items():
            if k not in names:
                continue
            ignored = self.knn_ignore.get(k, frozenset())
            if isinstance(v, list):  # multi-label field
                shown[k] = [x for x in v if x not in ignored]
            elif v is not None and v not in ignored:
                shown[k] = v
        return json.dumps(shown, sort_keys=True)

    # ------------------------------------------------------------------ main entry point
    def decide(self, item: Any, exclude: Iterable[str] = ()) -> Decision:
        exclude = frozenset(exclude)
        text = self.item_text(item)
        scores, fired, neighbors = self.score(item, text, exclude)
        labels, conf = self.labels_from_scores(scores)
        decision = Decision(
            agent=self.name,
            item_id=self.item_id(item),
            labels=labels,
            confidence=conf,
            fired_rules=[r.id for r in fired],
            neighbors=[(cid, round(sim, 3)) for cid, sim, _ in neighbors],
            genome_id=self.genome.id,
        )
        heuristic_extras = self.extras(item, text, labels, fired)
        decision.extras = dict(heuristic_extras)
        decision.rationale = self._heuristic_rationale(fired, neighbors)

        if self.llm is not None:
            try:
                data = self.llm.complete_json(
                    self.system_prompt(),
                    self.user_prompt(item, scores, fired, neighbors, exclude),
                    self.output_schema(),
                    name=f"{self.name}_decision",
                    effort=self.effort,
                )
            except (LLMError, BudgetExceeded) as e:
                decision.extras["llm_error"] = str(e)
            else:
                decision.used_llm = True
                for f in self.fields:
                    decision.labels[f.name] = sorted(set(data[f.name])) if f.multi else data[f.name]
                    decision.confidence[f.name] = 1.0 if decision.labels[f.name] == labels[f.name] else 0.6
                decision.rationale = data.get("rationale", "")
                llm_extras = {k: data[k] for k in self.extra_schema() if k in data}
                decision.extras = self.merge_llm_extras(item, text, decision.labels, fired, heuristic_extras, llm_extras)
        return decision

    def merge_llm_extras(self, item, text, labels, fired, heuristic_extras, llm_extras) -> dict[str, Any]:
        """Combine LLM free-text outputs with heuristic extras recomputed for the final labels."""
        merged = dict(self.extras(item, text, labels, fired))
        for k, v in llm_extras.items():
            if v not in ("", [], None):
                merged[k] = v
        return merged

    @staticmethod
    def _heuristic_rationale(fired: list[Rule], neighbors) -> str:
        bits = []
        if fired:
            bits.append("rules " + ", ".join(r.id for r in fired[:5]))
        if neighbors:
            bits.append("similar to " + ", ".join(f"{cid} ({sim:.2f})" for cid, sim, _ in neighbors[:3]))
        return "Heuristic decision from " + ("; ".join(bits) if bits else "priors only") + "."
