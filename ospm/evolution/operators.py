"""Mutation operators for non-parametric evolution.

GEPA (Agrawal et al., 2025, "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning")
mutates a system by *reflecting* on execution traces plus natural-language feedback. Here a genome has four
modules and each has a reflective operator:

* ``instructions`` – LLM reflection rewrites the natural-language policy (GEPA's core operator);
* ``playbook``     – rule induction: an LLM proposes rules, or (offline) a statistical analyst induces
                     discriminative phrase rules from failures and vetoes/demotes rules that misfired;
* ``params``       – bounded jitter of retrieval/score knobs;
* ``exemplars``    – re-picks pinned few-shot cases to cover the labels the agent gets wrong.

``merge`` is GEPA's system-aware crossover: for each module take the version that evolved away from the
common ancestor.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import Agent
from ..genome import Genome, Rule
from ..llm import LLM, BudgetExceeded, LLMError
from ..models import Decision
from ..text import content_tokens, ngrams, normalize, phrase_in, truncate
from .metrics import field_instance_score

PARAM_BOUNDS: dict[str, tuple[float, float, bool]] = {  # name: (low, high, integer)
    "knn_k": (0, 15, True),
    "knn_weight": (0.0, 3.0, False),
    "rule_weight": (0.2, 3.0, False),
    "base_weight": (0.0, 3.0, False),
    "min_sim": (0.0, 0.6, False),
    "multi_threshold": (0.3, 3.0, False),
    "llm_neighbors": (0, 8, True),
    "dup_threshold": (0.2, 0.9, False),
}
MAX_RULES = 80
MAX_INSTRUCTIONS_CHARS = 6000
# tokens that carry no transferable signal for induced rules (URL fragments, filler)
INDUCTION_STOPLIST = frozenset("http https www com org net io html see also use via etc file line".split())


@dataclass
class Trace:
    """One execution of the agent on a training example, with feedback (GEPA's mu_f)."""

    example_id: str
    text: str
    gold: dict[str, Any]
    pred: dict[str, Any]
    score: float
    fired: list[str]
    rationale: str
    feedback: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)


def make_trace(agent: Agent, example, decision: Decision, score: float) -> Trace:
    text = agent.item_text(example.item)
    scores, _, _ = agent.score(example.item, text, frozenset({example.id}))
    notes: list[str] = []
    for spec in agent.fields:
        gold = example.labels.get(spec.name)
        if gold is None:
            continue
        pred = decision.labels.get(spec.name)
        if field_instance_score(spec, pred, gold) < 1.0:
            if spec.multi:
                missed = sorted(set(gold) - set(pred or []))
                extra = sorted(set(pred or []) - set(gold))
                if missed:
                    notes.append(f"{spec.name}: missed {missed}")
                if extra:
                    notes.append(f"{spec.name}: false alarm {extra}")
            else:
                notes.append(f"{spec.name}: predicted {pred!r} but the correct answer is {gold!r}")
    fired = [r for r in agent.genome.rules if r.id in set(decision.fired_rules)]
    if fired:
        notes.append("rules fired: " + "; ".join(f"{r.id}->{r.field}={r.value}" for r in fired))
    feedback = "Correct." if not notes else " | ".join(notes)
    return Trace(example.id, text, dict(example.labels), dict(decision.labels), score, decision.fired_rules,
                 decision.rationale, feedback, scores)


# ---------------------------------------------------------------------------------------------- playbook
class RuleInducer:
    """Offline reflective mutation of the playbook: induce rules from failures, veto/demote misfiring rules.

    Statistics come from the *training* examples only (never val/test).
    """

    def __init__(self, agent_cls: type[Agent], train_texts: dict[str, str], train_labels: dict[str, dict],
                 min_pos: int = 3, min_precision: float = 0.75, min_lift: float = 1.5, max_df: float = 0.6):
        self.agent_cls = agent_cls
        self.labels = train_labels
        self.norm = {i: normalize(t) for i, t in train_texts.items()}
        self.raw = {i: t.lower() for i, t in train_texts.items()}
        self.min_pos = min_pos
        self.min_precision = min_precision
        self.min_lift = min_lift
        n = max(1, len(train_texts))
        df: dict[str, int] = {}
        for t in train_texts.values():
            for g in set(ngrams(content_tokens(t), 2)):
                df[g] = df.get(g, 0) + 1
        self.common = {g for g, c in df.items() if c / n > max_df}

    def _has(self, ex_id: str, fname: str, value: str) -> bool:
        v = self.labels[ex_id].get(fname)
        return value in v if isinstance(v, list) else v == value

    def _applicable(self, ex_id: str, fname: str) -> bool:
        return self.labels[ex_id].get(fname) is not None

    def _candidates(self, text: str) -> list[str]:
        seen: list[str] = []
        for g in ngrams(content_tokens(text), 2):
            if g in self.common or g in seen or len(g) < 3:
                continue
            if any(tok in INDUCTION_STOPLIST for tok in g.split()):
                continue
            seen.append(g)
        return seen

    def rule_precision(self, rule: Rule) -> tuple[int, int]:
        """(times fired, times correct) over the training examples."""
        fired = correct = 0
        for i in self.norm:
            if not self._applicable(i, rule.field):
                continue
            if rule.matches(self.raw[i], self.norm[i]):
                fired += 1
                correct += self._has(i, rule.field, rule.value)
        return fired, correct

    def base_rate(self, fname: str, value: str) -> float:
        applicable = [i for i in self.labels if self._applicable(i, fname)]
        if not applicable:
            return 1.0
        return sum(self._has(i, fname, value) for i in applicable) / len(applicable)

    def induce_positive(self, trace: Trace, fname: str, value: str, margin: float) -> Rule | None:
        """Best phrase from the failing item that is frequent, precise and *specific* (lift over base rate)."""
        base = max(self.base_rate(fname, value), 1e-6)
        best: tuple[float, str, int, int] | None = None
        for cand in self._candidates(trace.text):
            support = pos = 0
            for i, norm in self.norm.items():
                if not self._applicable(i, fname) or not phrase_in(cand, norm):
                    continue
                support += 1
                pos += self._has(i, fname, value)
            if pos < self.min_pos:
                continue
            precision = (pos + 0.5) / (support + 1.0)
            if precision < self.min_precision or precision / base < self.min_lift:
                continue
            score = precision * math.log1p(pos) - 0.01 * len(cand.split())
            if best is None or score > best[0] or (score == best[0] and cand < best[1]):
                best = (score, cand, pos, support)
        if best is None:
            return None
        _, cand, pos, support = best
        weight = round(min(2.5, max(0.4, margin + 0.25)), 3)
        return Rule(id=f"ind-{fname}-{value}-{cand.replace(' ', '_')}"[:60], field=fname, value=value, any_of=[cand],
                    weight=weight, rationale=f"induced: {pos}/{support} training items with this phrase are {value}",
                    origin="induced")

    def induce_veto(self, rule: Rule, trace: Trace) -> Rule | None:
        """Add a none_of phrase that blocks `rule` on items like this false positive."""
        fires = [i for i in self.norm if self._applicable(i, rule.field) and rule.matches(self.raw[i], self.norm[i])]
        good = [i for i in fires if self._has(i, rule.field, rule.value)]
        best: tuple[int, str] | None = None
        for cand in self._candidates(trace.text):
            if any(phrase_in(cand, self.norm[i]) for i in good):
                continue  # would veto a correct firing
            blocked_bad = sum(1 for i in fires if i not in good and phrase_in(cand, self.norm[i]))
            if blocked_bad == 0:
                continue
            if best is None or blocked_bad > best[0] or (blocked_bad == best[0] and len(cand) < len(best[1])):
                best = (blocked_bad, cand)
        if best is None:
            return None
        new = Rule.from_dict({**rule.__dict__})
        new.none_of = list(rule.none_of) + [best[1]]
        new.origin = "induced"
        new.rationale = (rule.rationale + "; " if rule.rationale else "") + f"vetoed by {best[1]!r}"
        return new

    def mutate(self, genome: Genome, traces: list[Trace], rng: random.Random, max_changes: int = 3) -> Genome | None:
        child = genome.clone()
        child.origin = "playbook-induction"
        changes: list[str] = []
        failures = [t for t in traces if t.score < 1.0]
        rng.shuffle(failures)
        for t in failures:
            if len(changes) >= max_changes:
                break
            for spec in self.agent_cls.fields:
                gold = t.gold.get(spec.name)
                if gold is None:
                    continue
                pred = t.pred.get(spec.name)
                fs = t.scores.get(spec.name, {})
                if spec.multi:
                    missed = sorted(set(gold) - set(pred or []))
                    extra = sorted(set(pred or []) - set(gold))
                    thr = genome.param("multi_threshold", 0.9)
                    for lab in missed:
                        r = self.induce_positive(t, spec.name, lab, thr - fs.get(lab, 0.0))
                        if r and child.add_rule(r):
                            changes.append(f"+rule {r.id}")
                    for lab in extra:
                        changes += self._repair(child, spec.name, lab, t)
                elif pred != gold:
                    margin = fs.get(pred, 0.0) - fs.get(gold, 0.0)
                    r = self.induce_positive(t, spec.name, gold, margin)
                    if r and child.add_rule(r):
                        changes.append(f"+rule {r.id}")
                    changes += self._repair(child, spec.name, pred, t)
        if not changes:
            return None
        if len(child.rules) > MAX_RULES:
            child.rules = sorted(child.rules, key=lambda r: -r.weight)[:MAX_RULES]
        child.notes = "; ".join(changes)
        return child

    def _repair(self, genome: Genome, fname: str, wrong_value: str, trace: Trace) -> list[str]:
        """Rules that pushed toward `wrong_value` on this item: veto them, or demote/remove imprecise ones."""
        changes: list[str] = []
        raw, norm = trace.text.lower(), normalize(trace.text)
        for rule in list(genome.rules):
            if rule.field != fname or rule.value != wrong_value or not rule.matches(raw, norm):
                continue
            vetoed = self.induce_veto(rule, trace)
            if vetoed is not None:
                pos = next(i for i, r in enumerate(genome.rules) if r is rule)
                genome.rules[pos] = vetoed
                changes.append(f"veto {rule.id} with {vetoed.none_of[-1]!r}")
                continue
            fired, correct = self.rule_precision(rule)
            precision = (correct + 0.5) / (fired + 1.0)
            if precision < 0.35:
                genome.rules.remove(rule)
                changes.append(f"-rule {rule.id} (precision {precision:.2f})")
            elif precision < 0.75:
                rule.weight = round(max(0.1, rule.weight * 0.7), 3)
                changes.append(f"demote {rule.id} -> {rule.weight}")
        return changes


# ---------------------------------------------------------------------------------------------- params
def mutate_params(genome: Genome, rng: random.Random, n_changes: int = 2) -> Genome | None:
    keys = [k for k in genome.params if k in PARAM_BOUNDS]
    if not keys:
        return None
    child = genome.clone()
    child.origin = "param-jitter"
    notes = []
    for k in rng.sample(keys, min(n_changes, len(keys))):
        lo, hi, is_int = PARAM_BOUNDS[k]
        cur = float(child.params[k])
        if is_int:
            new = cur + rng.choice([-2, -1, 1, 2])
        else:
            new = cur * rng.choice([0.5, 0.7, 1.4, 2.0]) if cur > 0 else rng.uniform(lo, hi) / 2
        new = min(hi, max(lo, new))
        new = float(round(new)) if is_int else round(new, 4)
        if new != cur:
            child.params[k] = new
            notes.append(f"{k}: {cur:g} -> {new:g}")
    if not notes:
        return None
    child.notes = "; ".join(notes)
    return child


# ---------------------------------------------------------------------------------------------- exemplars
def mutate_exemplars(genome: Genome, traces: list[Trace], train_labels: dict[str, dict], rng: random.Random,
                     k: int = 3) -> Genome | None:
    """Pin training cases whose gold labels match what the agent got wrong (helps the LLM policy)."""
    targets: list[tuple[str, Any]] = []
    for t in traces:
        if t.score >= 1.0:
            continue
        for fname, gold in t.gold.items():
            if gold is not None and t.pred.get(fname) != gold:
                targets.append((fname, gold))
    if not targets:
        return None
    chosen: list[str] = []
    failing = {t.example_id for t in traces if t.score < 1.0}
    for fname, gold in targets:
        pool = [i for i, lab in train_labels.items() if lab.get(fname) == gold and i not in failing and i not in chosen]
        if pool:
            chosen.append(rng.choice(sorted(pool)))
        if len(chosen) >= k:
            break
    if not chosen or sorted(chosen) == sorted(genome.exemplars):
        return None
    child = genome.clone()
    child.origin = "exemplar-selection"
    child.exemplars = chosen
    child.notes = "exemplars: " + ", ".join(chosen)
    return child


# ---------------------------------------------------------------------------------------------- LLM reflection
REFLECTION_SYSTEM = """You are the meta-optimizer of a self-improving AI project-management agent.
The agent's behaviour is fully determined by a natural-language operating policy plus a playbook of
structured rules; its model weights never change. You are shown the current policy and playbook, followed by
execution traces on training cases with the correct answers and feedback.

Your job (as in GEPA reflective prompt evolution):
1. Diagnose the systematic causes of the failures. Look for patterns, not individual cases.
2. Rewrite the operating policy so the agent gets these cases right *and generalises* to unseen cases:
   keep what works, add precise decision criteria and edge-case guidance, remove misleading guidance.
   Do not reference specific case ids or memorise individual items.
3. Optionally propose playbook rule changes. A rule fires when the item text contains any of `any_of`
   (token phrases; punctuation is ignored, matching is case-insensitive), all of `all_of`, none of `none_of`,
   and matches the optional regex `pattern` (Python syntax, case-insensitive, use "" for none).
   Rules add `weight` (0.1-3.0) of evidence for field=value. Prefer few, high-precision rules.
   Remove or down-weight rules that cause false positives.
Return JSON only."""


def _reflection_schema(agent_cls: type[Agent]) -> dict:
    fields = [f.name for f in agent_cls.fields]
    values = sorted({lab for f in agent_cls.fields for lab in f.labels})
    rule = {
        "type": "object",
        "properties": {
            "field": {"type": "string", "enum": fields},
            "value": {"type": "string", "enum": values},
            "any_of": {"type": "array", "items": {"type": "string"}},
            "all_of": {"type": "array", "items": {"type": "string"}},
            "none_of": {"type": "array", "items": {"type": "string"}},
            "pattern": {"type": "string"},
            "weight": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["field", "value", "any_of", "all_of", "none_of", "pattern", "weight", "rationale"],
        "additionalProperties": False,
    }
    reweight = {
        "type": "object",
        "properties": {"id": {"type": "string"}, "weight": {"type": "number"}},
        "required": ["id", "weight"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string"},
            "instructions": {"type": "string"},
            "add_rules": {"type": "array", "items": rule},
            "remove_rule_ids": {"type": "array", "items": {"type": "string"}},
            "reweight": {"type": "array", "items": reweight},
        },
        "required": ["diagnosis", "instructions", "add_rules", "remove_rule_ids", "reweight"],
        "additionalProperties": False,
    }


class LLMReflector:
    def __init__(self, llm: LLM, agent_cls: type[Agent], effort: str | None = "high", max_traces: int = 10):
        self.llm = llm
        self.agent_cls = agent_cls
        self.effort = effort
        self.max_traces = max_traces
        self.last_error: str | None = None

    def build_prompt(self, genome: Genome, traces: list[Trace]) -> str:
        fail = [t for t in traces if t.score < 1.0]
        ok = [t for t in traces if t.score >= 1.0]
        shown = fail[: self.max_traces] + ok[: max(0, min(3, self.max_traces - len(fail)))]
        label_space = "\n".join(
            f"- {f.name} ({'multi-label' if f.multi else 'single label'}): {', '.join(f.labels)}. {f.description}"
            for f in self.agent_cls.fields
        )
        rules = json.dumps(
            [{"id": r.id, "field": r.field, "value": r.value, "any_of": r.any_of, "all_of": r.all_of,
              "none_of": r.none_of, "pattern": r.pattern, "weight": r.weight} for r in genome.rules],
            indent=1,
        )
        blocks = []
        for n, t in enumerate(shown, 1):
            blocks.append(
                f"### Trace {n} (score {t.score:.2f})\nInput:\n{truncate(t.text, 1500)}\n"
                f"Agent output: {json.dumps(t.pred, sort_keys=True)}\nAgent rationale: {t.rationale}\n"
                f"Correct answer: {json.dumps(t.gold, sort_keys=True)}\nFeedback: {t.feedback}"
            )
        return (
            f"## Agent role\n{self.agent_cls.title}\n\n## Output label space\n{label_space}\n\n"
            f"## Current operating policy\n{genome.instructions}\n\n## Current playbook rules\n{rules}\n\n"
            f"## Execution traces\n" + "\n\n".join(blocks)
        )

    def mutate(self, genome: Genome, traces: list[Trace]) -> Genome | None:
        if not any(t.score < 1.0 for t in traces):
            return None
        self.last_error = None
        try:
            data = self.llm.complete_json(REFLECTION_SYSTEM, self.build_prompt(genome, traces),
                                          _reflection_schema(self.agent_cls), name="reflection", effort=self.effort)
        except (LLMError, BudgetExceeded) as e:
            self.last_error = str(e)
            return None
        child = genome.clone()
        child.origin = "llm-reflection"
        instr = (data.get("instructions") or "").strip()
        if instr:
            child.instructions = instr[:MAX_INSTRUCTIONS_CHARS]
        remove = set(data.get("remove_rule_ids") or [])
        child.rules = [r for r in child.rules if r.id not in remove]
        weights = {w["id"]: w["weight"] for w in data.get("reweight") or []}
        for r in child.rules:
            if r.id in weights:
                r.weight = round(min(3.0, max(0.1, float(weights[r.id]))), 3)
        added = 0
        for spec in data.get("add_rules") or []:
            try:
                field_spec = self.agent_cls.field(spec["field"])
            except KeyError:
                continue
            if spec["value"] not in field_spec.labels:
                continue
            rule = Rule.from_dict({**spec, "id": f"llm-{spec['field']}-{spec['value']}",
                                   "weight": min(3.0, max(0.1, float(spec["weight"]))), "origin": "llm"})
            if child.add_rule(rule):
                added += 1
        if len(child.rules) > MAX_RULES:
            child.rules = sorted(child.rules, key=lambda r: -r.weight)[:MAX_RULES]
        child.notes = f"reflection: {truncate(data.get('diagnosis', ''), 400)} (+{added} rules, -{len(remove)} rules)"
        return child


# ---------------------------------------------------------------------------------------------- merge
MODULES = ("instructions", "playbook", "params", "exemplars")


def _module(genome: Genome, name: str):
    if name == "instructions":
        return genome.instructions
    if name == "playbook":
        return sorted((r.signature(), round(r.weight, 4)) for r in genome.rules)
    if name == "params":
        return sorted(genome.params.items())
    return sorted(genome.exemplars)


def merge(a: Genome, b: Genome, ancestor: Genome, a_better: bool) -> Genome | None:
    """GEPA system-aware merge: per module, take the side that changed relative to the common ancestor;
    if both changed, playbooks are unioned and other modules come from the stronger parent."""
    child = (a if a_better else b).clone()
    child.parents = [a.id, b.id]
    child.generation = max(a.generation, b.generation) + 1
    child.origin = "merge"
    strong, weak = (a, b) if a_better else (b, a)
    picks = []
    for m in MODULES:
        ma, mb, m0 = _module(a, m), _module(b, m), _module(ancestor, m)
        if ma == mb:
            src, chosen = "both", a
        elif ma == m0:
            src, chosen = "b", b
        elif mb == m0:
            src, chosen = "a", a
        else:
            src, chosen = "both-changed", strong
        if m == "instructions":
            child.instructions = chosen.instructions
        elif m == "params":
            child.params = dict(chosen.params)
        elif m == "exemplars":
            child.exemplars = list(chosen.exemplars)
        elif src == "both-changed":
            # playbooks are sets of lessons: keep the stronger parent's rules and add the other's new ones
            src = "union"
            child.rules = [Rule.from_dict(dict(r.__dict__)) for r in strong.rules]
            for r in weak.rules:
                child.add_rule(Rule.from_dict(dict(r.__dict__)))
        else:
            child.rules = [Rule.from_dict(dict(r.__dict__)) for r in chosen.rules]
        picks.append(f"{m}<-{src}")
    if child.id in (a.id, b.id):
        return None
    child.notes = "merge: " + ", ".join(picks)
    return child
