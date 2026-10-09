"""Scoring: per-instance scores (used by GEPA's Pareto frontier) and aggregate field metrics (reported)."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..agents.base import Agent, FieldSpec
from ..models import Decision


def field_instance_score(spec: FieldSpec, pred: Any, gold: Any) -> float:
    if spec.multi:
        p, g = set(pred or []), set(gold or [])
        if not p and not g:
            return 1.0
        return 2 * len(p & g) / (len(p) + len(g))
    return 1.0 if pred == gold else 0.0


def instance_score(fields: Iterable[FieldSpec], pred: dict[str, Any], gold: dict[str, Any]) -> float:
    num = den = 0.0
    for spec in fields:
        if spec.name not in gold or gold[spec.name] is None:
            continue
        num += spec.weight * field_instance_score(spec, pred.get(spec.name), gold[spec.name])
        den += spec.weight
    return num / den if den else 1.0


def single_label_metrics(labels: tuple[str, ...], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    acc = sum(p == g for p, g in pairs) / len(pairs) if pairs else 0.0
    per: dict[str, dict[str, float]] = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in pairs if p == lab and g == lab)
        fp = sum(1 for p, g in pairs if p == lab and g != lab)
        fn = sum(1 for p, g in pairs if p != lab and g == lab)
        if tp + fp + fn == 0:
            continue  # label absent from both gold and predictions
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3), "support": tp + fn}
        f1s.append(f1)
    confusion = Counter(f"{g}->{p}" for p, g in pairs if p != g)
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "per_label": per, "top_confusions": confusion.most_common(5)}


def multi_label_metrics(pairs: list[tuple[list[str], list[str]]]) -> dict[str, Any]:
    tp = fp = fn = 0
    exact = 0
    for p, g in pairs:
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
        exact += ps == gs
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"micro_precision": round(prec, 4), "micro_recall": round(rec, 4), "micro_f1": round(f1, 4),
            "exact_match": round(exact / len(pairs), 4) if pairs else 0.0}


@dataclass
class EvalResult:
    score: float
    per_instance: dict[str, float]
    predictions: dict[str, Decision] = field(default_factory=dict)
    field_metrics: dict[str, Any] = field(default_factory=dict)
    llm_errors: int = 0

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"score": round(self.score, 4), "n": len(self.per_instance), "llm_errors": self.llm_errors}
        for name, m in self.field_metrics.items():
            out[name] = {k: v for k, v in m.items() if k not in ("per_label", "top_confusions")}
        return out


def evaluate(agent: Agent, examples: list, workers: int = 1,
             cache: dict[tuple[str, str], Decision] | None = None) -> EvalResult:
    """Run `agent` on labelled examples (leave-one-out w.r.t. its case memory) and score it.

    `cache` maps (genome_id, example_id) -> Decision so a genome is never re-run on the same example.
    """
    gid = agent.genome.id

    def run(ex) -> Decision:
        key = (gid, ex.id)
        if cache is not None and key in cache:
            return cache[key]
        dec = agent.decide(ex.item, exclude={ex.id})
        if cache is not None and "llm_error" not in dec.extras:  # never memoise a degraded fallback decision
            cache[key] = dec
        return dec

    if workers > 1 and agent.llm is not None and len(examples) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            decisions = list(pool.map(run, examples))
    else:
        decisions = [run(ex) for ex in examples]

    per_instance: dict[str, float] = {}
    preds: dict[str, Decision] = {}
    errors = 0
    for ex, dec in zip(examples, decisions):
        preds[ex.id] = dec
        per_instance[ex.id] = instance_score(agent.fields, dec.labels, ex.labels)
        errors += "llm_error" in dec.extras

    metrics: dict[str, Any] = {}
    for spec in agent.fields:
        pairs = [(preds[ex.id].labels[spec.name], ex.labels[spec.name]) for ex in examples
                 if ex.labels.get(spec.name) is not None]
        metrics[spec.name] = multi_label_metrics(pairs) if spec.multi else single_label_metrics(spec.labels, pairs)
    score = sum(per_instance.values()) / len(per_instance) if per_instance else 0.0
    return EvalResult(score, per_instance, preds, metrics, errors)
