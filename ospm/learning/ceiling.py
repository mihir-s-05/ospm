"""Ceiling check: how much could a perfect rubric help the reviewer?

    python -m ospm.learning.ceiling --data data/home-assistant --library data/results/rubric_ha_skills.json \
        --test 192 --budget 1.5 --out data/results/ceiling_ha.json

On the positive test PRs (those where maintainers raised concerns), each non-nit maintainer concern is rewritten
by the LLM as the general rule it applies (no file, line or PR-specific detail), and the reviewer is run with:
  oracle_items      - the mined rubric plus those rules as items of an extra area: the best case for any rubric,
                      the agent still has to notice which items the change violates
  oracle_violations - the mined rubric plus "maintainers found this PR violates these rules": an upper bound for
                      rule guidance of any kind; the agent still has to find where and say it
Rows of `none` / `rubric` / `evolved_rubric` come from an experiment result on the same test split (`--baseline`).
If even `oracle_violations` recall stays low, the bottleneck is the reviewer (or the scoring), not the rubric.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..harness.models import make_chat_model
from ..llm import BudgetExceeded, LLMError, UsageMeter, make_llm
from ..replay.concerns import concern_metrics
from ..replay.semantic import Matcher, semantic_metrics, semantic_score
from ..replay.tasks import ReplayTask, load_tasks
from ..replay.workspace import ReplayRepo
from .experiment import BudgetStop, JsonlCache, make_runner, pick, summarise
from .interventions import LearnConfig
from .postmortem import SkillLearner
from .skills import load as load_skills
from .stats import paired

SYSTEMS = ("oracle_items", "oracle_violations")
RULE_SYSTEM = """You turn a maintainer's code-review comment into the general, reusable review rule it applies,
written as a rubric item a reviewer could check on ANY pull request of this repository. Keep the project's own
helpers, APIs, files and conventions the rule is about; drop everything specific to this pull request: no file
paths, line numbers, variable names of this change, quotes of the comment, or the concrete fix. Return JSON only."""


def _schema() -> dict:
    item = {"type": "object", "properties": {"rule": {"type": "string"}, "check": {"type": "string"}},
            "required": ["rule", "check"], "additionalProperties": False}
    return {"type": "object", "properties": {"rules": {"type": "array", "items": item}}, "required": ["rules"],
            "additionalProperties": False}


def oracle_rules(llm, task: ReplayTask) -> list[dict]:
    """One {rule, check} per non-nit maintainer concern of `task` (cached by the reflect LLM's cache)."""
    gold = [c for c in task.truth["concerns"] if not c["nit"]] or task.truth["concerns"]
    user = "\n\n".join(f"## comment {n}\n{' '.join(c['body'].split())[:1200]}" for n, c in enumerate(gold, 1))
    d = llm.complete_json(RULE_SYSTEM, user + f"\n\nReturn exactly {len(gold)} rules, in order.", _schema(),
                          name="oracle_rule", effort="low")
    return [r for r in d["rules"] if r.get("rule")][: len(gold)]


def oracle_block(rules: list[dict], violated: bool) -> str:
    items = "\n".join(f"{n}. {r['rule']}\n   check: {r['check']}" for n, r in enumerate(rules, 1))
    if violated:
        return ("\n\n## Known problems\nThis repository's maintainers reviewed this pull request and found that it "
                "violates each of the following rules. Find where the change violates each one and comment there."
                f"\n\n{items}")
    return f"\n\n### rubric-this-change\nFurther rules of this repository that apply to this area.\n\n{items}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--library", required=True, help="the mined rubric (a *_skills.json)")
    p.add_argument("--baseline", default=None, help="experiment result on the same test split (none/rubric rows)")
    p.add_argument("--llm", default="openai")
    p.add_argument("--model", default=None)
    p.add_argument("--test", type=int, default=192)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget", type=float, default=1.5)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    data = Path(a.data)
    repo = ReplayRepo(data)
    repo.pool_size = a.workers
    tasks = [t for t in load_tasks(data / "tasks.jsonl") if t.kind == "concern"]
    test = [t for t in pick([t for t in tasks if t.split == "test"], a.test, newest=False) if t.truth["positive"]]
    meter = UsageMeter(a.budget)
    model = make_chat_model(a.llm, a.model, meter=meter)
    reflect = make_llm(a.llm, a.model, meter=meter, cache=JsonlCache(data / "reflect_cache.jsonl"))
    matcher = Matcher(data / "semantic_cache.jsonl")
    run = make_runner(repo, model, meter, data / "learning_cache.jsonl", data / "traces",
                      score_fn=lambda t, out: semantic_score(matcher, t, out))
    server = SkillLearner(repo, run, reflect, LearnConfig(workers=a.workers), log=log, trace_of=run.trace_of)
    server.skills = load_skills(Path(a.library))
    log(f"[ceiling] {len(test)} positive test PRs")
    stopped, jobs, rules = None, [], {}
    try:
        with ThreadPoolExecutor(a.workers) as ex:
            rules = dict(zip((t.id for t in test), ex.map(lambda t: oracle_rules(reflect, t), test)))
        for t in test:
            ctx, skills, _ = server.serve(t, "rubric")
            for s in SYSTEMS:
                jobs.append((t, s, ctx + oracle_block(rules[t.id], s == "oracle_violations"), skills))
        outs = server._run_many([(t, ctx, skills) for t, _, ctx, skills in jobs])
    except (BudgetStop, BudgetExceeded, LLMError) as e:
        stopped, outs = str(e), []
    rows: dict[str, list[dict]] = {s: [] for s in SYSTEMS}
    for (t, s, *_), (score, out) in zip(jobs, outs):
        rows[s].append({"task": t.id, "positive": True, "score": score, **concern_metrics(t, out),
                        **semantic_metrics(matcher, t, out)})
    if a.baseline:
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))["rows"]
        order = [r["task"] for r in rows[SYSTEMS[0]]]  # paired() compares rows by position
        by_task = {s: {r["task"]: r for r in v} for s, v in base.items()}
        rows = {**{s: [v[i] for i in order] for s, v in by_task.items() if all(i in v for i in order)}, **rows}
    result = {"repo": repo.store.meta("repo"), "test_positive_prs": len(rows[SYSTEMS[0]]), "stopped": stopped,
              "llm": meter.snapshot(), "summary": {s: summarise(v) for s, v in rows.items()},
              "paired_vs_none": paired(rows, "none") if "none" in rows else None,
              "rules": rules, "rows": rows}
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    repo.close()
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "rules")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
