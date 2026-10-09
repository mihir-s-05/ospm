"""Re-score a finished experiment's test outputs with semantic concern matching (no new episodes).

    python -m ospm.learning.rescore --data data/sglang --result data/results/learning_sglang_skills.json

Each system's extra context is rebuilt exactly as in the run (memory from the same training concerns, the saved
interventions and skill library), so the cached episode outputs are found under the same keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ..replay.semantic import Matcher, semantic_metrics
from ..replay.tasks import load_tasks
from ..replay.workspace import ReplayRepo
from ..team.roles import seed_team
from .experiment import EPISODE_VERSION
from .interventions import InterventionLearner, LearnConfig
from .interventions import load as load_interventions
from .postmortem import SkillLearner
from .skills import load as load_skills
from .skills import skills_hash


def episode_key(task_id: str, who: str, ctx: str, skills: tuple, role_ctx: tuple) -> str:
    """Must match experiment.make_runner's key."""
    key = f"{task_id}|{who}|{EPISODE_VERSION}|{hashlib.sha256(ctx.encode()).hexdigest()[:12]}"
    if skills:
        key += f"|{skills_hash(list(skills))}"
    if role_ctx:
        key += "|r" + hashlib.sha256(json.dumps(sorted(role_ctx)).encode()).hexdigest()[:12]
    return key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--entry", default=None, help="review | orchestrator (default: from the result, else review)")
    p.add_argument("--who", default=None, help="genome token of the cached episodes (default: current genome)")
    a = p.parse_args(argv)
    data, res_path = Path(a.data), Path(a.result)
    result = json.loads(res_path.read_text(encoding="utf-8"))
    systems = result.get("systems") or ["none", "memory", "interventions"]
    entry = a.entry or result.get("entry", "review")
    genome = seed_team()
    who = a.who or (f"{entry}:{genome.roles[entry].content_hash()}" if entry != "orchestrator" else genome.id)
    repo = ReplayRepo(data)
    tasks = {t.id: t for t in load_tasks(data / "tasks.jsonl")}
    train = [tasks[i] for i in result["train_tasks"]]
    learner = InterventionLearner(repo, None, None, LearnConfig(), log=lambda m: None)
    learner.remember(train)
    iv_path = res_path.with_name(res_path.stem + "_interventions.json")
    if "interventions" in systems and iv_path.exists():
        learner.interventions = load_interventions(iv_path)
    skiller = SkillLearner(repo, None, None, LearnConfig(), log=lambda m: None, trace_of=lambda *x: None, entry=entry)
    sk_path = res_path.with_name(res_path.stem + "_skills.json")
    if "skills" in systems and sk_path.exists():
        skiller.skills = load_skills(sk_path)
    cache: dict[str, Any] = {}
    for line in (data / "learning_cache.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            cache[d["key"]] = d["output"]
    matcher = Matcher(data / "semantic_cache.jsonl")
    ids = [r["task"] for r in result["rows"]["none"]]
    out: dict[str, Any] = {"result": str(res_path), "entry": entry, "who": who, "systems": {}}
    per: dict[str, dict[str, dict]] = {}
    for system in systems:
        rows, missing = [], 0
        for tid in ids:
            t = tasks[tid]
            job = (skiller.serve(t, system) if system in ("tool", "skills") else
                   (learner.serve_context(t, system), (), ()))
            key = episode_key(tid, who, *job)
            if key not in cache:
                missing += 1
                continue
            rows.append({"task": tid, "positive": t.truth["positive"], **semantic_metrics(matcher, t, cache[key])})
        per[system] = {r["task"]: r for r in rows}
        pos = [r for r in rows if r["positive"]]
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
        out["systems"][system] = {"n_pos": len(pos), "missing": missing,
                                  "sem_recall": mean([r["sem_recall"] for r in pos]),
                                  "soft_recall": mean([r["soft_recall"] for r in pos])}
    for system in systems[1:]:
        both = [tid for tid in per["none"] if tid in per[system] and per["none"][tid]["positive"]]
        w = sum(per[system][t]["soft_recall"] > per["none"][t]["soft_recall"] + 1e-9 for t in both)
        l_ = sum(per[system][t]["soft_recall"] < per["none"][t]["soft_recall"] - 1e-9 for t in both)
        out["systems"][system]["paired_soft_vs_none"] = {"wins": w, "losses": l_, "ties": len(both) - w - l_}
    out["jev_input_tokens"] = matcher.tokens
    res_path.with_name(res_path.stem + "_semantic.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
