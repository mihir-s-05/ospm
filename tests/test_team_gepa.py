"""Team-GEPA: blame-directed reflection, invocation-aware caching, cost-aware Pareto, merge, budget, repair."""

import json
from dataclasses import dataclass

import pytest

from ospm.evolution.team_gepa import (
    EpisodeCache,
    EpisodeOutcome,
    TeamCandidate,
    TeamGEPA,
    TeamGEPAConfig,
    repair_template,
)
from ospm.llm.fake import FakeLLM
from ospm.team.roles import TeamGenome, seed_team


@dataclass
class T:
    id: str
    kind: str
    event: str = "event"


LESSON = "Always search for duplicates before accepting."


def fake_episode(calls):
    """Triage tasks score 1.0 only once the triage playbook has the lesson; review tasks never use triage."""

    def run(genome: TeamGenome, task: T) -> EpisodeOutcome:
        calls.append((genome.id, task.id))
        roles = ["orchestrator", "triage"] if task.kind == "triage" else ["orchestrator", "review"]
        good = LESSON in genome.roles["triage"].playbook
        score = (1.0 if good else 0.2) if task.kind == "triage" else 0.6
        return EpisodeOutcome(task.id, score, "missed the duplicate" if score < 1 else "ok", 0.01,
                              {r: genome.roles[r].content_hash() for r in roles}, f"trace of {task.id}", "Submitted")
    return run


def reflector(blamed="triage", add=LESSON):
    def respond(system, user, schema):
        assert "Episodes" in user and "Feedback (what actually happened)" in user
        return {"blamed_role": blamed, "diagnosis": "duplicates are missed", "playbook_add": [add],
                "playbook_remove": [], "system_template": "", "step_limit": 0}
    return FakeLLM(respond)


def tasks(prefix, n, kind="triage"):
    return [T(f"{prefix}{i}", kind) for i in range(n)]


def test_blame_directed_evolution_improves_and_uses_cache():
    calls = []
    train = tasks("tr", 4) + tasks("rv", 2, "review")
    val = tasks("va", 3) + tasks("vr", 2, "review")
    g = TeamGEPA(seed_team(), train, val, tasks("te", 2), fake_episode(calls), reflector(),
                 config=TeamGEPAConfig(iterations=4, minibatch=3, workers=1, seed=1), log=lambda m: None)
    res = g.run()
    assert res["best"]["val"] > res["seed"]["val"]
    assert res["test"] == {"seed": 0.2, "best": 1.0}
    assert LESSON in g.best_genome.roles["triage"].playbook
    assert any(c["blamed"] == "triage" for c in res["candidates"])
    # invocation-aware caching: review episodes never invoke triage and every genome here shares the
    # orchestrator/review roles, so each review task runs at most once no matter how many genomes need it
    from collections import Counter

    review_runs = Counter(t for _, t in calls if t.startswith(("rv", "vr")))
    assert review_runs and max(review_runs.values()) == 1
    triage_runs = Counter(t for _, t in calls if t.startswith(("tr", "va")))
    assert max(triage_runs.values()) >= 2  # triage episodes do re-run when the triage role changes
    assert res["cache"]["hits"] > 0


def test_cache_semantics():
    cache = EpisodeCache()
    g = seed_team()
    o = EpisodeOutcome("t1", 0.5, "", 0.0, {"orchestrator": g.roles["orchestrator"].content_hash()})
    cache.put(o)
    other = g.clone()
    other.roles["security"].playbook.append("x")  # never ran in t1 -> cached result still valid
    assert cache.get("t1", other) is o
    other.roles["orchestrator"].playbook.append("y")  # the orchestrator ran -> must re-run
    assert cache.get("t1", other) is None
    assert cache.get("t2", g) is None


def test_cache_persists(tmp_path):
    p = tmp_path / "c.jsonl"
    g = seed_team()
    EpisodeCache(p).put(EpisodeOutcome("t1", 1.0, "f", 0.1, {"triage": g.roles["triage"].content_hash()}))
    assert EpisodeCache(p).get("t1", g).score == 1.0


def test_vanilla_round_robin_ignores_blame():
    seen = []

    def respond(system, user, schema):
        seen.append(user.split("You may only blame and edit: ")[1].split("\n")[0])
        return {"blamed_role": "triage", "diagnosis": "", "playbook_add": ["lesson"], "playbook_remove": [],
                "system_template": "", "step_limit": 0}
    g = TeamGEPA(seed_team(), tasks("tr", 3), tasks("va", 2), [], fake_episode([]), FakeLLM(respond),
                 config=TeamGEPAConfig(iterations=3, minibatch=2, workers=1, blame=False, evaluate_test=False),
                 log=lambda m: None)
    g.run()
    assert seen and all("," not in s for s in seen)  # exactly one allowed role per reflection
    assert set(seen) <= {"orchestrator", "triage"}


def test_reflection_cannot_edit_roles_that_did_not_run():
    g = TeamGEPA(seed_team(), tasks("tr", 2), tasks("va", 2), [], fake_episode([]), reflector(blamed="security"),
                 config=TeamGEPAConfig(iterations=1, workers=1, evaluate_test=False), log=lambda m: None)
    outs = [fake_episode([])(g.seed, t) for t in g.train]
    child, role = g.reflect(g.seed, outs, g.train)
    assert role in {"orchestrator", "triage"} and child.roles["security"].playbook == []


def test_cost_aware_dominance():
    g1, g2, g3 = seed_team(), seed_team().clone(), seed_team().clone()
    g2.roles["triage"].playbook.append("a")
    g3.roles["triage"].playbook.append("b")
    cheap = TeamCandidate(g1, {"a": 0.5, "b": 0.5}, {"a": 0.01, "b": 0.01}, 0, "seed")
    pricey = TeamCandidate(g2, {"a": 0.5, "b": 0.5}, {"a": 0.10, "b": 0.10}, 1, "reflect")
    spec = TeamCandidate(g3, {"a": 1.0, "b": 0.0}, {"a": 0.10, "b": 0.10}, 2, "reflect")
    assert TeamGEPA.dominates(cheap, pricey) and not TeamGEPA.dominates(pricey, cheap)
    eng = TeamGEPA(seed_team(), tasks("t", 1), tasks("v", 1), [], fake_episode([]), reflector(), log=lambda m: None)
    front = eng.front([cheap, pricey, spec])
    assert set(front) == {cheap.id, spec.id}  # same quality but 10x the cost is dominated


def test_merge_takes_role_versions_that_changed():
    base = seed_team()
    a = base.clone()
    a.roles["triage"].playbook.append("triage lesson")
    b = base.clone()
    b.roles["review"].playbook.append("review lesson")
    eng = TeamGEPA(base, tasks("t", 1), tasks("v", 1), [], fake_episode([]), reflector(), log=lambda m: None)
    ca = TeamCandidate(a, {"v0": 1}, {"v0": 0}, 1, "reflect")
    cb = TeamCandidate(b, {"v0": 1}, {"v0": 0}, 2, "reflect")
    child = eng.merge(ca, cb)
    assert child.roles["triage"].playbook == ["triage lesson"] and child.roles["review"].playbook == ["review lesson"]


def test_budget_stop_and_validation():
    eng = TeamGEPA(seed_team(), tasks("tr", 2), tasks("va", 2), [], fake_episode([]), reflector(),
                   budget_exhausted=lambda: True, log=lambda m: None)
    assert eng.run()["stop"] == "budget"
    with pytest.raises(ValueError):
        TeamGEPA(seed_team(), [], tasks("va", 1), [], fake_episode([]), reflector())


def test_repair_template_restores_placeholders():
    fixed = repair_template("review", "You review code. Be strict.")
    for p in ("{project}", "{project_context}", "{tools}", "{review_contract}", "{playbook}"):
        assert p in fixed
    ok = seed_team().roles["triage"].system_template
    assert repair_template("triage", ok) == ok


def test_genome_serialisation_roundtrip(tmp_path):
    g = seed_team().clone()
    g.roles["ci_triage"].playbook.append("check flaky history")
    back = TeamGenome.from_dict(json.loads(json.dumps(g.to_dict())))
    assert back.id == g.id and back.parents == g.parents
