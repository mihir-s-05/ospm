import json

from ospm.harness.env import LocalEnvironment
from ospm.learning import skills as sk
from ospm.learning.skills import Skill
from ospm.replay.engine import trace_name
from ospm.replay.workspace import write_skills


def S(name, paths=(), **kw):
    return Skill(name, f"use for {name}", f"# {name}\nstep 1\nstep 2", paths=list(paths), **kw)


def test_slug_and_validation():
    assert sk.slug("My Cool Skill!") == "my-cool-skill"
    assert S("Foo Bar").name == "foo-bar"
    assert len(sk.slug("x" * 100)) <= 48


def test_offered_matching_order_and_k():
    skills = [S("zeta"), S("beta", ["docs/*"]), S("alpha", ["vllm/*.py"]), S("off", ["vllm/*"], status="candidate"),
              S("other", role="triage"), S("gamma", ["tests/*"])]
    got = [s.name for s in sk.offered(skills, ["vllm/a.py", "docs/x.md"])]
    assert got == ["alpha", "beta", "zeta"]
    assert [s.name for s in sk.offered(skills, [])] == ["zeta"]
    assert [s.name for s in sk.offered(skills, ["vllm/a.py", "docs/x.md"], k=2)] == ["alpha", "beta"]
    assert [s.name for s in sk.offered(skills, [], role="triage")] == ["other"]


def test_skills_hash():
    a, b = S("a"), S("b")
    assert sk.skills_hash([]) == ""
    assert sk.skills_hash([a, b]) == sk.skills_hash([S("a"), S("b")])
    assert sk.skills_hash([a, b]) != sk.skills_hash([a])
    c = S("a")
    c.body += "!"
    assert sk.skills_hash([c, b]) != sk.skills_hash([a, b])
    d = S("a", ["x/*"], version=3)  # metadata outside name/description/body does not matter
    assert sk.skills_hash([d, b]) == sk.skills_hash([a, b])


def test_index_block():
    assert sk.index_block([]) == ""
    out = sk.index_block([S("a"), S("b")])
    assert "- a: use for a" in out and "- b: use for b" in out and "skill <name>" in out


def _call(cmd):
    return {"calls": [{"arguments": {"command": cmd}}]}


def test_loaded_skills():
    trace = {"steps": [_call("skill a && skill b"), _call("skill"), _call("ls; skill c | head"),
                       {"calls": [{"arguments": {}}]}, _call("echo skill zzz")],
             "children": [{"steps": [_call("skill d"), _call("skill a\nskill e")], "children": []}]}
    assert sk.loaded_skills(trace) == ["a", "b", "c", "d", "e"]
    assert sk.loaded_skills({}) == []


def test_save_load(tmp_path):
    skills = [S("a", ["x/*"], provenance=["pr1"]), S("b", status="candidate", validation={"n": 1})]
    sk.save(skills, tmp_path / "s.json")
    assert sk.load(tmp_path / "s.json") == skills
    json.loads((tmp_path / "s.json").read_text())


def test_skill_script(tmp_path):
    write_skills(tmp_path, [S("alpha"), S("beta")])
    env = LocalEnvironment(cwd=str(tmp_path), bin_dirs=[str(tmp_path)], timeout=30)
    try:
        r = env.execute("skill")
        assert r.returncode == 0 and "alpha: use for alpha" in r.output and "beta: use for beta" in r.output
        r = env.execute("skill beta")
        assert r.returncode == 0 and r.output.startswith("beta: use for beta") and "step 2" in r.output
        r = env.execute("skill nope")
        assert r.returncode == 1 and "alpha" in r.output and "beta" in r.output
        r = env.execute("skill ../etc")
        assert r.returncode == 1 and "unknown skill" in r.output
    finally:
        env.cleanup()


def test_skill_script_empty(tmp_path):
    write_skills(tmp_path, [])
    env = LocalEnvironment(cwd=str(tmp_path), bin_dirs=[str(tmp_path)], timeout=30)
    try:
        assert "no skills available" in env.execute("skill").output
    finally:
        env.cleanup()


def test_trace_name_unchanged_without_skills():
    import hashlib
    h = hashlib.sha256(b"ctx").hexdigest()[:6]
    assert trace_name("a:1@2", "g", "review", "ctx") == f"a_1_2__g__review__{h}.json.gz"
    assert trace_name("a:1@2", "g", "review") == "a_1_2__g__review__plain.json.gz"
    assert trace_name("a:1@2", "g", "review", "ctx", ()) == trace_name("a:1@2", "g", "review", "ctx")
    n1 = trace_name("a:1@2", "g", "review", "ctx", [S("a")])
    assert n1 != trace_name("a:1@2", "g", "review", "ctx") and n1 != trace_name("a:1@2", "g", "review", "ctx", [S("b")])
