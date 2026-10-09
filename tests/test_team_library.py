"""The team as a library of agent types held in the TeamGenome."""

import json
import re

from ospm.harness.env import SUBMIT_MARKER, LocalEnvironment
from ospm.harness.models import FakeChat, ModelTurn, ToolCall
from ospm.team.roles import (
    TeamGenome,
    make_validator,
    render_system,
    seed_library,
    seed_team,
    specialist,
    team_listing,
)
from ospm.team.runner import TeamSession, spawn_tool

BUILTIN_HASHES = {"triage": "06c3928e5cdd", "ci_triage": "c34c0287fdee", "review": "6e315a5b3cfd",
                  "security": "daf860a2adbb"}
REPORT = {"checked": ["x"], "evidence": "none", "unverified": [], "confidence": "high"}
CHILD_OUT = {"verdict": "approve", "comments": [], "report": REPORT}
NOTES_OUT = {"notes": [{"path": "a.py", "line": None, "note": "reviewers ask for tests"}], "summary": "s",
             "report": REPORT}


def test_builtin_hashes_are_stable():
    hashes = seed_team().role_hashes()
    assert {r: hashes[r] for r in BUILTIN_HASHES} == BUILTIN_HASHES
    assert {r: seed_library().role_hashes()[r] for r in BUILTIN_HASHES} == BUILTIN_HASHES
    g = seed_team().roles["review"]
    g.description = "changed"
    assert g.content_hash() == BUILTIN_HASHES["review"]  # description never enters the agent's hash
    g.contract = "review"
    assert g.content_hash() != BUILTIN_HASHES["review"]


def test_library_types_render_and_validate():
    lib = seed_library()
    assert lib.specialists() == ["triage", "ci_triage", "review", "security", "correctness", "design_docs",
                                 "history", "tests"]
    for r in lib.specialists():
        g = lib.roles[r]
        assert g.description
        text = render_system(g, "pytest", "CONTEXT")
        assert "CONTEXT" in text and "pytest" in text
        assert not re.findall(r"\{[a-z_]+\}", text), r
    assert lib.roles["history"].contract == "notes" and "hist reviews --path" in lib.roles["history"].system_template
    assert make_validator("tests", "review")(json.dumps({"verdict": "approve", "comments": []})) is None
    notes = make_validator("history", "notes")
    assert notes(json.dumps(NOTES_OUT)) is None
    assert notes(json.dumps({"summary": "x"})) and notes(json.dumps({"notes": [{"path": "a", "note": "n", "line": "x"}]}))
    assert make_validator("review")(json.dumps({"verdict": "bogus"}))


def test_orchestrator_sees_current_library():
    lib = seed_library()
    orch = lib.roles["orchestrator"]
    text = render_system(orch, "pytest", "CONTEXT", team_listing(lib))
    assert "{team}" not in text and all(f"- {r}: {lib.roles[r].description}" in text for r in lib.specialists())
    before = lib.id
    lib.roles["tests"].description = "Writes missing tests."
    assert lib.id != before
    assert "- tests: Writes missing tests." in render_system(orch, "pytest", "CONTEXT", team_listing(lib))
    assert "- correctness" not in team_listing(seed_team())


def test_round_trip_keeps_descriptions_and_contracts():
    lib = seed_library()
    back = TeamGenome.from_dict(json.loads(json.dumps(lib.to_dict())))
    assert back.id == lib.id and back.role_hashes() == lib.role_hashes()
    assert back.roles["history"].contract == "notes" and back.roles["tests"].description == lib.roles["tests"].description
    assert TeamGenome.from_dict(seed_team().to_dict()).id == seed_team().id


def submit_json(obj) -> ModelTurn:
    cmd = f"echo {SUBMIT_MARKER} && cat <<'EOF'\n{json.dumps(obj)}\nEOF"
    return ModelTurn("", [ToolCall("", "bash", {"command": cmd})])


def session(genome, tmp_path, policy):
    return TeamSession(genome, FakeChat(policy), lambda: LocalEnvironment(cwd=str(tmp_path)), "p", "c")


def test_spawn_enum_follows_genome(tmp_path):
    g = seed_team()
    del g.roles["security"]
    g.roles["linter"] = specialist("linter", "Check lint.", description="Runs linters.")
    g.roles["scout"] = specialist("scout", "Research.", contract="notes", description="Researches.")
    enum = spawn_tool(g.specialists()).parameters["properties"]["role"]["enum"]
    assert enum == ["triage", "ci_triage", "review", "linter", "scout"]
    sess = session(g, tmp_path, lambda v: submit_json(NOTES_OUT if "scout specialist" in v.system else CHILD_OUT))
    orch = sess.make_agent("orchestrator")
    assert "spawn_agent" in list(orch.tools)
    assert "unknown role" in sess._spawn({"role": "security", "prompt": "x"}, orch)
    aid = json.loads(sess._spawn({"role": "scout", "prompt": "x"}, orch))["agent_id"]
    out = json.loads(sess._wait({"agent_ids": [aid], "timeout_s": 10}, orch))
    note = out["notifications"][0]
    assert note["role"] == "scout" and note["exit_status"] == "Submitted" and note["final_message"]["notes"]
    sess.shutdown()


def test_notes_contract_is_enforced_on_new_type(tmp_path):
    g = seed_team()
    g.roles["scout"] = specialist("scout", "Research.", contract="notes")
    sess = session(g, tmp_path, lambda v: submit_json({"verdict": "approve", "report": REPORT}))
    scout = sess.make_agent("scout", parent=sess.make_agent("orchestrator"))
    assert "notes" in scout.validate(json.dumps({"verdict": "approve", "report": REPORT}))
    assert scout.validate(json.dumps(NOTES_OUT)) is None
