"""The team as a library of agent types held in the TeamGenome."""

import json
import re

from ospm.harness.env import SUBMIT_MARKER, LocalEnvironment
from ospm.harness.models import FakeChat, ModelTurn, ToolCall
from ospm.team.roles import (
    TeamGenome,
    make_validator,
    render_system,
    sdlc_library,
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


# ---------------------------------------------------------------------------------------------- beyond review
def test_seed_genomes_are_unchanged_by_the_extensions():
    # cached episodes are keyed by these ids; the sdlc types and the board must not change the seed teams
    assert seed_team().id == "5197c151e4a3" and seed_library().id == "dc741e6c3854"
    assert "patch" not in render_system(seed_team().roles["orchestrator"], "p", "c")


def test_sdlc_library_types_render_and_validate():
    lib = sdlc_library()
    new = ["planner", "implementer", "test_writer", "release"]
    assert set(seed_library().specialists()) | set(new) == set(lib.specialists())
    orch = render_system(lib.roles["orchestrator"], "p", "c", team_listing(lib))
    assert "implementation" in orch and "- patch: {" in orch and not re.findall(r"\{[a-z_]+\}", orch)
    for r in new:
        assert lib.roles[r].description and not re.findall(r"\{[a-z_]+\}", render_system(lib.roles[r], "p", "c"))
    assert "git stash" in lib.roles["implementer"].system_template
    ok = {"planner": {"approach": "a", "files": ["f.py"], "steps": ["s"], "risks": []},
          "implementer": {"status": "complete", "files": ["f.py"], "tests_run": [{"command": "ci-run", "passed": True}]},
          "test_writer": {"tests": [{"path": "t.py", "name": "test_x"}], "fails_before": True, "passes_after": None},
          "release": {"bump": "minor", "highlights": ["x"], "breaking": [], "notes": "## 1.2"}}
    for r, out in ok.items():
        check = make_validator(r, lib.roles[r].contract)
        assert check(json.dumps(out)) is None, r
    assert make_validator("implementer", "patch")(json.dumps({"status": "complete", "files": []}))
    assert make_validator("test_writer", "testcases")(json.dumps({"tests": [], "fails_before": "yes"}))
    assert make_validator("release", "release")(json.dumps({"bump": "huge", "highlights": [], "breaking": [], "notes": ""}))
    orch_check = make_validator("orchestrator")
    assert "'patch' is required" in orch_check(json.dumps({"event_kind": "implementation"}))
    assert orch_check(json.dumps({"event_kind": "implementation", "patch": ok["implementer"]})) is None
    assert orch_check(json.dumps({"event_kind": "release", "release": {"bump": "x"}}))


def test_team_board_is_shared_attributed_and_opt_in(tmp_path):
    g = seed_team()
    off = session(g, tmp_path, lambda v: submit_json(CHILD_OUT)).make_agent("orchestrator")
    assert "post_note" not in off.tools and "Notepads" not in off.cfg.system_template
    g.board = True
    assert g.id != seed_team().id and TeamGenome.from_dict(json.loads(json.dumps(g.to_dict()))).board
    sess = session(g, tmp_path, lambda v: submit_json(CHILD_OUT))
    orch = sess.make_agent("orchestrator")
    a, b = sess.make_agent("review", parent=orch), sess.make_agent("security", parent=orch)
    assert {"post_note", "read_notes"} <= set(a.tools) and "Notepads" in a.cfg.system_template
    assert "error" in sess._post_note({"kind": "gossip", "note": "x"}, a)
    sess._post_note({"kind": "covered", "note": "read the whole diff", "paths": ["src/a.py"]}, a)
    first = json.loads(sess._read_notes({"all": False}, b))["notes"]
    assert [(n["from"], n["role"], n["kind"], n["paths"]) for n in first] == [(a.id, "review", "covered", ["src/a.py"])]
    assert json.loads(sess._read_notes({"all": False}, b))["notes"] == []  # only what is new
    assert json.loads(sess._read_notes({"all": False}, a))["notes"] == []  # never your own notes
    sess._post_note({"kind": "finding", "note": "x" * 5000}, b)
    assert len(json.loads(sess._read_notes({"all": True}, orch))["notes"][-1]["note"]) == 800


def test_task_notepads_are_seen_by_their_agents_and_the_orchestrator(tmp_path):
    assert "pad" not in spawn_tool(["review"]).parameters["properties"]
    g = seed_team()
    g.board = True
    sess = session(g, tmp_path, lambda v: submit_json(CHILD_OUT))
    orch = sess.make_agent("orchestrator")
    assert "pad" in orch.tools["spawn_agent"].spec.parameters["properties"]
    ids = [json.loads(sess._spawn({"role": r, "prompt": "x", "pad": pad}, orch))["agent_id"]
           for r, pad in (("review", "Fix"), ("security", "fix"), ("triage", "other"), ("ci_triage", ""))]
    sess.shutdown()
    fix_a, fix_b, other, loner = (sess.agents[i] for i in ids)
    assert sess._pads == {fix_a.id: "fix", fix_b.id: "fix", other.id: "other"}

    def seen(agent):
        return sorted(n["note"] for n in json.loads(sess._read_notes({"all": True}, agent))["notes"])

    sess._post_note({"kind": "finding", "note": "fix-only", "pad": ""}, fix_a)
    sess._post_note({"kind": "covered", "note": "for everyone", "pad": "global"}, fix_a)
    assert "error" in sess._post_note({"kind": "finding", "note": "x", "pad": "other"}, fix_b)  # not its pad
    sess._post_note({"kind": "question", "note": "to other", "pad": "other"}, orch)  # the orchestrator may
    sess._post_note({"kind": "working_on", "note": "loner", "pad": ""}, loner)  # no task pad: global
    assert seen(fix_b) == ["fix-only", "for everyone", "loner"]
    assert seen(other) == ["for everyone", "loner", "to other"]
    assert seen(loner) == ["for everyone"]
    assert seen(orch) == ["fix-only", "for everyone", "loner"]  # every pad, never its own notes
    assert {n["pad"] for n in sess.board} == {"fix", "global", "other"}


def task_session(tmp_path, specialists=("review", "security")):
    g = seed_team()
    g.task_board = True
    sess = session(g, tmp_path, lambda v: submit_json(CHILD_OUT))
    orch = sess.make_agent("orchestrator")
    return sess, orch, [sess.make_agent(r, parent=orch) for r in specialists]


def call(fn, caller, **args):
    return json.loads(fn(args, caller))


def upd(sess, caller, task_id, revision, **kw):
    args = {"task_id": task_id, "revision": revision, "status": "", "owner": "", "note": "", "write_scope": []}
    return json.loads(sess._task_update({**args, **kw}, caller))


def new_task(sess, orch, title="t", **kw):
    args = {"title": title, "description": "", "depends_on": [], "write_scope": [], "owner": ""}
    return json.loads(sess._task_create({**args, **kw}, orch))


def test_task_board_is_opt_in(tmp_path):
    g = seed_team()
    off = session(g, tmp_path, lambda v: submit_json(CHILD_OUT)).make_agent("orchestrator")
    assert not {"task_create", "task_update", "task_list"} & set(off.tools) and "Task board" not in off.cfg.system_template
    g.task_board = True
    assert g.id != seed_team().id and TeamGenome.from_dict(json.loads(json.dumps(g.to_dict()))).task_board
    assert not TeamGenome.from_dict(seed_team().to_dict()).task_board
    sess, orch, (rev,) = task_session(tmp_path, ("review",))
    for a in (orch, rev):
        assert {"task_create", "task_update", "task_list"} <= set(a.tools) and "## Task board" in a.cfg.system_template
    assert "task_create" in orch.cfg.system_template.split("## Task board")[1]
    assert "task_create" not in rev.cfg.system_template.split("## Task board")[1]
    assert orch.tools["task_list"].spec.parameters["required"] == []


def test_task_create_is_orchestrator_only_and_checks_dependencies(tmp_path):
    sess, orch, (rev,) = task_session(tmp_path, ("review",))
    args = {"title": "t", "description": "", "depends_on": [], "write_scope": [], "owner": ""}
    assert "error" in json.loads(sess._task_create(args, rev))
    assert new_task(sess, orch, "first") == {"task_id": "t1", "revision": 1}
    assert "unknown dependencies" in new_task(sess, orch, depends_on=["t9"])["error"]
    assert "unknown owner" in new_task(sess, orch, owner="nobody")["error"]
    assert new_task(sess, orch, "second", depends_on=["t1"], owner=rev.id)["task_id"] == "t2"
    tasks = json.loads(sess._task_list({}, rev))["tasks"]
    assert [(t["id"], t["owner_role"], t["depends_on"]) for t in tasks] == [("t1", "", []), ("t2", "review", ["t1"])]


def test_task_update_is_compare_and_set(tmp_path):
    sess, orch, _ = task_session(tmp_path)
    new_task(sess, orch)
    ok = upd(sess, orch, "t1", 1, status="blocked", note="waiting")
    assert ok["revision"] == 2 and ok["status"] == "blocked"
    stale = upd(sess, orch, "t1", 1, status="pending")
    assert "stale" in stale["error"] and stale["task"]["revision"] == 2 and stale["task"]["status"] == "blocked"
    assert stale["task"]["notes"][0]["note"] == "waiting" and stale["task"]["updated_by"] == orch.id
    assert upd(sess, orch, "t1", 2, status="done", owner="")["revision"] == 3
    assert "unknown task_id" in upd(sess, orch, "t7", 1)["error"]
    assert "status must be" in upd(sess, orch, "t1", 3, status="finished")["error"]


def test_task_update_permissions(tmp_path):
    sess, orch, (a, b) = task_session(tmp_path)
    new_task(sess, orch, owner=a.id)
    new_task(sess, orch)
    assert "owned by" in upd(sess, b, "t1", 1, status="in_progress")["error"]
    assert "owned by" in upd(sess, b, "t1", 1, owner="me")["error"]  # cannot steal an owned task
    assert "owned by nobody" in upd(sess, b, "t2", 1, status="in_progress")["error"]  # claim first
    assert "owned by nobody" in upd(sess, b, "t2", 1, owner=a.id)["error"]  # assigning is not claiming
    assert upd(sess, b, "t2", 1, owner="me", note="mine")["revision"] == 2
    assert sess.tasks[1]["owner"] == b.id and sess.tasks[1]["updated_by"] == b.id
    assert upd(sess, b, "t2", 2, status="in_progress")["status"] == "in_progress"  # the owner may update
    assert upd(sess, a, "t1", 1, status="done")["status"] == "done"
    assert upd(sess, orch, "t2", 3, owner=a.id)["revision"] == 4  # the orchestrator may reassign


def test_task_dependencies_gate_progress(tmp_path):
    sess, orch, (a, _) = task_session(tmp_path)
    new_task(sess, orch)
    new_task(sess, orch, depends_on=["t1"], owner=a.id)
    new_task(sess, orch, depends_on=["t1", "t2"], owner=a.id)
    for status in ("in_progress", "done"):
        assert "t1" in upd(sess, a, "t2", 1, status=status)["error"]
    assert upd(sess, a, "t2", 1, status="blocked")["revision"] == 2  # other statuses are not gated
    upd(sess, orch, "t1", 1, status="done")
    assert upd(sess, a, "t2", 2, status="in_progress")["revision"] == 3
    err = upd(sess, a, "t3", 1, status="in_progress")["error"]
    assert "t2" in err and "t1" not in err
    assert upd(sess, a, "t2", 3, status="done")["revision"] == 4


def test_task_write_scope_overlap_warns_but_succeeds(tmp_path):
    sess, orch, (a, b) = task_session(tmp_path)
    new_task(sess, orch, owner=a.id, write_scope=["src/pkg/"])
    new_task(sess, orch, owner=b.id, write_scope=["src/pkg/mod.py"])  # inside the directory of t1
    new_task(sess, orch, owner=b.id, write_scope=["src/other.py", "tests"])
    new_task(sess, orch, owner=b.id, write_scope=["src/pkg"])  # same owner as t2, same dir as t1
    assert "warnings" not in upd(sess, a, "t1", 1, status="in_progress")
    out = upd(sess, b, "t2", 1, status="in_progress")
    assert out["status"] == "in_progress" and len(out["warnings"]) == 1
    assert "t1" in out["warnings"][0] and "src/pkg/mod.py" in out["warnings"][0]
    assert "warnings" not in upd(sess, b, "t3", 1, status="in_progress")  # disjoint
    assert "t1" in upd(sess, b, "t4", 1, status="in_progress")["warnings"][0]
    assert upd(sess, orch, "t3", 2, write_scope=["./src/pkg/"])["warnings"]  # the orchestrator edits the scope


def test_task_list_reports_conflicts(tmp_path):
    sess, orch, (a, b) = task_session(tmp_path)
    new_task(sess, orch, owner=a.id, write_scope=["src/a.py", "docs/"])
    new_task(sess, orch, owner=b.id, write_scope=["docs/guide.md"])
    new_task(sess, orch, owner=b.id, write_scope=["docs"])
    assert call(sess._task_list, orch)["conflicts"] == []
    upd(sess, a, "t1", 1, status="in_progress")
    upd(sess, b, "t2", 1, status="in_progress")
    upd(sess, b, "t3", 1, status="in_progress")  # same owner as t2: no conflict between them
    out = call(sess._task_list, orch)
    assert [c["tasks"] for c in out["conflicts"]] == [["t1", "t2"], ["t1", "t3"]]
    assert out["conflicts"][0]["paths"] == [["docs/", "docs/guide.md"]]
    upd(sess, b, "t2", 2, status="done")
    assert [c["tasks"] for c in call(sess._task_list, orch)["conflicts"]] == [["t1", "t3"]]


def test_task_limits_truncate(tmp_path):
    sess, orch, (a, _) = task_session(tmp_path)
    new_task(sess, orch, "x" * 500, description="d" * 5000, write_scope=[f"f{i}" for i in range(50)], owner=a.id)
    t = sess.tasks[0]
    assert (len(t["title"]), len(t["description"]), len(t["write_scope"])) == (120, 1500, 20)
    for i in range(1, 25):
        assert upd(sess, a, "t1", i, note="n" * 900)["revision"] == i + 1
    t = sess.tasks[0]
    assert len(t["notes"]) == 20 and {len(n["note"]) for n in t["notes"]} == {500} and t["notes"][0]["by"] == a.id
    for _ in range(29):
        new_task(sess, orch)
    assert len(sess.tasks) == 30 and "task limit" in new_task(sess, orch)["error"]
