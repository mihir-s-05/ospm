"""mini-swe-agent-style harness: sandbox, agent loop, compaction, team coordination, provider wire formats."""

import json
import os
import re
import time

import pytest

from ospm.harness.agent import SUMMARY_SYSTEM, Agent, AgentConfig, Tool
from ospm.harness.env import SUBMIT_MARKER, LocalEnvironment, extract_submission, format_observation
from ospm.harness.models import (
    AnthropicChat,
    FakeChat,
    ModelTurn,
    OpenAIChat,
    ToolCall,
    ToolResult,
    ToolSpec,
    bash_call,
)
from ospm.llm import UsageMeter
from ospm.team.roles import TeamGenome, make_validator, parse_json, render_system, seed_team
from ospm.team.runner import TeamSession, run_episode

SUBMIT = f"echo {SUBMIT_MARKER} && echo "


def submit_json(obj) -> ModelTurn:
    return bash_call(f"echo {SUBMIT_MARKER} && cat <<'EOF'\n{json.dumps(obj)}\nEOF")


@pytest.fixture
def env(tmp_path):
    e = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    yield e
    e.cleanup()


# ---------------------------------------------------------------------------------------------- sandbox
def test_env_runs_commands_statelessly(env, tmp_path):
    assert env.execute("echo hello && pwd").output.splitlines()[0] == "hello"
    env.execute("cd / && export FOO=1")
    assert env.execute("echo ${FOO:-unset}").output.strip() == "unset"  # nothing persists between actions
    (tmp_path / "f.txt").write_text("data")
    assert env.execute("cat f.txt").output == "data"


def test_env_hides_credentials(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    out = env.execute("env").output
    assert "sk-secret" not in out and "ghp_secret" not in out
    assert env.execute('echo "$HOME"').output.strip() != os.path.expanduser("~").replace("\\", "/")
    assert "GH_CONFIG_DIR" in out and "GIT_SSH_COMMAND=false" in out


@pytest.mark.parametrize("cmd", ["git push origin main", "ls && gh api user", "sudo rm x", "rm -rf /",
                                 "echo hi; git push"])
def test_env_denylist(env, cmd):
    res = env.execute(cmd)
    assert res.blocked and res.returncode == 126


def test_env_timeout_kills_process_tree(env):
    t = time.time()
    res = env.execute("sleep 30; echo never", timeout=2)
    assert res.timed_out and "never" not in res.output and time.time() - t < 20


def test_submission_and_observation_format():
    assert extract_submission(f"\n{SUBMIT_MARKER}\n{{\"a\": 1}}\n") == '{"a": 1}'
    assert extract_submission("not it\n" + SUBMIT_MARKER) is None
    from ospm.harness.env import ExecResult

    obs = format_observation(ExecResult("x" * 50000, 0), max_chars=1000)
    assert "<returncode>0</returncode>" in obs and "elided" in obs and len(obs) < 2000


# ---------------------------------------------------------------------------------------------- agent loop
def scripted(*turns):
    """Policy that plays the given turns in order (summary requests answered separately)."""
    it = iter(turns)

    def policy(view):
        if view.system == SUMMARY_SYSTEM:
            return "SUMMARY: found the bug in foo.py"
        return next(it)
    return policy


def test_agent_runs_and_submits(env):
    model = FakeChat(scripted(bash_call("echo hello"), submit_json({"answer": 42})))
    agent = Agent("triage", model, env, AgentConfig("sys"))
    res = agent.run(task="do it")
    assert res.exit_status == "Submitted" and json.loads(res.submission) == {"answer": 42}
    assert res.steps == 2 and agent.trace.steps[0].results[0]["output"].startswith("<returncode>0")
    assert "hello" in agent.trace.steps[0].results[0]["output"]


def test_format_error_then_recovery_and_validation(env):
    model = FakeChat(scripted("I think I'm done", submit_json({"wrong": 1}), submit_json({"ok": True})))
    agent = Agent("x", model, env, AgentConfig("sys"),
                  validate=lambda s: None if parse_json(s) and parse_json(s).get("ok") else "needs ok")
    res = agent.run(task="t")
    assert res.exit_status == "Submitted" and res.steps == 3
    assert agent.trace.steps[0].event == "format_error"
    assert "Submission rejected: needs ok" in agent.trace.steps[1].results[0]["output"]


def test_limits(env):
    model = FakeChat(lambda view: bash_call("echo loop"))
    res = Agent("x", model, env, AgentConfig("sys", step_limit=3)).run(task="t")
    assert res.exit_status.startswith("LimitsExceeded") and res.steps == 4  # 3 + the final turn it did not use
    meter = UsageMeter(budget_usd=0.0)
    res = Agent("x", FakeChat(lambda v: bash_call("echo"), meter=meter), env, AgentConfig("sys")).run(task="t")
    assert res.exit_status.startswith("ModelError") and "budget" in res.exit_status


def test_step_limit_warns_then_gives_one_final_turn_to_submit(env):
    from ospm.harness.agent import FINAL_TURN

    def policy(view):
        users = [m["text"] for m in view.transcript if m["role"] == "user"]
        if users and users[-1] == FINAL_TURN:
            return submit_json({"partial": True})
        return bash_call("echo still looking")

    agent = Agent("x", FakeChat(policy), env, AgentConfig("sys", step_limit=8))
    res = agent.run(task="t")
    assert res.exit_status == "Submitted" and json.loads(res.submission) == {"partial": True} and res.steps == 9
    assert [s.event for s in agent.trace.steps].count("final_turn") == 1
    notes = [m["text"] for m in agent.conv.view.transcript if m["role"] == "user" and "steps left" in m["text"]]
    assert len(notes) == 1 and "3 steps left" in notes[0]
    # answering the final turn in plain text (no tool call) still counts as the submission
    def plain(view):
        users = [m["text"] for m in view.transcript if m["role"] == "user"]
        return '{"partial": 1}' if users and users[-1] == FINAL_TURN else bash_call("echo still looking")

    res = Agent("x", FakeChat(plain), env, AgentConfig("sys", step_limit=4)).run(task="t")
    assert res.exit_status == "Submitted" and json.loads(res.submission) == {"partial": 1}


def test_simple_compaction(env):
    seen = {}

    def policy(view):
        if view.system == SUMMARY_SYSTEM:
            return "SUMMARY: the failing test is test_x"
        if view.transcript[0].get("compacted"):
            seen["after"] = view.transcript[0]["text"]
            return submit_json({"done": True})
        return bash_call("python -c \"print('y' * 4000)\"")

    agent = Agent("x", FakeChat(policy), env, AgentConfig("sys", compaction_tokens=1500, step_limit=10))
    res = agent.run(task="find the bug")
    assert res.exit_status == "Submitted" and agent.trace.compactions >= 1
    assert "find the bug" in seen["after"] and "SUMMARY: the failing test is test_x" in seen["after"]
    assert any(s.event == "compaction" for s in agent.trace.steps)


def test_text_mode(env):
    turns = iter(["Let me look.\n```mswea_bash_command\necho hi\n```",
                  "two blocks ```mswea_bash_command\na\n``` ```mswea_bash_command\nb\n```",
                  f"```mswea_bash_command\necho {SUBMIT_MARKER} && echo final\n```"])
    agent = Agent("x", FakeChat(lambda v: next(turns)), env, AgentConfig("sys", text_mode=True))
    res = agent.run(task="t")
    assert res.submission == "final" and agent.trace.steps[1].event == "format_error"


def test_follow_up_send(env):
    turns = iter([submit_json({"v": 1}), submit_json({"v": 2})])
    agent = Agent("x", FakeChat(lambda v: next(turns)), env, AgentConfig("sys"))
    assert json.loads(agent.run(task="t").submission) == {"v": 1}
    second = agent.send("are you sure?")
    assert json.loads(second.submission) == {"v": 2}
    assert agent.conv.view.transcript[-2]["text"] == "are you sure?" or any(
        e.get("text") == "are you sure?" for e in agent.conv.view.transcript)


def test_parallel_extra_tools(env):
    order = []

    def slow(args, agent):
        time.sleep(0.5)
        order.append(args["x"])
        return f"done {args['x']}"
    spec = ToolSpec("slow", "slow tool", {"type": "object", "properties": {"x": {"type": "string"}},
                                           "required": ["x"], "additionalProperties": False})
    turns = iter([ModelTurn("", [ToolCall("", "slow", {"x": "a"}), ToolCall("", "slow", {"x": "b"})]),
                  submit_json({})])
    agent = Agent("x", FakeChat(lambda v: next(turns)), env, AgentConfig("sys"), tools=[Tool(spec, slow)])
    t = time.time()
    agent.run(task="t")
    assert time.time() - t < 0.95 and sorted(order) == ["a", "b"]


# ---------------------------------------------------------------------------------------------- team
TRIAGE_OUT = {"kind": "bug", "decision": "accept", "duplicate_of": None, "components": ["topic: fixtures"],
              "reply": "thanks", "rationale": "r"}


REPORT = {"checked": ["hist issue 1"], "evidence": "none needed", "unverified": [], "confidence": "high"}
CHILD_OUT = {**TRIAGE_OUT, "report": REPORT}
ORCH_OUT = {"event_kind": "issue", "triage": TRIAGE_OUT, "review": None, "ci": [], "security": None, "summary": "ok"}


def tc(name, **args) -> ModelTurn:
    return ModelTurn("", [ToolCall("", name, args)])


def n_asst(view) -> int:
    return sum(1 for e in view.transcript if e["role"] == "assistant")


def tool_jsons(view) -> list[dict]:
    out = []
    for e in view.transcript:
        if e["role"] == "tool":
            try:
                out.append(json.loads(e["text"]))
            except json.JSONDecodeError:
                out.append({"raw": e["text"]})
    return out


def notes(view, kind=None) -> list[dict]:
    found = [n for o in tool_jsons(view) for n in o.get("notifications", [])]  # consumed by wait_for_agents
    found += [json.loads(m) for e in view.transcript if e["role"] == "user"  # or delivered before a step
              for m in re.findall(r"<notification>(.*?)</notification>", e.get("text") or "", re.DOTALL)]
    return [n for n in found if kind in (None, n["type"])]


def team(orch, child):
    def policy(view):
        if view.system == SUMMARY_SYSTEM:
            return "summary"
        return orch(view) if "lead maintainer agent" in view.system else child(view)
    return policy


def spin(view) -> ModelTurn:  # a specialist that works until it is closed
    return bash_call("sleep 0.02")


def run_team(policy, tmp_path, entry="orchestrator", genome=None):
    return run_episode(genome or seed_team(), FakeChat(policy), lambda: LocalEnvironment(cwd=str(tmp_path)),
                       task_id="t1", entry_role=entry, variables={"event": "issue #1", "as_of": "now"},
                       project="demo", project_context="ctx")


def test_team_parallel_spawn_and_wait(tmp_path):
    def orch(view):
        if n_asst(view) == 0:
            return ModelTurn("", [ToolCall("", "spawn_agent", {"role": "triage", "prompt": "triage #1"}),
                                  ToolCall("", "spawn_agent", {"role": "security", "prompt": "check"})])
        if len(notes(view, "finished")) < 2:
            return tc("wait_for_agents", agent_ids=[], timeout_s=10)
        return submit_json(ORCH_OUT)

    def child(view):
        return submit_json(CHILD_OUT if "issue-triage" in view.system else {"findings": [], "report": REPORT})

    ep = run_team(team(orch, child), tmp_path)
    assert ep.exit_status == "Submitted" and ep.output["triage"]["kind"] == "bug"
    assert ep.roles_used == {"orchestrator", "triage", "security"} and set(ep.role_hashes) == ep.roles_used
    assert len(ep.trace.children) == 2 and ep.steps >= 5
    assert all('"report"' in c.task and c.exit_status == "Submitted" for c in ep.trace.children)


def test_finished_notification_carries_final_message_with_report(tmp_path):
    seen = []

    def orch(view):
        if n_asst(view) == 0:
            return tc("spawn_agent", role="triage", prompt="triage #1")
        if not notes(view, "finished"):
            return tc("wait_for_agents", agent_ids=[], timeout_s=10)
        seen.extend(notes(view, "finished"))
        return submit_json(ORCH_OUT)

    run_team(team(orch, lambda v: submit_json(CHILD_OUT)), tmp_path)
    (note,) = seen
    assert note["role"] == "triage" and note["exit_status"] == "Submitted" and note["final_message"] == CHILD_OUT


def test_send_message_resumes_finished_agent(tmp_path):
    def orch(view):
        outs = tool_jsons(view)
        if n_asst(view) == 0:
            return tc("spawn_agent", role="triage", prompt="triage #1")
        aid = outs[0]["agent_id"]
        done = len(notes(view, "finished"))
        if done == 1 and not any(o.get("status") == "resumed" for o in outs):
            return tc("send_message", agent_id=aid, message="double-check the kind")
        if done < 2:
            return tc("wait_for_agents", agent_ids=[aid], timeout_s=10)
        return submit_json(ORCH_OUT)

    def child(view):
        return submit_json({**CHILD_OUT, "rationale": "second" if n_asst(view) else "first"})

    ep = run_team(team(orch, child), tmp_path)
    (c,) = ep.trace.children
    assert ep.exit_status == "Submitted" and len(c.steps) == 2  # the same child trace grew
    assert json.loads(c.submission)["rationale"] == "second"


def test_message_parent_reaches_parent(tmp_path):
    got = []

    def child(view):
        return tc("message_parent", message="found it early") if n_asst(view) == 0 else submit_json(CHILD_OUT)

    def orch(view):
        got.extend(notes(view, "message"))
        if n_asst(view) == 0:
            return tc("spawn_agent", role="triage", prompt="t")
        if not notes(view, "finished"):
            return tc("wait_for_agents", agent_ids=[], timeout_s=10)
        return submit_json(ORCH_OUT)

    ep = run_team(team(orch, child), tmp_path)
    assert ep.exit_status == "Submitted" and got[-1]["message"] == "found it early" and got[-1]["role"] == "triage"


def test_notifications_delivered_once_automatically_or_by_wait(tmp_path):
    sess = TeamSession(seed_team(), FakeChat(lambda v: "x"), lambda: LocalEnvironment(cwd=str(tmp_path)), "p", "c")
    orch = sess.make_agent("orchestrator")
    kid = sess.make_agent("triage", parent=orch)
    assert sess._message_parent({"message": "hi"}, kid) == "sent"
    (text,) = orch.pull()
    assert text.startswith("<notification>") and json.loads(text[14:-15])["message"] == "hi"
    assert orch.pull() == []  # popped: never delivered twice
    sess._message_parent({"message": "again"}, kid)
    assert json.loads(sess._wait({"agent_ids": [], "timeout_s": 5}, orch))["notifications"][0]["message"] == "again"
    assert orch.pull() == []


def test_close_agent_stops_running_specialist(tmp_path):
    def orch(view):
        n = n_asst(view)
        if n == 0:
            return tc("spawn_agent", role="triage", prompt="t")
        aid = tool_jsons(view)[0]["agent_id"]
        if n == 1:
            return tc("close_agent", agent_id=aid)
        if n == 2:
            return tc("wait_for_agents", agent_ids=[aid], timeout_s=10)
        return submit_json(ORCH_OUT)

    ep = run_team(team(orch, spin), tmp_path)
    assert ep.exit_status == "Submitted" and ep.trace.children[0].exit_status == "Closed"
    assert json.loads(ep.trace.steps[1].results[0]["output"])["status"] == "closed"


def test_orchestrator_cannot_submit_while_child_runs(tmp_path):
    def orch(view):
        n = n_asst(view)
        if n == 0:
            return tc("spawn_agent", role="triage", prompt="t")
        if n == 1:
            return submit_json(ORCH_OUT)
        aid = tool_jsons(view)[0]["agent_id"]
        if n == 2:
            return tc("close_agent", agent_id=aid)
        if n == 3:
            return tc("wait_for_agents", agent_ids=[], timeout_s=10)
        return submit_json(ORCH_OUT)

    ep = run_team(team(orch, spin), tmp_path)
    rejected = ep.trace.steps[1].results[0]["output"]
    assert "Submission rejected" in rejected and "1 agents are still running" in rejected
    assert ep.exit_status == "Submitted"


def test_spawned_specialist_must_report_but_standalone_need_not(tmp_path):
    def child(view):
        return submit_json(CHILD_OUT if n_asst(view) else TRIAGE_OUT)  # the first attempt has no report

    def orch(view):
        if n_asst(view) == 0:
            return tc("spawn_agent", role="triage", prompt="t")
        if not notes(view, "finished"):
            return tc("wait_for_agents", agent_ids=[], timeout_s=10)
        return submit_json(ORCH_OUT)

    ep = run_team(team(orch, child), tmp_path)
    (c,) = ep.trace.children
    assert "report" in c.steps[0].results[0]["output"] and c.steps[0].results[0]["is_error"]
    assert json.loads(c.submission)["report"] == REPORT and c.task.endswith("and your confidence.")
    solo = run_team(lambda v: submit_json(TRIAGE_OUT), tmp_path, entry="triage")
    assert solo.exit_status == "Submitted" and solo.output == TRIAGE_OUT and "report" not in solo.trace.task


def test_team_tool_errors(tmp_path):
    sess = TeamSession(seed_team(), FakeChat(lambda v: submit_json(CHILD_OUT)),
                       lambda: LocalEnvironment(cwd=str(tmp_path)), "p", "c", max_specialists=1)
    orch, other = sess.make_agent("orchestrator"), sess.make_agent("orchestrator")
    assert "unknown role" in sess._spawn({"role": "chef", "prompt": "x"}, orch)
    assert "unknown agent_id" in sess._send({"agent_id": "nope", "message": "x"}, orch)
    assert "unknown agent_id" in sess._close({"agent_id": "nope"}, orch)
    assert "unknown agent_id" in sess._wait({"agent_ids": ["nope"], "timeout_s": 1}, orch)
    aid = json.loads(sess._spawn({"role": "triage", "prompt": "x"}, orch))["agent_id"]
    assert "limit" in sess._spawn({"role": "triage", "prompt": "x"}, orch)
    assert "unknown agent_id" in sess._send({"agent_id": aid, "message": "x"}, other)  # not its child
    assert "unknown agent_id" in sess._close({"agent_id": aid}, other)
    sess._wait({"agent_ids": [aid], "timeout_s": 10}, orch)
    sess._close({"agent_id": aid}, orch)
    assert "closed" in json.loads(sess._send({"agent_id": aid, "message": "x"}, orch))["error"]
    sess.shutdown()


def test_run_episode_closes_leftover_agents(tmp_path):
    genome = seed_team()
    genome.roles["orchestrator"].step_limit = 2

    def orch(view):
        return tc("spawn_agent", role="triage", prompt="t") if n_asst(view) == 0 else bash_call("true")

    t = time.time()
    ep = run_team(team(orch, spin), tmp_path, genome=genome)
    assert ep.exit_status.startswith("LimitsExceeded")
    assert ep.trace.children[0].exit_status == "Closed" and time.time() - t < 5


def test_agent_post_is_delivered_before_next_step(env):
    seen = []

    def poke(args, agent):
        agent.post("news 1")
        agent.post("news 2")
        return "poked"
    spec = ToolSpec("poke", "post to self", {"type": "object", "properties": {}, "required": [],
                                              "additionalProperties": False})

    def policy(view):
        seen.append([e["text"] for e in view.transcript if e["role"] == "user"])
        return tc("poke") if len(seen) == 1 else submit_json({"ok": 1})
    agent = Agent("x", FakeChat(policy), env, AgentConfig("sys"), tools=[Tool(spec, poke)])
    assert json.loads(agent.run(task="t").submission) == {"ok": 1}
    assert seen[1][1:] == ["news 1", "news 2"]
    assert [s.event for s in agent.trace.steps].count("message") == 2


def test_agent_without_messages_is_unchanged_and_close_stops(env):
    turns = iter([bash_call("echo hi"), submit_json({"v": 1})])
    agent = Agent("x", FakeChat(lambda v: next(turns)), env, AgentConfig("sys"))
    assert json.loads(agent.run(task="t").submission) == {"v": 1}
    assert [e["role"] for e in agent.conv.view.transcript] == ["user", "assistant", "tool", "assistant", "tool"]
    assert all(s.event == "" for s in agent.trace.steps)
    agent.close()
    assert agent.closed
    res = agent.send("more")
    assert res.exit_status == "Closed" and res.submission == ""


def test_role_prompts_render_and_validators():
    g = seed_team()
    for rg in g.roles.values():
        text = render_system(rg, "pytest", "CONTEXT")
        assert "{tools}" not in text and "{playbook}" not in text and "CONTEXT" in text and SUBMIT_MARKER in text
    v = make_validator("triage")
    assert v(json.dumps(TRIAGE_OUT)) is None
    assert "decision" in v(json.dumps({**TRIAGE_OUT, "decision": "maybe"}))
    assert "duplicate_of" in v(json.dumps({**TRIAGE_OUT, "decision": "duplicate"}))
    assert make_validator("review")(json.dumps({"verdict": "approve", "comments": [{"path": "a", "line": "x"}]}))
    assert make_validator("orchestrator")(json.dumps({"event_kind": "issue"}))
    assert parse_json("```json\n{\"a\": 1}\n```") == {"a": 1} and parse_json("noise {\"a\": 2} tail") == {"a": 2}
    g2 = TeamGenome.from_dict(g.to_dict())
    assert g2.id == g.id
    g2.roles["review"].playbook.append("check changelog")
    assert g2.id != g.id and g2.role_hashes()["triage"] == g.role_hashes()["triage"]


# ---------------------------------------------------------------------------------------------- provider wire formats
httpx = pytest.importorskip("httpx")
TOOLS = [ToolSpec("bash", "run", {"type": "object", "properties": {"command": {"type": "string"}},
                                   "required": ["command"], "additionalProperties": False})]


def test_openai_tool_loop_wire_format():
    openai = pytest.importorskip("openai")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        out = ([{"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "bash",
                 "arguments": json.dumps({"command": "ls"}), "status": "completed"}]
               if len(bodies) == 1 else
               [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done", "annotations": []}]}])
        return httpx.Response(200, json={"id": "r", "object": "response", "created_at": 0, "model": "gpt-6-luna",
                                         "status": "completed", "output": out, "parallel_tool_calls": True,
                                         "tool_choice": "auto", "tools": [],
                                         "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                                                   "input_tokens_details": {"cached_tokens": 0},
                                                   "output_tokens_details": {"reasoning_tokens": 0}}})

    client = openai.OpenAI(api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    conv = OpenAIChat(client=client).start("SYS", TOOLS)
    conv.add_user("go")
    t1 = conv.query("low")
    assert t1.tool_calls[0].name == "bash" and t1.tool_calls[0].arguments == {"command": "ls"}
    conv.add_tool_results([ToolResult("call_1", "file.txt")])
    t2 = conv.query("low")
    assert t2.text == "done"
    second = bodies[1]
    assert second["tools"][0]["strict"] is True and second["instructions"] == "SYS"
    types = [i.get("type", i.get("role")) for i in second["input"]]
    assert types == ["user", "reasoning", "function_call", "function_call_output"]  # native items round-trip
    conv.compact("SUMMARY")
    assert conv.items == [{"role": "user", "content": "SUMMARY"}]


def test_anthropic_tool_loop_wire_format():
    anthropic = pytest.importorskip("anthropic")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        content = ([{"type": "thinking", "thinking": "", "signature": "sig"},
                    {"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "ls"}}]
                   if len(bodies) == 1 else [{"type": "text", "text": "done"}])
        return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                                         "content": content, "stop_reason": "tool_use" if len(bodies) == 1 else "end_turn",
                                         "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}})

    client = anthropic.Anthropic(api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    conv = AnthropicChat(client=client).start("SYS", TOOLS)
    conv.add_user("go")
    t1 = conv.query("medium")
    assert t1.tool_calls[0].id == "tu_1" and t1.tool_calls[0].arguments == {"command": "ls"}
    conv.add_tool_results([ToolResult("tu_1", "file.txt")])
    conv.query("medium")
    msgs = bodies[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0]["type"] == "thinking" and msgs[1]["content"][0]["signature"] == "sig"  # unchanged
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": "file.txt",
                                     "is_error": False}
    assert bodies[1]["fallbacks"] == "default" and bodies[1]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in bodies[1] and bodies[1]["output_config"] == {"effort": "medium"}
    conv.compact("SUMMARY")  # simple compaction: nothing older (no thinking) is replayed
    assert conv.messages == [{"role": "user", "content": [{"type": "text", "text": "SUMMARY"}]}]


def test_openai_server_side_compaction():
    openai = pytest.importorskip("openai")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        out = [{"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"}] if len(bodies) == 2 else []
        out.append({"type": "function_call", "id": f"fc_{len(bodies)}", "call_id": f"call_{len(bodies)}",
                    "name": "bash", "arguments": json.dumps({"command": "ls"}), "status": "completed"})
        return httpx.Response(200, json={"id": "r", "object": "response", "created_at": 0, "model": "gpt-6-luna",
                                         "status": "completed", "output": out, "parallel_tool_calls": True,
                                         "tool_choice": "auto", "tools": [],
                                         "usage": {"input_tokens": 1000, "output_tokens": 5, "total_tokens": 1005,
                                                   "input_tokens_details": {"cached_tokens": 900},
                                                   "output_tokens_details": {"reasoning_tokens": 0}}})

    client = openai.OpenAI(api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    model = OpenAIChat(client=client)
    conv = model.start("SYS", TOOLS)
    assert conv.enable_native_compaction(5000)
    conv.add_user("go")
    t1 = conv.query("low")
    assert bodies[0]["context_management"] == [{"type": "compaction", "compact_threshold": 5000}]
    assert not t1.compacted and t1.cached_tokens == 900
    assert t1.cost == pytest.approx((100 * 0.10 + 900 * 0.01 + 5 * 0.50) / 1e6)  # cached input at the cache price
    conv.add_tool_results([ToolResult("call_1", "file.txt")])
    t2 = conv.query("low")
    assert t2.compacted
    # items before the compaction item are dropped; the compaction item leads the next request
    conv.add_tool_results([ToolResult("call_2", "x")])
    conv.query("low")
    assert [i.get("type") for i in bodies[2]["input"]] == ["compaction", "function_call", "function_call_output"]


def test_agent_skips_local_compaction_when_provider_compacts(env):
    agent = Agent("x", FakeChat(lambda c: None), env, AgentConfig("sys", compaction_tokens=10))
    agent.conv = type("C", (), {"context_tokens": 10_000})()
    agent._native_compaction = True
    agent._maybe_compact()  # must not summarise: the provider owns compaction
    assert agent.trace.compactions == 0


def test_message_arriving_while_a_child_submits_resumes_it(tmp_path):
    from ospm.team.runner import TeamSession

    box = {}

    def kid_policy(view):
        if n_asst(view) == 0:  # the parent writes while this (submitting) step is in flight
            box["sess"]._send({"agent_id": box["kid"].id, "message": "also check X"}, box["orch"])
            return submit_json({**CHILD_OUT, "rationale": "first"})
        assert any("also check X" in (e.get("text") or "") for e in view.transcript if e["role"] == "user")
        return submit_json({**CHILD_OUT, "rationale": "second"})

    sess = TeamSession(seed_team(), FakeChat(kid_policy), lambda: LocalEnvironment(cwd=str(tmp_path)), "p", "c")
    box.update(sess=sess, orch=sess.make_agent("orchestrator"))
    out = json.loads(sess._spawn({"role": "triage", "prompt": "t"}, box["orch"]))
    box["kid"] = sess.agents[out["agent_id"]]
    deadline = time.time() + 10
    finished = []
    while not finished and time.time() < deadline:
        finished = [n for n in json.loads(sess._wait({"agent_ids": [], "timeout_s": 5}, box["orch"]))["notifications"]
                    if n["type"] == "finished"]
    assert [n["final_message"]["rationale"] for n in finished] == ["second"]  # no lost message, one notification
    sess.shutdown()
