"""Run one repository event through the agent team (or through a single specialist)."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..harness.agent import Agent, AgentConfig, Tool, Trace
from ..harness.env import LocalEnvironment
from ..harness.models import ChatModel, ToolSpec
from .roles import TeamGenome, make_validator, parse_json, render_system, team_listing


def _spec(name: str, description: str, props: dict) -> ToolSpec:
    return ToolSpec(name, description, {"type": "object", "properties": props, "required": list(props),
                                        "additionalProperties": False})


_STR = {"type": "string"}


def spawn_tool(roles: list[str]) -> ToolSpec:
    return _spec(
        "spawn_agent", "Start a specialist agent in the background on a task and return immediately with its "
        "agent_id. Its final message arrives later as a `finished` notification (see wait_for_agents).",
        {"role": {"type": "string", "enum": list(roles)},
         "prompt": {"type": "string", "description": "complete, self-contained instructions for the specialist"}})


SEND_TOOL = _spec(
    "send_message", "Send a message to a child agent: delivered at its next step if it is running; if it has "
    "finished it is resumed with the message (keeping its context).",
    {"agent_id": _STR, "message": _STR})
WAIT_TOOL = _spec(
    "wait_for_agents", "Block until a notification (finished / message) from your agents is pending, none of them "
    "is running any more, or the timeout passes. Returns and consumes all your pending notifications.",
    {"agent_ids": {"type": "array", "items": _STR, "description": "agents to wait for; empty = any of your agents"},
     "timeout_s": {"type": "integer", "description": "seconds to wait; 0 = default (300); at most 900"}})
CLOSE_TOOL = _spec("close_agent", "Stop a child agent for good (it cannot be resumed).", {"agent_id": _STR})
PARENT_TOOL = _spec(
    "message_parent", "Send a short non-blocking message to the agent that spawned you (e.g. an early finding or a "
    "question). Your final submission is delivered automatically when you finish.", {"message": _STR})

REPORT_NOTE = (
    "\n\n---\nYour final submission is the only thing the agent that spawned you will see (not your steps). Besides "
    "the output contract, include a top-level \"report\" field: what you checked (files, commands), the evidence "
    "for each finding, what you could not verify, and your confidence.")
WAIT_DEFAULT, WAIT_MAX = 300, 900


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


@dataclass
class Episode:
    task_id: str
    entry_role: str
    output: dict | None
    exit_status: str
    trace: Trace
    roles_used: set[str] = field(default_factory=set)
    role_hashes: dict[str, str] = field(default_factory=dict)  # genome hashes of the roles that ran

    @property
    def cost(self) -> float:
        return self.trace.total_cost()

    @property
    def steps(self) -> int:
        return self.trace.total_steps()


class TeamSession:
    """One event: the entry agent plus every specialist it spawns, sharing one workspace.

    Specialists run in background threads. Notifications for a parent (children's `finished` / `message`) sit in
    one queue per parent, consumed by `wait_for_agents` or delivered automatically before the parent's next step.
    """

    def __init__(self, genome: TeamGenome, model: ChatModel, env_factory: Callable[[], LocalEnvironment],
                 project: str, project_context: str, log: Callable[[str], None] | None = None,
                 max_specialists: int = 6, role_context: dict[str, str] | None = None):
        self.genome = genome
        self.role_context = role_context or {}  # appended to a spawned specialist's task (e.g. its skill index)
        self.model = model
        self.env_factory = env_factory
        self.project = project
        self.project_context = project_context
        self.log = log or (lambda m: None)
        self.max_specialists = max_specialists
        self.agents: dict[str, Agent] = {}
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._queues: dict[str, list[dict]] = {}  # parent id -> pending notifications
        self._running: set[str] = set()
        self._threads: list[threading.Thread] = []
        self.roles_used: set[str] = set()

    def _config(self, role: str) -> AgentConfig:
        g = self.genome.roles[role]
        return AgentConfig(system_template=render_system(g, self.project, self.project_context,
                                                     team_listing(self.genome) if role == "orchestrator" else ""),
                           instance_template=g.instance_template, step_limit=g.step_limit,
                           compaction_tokens=g.compaction_tokens, effort=g.effort)

    def make_agent(self, role: str, parent: Agent | None = None) -> Agent:
        tools = ([Tool(spawn_tool(self.genome.specialists()), self._spawn), Tool(SEND_TOOL, self._send),
                  Tool(WAIT_TOOL, self._wait), Tool(CLOSE_TOOL, self._close)] if role == "orchestrator" else [])
        if parent is not None:
            tools.append(Tool(PARENT_TOOL, self._message_parent))
        agent = Agent(role, self.model, self.env_factory(), self._config(role), tools=tools, parent=parent,
                      log=self.log)
        agent.validate = self._validator(agent)
        agent.pull = lambda: [f"<notification>{json.dumps(n)}</notification>" for n in self._pop(agent.id)]
        with self._lock:
            self.agents[agent.id] = agent
            self.roles_used.add(role)
        return agent

    def _validator(self, agent: Agent) -> Callable[[str], str | None]:
        base = make_validator(agent.role, self.genome.roles[agent.role].contract)

        def check(submission: str) -> str | None:
            with self._lock:
                busy = [a.id for a in self.agents.values() if a.parent is agent and a.id in self._running]
            if busy:
                return (f"{len(busy)} agents are still running ({', '.join(busy)}): "
                        "wait_for_agents or close_agent first")
            if (problem := base(submission)):
                return problem
            if agent.parent is not None and not (parse_json(submission) or {}).get("report"):
                return ('a top-level "report" field is required (what you checked, evidence per finding, what you '
                        "could not verify, confidence): your parent sees nothing but this submission")
            return None

        return check

    # ------------------------------------------------------------------ notifications
    def _notify(self, parent_id: str, note: dict) -> None:
        with self._cond:
            self._queues.setdefault(parent_id, []).append(note)
            self._cond.notify_all()

    def _pop(self, parent_id: str) -> list[dict]:
        with self._lock:
            return self._queues.pop(parent_id, [])

    def _message_parent(self, args: dict[str, Any], caller: Agent) -> str:
        self._notify(caller.parent.id, {"type": "message", "from": caller.id, "role": caller.role,
                                        "message": str(args.get("message", ""))})
        return "sent"

    # ------------------------------------------------------------------ background runs
    def _start(self, agent: Agent, fn: Callable[[], Any]) -> None:
        """Run `fn` (agent.run / agent.send) in a thread; call with the lock held."""
        self._running.add(agent.id)

        def target() -> None:
            try:
                res = fn()
                status, sub = res.exit_status, res.submission
            except Exception as e:  # keep the parent informed instead of losing the thread silently
                status, sub = f"Error: {type(e).__name__}: {str(e)[:300]}", ""
                agent.trace.exit_status = status
            with self._cond:
                self._running.discard(agent.id)
                late = [] if agent.closed else agent.take_inbox()
                if late:  # messages that arrived while it was submitting: continue with them instead of finishing
                    self._start(agent, lambda: agent.send("\n\n".join(late)))
                    return
                if agent.parent is not None and not agent.closed:
                    parsed = parse_json(sub) if sub else None
                    self._queues.setdefault(agent.parent.id, []).append(
                        {"type": "finished", "agent_id": agent.id, "role": agent.role, "exit_status": status,
                         "final_message": parsed if parsed is not None else sub})
                self._cond.notify_all()

        t = threading.Thread(target=target, name=agent.id, daemon=True)
        self._threads.append(t)
        t.start()

    def _child(self, agent_id: str, caller: Agent) -> Agent | str:
        agent = self.agents.get(agent_id)
        if agent is None or agent.parent is not caller:
            return _err(f"unknown agent_id {agent_id!r}: it is not one of your agents")
        return agent

    # ------------------------------------------------------------------ orchestrator tools
    def _spawn(self, args: dict[str, Any], caller: Agent) -> str:
        role, prompt = args.get("role"), str(args.get("prompt", ""))
        if role not in self.genome.specialists():
            return _err(f"unknown role {role!r}")
        with self._lock:
            if sum(1 for a in self.agents.values() if a.parent is caller) >= self.max_specialists:
                return _err(f"specialist limit ({self.max_specialists}) reached; message an existing agent instead")
            agent = self.make_agent(role, parent=caller)
            caller.trace.children.append(agent.trace)
            task = prompt + self.role_context.get(role, "") + REPORT_NOTE
            self._start(agent, lambda: agent.run(task=task))
        return json.dumps({"agent_id": agent.id, "role": role, "status": "running"})

    def _send(self, args: dict[str, Any], caller: Agent) -> str:
        message = f"<message from=\"{caller.id}\">\n{args.get('message', '')}\n</message>"
        with self._lock:
            agent = self._child(str(args.get("agent_id") or ""), caller)
            if isinstance(agent, str):
                return agent
            if agent.closed:
                return _err(f"agent {agent.id} is closed")
            if agent.id in self._running:
                agent.post(message)
                status = "delivered"
            else:
                self._start(agent, lambda: agent.send(message))
                status = "resumed"
        return json.dumps({"agent_id": agent.id, "status": status})

    def _wait(self, args: dict[str, Any], caller: Agent) -> str:
        ids = {str(i) for i in args.get("agent_ids") or []}
        timeout = min(int(args.get("timeout_s") or WAIT_DEFAULT), WAIT_MAX)
        deadline = time.monotonic() + timeout
        with self._cond:
            for i in ids:
                if isinstance(bad := self._child(i, caller), str):
                    return bad
            watched = ids or {a.id for a in self.agents.values() if a.parent is caller}
            while True:
                ready = any(not ids or n.get("agent_id", n.get("from")) in ids
                            for n in self._queues.get(caller.id, []))
                live = sorted(watched & self._running)
                left = deadline - time.monotonic()
                if ready or not live or left <= 0 or caller.closed:
                    break
                self._cond.wait(left)
            notes = self._queues.pop(caller.id, [])
        return json.dumps({"notifications": notes, "still_running": live})

    def _close(self, args: dict[str, Any], caller: Agent) -> str:
        with self._lock:
            agent = self._child(str(args.get("agent_id") or ""), caller)
            if isinstance(agent, str):
                return agent
            agent.close()
        return json.dumps({"agent_id": agent.id, "status": "closed"})

    def shutdown(self, join_timeout: float = 30.0) -> None:
        """End of the episode: close every agent still running and wait (bounded) for their threads."""
        with self._lock:
            for aid in list(self._running):
                self.agents[aid].close()
            threads = list(self._threads)
        deadline = time.monotonic() + join_timeout
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))


def run_episode(genome: TeamGenome, model: ChatModel, env_factory: Callable[[], LocalEnvironment], *,
                task_id: str, entry_role: str, variables: dict[str, Any], project: str, project_context: str,
                log: Callable[[str], None] | None = None, role_context: dict[str, str] | None = None) -> Episode:
    session = TeamSession(genome, model, env_factory, project, project_context, log, role_context=role_context)
    agent = session.make_agent(entry_role)
    try:
        res = agent.run(**variables)
        session.shutdown()
    finally:
        for a in list(session.agents.values()):
            a.env.cleanup()
    output = parse_json(res.submission) if res.submission else None
    return Episode(task_id, entry_role, output, res.exit_status, agent.trace, set(session.roles_used),
                   {r: genome.roles[r].content_hash() for r in session.roles_used})
