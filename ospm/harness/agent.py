"""A mini-swe-agent-v2-style agent: a linear history, one `bash` tool, independent subprocess actions.

Protocol (as in mini-swe-agent v2): the model acts through the native `bash` tool (or, in text mode, a single
```mswea_bash_command``` block); each command runs in a fresh subprocess; the agent finishes by running a command
whose output starts with the line ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` - everything after it is the
submission. Extensions for the multi-agent team:

* extra native tools (the orchestrator's ``spawn_agent`` etc.), executed concurrently when called in the same turn;
* ``send()`` continues a finished agent with a follow-up message; ``post()`` queues a message for a running agent
  (delivered before its next step); ``close()`` stops it;
* simple context compaction when the context grows past ``compaction_tokens``;
* hierarchical traces (sub-agent runs are children of the agent that spawned them).
"""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..llm.base import BudgetExceeded, LLMError
from .env import SUBMIT_MARKER, ExecResult, LocalEnvironment, extract_submission, format_observation
from .models import ChatModel, Conversation, ToolCall, ToolResult, ToolSpec

BASH_TOOL = ToolSpec(
    "bash",
    "Execute a bash command in the task workspace. Every call runs in a new subprocess: `cd` and environment "
    "variables do not persist between calls (chain them with && in one command). To finish, run a command whose "
    f"output starts with the line {SUBMIT_MARKER} followed by your final output.",
    {"type": "object", "properties": {"command": {"type": "string", "description": "the bash command"}},
     "required": ["command"], "additionalProperties": False},
)
_TEXT_ACTION = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)

FORMAT_ERROR_TOOLS = (
    "Your last response did not call any tool. Every response must call the `bash` tool (or another available "
    f"tool). If you are done, submit with: echo {SUBMIT_MARKER} && cat <<'EOF'\\n<final output>\\nEOF"
)
FORMAT_ERROR_TEXT = (
    "Your last response must contain exactly ONE ```mswea_bash_command``` code block with the command to run. "
    f"To finish, run a command whose output starts with {SUBMIT_MARKER}."
)
STEPS_LEFT_NOTE = (
    "Note: you have {n} steps left. Use them to confirm your most important findings, then submit (a run that "
    f"ends without a submission produces no output at all). Submit with a command whose output starts with "
    f"{SUBMIT_MARKER}."
)
FINAL_TURN = (
    "You have reached the step limit. This is your last turn: submit your final output now, based on what you have "
    f"found so far, with a command whose output starts with {SUBMIT_MARKER}. No other command will run."
)
SUMMARY_SYSTEM = (
    "You compress an agent's working transcript so it can continue without the full history. Write a dense, "
    "factual summary that preserves: the task goal; every concrete finding with evidence (file paths, line "
    "numbers, issue/PR numbers, test names, error lines, command results); what was ruled out; decisions made; "
    "open questions; and the immediate next steps. No preamble."
)


class InterruptAgentFlow(Exception):
    pass


class Submitted(InterruptAgentFlow):
    def __init__(self, submission: str):
        super().__init__(submission)
        self.submission = submission


class LimitsExceeded(InterruptAgentFlow):
    pass


@dataclass
class AgentConfig:
    system_template: str
    instance_template: str = "{task}"
    step_limit: int = 30
    cost_limit: float = 0.25  # USD for this agent's own model calls
    wall_time_limit: int = 1500
    # well inside gpt-6-luna's 272k short-context pricing tier; the repeated prefix is mostly cache reads
    compaction_tokens: int = 128_000
    native_compaction: bool = True  # use the provider's server-side compaction where available
    observation_max_chars: int = 8000
    command_timeout: int = 180
    effort: str | None = "medium"
    max_tokens: int = 16000
    text_mode: bool = False


@dataclass
class Step:
    n: int
    text: str
    calls: list[dict]
    results: list[dict]
    input_tokens: int
    output_tokens: int
    cost: float
    t: float
    event: str = ""  # "compaction" | "format_error" | "final_turn" | "message" (delivered inbox text in results) | ""


@dataclass
class Trace:
    agent_id: str
    role: str
    task: str
    parent: str = ""
    steps: list[Step] = field(default_factory=list)
    children: list["Trace"] = field(default_factory=list)
    submission: str = ""
    exit_status: str = ""
    cost: float = 0.0
    compactions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "role": self.role, "task": self.task, "parent": self.parent,
                "steps": [asdict(s) for s in self.steps], "children": [c.to_dict() for c in self.children],
                "submission": self.submission, "exit_status": self.exit_status, "cost": round(self.cost, 6),
                "compactions": self.compactions}

    def total_cost(self) -> float:
        return self.cost + sum(c.total_cost() for c in self.children)

    def total_steps(self) -> int:
        return len(self.steps) + sum(c.total_steps() for c in self.children)

    def roles(self) -> set[str]:
        out = {self.role}
        for c in self.children:
            out |= c.roles()
        return out

    def render(self, max_obs: int = 600, depth: int = 0) -> str:
        """Compact human/LLM-readable rendering (used for reflection and summaries)."""
        pad = "  " * depth
        lines = [f"{pad}## agent {self.agent_id} ({self.role}) exit={self.exit_status} cost=${self.cost:.4f}",
                 f"{pad}task: {self.task[:500]}"]
        for s in self.steps:
            if s.event == "compaction":
                lines.append(f"{pad}[step {s.n}] (context compacted)")
                continue
            if s.event == "message":
                lines.append(f"{pad}[step {s.n}] message: {s.results[0]['output'][:600]}")
                continue
            if s.text.strip():
                lines.append(f"{pad}[step {s.n}] thought: {s.text.strip()[:400]}")
            for c, r in itertools.zip_longest(s.calls, s.results, fillvalue={}):
                arg = c.get("arguments", {})
                shown = arg.get("command") if isinstance(arg, dict) and "command" in arg else json.dumps(arg)[:400]
                lines.append(f"{pad}[step {s.n}] {c.get('name', '?')}: {str(shown)[:400]}")
                out = str(r.get("output", ""))
                if out:
                    lines.append(f"{pad}  -> {out[:max_obs].replace(chr(10), chr(10) + pad + '     ')}")
        for child in self.children:
            lines.append(child.render(max_obs, depth + 1))
        if self.submission:
            lines.append(f"{pad}submission: {self.submission[:1500]}")
        return "\n".join(lines)


@dataclass
class AgentResult:
    agent_id: str
    submission: str
    exit_status: str
    steps: int
    cost: float


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[[dict[str, Any], "Agent"], str]
    parallel: bool = True  # may run concurrently with other calls of the same turn


def render(template: str, variables: dict[str, Any]) -> str:
    """Replace `{name}` placeholders for the given names only (JSON braces in templates are left alone)."""
    out = template
    for k, v in variables.items():
        out = out.replace("{" + k + "}", str(v))
    return out


_ids = itertools.count(1)
_ids_lock = threading.Lock()


def new_agent_id(role: str) -> str:
    with _ids_lock:
        return f"{role}-{next(_ids)}"


class Agent:
    def __init__(self, role: str, model: ChatModel, env: LocalEnvironment, config: AgentConfig, *,
                 tools: list[Tool] = (), validate: Callable[[str], str | None] | None = None,
                 parent: "Agent | None" = None, log: Callable[[str], None] | None = None):
        self.role = role
        self.id = new_agent_id(role)
        self.model = model
        self.env = env
        self.cfg = config
        self.tools = {t.spec.name: t for t in tools}
        self.validate = validate
        self.parent = parent
        self.log = log or (lambda m: None)
        self.conv: Conversation | None = None
        self._native_compaction = False
        self.trace = Trace(self.id, role, "", parent.id if parent else "")
        self.cost = 0.0
        self.n_steps = 0
        self._task_message = ""
        self._start = 0.0
        self.pull: Callable[[], list[str]] | None = None  # extra inbox source, drained with the posted messages
        self._inbox: list[str] = []
        self._inbox_lock = threading.Lock()
        self._closed = threading.Event()

    # ------------------------------------------------------------------ public API
    def run(self, **template_vars: Any) -> AgentResult:
        self._task_message = render(self.cfg.instance_template, template_vars)
        self.trace.task = self._task_message
        specs = [] if self.cfg.text_mode else [BASH_TOOL] + [t.spec for t in self.tools.values()]
        self.conv = self.model.start(render(self.cfg.system_template, template_vars), specs)
        self._native_compaction = (self.cfg.native_compaction and self.cfg.compaction_tokens > 0
                                   and self.conv.enable_native_compaction(self.cfg.compaction_tokens))
        self.conv.add_user(self._task_message)
        return self._loop()

    def send(self, message: str) -> AgentResult:
        """Follow-up after a submission: the same conversation continues until the next submission."""
        if self.conv is None:
            raise RuntimeError("agent has not been started")
        self.conv.add_user(message)
        self.trace.submission, self.trace.exit_status = "", ""
        return self._loop()

    def post(self, text: str) -> None:
        """Queue a message for the running agent; it is added to the conversation before the next step."""
        with self._inbox_lock:
            self._inbox.append(text)

    def close(self) -> None:
        """Stop the agent at its next iteration (exit_status "Closed"); a finished agent is just marked closed."""
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def take_inbox(self) -> list[str]:
        """Remove and return the queued messages (e.g. ones that arrived while the agent was submitting)."""
        with self._inbox_lock:
            msgs, self._inbox = self._inbox, []
        return msgs

    def _drain_inbox(self) -> None:
        msgs = self.take_inbox()
        if self.pull:
            msgs += self.pull()
        for m in msgs:
            self.conv.add_user(m)
            self.trace.steps.append(Step(self.n_steps, "", [], [{"id": "", "output": m, "is_error": False}],
                                         0, 0, 0.0, time.time(), event="message"))

    # ------------------------------------------------------------------ loop
    def _loop(self) -> AgentResult:
        self._start = time.time()
        steps_at_start = self.n_steps
        warned = False
        while True:
            if self._closed.is_set():
                return self._finish("", "Closed")
            try:
                self._drain_inbox()
                self._check_limits(steps_at_start)
                left = self.cfg.step_limit - (self.n_steps - steps_at_start)
                if 0 < self.cfg.step_limit and left <= 3 and not warned and self.cfg.step_limit > 6:
                    warned = True
                    self.conv.add_user(STEPS_LEFT_NOTE.format(n=left))
                self._maybe_compact()
                self._step()
            except Submitted as s:
                return self._finish(s.submission, "Submitted")
            except LimitsExceeded as e:
                if not str(e).startswith("step limit"):
                    return self._finish("", f"LimitsExceeded: {e}")
                return self._final_turn(str(e))
            except (LLMError, BudgetExceeded) as e:
                return self._finish("", f"ModelError: {str(e)[:300]}")

    def _final_turn(self, reason: str) -> AgentResult:
        """At the step limit the agent gets one last turn to submit what it has, instead of losing all its work."""
        assert self.conv is not None
        self.conv.add_user(FINAL_TURN)
        self.trace.steps.append(Step(self.n_steps, "", [], [], 0, 0, 0.0, time.time(), event="final_turn"))
        try:
            self._step()
        except Submitted as s:
            return self._finish(s.submission, "Submitted")
        except (LLMError, BudgetExceeded) as e:
            return self._finish("", f"ModelError: {str(e)[:300]}")
        except InterruptAgentFlow:
            pass
        last = self.trace.steps[-1] if self.trace.steps else None
        if last is not None and last.event == "format_error" and last.text.strip():
            # no tool call on the last turn: models often just write the answer, which is what was asked for
            return self._finish(extract_submission(last.text) or last.text.strip(), "Submitted")
        return self._finish("", f"LimitsExceeded: {reason}")

    def _finish(self, submission: str, status: str) -> AgentResult:
        self.trace.submission, self.trace.exit_status, self.trace.cost = submission, status, self.cost
        self.log(f"[{self.id}] {status} after {self.n_steps} steps (${self.cost:.4f})")
        return AgentResult(self.id, submission, status, self.n_steps, self.cost)

    def _check_limits(self, steps_at_start: int) -> None:
        if 0 < self.cfg.step_limit <= self.n_steps - steps_at_start:
            raise LimitsExceeded(f"step limit {self.cfg.step_limit}")
        if 0 < self.cfg.cost_limit <= self.cost:
            raise LimitsExceeded(f"cost limit ${self.cfg.cost_limit}")
        if 0 < self.cfg.wall_time_limit <= time.time() - self._start:
            raise LimitsExceeded(f"wall time {self.cfg.wall_time_limit}s")

    def _maybe_compact(self) -> None:
        assert self.conv is not None
        if (self._native_compaction or self.cfg.compaction_tokens <= 0
                or self.conv.context_tokens < self.cfg.compaction_tokens):
            return
        transcript = self.trace.render(max_obs=1500)
        summary = self.model.complete(SUMMARY_SYSTEM, f"Transcript to compress:\n\n{transcript[-400_000:]}",
                                      effort="low")
        last_obs = ""
        for s in reversed(self.trace.steps):
            if s.results:
                last_obs = "\n".join(str(r.get("output", ""))[:3000] for r in s.results)
                break
        message = (f"{self._task_message}\n\n<progress_summary>\n{summary.strip()}\n</progress_summary>\n\n"
                   f"<last_observation>\n{last_obs}\n</last_observation>\n\nThe context was compacted: the summary "
                   "above replaces your earlier turns. Continue the task from where it leaves off.")
        self.conv.compact(message)
        self.trace.compactions += 1
        self.trace.steps.append(Step(self.n_steps, "", [], [], 0, 0, 0.0, time.time(), event="compaction"))
        self.log(f"[{self.id}] context compacted")

    def _step(self) -> None:
        assert self.conv is not None
        turn = self.conv.query(self.cfg.effort, self.cfg.max_tokens)
        self.n_steps += 1
        self.cost += turn.cost
        if turn.compacted:  # server-side compaction happened inside this call
            self.trace.compactions += 1
            self.trace.steps.append(Step(self.n_steps, "", [], [], 0, 0, 0.0, time.time(), event="compaction"))
            self.log(f"[{self.id}] context compacted (server-side)")
        calls = turn.tool_calls
        if self.cfg.text_mode:
            actions = _TEXT_ACTION.findall(turn.text)
            if len(actions) != 1:
                self._record(turn, [], [], "format_error")
                self.conv.add_user(FORMAT_ERROR_TEXT)
                return
            calls = [ToolCall(f"text-{self.n_steps}", "bash", {"command": actions[0]})]
        elif not calls:
            self._record(turn, [], [], "format_error")
            self.conv.add_user(FORMAT_ERROR_TOOLS)
            return

        results: list[ToolResult | None] = [None] * len(calls)
        submission: str | None = None
        parallel = [(i, c) for i, c in enumerate(calls) if c.name in self.tools and self.tools[c.name].parallel]
        if len(parallel) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(parallel))) as pool:
                futs = {i: pool.submit(self._run_tool, c) for i, c in parallel}
                for i, f in futs.items():
                    results[i] = ToolResult(calls[i].id, f.result())
        for i, c in enumerate(calls):
            if results[i] is not None:
                continue
            if c.error:
                results[i] = ToolResult(c.id, f"Error: {c.error}", True)
            elif c.name == "bash":
                cmd = str(c.arguments.get("command", ""))
                res = self.env.execute(cmd, self.cfg.command_timeout)
                sub = extract_submission(res.output)
                if sub is not None:
                    problem = self.validate(sub) if self.validate else None
                    if problem:
                        results[i] = ToolResult(c.id, f"Submission rejected: {problem}\nFix it and submit again.",
                                                True)
                    else:
                        submission = sub
                        results[i] = ToolResult(c.id, "Submission received.")
                else:
                    results[i] = ToolResult(c.id, format_observation(res, self.cfg.observation_max_chars))
            elif c.name in self.tools:
                results[i] = ToolResult(c.id, self._run_tool(c))
            else:
                results[i] = ToolResult(c.id, f"Error: unknown tool {c.name!r}", True)
        final = [r for r in results if r is not None]
        self._record(turn, calls, final, "")
        if self.cfg.text_mode:
            self.conv.add_user("\n\n".join(r.output for r in final))
        else:
            self.conv.add_tool_results(final)
        if submission is not None:
            raise Submitted(submission)

    def _run_tool(self, call: ToolCall) -> str:
        try:
            return self.tools[call.name].fn(call.arguments, self)
        except Exception as e:  # a failing tool is reported to the model, not fatal
            return f"Error running {call.name}: {type(e).__name__}: {e}"

    def _record(self, turn, calls, results, event) -> None:
        self.trace.steps.append(Step(
            self.n_steps, turn.text, [{"name": c.name, "arguments": c.arguments, "id": c.id} for c in calls],
            [{"id": r.call_id, "output": r.output[:20000], "is_error": r.is_error} for r in results],
            turn.input_tokens, turn.output_tokens, round(turn.cost, 6), time.time(), event))


def run_command(env: LocalEnvironment, command: str) -> ExecResult:  # convenience for tests/tools
    return env.execute(command)
