"""Multi-turn, tool-calling chat models (provider-neutral) for the agent harness.

A `Conversation` keeps the provider's *native* message history (as mini-swe-agent v2 does: "we use the model's
native output as the basis for the message"), so reasoning / thinking items round-trip untouched. Histories are
append-only; the only rewrite is `compact()`, which performs *simple compaction* - the whole history is
replaced by one user message holding the task and a summary, and nothing older (no earlier turns, no earlier
thinking) is replayed. That is the shape Anthropic recommends for client-side compaction with preserved thinking
and it is equally valid for OpenAI's stateless Responses input.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.base import LLMError, UsageMeter

# parallel agents share one tokens-per-minute limit: retry 429s (with the SDK's backoff) instead of failing
MAX_RETRIES = 10


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema (object); strict-mode compatible


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""
    error: str = ""  # set when the arguments could not be parsed


@dataclass
class ModelTurn:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0  # whole prompt, including cached / cache-write tokens
    output_tokens: int = 0
    cost: float = 0.0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    compacted: bool = False  # the provider compacted the context server-side during this call


@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False


class Conversation(ABC):
    def __init__(self, model: "ChatModel", system: str, tools: list[ToolSpec]):
        self.model = model
        self.system = system
        self.tools = tools
        self.last_input_tokens = 0
        self.pending_chars = 0  # characters appended since the last query (for context-size estimates)
        self.native_compact_threshold = 0

    def enable_native_compaction(self, threshold: int) -> bool:
        """Let the provider compact the context itself past `threshold` tokens. False if it cannot."""
        return False

    @property
    def context_tokens(self) -> int:
        return self.last_input_tokens + self.pending_chars // 4

    @abstractmethod
    def add_user(self, text: str) -> None: ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> None: ...

    @abstractmethod
    def _query(self, effort: str | None, max_tokens: int) -> ModelTurn: ...

    @abstractmethod
    def compact(self, message: str) -> None:
        """Simple compaction: replace the entire history with a single user message."""

    def query(self, effort: str | None = "medium", max_tokens: int = 16000) -> ModelTurn:
        self.model.meter.check()
        turn = self._query(effort, max_tokens)
        turn.cost = self.model.meter_record(turn.input_tokens, turn.output_tokens, turn.cached_tokens,
                                            turn.cache_write_tokens)
        self.last_input_tokens = turn.input_tokens
        self.pending_chars = len(turn.text) + sum(len(c.raw) for c in turn.tool_calls)
        return turn


class ChatModel(ABC):
    provider = "abstract"

    def __init__(self, model: str, meter: UsageMeter | None = None):
        self.model = model
        self.meter = meter or UsageMeter()

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def meter_record(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
                     cache_write_tokens: int = 0) -> float:
        return self.meter.record(self.model, input_tokens, output_tokens, cached_tokens=cached_tokens,
                                 cache_write_tokens=cache_write_tokens)

    @abstractmethod
    def start(self, system: str, tools: list[ToolSpec]) -> Conversation: ...

    def complete(self, system: str, user: str, effort: str | None = "low", max_tokens: int = 8000) -> str:
        """One-shot text completion (used for summaries)."""
        conv = self.start(system, [])
        conv.add_user(user)
        return conv.query(effort, max_tokens).text


def _parse_args(raw: str | dict) -> tuple[dict, str]:
    if isinstance(raw, dict):
        return raw, ""
    try:
        val = json.loads(raw or "{}")
        return (val, "") if isinstance(val, dict) else ({}, "arguments must be a JSON object")
    except json.JSONDecodeError as e:
        return {}, f"invalid JSON arguments: {e}"


# ---------------------------------------------------------------------------------------------- OpenAI
class OpenAIConversation(Conversation):
    def __init__(self, model: "OpenAIChat", system: str, tools: list[ToolSpec]):
        super().__init__(model, system, tools)
        self.items: list[dict] = []

    def add_user(self, text: str) -> None:
        self.items.append({"role": "user", "content": text})
        self.pending_chars += len(text)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.items.append({"type": "function_call_output", "call_id": r.call_id, "output": r.output})
            self.pending_chars += len(r.output)

    def compact(self, message: str) -> None:
        self.items = [{"role": "user", "content": message}]
        self.last_input_tokens, self.pending_chars = 0, len(message)

    def enable_native_compaction(self, threshold: int) -> bool:
        # Responses API server-side compaction: past the threshold the server compacts inside the call and returns
        # an opaque `compaction` item that carries prior state (reasoning included) into later turns
        self.native_compact_threshold = threshold
        return True

    def _query(self, effort, max_tokens) -> ModelTurn:
        m: OpenAIChat = self.model  # type: ignore[assignment]
        kwargs: dict[str, Any] = {"model": m.model, "instructions": self.system, "input": self.items,
                                  "max_output_tokens": max_tokens}
        if self.native_compact_threshold:
            kwargs["extra_body"] = {"context_management": [
                {"type": "compaction", "compact_threshold": self.native_compact_threshold}]}
        if self.tools:
            kwargs["tools"] = [{"type": "function", "name": t.name, "description": t.description,
                                "parameters": t.parameters, "strict": True} for t in self.tools]
        if effort and m.reasoning:
            kwargs["reasoning"] = {"effort": effort}
        try:
            resp = m.client.responses.create(**kwargs)
        except Exception as e:
            raise LLMError(f"{m.name} request failed: {type(e).__name__}: {str(e)[:300]}") from e
        if getattr(resp, "status", "completed") not in ("completed", None):
            raise LLMError(f"{m.name} response {resp.status}: {getattr(resp, 'incomplete_details', None)}")
        calls = []
        compacted = False
        for o in resp.output:
            if o.type == "compaction":
                # everything before the latest compaction item is carried by it: drop it to keep requests small
                self.items = []
                compacted = True
            self.items.append(o.model_dump(exclude_none=True))  # native items, reasoning included
            if o.type == "function_call":
                args, err = _parse_args(o.arguments)
                calls.append(ToolCall(o.call_id, o.name, args, o.arguments or "", err))
        u = resp.usage
        cached = int(getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0)
        return ModelTurn(resp.output_text or "", calls, int(u.input_tokens), int(u.output_tokens),
                         cached_tokens=cached, compacted=compacted)


class OpenAIChat(ChatModel):
    provider = "openai"

    def __init__(self, model: str = "gpt-6-luna", *, client: Any = None, reasoning: bool = True,
                 meter: UsageMeter | None = None):
        super().__init__(model, meter)
        if client is None:
            from openai import OpenAI

            client = OpenAI(max_retries=MAX_RETRIES)
        self.client = client
        self.reasoning = reasoning

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return OpenAIConversation(self, system, tools)


# ---------------------------------------------------------------------------------------------- Anthropic
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicConversation(Conversation):
    def __init__(self, model: "AnthropicChat", system: str, tools: list[ToolSpec]):
        super().__init__(model, system, tools)
        self.messages: list[dict] = []

    def _append_user_content(self, blocks: list) -> None:
        # consecutive user content is merged into one user message (the API requires alternation)
        if self.messages and self.messages[-1]["role"] == "user":
            prev = self.messages[-1]["content"]
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            self.messages[-1]["content"] = prev + blocks
        else:
            self.messages.append({"role": "user", "content": blocks})

    def add_user(self, text: str) -> None:
        self._append_user_content([{"type": "text", "text": text}])
        self.pending_chars += len(text)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        blocks = [{"type": "tool_result", "tool_use_id": r.call_id, "content": r.output, "is_error": r.is_error}
                  for r in results]
        # tool results must come first in their user message
        self.messages.append({"role": "user", "content": blocks})
        self.pending_chars += sum(len(r.output) for r in results)

    def compact(self, message: str) -> None:
        self.messages = [{"role": "user", "content": [{"type": "text", "text": message}]}]
        self.last_input_tokens, self.pending_chars = 0, len(message)

    def _query(self, effort, max_tokens) -> ModelTurn:
        m: AnthropicChat = self.model  # type: ignore[assignment]
        kwargs: dict[str, Any] = {
            "model": m.model, "max_tokens": max_tokens,
            "system": [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            "messages": self.messages,
        }
        if self.tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                               for t in self.tools]
        extra: dict[str, Any] = {"cache_control": {"type": "ephemeral"}}  # auto-cache the growing transcript
        if effort:
            extra["output_config"] = {"effort": effort}
        try:
            if m.fallbacks:
                extra["fallbacks"] = "default"
                resp = m.client.beta.messages.create(betas=[FALLBACK_BETA], extra_body=extra, **kwargs)
            else:
                resp = m.client.messages.create(extra_body=extra, **kwargs)
        except Exception as e:
            raise LLMError(f"{m.name} request failed: {type(e).__name__}: {str(e)[:300]}") from e
        if resp.stop_reason == "refusal":
            raise LLMError(f"{m.name} refused ({getattr(getattr(resp, 'stop_details', None), 'category', None)})")
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{m.name} hit max_tokens={max_tokens}")
        # append the assistant content unchanged (thinking blocks must round-trip byte-identically)
        self.messages.append({"role": "assistant", "content": resp.content})
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else {}, json.dumps(b.input))
                 for b in resp.content if getattr(b, "type", "") == "tool_use"]
        u = resp.usage
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        in_tok = int(getattr(u, "input_tokens", 0) or 0) + read + write
        return ModelTurn(text, calls, in_tok, int(u.output_tokens), cached_tokens=read, cache_write_tokens=write)


class AnthropicChat(ChatModel):
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", *, client: Any = None, fallbacks: bool = True,
                 meter: UsageMeter | None = None):
        super().__init__(model, meter)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=MAX_RETRIES)
        self.client = client
        self.fallbacks = fallbacks

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return AnthropicConversation(self, system, tools)


# ---------------------------------------------------------------------------------------------- scripted fake
@dataclass
class FakeView:
    """What a scripted model sees: the system prompt and a flat, provider-neutral transcript."""

    system: str
    tools: list[str]
    transcript: list[dict] = field(default_factory=list)  # {"role": user|assistant|tool, "text"/"calls"/...}

    @property
    def last(self) -> dict:
        return self.transcript[-1]

    def last_tool_outputs(self) -> list[str]:
        outs = []
        for e in reversed(self.transcript):
            if e["role"] != "tool":
                break
            outs.append(e["text"])
        return list(reversed(outs))


class FakeConversation(Conversation):
    def __init__(self, model: "FakeChat", system: str, tools: list[ToolSpec]):
        super().__init__(model, system, tools)
        self.view = FakeView(system, [t.name for t in tools])
        self.n = 0

    def add_user(self, text: str) -> None:
        self.view.transcript.append({"role": "user", "text": text})
        self.pending_chars += len(text)

    def add_tool_results(self, results: list[ToolResult]) -> None:
        for r in results:
            self.view.transcript.append({"role": "tool", "id": r.call_id, "text": r.output, "error": r.is_error})
            self.pending_chars += len(r.output)

    def compact(self, message: str) -> None:
        self.view.transcript = [{"role": "user", "text": message, "compacted": True}]
        self.last_input_tokens, self.pending_chars = 0, len(message)

    def _query(self, effort, max_tokens) -> ModelTurn:
        m: FakeChat = self.model  # type: ignore[assignment]
        out = m.policy(self.view)
        if isinstance(out, str):
            out = ModelTurn(out, [])
        for i, c in enumerate(out.tool_calls):
            if not c.id:
                c.id = f"call_{self.n}_{i}"
            c.raw = c.raw or json.dumps(c.arguments)
        self.n += 1
        self.view.transcript.append({"role": "assistant", "text": out.text,
                                     "calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                                               for c in out.tool_calls]})
        chars = len(self.system) + sum(len(json.dumps(e)) for e in self.view.transcript)
        out.input_tokens = out.input_tokens or chars // 4
        out.output_tokens = out.output_tokens or 20
        return out


class FakeChat(ChatModel):
    """Deterministic model driven by a Python policy: policy(view) -> ModelTurn | str."""

    provider = "fake"

    def __init__(self, policy: Callable[[FakeView], ModelTurn | str], model: str = "fake-chat",
                 meter: UsageMeter | None = None):
        super().__init__(model, meter)
        self.policy = policy

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return FakeConversation(self, system, tools)


def bash_call(command: str) -> ModelTurn:
    """Helper for scripted policies."""
    return ModelTurn("", [ToolCall("", "bash", {"command": command})])


def make_chat_model(provider: str, model: str | None = None, meter: UsageMeter | None = None) -> ChatModel:
    from ..llm import DEFAULT_MODELS

    provider = provider.lower()
    if provider == "openai":
        return OpenAIChat(model or DEFAULT_MODELS["openai"], meter=meter)
    if provider == "anthropic":
        return AnthropicChat(model or DEFAULT_MODELS["anthropic"], meter=meter)
    raise ValueError(f"no chat model for provider {provider!r}")
