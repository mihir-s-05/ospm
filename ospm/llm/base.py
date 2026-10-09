"""Provider-neutral LLM interface with JSON-schema outputs, response caching and a hard spend cap."""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Any failure to obtain a valid structured response."""


class BudgetExceeded(LLMError):
    """Raised before a call that would start once the configured USD budget is spent."""


# USD per 1M tokens: (input, output). Unknown models use a deliberately conservative default.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.0, 10.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
DEFAULT_PRICE = (10.0, 50.0)


# USD per 1M tokens: (cached input read, cache write). OpenAI caches automatically and charges no write premium;
# Anthropic charges 1.25x input for a (5-minute) cache write. Unknown models: no caching discount.
CACHE_PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.01, 0.10),
    "gpt-6.1-sol": (0.10, 2.0),
    "claude-fable-5-1": (1.0, 12.5),
    "claude-opus-5-5": (0.40, 5.0),
    "claude-sonnet-5-5": (0.20, 2.5),
    "claude-haiku-5-5": (0.01, 0.125),
}


def price_for(model: str) -> tuple[float, float]:
    return PRICES.get(model, DEFAULT_PRICE)


def call_cost(model: str, input_tokens: int, output_tokens: int, *, cached_tokens: int = 0,
              cache_write_tokens: int = 0, discount: float = 1.0) -> float:
    """USD for one call. `input_tokens` is the total prompt size, including the cached and cache-write parts."""
    pin, pout = price_for(model)
    pread, pwrite = CACHE_PRICES.get(model, (pin, pin))
    plain = max(0, input_tokens - cached_tokens - cache_write_tokens)
    return discount * (plain * pin + cached_tokens * pread + cache_write_tokens * pwrite + output_tokens * pout) / 1e6


class ResponseCache(Protocol):
    def get(self, key: str) -> dict | None: ...
    def put(self, key: str, value: dict) -> None: ...


class UsageMeter:
    """Thread-safe token/cost accounting shared by every LLM call in a process.

    The budget check happens *before* each call, so concurrent in-flight calls can overshoot the
    budget by at most (workers x cost of one call).
    """

    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = budget_usd
        self.calls = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.errors = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.budget_usd is not None and self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"LLM budget of ${self.budget_usd:.2f} exhausted (spent ${self.cost_usd:.4f}). "
                    "Raise OSPM_BUDGET_USD / --budget to continue."
                )

    def record(self, model: str, input_tokens: int, output_tokens: int, discount: float = 1.0, *,
               cached_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        """Record one call and return its cost. `discount` scales the list price (0.5 for the OpenAI Batch API)."""
        cost = call_cost(model, input_tokens, output_tokens, cached_tokens=cached_tokens,
                         cache_write_tokens=cache_write_tokens, discount=discount)
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.cached_tokens += cached_tokens
            self.output_tokens += output_tokens
            self.cost_usd += cost
        return cost

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def record_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "errors": self.errors,
                "input_tokens": self.input_tokens,
                "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "budget_usd": self.budget_usd,
            }


@dataclass
class RawResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


class LLM(ABC):
    provider: str = "abstract"

    def __init__(self, model: str, meter: UsageMeter | None = None, cache: ResponseCache | None = None):
        self.model = model
        self.meter = meter or UsageMeter()
        self.cache = cache

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @abstractmethod
    def _call(
        self, system: str, user: str, schema: dict, name: str, effort: str | None, max_tokens: int
    ) -> RawResult: ...

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        name: str = "output",
        effort: str | None = "medium",
        max_tokens: int = 16000,
    ) -> dict[str, Any]:
        """Return a dict that validates against `schema`, or raise LLMError."""
        key = hashlib.sha256(
            json.dumps([self.provider, self.model, system, user, schema, effort], sort_keys=True).encode()
        ).hexdigest()
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.meter.record_cache_hit()
                return hit
        self.meter.check()
        try:
            raw = self._call(system, user, schema, name, effort, max_tokens)
        except LLMError:
            self.meter.record_error()
            raise
        self.meter.record(self.model, raw.input_tokens, raw.output_tokens, cached_tokens=raw.cached_tokens)
        try:
            data = json.loads(raw.text)
        except json.JSONDecodeError as e:
            self.meter.record_error()
            raise LLMError(f"{self.name} returned non-JSON output: {raw.text[:200]!r}") from e
        problems = validate(data, schema)
        if problems:
            self.meter.record_error()
            raise LLMError(f"{self.name} output failed schema validation: {problems[:3]}")
        if self.cache is not None:
            self.cache.put(key, data)
        return data


def validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-schema validator covering the subset this project emits."""
    errs: list[str] = []
    typ = schema.get("type")
    types = typ if isinstance(typ, list) else [typ] if typ else []
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    if types and not any(checks[t](value) for t in types if t in checks):
        return [f"{path}: expected {typ}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in enum")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errs.append(f"{path}: missing {req!r}")
        if schema.get("additionalProperties") is False:
            for k in value:
                if k not in props:
                    errs.append(f"{path}: unexpected property {k!r}")
        for k, sub in props.items():
            if k in value:
                errs.extend(validate(value[k], sub, f"{path}.{k}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errs.extend(validate(item, schema["items"], f"{path}[{i}]"))
    return errs
