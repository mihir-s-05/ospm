"""Deterministic in-process LLM for tests and offline demos."""

from __future__ import annotations

import json
from typing import Callable

from .base import LLM, RawResult, ResponseCache, UsageMeter

Responder = Callable[[str, str, dict], dict]


class FakeLLM(LLM):
    provider = "fake"

    def __init__(
        self,
        responder: Responder,
        model: str = "fake-model",
        meter: UsageMeter | None = None,
        cache: ResponseCache | None = None,
    ):
        super().__init__(model, meter, cache)
        self.responder = responder
        self.requests: list[dict] = []

    def _call(self, system, user, schema, name, effort, max_tokens) -> RawResult:
        self.requests.append({"system": system, "user": user, "schema": schema, "name": name, "effort": effort})
        out = self.responder(system, user, schema)
        return RawResult(text=json.dumps(out), input_tokens=(len(system) + len(user)) // 4, output_tokens=50)
