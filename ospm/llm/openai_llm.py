"""OpenAI backend (Responses API with strict JSON-schema output)."""

from __future__ import annotations

from typing import Any

from .base import LLM, LLMError, RawResult, ResponseCache, UsageMeter


class OpenAILLM(LLM):
    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        client: Any = None,
        reasoning: bool = True,
        meter: UsageMeter | None = None,
        cache: ResponseCache | None = None,
    ):
        super().__init__(model, meter, cache)
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - depends on environment
                raise LLMError("The 'openai' package is not installed: pip install 'ospm[openai]'") from e
            client = OpenAI(max_retries=10)  # reads OPENAI_API_KEY; patient on 429s (tokens-per-minute limits)
        self.client = client
        self.reasoning = reasoning

    def _call(self, system, user, schema, name, effort, max_tokens) -> RawResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
        }
        if self.reasoning and effort:
            kwargs["reasoning"] = {"effort": effort}
        try:
            resp = self.client.responses.create(**kwargs)
        except Exception as e:  # openai.APIError and transport errors; SDK already retried transient ones
            raise LLMError(f"{self.name} request failed: {type(e).__name__}: {str(e)[:300]}") from e
        status = getattr(resp, "status", "completed")
        if status != "completed":
            raise LLMError(f"{self.name} response {status}: {getattr(resp, 'incomplete_details', None)}")
        text = getattr(resp, "output_text", "") or ""
        if not text.strip():
            raise LLMError(f"{self.name} returned no text (possible refusal)")
        usage = getattr(resp, "usage", None)
        return RawResult(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_tokens=int(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0),
        )
