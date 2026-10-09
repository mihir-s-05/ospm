"""Anthropic Claude backend (Messages API, structured outputs, server-side refusal fallbacks)."""

from __future__ import annotations

from typing import Any

from .base import LLM, LLMError, RawResult, ResponseCache, UsageMeter

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM(LLM):
    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        client: Any = None,
        fallbacks: bool = True,
        meter: UsageMeter | None = None,
        cache: ResponseCache | None = None,
    ):
        super().__init__(model, meter, cache)
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on environment
                raise LLMError("The 'anthropic' package is not installed: pip install 'ospm[anthropic]'") from e
            client = anthropic.Anthropic()  # ANTHROPIC_API_KEY / auth token / `ant auth login` profile
        self.client = client
        self.fallbacks = fallbacks

    def _call(self, system, user, schema, name, effort, max_tokens) -> RawResult:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        extra_body: dict[str, Any] = {"output_config": output_config}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            # The system prompt (instructions + playbook) is stable per genome, so cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
        }
        try:
            if self.fallbacks:
                # On a safety-classifier decline the API re-runs the request on Anthropic's
                # recommended fallback model inside the same call.
                extra_body["fallbacks"] = "default"
                resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], extra_body=extra_body, **kwargs)
            else:
                resp = self.client.messages.create(extra_body=extra_body, **kwargs)
        except Exception as e:  # anthropic.APIError hierarchy; SDK already retried 408/409/429/5xx
            raise LLMError(f"{self.name} request failed: {type(e).__name__}: {str(e)[:300]}") from e

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMError(f"{self.name} refused the request ({getattr(details, 'category', None)})")
        if stop == "max_tokens":
            raise LLMError(f"{self.name} hit max_tokens={max_tokens} before finishing")
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        in_tok = sum(
            int(getattr(usage, f, 0) or 0)
            for f in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return RawResult(text=text, input_tokens=in_tok, output_tokens=int(getattr(usage, "output_tokens", 0) or 0))
