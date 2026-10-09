"""LLM backends. Agents work without any LLM (heuristic mode); an LLM upgrades their judgement."""

from __future__ import annotations

import os

from .base import LLM, PRICES, BudgetExceeded, LLMError, ResponseCache, UsageMeter, price_for, validate

DEFAULT_MODELS = {"openai": "gpt-6-luna", "anthropic": "claude-opus-5-5"}
DEFAULT_BUDGET_USD = 2.0


def make_llm(
    provider: str | None = None,
    model: str | None = None,
    *,
    budget_usd: float | None = None,
    cache: ResponseCache | None = None,
    meter: UsageMeter | None = None,
) -> LLM | None:
    """Build an LLM client.

    provider: "none" | "openai" | "anthropic" | "auto" (default: $OSPM_LLM or "none").
    "auto" picks Anthropic if ANTHROPIC_API_KEY is set, else OpenAI if OPENAI_API_KEY is set, else none.
    budget_usd defaults to $OSPM_BUDGET_USD or 2.0 and is a hard per-process cap.
    """
    provider = (provider or os.environ.get("OSPM_LLM") or "none").lower()
    if provider == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "none"
    if provider in ("none", "off", ""):
        return None
    if budget_usd is None:
        budget_usd = float(os.environ.get("OSPM_BUDGET_USD", DEFAULT_BUDGET_USD))
    meter = meter or UsageMeter(budget_usd)
    model = model or os.environ.get("OSPM_MODEL") or DEFAULT_MODELS.get(provider)
    if provider == "openai":
        from .openai_llm import OpenAILLM

        return OpenAILLM(model, meter=meter, cache=cache)
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM(model, meter=meter, cache=cache)
    raise ValueError(f"unknown LLM provider {provider!r} (expected none|openai|anthropic|auto)")


__all__ = [
    "LLM",
    "LLMError",
    "BudgetExceeded",
    "UsageMeter",
    "ResponseCache",
    "PRICES",
    "price_for",
    "validate",
    "make_llm",
    "DEFAULT_MODELS",
]
