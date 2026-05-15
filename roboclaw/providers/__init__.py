"""LLM provider abstraction module."""

from __future__ import annotations

# Lazy imports — OpenAICodexProvider depends on oauth_cli_kit (private package),
# so we don't force it at import time.
__all__ = ["LLMProvider", "LLMResponse", "LiteLLMProvider", "OpenAICodexProvider", "AzureOpenAIProvider"]


def __getattr__(name: str):
    if name in ("LLMProvider", "LLMResponse"):
        from roboclaw.providers.base import LLMProvider, LLMResponse  # noqa: PLC0415
        return {"LLMProvider": LLMProvider, "LLMResponse": LLMResponse}[name]
    if name == "LiteLLMProvider":
        from roboclaw.providers.litellm_provider import LiteLLMProvider  # noqa: PLC0415
        return LiteLLMProvider
    if name == "OpenAICodexProvider":
        from roboclaw.providers.openai_codex_provider import OpenAICodexProvider  # noqa: PLC0415
        return OpenAICodexProvider
    if name == "AzureOpenAIProvider":
        from roboclaw.providers.azure_openai_provider import AzureOpenAIProvider  # noqa: PLC0415
        return AzureOpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
