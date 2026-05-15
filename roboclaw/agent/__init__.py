"""Agent core module."""

from __future__ import annotations

# Lazy imports so that importing sub-modules (e.g. roboclaw.agent.tools.*)
# does not force the full AgentLoop / provider dependency chain.
__all__ = ["AgentLoop", "ContextBuilder", "MemoryStore", "SkillsLoader"]


def __getattr__(name: str):
    if name == "AgentLoop":
        from roboclaw.agent.loop import AgentLoop
        return AgentLoop
    if name == "ContextBuilder":
        from roboclaw.agent.context import ContextBuilder
        return ContextBuilder
    if name == "MemoryStore":
        from roboclaw.agent.memory import MemoryStore
        return MemoryStore
    if name == "SkillsLoader":
        from roboclaw.agent.skills import SkillsLoader
        return SkillsLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
