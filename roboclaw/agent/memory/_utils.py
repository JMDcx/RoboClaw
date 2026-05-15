"""Internal helpers for the personalized memory package."""

from __future__ import annotations


def _safe_user_id(user_id: str) -> str:
    safe = user_id.replace("/", "_").replace(":", "_")
    return safe[:64]
