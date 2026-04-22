"""Independent-service adapter for AnyDex grasp planning."""

from __future__ import annotations

import os
from typing import Any

import httpx


class AnyDexBackendError(RuntimeError):
    """Raised when the external AnyDex backend cannot serve a request."""


class AnyDexBackendUnavailable(AnyDexBackendError):
    """Raised when no backend is configured."""


class AnyDexBackendClient:
    """Thin JSON client for an external AnyDex-inspired grasp planner."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 10.0) -> None:
        self.base_url = (base_url or os.getenv("ROBOCLAW_ANYDEX_BACKEND_URL") or "").strip().rstrip("/")
        timeout_raw = os.getenv("ROBOCLAW_ANYDEX_TIMEOUT_S")
        self.timeout_s = float(timeout_raw) if timeout_raw else float(timeout_s)

    async def plan_grasp(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Request grasp candidates from the configured AnyDex backend."""
        if not self.base_url:
            raise AnyDexBackendUnavailable(
                "ROBOCLAW_ANYDEX_BACKEND_URL is not configured."
            )
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(f"{self.base_url}/v1/grasp-plan", json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise AnyDexBackendError(
                f"AnyDex backend HTTP {exc.response.status_code}: {detail or exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AnyDexBackendError(f"AnyDex backend request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise AnyDexBackendError("AnyDex backend returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise AnyDexBackendError("AnyDex backend returned a non-object response.")
        return data
