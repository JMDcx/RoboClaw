"""Adapter protocols for embodied execution."""

from __future__ import annotations

from typing import Any, Protocol


class EmbodiedAdapter(Protocol):
    """Execution adapter that binds assemblies to real or simulated carriers."""

    adapter_id: str
    assembly_id: str

    def probe_env(self) -> dict[str, Any]:
        """Inspect the execution environment without mutating state."""

    def check_dependencies(self) -> dict[str, Any]:
        """Check dependencies declared by the adapter lifecycle contract."""

    async def connect(self, *, target_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Connect to one execution target."""

    async def disconnect(self) -> dict[str, Any]:
        """Disconnect from the active target."""

    async def ready(self) -> dict[str, Any]:
        """Return readiness state for command execution."""

    async def health(self) -> dict[str, Any]:
        """Return adapter health mode, including degraded state when applicable."""

    async def check_compatibility(self) -> dict[str, Any]:
        """Validate runtime component versions against compatibility constraints."""

    async def stop(self, *, scope: str = "all") -> dict[str, Any]:
        """Stop active tasks or motion."""

    async def reset(self, *, mode: str = "home") -> dict[str, Any]:
        """Reset adapter state to a known mode."""

    async def recover(self, *, strategy: str | None = None) -> dict[str, Any]:
        """Apply recovery strategy after failures."""

    async def get_state(self) -> dict[str, Any]:
        """Return normalized runtime state."""

    async def execute_primitive(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute one normalized primitive."""

    async def capture_sensor(self, sensor_id: str, mode: str = "latest") -> dict[str, Any]:
        """Capture one sensor payload."""

    async def debug_snapshot(self) -> dict[str, Any]:
        """Collect a debug bundle."""
