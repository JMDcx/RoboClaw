"""Adapter protocols for embodied execution."""

from __future__ import annotations

from typing import Any, Protocol

from roboclaw.embodied.execution.integration.adapters.model import (
    AdapterOperationResult,
    AdapterStateSnapshot,
    CompatibilityCheckResult,
    DebugSnapshotResult,
    DependencyCheckResult,
    EnvironmentProbeResult,
    HealthReport,
    PrimitiveExecutionResult,
    ReadinessReport,
    SensorCaptureResult,
)


class EmbodiedAdapter(Protocol):
    """Execution adapter that binds assemblies to real or simulated carriers."""

    adapter_id: str
    assembly_id: str

    def probe_env(self) -> EnvironmentProbeResult:
        """Inspect the execution environment without mutating state."""

    def check_dependencies(self) -> DependencyCheckResult:
        """Check dependencies declared by the adapter lifecycle contract."""

    async def connect(
        self,
        *,
        target_id: str,
        config: dict[str, Any] | None = None,
    ) -> AdapterOperationResult:
        """Connect to one execution target."""

    async def disconnect(self) -> AdapterOperationResult:
        """Disconnect from the active target."""

    async def ready(self) -> ReadinessReport:
        """Return readiness state for command execution."""

    async def health(self) -> HealthReport:
        """Return adapter health mode, including degraded state when applicable."""

    async def check_compatibility(self) -> CompatibilityCheckResult:
        """Validate runtime component versions against compatibility constraints."""

    async def stop(self, *, scope: str = "all") -> AdapterOperationResult:
        """Stop active tasks or motion."""

    async def reset(self, *, mode: str = "home") -> AdapterOperationResult:
        """Reset adapter state to a known mode."""

    async def recover(self, *, strategy: str | None = None) -> AdapterOperationResult:
        """Apply recovery strategy after failures."""

    async def get_state(self) -> AdapterStateSnapshot:
        """Return normalized runtime state."""

    async def execute_primitive(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> PrimitiveExecutionResult:
        """Execute one normalized primitive."""

    async def capture_sensor(self, sensor_id: str, mode: str = "latest") -> SensorCaptureResult:
        """Capture one sensor payload."""

    async def debug_snapshot(self) -> DebugSnapshotResult:
        """Collect a debug bundle."""
