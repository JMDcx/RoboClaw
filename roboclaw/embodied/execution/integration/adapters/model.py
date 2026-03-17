"""Adapter registration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily, TransportKind

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python < 3.11 fallback for local tooling.
    class StrEnum(str, Enum):
        """Fallback for Python versions without enum.StrEnum."""


class AdapterOperation(StrEnum):
    """Lifecycle operation names exposed by all adapters."""

    DEPENDENCY_CHECK = "dependency_check"
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    READY = "ready"
    STOP = "stop"
    RESET = "reset"
    RECOVER = "recover"


class DependencyKind(StrEnum):
    """Dependency kinds checked before adapter activation."""

    BINARY = "binary"
    ENV_VAR = "env_var"
    DEVICE = "device"
    NETWORK = "network"
    ROS2_NODE = "ros2_node"
    ROS2_TOPIC = "ros2_topic"
    ROS2_SERVICE = "ros2_service"
    ROS2_ACTION = "ros2_action"
    OTHER = "other"


class ErrorCategory(StrEnum):
    """Normalized adapter error taxonomy."""

    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    COMMAND = "command"
    SAFETY = "safety"
    INTERNAL = "internal"
    OTHER = "other"


class AdapterHealthMode(StrEnum):
    """Normalized adapter health mode."""

    READY = "ready"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    TELEMETRY_ONLY = "telemetry_only"
    UNAVAILABLE = "unavailable"


class CompatibilityComponent(StrEnum):
    """Component family referenced by compatibility constraints."""

    TRANSPORT = "transport"
    BRIDGE = "bridge"
    WORKSPACE_SCHEMA = "workspace_schema"
    ROBOT_SCHEMA = "robot_schema"
    SENSOR_SCHEMA = "sensor_schema"
    ADAPTER_RUNTIME = "adapter_runtime"
    OTHER = "other"


@dataclass(frozen=True)
class DependencySpec:
    """One dependency required by an adapter binding."""

    id: str
    kind: DependencyKind
    description: str
    required: bool = True
    checker: str | None = None
    hint: str | None = None


@dataclass(frozen=True)
class OperationTimeout:
    """Timeout and retry policy for one lifecycle operation."""

    operation: AdapterOperation
    timeout_s: float
    retries: int = 0
    backoff_s: float = 0.0

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError(f"Operation timeout for '{self.operation}' must be > 0.")
        if self.retries < 0:
            raise ValueError(f"Retry count for '{self.operation}' cannot be negative.")
        if self.backoff_s < 0:
            raise ValueError(f"Backoff for '{self.operation}' cannot be negative.")


@dataclass(frozen=True)
class TimeoutPolicy:
    """Default and per-operation timeout behavior."""

    default_timeout_s: float = 30.0
    operations: tuple[OperationTimeout, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.default_timeout_s <= 0:
            raise ValueError("Default timeout must be > 0.")
        operation_names = [spec.operation for spec in self.operations]
        if len(set(operation_names)) != len(operation_names):
            raise ValueError("Duplicate timeout overrides are not allowed.")

    def timeout_for(self, operation: AdapterOperation) -> OperationTimeout:
        for item in self.operations:
            if item.operation == operation:
                return item
        return OperationTimeout(operation=operation, timeout_s=self.default_timeout_s)


@dataclass(frozen=True)
class ErrorCodeSpec:
    """One machine-readable error code in adapter taxonomy."""

    code: str
    category: ErrorCategory
    description: str
    recoverable: bool = True
    retryable: bool = False
    related_operation: AdapterOperation | None = None


@dataclass(frozen=True)
class DegradedModeSpec:
    """One allowed degraded mode and capability impact."""

    mode: AdapterHealthMode
    description: str
    available_capabilities: tuple[CapabilityFamily, ...] = field(default_factory=tuple)
    blocked_capabilities: tuple[CapabilityFamily, ...] = field(default_factory=tuple)
    allowed_operations: tuple[AdapterOperation, ...] = field(default_factory=tuple)
    entered_on_error_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.mode == AdapterHealthMode.READY:
            raise ValueError("Degraded mode spec cannot use AdapterHealthMode.READY.")
        if not self.description.strip():
            raise ValueError(f"Degraded mode '{self.mode.value}' description cannot be empty.")
        overlap = set(self.available_capabilities) & set(self.blocked_capabilities)
        if overlap:
            names = ", ".join(sorted(item.value for item in overlap))
            raise ValueError(
                f"Degraded mode '{self.mode.value}' has capabilities both available and blocked: {names}."
            )
        for code in self.entered_on_error_codes:
            if not code.strip():
                raise ValueError(
                    f"Degraded mode '{self.mode.value}' entered_on_error_codes cannot contain empty values."
                )


@dataclass(frozen=True)
class VersionConstraint:
    """Version requirement for one compatibility component."""

    component: CompatibilityComponent
    target: str
    requirement: str
    required: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Version constraint target cannot be empty.")
        if not self.requirement.strip():
            raise ValueError(f"Version constraint requirement for '{self.target}' cannot be empty.")


@dataclass(frozen=True)
class AdapterCompatibilitySpec:
    """Compatibility/version contract for one adapter binding."""

    adapter_api_version: str = "1.0"
    constraints: tuple[VersionConstraint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.adapter_api_version.strip():
            raise ValueError("Adapter compatibility adapter_api_version cannot be empty.")
        keys = [(item.component, item.target) for item in self.constraints]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Adapter compatibility constraints cannot contain duplicate component/target pairs."
            )

    def for_component(self, component: CompatibilityComponent) -> tuple[VersionConstraint, ...]:
        """Return constraints for one component family."""

        return tuple(item for item in self.constraints if item.component == component)


_REQUIRED_ADAPTER_OPERATIONS = (
    AdapterOperation.DEPENDENCY_CHECK,
    AdapterOperation.CONNECT,
    AdapterOperation.DISCONNECT,
    AdapterOperation.READY,
    AdapterOperation.STOP,
    AdapterOperation.RESET,
    AdapterOperation.RECOVER,
)


_DEFAULT_ADAPTER_ERROR_CODES = (
    ErrorCodeSpec(
        code="DEP_MISSING",
        category=ErrorCategory.DEPENDENCY,
        description="Required dependency is missing or unavailable.",
        recoverable=False,
        related_operation=AdapterOperation.DEPENDENCY_CHECK,
    ),
    ErrorCodeSpec(
        code="CONNECT_TIMEOUT",
        category=ErrorCategory.TIMEOUT,
        description="Connection timed out before adapter became ready.",
        recoverable=True,
        retryable=True,
        related_operation=AdapterOperation.CONNECT,
    ),
    ErrorCodeSpec(
        code="TRANSPORT_UNAVAILABLE",
        category=ErrorCategory.TRANSPORT,
        description="Underlying transport is unavailable.",
        recoverable=True,
        retryable=True,
    ),
    ErrorCodeSpec(
        code="RESET_FAILED",
        category=ErrorCategory.COMMAND,
        description="Reset command failed.",
        recoverable=True,
        retryable=False,
        related_operation=AdapterOperation.RESET,
    ),
    ErrorCodeSpec(
        code="RECOVER_FAILED",
        category=ErrorCategory.INTERNAL,
        description="Recovery strategy failed to restore readiness.",
        recoverable=False,
        retryable=False,
        related_operation=AdapterOperation.RECOVER,
    ),
)


@dataclass(frozen=True)
class AdapterLifecycleContract:
    """Lifecycle behavior contract for one adapter binding."""

    operations: tuple[AdapterOperation, ...] = field(default_factory=lambda: _REQUIRED_ADAPTER_OPERATIONS)
    readiness_probe: str = "ready"
    dependencies: tuple[DependencySpec, ...] = field(default_factory=tuple)
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    error_codes: tuple[ErrorCodeSpec, ...] = field(default_factory=lambda: _DEFAULT_ADAPTER_ERROR_CODES)

    def __post_init__(self) -> None:
        operation_set = set(self.operations)
        missing = set(_REQUIRED_ADAPTER_OPERATIONS) - operation_set
        if missing:
            missing_ids = ", ".join(sorted(op.value for op in missing))
            raise ValueError(f"Adapter lifecycle is missing required operations: {missing_ids}.")
        if len(operation_set) != len(self.operations):
            raise ValueError("Adapter lifecycle operations cannot contain duplicates.")

        dependency_ids = [dep.id for dep in self.dependencies]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ValueError("Adapter lifecycle dependencies cannot contain duplicate ids.")

        error_codes = [item.code for item in self.error_codes]
        if len(set(error_codes)) != len(error_codes):
            raise ValueError("Adapter lifecycle error codes cannot contain duplicates.")

    def supports(self, operation: AdapterOperation) -> bool:
        return operation in set(self.operations)


DEFAULT_ADAPTER_LIFECYCLE = AdapterLifecycleContract()


DEFAULT_ADAPTER_COMPATIBILITY = AdapterCompatibilitySpec()


@dataclass(frozen=True)
class AdapterBinding:
    """Static binding between an assembly and an implementation entrypoint."""

    id: str
    assembly_id: str
    transport: TransportKind
    implementation: str
    supported_targets: tuple[str, ...]
    bridge_id: str | None = None
    lifecycle: AdapterLifecycleContract = field(default_factory=lambda: DEFAULT_ADAPTER_LIFECYCLE)
    degraded_modes: tuple[DegradedModeSpec, ...] = field(default_factory=tuple)
    compatibility: AdapterCompatibilitySpec = field(
        default_factory=lambda: DEFAULT_ADAPTER_COMPATIBILITY
    )
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.supported_targets:
            raise ValueError(f"Adapter '{self.id}' must support at least one execution target.")
        if len(set(self.supported_targets)) != len(self.supported_targets):
            raise ValueError(f"Adapter '{self.id}' has duplicate supported targets.")
        if self.bridge_id is not None and not self.bridge_id.strip():
            raise ValueError(f"Adapter '{self.id}' bridge_id cannot be empty when specified.")

        transport_constraints = self.compatibility.for_component(CompatibilityComponent.TRANSPORT)
        if not transport_constraints:
            raise ValueError(
                f"Adapter '{self.id}' compatibility must declare at least one transport constraint."
            )
        if self.bridge_id is not None:
            bridge_constraints = tuple(
                item
                for item in self.compatibility.for_component(CompatibilityComponent.BRIDGE)
                if item.target == self.bridge_id
            )
            if not bridge_constraints:
                raise ValueError(
                    f"Adapter '{self.id}' bridge_id '{self.bridge_id}' is missing a matching bridge compatibility constraint."
                )

        known_error_codes = {item.code for item in self.lifecycle.error_codes}
        mode_ids = [mode.mode for mode in self.degraded_modes]
        if len(set(mode_ids)) != len(mode_ids):
            raise ValueError(f"Adapter '{self.id}' has duplicate degraded mode entries.")
        for mode in self.degraded_modes:
            unknown_codes = set(mode.entered_on_error_codes) - known_error_codes
            if unknown_codes:
                names = ", ".join(sorted(unknown_codes))
                raise ValueError(
                    f"Adapter '{self.id}' degraded mode '{mode.mode.value}' references unknown error codes: {names}."
                )


@dataclass(frozen=True)
class EnvironmentProbeResult:
    """Structured result for probing runtime environment readiness."""

    adapter_id: str
    assembly_id: str
    transport: TransportKind | None = None
    available_targets: tuple[str, ...] = field(default_factory=tuple)
    detected_dependencies: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencyCheckItem:
    """One dependency check outcome."""

    dependency_id: str
    kind: DependencyKind
    required: bool
    available: bool
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.dependency_id.strip():
            raise ValueError("Dependency check item dependency_id cannot be empty.")
        if self.message is not None and not self.message.strip():
            raise ValueError(
                f"Dependency check item '{self.dependency_id}' message cannot be empty when specified."
            )


@dataclass(frozen=True)
class DependencyCheckResult:
    """Structured result for adapter dependency checks."""

    adapter_id: str
    ok: bool
    items: tuple[DependencyCheckItem, ...] = field(default_factory=tuple)
    checked_dependencies: tuple[str, ...] = field(default_factory=tuple)
    missing_required: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        item_ids = [item.dependency_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("Dependency check items cannot contain duplicate dependency_id values.")
        if self.ok and self.missing_required:
            raise ValueError("Dependency check result cannot be ok=True with missing required dependencies.")
        if any(not dep_id.strip() for dep_id in self.checked_dependencies):
            raise ValueError("checked_dependencies cannot contain empty values.")
        if any(not dep_id.strip() for dep_id in self.missing_required):
            raise ValueError("missing_required cannot contain empty values.")


@dataclass(frozen=True)
class AdapterOperationResult:
    """Structured result for one adapter operation call."""

    operation: AdapterOperation
    ok: bool
    target_id: str | None = None
    message: str | None = None
    error_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_id is not None and not self.target_id.strip():
            raise ValueError(
                f"Adapter operation result '{self.operation.value}' target_id cannot be empty when specified."
            )
        if self.message is not None and not self.message.strip():
            raise ValueError(
                f"Adapter operation result '{self.operation.value}' message cannot be empty when specified."
            )
        if self.error_code is not None and not self.error_code.strip():
            raise ValueError(
                f"Adapter operation result '{self.operation.value}' error_code cannot be empty when specified."
            )
        if self.ok and self.error_code is not None:
            raise ValueError(
                f"Adapter operation result '{self.operation.value}' cannot include error_code when ok=True."
            )


@dataclass(frozen=True)
class ReadinessReport:
    """Structured readiness report for command execution."""

    ready: bool
    target_id: str | None = None
    blocked_operations: tuple[AdapterOperation, ...] = field(default_factory=tuple)
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_id is not None and not self.target_id.strip():
            raise ValueError("Readiness report target_id cannot be empty when specified.")
        if self.ready and self.blocked_operations:
            raise ValueError("Readiness report cannot block operations when ready=True.")
        if self.message is not None and not self.message.strip():
            raise ValueError("Readiness report message cannot be empty when specified.")


@dataclass(frozen=True)
class HealthReport:
    """Structured adapter health report."""

    mode: AdapterHealthMode
    healthy: bool
    error_codes: tuple[str, ...] = field(default_factory=tuple)
    blocked_operations: tuple[AdapterOperation, ...] = field(default_factory=tuple)
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode == AdapterHealthMode.READY and not self.healthy:
            raise ValueError("Health report mode READY requires healthy=True.")
        if self.message is not None and not self.message.strip():
            raise ValueError("Health report message cannot be empty when specified.")
        if any(not code.strip() for code in self.error_codes):
            raise ValueError("Health report error_codes cannot contain empty values.")


@dataclass(frozen=True)
class CompatibilityCheckItem:
    """One compatibility check evaluation result."""

    component: CompatibilityComponent
    target: str
    requirement: str
    satisfied: bool
    required: bool = True
    detected_version: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Compatibility check item target cannot be empty.")
        if not self.requirement.strip():
            raise ValueError(
                f"Compatibility check item '{self.target}' requirement cannot be empty."
            )
        if self.detected_version is not None and not self.detected_version.strip():
            raise ValueError(
                f"Compatibility check item '{self.target}' detected_version cannot be empty when specified."
            )
        if self.message is not None and not self.message.strip():
            raise ValueError(
                f"Compatibility check item '{self.target}' message cannot be empty when specified."
            )


@dataclass(frozen=True)
class CompatibilityCheckResult:
    """Structured adapter compatibility evaluation result."""

    adapter_api_version: str
    compatible: bool
    checks: tuple[CompatibilityCheckItem, ...] = field(default_factory=tuple)
    blocking_failures: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.adapter_api_version.strip():
            raise ValueError("Compatibility check result adapter_api_version cannot be empty.")
        if self.compatible and self.blocking_failures:
            raise ValueError("Compatibility check result cannot be compatible=True with blocking_failures.")
        keys = [(item.component, item.target) for item in self.checks]
        if len(set(keys)) != len(keys):
            raise ValueError("Compatibility check items cannot contain duplicate component/target pairs.")
        if any(not failure.strip() for failure in self.blocking_failures):
            raise ValueError("blocking_failures cannot contain empty values.")


@dataclass(frozen=True)
class AdapterStateSnapshot:
    """Structured normalized adapter state snapshot."""

    source: str = "adapter"
    target_id: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    updated_fields: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("Adapter state snapshot source cannot be empty.")
        if self.target_id is not None and not self.target_id.strip():
            raise ValueError("Adapter state snapshot target_id cannot be empty when specified.")
        if any(not field_name.strip() for field_name in self.updated_fields):
            raise ValueError("Adapter state snapshot updated_fields cannot contain empty values.")


@dataclass(frozen=True)
class PrimitiveExecutionResult:
    """Structured result for primitive execution."""

    primitive_name: str
    accepted: bool
    completed: bool | None = None
    status: str = "accepted"
    message: str | None = None
    error_code: str | None = None
    output: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.primitive_name.strip():
            raise ValueError("Primitive execution result primitive_name cannot be empty.")
        if not self.status.strip():
            raise ValueError(
                f"Primitive execution result '{self.primitive_name}' status cannot be empty."
            )
        if self.message is not None and not self.message.strip():
            raise ValueError(
                f"Primitive execution result '{self.primitive_name}' message cannot be empty when specified."
            )
        if self.error_code is not None and not self.error_code.strip():
            raise ValueError(
                f"Primitive execution result '{self.primitive_name}' error_code cannot be empty when specified."
            )
        if self.accepted and self.error_code is not None:
            raise ValueError(
                f"Primitive execution result '{self.primitive_name}' cannot include error_code when accepted=True."
            )


@dataclass(frozen=True)
class SensorCaptureResult:
    """Structured result for one sensor capture request."""

    sensor_id: str
    mode: str = "latest"
    captured: bool = True
    media_type: str | None = None
    payload_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.sensor_id.strip():
            raise ValueError("Sensor capture result sensor_id cannot be empty.")
        if not self.mode.strip():
            raise ValueError(
                f"Sensor capture result '{self.sensor_id}' mode cannot be empty."
            )
        if self.media_type is not None and not self.media_type.strip():
            raise ValueError(
                f"Sensor capture result '{self.sensor_id}' media_type cannot be empty when specified."
            )
        if self.payload_ref is not None and not self.payload_ref.strip():
            raise ValueError(
                f"Sensor capture result '{self.sensor_id}' payload_ref cannot be empty when specified."
            )
        if self.message is not None and not self.message.strip():
            raise ValueError(
                f"Sensor capture result '{self.sensor_id}' message cannot be empty when specified."
            )


@dataclass(frozen=True)
class DebugSnapshotResult:
    """Structured debug snapshot result."""

    captured: bool
    summary: str
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    payload: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("Debug snapshot result summary cannot be empty.")
        if any(not artifact.strip() for artifact in self.artifacts):
            raise ValueError("Debug snapshot result artifacts cannot contain empty values.")
        if self.message is not None and not self.message.strip():
            raise ValueError("Debug snapshot result message cannot be empty when specified.")
