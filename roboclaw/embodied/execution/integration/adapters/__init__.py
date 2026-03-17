"""Adapter exports."""

from roboclaw.embodied.execution.integration.adapters.model import (
    AdapterBinding,
    AdapterCompatibilitySpec,
    AdapterHealthMode,
    AdapterLifecycleContract,
    AdapterOperation,
    CompatibilityComponent,
    DEFAULT_ADAPTER_LIFECYCLE,
    DEFAULT_ADAPTER_COMPATIBILITY,
    DegradedModeSpec,
    DependencyKind,
    DependencySpec,
    ErrorCategory,
    ErrorCodeSpec,
    OperationTimeout,
    TimeoutPolicy,
    VersionConstraint,
)
from roboclaw.embodied.execution.integration.adapters.protocols import EmbodiedAdapter
from roboclaw.embodied.execution.integration.adapters.registry import AdapterRegistry

__all__ = [
    "AdapterBinding",
    "AdapterCompatibilitySpec",
    "AdapterHealthMode",
    "AdapterLifecycleContract",
    "AdapterOperation",
    "AdapterRegistry",
    "CompatibilityComponent",
    "DEFAULT_ADAPTER_COMPATIBILITY",
    "DEFAULT_ADAPTER_LIFECYCLE",
    "DegradedModeSpec",
    "DependencyKind",
    "DependencySpec",
    "EmbodiedAdapter",
    "ErrorCategory",
    "ErrorCodeSpec",
    "OperationTimeout",
    "TimeoutPolicy",
    "VersionConstraint",
]
