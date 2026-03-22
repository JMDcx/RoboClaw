"""Registry for framework-owned built-in embodiments."""

from __future__ import annotations

from typing import Any

from roboclaw.embodied.builtins.model import BuiltinEmbodiment
from roboclaw.embodied.execution.orchestration.runtime.calibration import (
    CalibrationDriver,
)
from roboclaw.embodied.probes import ProbeProvider
from roboclaw.utils.helpers import normalize_token as _normalize_token

_BUILTINS_BY_ID: dict[str, BuiltinEmbodiment] = {}
_BUILTINS_BY_ROBOT_ID: dict[str, BuiltinEmbodiment] = {}
_CALIBRATION_DRIVERS: dict[str, CalibrationDriver] = {}
_PROBE_PROVIDERS: dict[str, ProbeProvider] = {}
_DEFAULTS_LOADED = False


def register_builtin_embodiment(
    embodiment: BuiltinEmbodiment,
    *,
    calibration_driver: CalibrationDriver | None = None,
    probe_provider: ProbeProvider | None = None,
) -> None:
    """Register one framework-owned built-in embodiment."""

    if calibration_driver is not None:
        _CALIBRATION_DRIVERS[calibration_driver.id] = calibration_driver
    if probe_provider is not None:
        _PROBE_PROVIDERS[probe_provider.id] = probe_provider
    _BUILTINS_BY_ID[embodiment.id] = embodiment
    _BUILTINS_BY_ROBOT_ID[embodiment.robot.id] = embodiment


def _ensure_defaults_loaded() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    from roboclaw.embodied.builtins import piperx  # noqa: F401
    from roboclaw.embodied.builtins import so101  # noqa: F401

    _DEFAULTS_LOADED = True


def list_builtin_embodiments() -> tuple[BuiltinEmbodiment, ...]:
    """List all built-in embodiments."""

    _ensure_defaults_loaded()
    return tuple(_BUILTINS_BY_ID.values())


def get_builtin_embodiment(embodiment_id: str | None) -> BuiltinEmbodiment | None:
    """Resolve one built-in embodiment by id."""

    _ensure_defaults_loaded()
    if embodiment_id is None:
        return None
    normalized = _normalize_token(embodiment_id)
    for key, embodiment in _BUILTINS_BY_ID.items():
        if _normalize_token(key) == normalized:
            return embodiment
    return None


def get_builtin_embodiment_for_robot(robot_id: str | None) -> BuiltinEmbodiment | None:
    """Resolve one built-in embodiment by robot id."""

    _ensure_defaults_loaded()
    if robot_id is None:
        return None
    normalized = _normalize_token(robot_id)
    for key, embodiment in _BUILTINS_BY_ROBOT_ID.items():
        if _normalize_token(key) == normalized:
            return embodiment
    return None


def list_builtin_robot_aliases() -> dict[str, tuple[str, ...]]:
    """Return onboarding aliases keyed by robot id."""

    aliases: dict[str, tuple[str, ...]] = {}
    for embodiment in list_builtin_embodiments():
        aliases[embodiment.robot.id] = tuple(dict.fromkeys((embodiment.robot.id, *embodiment.onboarding_aliases)))
    return aliases


def list_supported_robot_labels() -> tuple[str, ...]:
    """Return human-readable supported robot labels."""

    labels: list[str] = []
    for embodiment in list_builtin_embodiments():
        if embodiment.robot.name not in labels:
            labels.append(embodiment.robot.name)
    return tuple(labels)


def list_ros2_profiles() -> tuple[Any, ...]:
    """List built-in ROS2 embodiment profiles."""

    profiles = [
        embodiment.ros2_profile
        for embodiment in list_builtin_embodiments()
        if embodiment.ros2_profile is not None
    ]
    return tuple(profiles)


def get_ros2_profile_from_builtins(profile_or_robot_id: str | None) -> Any | None:
    """Resolve one built-in ROS2 profile by profile id or robot id."""

    if profile_or_robot_id is None:
        return None
    normalized = _normalize_token(profile_or_robot_id)
    for embodiment in list_builtin_embodiments():
        profile = embodiment.ros2_profile
        if profile is None:
            continue
        if _normalize_token(profile.id) == normalized or _normalize_token(profile.robot_id) == normalized:
            return profile
    return None


def get_control_surface_runtime_factory(profile_or_robot_id: str | None) -> Any | None:
    """Resolve one built-in control-surface runtime factory."""

    if profile_or_robot_id is None:
        return None
    normalized = _normalize_token(profile_or_robot_id)
    for embodiment in list_builtin_embodiments():
        profile = embodiment.ros2_profile
        if profile is None:
            continue
        if _normalize_token(profile.id) == normalized or _normalize_token(profile.robot_id) == normalized:
            return embodiment.control_surface_runtime_factory
    return None


def get_builtin_calibration_driver(driver_id: str | None) -> CalibrationDriver | None:
    """Resolve one registered calibration driver."""

    _ensure_defaults_loaded()
    if driver_id is None:
        return None
    return _CALIBRATION_DRIVERS.get(driver_id)


def get_builtin_probe_provider(provider_id: str | None) -> ProbeProvider | None:
    """Resolve one registered onboarding probe provider."""

    _ensure_defaults_loaded()
    if provider_id is None:
        return None
    return _PROBE_PROVIDERS.get(provider_id)


__all__ = [
    "get_builtin_calibration_driver",
    "get_builtin_embodiment",
    "get_builtin_embodiment_for_robot",
    "get_builtin_probe_provider",
    "get_control_surface_runtime_factory",
    "get_ros2_profile_from_builtins",
    "list_builtin_embodiments",
    "list_builtin_robot_aliases",
    "list_ros2_profiles",
    "list_supported_robot_labels",
    "register_builtin_embodiment",
]
