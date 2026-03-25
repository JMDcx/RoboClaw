"""Setup management — single source of truth for the user's embodied configuration."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

SETUP_PATH = Path("~/.roboclaw/workspace/embodied/setup.json").expanduser()

_ARM_TYPES = ("so101_follower", "so101_leader")
_ARM_FIELDS = {"alias", "type", "port", "calibration_dir", "calibrated"}
_CAMERA_FIELDS = {"by_path", "by_id", "dev", "width", "height"}
_VALID_TOP_KEYS = {"version", "arms", "cameras", "datasets", "policies", "scanned_ports", "scanned_cameras"}

_CALIBRATION_ROOT = Path("~/.roboclaw/workspace/embodied/calibration").expanduser()

_DEFAULT_SETUP: dict[str, Any] = {
    "version": 2,
    "arms": [],
    "cameras": {},
    "datasets": {
        "root": str(Path("~/.roboclaw/workspace/embodied/datasets").expanduser()),
    },
    "policies": {
        "root": str(Path("~/.roboclaw/workspace/embodied/policies").expanduser()),
    },
    "scanned_ports": [],
    "scanned_cameras": [],
}


def load_setup(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Load setup.json, return defaults if not found. Refreshes calibration state from disk."""
    if not path.exists():
        return copy.deepcopy(_DEFAULT_SETUP)
    setup = json.loads(path.read_text(encoding="utf-8"))
    if _refresh_calibration_state(setup):
        save_setup(setup, path)
    return setup


def save_setup(setup: dict[str, Any], path: Path = SETUP_PATH) -> None:
    """Write setup.json, creating parent dirs if needed."""
    _validate_setup(setup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_setup_with_scan(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Create setup.json with auto-detected hardware. Called during onboard."""
    from roboclaw.embodied.scan import scan_cameras, scan_serial_ports

    setup = copy.deepcopy(_DEFAULT_SETUP)
    setup["scanned_ports"] = scan_serial_ports()
    setup["scanned_cameras"] = scan_cameras()
    save_setup(setup, path)
    return setup


def ensure_setup(path: Path = SETUP_PATH) -> dict[str, Any]:
    """Load setup.json if exists, otherwise create with defaults (no scan) and return."""
    if path.exists():
        return load_setup(path)
    defaults = copy.deepcopy(_DEFAULT_SETUP)
    save_setup(defaults, path)
    return defaults



def mark_arm_calibrated(alias: str, path: Path = SETUP_PATH) -> dict[str, Any]:
    """Mark an arm as calibrated by alias."""
    setup = load_setup(path)
    arm = find_arm(setup.get("arms", []), alias)
    if not arm:
        raise ValueError(f"No arm with alias '{alias}' in setup.")
    arm["calibrated"] = True
    save_setup(setup, path)
    return setup


def _resolve_port(port: str, scanned_ports: list[dict]) -> str:
    """Resolve a volatile port (e.g. /dev/ttyACM0) to a stable by_id path.

    If port already starts with /dev/serial/by-id/ or /dev/serial/by-path/, keep as-is.
    Otherwise look up in scanned_ports and prefer by_id > by_path > original.
    """
    if port.startswith("/dev/serial/"):
        return port
    for entry in scanned_ports:
        if entry.get("dev") != port:
            continue
        by_id = entry.get("by_id", "")
        if by_id:
            return by_id
        return port
    return port


def _extract_serial_number(port: str) -> str:
    """Extract serial number from a by_id port path.

    E.g. "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14032630-if00" -> "5B14032630"
    Falls back to the full filename if no pattern matches.
    """
    filename = Path(port).name
    # Match serial number: last segment before optional -ifNN suffix
    m = re.search(r"_([A-Za-z0-9]+)(?:-if\d+)?$", filename)
    if m:
        return m.group(1)
    return filename


# ── Structured mutators (exposed as agent actions) ──────────────────


def set_arm(
    alias: str, arm_type: str, port: str, *, path: Path = SETUP_PATH,
) -> dict[str, Any]:
    """Add or update an arm by alias. Auto-fills calibration_dir and calibration state."""
    if arm_type not in _ARM_TYPES:
        raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
    if not port:
        raise ValueError("Arm port is required.")
    if not alias:
        raise ValueError("Arm alias is required.")
    from roboclaw.embodied.scan import scan_serial_ports
    setup = load_setup(path)
    port = _resolve_port(port, scan_serial_ports())
    serial = _extract_serial_number(port)
    calibration_dir = _CALIBRATION_ROOT / serial
    _migrate_none_calibration_file(calibration_dir, serial)
    existing = find_arm(setup.setdefault("arms", []), alias)
    _ensure_unique_port(setup["arms"], alias, port)
    entry: dict[str, Any] = {
        "alias": alias,
        "type": arm_type,
        "port": port,
        "calibration_dir": str(calibration_dir),
        "calibrated": _has_calibration_file(calibration_dir, serial),
    }
    arms = setup["arms"]
    if existing is not None:
        idx = arms.index(existing)
        arms[idx] = entry
    else:
        arms.append(entry)
    save_setup(setup, path)
    return setup


def arm_display_name(arm: dict) -> str:
    """Return user-friendly display name: the arm's alias."""
    return arm.get("alias", "unnamed")


def find_arm(arms: list[dict], alias: str) -> dict | None:
    """Find an arm in the arms list by alias. Returns the dict or None."""
    for arm in arms:
        if arm.get("alias") == alias:
            return arm
    return None


def remove_arm(alias: str, path: Path = SETUP_PATH) -> dict[str, Any]:
    """Remove an arm by alias."""
    setup = load_setup(path)
    arms = setup.get("arms", [])
    arm = find_arm(arms, alias)
    if arm is None:
        raise ValueError(f"No arm with alias '{alias}' in setup.")
    arms.remove(arm)
    save_setup(setup, path)
    return setup


def rename_arm(old_alias: str, new_alias: str, *, path: Path = SETUP_PATH) -> dict[str, Any]:
    """Rename an arm alias without changing any hardware-backed fields."""
    if not old_alias:
        raise ValueError("Old arm alias is required.")
    if not new_alias:
        raise ValueError("New arm alias is required.")
    setup = load_setup(path)
    arms = setup.get("arms", [])
    arm = find_arm(arms, old_alias)
    if arm is None:
        raise ValueError(f"No arm with alias '{old_alias}' in setup.")
    if old_alias != new_alias and find_arm(arms, new_alias) is not None:
        raise ValueError(f"Arm alias '{new_alias}' already exists.")
    arm["alias"] = new_alias
    save_setup(setup, path)
    return setup


def set_camera(name: str, camera_index: int, path: Path = SETUP_PATH) -> dict[str, Any]:
    """Add or update a camera by picking from scanned_cameras by index."""
    setup = load_setup(path)
    scanned = setup.get("scanned_cameras", [])
    if camera_index < 0 or camera_index >= len(scanned):
        raise ValueError(
            f"camera_index {camera_index} out of range. "
            f"scanned_cameras has {len(scanned)} entries."
        )
    source = scanned[camera_index]
    entry = {field: source[field] for field in _CAMERA_FIELDS if field in source}
    setup.setdefault("cameras", {})[name] = entry
    save_setup(setup, path)
    return setup


def remove_camera(name: str, path: Path = SETUP_PATH) -> dict[str, Any]:
    """Remove a camera by name."""
    setup = load_setup(path)
    cameras = setup.get("cameras", {})
    if name not in cameras:
        raise ValueError(f"No camera named '{name}' in setup.")
    del cameras[name]
    save_setup(setup, path)
    return setup


# ── Validation ───────────────────────────────────────────────────────


def _validate_setup(setup: dict[str, Any]) -> None:
    """Validate setup against schema. Raises ValueError on invalid data."""
    invalid_top = set(setup.keys()) - _VALID_TOP_KEYS
    if invalid_top:
        raise ValueError(f"Unknown top-level keys: {invalid_top}")
    _validate_arms(setup.get("arms", []))
    _validate_cameras(setup.get("cameras", {}))


def _validate_arms(arms: Any) -> None:
    """Validate all arm entries. Arms is a list of dicts."""
    if not isinstance(arms, list):
        raise ValueError("'arms' must be a list.")
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError(f"Each arm entry must be a dict, got {type(arm).__name__}.")
        alias = arm.get("alias", "<unknown>")
        bad_fields = set(arm.keys()) - _ARM_FIELDS
        if bad_fields:
            raise ValueError(f"Arm '{alias}' has unknown fields: {bad_fields}")
        arm_type = arm.get("type")
        if arm_type is not None and arm_type not in _ARM_TYPES:
            raise ValueError(f"Arm '{alias}' has invalid type '{arm_type}'.")


def _validate_cameras(cameras: Any) -> None:
    """Validate all camera entries."""
    if not isinstance(cameras, dict):
        raise ValueError("'cameras' must be a dict.")
    for name, cam in cameras.items():
        if not isinstance(cam, dict):
            raise ValueError(f"Camera '{name}' must be a dict.")
        bad_fields = set(cam.keys()) - _CAMERA_FIELDS
        if bad_fields:
            raise ValueError(f"Camera '{name}' has unknown fields: {bad_fields}")


def _ensure_unique_port(arms: list[dict], alias: str, port: str) -> None:
    for arm in arms:
        if arm.get("alias") == alias:
            continue
        if arm.get("port") == port:
            raise ValueError(f"Port '{port}' is already assigned to arm '{arm['alias']}'.")


def _refresh_calibration_state(setup: dict[str, Any]) -> bool:
    """Migrate None.json and recompute calibrated from disk for all arms. Returns True if anything changed."""
    changed = False
    for arm in setup.get("arms", []):
        cal_dir = Path(arm.get("calibration_dir", ""))
        serial = cal_dir.name
        if not serial or not cal_dir.exists():
            continue
        _migrate_none_calibration_file(cal_dir, serial)
        on_disk = _has_calibration_file(cal_dir, serial)
        if arm.get("calibrated") != on_disk:
            arm["calibrated"] = on_disk
            changed = True
    return changed


def _has_calibration_file(calibration_dir: Path, serial: str) -> bool:
    return (calibration_dir / f"{serial}.json").exists()


def _migrate_none_calibration_file(calibration_dir: Path, serial: str) -> None:
    legacy = calibration_dir / "None.json"
    target = calibration_dir / f"{serial}.json"
    if legacy.exists() and not target.exists():
        legacy.rename(target)

