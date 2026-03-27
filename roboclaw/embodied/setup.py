"""Setup management — single source of truth for the user's embodied configuration."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ARM_TYPES = ("so101_follower", "so101_leader")
_ARM_FIELDS = {"alias", "type", "port", "calibration_dir", "calibrated"}
_CAMERA_FIELDS = {"alias", "port", "width", "height", "fps"}
_VALID_TOP_KEYS = {"version", "arms", "cameras", "datasets", "policies", "scanned_ports", "scanned_cameras"}


def get_roboclaw_home(home: str | Path | None = None) -> Path:
    """Return RoboClaw home directory, honoring ROBOCLAW_HOME env var."""
    if home is not None:
        return Path(home).expanduser()
    return Path(os.environ.get("ROBOCLAW_HOME", "~/.roboclaw")).expanduser()


def get_setup_path(home: Path | None = None) -> Path:
    """Return the setup.json path under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "setup.json"


def get_calibration_root(home: Path | None = None) -> Path:
    """Return the calibration directory under *home*."""
    return (home or get_roboclaw_home()) / "workspace" / "embodied" / "calibration"


def _default_setup(home: Path | None = None) -> dict[str, Any]:
    """Build a fresh default setup dict with paths under *home*."""
    base = (home or get_roboclaw_home()) / "workspace" / "embodied"
    return {
        "version": 2,
        "arms": [],
        "cameras": [],
        "datasets": {"root": str(base / "datasets")},
        "policies": {"root": str(base / "policies")},
        "scanned_ports": [],
        "scanned_cameras": [],
    }


def load_setup(path: Path | None = None) -> dict[str, Any]:
    """Load setup.json, return defaults if not found. Refreshes calibration state from disk."""
    path = path or get_setup_path()
    if not path.exists():
        return _default_setup()
    setup = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(setup.get("cameras"), dict):
        setup["cameras"] = []
        save_setup(setup, path)
    if _refresh_calibration_state(setup):
        save_setup(setup, path)
    return setup


def save_setup(setup: dict[str, Any], path: Path | None = None) -> None:
    """Write setup.json, creating parent dirs if needed."""
    path = path or get_setup_path()
    _validate_setup(setup)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def create_setup_with_scan(path: Path | None = None) -> dict[str, Any]:
    """Create setup.json with auto-detected hardware. Called during onboard."""
    from roboclaw.embodied.scan import scan_cameras, scan_serial_ports

    path = path or get_setup_path()
    setup = _default_setup()
    setup["scanned_ports"] = scan_serial_ports()
    setup["scanned_cameras"] = scan_cameras()
    save_setup(setup, path)
    return setup


def ensure_setup(path: Path | None = None) -> dict[str, Any]:
    """Load setup.json if exists, otherwise create with defaults (no scan) and return."""
    path = path or get_setup_path()
    if path.exists():
        return load_setup(path)
    defaults = _default_setup()
    save_setup(defaults, path)
    return defaults



def mark_arm_calibrated(alias: str, path: Path | None = None) -> dict[str, Any]:
    """Mark an arm as calibrated by alias."""
    path = path or get_setup_path()
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
    alias: str, arm_type: str, port: str, *, path: Path | None = None,
) -> dict[str, Any]:
    """Add or update an arm by alias. Auto-fills calibration_dir and calibration state."""
    if arm_type not in _ARM_TYPES:
        raise ValueError(f"Invalid arm_type '{arm_type}'. Must be one of {_ARM_TYPES}.")
    if not port:
        raise ValueError("Arm port is required.")
    if not alias:
        raise ValueError("Arm alias is required.")
    from roboclaw.embodied.scan import scan_serial_ports
    path = path or get_setup_path()
    setup = load_setup(path)
    port = _resolve_port(port, scan_serial_ports())
    serial = _extract_serial_number(port)
    calibration_dir = get_calibration_root() / serial
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


def remove_arm(alias: str, path: Path | None = None) -> dict[str, Any]:
    """Remove an arm by alias."""
    path = path or get_setup_path()
    setup = load_setup(path)
    arms = setup.get("arms", [])
    arm = find_arm(arms, alias)
    if arm is None:
        raise ValueError(f"No arm with alias '{alias}' in setup.")
    arms.remove(arm)
    save_setup(setup, path)
    return setup


def rename_arm(old_alias: str, new_alias: str, *, path: Path | None = None) -> dict[str, Any]:
    """Rename an arm alias without changing any hardware-backed fields."""
    path = path or get_setup_path()
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


def set_camera(name: str, camera_index: int, path: Path | None = None) -> dict[str, Any]:
    """Add or update a camera by picking from scanned_cameras by index."""
    path = path or get_setup_path()
    if not name:
        raise ValueError("Camera alias is required.")
    setup = load_setup(path)
    scanned = setup.get("scanned_cameras", [])
    if camera_index < 0 or camera_index >= len(scanned):
        raise ValueError(
            f"camera_index {camera_index} out of range. "
            f"scanned_cameras has {len(scanned)} entries."
        )
    source = scanned[camera_index]
    port = source.get("by_path") or source.get("by_id") or source.get("dev", "")
    if not port:
        raise ValueError(f"Scanned camera at index {camera_index} has no usable path.")
    entry = {
        "alias": name,
        "port": port,
        "width": source.get("width", 640),
        "height": source.get("height", 480),
    }
    cameras = setup.setdefault("cameras", [])
    existing = find_camera(cameras, name)
    if existing is not None:
        cameras[cameras.index(existing)] = entry
    else:
        cameras.append(entry)
    save_setup(setup, path)
    return setup


def remove_camera(name: str, path: Path | None = None) -> dict[str, Any]:
    """Remove a camera by alias."""
    path = path or get_setup_path()
    setup = load_setup(path)
    cameras = setup.get("cameras", [])
    cam = find_camera(cameras, name)
    if cam is None:
        raise ValueError(f"No camera with alias '{name}' in setup.")
    cameras.remove(cam)
    save_setup(setup, path)
    return setup


def find_camera(cameras: list[dict], alias: str) -> dict | None:
    """Find a camera in the cameras list by alias. Returns the dict or None."""
    for cam in cameras:
        if cam.get("alias") == alias:
            return cam
    return None


# ── Validation ───────────────────────────────────────────────────────


def _validate_setup(setup: dict[str, Any]) -> None:
    """Validate setup against schema. Raises ValueError on invalid data."""
    invalid_top = set(setup.keys()) - _VALID_TOP_KEYS
    if invalid_top:
        raise ValueError(f"Unknown top-level keys: {invalid_top}")
    _validate_arms(setup.get("arms", []))
    _validate_cameras(setup.get("cameras", []))


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
    if not isinstance(cameras, list):
        raise ValueError("'cameras' must be a list.")
    for cam in cameras:
        if not isinstance(cam, dict):
            raise ValueError(f"Each camera must be a dict, got {type(cam).__name__}.")
        alias = cam.get("alias")
        if not alias:
            raise ValueError("Camera entry missing required 'alias' field.")
        if not cam.get("port"):
            raise ValueError(f"Camera '{alias}' missing required 'port' field.")
        bad = set(cam.keys()) - _CAMERA_FIELDS
        if bad:
            raise ValueError(f"Camera '{alias}' has unknown fields: {bad}")


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
