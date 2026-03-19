"""Runtime path helpers derived from the active config context."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from roboclaw.config.loader import get_config_path
from roboclaw.utils.helpers import ensure_dir

WORKSPACE_PATH_ENV = "ROBOCLAW_WORKSPACE_PATH"
CALIBRATION_SUBDIR = "calibration"
LEGACY_CALIBRATION_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "lerobot"
    / "calibration"
    / "robots"
)
LEGACY_CALIBRATION_ROBOT_DIRS = {
    "so101": ("so_follower", "so100_follower"),
}
_current_workspace_path: Path | None = None


def set_workspace_path(path: Path | None) -> None:
    """Set the active runtime workspace path override."""
    global _current_workspace_path
    _current_workspace_path = path


def get_data_dir() -> Path:
    """Return the instance-level runtime data directory."""
    return ensure_dir(get_config_path().parent)


def get_calibration_root() -> Path:
    """Return the canonical calibration root under the active config root."""
    return ensure_dir(get_data_dir() / CALIBRATION_SUBDIR)


def get_robot_calibration_dir(robot_name: str) -> Path:
    """Return the canonical calibration directory for one robot family."""
    robot = robot_name.strip()
    if not robot:
        raise ValueError("robot_name cannot be empty.")
    return ensure_dir(get_calibration_root() / robot)


def _legacy_calibration_dirs(robot_name: str) -> tuple[Path, ...]:
    robot = robot_name.strip().lower()
    return tuple(LEGACY_CALIBRATION_ROOT / name for name in LEGACY_CALIBRATION_ROBOT_DIRS.get(robot, ()))


def find_legacy_calibration_file(robot_name: str, calibration_id: str) -> Path | None:
    """Return a legacy calibration file if one exists."""
    filename = f"{calibration_id}.json"
    for root in _legacy_calibration_dirs(robot_name):
        candidate = root / filename
        if candidate.exists():
            return candidate
    return None


def ensure_robot_calibration_file(robot_name: str, calibration_id: str) -> Path:
    """Ensure the canonical calibration file exists, migrating from legacy cache if needed."""
    canonical = get_robot_calibration_dir(robot_name) / f"{calibration_id}.json"
    if canonical.exists():
        return canonical

    legacy = find_legacy_calibration_file(robot_name, calibration_id)
    if legacy is None:
        return canonical

    canonical.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, canonical)
    return canonical


def get_runtime_subdir(name: str) -> Path:
    """Return a named runtime subdirectory under the instance data dir."""
    return ensure_dir(get_data_dir() / name)


def get_media_dir(channel: str | None = None) -> Path:
    """Return the media directory, optionally namespaced per channel."""
    base = get_runtime_subdir("media")
    return ensure_dir(base / channel) if channel else base


def get_cron_dir() -> Path:
    """Return the cron storage directory."""
    return get_runtime_subdir("cron")


def get_logs_dir() -> Path:
    """Return the logs directory."""
    return get_runtime_subdir("logs")


def get_calibration_dir(robot_name: str | None = None) -> Path:
    """Return the calibration directory, optionally namespaced per robot."""
    root = get_calibration_root()
    return ensure_dir(root / robot_name) if robot_name else root


def get_workspace_path(workspace: str | None = None) -> Path:
    """Resolve and ensure the agent workspace path."""
    env_path = os.environ.get(WORKSPACE_PATH_ENV)
    if _current_workspace_path is not None:
        path = _current_workspace_path.expanduser()
    elif env_path:
        path = Path(env_path).expanduser()
    elif workspace:
        path = Path(workspace).expanduser()
    else:
        path = Path.home() / ".roboclaw" / "workspace"
    return ensure_dir(path)


def get_cli_history_path() -> Path:
    """Return the shared CLI history file path."""
    return Path.home() / ".roboclaw" / "history" / "cli_history"


def get_bridge_install_dir() -> Path:
    """Return the shared WhatsApp bridge installation directory."""
    return Path.home() / ".roboclaw" / "bridge"


def get_legacy_sessions_dir() -> Path:
    """Return the legacy global session directory used for migration fallback."""
    return Path.home() / ".roboclaw" / "sessions"


def resolve_serial_by_id_path(device_path: str) -> Path | None:
    """Resolve a /dev/tty* path back to its /dev/serial/by-id symlink."""
    candidate = Path(device_path).expanduser()
    serial_dir = Path("/dev/serial/by-id")
    if str(candidate).startswith("/dev/serial/by-id/"):
        return candidate if candidate.exists() else None
    if not candidate.exists() or not serial_dir.exists():
        return None

    try:
        device_real = candidate.resolve()
    except FileNotFoundError:
        return None

    for link in serial_dir.iterdir():
        try:
            if link.resolve() == device_real:
                return link
        except FileNotFoundError:
            continue
    return None


def resolve_active_serial_device_path(by_id_path: str) -> Path:
    """Resolve a /dev/serial/by-id path to the current device node at runtime."""
    candidate = Path(by_id_path).expanduser()
    if not str(candidate).startswith("/dev/serial/by-id/"):
        raise ValueError(f"Expected a /dev/serial/by-id path, got '{by_id_path}'.")
    if not candidate.exists():
        raise FileNotFoundError(f"Serial device link '{candidate}' does not exist.")
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved serial device '{resolved}' does not exist.")
    return resolved
