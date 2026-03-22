"""Snapshot capture helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from roboclaw.embodied.perception.camera import CameraManager


@dataclass(frozen=True)
class SnapshotResult:
    """Result for one snapshot attempt."""

    camera_id: str
    path: str
    timestamp: str
    ok: bool
    message: str | None = None


def take_snapshot(camera: CameraManager, output_dir: str = "/tmp/roboclaw_snapshots") -> SnapshotResult:
    """Grab one JPEG frame and persist it to disk."""

    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        safe_camera_id = camera.camera_id.replace("/", "_")
        filename_timestamp = timestamp.replace(":", "-")
        path = output_path / f"snapshot_{safe_camera_id}_{filename_timestamp}.jpg"
        path.write_bytes(camera.grab_frame())
        return SnapshotResult(
            camera_id=camera.camera_id,
            path=str(path),
            timestamp=timestamp,
            ok=True,
        )
    except Exception as exc:
        return SnapshotResult(
            camera_id=camera.camera_id,
            path="",
            timestamp=timestamp,
            ok=False,
            message=str(exc),
        )
