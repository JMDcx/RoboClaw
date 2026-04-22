"""Image sources for perception."""

from __future__ import annotations

from pathlib import Path

from roboclaw.sim_camera import IsaacSimCameraReader, SimCameraFrame


class SimCameraSource:
    """Read the latest RGB-D frame from the Isaac Lab head camera."""

    def __init__(self, workspace: Path, camera_name: str = "head") -> None:
        self._reader = IsaacSimCameraReader(camera_name=camera_name)
        self._cache_dir = workspace / ".roboclaw_tmp" / "sim_camera"

    def get_latest_frame(self) -> tuple[SimCameraFrame, Path, Path | None, Path | None]:
        """Return the latest saved RGB-D frame plus artifact paths."""
        return self._reader.save_latest_rgbd_frame(self._cache_dir)
