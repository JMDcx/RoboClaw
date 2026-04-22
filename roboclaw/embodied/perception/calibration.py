"""Camera calibration helpers for simulated RGB-D perception."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class CameraIntrinsics:
    """Minimal pinhole intrinsics for one camera stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    focal_length_mm: float
    horizontal_aperture_mm: float
    near_plane_m: float = 0.1
    far_plane_m: float = 1.0e5
    camera_model: str = "pinhole"
    source: str = "g1_front_camera_preset"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def derive_head_camera_intrinsics(width: int, height: int) -> CameraIntrinsics:
    """Derive G1 front-camera intrinsics from the current Isaac Lab preset."""
    focal_length_mm = 7.6
    horizontal_aperture_mm = 20.0
    fx = float(width) * focal_length_mm / horizontal_aperture_mm
    fy = float(height) * focal_length_mm / horizontal_aperture_mm
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return CameraIntrinsics(
        width=int(width),
        height=int(height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=horizontal_aperture_mm,
    )


def save_head_camera_intrinsics(path: Path, intrinsics: CameraIntrinsics) -> Path:
    """Persist derived intrinsics for downstream tools and backend services."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(intrinsics.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
