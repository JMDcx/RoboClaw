"""Structured schemas for perception outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DetectedObject:
    """One detected and temporally-tracked object."""

    object_id: str
    raw_class_name: str
    task_label: str
    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask_rle: dict[str, Any] | None
    center_xy: list[float]
    stable: bool
    age: int
    visibility: float
    pickable: bool
    container_candidate: bool
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)


@dataclass(slots=True)
class PerceptionFrame:
    """Structured result for one analyzed frame."""

    frame_id: str
    timestamp_ms: int
    camera_name: str
    image_path: str
    depth_path: str | None = None
    depth_visualization_path: str | None = None
    overlay_path: str | None = None
    has_depth: bool = False
    camera_intrinsics: dict[str, Any] | None = None
    camera_extrinsics: dict[str, Any] | None = None
    camera_calibration_path: str | None = None
    objects: list[DetectedObject] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "camera_name": self.camera_name,
            "image_path": self.image_path,
            "depth_path": self.depth_path,
            "depth_visualization_path": self.depth_visualization_path,
            "overlay_path": self.overlay_path,
            "has_depth": self.has_depth,
            "camera_intrinsics": self.camera_intrinsics,
            "camera_extrinsics": self.camera_extrinsics,
            "camera_calibration_path": self.camera_calibration_path,
            "objects": [obj.to_dict() for obj in self.objects],
        }
