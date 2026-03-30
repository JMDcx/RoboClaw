"""Structured schemas for perception outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DetectedObject:
    """One detected and temporally-tracked object."""

    track_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask_rle: dict[str, Any] | None
    center_xy: list[float]
    stable: bool
    age: int
    visibility: float
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
    overlay_path: str | None
    objects: list[DetectedObject]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "camera_name": self.camera_name,
            "image_path": self.image_path,
            "overlay_path": self.overlay_path,
            "objects": [obj.to_dict() for obj in self.objects],
        }
