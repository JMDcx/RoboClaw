"""Perception service for RGB scene analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roboclaw.embodied.perception.models import YoloSegDetector
from roboclaw.embodied.perception.schemas import PerceptionFrame
from roboclaw.embodied.perception.sources import SimCameraSource
from roboclaw.embodied.perception.tracking import TemporalObjectMemory


class PerceptionService:
    """Glue source, detector, tracker, and overlay generation together."""

    def __init__(
        self,
        workspace: Path,
        *,
        class_names: list[str] | None = None,
        detector: YoloSegDetector | None = None,
        tracker: TemporalObjectMemory | None = None,
    ) -> None:
        self._workspace = workspace
        self._source = SimCameraSource(workspace=workspace, camera_name="head")
        self._detector = detector or YoloSegDetector(class_names=class_names)
        self._tracker = tracker or TemporalObjectMemory()
        self._overlay_dir = workspace / ".roboclaw_tmp" / "perception"

    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        """Analyze the latest RGB frame and return structured scene objects."""
        if camera_name != "head":
            raise ValueError("Only camera_name='head' is supported for perception v1.")
        frame, image_path = self._source.get_latest_frame()
        detections = self._detector.detect(image_path)
        objects = self._tracker.update(detections)
        overlay_path = self._write_overlay(
            image_path=image_path,
            timestamp_ms=frame.timestamp_ms,
            objects=objects,
        )
        return PerceptionFrame(
            frame_id=f"{camera_name}_{frame.timestamp_ms}",
            timestamp_ms=frame.timestamp_ms,
            camera_name=camera_name,
            image_path=str(image_path),
            overlay_path=str(overlay_path) if overlay_path else None,
            objects=objects,
        )

    def _write_overlay(self, image_path: Path, timestamp_ms: int, objects) -> Path | None:
        cv2 = _import_cv2()
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox_xyxy]
            color = (0, 200, 0) if obj.stable else (0, 140, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name}#{obj.track_id} {obj.confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(16, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            if obj.mask_rle:
                _draw_mask_overlay(image, obj.mask_rle, color)
        self._overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = self._overlay_dir / f"overlay_{timestamp_ms}.png"
        if not cv2.imwrite(str(overlay_path), image):
            return None
        return overlay_path


def _draw_mask_overlay(image: np.ndarray, mask_rle: dict, color: tuple[int, int, int]) -> None:
    if "size" not in mask_rle or "counts" not in mask_rle:
        return
    h, w = mask_rle["size"]
    flat = np.zeros(int(h) * int(w), dtype=np.uint8)
    current = 0
    index = 0
    for count in mask_rle["counts"]:
        count_int = int(count)
        if count_int <= 0:
            continue
        if current == 1:
            flat[index:index + count_int] = 1
        index += count_int
        current = 1 - current
        if index >= flat.size:
            break
    mask = flat.reshape(int(h), int(w)).astype(bool)
    image[mask] = (0.65 * image[mask] + 0.35 * np.array(color)).astype(np.uint8)


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required for perception overlays. "
            "Install opencv-python-headless in the RoboClaw runtime."
        ) from exc
    return cv2
