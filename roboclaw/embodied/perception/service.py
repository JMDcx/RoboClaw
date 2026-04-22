"""Perception service for RGB scene analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from roboclaw.embodied.embodiment.g1_camera import head_camera_extrinsics_dict
from roboclaw.embodied.perception.calibration import CameraIntrinsics, derive_head_camera_intrinsics, save_head_camera_intrinsics
from roboclaw.embodied.perception.geometry import enrich_detection_with_depth
from roboclaw.embodied.perception.models import YoloSegDetector
from roboclaw.embodied.perception.schemas import PerceptionFrame
from roboclaw.embodied.perception.sim_segmentation import SimTabletopObjectDetector
from roboclaw.embodied.perception.sources import SimCameraSource
from roboclaw.embodied.perception.tracking import TemporalObjectMemory


class PerceptionService:
    """Glue source, detector, tracker, geometry enrichment, and overlays together."""

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
        self._sim_fallback = SimTabletopObjectDetector()
        self._tracker = tracker or TemporalObjectMemory()
        self._overlay_dir = workspace / ".roboclaw_tmp" / "perception"
        self._calibration_path = self._overlay_dir / "head_camera_intrinsics.json"

    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        """Analyze the latest RGB-D frame and return structured scene objects."""
        if camera_name != "head":
            raise ValueError("Only camera_name='head' is supported for perception v1.")
        frame, image_path, depth_path, depth_vis_path = self._source.get_latest_frame()
        intrinsics = derive_head_camera_intrinsics(frame.width, frame.height)
        calibration_path = save_head_camera_intrinsics(self._calibration_path, intrinsics)
        detections = self._detector.detect(image_path)
        sim_detections = self._sim_fallback.detect(frame.image_bgr, frame.depth_m)
        detections = _merge_detections(detections, sim_detections)
        detections = [
            self._annotate_detection(idx, det, depth_m=frame.depth_m, intrinsics=intrinsics)
            for idx, det in enumerate(detections, start=1)
        ]
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
            depth_path=str(depth_path) if depth_path else None,
            depth_visualization_path=str(depth_vis_path) if depth_vis_path else None,
            overlay_path=str(overlay_path) if overlay_path else None,
            has_depth=frame.depth_m is not None,
            camera_intrinsics=intrinsics.to_dict(),
            camera_extrinsics=head_camera_extrinsics_dict(),
            camera_calibration_path=str(calibration_path),
            objects=objects,
        )

    def _annotate_detection(
        self,
        index: int,
        detection: dict,
        *,
        depth_m: np.ndarray | None,
        intrinsics: CameraIntrinsics,
    ) -> dict:
        raw_class_name = str(detection.get("raw_class_name") or detection.get("class_name") or "unknown")
        task_label, pickable, container_candidate = _task_semantics(raw_class_name)
        annotated = dict(detection)
        annotated["object_id"] = f"object_{index}"
        annotated["raw_class_name"] = raw_class_name
        annotated["task_label"] = task_label
        annotated["pickable"] = pickable
        annotated["container_candidate"] = container_candidate
        return enrich_detection_with_depth(
            annotated,
            depth_m=depth_m,
            intrinsics=intrinsics if depth_m is not None else None,
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


def _task_semantics(raw_class_name: str) -> tuple[str, bool, bool]:
    name = raw_class_name.strip().lower()
    if name in {"bowl", "box", "basket", "bin"}:
        return "container", False, True
    if name in {"person", "airplane", "car", "truck"}:
        return "robot_part", False, False
    if name:
        return "object", True, False
    return "unknown", False, False


def _merge_detections(
    yolo_detections: list[dict],
    sim_detections: list[dict],
) -> list[dict]:
    """Prefer sim fallback detections when they overlap likely YOLO misclassifications."""
    if not sim_detections:
        return list(yolo_detections)

    merged: list[dict] = []
    for detection in yolo_detections:
        bbox = detection.get("bbox_xyxy") or []
        should_drop = False
        for sim_detection in sim_detections:
            sim_bbox = sim_detection.get("bbox_xyxy") or []
            if len(bbox) != 4 or len(sim_bbox) != 4:
                continue
            if _bbox_iou_xyxy(bbox, sim_bbox) < 0.20:
                continue
            should_drop = True
            break
        if not should_drop:
            merged.append(detection)
    return [*sim_detections, *merged]


def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
