"""Sim-only RGB-D segmentation helpers for tabletop objects in Isaac Lab."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboclaw.embodied.perception.models import _encode_binary_mask


class SimTabletopObjectDetector:
    """Detect simple tabletop objects from RGB-D without relying on class labels."""

    def __init__(
        self,
        *,
        red_hue_tolerance_deg: int = 15,
        min_red_saturation: int = 80,
        min_red_value: int = 60,
        min_area_px: int = 120,
        max_area_ratio: float = 0.20,
        min_aspect_ratio: float = 0.60,
        max_aspect_ratio: float = 1.60,
        max_candidates: int = 3,
    ) -> None:
        self._red_hue_tolerance_deg = int(red_hue_tolerance_deg)
        self._min_red_saturation = int(min_red_saturation)
        self._min_red_value = int(min_red_value)
        self._min_area_px = int(min_area_px)
        self._max_area_ratio = float(max_area_ratio)
        self._min_aspect_ratio = float(min_aspect_ratio)
        self._max_aspect_ratio = float(max_aspect_ratio)
        self._max_candidates = int(max_candidates)

    def detect(self, image_bgr: np.ndarray, depth_m: np.ndarray | None) -> list[dict[str, Any]]:
        """Return sim-only detections for red tabletop block-like objects."""
        if depth_m is None or image_bgr.size == 0:
            return []
        if image_bgr.shape[:2] != depth_m.shape[:2]:
            return []

        cv2 = _import_cv2()
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
        if int(valid_depth.sum()) < 100:
            return []

        low_red_1 = np.array([0, self._min_red_saturation, self._min_red_value], dtype=np.uint8)
        high_red_1 = np.array([self._red_hue_tolerance_deg, 255, 255], dtype=np.uint8)
        low_red_2 = np.array([180 - self._red_hue_tolerance_deg, self._min_red_saturation, self._min_red_value], dtype=np.uint8)
        high_red_2 = np.array([179, 255, 255], dtype=np.uint8)
        red_mask = cv2.inRange(hsv, low_red_1, high_red_1) | cv2.inRange(hsv, low_red_2, high_red_2)

        candidate_mask = (red_mask.astype(bool) & valid_depth).astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[dict[str, Any]] = []
        image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_area_px or area > image_area * self._max_area_ratio:
                continue
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width <= 1.0 or height <= 1.0:
                continue
            major = max(float(width), float(height))
            minor = max(1.0, min(float(width), float(height)))
            aspect_ratio = major / minor
            if aspect_ratio < self._min_aspect_ratio or aspect_ratio > self._max_aspect_ratio:
                continue

            mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, color=1, thickness=-1)
            mask_bool = mask.astype(bool)
            if int(mask_bool.sum()) < self._min_area_px:
                continue

            ys, xs = np.nonzero(mask_bool)
            x1 = float(xs.min())
            y1 = float(ys.min())
            x2 = float(xs.max() + 1)
            y2 = float(ys.max() + 1)

            masked_hsv = hsv[mask_bool]
            mean_depth = float(depth_m[mask_bool].mean())
            mean_saturation = float(masked_hsv[:, 1].mean())
            mean_value = float(masked_hsv[:, 2].mean())
            squareness_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / max(1.0 - self._min_aspect_ratio, self._max_aspect_ratio - 1.0))
            saturation_score = max(0.0, min(1.0, mean_saturation / 255.0))
            confidence = min(0.99, 0.55 + 0.20 * squareness_score + 0.25 * saturation_score)

            detections.append(
                {
                    "object_id": "",
                    "raw_class_name": "sim_red_block",
                    "task_label": "object",
                    "class_name": "sim_red_block",
                    "confidence": confidence,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "mask_rle": _encode_binary_mask(mask.astype(np.float32)),
                    "center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "visibility": 1.0,
                    "pickable": True,
                    "container_candidate": False,
                    "attributes": {
                        "detector_source": "sim_fallback",
                        "segmentation_mode": "red_compact_tabletop",
                        "sim_aspect_ratio": aspect_ratio,
                        "sim_mean_saturation": mean_saturation,
                        "sim_mean_value": mean_value,
                        "sim_mean_depth_m": mean_depth,
                    },
                }
            )

        detections.sort(
            key=lambda item: (
                float(item["confidence"]),
                -abs(float(item["attributes"].get("sim_aspect_ratio") or 0.0) - 1.0),
            ),
            reverse=True,
        )
        return detections[: self._max_candidates]


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV (cv2) is required for sim-only tabletop segmentation. "
            "Install opencv-python-headless in the RoboClaw runtime."
        ) from exc
    return cv2
