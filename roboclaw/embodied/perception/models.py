"""Detection models for perception v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class YoloSegDetector:
    """Thin wrapper around an Ultralytics segmentation model."""

    def __init__(
        self,
        *,
        model_path: str = "yolov8n-seg.pt",
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.25,
    ) -> None:
        self._model_path = model_path
        self._class_names = list(class_names or [])
        self._confidence_threshold = float(confidence_threshold)
        self._model = None

    def detect(self, image_path: Path) -> list[dict[str, Any]]:
        """Run one segmentation pass on one RGB image."""
        model = self._ensure_model()
        result = model.predict(
            source=str(image_path),
            conf=self._confidence_threshold,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        masks = getattr(result, "masks", None)
        mask_data = getattr(masks, "data", None) if masks is not None else None
        names = getattr(result, "names", {}) or {}
        detections: list[dict[str, Any]] = []
        total = len(boxes)
        for idx in range(total):
            conf = float(boxes.conf[idx].item())
            cls_idx = int(boxes.cls[idx].item())
            bbox = [float(v) for v in boxes.xyxy[idx].tolist()]
            center_xy = [
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            ]
            class_name = self._resolve_class_name(cls_idx, names)
            if self._class_names and class_name not in self._class_names:
                continue
            mask_rle = None
            if mask_data is not None and idx < len(mask_data):
                mask_rle = _encode_binary_mask(mask_data[idx].detach().cpu().numpy())
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": bbox,
                    "mask_rle": mask_rle,
                    "center_xy": center_xy,
                    "visibility": 1.0,
                    "attributes": {},
                }
            )
        return detections

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for perception v1. "
                "Install the 'ultralytics' package in the RoboClaw runtime."
            ) from exc
        self._model = YOLO(self._model_path)
        return self._model

    def _resolve_class_name(self, cls_idx: int, names: dict[int, str] | dict[str, str]) -> str:
        if cls_idx in names:
            return str(names[cls_idx])
        key = str(cls_idx)
        if key in names:
            return str(names[key])
        return f"class_{cls_idx}"


def _encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask using a simple row-major run-length format."""
    binary = (mask > 0.5).astype(np.uint8).reshape(-1)
    counts: list[int] = []
    current = 0
    run = 0
    for value in binary:
        value_int = int(value)
        if value_int == current:
            run += 1
            continue
        counts.append(run)
        run = 1
        current = value_int
    counts.append(run)
    return {
        "size": [int(mask.shape[0]), int(mask.shape[1])],
        "counts": counts,
    }
