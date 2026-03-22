"""Synchronous camera management utilities."""

from __future__ import annotations

import importlib
from typing import Any


class CameraManager:
    """Minimal synchronous OpenCV camera wrapper."""

    def __init__(self, camera_id: str, config: dict[str, int] | None = None) -> None:
        self.camera_id = camera_id
        self.config = {
            "device_index": 0,
            "width": 640,
            "height": 480,
            "fps": 30,
        }
        if config:
            self.config.update(config)
        self._capture: Any | None = None

    def _import_cv2(self) -> Any:
        try:
            return importlib.import_module("cv2")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Python package 'opencv-python' is not installed.") from exc

    @property
    def is_opened(self) -> bool:
        return self._capture is not None and bool(self._capture.isOpened())

    def open(self) -> None:
        if self.is_opened:
            return
        cv2 = self._import_cv2()
        capture = cv2.VideoCapture(int(self.config["device_index"]))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Camera '{self.camera_id}' could not be opened.")
        capture.set(getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3), int(self.config["width"]))
        capture.set(getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4), int(self.config["height"]))
        capture.set(getattr(cv2, "CAP_PROP_FPS", 5), int(self.config["fps"]))
        self._capture = capture

    def close(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None

    def grab_frame(self) -> bytes:
        if not self.is_opened:
            raise RuntimeError(f"Camera '{self.camera_id}' is not opened.")
        capture = self._capture
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Camera '{self.camera_id}' failed to read a frame.")
        cv2 = self._import_cv2()
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise RuntimeError(f"Camera '{self.camera_id}' failed to encode a frame as JPEG.")
        if hasattr(encoded, "tobytes"):
            return encoded.tobytes()
        return bytes(encoded)
