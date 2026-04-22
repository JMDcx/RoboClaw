"""Shared G1 camera geometry helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


# Calibrated from the Isaac Lab G1 tabletop scenes currently used by RoboClaw.
HEAD_CAMERA_ORIGIN_BASE = np.array([0.05372197, 0.01753595, 0.47386146], dtype=np.float64)
HEAD_CAMERA_TO_BASE_ROTATION = np.array(
    [
        [0.0, -0.73855292, 0.67419551],
        [-1.0, 0.0, 0.0],
        [0.0, -0.67419551, -0.73855292],
    ],
    dtype=np.float64,
)


def head_camera_extrinsics_dict() -> dict[str, Any]:
    """Return the fixed head-camera extrinsics in a JSON-friendly form."""
    return {
        "translation_base": [float(value) for value in HEAD_CAMERA_ORIGIN_BASE],
        "rotation_camera_to_base": [
            [float(entry) for entry in row]
            for row in HEAD_CAMERA_TO_BASE_ROTATION.tolist()
        ],
    }


def transform_camera_translation_to_base(translation_cam: list[float] | tuple[float, float, float]) -> list[float]:
    """Transform one camera-frame XYZ point into the G1 base frame."""
    camera_point = np.asarray(translation_cam, dtype=np.float64)
    base_point = HEAD_CAMERA_ORIGIN_BASE + HEAD_CAMERA_TO_BASE_ROTATION @ camera_point
    return [float(value) for value in base_point]


def transform_camera_vector_to_base(vector_cam: list[float] | tuple[float, float, float]) -> list[float]:
    """Transform one camera-frame direction vector into the G1 base frame."""
    camera_vector = np.asarray(vector_cam, dtype=np.float64)
    base_vector = HEAD_CAMERA_TO_BASE_ROTATION @ camera_vector
    return [float(value) for value in base_vector]
