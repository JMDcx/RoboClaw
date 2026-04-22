"""Geometry helpers for RGB-D grasp-oriented perception."""

from __future__ import annotations

from typing import Any

import numpy as np

from roboclaw.embodied.perception.calibration import CameraIntrinsics


def enrich_detection_with_depth(
    detection: dict[str, Any],
    *,
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
) -> dict[str, Any]:
    """Attach object-level 3D geometry summaries derived from depth."""
    enriched = dict(detection)
    attributes = dict(enriched.get("attributes") or {})
    attributes["has_depth"] = bool(depth_m is not None and intrinsics is not None)
    if depth_m is None or intrinsics is None:
        enriched["attributes"] = attributes
        return enriched

    mask = _build_detection_mask(
        depth_shape=depth_m.shape[:2],
        bbox_xyxy=enriched.get("bbox_xyxy") or [],
        mask_rle=enriched.get("mask_rle"),
    )
    if mask is None:
        enriched["attributes"] = attributes
        return enriched

    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    point_count = int(valid_mask.sum())
    attributes["grasp_region_point_count"] = point_count
    if point_count == 0:
        attributes["mask_depth_stats"] = {
            "valid_point_count": 0,
            "min_m": None,
            "max_m": None,
            "mean_m": None,
            "median_m": None,
            "std_m": None,
        }
        enriched["attributes"] = attributes
        return enriched

    ys, xs = np.nonzero(valid_mask)
    zs = depth_m[valid_mask].astype(np.float32, copy=False)
    points = deproject_pixels(xs, ys, zs, intrinsics)
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    extent_xyz = (max_xyz - min_xyz).tolist()
    centroid = points.mean(axis=0).tolist()

    principal_axis = None
    principal_axis_hint = "compact"
    if point_count >= 3:
        centered = points - points.mean(axis=0, keepdims=True)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        principal_axis = vh[0].astype(np.float32).tolist()
        elongation = float(singular_values[0] / max(singular_values[-1], 1e-6))
        principal_axis_hint = "elongated" if elongation >= 2.0 else "compact"
        attributes["shape_elongation"] = elongation

    attributes["mask_depth_stats"] = {
        "valid_point_count": point_count,
        "min_m": float(zs.min()),
        "max_m": float(zs.max()),
        "mean_m": float(zs.mean()),
        "median_m": float(np.median(zs)),
        "std_m": float(zs.std()),
    }
    attributes["centroid_3d"] = [float(v) for v in centroid]
    attributes["extent_3d"] = [float(v) for v in extent_xyz]
    attributes["principal_axis_3d"] = [float(v) for v in principal_axis] if principal_axis is not None else None
    attributes["principal_axis_hint"] = principal_axis_hint
    attributes["top_surface_height"] = float(zs.min())
    enriched["attributes"] = attributes
    return enriched


def deproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert camera pixels plus depth to camera-frame XYZ points."""
    x = (xs.astype(np.float32) - intrinsics.cx) * zs / intrinsics.fx
    y = (ys.astype(np.float32) - intrinsics.cy) * zs / intrinsics.fy
    return np.stack([x, y, zs.astype(np.float32)], axis=1)


def extract_detection_point_cloud(
    *,
    depth_m: np.ndarray | None,
    intrinsics: CameraIntrinsics | None,
    bbox_xyxy: list[float],
    mask_rle: dict[str, Any] | None,
    voxel_size_m: float | None = 0.005,
    max_points: int | None = 2048,
) -> np.ndarray:
    """Return a cropped camera-frame point cloud for one detection."""
    if depth_m is None or intrinsics is None:
        return np.zeros((0, 3), dtype=np.float32)
    mask = _build_detection_mask(
        depth_shape=depth_m.shape[:2],
        bbox_xyxy=bbox_xyxy,
        mask_rle=mask_rle,
    )
    if mask is None:
        return np.zeros((0, 3), dtype=np.float32)
    valid_mask = mask & np.isfinite(depth_m) & (depth_m > 0)
    if not valid_mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    ys, xs = np.nonzero(valid_mask)
    zs = depth_m[valid_mask].astype(np.float32, copy=False)
    points = deproject_pixels(xs, ys, zs, intrinsics)
    if voxel_size_m and voxel_size_m > 0.0 and len(points) > 1:
        points = _voxel_downsample(points, voxel_size_m)
    if max_points is not None and max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
        points = points[indices]
    return points.astype(np.float32, copy=False)


def point_cloud_to_list(points: np.ndarray) -> list[list[float]]:
    """Convert an `(N, 3)` point cloud to a JSON-friendly list."""
    if points.size == 0:
        return []
    return [[float(value) for value in point] for point in points.tolist()]


def _build_detection_mask(
    *,
    depth_shape: tuple[int, int],
    bbox_xyxy: list[float],
    mask_rle: dict[str, Any] | None,
) -> np.ndarray | None:
    h, w = depth_shape
    if mask_rle:
        decoded = decode_binary_mask(mask_rle)
        if decoded is not None and decoded.shape == (h, w):
            return decoded
    if len(bbox_xyxy) != 4:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def decode_binary_mask(mask_rle: dict[str, Any]) -> np.ndarray | None:
    """Decode the simple row-major run-length format used by perception v1."""
    if "size" not in mask_rle or "counts" not in mask_rle:
        return None
    h, w = [int(v) for v in mask_rle["size"]]
    flat = np.zeros(h * w, dtype=np.uint8)
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
    return flat.reshape(h, w).astype(bool)


def _voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    quantized = np.floor(points / float(voxel_size_m)).astype(np.int32)
    _, unique_indices = np.unique(quantized, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices]
