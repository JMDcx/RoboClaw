"""Lightweight temporal object tracking for perception v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roboclaw.embodied.perception.schemas import DetectedObject


@dataclass(slots=True)
class _TrackState:
    track_id: int
    class_name: str
    bbox_xyxy: list[float]
    confidence_history: list[float] = field(default_factory=list)
    last_center_xy: list[float] = field(default_factory=list)
    last_mask_rle: dict[str, Any] | None = None
    age: int = 0
    missed_frames: int = 0
    stable: bool = False


class TemporalObjectMemory:
    """Track-by-detection memory using class, IoU, and confidence history."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.4,
        stable_frames: int = 3,
        stable_confidence: float = 0.45,
        max_missed_frames: int = 2,
    ) -> None:
        self._iou_threshold = float(iou_threshold)
        self._stable_frames = int(stable_frames)
        self._stable_confidence = float(stable_confidence)
        self._max_missed_frames = int(max_missed_frames)
        self._next_track_id = 1
        self._tracks: list[_TrackState] = []

    def update(self, detections: list[dict[str, Any]]) -> list[DetectedObject]:
        """Associate new detections with existing tracks and emit structured objects."""
        matched_track_ids: set[int] = set()
        objects: list[DetectedObject] = []

        for detection in detections:
            track = self._match_track(detection)
            if track is None:
                track = self._create_track(detection)
                self._tracks.append(track)
            else:
                self._update_track(track, detection)
            matched_track_ids.add(track.track_id)
            objects.append(self._to_object(track, detection))

        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.missed_frames += 1

        self._tracks = [t for t in self._tracks if t.missed_frames <= self._max_missed_frames]
        return objects

    def _match_track(self, detection: dict[str, Any]) -> _TrackState | None:
        best_track: _TrackState | None = None
        best_score = 0.0
        for track in self._tracks:
            if track.class_name != detection["class_name"]:
                continue
            iou = _bbox_iou(track.bbox_xyxy, detection["bbox_xyxy"])
            if iou < self._iou_threshold or iou <= best_score:
                continue
            best_score = iou
            best_track = track
        return best_track

    def _create_track(self, detection: dict[str, Any]) -> _TrackState:
        track = _TrackState(
            track_id=self._next_track_id,
            class_name=str(detection["class_name"]),
            bbox_xyxy=list(detection["bbox_xyxy"]),
            confidence_history=[float(detection["confidence"])],
            last_center_xy=list(detection["center_xy"]),
            last_mask_rle=detection.get("mask_rle"),
            age=1,
            missed_frames=0,
            stable=False,
        )
        self._next_track_id += 1
        return track

    def _update_track(self, track: _TrackState, detection: dict[str, Any]) -> None:
        track.bbox_xyxy = list(detection["bbox_xyxy"])
        track.last_center_xy = list(detection["center_xy"])
        track.last_mask_rle = detection.get("mask_rle")
        track.confidence_history.append(float(detection["confidence"]))
        track.confidence_history = track.confidence_history[-max(self._stable_frames, 5):]
        track.age += 1
        track.missed_frames = 0
        recent = track.confidence_history[-self._stable_frames:]
        if len(recent) >= self._stable_frames and sum(recent) / len(recent) >= self._stable_confidence:
            track.stable = True

    def _to_object(self, track: _TrackState, detection: dict[str, Any]) -> DetectedObject:
        return DetectedObject(
            object_id=f"object_{track.track_id}",
            raw_class_name=str(detection.get("raw_class_name") or detection["class_name"]),
            task_label=str(detection.get("task_label") or "unknown"),
            track_id=track.track_id,
            class_name=str(detection["class_name"]),
            confidence=float(detection["confidence"]),
            bbox_xyxy=list(detection["bbox_xyxy"]),
            mask_rle=detection.get("mask_rle"),
            center_xy=list(detection["center_xy"]),
            stable=track.stable,
            age=track.age,
            visibility=float(detection.get("visibility", 1.0)),
            pickable=bool(detection.get("pickable")),
            container_candidate=bool(detection.get("container_candidate")),
            attributes=dict(detection.get("attributes") or {}),
        )


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
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
