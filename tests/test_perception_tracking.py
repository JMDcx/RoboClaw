from roboclaw.embodied.perception.tracking import TemporalObjectMemory


def _det(class_name: str, bbox, confidence: float = 0.9):
    x1, y1, x2, y2 = bbox
    return {
        "class_name": class_name,
        "confidence": confidence,
        "bbox_xyxy": [x1, y1, x2, y2],
        "mask_rle": None,
        "center_xy": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
        "visibility": 1.0,
        "attributes": {},
    }


def test_temporal_memory_reuses_track_id_across_frames() -> None:
    memory = TemporalObjectMemory(stable_frames=2)
    first = memory.update([_det("cup", [10, 10, 50, 50])])
    second = memory.update([_det("cup", [12, 12, 52, 52])])
    assert first[0].track_id == second[0].track_id


def test_temporal_memory_marks_object_stable_after_repeated_frames() -> None:
    memory = TemporalObjectMemory(stable_frames=2, stable_confidence=0.5)
    memory.update([_det("cup", [10, 10, 50, 50], confidence=0.8)])
    second = memory.update([_det("cup", [11, 11, 51, 51], confidence=0.9)])
    assert second[0].stable is True


def test_temporal_memory_creates_new_track_for_different_class() -> None:
    memory = TemporalObjectMemory(stable_frames=2)
    first = memory.update([_det("cup", [10, 10, 50, 50])])
    second = memory.update([_det("bottle", [10, 10, 50, 50])])
    assert first[0].track_id != second[0].track_id
