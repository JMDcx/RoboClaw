from pathlib import Path

import cv2
import numpy as np

from roboclaw.embodied.perception.calibration import CameraIntrinsics
from roboclaw.embodied.perception.geometry import extract_detection_point_cloud
from roboclaw.embodied.perception.service import PerceptionService
from roboclaw.embodied.perception.tracking import TemporalObjectMemory
from roboclaw.sim_camera import SimCameraFrame


class _StubSource:
    def __init__(self, tmp_path: Path) -> None:
        self._image_path = tmp_path / "frame.png"
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        cv2.imwrite(str(self._image_path), image)
        self._depth = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
                [0.0, 0.5, 0.6, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._depth_path = tmp_path / "frame_depth.npy"
        np.save(self._depth_path, self._depth)
        self._depth_vis_path = tmp_path / "frame_depth.png"
        cv2.imwrite(str(self._depth_vis_path), np.zeros((4, 4, 3), dtype=np.uint8))

    def get_latest_frame(self):
        frame = SimCameraFrame(
            timestamp_ms=123,
            width=4,
            height=4,
            encoding="raw_bgr",
            image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=np.load(self._depth_path),
            fresh=True,
        )
        return frame, self._image_path, self._depth_path, self._depth_vis_path


class _StubDetector:
    def detect(self, image_path: Path):
        return [
            {
                "object_id": "",
                "raw_class_name": "cup",
                "task_label": "unknown",
                "class_name": "cup",
                "confidence": 0.9,
                "bbox_xyxy": [1.0, 1.0, 3.0, 3.0],
                "mask_rle": None,
                "center_xy": [2.0, 2.0],
                "visibility": 1.0,
                "pickable": False,
                "container_candidate": False,
                "attributes": {},
            }
        ]


class _StubLaptopDetector:
    def detect(self, image_path: Path):
        return [
            {
                "object_id": "",
                "raw_class_name": "laptop",
                "task_label": "unknown",
                "class_name": "laptop",
                "confidence": 0.45,
                "bbox_xyxy": [1.0, 1.0, 7.0, 3.0],
                "mask_rle": None,
                "center_xy": [4.0, 2.0],
                "visibility": 1.0,
                "pickable": False,
                "container_candidate": False,
                "attributes": {},
            }
        ]


def test_perception_service_enriches_objects_with_depth_geometry(tmp_path: Path) -> None:
    service = PerceptionService(
        workspace=tmp_path,
        detector=_StubDetector(),
        tracker=TemporalObjectMemory(stable_frames=1),
    )
    service._source = _StubSource(tmp_path)

    frame = service.analyze_scene(camera_name="head")

    assert frame.has_depth is True
    assert frame.depth_path is not None
    assert frame.camera_calibration_path is not None
    assert Path(frame.camera_calibration_path).exists()
    obj = frame.objects[0]
    assert obj.attributes["grasp_region_point_count"] > 0
    assert obj.attributes["centroid_3d"] is not None
    assert obj.attributes["extent_3d"] is not None
    assert obj.attributes["mask_depth_stats"]["valid_point_count"] > 0


def test_perception_service_adds_sim_fallback_red_block_and_suppresses_overlap(tmp_path: Path) -> None:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[4:10, 5:11] = np.array([0, 0, 255], dtype=np.uint8)
    image_path = tmp_path / "red_block.png"
    cv2.imwrite(str(image_path), image)
    depth = np.full((16, 16), 0.7, dtype=np.float32)
    depth[4:10, 5:11] = 0.45
    depth_path = tmp_path / "red_block_depth.npy"
    np.save(depth_path, depth)
    depth_vis_path = tmp_path / "red_block_depth.png"
    cv2.imwrite(str(depth_vis_path), np.zeros((16, 16, 3), dtype=np.uint8))

    class _BlockSource:
        def get_latest_frame(self):
            frame = SimCameraFrame(
                timestamp_ms=124,
                width=16,
                height=16,
                encoding="raw_bgr",
                image_bgr=image.copy(),
                depth_m=depth.copy(),
                fresh=True,
            )
            return frame, image_path, depth_path, depth_vis_path

    service = PerceptionService(
        workspace=tmp_path,
        detector=_StubLaptopDetector(),
        tracker=TemporalObjectMemory(stable_frames=1),
    )
    service._source = _BlockSource()

    frame = service.analyze_scene(camera_name="head")

    assert len(frame.objects) == 1
    obj = frame.objects[0]
    assert obj.raw_class_name == "sim_red_block"
    assert obj.pickable is True
    assert obj.attributes["detector_source"] == "sim_fallback"
    assert obj.attributes["grasp_region_point_count"] > 0


def test_extract_detection_point_cloud_prefers_mask_when_available() -> None:
    depth = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.5, 0.0],
            [0.0, 0.6, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(width=4, height=4, fx=1.0, fy=1.0, cx=0.0, cy=0.0, focal_length_mm=1.0, horizontal_aperture_mm=1.0)
    mask_rle = {
        "size": [4, 4],
        "counts": [5, 2, 9],
    }
    points = extract_detection_point_cloud(
        depth_m=depth,
        intrinsics=intrinsics,
        bbox_xyxy=[1.0, 1.0, 3.0, 3.0],
        mask_rle=mask_rle,
        voxel_size_m=None,
        max_points=None,
    )
    assert points.shape == (2, 3)
    assert np.allclose(points[:, 2], np.array([0.4, 0.5], dtype=np.float32))


def test_extract_detection_point_cloud_falls_back_to_bbox() -> None:
    depth = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.5, 0.0],
            [0.0, 0.6, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(width=4, height=4, fx=1.0, fy=1.0, cx=0.0, cy=0.0, focal_length_mm=1.0, horizontal_aperture_mm=1.0)
    points = extract_detection_point_cloud(
        depth_m=depth,
        intrinsics=intrinsics,
        bbox_xyxy=[1.0, 1.0, 3.0, 3.0],
        mask_rle=None,
        voxel_size_m=None,
        max_points=None,
    )
    assert points.shape == (4, 3)
    assert np.allclose(np.sort(points[:, 2]), np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32))
