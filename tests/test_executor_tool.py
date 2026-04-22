import json
from pathlib import Path

import pytest

from roboclaw.agent.tools.executor import ExecutorTool
from roboclaw.embodied.perception.schemas import DetectedObject, PerceptionFrame
from roboclaw.grasp.anydex_backend import AnyDexBackendUnavailable


class _StubPerception:
    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        return PerceptionFrame(
            frame_id="head_2",
            timestamp_ms=2,
            camera_name=camera_name,
            image_path="/tmp/image.png",
            depth_path="/tmp/depth.npy",
            overlay_path=None,
            has_depth=True,
            camera_intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "width": 4, "height": 4},
            camera_extrinsics={"translation_base": [0.0, 0.0, 0.0], "rotation_camera_to_base": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            objects=[],
        )


class _StubPickPerception:
    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        return PerceptionFrame(
            frame_id="head_3",
            timestamp_ms=3,
            camera_name=camera_name,
            image_path="/tmp/image.png",
            depth_path="/tmp/depth.npy",
            overlay_path=None,
            has_depth=True,
            camera_intrinsics={"fx": 243.2, "fy": 182.4, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480},
            camera_extrinsics={"translation_base": [0.0, 0.0, 0.0], "rotation_camera_to_base": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            objects=[
                DetectedObject(
                    object_id="object_1",
                    raw_class_name="cup",
                    task_label="object",
                    track_id=1,
                    class_name="cup",
                    confidence=0.95,
                    bbox_xyxy=[10.0, 10.0, 20.0, 20.0],
                    mask_rle=None,
                    center_xy=[15.0, 15.0],
                    stable=True,
                    age=5,
                    visibility=1.0,
                    pickable=True,
                    container_candidate=False,
                    attributes={
                        "centroid_3d": [0.1, 0.0, 0.5],
                        "extent_3d": [0.05, 0.05, 0.1],
                    },
                )
            ],
        )


class _StubBackend:
    async def plan_grasp(self, payload):
        return {
            "candidate_grasps": [
                {
                    "candidate_id": "candidate_1",
                    "score": 0.93,
                    "grasp_pose_cam": {"translation": [0.1, 0.0, 0.5]},
                    "approach_vector_cam": [0.0, 0.0, -1.0],
                    "contact_center_cam": [0.1, 0.0, 0.45],
                    "jaw_or_hand_width_hint": 0.04,
                    "grasp_depth_m": 0.03,
                    "hand_type": "dex1",
                    "rank": 1,
                    "validity_flags": {"collision_free": True},
                    "debug_source": "anydex",
                }
            ],
            "selected_grasp": {
                "candidate_id": "candidate_1",
                "score": 0.93,
                "grasp_pose_cam": {"translation": [0.1, 0.0, 0.5]},
                "approach_vector_cam": [0.0, 0.0, -1.0],
                "hand_type": "dex1",
            },
            "backend_debug": {"model": "stub"},
            "failure_reason": None,
        }


class _UnavailableBackend:
    async def plan_grasp(self, payload):
        raise AnyDexBackendUnavailable("backend not configured")


@pytest.mark.asyncio
async def test_executor_inspect_returns_structured_report(tmp_path: Path) -> None:
    tool = ExecutorTool(workspace=tmp_path)
    tool._perception = _StubPerception()

    payload = json.loads(await tool.execute(action_type="inspect"))
    assert payload["action_type"] == "inspect"
    assert payload["status"] == "success"
    assert payload["observed_effect"]["scene_frame_id"] == "head_2"


@pytest.mark.asyncio
async def test_executor_place_requires_container_id(tmp_path: Path) -> None:
    tool = ExecutorTool(workspace=tmp_path)
    payload = json.loads(
        await tool.execute(action_type="place", target_object_id="object_1")
    )
    assert payload["status"] == "terminal_failure"
    assert "target_container_id" in payload["reason"]


@pytest.mark.asyncio
async def test_executor_pick_returns_anydex_grasp_and_persists_state(tmp_path: Path) -> None:
    tool = ExecutorTool(workspace=tmp_path)
    tool.set_context("cli", "session-pick")
    tool._perception = _StubPickPerception()
    tool._backend = _StubBackend()
    tool._extract_target_point_cloud = lambda frame, target: [[0.1, 0.0, 0.5], [0.1, 0.0, 0.45]]
    payload = json.loads(
        await tool.execute(action_type="pick", target_object_id="object_1", retry_budget=2)
    )
    assert payload["status"] == "success"
    assert payload["observed_effect"]["target_object_id"] == "object_1"
    assert payload["observed_effect"]["selected_grasp"]["hand_type"] == "dex1"
    assert payload["observed_effect"]["planned_only"] is True
    assert len(payload["observed_effect"]["cropped_point_cloud"]) == 2

    state = tool._task_state.load("cli:session-pick")
    assert state.grasp_backend == "anydex"
    assert state.grasp_frame_id == "head_3"
    assert state.selected_grasp is not None
    assert state.selected_grasp["candidate_id"] == "candidate_1"
    assert len(state.candidate_grasps) == 1


@pytest.mark.asyncio
async def test_executor_pick_returns_retryable_failure_when_backend_is_unavailable(tmp_path: Path) -> None:
    tool = ExecutorTool(workspace=tmp_path)
    tool._perception = _StubPickPerception()
    tool._backend = _UnavailableBackend()
    tool._extract_target_point_cloud = lambda frame, target: [[0.1, 0.0, 0.5]]
    payload = json.loads(
        await tool.execute(action_type="pick", target_object_id="object_1", retry_budget=2)
    )
    assert payload["status"] == "retryable_failure"
    assert "anydex_backend_unavailable" in payload["reason"]


@pytest.mark.asyncio
async def test_executor_pick_executes_selected_grasp_when_dex1_is_configured(tmp_path: Path, monkeypatch) -> None:
    tool = ExecutorTool(workspace=tmp_path)
    tool._perception = _StubPickPerception()
    tool._backend = _StubBackend()
    tool._extract_target_point_cloud = lambda frame, target: [[0.1, 0.0, 0.5]]

    class _StubController:
        async def status(self, config):
            return {"connected": False}

        async def connect(self, config):
            return {"ok": True}

    class _StubRunner:
        def __init__(self, workspace, controller):
            self.workspace = workspace
            self.controller = controller

        async def run(self, **kwargs):
            return {
                "status": "success",
                "target_object_id": "object_1",
                "scene_frame_id": "head_3",
                "verification": {"success_boolean": True},
            }

    monkeypatch.setattr("roboclaw.agent.tools.executor.load_setup", lambda: {
        "unitree_g1": {
            "enabled": True,
            "connected": False,
            "mode": "sim",
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
            "sim_runtime": "isaaclab",
        }
    })
    tool._g1_controller = _StubController()
    monkeypatch.setattr("roboclaw.agent.tools.executor.G1Dex1GraspRunner", _StubRunner)

    payload = json.loads(await tool.execute(action_type="pick", target_object_id="object_1"))
    assert payload["status"] == "success"
    assert payload["observed_effect"]["planned_only"] is False
    assert payload["observed_effect"]["execution"]["verification"]["success_boolean"] is True
