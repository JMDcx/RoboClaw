"""Structured executor tool for high-level tidyup actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.embodied.embodiment.g1_dex1_grasp import G1Dex1GraspRunner
from roboclaw.embodied.perception.geometry import extract_detection_point_cloud, point_cloud_to_list
from roboclaw.embodied.perception.service import PerceptionService
from roboclaw.embodied.perception.calibration import CameraIntrinsics
from roboclaw.embodied.perception.schemas import PerceptionFrame
from roboclaw.embodied.setup import load_setup
from roboclaw.embodied.tasking import ExecutionReport, SceneObject, TaskStateStore
from roboclaw.grasp.anydex_backend import AnyDexBackendClient, AnyDexBackendError, AnyDexBackendUnavailable


class ExecutorTool(Tool):
    """Consume structured high-level actions and emit structured execution reports."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._perception = PerceptionService(workspace=workspace)
        self._task_state = TaskStateStore(workspace)
        self._backend = AnyDexBackendClient()
        self._g1_controller = None
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "executor"

    @property
    def description(self) -> str:
        return (
            "Execute a structured high-level tidyup action. "
            "Use this for inspect, pick, place, verify, and reset action contracts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["inspect", "pick", "place", "verify", "reset"],
                },
                "target_object_id": {"type": "string"},
                "target_container_id": {"type": "string"},
                "constraints": {"type": "object"},
                "retry_budget": {"type": "integer", "minimum": 0},
            },
            "required": ["action_type"],
        }

    async def execute(
        self,
        action_type: str,
        target_object_id: str = "",
        target_container_id: str = "",
        constraints: dict[str, Any] | None = None,
        retry_budget: int = 1,
        **kwargs: Any,
    ) -> str:
        constraints = dict(constraints or {})
        if action_type == "inspect":
            frame = self._perception.analyze_scene(camera_name="head")
            report = ExecutionReport(
                action_type=action_type,
                status="success",
                reason="scene_refreshed",
                attempt_count=1,
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "object_count": len(frame.objects),
                },
                next_recommendation="plan_next_action",
            )
            return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if action_type == "verify":
            frame = self._perception.analyze_scene(camera_name="head")
            report = ExecutionReport(
                action_type=action_type,
                status="success",
                reason="verification_complete",
                attempt_count=1,
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "stable_object_count": len([obj for obj in frame.objects if obj.stable]),
                    "target_object_id": target_object_id or None,
                    "target_container_id": target_container_id or None,
                },
                next_recommendation="complete_or_replan",
            )
            return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if action_type == "reset":
            report = ExecutionReport(
                action_type=action_type,
                status="success",
                reason="reset_scaffold_complete",
                attempt_count=1,
                observed_effect={"constraints": constraints},
                next_recommendation="inspect",
            )
            return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        if action_type == "pick":
            report = await self._plan_pick(
                target_object_id=target_object_id,
                target_container_id=target_container_id,
                constraints=constraints,
                retry_budget=retry_budget,
            )
            return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        report = self._pick_or_place_report(
            action_type=action_type,
            target_object_id=target_object_id,
            target_container_id=target_container_id,
            retry_budget=retry_budget,
        )
        return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)

    async def _plan_pick(
        self,
        *,
        target_object_id: str,
        target_container_id: str,
        constraints: dict[str, Any],
        retry_budget: int,
    ) -> ExecutionReport:
        base_failure = self._pick_or_place_report(
            action_type="pick",
            target_object_id=target_object_id,
            target_container_id=target_container_id,
            retry_budget=retry_budget,
        )
        if base_failure.status == "terminal_failure":
            return base_failure

        try:
            frame = self._perception.analyze_scene(camera_name="head")
        except Exception as exc:
            return ExecutionReport(
                action_type="pick",
                status="retryable_failure",
                reason=f"perception_failed: {exc}",
                attempt_count=max(1, retry_budget),
                observed_effect={
                    "target_object_id": target_object_id or None,
                },
                next_recommendation="inspect_or_retry",
            )
        state = self._task_state.load(self._session_key)
        state.scene_frame_id = frame.frame_id
        state.objects = [SceneObject.from_dict(obj.to_dict()) for obj in frame.objects]
        self._task_state.save(self._session_key, state)

        target_object = next((obj for obj in frame.objects if obj.object_id == target_object_id), None)
        if target_object is None:
            return ExecutionReport(
                action_type="pick",
                status="retryable_failure",
                reason=f"target object '{target_object_id}' not found in current RGB-D scene",
                attempt_count=max(1, retry_budget),
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "target_object_id": target_object_id or None,
                    "available_object_ids": [obj.object_id for obj in frame.objects],
                },
                next_recommendation="inspect_or_retry",
            )

        if not frame.has_depth or not frame.depth_path or not frame.camera_intrinsics:
            return ExecutionReport(
                action_type="pick",
                status="retryable_failure",
                reason="depth_unavailable_for_grasp_planning",
                attempt_count=max(1, retry_budget),
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "target_object_id": target_object_id,
                    "has_depth": frame.has_depth,
                },
                next_recommendation="inspect_or_retry",
            )

        cropped_point_cloud = self._extract_target_point_cloud(frame, target_object)
        if not cropped_point_cloud:
            return ExecutionReport(
                action_type="pick",
                status="retryable_failure",
                reason="target_point_cloud_empty",
                attempt_count=max(1, retry_budget),
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "target_object_id": target_object_id,
                },
                next_recommendation="inspect_or_retry",
            )

        request = {
            "frame_id": frame.frame_id,
            "rgb_path": frame.image_path,
            "depth_path": frame.depth_path,
            "camera_intrinsics": frame.camera_intrinsics,
            "camera_extrinsics": frame.camera_extrinsics,
            "target_object": target_object.to_dict(),
            "object_mask": target_object.mask_rle,
            "cropped_point_cloud": cropped_point_cloud,
            "scene_constraints": {
                "target_container_id": target_container_id or None,
                "retry_budget": max(1, retry_budget),
                "constraints": constraints,
            },
            "gripper_profile": "dex1_parallel_gripper",
            "robot_profile": "g1_right_arm_head_camera",
        }
        try:
            backend_result = await self._backend.plan_grasp(request)
        except AnyDexBackendUnavailable as exc:
            return self._backend_failure_report(
                retry_budget=retry_budget,
                target_object_id=target_object_id,
                frame_id=frame.frame_id,
                reason=f"anydex_backend_unavailable: {exc}",
            )
        except AnyDexBackendError as exc:
            return self._backend_failure_report(
                retry_budget=retry_budget,
                target_object_id=target_object_id,
                frame_id=frame.frame_id,
                reason=str(exc),
            )

        candidate_grasps = self._normalize_candidates(backend_result.get("candidate_grasps") or [])
        selected_grasp = self._normalize_selected_grasp(
            backend_result.get("selected_grasp"),
            candidate_grasps,
        )
        if not candidate_grasps or selected_grasp is None:
            failure_reason = str(backend_result.get("failure_reason") or "backend returned no valid grasps")
            return self._backend_failure_report(
                retry_budget=retry_budget,
                target_object_id=target_object_id,
                frame_id=frame.frame_id,
                reason=failure_reason,
            )

        state.target_object_id = target_object_id
        if target_container_id:
            state.container_object_id = target_container_id
        state.candidate_grasps = candidate_grasps
        state.selected_grasp = selected_grasp
        state.grasp_backend = "anydex"
        state.grasp_frame_id = frame.frame_id
        self._task_state.save(self._session_key, state)

        execution_result = await self._execute_selected_grasp(
            frame=frame,
            target_object_id=target_object_id,
            selected_grasp=selected_grasp,
        )
        executed = execution_result is not None
        if executed:
            state.last_executor_report = execution_result
            self._task_state.save(self._session_key, state)
        if executed and str(execution_result.get("status") or "") != "success":
            return ExecutionReport(
                action_type="pick",
                status="retryable_failure",
                reason=str(execution_result.get("failure_reason") or "grasp_execution_failed"),
                attempt_count=max(1, retry_budget),
                observed_effect={
                    "scene_frame_id": frame.frame_id,
                    "grasp_frame_id": frame.frame_id,
                    "target_object_id": target_object_id,
                    "selected_grasp": selected_grasp,
                    "execution": execution_result,
                },
                next_recommendation="inspect_or_retry",
            )

        report = ExecutionReport(
            action_type="pick",
            status="success",
            reason="grasp_selected_via_anydex_and_executed" if executed else "grasp_selected_via_anydex",
            attempt_count=1,
            observed_effect={
                "scene_frame_id": frame.frame_id,
                "grasp_frame_id": frame.frame_id,
                "target_object_id": target_object_id,
                "target_container_id": target_container_id or None,
                "grasp_backend": "anydex",
                "candidate_grasps": candidate_grasps,
                "selected_grasp": selected_grasp,
                "backend_debug": backend_result.get("backend_debug") or {},
                "depth_path": frame.depth_path,
                "camera_intrinsics": frame.camera_intrinsics,
                "camera_extrinsics": frame.camera_extrinsics,
                "cropped_point_cloud": cropped_point_cloud,
                "planned_only": not executed,
                "execution": execution_result,
            },
            next_recommendation="verify" if executed else "record_execution_or_continue",
        )
        return report

    @staticmethod
    def _pick_or_place_report(
        *,
        action_type: str,
        target_object_id: str,
        target_container_id: str,
        retry_budget: int,
    ) -> ExecutionReport:
        missing = []
        if action_type in {"pick", "place"} and not target_object_id:
            missing.append("target_object_id")
        if action_type == "place" and not target_container_id:
            missing.append("target_container_id")
        if missing:
            return ExecutionReport(
                action_type=action_type,
                status="terminal_failure",
                reason=f"missing required fields: {', '.join(missing)}",
                attempt_count=1,
                observed_effect={},
                next_recommendation="replan",
            )
        return ExecutionReport(
            action_type=action_type,
            status="retryable_failure",
            reason=(
                f"{action_type}_target scaffold is wired, but the low-level "
                "grasp/place skill is not implemented yet."
            ),
            attempt_count=max(1, retry_budget),
            observed_effect={
                "target_object_id": target_object_id or None,
                "target_container_id": target_container_id or None,
            },
            next_recommendation="replan_or_attach_grasp_skill",
        )

    @staticmethod
    def _backend_failure_report(
        *,
        retry_budget: int,
        target_object_id: str,
        frame_id: str,
        reason: str,
    ) -> ExecutionReport:
        return ExecutionReport(
            action_type="pick",
            status="retryable_failure",
            reason=reason,
            attempt_count=max(1, retry_budget),
            observed_effect={
                "scene_frame_id": frame_id,
                "target_object_id": target_object_id or None,
            },
            next_recommendation="inspect_or_retry",
        )

    @staticmethod
    def _normalize_candidates(items: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            candidate.setdefault("candidate_id", f"candidate_{index}")
            candidate.setdefault("score", 0.0)
            candidate.setdefault("grasp_pose_cam", {})
            candidate.setdefault("approach_vector_cam", None)
            candidate.setdefault("contact_center_cam", None)
            candidate.setdefault("jaw_or_hand_width_hint", None)
            candidate.setdefault("grasp_width_m", candidate.get("jaw_or_hand_width_hint"))
            candidate.setdefault("grasp_depth_m", None)
            candidate.setdefault("hand_type", "dex1")
            candidate.setdefault("pregrasp_offset_m", 0.10)
            candidate.setdefault("lift_height_m", 0.12)
            candidate.setdefault("hand_shape_id", None)
            candidate.setdefault("rank", index)
            candidate.setdefault("validity_flags", {})
            candidate.setdefault("debug_source", "anydex")
            normalized.append(candidate)
        normalized.sort(key=lambda item: float(item.get("rank") or 0))
        return normalized

    @staticmethod
    def _normalize_selected_grasp(
        selected: Any,
        candidate_grasps: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if isinstance(selected, dict):
            result = dict(selected)
            result.setdefault("hand_type", "dex1")
            result.setdefault("grasp_width_m", result.get("jaw_or_hand_width_hint"))
            result.setdefault("pregrasp_offset_m", 0.10)
            result.setdefault("lift_height_m", 0.12)
            result.setdefault("debug_source", "anydex")
            return result
        if candidate_grasps:
            return dict(candidate_grasps[0])
        return None

    def _get_g1_controller(self) -> Any:
        if self._g1_controller is None:
            from roboclaw.embodied.embodiment.g1 import UnitreeG1Controller

            self._g1_controller = UnitreeG1Controller()
        return self._g1_controller

    def _extract_target_point_cloud(self, frame: PerceptionFrame, target_object: Any) -> list[list[float]]:
        if not frame.depth_path or not frame.camera_intrinsics:
            return []
        try:
            import numpy as np
        except Exception:
            return []
        try:
            depth_m = np.load(frame.depth_path)
        except Exception:
            return []
        intrinsics = CameraIntrinsics(**frame.camera_intrinsics)
        point_cloud = extract_detection_point_cloud(
            depth_m=depth_m,
            intrinsics=intrinsics,
            bbox_xyxy=target_object.bbox_xyxy,
            mask_rle=target_object.mask_rle,
        )
        return point_cloud_to_list(point_cloud)

    async def _execute_selected_grasp(
        self,
        *,
        frame: PerceptionFrame,
        target_object_id: str,
        selected_grasp: dict[str, Any],
    ) -> dict[str, Any] | None:
        setup = load_setup()
        config = dict(setup.get("unitree_g1", {}))
        if not config.get("enabled") or str(config.get("robot_variant") or "") != "g129_dex1":
            return None
        try:
            controller = self._get_g1_controller()
            if not (await controller.status(config)).get("connected"):
                await controller.connect(config)
            runner = G1Dex1GraspRunner(workspace=self._workspace, controller=controller)
            return await runner.run(
                config=config,
                frame=frame,
                target_object_id=target_object_id,
                selected_grasp=selected_grasp,
                arm_side="right",
            )
        except Exception as exc:
            return {
                "status": "failed",
                "failure_reason": f"dex1_execution_failed: {exc}",
                "target_object_id": target_object_id,
                "scene_frame_id": frame.frame_id,
                "selected_grasp": dict(selected_grasp),
            }
