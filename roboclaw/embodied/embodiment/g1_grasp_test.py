"""Sim-only G1 + Inspire grasp test runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.embodied.embodiment.g1_camera import (
    transform_camera_translation_to_base,
    transform_camera_vector_to_base,
)
from roboclaw.embodied.embodiment.sim_feedback import IsaacSimFeedbackReader
from roboclaw.embodied.perception.service import PerceptionService
from roboclaw.embodied.perception.schemas import DetectedObject, PerceptionFrame
from roboclaw.embodied.unitree_xr import ArmCommandAdapter, ArmIKError, ArmIKRequest, G1ArmIKSolver


@dataclass(slots=True)
class G1InspireGraspTestRunner:
    """Run one structured sim-only grasp test."""

    workspace: Path
    controller: Any
    perception: PerceptionService | None = None
    feedback: IsaacSimFeedbackReader | None = None
    _solver: G1ArmIKSolver = field(init=False, repr=False)
    _adapter: ArmCommandAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.perception is None:
            self.perception = PerceptionService(workspace=self.workspace)
        if self.feedback is None:
            self.feedback = IsaacSimFeedbackReader()
        self._solver = G1ArmIKSolver()
        self._adapter = ArmCommandAdapter()

    async def run(
        self,
        *,
        config: dict[str, Any],
        target_object_id: str = "",
        arm_side: str = "right",
        camera_name: str = "head",
        grasp_mode: str = "top_grasp",
        pregrasp_offset_m: float = 0.12,
        descend_offset_m: float = 0.08,
        lift_height_m: float = 0.12,
        open_preset: str = "open",
        close_preset: str = "grasp",
    ) -> dict[str, Any]:
        if camera_name != "head":
            return self._failure("Only camera_name='head' is supported.", arm_side=arm_side)
        if grasp_mode != "top_grasp":
            return self._failure("Only grasp_mode='top_grasp' is supported in v1.", arm_side=arm_side)
        if str(config.get("robot_variant") or "") != "g129_inspire":
            return self._failure("g1_inspire_grasp_test requires robot_variant='g129_inspire'.", arm_side=arm_side)

        frame = self.perception.analyze_scene(camera_name=camera_name)
        if not frame.has_depth:
            return self._failure("Depth is required for g1_inspire_grasp_test.", arm_side=arm_side, frame=frame)

        target = self._select_target(frame, target_object_id=target_object_id)
        if isinstance(target, dict):
            target["arm_side"] = arm_side
            target["scene_frame_id"] = frame.frame_id
            return target

        verification_before = self.feedback.verify_object_lift()
        grasp_pose_cam = self._estimate_grasp_pose_cam(target)
        grasp_pose_base = self._camera_pose_to_base(grasp_pose_cam)
        pregrasp_pose_base = self._offset_pose(grasp_pose_base, dz=pregrasp_offset_m)
        lift_pose_base = self._offset_pose(grasp_pose_base, dz=lift_height_m)
        current_joint_positions = await self._get_current_joint_positions(config)

        try:
            pregrasp_solution = self._solver.solve(
                ArmIKRequest(
                    side=arm_side,
                    target_translation_base=pregrasp_pose_base["translation"],
                    approach_vector_base=grasp_pose_base["approach_vector"],
                    wrist_yaw_hint_rad=float(grasp_pose_base["wrist_yaw_hint_rad"]),
                    current_joint_positions=current_joint_positions,
                )
            )
            grasp_solution = self._solver.solve(
                ArmIKRequest(
                    side=arm_side,
                    target_translation_base=grasp_pose_base["translation"],
                    approach_vector_base=grasp_pose_base["approach_vector"],
                    wrist_yaw_hint_rad=float(grasp_pose_base["wrist_yaw_hint_rad"]),
                    current_joint_positions=pregrasp_solution.joint_positions,
                )
            )
            lift_solution = self._solver.solve(
                ArmIKRequest(
                    side=arm_side,
                    target_translation_base=lift_pose_base["translation"],
                    approach_vector_base=grasp_pose_base["approach_vector"],
                    wrist_yaw_hint_rad=float(grasp_pose_base["wrist_yaw_hint_rad"]),
                    current_joint_positions=grasp_solution.joint_positions,
                )
            )
        except ArmIKError as exc:
            return self._failure(str(exc), arm_side=arm_side, frame=frame, target_object_id=target.object_id)

        pregrasp_cmd = self._adapter.adapt(
            side=arm_side,
            current_joint_positions=current_joint_positions,
            target_joint_positions=pregrasp_solution.joint_positions,
        )
        grasp_cmd = self._adapter.adapt(
            side=arm_side,
            current_joint_positions=pregrasp_solution.joint_positions,
            target_joint_positions=grasp_solution.joint_positions,
        )
        lift_cmd = self._adapter.adapt(
            side=arm_side,
            current_joint_positions=grasp_solution.joint_positions,
            target_joint_positions=lift_solution.joint_positions,
        )

        try:
            await self.controller.move_joint(config, pregrasp_cmd, hold_seconds=1.0)
            await self.controller.hand_preset(config, open_preset, side=arm_side, hold_seconds=0.5)
            await self.controller.move_joint(config, grasp_cmd, hold_seconds=1.0)
            if descend_offset_m > 0:
                descend_pose = self._offset_pose(grasp_pose_base, dz=-descend_offset_m)
                descend_solution = self._solver.solve(
                    ArmIKRequest(
                        side=arm_side,
                        target_translation_base=descend_pose["translation"],
                        approach_vector_base=grasp_pose_base["approach_vector"],
                        wrist_yaw_hint_rad=float(grasp_pose_base["wrist_yaw_hint_rad"]),
                        current_joint_positions=grasp_solution.joint_positions,
                    )
                )
                descend_cmd = self._adapter.adapt(
                    side=arm_side,
                    current_joint_positions=grasp_solution.joint_positions,
                    target_joint_positions=descend_solution.joint_positions,
                )
                await self.controller.move_joint(config, descend_cmd, hold_seconds=0.8)
                grasp_cmd = descend_cmd
                grasp_solution = descend_solution
            await self.controller.hand_preset(config, close_preset, side=arm_side, hold_seconds=0.7)
            await asyncio.sleep(0.25)
            await self.controller.move_joint(config, lift_cmd, hold_seconds=1.0)
            await asyncio.sleep(0.25)
        except ArmIKError as exc:
            return self._failure(str(exc), arm_side=arm_side, frame=frame, target_object_id=target.object_id)
        except Exception as exc:
            return self._failure(
                f"Execution failed: {exc}",
                arm_side=arm_side,
                frame=frame,
                target_object_id=target.object_id,
            )

        verification_after = self.feedback.verify_object_lift()
        verification = {
            "object_height_before": verification_before.object_height_after,
            "object_height_after": verification_after.object_height_after,
            "height_delta": _delta(verification_before.object_height_after, verification_after.object_height_after),
            "reward_snapshot": verification_after.reward_snapshot,
            "success_boolean": verification_after.success_boolean,
        }

        return {
            "status": "success" if verification_after.success_boolean else "failed",
            "target_object_id": target.object_id,
            "arm_side": arm_side,
            "scene_frame_id": frame.frame_id,
            "grasp_pose_cam": grasp_pose_cam,
            "grasp_pose_base": grasp_pose_base,
            "pregrasp_joint_positions": pregrasp_cmd,
            "grasp_joint_positions": grasp_cmd,
            "lift_joint_positions": lift_cmd,
            "verification": verification,
            "failure_reason": None if verification_after.success_boolean else "verification_failed",
        }

    async def _get_current_joint_positions(self, config: dict[str, Any]) -> dict[str, float]:
        status = await self.controller.status(config)
        return {
            str(name): float(value)
            for name, value in (status.get("joint_positions") or {}).items()
        }

    def _select_target(self, frame: PerceptionFrame, *, target_object_id: str) -> DetectedObject | dict[str, Any]:
        candidates = [obj for obj in frame.objects if obj.pickable and obj.stable]
        pickable_objects = [obj for obj in frame.objects if obj.pickable]
        if target_object_id:
            for obj in pickable_objects:
                if obj.object_id == target_object_id:
                    if obj.stable:
                        return obj
                    detector_source = str(obj.attributes.get("detector_source") or "").strip().lower()
                    if detector_source == "sim_fallback":
                        return obj
                    return self._failure(
                        f"Target object '{target_object_id}' is pickable but not stable yet.",
                        available_object_ids=[obj.object_id for obj in pickable_objects],
                    )
            return self._failure(
                f"Target object '{target_object_id}' not found among pickable objects.",
                available_object_ids=[obj.object_id for obj in pickable_objects],
            )
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return self._failure(
                "Multiple stable pickable objects detected; specify target_object_id.",
                available_object_ids=[obj.object_id for obj in candidates],
            )
        sim_fallback_pickables = [
            obj
            for obj in pickable_objects
            if str(obj.attributes.get("detector_source") or "").strip().lower() == "sim_fallback"
        ]
        if len(sim_fallback_pickables) == 1:
            return sim_fallback_pickables[0]
        if len(sim_fallback_pickables) > 1:
            return self._failure(
                "Multiple sim_fallback pickable objects detected; specify target_object_id.",
                available_object_ids=[obj.object_id for obj in sim_fallback_pickables],
            )
        if pickable_objects:
            return self._failure(
                "No stable pickable object detected in the current scene.",
                available_object_ids=[obj.object_id for obj in pickable_objects],
            )
        return self._failure("No stable pickable object detected in the current scene.")

    def _estimate_grasp_pose_cam(self, target: DetectedObject) -> dict[str, Any]:
        centroid = target.attributes.get("centroid_3d")
        extent = target.attributes.get("extent_3d") or [0.04, 0.04, 0.08]
        principal_axis = target.attributes.get("principal_axis_3d") or [1.0, 0.0, 0.0]
        if not centroid or len(centroid) != 3:
            raise ArmIKError("Target object does not have a valid centroid_3d for grasp planning.")
        translation = [float(centroid[0]), float(centroid[1]), float(centroid[2] - 0.5 * float(extent[2]))]
        yaw_hint = float(np.arctan2(float(principal_axis[1]), float(principal_axis[0])))
        return {
            "translation": translation,
            "approach_vector": [0.0, 0.0, 1.0],
            "wrist_yaw_hint_rad": yaw_hint,
        }

    def _camera_pose_to_base(self, pose_cam: dict[str, Any]) -> dict[str, Any]:
        return {
            "translation": transform_camera_translation_to_base(pose_cam["translation"]),
            "approach_vector": transform_camera_vector_to_base(pose_cam["approach_vector"]),
            "wrist_yaw_hint_rad": float(pose_cam["wrist_yaw_hint_rad"]),
        }

    @staticmethod
    def _offset_pose(pose: dict[str, Any], *, dz: float) -> dict[str, Any]:
        result = dict(pose)
        translation = list(result["translation"])
        translation[2] += float(dz)
        result["translation"] = translation
        return result

    @staticmethod
    def _failure(reason: str, **extra: Any) -> dict[str, Any]:
        payload = {
            "status": "failed",
            "target_object_id": extra.pop("target_object_id", None),
            "arm_side": extra.pop("arm_side", None),
            "scene_frame_id": getattr(extra.pop("frame", None), "frame_id", None),
            "grasp_pose_cam": None,
            "grasp_pose_base": None,
            "pregrasp_joint_positions": None,
            "grasp_joint_positions": None,
            "lift_joint_positions": None,
            "verification": None,
            "failure_reason": reason,
        }
        payload.update(extra)
        return payload


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)
