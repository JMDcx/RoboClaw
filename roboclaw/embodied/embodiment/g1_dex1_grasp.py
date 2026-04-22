"""Sim-only G1 + Dex1 grasp runner driven by a selected grasp pose."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboclaw.embodied.embodiment.g1_camera import (
    transform_camera_translation_to_base,
    transform_camera_vector_to_base,
)
from roboclaw.embodied.embodiment.sim_feedback import IsaacSimFeedbackReader
from roboclaw.embodied.perception.schemas import DetectedObject, PerceptionFrame
from roboclaw.embodied.unitree_xr import ArmCommandAdapter, ArmIKError, ArmIKRequest, G1ArmIKSolver

_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)


@dataclass(slots=True)
class G1Dex1GraspRunner:
    """Execute one selected Dex1 grasp in Isaac Lab."""

    workspace: Path
    controller: Any
    feedback: IsaacSimFeedbackReader | None = None
    _solver: G1ArmIKSolver = field(init=False, repr=False)
    _adapter: ArmCommandAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feedback is None:
            self.feedback = IsaacSimFeedbackReader()
        self._solver = G1ArmIKSolver()
        self._adapter = ArmCommandAdapter(max_delta_rad=0.4)

    async def run(
        self,
        *,
        config: dict[str, Any],
        frame: PerceptionFrame,
        target_object_id: str,
        selected_grasp: dict[str, Any],
        arm_side: str = "right",
    ) -> dict[str, Any]:
        if arm_side != "right":
            return self._failure("Dex1 grasp v1 supports only arm_side='right'.", arm_side=arm_side, frame=frame)
        if str(config.get("robot_variant") or "") != "g129_dex1":
            return self._failure("g1_dex1_grasp requires robot_variant='g129_dex1'.", arm_side=arm_side, frame=frame)

        target = next((obj for obj in frame.objects if obj.object_id == target_object_id), None)
        if target is None:
            return self._failure(
                f"Target object '{target_object_id}' not found in the current scene.",
                arm_side=arm_side,
                frame=frame,
                target_object_id=target_object_id,
            )

        try:
            grasp_pose_cam = self._resolve_grasp_pose_cam(target, selected_grasp)
        except ArmIKError as exc:
            return self._failure(str(exc), arm_side=arm_side, frame=frame, target_object_id=target.object_id)

        grasp_pose_base = self._camera_pose_to_base(grasp_pose_cam)
        pregrasp_pose_base = self._offset_along_approach(
            grasp_pose_base,
            distance=float(selected_grasp.get("pregrasp_offset_m") or 0.10),
        )
        lift_pose_base = self._offset_vertical(
            grasp_pose_base,
            dz=float(selected_grasp.get("lift_height_m") or 0.12),
        )

        verification_before = self.feedback.verify_object_lift()
        current_joint_positions = await self._get_current_joint_positions(config)

        try:
            pregrasp_solution = self._solver.solve(
                ArmIKRequest(
                    side=arm_side,
                    target_translation_base=pregrasp_pose_base["translation"],
                    approach_vector_base=pregrasp_pose_base["approach_vector"],
                    wrist_yaw_hint_rad=float(pregrasp_pose_base["wrist_yaw_hint_rad"]),
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
                    approach_vector_base=lift_pose_base["approach_vector"],
                    wrist_yaw_hint_rad=float(lift_pose_base["wrist_yaw_hint_rad"]),
                    current_joint_positions=grasp_solution.joint_positions,
                )
            )
        except ArmIKError as exc:
            return self._failure(str(exc), arm_side=arm_side, frame=frame, target_object_id=target.object_id)

        try:
            await self.controller.gripper_open(config, side="right", hold_seconds=0.3)
            pregrasp_joint_positions = await self._execute_solution(
                config,
                arm_side=arm_side,
                current_joint_positions=current_joint_positions,
                target_joint_positions=pregrasp_solution.joint_positions,
            )
            grasp_joint_positions = await self._execute_solution(
                config,
                arm_side=arm_side,
                current_joint_positions=pregrasp_solution.joint_positions,
                target_joint_positions=grasp_solution.joint_positions,
            )
            await self.controller.gripper_close(config, side="right", hold_seconds=0.4)
            await asyncio.sleep(0.2)
            lift_joint_positions = await self._execute_solution(
                config,
                arm_side=arm_side,
                current_joint_positions=grasp_solution.joint_positions,
                target_joint_positions=lift_solution.joint_positions,
            )
            await asyncio.sleep(0.2)
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
            "selected_grasp": dict(selected_grasp),
            "grasp_pose_cam": grasp_pose_cam,
            "grasp_pose_base": grasp_pose_base,
            "pregrasp_joint_positions": pregrasp_joint_positions,
            "grasp_joint_positions": grasp_joint_positions,
            "lift_joint_positions": lift_joint_positions,
            "verification": verification,
            "failure_reason": None if verification_after.success_boolean else "verification_failed",
        }

    async def _get_current_joint_positions(self, config: dict[str, Any]) -> dict[str, float]:
        status = await self.controller.status(config)
        return {
            str(name): float(value)
            for name, value in (status.get("joint_positions") or {}).items()
        }

    async def _execute_solution(
        self,
        config: dict[str, Any],
        *,
        arm_side: str,
        current_joint_positions: dict[str, float],
        target_joint_positions: dict[str, float],
    ) -> dict[str, float]:
        commands = self._segment_joint_targets(
            arm_side=arm_side,
            current_joint_positions=current_joint_positions,
            target_joint_positions=target_joint_positions,
        )
        applied = dict(current_joint_positions)
        last_command: dict[str, float] = {}
        for command in commands:
            await self.controller.move_joint(config, command, hold_seconds=0.20)
            applied.update(command)
            last_command = command
        return last_command or {}

    def _segment_joint_targets(
        self,
        *,
        arm_side: str,
        current_joint_positions: dict[str, float],
        target_joint_positions: dict[str, float],
        max_step_rad: float = 0.35,
    ) -> list[dict[str, float]]:
        relevant_targets = {
            name: float(target_joint_positions[name])
            for name in _RIGHT_ARM_JOINTS
            if name in target_joint_positions
        }
        if not relevant_targets:
            return []
        max_delta = max(
            abs(float(relevant_targets[name]) - float(current_joint_positions.get(name, 0.0)))
            for name in relevant_targets
        )
        steps = max(1, int(math.ceil(max_delta / max_step_rad)))
        commands: list[dict[str, float]] = []
        staged_current = dict(current_joint_positions)
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            blended = {
                name: float(current_joint_positions.get(name, 0.0))
                + (float(relevant_targets[name]) - float(current_joint_positions.get(name, 0.0))) * alpha
                for name in relevant_targets
            }
            command = self._adapter.adapt(
                side=arm_side,
                current_joint_positions=staged_current,
                target_joint_positions=blended,
            )
            commands.append(command)
            staged_current.update(command)
        return commands

    def _resolve_grasp_pose_cam(self, target: DetectedObject, selected_grasp: dict[str, Any]) -> dict[str, Any]:
        pose = dict(selected_grasp.get("grasp_pose_cam") or {})
        translation = pose.get("translation") or selected_grasp.get("contact_center_cam")
        if translation is None:
            translation = target.attributes.get("centroid_3d")
        if not isinstance(translation, (list, tuple)) or len(translation) != 3:
            raise ArmIKError("Selected grasp does not provide a valid grasp_pose_cam.translation.")

        approach = (
            selected_grasp.get("approach_vector_cam")
            or pose.get("approach_vector")
            or pose.get("normal")
            or pose.get("z_axis")
            or [0.0, 0.0, 1.0]
        )
        if not isinstance(approach, (list, tuple)) or len(approach) != 3:
            raise ArmIKError("Selected grasp does not provide a valid approach vector.")
        approach_vector = _normalize_vector(np.asarray(approach, dtype=np.float64))

        wrist_yaw_hint_rad = selected_grasp.get("wrist_yaw_hint_rad")
        if wrist_yaw_hint_rad is None:
            reference = (
                pose.get("binormal")
                or pose.get("x_axis")
                or target.attributes.get("principal_axis_3d")
                or [1.0, 0.0, 0.0]
            )
            if not isinstance(reference, (list, tuple)) or len(reference) != 3:
                reference = [1.0, 0.0, 0.0]
            wrist_yaw_hint_rad = float(np.arctan2(float(reference[1]), float(reference[0])))
        return {
            "translation": [float(value) for value in translation],
            "approach_vector": [float(value) for value in approach_vector.tolist()],
            "wrist_yaw_hint_rad": float(wrist_yaw_hint_rad),
        }

    def _camera_pose_to_base(self, pose_cam: dict[str, Any]) -> dict[str, Any]:
        return {
            "translation": transform_camera_translation_to_base(pose_cam["translation"]),
            "approach_vector": transform_camera_vector_to_base(pose_cam["approach_vector"]),
            "wrist_yaw_hint_rad": float(pose_cam["wrist_yaw_hint_rad"]),
        }

    @staticmethod
    def _offset_along_approach(pose: dict[str, Any], *, distance: float) -> dict[str, Any]:
        result = dict(pose)
        translation = np.asarray(result["translation"], dtype=np.float64)
        approach_vector = _normalize_vector(np.asarray(result["approach_vector"], dtype=np.float64))
        result["translation"] = [float(value) for value in (translation - approach_vector * float(distance))]
        result["approach_vector"] = [float(value) for value in approach_vector.tolist()]
        return result

    @staticmethod
    def _offset_vertical(pose: dict[str, Any], *, dz: float) -> dict[str, Any]:
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
            "selected_grasp": extra.pop("selected_grasp", None),
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


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)
