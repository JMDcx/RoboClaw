"""Minimal vendored arm IK inspired by xr_teleoperate robot_arm_ik.py.

This is intentionally a compact internal adaptation for RoboClaw's sim-only
grasp test path. It keeps the public surface area small and returns joint-space
targets that can be sent through the existing G1 lowcmd controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ArmIKError(RuntimeError):
    """Raised when the sim-only arm IK cannot find a usable solution."""


@dataclass(slots=True)
class ArmIKRequest:
    """Pose request for one G1 arm."""

    side: str
    target_translation_base: list[float]
    approach_vector_base: list[float]
    wrist_yaw_hint_rad: float = 0.0
    current_joint_positions: dict[str, float] | None = None


@dataclass(slots=True)
class ArmIKSolution:
    """Joint-space result for one G1 arm."""

    joint_positions: dict[str, float]
    final_position_error_m: float
    iterations: int


class G1ArmIKSolver:
    """Small numerical IK solver for the G1 7DoF arm."""

    _LINK_LENGTHS = (0.28, 0.25, 0.12)
    _SHOULDER_POS = {
        "left": np.array([0.05, 0.20, 1.12], dtype=np.float64),
        "right": np.array([0.05, -0.20, 1.12], dtype=np.float64),
    }
    _JOINT_ORDER = {
        "left": (
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist_roll",
            "left_wrist_pitch",
            "left_wrist_yaw",
        ),
        "right": (
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
            "right_wrist_roll",
            "right_wrist_pitch",
            "right_wrist_yaw",
        ),
    }
    _JOINT_LIMITS = np.array(
        [
            (-2.5, 2.5),
            (-2.0, 2.0),
            (-2.7, 2.7),
            (-0.1, 2.7),
            (-2.8, 2.8),
            (-1.8, 1.8),
            (-2.8, 2.8),
        ],
        dtype=np.float64,
    )

    def solve(self, request: ArmIKRequest) -> ArmIKSolution:
        side = request.side.strip().lower()
        if side not in self._JOINT_ORDER:
            raise ArmIKError(f"Unsupported arm side '{request.side}'.")
        target = np.asarray(request.target_translation_base, dtype=np.float64)
        if target.shape != (3,):
            raise ArmIKError("target_translation_base must be a 3D vector.")

        seed = self._initial_seed(request, side)
        solution, error, iterations = self._iterate(side=side, target=target, seed=seed)
        if error > 0.08:
            raise ArmIKError(f"IK residual too large ({error:.3f} m).")

        q = solution.copy()
        q[4] = float(np.clip(self._wrist_roll_for_side(side), *self._JOINT_LIMITS[4]))
        q[5] = float(np.clip(self._wrist_pitch_for_approach(request.approach_vector_base), *self._JOINT_LIMITS[5]))
        q[6] = float(np.clip(request.wrist_yaw_hint_rad, *self._JOINT_LIMITS[6]))
        q = np.clip(q, self._JOINT_LIMITS[:, 0], self._JOINT_LIMITS[:, 1])
        joint_map = {
            joint_name: float(q[idx])
            for idx, joint_name in enumerate(self._JOINT_ORDER[side])
        }
        return ArmIKSolution(
            joint_positions=joint_map,
            final_position_error_m=float(error),
            iterations=iterations,
        )

    def _initial_seed(self, request: ArmIKRequest, side: str) -> np.ndarray:
        current = request.current_joint_positions or {}
        defaults = {
            "left_shoulder_pitch": 0.2,
            "left_shoulder_roll": 0.9,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 1.2,
            "left_wrist_roll": 0.0,
            "left_wrist_pitch": -0.4,
            "left_wrist_yaw": 0.0,
            "right_shoulder_pitch": -0.2,
            "right_shoulder_roll": -0.9,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 1.2,
            "right_wrist_roll": 0.0,
            "right_wrist_pitch": -0.4,
            "right_wrist_yaw": 0.0,
        }
        return np.array(
            [
                float(current.get(joint_name, defaults[joint_name]))
                for joint_name in self._JOINT_ORDER[side]
            ],
            dtype=np.float64,
        )

    def _iterate(self, *, side: str, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float, int]:
        q = seed.copy()
        for iteration in range(1, 81):
            current = self._fk_position(side, q)
            error_vec = target - current
            error = float(np.linalg.norm(error_vec))
            if error < 0.025:
                return q, error, iteration
            jacobian = self._position_jacobian(side, q)
            dq = np.linalg.pinv(jacobian) @ error_vec
            q += 0.6 * dq
            q = np.clip(q, self._JOINT_LIMITS[:, 0], self._JOINT_LIMITS[:, 1])
        final_error = float(np.linalg.norm(target - self._fk_position(side, q)))
        return q, final_error, 80

    def _position_jacobian(self, side: str, q: np.ndarray) -> np.ndarray:
        base = self._fk_position(side, q)
        jacobian = np.zeros((3, len(q)), dtype=np.float64)
        eps = 1e-3
        for idx in range(len(q)):
            perturbed = q.copy()
            perturbed[idx] += eps
            jacobian[:, idx] = (self._fk_position(side, perturbed) - base) / eps
        return jacobian

    def _fk_position(self, side: str, q: np.ndarray) -> np.ndarray:
        shoulder = self._SHOULDER_POS[side]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = shoulder
        transform = transform @ _rot_y(q[0]) @ _rot_x(q[1]) @ _rot_z(q[2])
        transform = transform @ _translate(self._LINK_LENGTHS[0], 0.0, 0.0)
        transform = transform @ _rot_y(q[3])
        transform = transform @ _translate(self._LINK_LENGTHS[1], 0.0, 0.0)
        transform = transform @ _rot_x(q[4]) @ _rot_y(q[5]) @ _rot_z(q[6])
        transform = transform @ _translate(self._LINK_LENGTHS[2], 0.0, 0.0)
        return transform[:3, 3]

    @staticmethod
    def _wrist_pitch_for_approach(approach_vector_base: list[float]) -> float:
        if len(approach_vector_base) != 3:
            return -0.6
        vec = np.asarray(approach_vector_base, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return -0.6
        vec /= norm
        return float(np.clip(np.arctan2(-vec[2], max(1e-6, vec[0])), -1.4, 0.4))

    @staticmethod
    def _wrist_roll_for_side(side: str) -> float:
        return -1.2 if side == "right" else 1.2


def _translate(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = [x, y, z]
    return matrix


def _rot_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rot_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0], [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rot_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
