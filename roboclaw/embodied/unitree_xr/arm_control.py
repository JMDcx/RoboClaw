"""Minimal arm command adaptation inspired by xr_teleoperate robot_arm.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_FULL_ARM_ORDER = {
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

_JOINT_LIMITS = {
    "shoulder_pitch": (-2.5, 2.5),
    "shoulder_roll": (-2.0, 2.0),
    "shoulder_yaw": (-2.7, 2.7),
    "elbow": (-0.1, 2.7),
    "wrist_roll": (-2.8, 2.8),
    "wrist_pitch": (-1.8, 1.8),
    "wrist_yaw": (-2.8, 2.8),
}


@dataclass(slots=True)
class ArmCommandAdapter:
    """Convert IK outputs into bounded lowcmd joint maps."""

    max_delta_rad: float = 1.25

    def adapt(
        self,
        *,
        side: str,
        current_joint_positions: dict[str, Any],
        target_joint_positions: dict[str, float],
    ) -> dict[str, float]:
        side_key = side.strip().lower()
        if side_key not in _FULL_ARM_ORDER:
            raise ValueError(f"Unsupported arm side '{side}'.")

        result: dict[str, float] = {}
        for joint_name in _FULL_ARM_ORDER[side_key]:
            if joint_name not in target_joint_positions:
                continue
            target_value = float(target_joint_positions[joint_name])
            current_value = float(current_joint_positions.get(joint_name, 0.0))
            delta = max(-self.max_delta_rad, min(self.max_delta_rad, target_value - current_value))
            bounded = current_value + delta
            limited = _clamp_joint(joint_name, bounded)
            result[joint_name] = limited
        return result


def _clamp_joint(joint_name: str, value: float) -> float:
    suffix = joint_name.split("_", 1)[1]
    low, high = _JOINT_LIMITS[suffix]
    return max(low, min(high, float(value)))
