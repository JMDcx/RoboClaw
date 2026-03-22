"""MuJoCo-backed control runtime with the same interface as So101FeetechRuntime.

This runtime plugs into the existing Ros2ControlSurfaceServer, making
simulation transparent to the ROS2 adapter layer above.
"""

from __future__ import annotations

from typing import Any

from roboclaw.embodied.simulation.mujoco_runtime import MujocoRuntime


class MujocoControlRuntime:
    """Drop-in replacement for hardware runtimes (So101FeetechRuntime, etc.)."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_mapping: dict[str, str] | None = None,
        gripper_actuator: str = "gripper",
        gripper_open_value: float = 1.0,
        gripper_close_value: float = 0.0,
    ) -> None:
        self._mujoco = MujocoRuntime(model_path, joint_mapping=joint_mapping)
        self._gripper_actuator = gripper_actuator
        self._gripper_open = gripper_open_value
        self._gripper_close = gripper_close_value

    @property
    def connected(self) -> bool:
        return self._mujoco.is_running

    def connect(self) -> None:
        self._mujoco.start()

    def disconnect(self) -> None:
        self._mujoco.stop()

    def open_gripper(self) -> dict[str, Any]:
        self._mujoco.set_joint_targets({self._gripper_actuator: self._gripper_open})
        self._mujoco.step()
        return self.snapshot()

    def close_gripper(self) -> dict[str, Any]:
        self._mujoco.set_joint_targets({self._gripper_actuator: self._gripper_close})
        self._mujoco.step()
        return self.snapshot()

    def go_home(self) -> dict[str, Any]:
        self._mujoco.reset()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        positions = self._mujoco.get_joint_positions() if self.connected else {}
        gripper_val = positions.get(self._gripper_actuator)
        gripper_pct = None
        if gripper_val is not None and (self._gripper_open - self._gripper_close) != 0:
            gripper_pct = (gripper_val - self._gripper_close) / (self._gripper_open - self._gripper_close)
            gripper_pct = max(0.0, min(1.0, gripper_pct)) * 100.0
        return {
            "connected": self.connected,
            "joint_positions": positions,
            "gripper_percent": gripper_pct,
            "simulator": "mujoco",
        }
