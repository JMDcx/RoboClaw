"""Launcher helpers for embodied MuJoCo simulation."""

from __future__ import annotations

from pathlib import Path

from roboclaw.embodied.builtins.model import BuiltinEmbodiment
from roboclaw.embodied.simulation.mujoco_runtime import MujocoRuntime


class SimulationLauncher:
    """Create and manage MuJoCo runtimes."""

    def __init__(self, model_path: str, joint_mapping: dict[str, str] | None = None) -> None:
        self.model_path = model_path
        self.joint_mapping = joint_mapping or {}

    def launch(self) -> MujocoRuntime:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Simulation model file not found: {model_path}")
        runtime = MujocoRuntime(str(model_path), joint_mapping=self.joint_mapping)
        runtime.start()
        return runtime

    def shutdown(self, runtime: MujocoRuntime) -> None:
        runtime.stop()

    @classmethod
    def from_builtin(cls, builtin: BuiltinEmbodiment) -> "SimulationLauncher":
        if builtin.sim_model_path is None:
            raise ValueError(f"Builtin embodiment '{builtin.id}' does not define sim_model_path.")
        return cls(model_path=builtin.sim_model_path, joint_mapping=builtin.sim_joint_mapping)
