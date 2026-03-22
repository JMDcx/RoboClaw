"""Simulation module — MuJoCo physics runtime for embodied simulation."""

import os
import sys

from roboclaw.embodied.simulation.launcher import SimulationLauncher
from roboclaw.embodied.simulation.mujoco_control_runtime import MujocoControlRuntime
from roboclaw.embodied.simulation.mujoco_runtime import MujocoRuntime
from roboclaw.embodied.simulation.session import SimulationSession
from roboclaw.embodied.simulation.viewer import SimulationViewer


def resolve_viewer_mode(mode: str) -> str:
    """Resolve 'auto' viewer mode to 'native' or 'web'."""
    if mode != "auto":
        return mode
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        return "native"
    return "web"


__all__ = ["MujocoControlRuntime", "MujocoRuntime", "SimulationLauncher", "SimulationSession", "SimulationViewer", "resolve_viewer_mode"]
