"""Shared MuJoCo simulation session lifecycle."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from roboclaw.embodied.simulation.mujoco_runtime import MujocoRuntime


class SimulationSession:
    """Runs MuJoCo stepping alongside either a native or web viewer."""

    def __init__(
        self,
        runtime: MujocoRuntime,
        *,
        viewer_mode: str = "auto",
        viewer_port: int = 9878,
    ) -> None:
        self._runtime = runtime
        self._viewer_mode = viewer_mode
        self._viewer_port = int(viewer_port)

    def run(self, node: Any, lock: threading.RLock) -> None:
        mode = self._resolve_mode()
        if mode == "native":
            self._run_native(node, lock)
            return
        if mode == "web":
            self._run_web(node, lock)
            return
        raise ValueError(f"Unsupported viewer mode: {mode}")

    def _resolve_mode(self) -> str:
        from roboclaw.embodied.simulation import resolve_viewer_mode
        return resolve_viewer_mode(self._viewer_mode)

    def _run_native(self, node: Any, lock: threading.RLock) -> None:
        import mujoco
        import mujoco.viewer
        import rclpy

        model, data = self._runtime.model, self._runtime.data
        print("ROBOCLAW_SIM_VIEWER_MODE=native", flush=True)
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                with lock:
                    mujoco.mj_step(model, data)
                viewer.sync()
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

    def _run_web(self, node: Any, lock: threading.RLock) -> None:
        os.environ.setdefault("MUJOCO_GL", "osmesa")

        import mujoco
        import rclpy

        from roboclaw.embodied.simulation.viewer import SimulationViewer

        model, data = self._runtime.model, self._runtime.data
        viewer = SimulationViewer(self._runtime, port=self._viewer_port)
        viewer.start()
        print(f"ROBOCLAW_SIM_VIEWER_URL=http://0.0.0.0:{viewer.port}", flush=True)
        print("ROBOCLAW_SIM_VIEWER_MODE=web", flush=True)
        stop_event = threading.Event()

        def physics_loop() -> None:
            while not stop_event.is_set():
                step_start = time.time()
                with lock:
                    mujoco.mj_step(model, data)
                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

        physics_thread = threading.Thread(target=physics_loop, daemon=True)
        physics_thread.start()
        try:
            rclpy.spin(node)
        finally:
            stop_event.set()
            viewer.stop()
            physics_thread.join(timeout=2.0)
