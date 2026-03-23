"""Embodied tool — bridges agent to the embodied robotics layer."""

from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool

_ACTIONS = [
    "doctor",
    "calibrate",
    "teleoperate",
    "record",
    "train",
    "run_policy",
    "job_status",
]

_DEFAULT_PORT = "/dev/ttyACM0"
_DEFAULT_CALIBRATION_DIR = Path("~/.roboclaw/workspace/embodied/calibration/so101").expanduser()
_DATASET_ROOT = Path("~/.roboclaw/workspace/embodied/datasets").expanduser()
_POLICY_OUTPUT = Path("~/.roboclaw/workspace/embodied/policies").expanduser()
_LOGS_DIR = Path("~/.roboclaw/workspace/embodied/jobs").expanduser()


class EmbodiedTool(Tool):
    """Control embodied robots via the agent."""

    @property
    def name(self) -> str:
        return "embodied"

    @property
    def description(self) -> str:
        return (
            "Control embodied robots — connect, calibrate, collect data, "
            "train policies, and run inference."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ACTIONS,
                    "description": "The action to perform.",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port for the robot.",
                },
                "calibration_dir": {
                    "type": "string",
                    "description": "Directory for calibration data.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Name for the dataset.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for recording.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record or run.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for recording.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of training steps.",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Path to a trained policy checkpoint.",
                },
                "job_id": {
                    "type": "string",
                    "description": "ID of a background training job.",
                },
                "device": {
                    "type": "string",
                    "description": "Device for training (default: cuda).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.learning.act import ACTPipeline
        from roboclaw.embodied.runner import LocalLeRobotRunner

        action = kwargs.get("action", "")
        port = kwargs.get("port", _DEFAULT_PORT)
        calibration_dir = kwargs.get("calibration_dir", str(_DEFAULT_CALIBRATION_DIR))
        runner = LocalLeRobotRunner()

        if action == "doctor":
            controller = SO101Controller()
            return await self._run(runner, controller.doctor())

        if action == "calibrate":
            controller = SO101Controller()
            return await self._run(runner, controller.calibrate(robot_port=port, calibration_dir=calibration_dir))

        if action == "teleoperate":
            controller = SO101Controller()
            return await self._run(runner, controller.teleoperate(robot_port=port, calibration_dir=calibration_dir))

        if action == "record":
            controller = SO101Controller()
            argv = controller.record(
                robot_port=port,
                calibration_dir=calibration_dir,
                dataset_name=kwargs.get("dataset_name", "default"),
                task=kwargs.get("task", "default_task"),
                num_episodes=kwargs.get("num_episodes", 10),
                fps=kwargs.get("fps", 30),
            )
            return await self._run(runner, argv)

        if action == "train":
            pipeline = ACTPipeline()
            return await self._handle_train(runner, pipeline, kwargs)

        if action == "run_policy":
            controller = SO101Controller()
            pipeline = ACTPipeline()
            checkpoint = kwargs.get("checkpoint_path") or pipeline.checkpoint_path(str(_POLICY_OUTPUT))
            argv = controller.run_policy(
                robot_port=port,
                calibration_dir=calibration_dir,
                checkpoint_path=checkpoint,
                num_episodes=kwargs.get("num_episodes", 1),
            )
            return await self._run(runner, argv)

        if action == "job_status":
            job_id = kwargs.get("job_id", "")
            status = await runner.job_status(job_id=job_id, log_dir=_LOGS_DIR)
            return "\n".join(f"{k}: {v}" for k, v in status.items())

        return f"Unknown action: {action}"

    @staticmethod
    async def _run(runner: Any, argv: list[str]) -> str:
        returncode, stdout, stderr = await runner.run(argv)
        if returncode != 0:
            return f"Command failed (exit {returncode}).\nstdout: {stdout}\nstderr: {stderr}"
        return stdout or "Done."

    @staticmethod
    async def _handle_train(runner: Any, pipeline: Any, kwargs: dict[str, Any]) -> str:
        dataset_name = kwargs.get("dataset_name", "default")
        dataset_path = str(_DATASET_ROOT / dataset_name)
        argv = pipeline.train(
            dataset_path=dataset_path,
            output_dir=str(_POLICY_OUTPUT),
            steps=kwargs.get("steps", 100_000),
            device=kwargs.get("device", "cuda"),
        )
        job_id = await runner.run_detached(argv=argv, log_dir=_LOGS_DIR)
        return f"Training started. Job ID: {job_id}"
