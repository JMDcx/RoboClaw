"""Embodied tool - bridges agent to the embodied robotics layer."""

from __future__ import annotations

import json
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from roboclaw.agent.tools.base import Tool

_ACTIONS = [
    "doctor",
    "identify",
    "describe",
    "calibrate",
    "teleoperate",
    "record",
    "replay",
    "train",
    "run_policy",
    "job_status",
    "setup_show",
    "set_arm",
    "rename_arm",
    "remove_arm",
    "set_camera",
    "remove_camera",
    "g1_setup",
    "g1_connect",
    "g1_status",
    "g1_move_joint",
    "g1_go_named_pose",
    "g1_gripper_status",
    "g1_gripper_move",
    "g1_gripper_open",
    "g1_gripper_close",
    "g1_hand_status",
    "g1_hand_move",
    "g1_hand_preset",
    "g1_inspire_grasp_test",
]

_ACTION_DESCRIPTIONS = {
    "doctor": "Check LeRobot availability and show the current embodied setup.",
    "identify": "Launch the interactive arm-identification flow for detected serial ports.",
    "describe": "Explain adjustable parameters for a target embodied action.",
    "calibrate": "Calibrate one or more configured arms. If arms is omitted, calibrate every uncalibrated arm.",
    "teleoperate": "Run live teleoperation. Select arms with a comma-separated port list.",
    "record": "Record a dataset with one follower/leader pair or two pairs for bimanual capture.",
    "replay": "Replay a recorded dataset episode on one or two follower arms.",
    "train": "Start ACT training for a recorded dataset as a detached job.",
    "run_policy": "Run a trained policy on one follower arm and optionally cameras.",
    "job_status": "Inspect the status and recent logs for a detached training job.",
    "setup_show": "Show the embodied setup JSON with configured arms, cameras, and roots.",
    "set_arm": "Create or update one configured arm alias.",
    "rename_arm": "Rename an existing configured arm alias.",
    "remove_arm": "Remove one configured arm alias.",
    "set_camera": "Assign a scanned camera to a stable camera name.",
    "remove_camera": "Remove a configured camera.",
    "g1_gripper_status": "Inspect the latest Dex1 gripper DDS state and command targets for Unitree G1.",
    "g1_gripper_move": "Send a normalized Dex1 close amount for Unitree G1 simulation.",
    "g1_gripper_open": "Open the right-side Dex1 gripper in Unitree G1 simulation.",
    "g1_gripper_close": "Close the right-side Dex1 gripper in Unitree G1 simulation.",
    "g1_hand_status": "Inspect the latest Inspire hand DDS state and command targets for Unitree G1.",
    "g1_hand_move": "Send normalized Inspire hand joint targets for Unitree G1 simulation.",
    "g1_hand_preset": "Execute a named Inspire hand preset such as open or grasp.",
    "g1_inspire_grasp_test": "Run a sim-only RGB-D grasp test for G1 + Inspire in Isaac Lab.",
}

_LOGS_DIR = Path("~/.roboclaw/workspace/embodied/jobs").expanduser()
_NO_TTY_MSG = "This action requires a local terminal. Run: roboclaw agent"
_BIMANUAL_ID = "bimanual"
_DEFAULT_REPLAY_ROOT = Path("~/.cache/huggingface/lerobot").expanduser()
_DATASET_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


class EmbodiedTool(Tool):
    """Control embodied robots via the agent."""

    def __init__(self, tty_handoff: Any = None):
        self._tty_handoff = tty_handoff
        self._g1_controller = None

    @property
    def name(self) -> str:
        return "embodied"

    @property
    def description(self) -> str:
        return (
            "Control embodied robots — connect, calibrate, collect data, train policies, and run inference. "
            "ALWAYS use this tool for robot/arm/hardware questions. "
            "NEVER use exec for /dev queries. "
            "Use setup_show to view current config. "
            "Use set_arm(name, arm_type, port) to add/update arms by alias. "
            "Use remove_arm(name) to remove an arm. "
            "Use set_camera/remove_camera to configure cameras (picks from scanned_cameras by index). "
            "For teleoperate/record, specify follower_names and leader_names (comma-separated aliases). "
            "1+1 = single arm, 2+2 = bimanual. "
            "For Unitree G1 Isaac Lab simulation, use g1_setup, g1_connect, g1_status, g1_move_joint, g1_go_named_pose, "
            "g1_gripper_status, g1_gripper_move, g1_gripper_open, g1_gripper_close, g1_hand_status, g1_hand_move, "
            "g1_hand_preset, and g1_inspire_grasp_test. "
            "Actions: setup_show, identify, describe, calibrate, teleoperate, record, replay, train, run_policy, "
            "job_status, set_arm, rename_arm, remove_arm, set_camera, remove_camera, g1_setup, g1_connect, "
            "g1_status, g1_move_joint, g1_go_named_pose, g1_gripper_status, g1_gripper_move, g1_gripper_open, "
            "g1_gripper_close, g1_hand_status, g1_hand_move, g1_hand_preset, g1_inspire_grasp_test."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        params = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ACTIONS,
                    "description": "The action to perform.",
                },
                "target_action": {
                    "type": "string",
                    "description": "Action name to describe.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for record, replay, or train.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for recording.",
                },
                "use_cameras": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether recording or policy execution should include configured cameras.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record or run.",
                },
                "episode": {
                    "type": "integer",
                    "description": "Episode index to replay.",
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
                "alias": {
                    "type": "string",
                    "description": "Arm alias for set_arm, rename_arm, or remove_arm.",
                },
                "new_alias": {
                    "type": "string",
                    "description": "New arm alias for rename_arm.",
                },
                "arm_type": {
                    "type": "string",
                    "enum": ["so101_follower", "so101_leader"],
                    "description": "Arm hardware type for set_arm.",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path for set_arm.",
                },
                "camera_name": {
                    "type": "string",
                    "description": "Camera name like front or side.",
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Index into scanned_cameras for set_camera.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
                },
                "network_interface": {
                    "type": "string",
                    "description": "DDS network interface for Unitree G1 simulation.",
                },
                "dds_domain": {
                    "type": "integer",
                    "description": "DDS domain for Unitree G1 simulation.",
                    "minimum": 0,
                },
                "joint_positions": {
                    "type": "object",
                    "description": "Joint position map for G1 move_joint. May also be passed as a JSON string.",
                },
                "pose_name": {
                    "type": "string",
                    "description": "Named G1 pose to execute: home, ready, or folded.",
                },
                "hold_seconds": {
                    "type": "number",
                    "description": "How long to keep publishing a G1 command.",
                    "minimum": 0.02,
                },
                "close_amount": {
                    "type": "number",
                    "description": "Normalized close amount for the Dex1 gripper. 0=open, 1=closed.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "robot_variant": {
                    "type": "string",
                    "enum": ["g129_dex1", "g129_inspire"],
                    "description": "Unitree G1 sim variant for g1_setup.",
                },
                "side": {
                    "type": "string",
                    "enum": ["left", "right", "both"],
                    "description": "Which Inspire hand side to command.",
                },
                "hand_positions": {
                    "type": "object",
                    "description": "Normalized Inspire hand position map for G1 hand control. May also be passed as a JSON string.",
                },
                "preset_name": {
                    "type": "string",
                    "description": "Named G1 hand preset to execute: open, grasp, pinch, tripod, or relax.",
                },
                "target_object_id": {
                    "type": "string",
                    "description": "Optional target object_id from perception for g1_inspire_grasp_test.",
                },
                "arm_side": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Which G1 arm to use for the Inspire grasp test.",
                },
                "grasp_mode": {
                    "type": "string",
                    "description": "Grasp mode for g1_inspire_grasp_test. v1 supports top_grasp.",
                },
                "pregrasp_offset_m": {
                    "type": "number",
                    "description": "Vertical pregrasp offset in meters for g1_inspire_grasp_test.",
                },
                "descend_offset_m": {
                    "type": "number",
                    "description": "Vertical descend offset in meters for g1_inspire_grasp_test.",
                },
                "lift_height_m": {
                    "type": "number",
                    "description": "Lift height in meters for g1_inspire_grasp_test.",
                },
                "open_preset": {
                    "type": "string",
                    "description": "Hand preset to open the Inspire hand before grasping.",
                },
                "close_preset": {
                    "type": "string",
                    "description": "Hand preset to close the Inspire hand during grasping.",
                },
            },
            "required": ["action"],
        }
        params["properties"].pop("port", None)
        return params

    async def execute(self, **kwargs: Any) -> str:
        from roboclaw.embodied.setup import ensure_setup, load_setup

        action = kwargs.get("action", "")

        if action == "setup_show":
            return json.dumps(load_setup(), indent=2, ensure_ascii=False)
        if action == "describe":
            return self._do_describe(kwargs)
        if action in {"set_arm", "rename_arm", "remove_arm", "set_camera", "remove_camera", "g1_setup"}:
            return self._handle_setup_action(action, kwargs)

        setup = ensure_setup()

        try:
            if action == "doctor":
                return await self._do_doctor(setup)
            if action == "g1_connect":
                return await self._do_g1_connect(setup)
            if action == "g1_status":
                return await self._do_g1_status(setup)
            if action == "g1_move_joint":
                return await self._do_g1_move_joint(setup, kwargs)
            if action == "g1_go_named_pose":
                return await self._do_g1_go_named_pose(setup, kwargs)
            if action == "g1_gripper_status":
                return await self._do_g1_gripper_status(setup)
            if action == "g1_gripper_move":
                return await self._do_g1_gripper_move(setup, kwargs)
            if action == "g1_gripper_open":
                return await self._do_g1_gripper_open(setup, kwargs)
            if action == "g1_gripper_close":
                return await self._do_g1_gripper_close(setup, kwargs)
            if action == "g1_hand_status":
                return await self._do_g1_hand_status(setup)
            if action == "g1_hand_move":
                return await self._do_g1_hand_move(setup, kwargs)
            if action == "g1_hand_preset":
                return await self._do_g1_hand_preset(setup, kwargs)
            if action == "g1_inspire_grasp_test":
                return await self._do_g1_inspire_grasp_test(setup, kwargs)
            if action == "identify":
                return await self._do_identify(setup)
            if action == "calibrate":
                return await self._do_calibrate(setup, kwargs)
            if action == "teleoperate":
                return await self._do_teleoperate(setup, kwargs)
            if action == "record":
                return await self._do_record(setup, kwargs)
            if action == "replay":
                return await self._do_replay(setup, kwargs)
            if action == "train":
                return await self._do_train(setup, kwargs)
            if action == "run_policy":
                return await self._do_run_policy(setup, kwargs)
            if action == "job_status":
                return await self._do_job_status(kwargs)
        except ActionError as exc:
            return str(exc)
        return f"Unknown action: {action}"

    def _do_describe(self, kwargs: dict[str, Any]) -> str:
        target_action = kwargs.get("target_action", "")
        if not target_action:
            return json.dumps(_ACTION_DESCRIPTIONS, indent=2, ensure_ascii=False)
        if target_action not in _ACTION_DESCRIPTIONS:
            return f"Unknown target_action: {target_action}"
        return f"{target_action}: {_ACTION_DESCRIPTIONS[target_action]}"

    def _handle_setup_action(self, action: str, kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.setup import (
            clear_unitree_g1,
            remove_arm,
            remove_camera,
            rename_arm,
            set_arm,
            set_camera,
            set_unitree_g1,
        )

        if action == "g1_setup":
            return self._do_g1_setup(kwargs, set_unitree_g1, clear_unitree_g1)
        if action == "set_arm":
            return self._do_set_arm(kwargs, set_arm)
        if action == "rename_arm":
            return self._do_rename_arm(kwargs, rename_arm)
        if action == "remove_arm":
            return self._do_remove_arm(kwargs, remove_arm)
        if action == "set_camera":
            return self._do_set_camera(kwargs, set_camera)
        return self._do_remove_camera(kwargs, remove_camera)

    @staticmethod
    def _do_set_arm(kwargs: dict[str, Any], fn: Any) -> str:
        from roboclaw.embodied.setup import arm_display_name, find_arm

        alias = kwargs.get("alias", "")
        arm_type = kwargs.get("arm_type", "")
        port = kwargs.get("port", "")
        if not all([alias, arm_type, port]):
            return "set_arm requires alias, arm_type, and port."
        updated = fn(alias, arm_type, port)
        arm = find_arm(updated["arms"], alias)
        display = arm_display_name(arm)
        return f"Arm '{display}' configured.\n{json.dumps(arm, indent=2)}"

    @staticmethod
    def _do_remove_arm(kwargs: dict[str, Any], fn: Any) -> str:
        alias = kwargs.get("alias", "")
        if not alias:
            return "remove_arm requires alias."
        fn(alias)
        return f"Arm '{alias}' removed."

    @staticmethod
    def _do_rename_arm(kwargs: dict[str, Any], fn: Any) -> str:
        from roboclaw.embodied.setup import find_arm

        old_alias = kwargs.get("alias", "")
        new_alias = kwargs.get("new_alias", "")
        if not old_alias or not new_alias:
            return "rename_arm requires alias and new_alias."
        updated = fn(old_alias, new_alias)
        arm = find_arm(updated["arms"], new_alias)
        return f"Arm renamed from '{old_alias}' to '{new_alias}'.\n{json.dumps(arm, indent=2)}"

    @staticmethod
    def _do_set_camera(kwargs: dict[str, Any], fn: Any) -> str:
        name = kwargs.get("camera_name", "")
        index = kwargs.get("camera_index")
        if not name or index is None:
            return "set_camera requires camera_name and camera_index."
        updated = fn(name, index)
        return f"Camera '{name}' configured.\n{json.dumps(updated['cameras'][name], indent=2)}"

    @staticmethod
    def _do_g1_setup(kwargs: dict[str, Any], set_fn: Any, clear_fn: Any) -> str:
        network_interface = str(kwargs.get("network_interface", "")).strip()
        if not network_interface:
            return "g1_setup requires network_interface."
        dds_domain = kwargs.get("dds_domain", 1)
        if isinstance(dds_domain, str):
            try:
                dds_domain = int(dds_domain)
            except ValueError:
                return "g1_setup requires dds_domain to be an integer."
        if dds_domain is None:
            dds_domain = 1
        robot_variant = str(kwargs.get("robot_variant") or "g129_dex1").strip() or "g129_dex1"
        try:
            updated = set_fn(
                network_interface=network_interface,
                dds_domain=dds_domain,
                enabled=True,
                connected=False,
                mode="sim",
                robot_variant=robot_variant,
                motion_source="lowcmd",
                sim_runtime="isaaclab",
            )
        except ValueError as exc:
            return str(exc)
        return f"Unitree G1 simulation configured.\n{json.dumps(updated['unitree_g1'], indent=2)}"

    @staticmethod
    def _do_remove_camera(kwargs: dict[str, Any], fn: Any) -> str:
        name = kwargs.get("camera_name", "")
        if not name:
            return "remove_camera requires camera_name."
        fn(name)
        return f"Camera '{name}' removed."

    async def _do_doctor(self, setup: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner

        result = await self._run(LocalLeRobotRunner(), SO101Controller().doctor())
        return result + f"\n\nCurrent setup:\n{json.dumps(setup, indent=2, ensure_ascii=False)}"

    async def _do_identify(self, setup: dict[str, Any]) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        if not self._tty_handoff:
            return _NO_TTY_MSG
        ports = setup.get("scanned_ports", [])
        if not ports:
            return "No serial ports detected."
        argv = [sys.executable, "-m", "roboclaw.embodied.identify", json.dumps(ports)]
        rc = await self._run_tty(LocalLeRobotRunner(), argv, "identify-arms")
        if rc == 0:
            return "Arm identification complete."
        return f"Arm identification failed (exit {rc})."

    def _get_g1_controller(self) -> Any:
        if self._g1_controller is None:
            from roboclaw.embodied.embodiment.g1 import UnitreeG1Controller

            self._g1_controller = UnitreeG1Controller()
        return self._g1_controller

    async def _do_g1_connect(self, setup: dict[str, Any]) -> str:
        from roboclaw.embodied.setup import set_unitree_g1

        config = dict(setup.get("unitree_g1", {}))
        if not config.get("enabled"):
            return "Unitree G1 is not configured. Use g1_setup first."
        try:
            result = await self._get_g1_controller().connect(config)
            set_unitree_g1(connected=True)
        except Exception as exc:
            return f"G1 connect failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_status(self, setup: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        result = await self._get_g1_controller().status(config)
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_move_joint(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        joint_positions = kwargs.get("joint_positions")
        if isinstance(joint_positions, str):
            try:
                joint_positions = json.loads(joint_positions)
            except json.JSONDecodeError as exc:
                return f"joint_positions must be valid JSON: {exc}"
        if not isinstance(joint_positions, dict) or not joint_positions:
            return "g1_move_joint requires joint_positions as a non-empty object."
        try:
            result = await self._get_g1_controller().move_joint(
                config,
                joint_positions,
                hold_seconds=kwargs.get("hold_seconds", 2.0),
            )
        except Exception as exc:
            return f"G1 move_joint failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_go_named_pose(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        pose_name = str(kwargs.get("pose_name") or "").strip()
        if not pose_name:
            return "g1_go_named_pose requires pose_name."
        try:
            result = await self._get_g1_controller().go_named_pose(
                config,
                pose_name,
                hold_seconds=kwargs.get("hold_seconds", 4.0),
            )
        except Exception as exc:
            return f"G1 go_named_pose failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_gripper_status(self, setup: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        try:
            result = await self._get_g1_controller().gripper_status(config)
        except Exception as exc:
            return f"G1 gripper_status failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_gripper_move(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        if "close_amount" not in kwargs:
            return "g1_gripper_move requires close_amount."
        try:
            result = await self._get_g1_controller().gripper_move(
                config,
                kwargs.get("close_amount"),
                side=kwargs.get("side", "right"),
                hold_seconds=kwargs.get("hold_seconds", 0.5),
            )
        except Exception as exc:
            return f"G1 gripper_move failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_gripper_open(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        try:
            result = await self._get_g1_controller().gripper_open(
                config,
                side=kwargs.get("side", "right"),
                hold_seconds=kwargs.get("hold_seconds", 0.5),
            )
        except Exception as exc:
            return f"G1 gripper_open failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_gripper_close(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        try:
            result = await self._get_g1_controller().gripper_close(
                config,
                side=kwargs.get("side", "right"),
                hold_seconds=kwargs.get("hold_seconds", 0.5),
            )
        except Exception as exc:
            return f"G1 gripper_close failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_hand_status(self, setup: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        try:
            result = await self._get_g1_controller().hand_status(config)
        except Exception as exc:
            return f"G1 hand_status failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_hand_move(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        hand_positions = kwargs.get("hand_positions")
        if isinstance(hand_positions, str):
            try:
                hand_positions = json.loads(hand_positions)
            except json.JSONDecodeError as exc:
                return f"hand_positions must be valid JSON: {exc}"
        if not isinstance(hand_positions, dict) or not hand_positions:
            return "g1_hand_move requires hand_positions as a non-empty object."
        try:
            result = await self._get_g1_controller().hand_move(
                config,
                hand_positions,
                side=kwargs.get("side", "both"),
                hold_seconds=kwargs.get("hold_seconds", 0.5),
            )
        except Exception as exc:
            return f"G1 hand_move failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_hand_preset(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        preset_name = str(kwargs.get("preset_name") or "").strip()
        if not preset_name:
            return "g1_hand_preset requires preset_name."
        try:
            result = await self._get_g1_controller().hand_preset(
                config,
                preset_name,
                side=kwargs.get("side", "both"),
                hold_seconds=kwargs.get("hold_seconds", 0.5),
            )
        except Exception as exc:
            return f"G1 hand_preset failed: {exc}"
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_g1_inspire_grasp_test(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        config = dict(setup.get("unitree_g1", {}))
        if not config.get("enabled"):
            return "Unitree G1 is not configured. Use g1_setup first."
        from roboclaw.embodied.embodiment.g1_grasp_test import G1InspireGraspTestRunner

        runner = G1InspireGraspTestRunner(
            workspace=Path.cwd(),
            controller=self._get_g1_controller(),
        )
        try:
            result = await runner.run(
                config=config,
                target_object_id=str(kwargs.get("target_object_id") or "").strip(),
                arm_side=str(kwargs.get("arm_side") or "right").strip() or "right",
                camera_name=str(kwargs.get("camera_name") or "head").strip() or "head",
                grasp_mode=str(kwargs.get("grasp_mode") or "top_grasp").strip() or "top_grasp",
                pregrasp_offset_m=float(kwargs.get("pregrasp_offset_m", 0.12)),
                descend_offset_m=float(kwargs.get("descend_offset_m", 0.08)),
                lift_height_m=float(kwargs.get("lift_height_m", 0.12)),
                open_preset=str(kwargs.get("open_preset") or "open").strip() or "open",
                close_preset=str(kwargs.get("close_preset") or "grasp").strip() or "grasp",
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "failure_reason": f"g1_inspire_grasp_test crashed: {exc}",
            }
        return json.dumps(result, indent=2, ensure_ascii=False)

    async def _do_calibrate(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.setup import arm_display_name, mark_arm_calibrated

        configured = setup.get("arms", [])
        if not configured:
            return "No arms configured."
        selected = self._resolve_action_arms(setup, kwargs)
        targets = selected if kwargs.get("arms", "") else [arm for arm in selected if not arm.get("calibrated")]
        if not targets:
            return "All arms are already calibrated."
        if not self._tty_handoff:
            return _NO_TTY_MSG
        controller = SO101Controller()
        runner = LocalLeRobotRunner()
        succeeded = 0
        failed = 0
        results: list[str] = []
        for arm in targets:
            display = arm_display_name(arm)
            argv = controller.calibrate(
                arm["type"],
                arm["port"],
                arm.get("calibration_dir", ""),
                _arm_id(arm),
            )
            rc = await self._run_tty(runner, argv, f"Calibrating: {display}")
            if _is_interrupted(rc):
                return "interrupted"
            if rc == 0:
                succeeded += 1
                mark_arm_calibrated(arm["alias"])
                results.append(f"{display}: OK")
                continue
            failed += 1
            results.append(f"{display}: FAILED (exit {rc})")
        return (
            f"{succeeded} succeeded, {failed} failed.\n"
            + "\n".join(results)
            + "\nNote: wrist_roll is auto-calibrated by LeRobot (expected)."
        )

    async def _do_teleoperate(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.setup import arm_display_name

        if not self._tty_handoff:
            return _NO_TTY_MSG
        grouped = _group_arms(self._resolve_action_arms(setup, kwargs))
        error = _validate_pairing(grouped["followers"], grouped["leaders"])
        if error:
            return error
        controller = SO101Controller()
        followers = grouped["followers"]
        leaders = grouped["leaders"]
        if len(followers) == 1:
            follower = followers[0]
            leader = leaders[0]
            argv = controller.teleoperate(
                robot_type=follower["type"],
                robot_port=follower["port"],
                robot_cal_dir=follower["calibration_dir"],
                robot_id=_arm_id(follower),
                teleop_type=leader["type"],
                teleop_port=leader["port"],
                teleop_cal_dir=leader["calibration_dir"],
                teleop_id=_arm_id(leader),
            )
            label = f"lerobot-teleoperate ({arm_display_name(follower)} + {arm_display_name(leader)})"
            rc = await self._run_tty(LocalLeRobotRunner(), argv, label)
            if _is_interrupted(rc):
                return "interrupted"
            return "Teleoperation finished." if rc == 0 else f"Teleoperation failed (exit {rc})."
        with _bimanual_cal_dirs(followers, leaders) as (robot_dir, teleop_dir):
            argv = controller.teleoperate_bimanual(
                robot_id=_BIMANUAL_ID,
                robot_cal_dir=robot_dir,
                left_robot=followers[0],
                right_robot=followers[1],
                teleop_id=_BIMANUAL_ID,
                teleop_cal_dir=teleop_dir,
                left_teleop=leaders[0],
                right_teleop=leaders[1],
            )
            rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-teleoperate (bimanual)")
        if _is_interrupted(rc):
            return "interrupted"
        return "Teleoperation finished." if rc == 0 else f"Teleoperation failed (exit {rc})."

    async def _do_record(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner

        if not self._tty_handoff:
            return _NO_TTY_MSG
        grouped = _group_arms(self._resolve_action_arms(setup, kwargs))
        error = _validate_pairing(grouped["followers"], grouped["leaders"])
        if error:
            return error
        dataset_name = kwargs.get("dataset_name", "default")
        error = _validate_dataset_name(dataset_name)
        if error:
            return error
        controller = SO101Controller()
        cameras = {} if kwargs.get("use_cameras") is False else self._resolve_cameras(setup)
        dataset_root = _dataset_root(setup)
        record_kwargs = {
            "cameras": cameras,
            "repo_id": f"local/{dataset_name}",
            "task": kwargs.get("task", "default_task"),
            "dataset_root": str(dataset_root),
            "push_to_hub": False,
            "fps": kwargs.get("fps", 30),
            "num_episodes": kwargs.get("num_episodes", 10),
        }
        followers = grouped["followers"]
        leaders = grouped["leaders"]
        if len(followers) == 1:
            follower = followers[0]
            leader = leaders[0]
            argv = controller.record(
                robot_type=follower["type"],
                robot_port=follower["port"],
                robot_cal_dir=follower["calibration_dir"],
                robot_id=_arm_id(follower),
                teleop_type=leader["type"],
                teleop_port=leader["port"],
                teleop_cal_dir=leader["calibration_dir"],
                teleop_id=_arm_id(leader),
                **record_kwargs,
            )
            rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-record")
            if _is_interrupted(rc):
                return "interrupted"
            return "Recording finished." if rc == 0 else f"Recording failed (exit {rc})."
        with _bimanual_cal_dirs(followers, leaders) as (robot_dir, teleop_dir):
            argv = controller.record_bimanual(
                robot_id=_BIMANUAL_ID,
                robot_cal_dir=robot_dir,
                left_robot=followers[0],
                right_robot=followers[1],
                teleop_id=_BIMANUAL_ID,
                teleop_cal_dir=teleop_dir,
                left_teleop=leaders[0],
                right_teleop=leaders[1],
                **record_kwargs,
            )
            rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-record")
        if _is_interrupted(rc):
            return "interrupted"
        return "Recording finished." if rc == 0 else f"Recording failed (exit {rc})."

    async def _do_replay(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner

        if not self._tty_handoff:
            return _NO_TTY_MSG
        selected = self._resolve_action_arms(setup, kwargs)
        grouped = _group_arms(selected)
        if kwargs.get("arms", "") and grouped["leaders"]:
            return "Replay only supports follower arms. Remove leader arm ports from arms."
        followers = grouped["followers"]
        if not followers:
            return "No follower arm configured."
        if len(followers) not in {1, 2}:
            return f"Unsupported follower arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
        dataset_name = kwargs.get("dataset_name", "default")
        error = _validate_dataset_name(dataset_name)
        if error:
            return error
        dataset_root = _dataset_root(setup, fallback=_DEFAULT_REPLAY_ROOT)
        episode = kwargs.get("episode", 0)
        controller = SO101Controller()
        if len(followers) == 1:
            follower = followers[0]
            argv = controller.replay(
                robot_type=follower["type"],
                robot_port=follower["port"],
                robot_cal_dir=follower["calibration_dir"],
                robot_id=_arm_id(follower),
                repo_id=f"local/{dataset_name}",
                dataset_root=str(dataset_root),
                episode=episode,
            )
            rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-replay")
            if _is_interrupted(rc):
                return "interrupted"
            return "Replay finished." if rc == 0 else f"Replay failed (exit {rc})."
        with _bimanual_cal_dirs(followers, []) as (robot_dir, _):
            argv = controller.replay_bimanual(
                robot_id=_BIMANUAL_ID,
                robot_cal_dir=robot_dir,
                left_robot=followers[0],
                right_robot=followers[1],
                repo_id=f"local/{dataset_name}",
                dataset_root=str(dataset_root),
                episode=episode,
            )
            rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-replay (bimanual)")
        if _is_interrupted(rc):
            return "interrupted"
        return "Replay finished." if rc == 0 else f"Replay failed (exit {rc})."

    async def _do_train(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.learning.act import ACTPipeline
        from roboclaw.embodied.runner import LocalLeRobotRunner

        dataset_name = kwargs.get("dataset_name", "default")
        error = _validate_dataset_name(dataset_name)
        if error:
            return error
        dataset_root = _dataset_root(setup)
        policies_root = setup.get("policies", {}).get("root", "")
        argv = ACTPipeline().train(
            repo_id=f"local/{dataset_name}",
            dataset_root=str(dataset_root),
            output_dir=policies_root,
            steps=kwargs.get("steps", 100_000),
            device=kwargs.get("device", "cuda"),
        )
        job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_LOGS_DIR)
        return f"Training started. Job ID: {job_id}"

    async def _do_run_policy(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.learning.act import ACTPipeline
        from roboclaw.embodied.runner import LocalLeRobotRunner

        grouped = _group_arms(self._resolve_action_arms(setup, kwargs))
        followers = grouped["followers"]
        if not followers:
            return "No follower arm configured."
        if len(followers) != 1:
            return "run_policy requires exactly 1 follower arm. Provide arms with a single follower port."
        follower = followers[0]
        cameras = {} if kwargs.get("use_cameras") is False else self._resolve_cameras(setup)
        policies_root = setup.get("policies", {}).get("root", "")
        checkpoint = kwargs.get("checkpoint_path") or ACTPipeline().checkpoint_path(policies_root)
        argv = SO101Controller().run_policy(
            robot_type=follower["type"],
            robot_port=follower["port"],
            robot_cal_dir=follower["calibration_dir"],
            robot_id=_arm_id(follower),
            cameras=cameras,
            policy_path=checkpoint,
            num_episodes=kwargs.get("num_episodes", 1),
        )
        return await self._run(LocalLeRobotRunner(), argv)

    async def _do_job_status(self, kwargs: dict[str, Any]) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        job_id = kwargs.get("job_id", "")
        status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_LOGS_DIR)
        return "\n".join(f"{key}: {value}" for key, value in status.items())

    def _resolve_cameras(self, setup: dict[str, Any]) -> dict[str, dict]:
        cameras = setup.get("cameras", {})
        result = {}
        for name, cam in cameras.items():
            path = cam.get("by_path") or cam.get("dev", "")
            if not path:
                continue
            result[name] = {"type": "opencv", "index_or_path": path}
        return result

    def _resolve_action_arms(self, setup: dict[str, Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return _resolve_arms(setup, kwargs.get("arms", ""))
        except ValueError as exc:
            raise ActionError(str(exc)) from exc

    async def _run_tty(self, runner: Any, argv: list[str], label: str) -> int:
        await self._tty_handoff(start=True, label=label)
        try:
            return await runner.run_interactive(argv)
        finally:
            await self._tty_handoff(start=False, label=label)

    @staticmethod
    async def _run(runner: Any, argv: list[str]) -> str:
        returncode, stdout, stderr = await runner.run(argv)
        if returncode != 0:
            return f"Command failed (exit {returncode}).\nstdout: {stdout}\nstderr: {stderr}"
        return stdout or "Done."


class ActionError(Exception):
    """User-facing embodied action error."""


def _resolve_arms(setup: dict[str, Any], arms_str: str) -> list[dict[str, Any]]:
    configured = setup.get("arms", [])
    if not configured:
        return []
    ports = _split_arm_tokens(arms_str)
    if not ports:
        return list(configured)
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for port in ports:
        if port in seen:
            raise ValueError(f"Duplicate arm port '{port}' in arms.")
        seen.add(port)
        arm = next((item for item in configured if item.get("port") == port), None)
        if arm is None:
            raise ValueError(f"No arm with port '{port}' found in setup.")
        resolved.append(arm)
    return resolved


def _group_arms(arms: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"followers": [], "leaders": []}
    for arm in arms:
        arm_type = arm.get("type", "")
        if "follower" in arm_type:
            grouped["followers"].append(arm)
            continue
        if "leader" in arm_type:
            grouped["leaders"].append(arm)
    return grouped


def _split_arm_tokens(arms_str: str) -> list[str]:
    if not arms_str:
        return []
    return [token.strip() for token in arms_str.split(",") if token.strip()]


def _validate_pairing(followers: list[dict[str, Any]], leaders: list[dict[str, Any]]) -> str | None:
    if not followers:
        return "No follower arms configured."
    if not leaders:
        return "No leader arms configured."
    if len(followers) != len(leaders):
        return f"Follower/leader count mismatch: {len(followers)} followers, {len(leaders)} leaders."
    if len(followers) not in {1, 2}:
        return f"Unsupported arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
    return None


def _dataset_root(setup: dict[str, Any], fallback: Path | None = None) -> Path:
    root = setup.get("datasets", {}).get("root", "")
    if root:
        return Path(root).expanduser()
    if fallback is not None:
        return fallback.expanduser()
    return Path("~/.roboclaw/workspace/embodied/datasets").expanduser()


def _arm_id(arm: dict[str, Any]) -> str:
    arm_id = Path(arm.get("calibration_dir", "")).name
    if not arm_id:
        raise ValueError(f"Arm '{arm.get('alias', 'unknown')}' has no serial-based calibration_dir.")
    return arm_id


def _is_interrupted(returncode: int) -> bool:
    return returncode in {130, -2}


def _validate_dataset_name(dataset_name: str) -> str | None:
    if not dataset_name or not _DATASET_SLUG_RE.match(dataset_name):
        return "dataset_name must be a non-empty ASCII slug (letters, numbers, underscores, hyphens)."
    return None


@contextmanager
def _bimanual_cal_dirs(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
):
    with TemporaryDirectory(prefix="roboclaw-bimanual-robot-") as robot_dir:
        with TemporaryDirectory(prefix="roboclaw-bimanual-teleop-") as teleop_dir:
            _stage_bimanual_arm_pair(followers[0], followers[1], robot_dir)
            if leaders:
                _stage_bimanual_arm_pair(leaders[0], leaders[1], teleop_dir)
            yield robot_dir, teleop_dir


def _stage_bimanual_arm_pair(left_arm: dict[str, Any], right_arm: dict[str, Any], target_dir: str) -> None:
    target = Path(target_dir)
    for side, arm in [("left", left_arm), ("right", right_arm)]:
        serial = _arm_id(arm)
        source = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
        shutil.copy2(source, target / f"bimanual_{side}.json")
