"""Embodied tool groups - bridges agent to the embodied robotics layer."""

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
from roboclaw.embodied.hand_actions import (
    _do_hand_close,
    _do_hand_open,
    _do_hand_pose,
    _do_hand_status,
)
from roboclaw.embodied.setup import get_roboclaw_home


def _logs_dir() -> Path:
    """Return the embodied jobs log directory under ROBOCLAW_HOME."""
    return get_roboclaw_home() / "workspace" / "embodied" / "jobs"

_NO_TTY_MSG = "This action requires a local terminal. Run: roboclaw agent"
_BIMANUAL_ID = "bimanual"
_DEFAULT_REPLAY_ROOT = Path("~/.cache/huggingface/lerobot").expanduser()
_DATASET_SLUG_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

_ACTION_DESCRIPTIONS = {
    "doctor": "Check LeRobot availability and show the current embodied setup.",
    "identify": "Launch the interactive arm-identification flow for detected serial ports.",
    "describe": "Explain adjustable parameters for a target embodied action.",
    "calibrate": "Calibrate one or more configured arms. If arms is omitted, calibrate every uncalibrated arm.",
    "teleoperate": "Run live teleoperation. Select arms with a comma-separated port list.",
    "record": "Record a dataset with one follower/leader pair or two pairs for bimanual capture.",
    "replay": "Replay a recorded dataset episode on one or two follower arms.",
    "train": "Start ACT training for a recorded dataset as a detached job.",
    "run_policy": "Run a trained policy: use embodied_control(action='record', checkpoint_path='...') with one follower arm.",
    "job_status": "Inspect the status and recent logs for a detached training job.",
    "setup_show": "Show the embodied setup JSON with configured arms, cameras, and roots.",
    "set_arm": "Create or update one configured arm alias.",
    "rename_arm": "Rename an existing configured arm alias.",
    "remove_arm": "Remove one configured arm alias.",
    "set_camera": "Assign a scanned camera to a stable camera name.",
    "preview_cameras": "Capture one preview image for each scanned camera.",
    "remove_camera": "Remove a configured camera.",
    "set_hand": "Create or update one configured hand alias.",
    "remove_hand": "Remove a configured hand alias.",
    "hand_open": "Open all fingers of a dexterous hand.",
    "hand_close": "Close all fingers of a dexterous hand.",
    "hand_pose": "Set individual finger positions on a dexterous hand.",
    "hand_status": "Read current finger angles and forces from a dexterous hand.",
}

# ---------------------------------------------------------------------------
# Tool group specifications
# ---------------------------------------------------------------------------

_SETUP_ACTIONS = [
    "setup_show", "set_arm", "remove_arm", "rename_arm",
    "set_camera", "preview_cameras", "remove_camera", "describe", "doctor",
    "set_hand", "remove_hand",
]

_TOOL_GROUPS: dict[str, dict[str, Any]] = {
    "embodied_setup": {
        "description": (
            "Configure robot hardware: show setup, add/remove/rename arms, "
            "add/remove cameras, describe actions, check environment. "
            "Also manages dexterous hands: set_hand, remove_hand."
        ),
        "actions": _SETUP_ACTIONS,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _SETUP_ACTIONS,
                    "description": "The action to perform.",
                },
                "alias": {
                    "type": "string",
                    "description": "Arm alias for set_arm, rename_arm, remove_arm, or hand alias for set_hand/remove_hand.",
                },
                "arm_type": {
                    "type": "string",
                    "enum": ["so101_follower", "so101_leader"],
                    "description": "Arm hardware type for set_arm.",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path for set_arm or set_hand.",
                },
                "new_alias": {
                    "type": "string",
                    "description": "New arm alias for rename_arm.",
                },
                "camera_name": {
                    "type": "string",
                    "description": "Camera name like front or side.",
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Index into the live-detected camera list for set_camera.",
                },
                "target_action": {
                    "type": "string",
                    "description": "Action name to describe.",
                },
                "hand_type": {
                    "type": "string",
                    "enum": ["inspire_rh56", "revo2"],
                    "description": "Hand hardware type for set_hand.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_hardware": {
        "description": "Hardware identification and calibration for configured arms.",
        "actions": ["identify", "calibrate"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["identify", "calibrate"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_control": {
        "description": (
            "Teleoperate or record datasets. "
            "With checkpoint_path, record runs policy inference instead of teleop recording."
        ),
        "actions": ["teleoperate", "record"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["teleoperate", "record"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for record.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for recording.",
                },
                "use_cameras": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether recording or policy should include configured cameras.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record or run.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for recording.",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Path to a trained policy checkpoint (turns record into policy inference).",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_replay": {
        "description": "Replay a recorded dataset episode on follower arms.",
        "actions": ["replay"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["replay"],
                    "description": "The action to perform.",
                },
                "arms": {
                    "type": "string",
                    "description": "Comma-separated arm port paths (by-id from setup_show).",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for replay.",
                },
                "episode": {
                    "type": "integer",
                    "description": "Episode index to replay.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for replay.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_train": {
        "description": "Train a policy on a recorded dataset or check training job status.",
        "actions": ["train", "job_status"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["train", "job_status"],
                    "description": "The action to perform.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset slug for training.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of training steps.",
                },
                "device": {
                    "type": "string",
                    "description": "Device for training (default: cuda).",
                },
                "job_id": {
                    "type": "string",
                    "description": "ID of a background training job.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
    "embodied_hand": {
        "description": "Control a dexterous hand: open, close, set finger pose, read status.",
        "actions": ["hand_open", "hand_close", "hand_pose", "hand_status"],
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["hand_open", "hand_close", "hand_pose", "hand_status"],
                    "description": "The hand action to perform.",
                },
                "hand_name": {
                    "type": "string",
                    "description": "Hand alias. Uses first configured hand if omitted.",
                },
                "positions": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "6 finger positions 0-1000 [little, ring, middle, index, thumb_bend, thumb_rotation] for hand_pose.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# EmbodiedToolGroup
# ---------------------------------------------------------------------------

class EmbodiedToolGroup(Tool):
    """A single tool group that dispatches to shared action functions."""

    def __init__(self, group_name: str, spec: dict[str, Any], tty_handoff: Any = None):
        self._group_name = group_name
        self._spec = spec
        self._tty_handoff = tty_handoff

    @property
    def name(self) -> str:
        return self._group_name

    @property
    def description(self) -> str:
        return self._spec["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self._spec["parameters"]

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        if action not in self._spec["actions"]:
            return f"Unknown action '{action}' for tool {self._group_name}."
        return await _dispatch(action, kwargs, self._tty_handoff)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_embodied_tools(tty_handoff: Any = None) -> list[EmbodiedToolGroup]:
    """Return a list of EmbodiedToolGroup instances for all 6 groups."""
    return [
        EmbodiedToolGroup(name, spec, tty_handoff=tty_handoff)
        for name, spec in _TOOL_GROUPS.items()
    ]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Actions that don't need ensure_setup (sync functions)
_NO_SETUP_ACTIONS = frozenset({
    "setup_show", "describe", "set_arm", "rename_arm",
    "remove_arm", "set_camera", "preview_cameras", "remove_camera",
    "set_hand", "remove_hand",
})


async def _dispatch(action: str, kwargs: dict[str, Any], tty_handoff: Any) -> str:
    if action in _NO_SETUP_ACTIONS:
        return _SYNC_DISPATCH[action](kwargs)

    from roboclaw.embodied.setup import ensure_setup
    setup = ensure_setup()
    try:
        return await _ASYNC_DISPATCH[action](setup, kwargs, tty_handoff)
    except ActionError as exc:
        return str(exc)


# ---------------------------------------------------------------------------
# Action implementations (sync, no setup required)
# ---------------------------------------------------------------------------

def _do_setup_show(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.scan import scan_cameras, scan_serial_ports
    from roboclaw.embodied.setup import load_setup

    setup = load_setup()
    setup["scanned_ports"] = scan_serial_ports()
    setup["scanned_cameras"] = scan_cameras()
    return json.dumps(setup, indent=2, ensure_ascii=False)


def _do_describe(kwargs: dict[str, Any]) -> str:
    target_action = kwargs.get("target_action", "")
    if not target_action:
        return json.dumps(_ACTION_DESCRIPTIONS, indent=2, ensure_ascii=False)
    if target_action not in _ACTION_DESCRIPTIONS:
        return f"Unknown target_action: {target_action}"
    return f"{target_action}: {_ACTION_DESCRIPTIONS[target_action]}"


def _do_set_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import arm_display_name, find_arm, set_arm

    alias = kwargs.get("alias", "")
    arm_type = kwargs.get("arm_type", "")
    port = kwargs.get("port", "")
    if not all([alias, arm_type, port]):
        return "set_arm requires alias, arm_type, and port."
    updated = set_arm(alias, arm_type, port)
    arm = find_arm(updated["arms"], alias)
    display = arm_display_name(arm)
    return f"Arm '{display}' configured.\n{json.dumps(arm, indent=2)}"


def _do_rename_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_arm, rename_arm

    old_alias = kwargs.get("alias", "")
    new_alias = kwargs.get("new_alias", "")
    if not old_alias or not new_alias:
        return "rename_arm requires alias and new_alias."
    updated = rename_arm(old_alias, new_alias)
    arm = find_arm(updated["arms"], new_alias)
    return f"Arm renamed from '{old_alias}' to '{new_alias}'.\n{json.dumps(arm, indent=2)}"


def _do_remove_arm(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_arm

    alias = kwargs.get("alias", "")
    if not alias:
        return "remove_arm requires alias."
    remove_arm(alias)
    return f"Arm '{alias}' removed."


def _do_set_camera(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_camera, set_camera

    name = kwargs.get("camera_name", "")
    index = kwargs.get("camera_index")
    if not name or index is None:
        return "set_camera requires camera_name and camera_index."
    updated = set_camera(name, index)
    cam = find_camera(updated["cameras"], name)
    return f"Camera '{name}' configured.\n{json.dumps(cam, indent=2)}"


def _do_preview_cameras(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.scan import capture_camera_frames, scan_cameras

    scanned_cameras = scan_cameras()
    if not scanned_cameras:
        return "No cameras detected."
    try:
        previews = capture_camera_frames(scanned_cameras, _camera_previews_dir())
    except RuntimeError as exc:
        return f"Camera preview failed: {exc}"
    if not previews:
        return "No camera previews captured."
    return json.dumps(previews, indent=2, ensure_ascii=False)


def _do_remove_camera(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_camera

    name = kwargs.get("camera_name", "")
    if not name:
        return "remove_camera requires camera_name."
    remove_camera(name)
    return f"Camera '{name}' removed."


def _do_set_hand(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import find_hand, set_hand

    alias = kwargs.get("alias", "")
    hand_type = kwargs.get("hand_type", "")
    port = kwargs.get("port", "")
    if not all([alias, hand_type, port]):
        return "set_hand requires alias, hand_type, and port."
    updated = set_hand(alias, hand_type, port)
    hand = find_hand(updated["hands"], alias)
    return f"Hand '{alias}' configured.\n{json.dumps(hand, indent=2)}"


def _do_remove_hand(kwargs: dict[str, Any]) -> str:
    from roboclaw.embodied.setup import remove_hand

    alias = kwargs.get("alias", "")
    if not alias:
        return "remove_hand requires alias."
    remove_hand(alias)
    return f"Hand '{alias}' removed."


_SYNC_DISPATCH: dict[str, Any] = {
    "setup_show": _do_setup_show,
    "describe": _do_describe,
    "set_arm": _do_set_arm,
    "rename_arm": _do_rename_arm,
    "remove_arm": _do_remove_arm,
    "set_camera": _do_set_camera,
    "preview_cameras": _do_preview_cameras,
    "remove_camera": _do_remove_camera,
    "set_hand": _do_set_hand,
    "remove_hand": _do_remove_hand,
}


# ---------------------------------------------------------------------------
# Action implementations (async, require setup)
# ---------------------------------------------------------------------------

async def _do_doctor(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.runner import LocalLeRobotRunner

    result = await _run(LocalLeRobotRunner(), SO101Controller().doctor())
    return result + f"\n\nCurrent setup:\n{json.dumps(setup, indent=2, ensure_ascii=False)}"


async def _do_identify(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.scan import scan_serial_ports

    if not tty_handoff:
        return _NO_TTY_MSG
    ports = scan_serial_ports()
    if not ports:
        return "No serial ports detected."
    argv = [sys.executable, "-m", "roboclaw.embodied.identify", json.dumps(ports)]
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "identify-arms")
    if rc == 0:
        return "Arm identification complete."
    return _format_tty_failure("Arm identification failed", rc, stderr_text)


async def _do_calibrate(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.setup import arm_display_name, mark_arm_calibrated

    configured = setup.get("arms", [])
    if not configured:
        return "No arms configured."
    selected = _resolve_action_arms(setup, kwargs)
    targets = selected if kwargs.get("arms", "") else [a for a in selected if not a.get("calibrated")]
    if not targets:
        return "All arms are already calibrated."
    if not tty_handoff:
        return _NO_TTY_MSG
    controller = SO101Controller()
    runner = LocalLeRobotRunner()
    succeeded = 0
    failed = 0
    results: list[str] = []
    for arm in targets:
        display = arm_display_name(arm)
        argv = controller.calibrate(
            arm["type"], arm["port"], arm.get("calibration_dir", ""), _arm_id(arm),
        )
        rc, stderr_text = await _run_tty(tty_handoff, runner, argv, f"Calibrating: {display}")
        if _is_interrupted(rc):
            return "interrupted"
        if rc == 0:
            succeeded += 1
            mark_arm_calibrated(arm["alias"])
            results.append(f"{display}: OK")
            continue
        failed += 1
        results.append(_format_tty_failure(f"{display}: FAILED", rc, stderr_text))
    return (
        f"{succeeded} succeeded, {failed} failed.\n"
        + "\n".join(results)
        + "\nNote: wrist_roll is auto-calibrated by LeRobot (expected)."
    )


async def _do_teleoperate(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller

    if not tty_handoff:
        return _NO_TTY_MSG
    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    error = _validate_pairing(grouped["followers"], grouped["leaders"])
    if error:
        return error
    controller = SO101Controller()
    followers = grouped["followers"]
    leaders = grouped["leaders"]
    cameras = _resolve_cameras(setup)
    if len(followers) == 1:
        return await _teleoperate_single(controller, followers[0], leaders[0], cameras, tty_handoff)
    return await _teleoperate_bimanual(controller, followers, leaders, cameras, tty_handoff)


async def _teleoperate_single(
    controller: Any, follower: dict, leader: dict, cameras: dict, tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.setup import arm_display_name

    argv = controller.teleoperate(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        teleop_type=leader["type"],
        teleop_port=leader["port"],
        teleop_cal_dir=leader["calibration_dir"],
        teleop_id=_arm_id(leader),
        cameras=cameras,
    )
    label = f"lerobot-teleoperate ({arm_display_name(follower)} + {arm_display_name(leader)})"
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, label)
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Teleoperation finished."
    return _format_tty_failure("Teleoperation failed", rc, stderr_text)


async def _teleoperate_bimanual(
    controller: Any,
    followers: list[dict],
    leaders: list[dict],
    cameras: dict,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

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
            cameras=cameras,
        )
        rc, stderr_text = await _run_tty(
            tty_handoff, LocalLeRobotRunner(), argv, "lerobot-teleoperate (bimanual)",
        )
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Teleoperation finished."
    return _format_tty_failure("Teleoperation failed", rc, stderr_text)


async def _do_record(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    if kwargs.get("checkpoint_path"):
        return await _do_run_policy(setup, kwargs, tty_handoff)

    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.runner import LocalLeRobotRunner

    if not tty_handoff:
        return _NO_TTY_MSG
    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    error = _validate_pairing(grouped["followers"], grouped["leaders"])
    if error:
        return error
    dataset_name = kwargs.get("dataset_name", "default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    controller = SO101Controller()
    cameras = {} if kwargs.get("use_cameras") is False else _resolve_cameras(setup)
    record_kwargs = _build_record_kwargs(setup, kwargs, cameras, dataset_name)
    followers = grouped["followers"]
    leaders = grouped["leaders"]
    if len(followers) == 1:
        return await _record_single(controller, followers[0], leaders[0], record_kwargs, tty_handoff)
    return await _record_bimanual(controller, followers, leaders, record_kwargs, tty_handoff)


def _build_record_kwargs(
    setup: dict[str, Any], kwargs: dict[str, Any], cameras: dict, dataset_name: str,
) -> dict[str, Any]:
    return {
        "cameras": cameras,
        "repo_id": f"local/{dataset_name}",
        "task": kwargs.get("task", "default_task"),
        "dataset_root": str(_dataset_path(setup, dataset_name)),
        "push_to_hub": False,
        "fps": kwargs.get("fps", 30),
        "num_episodes": kwargs.get("num_episodes", 10),
    }


async def _record_single(
    controller: Any, follower: dict, leader: dict, record_kwargs: dict, tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

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
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-record")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Recording finished."
    return _format_tty_failure("Recording failed", rc, stderr_text)


async def _record_bimanual(
    controller: Any,
    followers: list[dict],
    leaders: list[dict],
    record_kwargs: dict,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

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
        rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-record")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Recording finished."
    return _format_tty_failure("Recording failed", rc, stderr_text)


async def _do_run_policy(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    """Run a trained policy - called from record when checkpoint_path is set."""
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    followers = grouped["followers"]
    if not followers:
        return "No follower arm configured."
    if len(followers) != 1:
        return "run_policy requires exactly 1 follower arm. Provide arms with a single follower port."
    follower = followers[0]
    cameras = {} if kwargs.get("use_cameras") is False else _resolve_cameras(setup)
    policies_root = setup.get("policies", {}).get("root", "")
    checkpoint = kwargs.get("checkpoint_path") or ACTPipeline().checkpoint_path(policies_root)
    dataset_name = kwargs.get("dataset_name", "eval_default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    if not dataset_name.startswith("eval_"):
        dataset_name = f"eval_{dataset_name}"
    dataset_root = _dataset_path(setup, dataset_name)
    argv = SO101Controller().run_policy(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        cameras=cameras,
        policy_path=checkpoint,
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        task=kwargs.get("task", "eval"),
        num_episodes=kwargs.get("num_episodes", 1),
    )
    return await _run(LocalLeRobotRunner(), argv)


async def _do_replay(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.so101 import SO101Controller
    from roboclaw.embodied.runner import LocalLeRobotRunner

    if not tty_handoff:
        return _NO_TTY_MSG
    selected = _resolve_action_arms(setup, kwargs)
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
    dataset_root = _dataset_path(setup, dataset_name, fallback=_DEFAULT_REPLAY_ROOT)
    episode = kwargs.get("episode", 0)
    fps = kwargs.get("fps", 30)
    controller = SO101Controller()
    if len(followers) == 1:
        return await _replay_single(controller, followers[0], dataset_name, dataset_root, episode, fps, tty_handoff)
    return await _replay_bimanual(controller, followers, dataset_name, dataset_root, episode, fps, tty_handoff)


async def _replay_single(
    controller: Any, follower: dict,
    dataset_name: str, dataset_root: Path, episode: int, fps: int,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    argv = controller.replay(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        episode=episode,
        fps=fps,
    )
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-replay")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Replay finished."
    return _format_tty_failure("Replay failed", rc, stderr_text)


async def _replay_bimanual(
    controller: Any, followers: list[dict],
    dataset_name: str, dataset_root: Path, episode: int, fps: int,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    with _bimanual_cal_dirs(followers, []) as (robot_dir, _):
        argv = controller.replay_bimanual(
            robot_id=_BIMANUAL_ID,
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            repo_id=f"local/{dataset_name}",
            dataset_root=str(dataset_root),
            episode=episode,
            fps=fps,
        )
        rc, stderr_text = await _run_tty(
            tty_handoff, LocalLeRobotRunner(), argv, "lerobot-replay (bimanual)",
        )
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Replay finished."
    return _format_tty_failure("Replay failed", rc, stderr_text)


async def _do_train(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    dataset_name = kwargs.get("dataset_name", "default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    dataset_root = _dataset_path(setup, dataset_name)
    policies_root = setup.get("policies", {}).get("root", "")
    argv = ACTPipeline().train(
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        output_dir=policies_root,
        steps=kwargs.get("steps", 100_000),
        device=kwargs.get("device", "cuda"),
    )
    job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_logs_dir())
    return f"Training started. Job ID: {job_id}"


async def _do_job_status(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    job_id = kwargs.get("job_id", "")
    status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_logs_dir())
    return "\n".join(f"{key}: {value}" for key, value in status.items())


_ASYNC_DISPATCH: dict[str, Any] = {
    "doctor": _do_doctor,
    "identify": _do_identify,
    "calibrate": _do_calibrate,
    "teleoperate": _do_teleoperate,
    "record": _do_record,
    "replay": _do_replay,
    "train": _do_train,
    "job_status": _do_job_status,
    "hand_open": _do_hand_open,
    "hand_close": _do_hand_close,
    "hand_pose": _do_hand_pose,
    "hand_status": _do_hand_status,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class ActionError(Exception):
    """User-facing embodied action error."""


def _resolve_cameras(setup: dict[str, Any]) -> dict[str, dict]:
    cameras = setup.get("cameras", [])
    result = {}
    for cam in cameras:
        alias = cam.get("alias", "")
        port = cam.get("port", "")
        if not alias or not port:
            continue
        config = {
            "type": "opencv",
            "index_or_path": port,
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps") or 30,
        }
        fourcc = cam.get("fourcc")
        if fourcc is not None:
            config["fourcc"] = fourcc
        result[alias] = config
    return result


def _resolve_action_arms(setup: dict[str, Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return _resolve_arms(setup, kwargs.get("arms", ""))
    except ValueError as exc:
        raise ActionError(str(exc)) from exc


async def _run_tty(
    tty_handoff: Any, runner: Any, argv: list[str], label: str,
) -> tuple[int, str]:
    await tty_handoff(start=True, label=label)
    try:
        return await runner.run_interactive(argv)
    finally:
        await tty_handoff(start=False, label=label)


async def _run(runner: Any, argv: list[str]) -> str:
    returncode, stdout, stderr = await runner.run(argv)
    if returncode != 0:
        return f"Command failed (exit {returncode}).\nstdout: {stdout}\nstderr: {stderr}"
    return stdout or "Done."


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
    grouped: dict[str, list[dict[str, Any]]] = {"followers": [], "leaders": []}
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
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"


def _dataset_path(
    setup: dict[str, Any], dataset_name: str, fallback: Path | None = None,
) -> Path:
    return _dataset_root(setup, fallback) / "local" / dataset_name


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


def _camera_previews_dir() -> Path:
    return get_roboclaw_home() / "workspace" / "embodied" / "camera_previews"


def _format_tty_failure(prefix: str, returncode: int, stderr_text: str) -> str:
    message = f"{prefix} (exit {returncode})."
    stderr_text = stderr_text.strip()
    if not stderr_text:
        return message
    return f"{message}\nstderr: {stderr_text}"


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


def _stage_bimanual_arm_pair(
    left_arm: dict[str, Any], right_arm: dict[str, Any], target_dir: str,
) -> None:
    target = Path(target_dir)
    for side, arm in [("left", left_arm), ("right", right_arm)]:
        serial = _arm_id(arm)
        source = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
        shutil.copy2(source, target / f"bimanual_{side}.json")
