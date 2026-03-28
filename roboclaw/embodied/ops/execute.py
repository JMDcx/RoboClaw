"""Asynchronous embodied action execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from roboclaw.embodied.hand_actions import (
    _do_hand_close,
    _do_hand_open,
    _do_hand_pose,
    _do_hand_status,
)
from roboclaw.embodied.ops.helpers import (
    _BIMANUAL_ID,
    _DEFAULT_REPLAY_ROOT,
    _NO_TTY_MSG,
    _arm_id,
    _bimanual_cal_dirs,
    _dataset_path,
    _format_tty_failure,
    _group_arms,
    _is_interrupted,
    _logs_dir,
    _resolve_action_arms,
    _run,
    _run_tty,
    _validate_dataset_name,
    _validate_pairing,
)
from roboclaw.embodied.sensor.camera import resolve_cameras


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
    cameras = resolve_cameras(setup)
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
    cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(setup)
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
    cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(setup)
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
        output_dir=str(Path(policies_root).expanduser() / dataset_name),
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


ASYNC_DISPATCH: dict[str, Any] = {
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
