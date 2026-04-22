#!/usr/bin/env python3
"""Standalone Isaac-Lab reach test for the tabletop red block.

This script bypasses RoboClaw agent / perception, reads the red block's true
world pose directly from Isaac Lab, and then drives the same official-style
right-arm IK path used by ``run_right_arm_ik_keyboard_dex1.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_roots() -> tuple[Path, Path]:
    roboclaw_root = Path(__file__).resolve().parents[1]
    unitree_root = Path(os.environ.get("UNITREE_SIM_ROOT", "/home/xinyuan/unitree_sim_isaaclab")).resolve()
    return roboclaw_root, unitree_root


ROBOCLAW_ROOT, UNITREE_ROOT = _resolve_repo_roots()
os.environ.setdefault("PROJECT_ROOT", str(UNITREE_ROOT))

for path in (str(ROBOCLAW_ROOT), str(UNITREE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Reach the true red block pose with Isaac Lab's official right-arm IK.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PickPlace-RedBlock-G129-Dex1-Joint",
    help="Gym task id to launch.",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--settle-steps", type=int, default=60, help="Initial settle steps after reset.")
parser.add_argument("--seed-steps", type=int, default=90, help="Warm-up steps to move into the seed arm pose.")
parser.add_argument("--phase-hold-steps", type=int, default=45, help="Hold steps after each Cartesian phase.")
parser.add_argument(
    "--ee-body-name",
    type=str,
    default="right_hand_base_link",
    help="Displayed end-effector body name, matching the Dex1 keyboard script.",
)
parser.add_argument("--joint-step-limit", type=float, default=0.05, help="Maximum arm joint delta per IK update.")
parser.add_argument("--gripper-open", type=float, default=-0.02, help="Dex1 open joint target.")
parser.add_argument("--gripper-closed", type=float, default=0.024, help="Dex1 closed joint target.")
parser.add_argument("--pregrasp-height", type=float, default=0.18, help="Clearance above the block before descending.")
parser.add_argument("--clearance-height", type=float, default=0.22, help="Initial raise height at the current XY before moving over the block.")
parser.add_argument("--lift-offset", type=float, default=0.12, help="Vertical lift after closing.")
parser.add_argument("--grasp-top-offset", type=float, default=0.045, help="Height above the block center for contact.")
parser.add_argument("--grasp-offset-x", type=float, default=0.055, help="Grasp center offset +X from right_hand_base_link.")
parser.add_argument("--grasp-offset-y", type=float, default=0.0, help="Grasp center offset +Y from right_hand_base_link.")
parser.add_argument("--grasp-offset-z", type=float, default=0.0, help="Grasp center offset +Z from right_hand_base_link.")
parser.add_argument("--topdown-yaw-deg", type=float, default=0.0, help="Extra yaw rotation for the top-down grasp in degrees.")
parser.add_argument("--print-body-names", action="store_true", help="Print all robot body names.")
AppLauncher.add_app_launcher_args(parser)
ARGS = parser.parse_args()

app_launcher = AppLauncher(ARGS)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch

import tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_mul
from imitation_learning.isaac_action_utils import joint_command_to_policy_action
from tasks.utils.parse_cfg import parse_env_cfg


RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

RIGHT_GRIPPER_JOINTS = (
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
)

RIGHT_ARM_SEED = {
    "right_shoulder_pitch_joint": -0.10,
    "right_shoulder_roll_joint": 0.72,
    "right_shoulder_yaw_joint": 0.10,
    "right_elbow_joint": 1.15,
    "right_wrist_roll_joint": -0.02,
    "right_wrist_pitch_joint": 0.02,
    "right_wrist_yaw_joint": 0.00,
}

SOLVER_REFERENCE_BODY_NAME = "right_wrist_yaw_link"


@dataclass(slots=True)
class ReachDiagnostics:
    block_position_world: list[float]
    clearance_grasp_world: list[float]
    target_grasp_world: list[float]
    pregrasp_grasp_world: list[float]
    lift_grasp_world: list[float]
    hand_base_clearance_world: list[float]
    hand_base_target_world: list[float]
    hand_base_pregrasp_world: list[float]
    hand_base_lift_world: list[float]
    target_quaternion_world_wxyz: list[float]


def _resolve_joint_ids(all_joint_names: list[str], requested_names: tuple[str, ...]) -> list[int]:
    joint_map = {name: idx for idx, name in enumerate(all_joint_names)}
    missing = [name for name in requested_names if name not in joint_map]
    if missing:
        raise RuntimeError(f"Missing joints in robot articulation: {missing}")
    return [joint_map[name] for name in requested_names]


def _load_official_solver_cls():
    module_path = UNITREE_ROOT / "tasks" / "common_controllers" / "official_style_g1_right_arm_ik.py"
    spec = importlib.util.spec_from_file_location("official_style_g1_right_arm_ik", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load solver module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OfficialStyleG1RightArmIK


def _make_sphere_marker(prim_path: str, color: tuple[float, float, float], radius: float) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.2),
            )
        },
    )
    return VisualizationMarkers(cfg)


def _set_named_joint_targets(target: torch.Tensor, joint_to_index: dict[str, int], values: dict[str, float]) -> None:
    for name, value in values.items():
        if name in joint_to_index:
            target[:, joint_to_index[name]] = float(value)


def _set_gripper_targets(target: torch.Tensor, joint_to_index: dict[str, int], close_value: float) -> None:
    for joint_name in RIGHT_GRIPPER_JOINTS:
        if joint_name in joint_to_index:
            target[:, joint_to_index[joint_name]] = float(close_value)


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _world_pose_to_root_pose(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_pos_root = quat_apply(_quat_conjugate(root_quat_w), target_pos_w - root_pos_w)
    root_quat_inv = _quat_conjugate(root_quat_w)
    target_quat_root = _normalize_quat(quat_mul(root_quat_inv, target_quat_w))
    return target_pos_root, target_quat_root


def _quat_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        diag = np.diag(m)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat


def _pose_from_wxyz(position_w: torch.Tensor, quat_wxyz: torch.Tensor) -> np.ndarray:
    w = float(quat_wxyz[0].item())
    x = float(quat_wxyz[1].item())
    y = float(quat_wxyz[2].item())
    z = float(quat_wxyz[3].item())
    norm = max(w * w + x * x + y * y + z * z, 1e-12)
    s = 2.0 / norm
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )
    pose[:3, 3] = position_w.detach().cpu().numpy().astype(np.float64)
    return pose


def _get_grasp_offset_b(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [[ARGS.grasp_offset_x, ARGS.grasp_offset_y, ARGS.grasp_offset_z]],
        device=device,
        dtype=dtype,
    )


def _hand_base_to_grasp_world(base_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
    offset_b = _get_grasp_offset_b(base_pos_w.device, base_pos_w.dtype).expand(base_pos_w.shape[0], -1)
    return base_pos_w + quat_apply(base_quat_w, offset_b)


def _grasp_world_to_hand_base_target(grasp_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
    offset_b = _get_grasp_offset_b(grasp_pos_w.device, grasp_pos_w.dtype).expand(grasp_pos_w.shape[0], -1)
    return grasp_pos_w - quat_apply(base_quat_w, offset_b)


def _rotation_about_world_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(float(degrees))
    c = float(np.cos(radians))
    s = float(np.sin(radians))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _topdown_quat_w(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base_rotation = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotation = _rotation_about_world_z(ARGS.topdown_yaw_deg) @ base_rotation
    quat = _quat_from_rotation_matrix(rotation)
    return torch.tensor([quat], device=device, dtype=dtype)


def _solver_ee_offset_from_current_pose(
    wrist_pos_w: torch.Tensor,
    wrist_quat_w: torch.Tensor,
    hand_base_pos_w: torch.Tensor,
    hand_base_quat_w: torch.Tensor,
) -> tuple[float, float, float]:
    grasp_pos_w = _hand_base_to_grasp_world(hand_base_pos_w, hand_base_quat_w)
    offset_in_wrist = quat_apply(_quat_conjugate(wrist_quat_w), grasp_pos_w - wrist_pos_w)
    return tuple(float(v) for v in offset_in_wrist[0].detach().cpu().tolist())


def _hold_steps(env, joint_target: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    terminated = torch.tensor([False], device=env.device)
    truncated = torch.tensor([False], device=env.device)
    for _ in range(max(0, int(steps))):
        raw_action = joint_command_to_policy_action(env, joint_target)
        _, _, terminated, truncated, _ = env.step(raw_action)
    return terminated, truncated


def _print_diagnostics(diag: ReachDiagnostics) -> None:
    print("=" * 88)
    print("[target] world-frame diagnostics")
    print(f"  block_position_world      : {np.array(diag.block_position_world)}")
    print(f"  clearance_grasp_world     : {np.array(diag.clearance_grasp_world)}")
    print(f"  target_grasp_world        : {np.array(diag.target_grasp_world)}")
    print(f"  pregrasp_grasp_world      : {np.array(diag.pregrasp_grasp_world)}")
    print(f"  lift_grasp_world          : {np.array(diag.lift_grasp_world)}")
    print(f"  hand_base_clearance_world : {np.array(diag.hand_base_clearance_world)}")
    print(f"  hand_base_target_world    : {np.array(diag.hand_base_target_world)}")
    print(f"  hand_base_pregrasp_world  : {np.array(diag.hand_base_pregrasp_world)}")
    print(f"  hand_base_lift_world      : {np.array(diag.hand_base_lift_world)}")
    print(f"  target_quat_world_wxyz    : {np.array(diag.target_quaternion_world_wxyz)}")
    print("=" * 88)


def _build_phase_targets(
    block_pos_w: torch.Tensor,
    current_grasp_world: torch.Tensor,
) -> tuple[ReachDiagnostics, dict[str, torch.Tensor], torch.Tensor]:
    target_quat_w = _topdown_quat_w(block_pos_w.device, block_pos_w.dtype)
    target_grasp_world = block_pos_w.clone()
    target_grasp_world[:, 2] += float(ARGS.grasp_top_offset)

    clearance_grasp_world = current_grasp_world.clone()
    clearance_grasp_world[:, 2] = torch.maximum(
        clearance_grasp_world[:, 2],
        target_grasp_world[:, 2] + float(ARGS.clearance_height),
    )

    pregrasp_grasp_world = target_grasp_world.clone()
    pregrasp_grasp_world[:, 2] += float(ARGS.pregrasp_height)

    lift_grasp_world = target_grasp_world.clone()
    lift_grasp_world[:, 2] += float(ARGS.lift_offset)

    hand_base_clearance_world = _grasp_world_to_hand_base_target(clearance_grasp_world, target_quat_w)
    hand_base_target_world = _grasp_world_to_hand_base_target(target_grasp_world, target_quat_w)
    hand_base_pregrasp_world = _grasp_world_to_hand_base_target(pregrasp_grasp_world, target_quat_w)
    hand_base_lift_world = _grasp_world_to_hand_base_target(lift_grasp_world, target_quat_w)

    diag = ReachDiagnostics(
        block_position_world=block_pos_w[0].detach().cpu().tolist(),
        clearance_grasp_world=clearance_grasp_world[0].detach().cpu().tolist(),
        target_grasp_world=target_grasp_world[0].detach().cpu().tolist(),
        pregrasp_grasp_world=pregrasp_grasp_world[0].detach().cpu().tolist(),
        lift_grasp_world=lift_grasp_world[0].detach().cpu().tolist(),
        hand_base_clearance_world=hand_base_clearance_world[0].detach().cpu().tolist(),
        hand_base_target_world=hand_base_target_world[0].detach().cpu().tolist(),
        hand_base_pregrasp_world=hand_base_pregrasp_world[0].detach().cpu().tolist(),
        hand_base_lift_world=hand_base_lift_world[0].detach().cpu().tolist(),
        target_quaternion_world_wxyz=target_quat_w[0].detach().cpu().tolist(),
    )
    phases = {
        "clearance": clearance_grasp_world,
        "pregrasp": pregrasp_grasp_world,
        "grasp": target_grasp_world,
        "lift": lift_grasp_world,
    }
    return diag, phases, target_quat_w


def _solve_phase(
    official_ik,
    robot,
    right_arm_ids: list[int],
    target_grasp_world: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> np.ndarray:
    root_pose_w = robot.data.root_state_w[:, 0:7]
    joint_pos = robot.data.joint_pos[:, right_arm_ids]
    hand_base_target_pos_w = _grasp_world_to_hand_base_target(target_grasp_world, target_quat_w)
    target_pos_root, target_quat_root = _world_pose_to_root_pose(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        hand_base_target_pos_w,
        target_quat_w,
    )
    target_pose_np = _pose_from_wxyz(target_pos_root[0], target_quat_root[0])
    current_q_np = joint_pos[0].detach().cpu().numpy().astype(np.float64)
    current_dq_np = robot.data.joint_vel[0, right_arm_ids].detach().cpu().numpy().astype(np.float64)
    arm_pos_des_np, _ = official_ik.solve(target_pose_np, current_q=current_q_np, current_dq=current_dq_np)
    if not np.all(np.isfinite(arm_pos_des_np)):
        raise RuntimeError("Official solver returned non-finite joints.")
    return arm_pos_des_np


def main() -> None:
    env_cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.seed = 42
    env = gym.make(ARGS.task, cfg=env_cfg).unwrapped
    env.seed(42)
    env.sim.reset()
    env.reset()

    scene = env.scene
    robot = scene["robot"]
    obj_asset = scene["object"]
    joint_names = list(robot.data.joint_names)
    body_names = list(getattr(robot.data, "body_names", []))
    if ARGS.print_body_names:
        print("Robot body names:")
        for idx, name in enumerate(body_names):
            print(f"  [{idx:03d}] {name}")

    if ARGS.ee_body_name not in body_names:
        raise RuntimeError(f"Body '{ARGS.ee_body_name}' not found in robot body names.")
    if SOLVER_REFERENCE_BODY_NAME not in body_names:
        raise RuntimeError(f"Solver reference body '{SOLVER_REFERENCE_BODY_NAME}' not found in robot body names.")

    joint_to_index = {name: idx for idx, name in enumerate(joint_names)}
    right_arm_ids = _resolve_joint_ids(joint_names, RIGHT_ARM_JOINTS)
    ee_body_id = body_names.index(ARGS.ee_body_name)
    solver_ref_body_id = body_names.index(SOLVER_REFERENCE_BODY_NAME)
    pregrasp_marker = _make_sphere_marker("/World/Visuals/redblock_pregrasp_target", (0.2, 0.45, 1.0), 0.018)
    grasp_marker = _make_sphere_marker("/World/Visuals/redblock_grasp_target", (1.0, 0.25, 0.25), 0.018)
    lift_marker = _make_sphere_marker("/World/Visuals/redblock_lift_target", (0.75, 0.35, 1.0), 0.018)
    active_target_marker = _make_sphere_marker("/World/Visuals/redblock_active_target", (1.0, 1.0, 1.0), 0.014)
    live_grasp_marker = _make_sphere_marker("/World/Visuals/redblock_live_grasp", (0.2, 1.0, 0.25), 0.016)

    default_joint_target = (
        robot.data.default_joint_pos.clone() if hasattr(robot.data, "default_joint_pos") else robot.data.joint_pos.clone()
    )
    full_joint_target = default_joint_target.clone()
    _set_named_joint_targets(full_joint_target, joint_to_index, RIGHT_ARM_SEED)
    _set_gripper_targets(full_joint_target, joint_to_index, float(ARGS.gripper_open))

    _hold_steps(env, default_joint_target, ARGS.settle_steps)
    _hold_steps(env, full_joint_target, ARGS.seed_steps)

    ee_pose_w = robot.data.body_state_w[:, ee_body_id, 0:7].clone()
    solver_ref_pose_w = robot.data.body_state_w[:, solver_ref_body_id, 0:7].clone()
    current_grasp_world = _hand_base_to_grasp_world(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    if hasattr(obj_asset.data, "root_state_w"):
        block_pos_w = obj_asset.data.root_state_w[:, 0:3].clone()
    else:
        block_pos_w = obj_asset.data.root_pos_w.clone()
    print(f"[block] world position : {block_pos_w[0].detach().cpu().tolist()}")

    diag, phases, target_quat_w = _build_phase_targets(block_pos_w, current_grasp_world)
    _print_diagnostics(diag)
    clearance_marker = _make_sphere_marker("/World/Visuals/redblock_clearance_target", (1.0, 0.8, 0.2), 0.018)
    pregrasp_marker.visualize(translations=phases["pregrasp"])
    grasp_marker.visualize(translations=phases["grasp"])
    lift_marker.visualize(translations=phases["lift"])
    clearance_marker.visualize(translations=phases["clearance"])
    active_target_marker.visualize(translations=phases["clearance"])
    live_grasp_marker.visualize(
        translations=_hand_base_to_grasp_world(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    )

    solver_ee_offset = _solver_ee_offset_from_current_pose(
        solver_ref_pose_w[:, 0:3],
        solver_ref_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )
    solver_cls = _load_official_solver_cls()
    official_ik = solver_cls(ee_offset=solver_ee_offset)
    print(f"[IK] official solver ee_offset: {solver_ee_offset}")

    for phase_name in ("clearance", "pregrasp", "grasp"):
        print(f"[IK] solving phase: {phase_name}")
        clearance_marker.visualize(translations=phases["clearance"])
        pregrasp_marker.visualize(translations=phases["pregrasp"])
        grasp_marker.visualize(translations=phases["grasp"])
        lift_marker.visualize(translations=phases["lift"])
        active_target_marker.visualize(translations=phases[phase_name])
        live_grasp_marker.visualize(
            translations=_hand_base_to_grasp_world(
                robot.data.body_state_w[:, ee_body_id, 0:3],
                robot.data.body_state_w[:, ee_body_id, 3:7],
            )
        )
        arm_pos_des_np = _solve_phase(
            official_ik,
            robot,
            right_arm_ids,
            phases[phase_name],
            target_quat_w,
        )
        joint_pos = robot.data.joint_pos[:, right_arm_ids]
        arm_pos_des = torch.tensor(arm_pos_des_np, device=env.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
        arm_delta = torch.clamp(arm_pos_des - joint_pos, min=-ARGS.joint_step_limit, max=ARGS.joint_step_limit)
        full_joint_target[:, right_arm_ids] = joint_pos + arm_delta
        terminated, truncated = _hold_steps(env, full_joint_target, ARGS.phase_hold_steps)
        done = bool(terminated.item() if torch.is_tensor(terminated) else terminated) or bool(
            truncated.item() if torch.is_tensor(truncated) else truncated
        )
        if done:
            print(f"[env] environment terminated during phase '{phase_name}'")
            simulation_app.close()
            return

    _set_gripper_targets(full_joint_target, joint_to_index, float(ARGS.gripper_closed))
    _hold_steps(env, full_joint_target, ARGS.phase_hold_steps)

    print("[IK] solving phase: lift")
    clearance_marker.visualize(translations=phases["clearance"])
    pregrasp_marker.visualize(translations=phases["pregrasp"])
    grasp_marker.visualize(translations=phases["grasp"])
    lift_marker.visualize(translations=phases["lift"])
    active_target_marker.visualize(translations=phases["lift"])
    live_grasp_marker.visualize(
        translations=_hand_base_to_grasp_world(
            robot.data.body_state_w[:, ee_body_id, 0:3],
            robot.data.body_state_w[:, ee_body_id, 3:7],
        )
    )
    arm_pos_des_np = _solve_phase(
        official_ik,
        robot,
        right_arm_ids,
        phases["lift"],
        target_quat_w,
    )
    joint_pos = robot.data.joint_pos[:, right_arm_ids]
    arm_pos_des = torch.tensor(arm_pos_des_np, device=env.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
    arm_delta = torch.clamp(arm_pos_des - joint_pos, min=-ARGS.joint_step_limit, max=ARGS.joint_step_limit)
    full_joint_target[:, right_arm_ids] = joint_pos + arm_delta
    _hold_steps(env, full_joint_target, ARGS.phase_hold_steps)

    print("[done] executed above-block pregrasp -> vertical descend -> close -> vertical lift.")
    print("[done] yellow = clearance, blue = pregrasp, red = grasp, purple = lift, white = active target, green = live grasp center.")
    print("[done] leave the simulator open to inspect the final posture; Ctrl+C to exit.")
    try:
        while simulation_app.is_running():
            clearance_marker.visualize(translations=phases["clearance"])
            pregrasp_marker.visualize(translations=phases["pregrasp"])
            grasp_marker.visualize(translations=phases["grasp"])
            lift_marker.visualize(translations=phases["lift"])
            active_target_marker.visualize(translations=phases["lift"])
            live_grasp_marker.visualize(
                translations=_hand_base_to_grasp_world(
                    robot.data.body_state_w[:, ee_body_id, 0:3],
                    robot.data.body_state_w[:, ee_body_id, 3:7],
                )
            )
            _hold_steps(env, full_joint_target, 1)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
