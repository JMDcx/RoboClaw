"""Standalone MuJoCo simulation ROS2 node.

Provides the same ROS2 services as the real-hardware control surface server,
but backed by MuJoCo physics instead of serial servos.

Usage:
    python -m roboclaw.embodied.simulation.mujoco_ros2_node \
        --model-path path/to/robot.urdf \
        --namespace /roboclaw/so101_setup/sim
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo simulation ROS2 node")
    parser.add_argument("--model-path", required=True, help="Path to URDF/MJCF model")
    parser.add_argument("--namespace", required=True, help="ROS2 namespace")
    parser.add_argument("--joint-mapping", default="{}", help="JSON joint name mapping")
    parser.add_argument("--viewer-port", type=int, default=0, help="HTTP port for the optional simulation viewer")
    parser.add_argument("--viewer-mode", choices=["native", "web", "auto"], default="auto")
    parser.add_argument("--gripper-actuator", default="gripper")
    parser.add_argument("--gripper-open-value", type=float, default=1.0)
    parser.add_argument("--gripper-close-value", type=float, default=0.0)
    parser.add_argument("--state-rate-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    import os

    from roboclaw.embodied.simulation import resolve_viewer_mode

    args = _parse_args()

    # Resolve viewer mode and set MUJOCO_GL *before* any MuJoCo import/model load.
    viewer_mode = resolve_viewer_mode(args.viewer_mode)
    if viewer_mode == "web":
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from roboclaw.embodied.simulation.mujoco_control_runtime import MujocoControlRuntime

    mapping = json.loads(args.joint_mapping) if args.joint_mapping else {}

    runtime = MujocoControlRuntime(
        model_path=args.model_path,
        joint_mapping=mapping or None,
        gripper_actuator=args.gripper_actuator,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
    )
    runtime.connect()

    node = None
    try:
        rclpy.init()
        ns = args.namespace.rstrip("/")
        lock = threading.RLock()
        last_primitive: list[str | None] = [None]
        last_error: list[str | None] = [None]

        node = Node("roboclaw_mujoco_sim")

        string_cls = String
        state_pub = node.create_publisher(String, f"{ns}/state", 10)
        health_pub = node.create_publisher(String, f"{ns}/health", 10)
        events_pub = node.create_publisher(String, f"{ns}/events", 10)

        def publish_status() -> None:
            with lock:
                try:
                    snap = runtime.snapshot()
                except Exception as exc:
                    snap = {"connected": runtime.connected, "snapshot_error": str(exc)}
            payload = {
                "profile_id": "mujoco_sim",
                "robot_id": "sim",
                "connected": snap.get("connected"),
                "gripper_percent": snap.get("gripper_percent"),
                "joint_positions": snap.get("joint_positions", {}),
                "last_primitive": last_primitive[0],
                "last_error": last_error[0],
                "simulator": "mujoco",
            }
            state_pub.publish(string_cls(data=json.dumps(payload, ensure_ascii=False)))
            health_pub.publish(string_cls(data=json.dumps({"ok": last_error[0] is None, "last_error": last_error[0]})))
            events_pub.publish(string_cls(data=json.dumps({"primitive": last_primitive[0], "error": last_error[0]})))

        node.create_timer(max(0.25, 1.0 / max(args.state_rate_hz, 0.1)), publish_status)

        def _success(response: Any, message: str) -> Any:
            last_error[0] = None
            response.success = True
            response.message = message
            return response

        def _failure(response: Any, exc: Exception) -> Any:
            last_error[0] = str(exc)
            response.success = False
            response.message = str(exc)
            return response

        def handle_connect(_req: Any, resp: Any) -> Any:
            with lock:
                try:
                    runtime.connect()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "connected (simulation)")

        def handle_disconnect(_req: Any, resp: Any) -> Any:
            with lock:
                runtime.disconnect()
            return _success(resp, "disconnected")

        def handle_stop(_req: Any, resp: Any) -> Any:
            return _success(resp, "stop acknowledged")

        def handle_reset(_req: Any, resp: Any) -> Any:
            with lock:
                try:
                    last_primitive[0] = "go_named_pose"
                    runtime.go_home()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "reset to home")

        def handle_gripper_open(_req: Any, resp: Any) -> Any:
            with lock:
                try:
                    last_primitive[0] = "gripper_open"
                    runtime.open_gripper()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "gripper opened (sim)")

        def handle_gripper_close(_req: Any, resp: Any) -> Any:
            with lock:
                try:
                    last_primitive[0] = "gripper_close"
                    runtime.close_gripper()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "gripper closed (sim)")

        def handle_go_home(_req: Any, resp: Any) -> Any:
            with lock:
                try:
                    last_primitive[0] = "go_named_pose"
                    runtime.go_home()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "home pose reached (sim)")

        def handle_debug_snapshot(_req: Any, resp: Any) -> Any:
            with lock:
                snap = runtime.snapshot()
            return _success(resp, json.dumps(snap, ensure_ascii=False))

        def handle_recover(_req: Any, resp: Any) -> Any:
            with lock:
                runtime.disconnect()
                try:
                    runtime.connect()
                except Exception as exc:
                    return _failure(resp, exc)
            return _success(resp, "recovered (sim)")

        node.create_service(Trigger, f"{ns}/connect", handle_connect)
        node.create_service(Trigger, f"{ns}/disconnect", handle_disconnect)
        node.create_service(Trigger, f"{ns}/stop", handle_stop)
        node.create_service(Trigger, f"{ns}/reset", handle_reset)
        node.create_service(Trigger, f"{ns}/recover", handle_recover)
        node.create_service(Trigger, f"{ns}/debug_snapshot", handle_debug_snapshot)
        node.create_service(Trigger, f"{ns}/primitive_gripper_open", handle_gripper_open)
        node.create_service(Trigger, f"{ns}/primitive_gripper_close", handle_gripper_close)
        node.create_service(Trigger, f"{ns}/primitive_go_home", handle_go_home)

        from roboclaw.embodied.simulation.session import SimulationSession

        session = SimulationSession(runtime._mujoco, viewer_mode=args.viewer_mode, viewer_port=args.viewer_port)
        session.run(node, lock)
    finally:
        runtime.disconnect()
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
