"""Minimal Unitree G1 Isaac Lab controller using SDK2 DDS lowcmd."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_DDS_DOMAIN = 1
_DEFAULT_KP = 60.0
_DEFAULT_KD = 1.5
_DEFAULT_DQ = 0.0
_DEFAULT_TAU = 0.0
_DEFAULT_PUBLISH_HZ = 50.0
_DEFAULT_MOVE_HOLD_S = 2.0
_DEFAULT_NAMED_POSE_HOLD_S = 4.0
_LOWSTATE_TOPIC = "rt/lowstate"
_LOWCMD_TOPIC = "rt/lowcmd"
_K_NOT_USED_JOINT = 29
_G1_JOINTS = {
    "waist_yaw": 12,
    "waist_roll": 13,
    "waist_pitch": 14,
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
}
_REQUIRED_JOINT_INDICES = tuple(sorted(_G1_JOINTS.values()))
_READY_POSE = {
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": math.pi / 2,
    "left_shoulder_yaw": 0.0,
    "left_elbow": math.pi / 2,
    "left_wrist_roll": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": -math.pi / 2,
    "right_shoulder_yaw": 0.0,
    "right_elbow": math.pi / 2,
    "right_wrist_roll": 0.0,
    "waist_yaw": 0.0,
    "waist_roll": 0.0,
    "waist_pitch": 0.0,
}
_NAMED_POSES = {
    "home": {name: 0.0 for name in _G1_JOINTS},
    "ready": dict(_READY_POSE),
    "folded": {
        "left_shoulder_pitch": 0.2,
        "left_shoulder_roll": 0.6,
        "left_shoulder_yaw": 0.0,
        "left_elbow": 1.2,
        "left_wrist_roll": 0.0,
        "right_shoulder_pitch": -0.2,
        "right_shoulder_roll": -0.6,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 1.2,
        "right_wrist_roll": 0.0,
        "waist_yaw": 0.0,
        "waist_roll": 0.0,
        "waist_pitch": 0.0,
    },
}


class UnitreeG1Controller:
    """Minimal sim-only G1 controller for Isaac Lab lowcmd."""

    def __init__(self, handshake_timeout_s: float = 2.0) -> None:
        self.handshake_timeout_s = handshake_timeout_s
        self._channel_initialized = False
        self._connected = False
        self._subscriber: Any | None = None
        self._publisher: Any | None = None
        self._last_lowstate: Any | None = None
        self._last_lowstate_ts: float | None = None
        self._last_error: str | None = None
        self._lowstate_ready = threading.Event()

    async def doctor(self) -> dict[str, Any]:
        return {
            "sdk_available": self._module_exists("unitree_sdk2py"),
            "cyclonedds_available": self._module_exists("cyclonedds"),
            "command_topic": _LOWCMD_TOPIC,
            "lowstate_topic": _LOWSTATE_TOPIC,
        }

    async def status(self, config: dict[str, Any]) -> dict[str, Any]:
        details = self._debug_details(config)
        details["config"] = dict(config)
        details["connected"] = self._connected
        details["received_lowstate"] = self._received_lowstate
        details["joint_positions"] = self._current_joint_targets()
        return details

    async def connect(self, config: dict[str, Any]) -> dict[str, Any]:
        self._validate_config(config)
        channel_mod = self._import_channel_module()
        msg_mod = self._import_message_module()
        self._initialize_channel_factory(
            channel_mod.ChannelFactoryInitialize,
            domain=self._dds_domain(config),
            network_interface=self._network_interface(config),
        )
        self._subscriber = self._make_subscriber(channel_mod, msg_mod)
        self._publisher = self._make_publisher(channel_mod, msg_mod)
        self._channel_initialized = True
        self._connected = False
        self._last_error = None
        self._lowstate_ready.clear()
        self._last_lowstate = None
        self._last_lowstate_ts = None
        received = await asyncio.to_thread(self._lowstate_ready.wait, self.handshake_timeout_s)
        if not received:
            self._last_error = self._lowstate_timeout_message(config)
            raise RuntimeError(self._last_error)
        if not self._supports_required_joint_indices():
            self._last_error = "LowState arrived but does not expose the expected G1 arm joint indices."
            raise RuntimeError(self._last_error)
        self._connected = True
        return {
            "ok": True,
            "command_topic": _LOWCMD_TOPIC,
            "network_interface": self._network_interface(config),
            "dds_domain": self._dds_domain(config),
            "joint_count": self._joint_count(),
        }

    async def move_joint(
        self,
        config: dict[str, Any],
        positions: dict[str, Any],
        hold_seconds: float = _DEFAULT_MOVE_HOLD_S,
    ) -> dict[str, Any]:
        self._ensure_connected()
        invalid = [name for name in positions if name not in _G1_JOINTS]
        if invalid:
            raise ValueError(f"Unknown G1 joints: {', '.join(sorted(invalid))}")
        try:
            normalized = {name: float(value) for name, value in positions.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Joint target values must be numeric: {exc}") from exc
        low_cmd = self._build_low_cmd(normalized)
        writes = await self._publish_low_cmd(low_cmd, hold_seconds=hold_seconds)
        return {
            "ok": True,
            "positions": normalized,
            "hold_seconds": self._coerce_hold_seconds(hold_seconds, _DEFAULT_MOVE_HOLD_S),
            "write_count": writes,
            "command_topic": _LOWCMD_TOPIC,
            "network_interface": self._network_interface(config),
            "dds_domain": self._dds_domain(config),
        }

    async def go_named_pose(
        self,
        config: dict[str, Any],
        pose_name: str,
        hold_seconds: float = _DEFAULT_NAMED_POSE_HOLD_S,
    ) -> dict[str, Any]:
        pose_key = str(pose_name or "").strip().lower()
        if pose_key not in _NAMED_POSES:
            raise ValueError(f"Unknown G1 named pose '{pose_name}'. Available: {', '.join(sorted(_NAMED_POSES))}.")
        result = await self.move_joint(config, _NAMED_POSES[pose_key], hold_seconds=hold_seconds)
        result["named_pose"] = pose_key
        return result

    @property
    def _received_lowstate(self) -> bool:
        return self._last_lowstate is not None

    def _validate_config(self, config: dict[str, Any]) -> None:
        network_interface = self._network_interface(config)
        if not network_interface:
            raise ValueError("Unitree G1 network_interface is required.")
        if str(config.get("mode") or "sim") != "sim":
            raise ValueError("Unitree G1 controller currently supports only sim mode.")
        if str(config.get("robot_variant") or "g129_dex1") != "g129_dex1":
            raise ValueError("Unitree G1 sim requires robot_variant 'g129_dex1'.")
        if str(config.get("motion_source") or "lowcmd") != "lowcmd":
            raise ValueError("Unitree G1 sim requires motion_source 'lowcmd'.")
        domain = self._dds_domain(config)
        if domain < 0:
            raise ValueError("Unitree G1 dds_domain must be >= 0.")

    def _ensure_connected(self) -> None:
        if not self._connected or self._publisher is None:
            raise RuntimeError("Unitree G1 controller is not connected.")

    def _network_interface(self, config: dict[str, Any]) -> str:
        return str(config.get("network_interface") or "").strip()

    def _dds_domain(self, config: dict[str, Any]) -> int:
        value = config.get("dds_domain")
        return _DEFAULT_DDS_DOMAIN if value is None else int(value)

    def _module_exists(self, name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False

    def _import_channel_module(self) -> Any:
        return importlib.import_module("unitree_sdk2py.core.channel")

    def _import_message_module(self) -> dict[str, Any]:
        low_cmd_default = importlib.import_module("unitree_sdk2py.idl.default")
        low_types = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
        return {
            "LowCmdDefault": getattr(low_cmd_default, "unitree_hg_msg_dds__LowCmd_", None),
            "LowState": getattr(low_types, "LowState_", None),
            "LowCmd": getattr(low_types, "LowCmd_", None),
        }

    def _initialize_channel_factory(self, factory: Any, *, domain: int, network_interface: str) -> None:
        try:
            factory(domain, network_interface)
        except TypeError:
            factory(domain)

    def _make_subscriber(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        subscriber = channel_mod.ChannelSubscriber(_LOWSTATE_TOPIC, msg_mod["LowState"])
        subscriber.Init(self._lowstate_handler, 10)
        return subscriber

    def _make_publisher(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        publisher = channel_mod.ChannelPublisher(_LOWCMD_TOPIC, msg_mod["LowCmd"])
        publisher.Init()
        return publisher

    def _lowstate_handler(self, msg: Any) -> None:
        self._last_lowstate = msg
        self._last_lowstate_ts = time.time()
        self._lowstate_ready.set()

    def _joint_count(self) -> int | None:
        state = self._last_lowstate
        if state is None or not hasattr(state, "motor_state"):
            return None
        try:
            return len(getattr(state, "motor_state") or [])
        except Exception:
            return None

    def _supports_required_joint_indices(self) -> bool:
        joint_count = self._joint_count()
        return joint_count is not None and joint_count > max(_REQUIRED_JOINT_INDICES)

    def _lowstate_timeout_message(self, config: dict[str, Any]) -> str:
        return (
            f"Timed out waiting for the first G1 lowstate frame on interface '{self._network_interface(config)}' "
            f"(domain={self._dds_domain(config)}). For sim mode, verify unitree_sim_isaaclab is already running, "
            "Dex1 DDS is enabled, and the DDS domain/interface match RoboClaw."
        )

    def _current_joint_targets(self) -> dict[str, float]:
        state = self._last_lowstate
        motor_state = getattr(state, "motor_state", []) if state is not None else []
        return {
            name: float(getattr(motor_state[index], "q", 0.0))
            for name, index in _G1_JOINTS.items()
            if index < len(motor_state)
        }

    def _build_low_cmd(self, targets: dict[str, float]) -> Any:
        msg_mod = self._import_message_module()
        factory = msg_mod.get("LowCmdDefault") or msg_mod.get("LowCmd")
        if factory is None:
            raise RuntimeError("Unitree SDK2 message module does not expose a LowCmd constructor.")
        low_cmd = factory()
        motor_cmd = getattr(low_cmd, "motor_cmd", None)
        if motor_cmd is None:
            raise RuntimeError("Unitree LowCmd object is missing motor_cmd.")
        if len(motor_cmd) <= _K_NOT_USED_JOINT:
            raise RuntimeError("Unitree LowCmd object does not expose the expected command slots.")
        for command in motor_cmd:
            command.mode = 1
            command.q = 0.0
            command.dq = _DEFAULT_DQ
            command.tau = _DEFAULT_TAU
            command.kp = _DEFAULT_KP
            command.kd = _DEFAULT_KD
        for joint_name, index in _G1_JOINTS.items():
            if joint_name in targets:
                motor_cmd[index].q = targets[joint_name]
        crc = self._compute_crc(low_cmd)
        if crc is not None and hasattr(low_cmd, "crc"):
            low_cmd.crc = crc
        return low_cmd

    def _compute_crc(self, low_cmd: Any) -> int | None:
        try:
            crc_mod = importlib.import_module("unitree_sdk2py.utils.crc")
        except Exception:
            return None
        crc_cls = getattr(crc_mod, "CRC", None)
        if crc_cls is None:
            return None
        try:
            return crc_cls().Crc(low_cmd)
        except Exception:
            return None

    async def _publish_low_cmd(
        self,
        low_cmd: Any,
        *,
        hold_seconds: float,
        publish_hz: float = _DEFAULT_PUBLISH_HZ,
    ) -> int:
        if self._publisher is None or not hasattr(self._publisher, "Write"):
            raise RuntimeError("Unitree G1 lowcmd publisher is unavailable.")
        duration = max(0.02, float(hold_seconds))
        hz = max(1.0, float(publish_hz))
        interval = 1.0 / hz
        repeats = max(1, int(round(duration * hz)))
        count = 0
        for _ in range(repeats):
            self._publisher.Write(low_cmd)
            count += 1
            await asyncio.sleep(interval)
        return count

    def _coerce_hold_seconds(self, value: Any, default: float) -> float:
        if value is None:
            return default
        try:
            hold_seconds = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.02, hold_seconds)

    def _debug_details(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "network_interface": self._network_interface(config),
            "dds_domain": self._dds_domain(config),
            "mode": str(config.get("mode") or "sim"),
            "robot_variant": str(config.get("robot_variant") or "g129_dex1"),
            "motion_source": str(config.get("motion_source") or "lowcmd"),
            "sim_runtime": str(config.get("sim_runtime") or "isaaclab"),
            "command_topic": _LOWCMD_TOPIC,
            "received_lowstate": self._received_lowstate,
            "joint_count": self._joint_count(),
            "last_lowstate_timestamp": self._last_lowstate_ts,
            "channel_initialized": self._channel_initialized,
            "connected": self._connected,
            "last_error": self._last_error,
        }
