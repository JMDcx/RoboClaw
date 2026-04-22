"""Minimal Unitree G1 Isaac Lab controller using SDK2 DDS lowcmd."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
import threading
import time
from typing import Any

_DEFAULT_DDS_DOMAIN = 1
_DEFAULT_KP = 60.0
_DEFAULT_KD = 1.5
_DEFAULT_DQ = 0.0
_DEFAULT_TAU = 0.0
_DEFAULT_PUBLISH_HZ = 50.0
_DEFAULT_MOVE_HOLD_S = 2.0
_DEFAULT_NAMED_POSE_HOLD_S = 4.0
_DEFAULT_HAND_HOLD_S = 0.5
_LOWSTATE_TOPIC = "rt/lowstate"
_LOWCMD_TOPIC = "rt/lowcmd"
_INSPIRE_STATE_TOPIC = "rt/inspire/state"
_INSPIRE_CMD_TOPIC = "rt/inspire/cmd"
_DEX1_LEFT_STATE_TOPIC = "rt/dex1/left/state"
_DEX1_LEFT_CMD_TOPIC = "rt/dex1/left/cmd"
_DEX1_RIGHT_STATE_TOPIC = "rt/dex1/right/state"
_DEX1_RIGHT_CMD_TOPIC = "rt/dex1/right/cmd"
_K_NOT_USED_JOINT = 29
_G1_VARIANTS = {"g129_dex1", "g129_inspire"}
_G1_JOINTS = {
    "waist_yaw": 12,
    "waist_roll": 13,
    "waist_pitch": 14,
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}
_REQUIRED_JOINT_INDICES = tuple(sorted(_G1_JOINTS.values()))
_READY_POSE = {
    "left_shoulder_pitch": 0.0,
    "left_shoulder_roll": math.pi / 2,
    "left_shoulder_yaw": 0.0,
    "left_elbow": math.pi / 2,
    "left_wrist_roll": 0.0,
    "left_wrist_pitch": 0.0,
    "left_wrist_yaw": 0.0,
    "right_shoulder_pitch": 0.0,
    "right_shoulder_roll": -math.pi / 2,
    "right_shoulder_yaw": 0.0,
    "right_elbow": math.pi / 2,
    "right_wrist_roll": 0.0,
    "right_wrist_pitch": 0.0,
    "right_wrist_yaw": 0.0,
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
        "left_wrist_pitch": 0.0,
        "left_wrist_yaw": 0.0,
        "right_shoulder_pitch": -0.2,
        "right_shoulder_roll": -0.6,
        "right_shoulder_yaw": 0.0,
        "right_elbow": 1.2,
        "right_wrist_roll": 0.0,
        "right_wrist_pitch": 0.0,
        "right_wrist_yaw": 0.0,
        "waist_yaw": 0.0,
        "waist_roll": 0.0,
        "waist_pitch": 0.0,
    },
}
_HAND_JOINT_ORDER = (
    "right_pinky",
    "right_ring",
    "right_middle",
    "right_index",
    "right_thumb_bend",
    "right_thumb_rotation",
    "left_pinky",
    "left_ring",
    "left_middle",
    "left_index",
    "left_thumb_bend",
    "left_thumb_rotation",
)
_HAND_JOINTS = {name: index for index, name in enumerate(_HAND_JOINT_ORDER)}
_HAND_FINGER_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
_HAND_SIDES = {"left", "right", "both"}
_DEX1_GRIPPER_SIDES = {"right"}
_DEX1_GRIPPER_OPEN_Q = 5.4
_DEX1_GRIPPER_CLOSED_Q = 0.0
_HAND_OPEN = {
    "right_pinky": 0.05,
    "right_ring": 0.05,
    "right_middle": 0.05,
    "right_index": 0.05,
    "right_thumb_bend": 0.15,
    "right_thumb_rotation": 0.35,
    "left_pinky": 0.05,
    "left_ring": 0.05,
    "left_middle": 0.05,
    "left_index": 0.05,
    "left_thumb_bend": 0.15,
    "left_thumb_rotation": 0.35,
}
_HAND_JOINT_LIMITS = {
    "pinky": (0.0, 1.7),
    "ring": (0.0, 1.7),
    "middle": (0.0, 1.7),
    "index": (0.0, 1.7),
    "thumb_bend": (0.0, 0.5),
    "thumb_rotation": (-0.1, 1.3),
}
_HAND_PRESETS = {
    "open": {
        "pinky": 0.05,
        "ring": 0.05,
        "middle": 0.05,
        "index": 0.05,
        "thumb_bend": 0.15,
        "thumb_rotation": 0.35,
    },
    "grasp": {
        "pinky": 1.0,
        "ring": 1.0,
        "middle": 1.0,
        "index": 1.0,
        "thumb_bend": 1.0,
        "thumb_rotation": 1.0,
    },
    "pinch": {
        "pinky": 0.8,
        "ring": 0.8,
        "middle": 0.8,
        "index": 0.15,
        "thumb_bend": 0.1,
        "thumb_rotation": 0.15,
    },
    "tripod": {
        "pinky": 0.8,
        "ring": 0.8,
        "middle": 0.2,
        "index": 0.2,
        "thumb_bend": 0.1,
        "thumb_rotation": 0.2,
    },
    "relax": {finger: 0.6 for finger in _HAND_FINGER_ORDER},
}


class UnitreeG1Controller:
    """Minimal sim-only G1 controller for Isaac Lab lowcmd."""

    def __init__(self, handshake_timeout_s: float = 2.0) -> None:
        self.handshake_timeout_s = handshake_timeout_s
        self._channel_initialized = False
        self._connected = False
        self._subscriber: Any | None = None
        self._publisher: Any | None = None
        self._hand_subscriber: Any | None = None
        self._hand_publisher: Any | None = None
        self._gripper_subscriber: Any | None = None
        self._gripper_publisher: Any | None = None
        self._gripper_left_subscriber: Any | None = None
        self._gripper_left_publisher: Any | None = None
        self._last_lowstate: Any | None = None
        self._last_lowstate_ts: float | None = None
        self._last_hand_state: Any | None = None
        self._last_hand_state_ts: float | None = None
        self._last_gripper_state: Any | None = None
        self._last_gripper_state_ts: float | None = None
        self._last_left_gripper_state: Any | None = None
        self._last_left_gripper_state_ts: float | None = None
        self._last_hand_targets: dict[str, float] = dict(_HAND_OPEN)
        self._last_gripper_targets: dict[str, float] = {"left": 0.0, "right": 0.0}
        self._last_error: str | None = None
        self._lowstate_ready = threading.Event()

    async def doctor(self) -> dict[str, Any]:
        return {
            "sdk_available": self._module_exists("unitree_sdk2py"),
            "cyclonedds_available": self._module_exists("cyclonedds"),
            "command_topic": _LOWCMD_TOPIC,
            "lowstate_topic": _LOWSTATE_TOPIC,
            "inspire_command_topic": _INSPIRE_CMD_TOPIC,
            "inspire_state_topic": _INSPIRE_STATE_TOPIC,
            "dex1_left_command_topic": _DEX1_LEFT_CMD_TOPIC,
            "dex1_left_state_topic": _DEX1_LEFT_STATE_TOPIC,
            "dex1_right_command_topic": _DEX1_RIGHT_CMD_TOPIC,
            "dex1_right_state_topic": _DEX1_RIGHT_STATE_TOPIC,
        }

    async def status(self, config: dict[str, Any]) -> dict[str, Any]:
        details = self._debug_details(config)
        details["config"] = dict(config)
        details["connected"] = self._connected
        details["received_lowstate"] = self._received_lowstate
        details["joint_positions"] = self._current_joint_targets()
        details["received_handstate"] = self._received_handstate
        details["hand_positions"] = self._current_hand_targets()
        details["hand_targets"] = dict(self._last_hand_targets)
        details["received_gripper_state"] = self._received_gripper_state
        details["gripper_close_amounts"] = self._current_gripper_targets()
        details["gripper_targets"] = dict(self._last_gripper_targets)
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
        if self._is_inspire_variant(config):
            hand_msg_mod = self._import_hand_message_module()
            self._hand_subscriber = self._make_hand_subscriber(channel_mod, hand_msg_mod)
            self._hand_publisher = self._make_hand_publisher(channel_mod, hand_msg_mod)
            self._gripper_subscriber = None
            self._gripper_publisher = None
            self._gripper_left_subscriber = None
            self._gripper_left_publisher = None
            self._last_gripper_state = None
            self._last_gripper_state_ts = None
            self._last_left_gripper_state = None
            self._last_left_gripper_state_ts = None
            self._last_gripper_targets = {"left": 0.0, "right": 0.0}
        elif self._is_dex1_variant(config):
            gripper_msg_mod = self._import_gripper_message_module()
            self._gripper_left_subscriber = self._make_left_gripper_subscriber(channel_mod, gripper_msg_mod)
            self._gripper_left_publisher = self._make_left_gripper_publisher(channel_mod, gripper_msg_mod)
            self._gripper_subscriber = self._make_gripper_subscriber(channel_mod, gripper_msg_mod)
            self._gripper_publisher = self._make_gripper_publisher(channel_mod, gripper_msg_mod)
            self._hand_subscriber = None
            self._hand_publisher = None
            self._last_hand_state = None
            self._last_hand_state_ts = None
            self._last_hand_targets = dict(_HAND_OPEN)
        else:
            self._hand_subscriber = None
            self._hand_publisher = None
            self._gripper_subscriber = None
            self._gripper_publisher = None
            self._gripper_left_subscriber = None
            self._gripper_left_publisher = None
            self._last_hand_state = None
            self._last_hand_state_ts = None
            self._last_gripper_state = None
            self._last_gripper_state_ts = None
            self._last_left_gripper_state = None
            self._last_left_gripper_state_ts = None
            self._last_hand_targets = dict(_HAND_OPEN)
            self._last_gripper_targets = {"left": 0.0, "right": 0.0}
        self._connected = True
        return {
            "ok": True,
            "command_topic": _LOWCMD_TOPIC,
            "network_interface": self._network_interface(config),
            "dds_domain": self._dds_domain(config),
            "joint_count": self._joint_count(),
            "robot_variant": self._robot_variant(config),
            "hand_command_topic": _INSPIRE_CMD_TOPIC if self._is_inspire_variant(config) else None,
            "hand_state_topic": _INSPIRE_STATE_TOPIC if self._is_inspire_variant(config) else None,
            "gripper_command_topic": _DEX1_RIGHT_CMD_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_state_topic": _DEX1_RIGHT_STATE_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_left_command_topic": _DEX1_LEFT_CMD_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_left_state_topic": _DEX1_LEFT_STATE_TOPIC if self._is_dex1_variant(config) else None,
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
        writes = await self._publish_message(self._publisher, low_cmd, hold_seconds=hold_seconds)
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

    async def hand_status(self, config: dict[str, Any]) -> dict[str, Any]:
        self._ensure_inspire_hand_supported(config)
        return {
            "ok": True,
            "robot_variant": self._robot_variant(config),
            "hand_command_topic": _INSPIRE_CMD_TOPIC,
            "hand_state_topic": _INSPIRE_STATE_TOPIC,
            "received_handstate": self._received_handstate,
            "hand_positions": self._current_hand_targets(),
            "hand_targets": dict(self._last_hand_targets),
            "last_hand_state_timestamp": self._last_hand_state_ts,
            "connected": self._connected,
        }

    async def hand_move(
        self,
        config: dict[str, Any],
        positions: dict[str, Any],
        *,
        side: str = "both",
        hold_seconds: float = _DEFAULT_HAND_HOLD_S,
    ) -> dict[str, Any]:
        self._ensure_inspire_hand_supported(config)
        normalized_side = self._coerce_hand_side(side)
        expanded = self._expand_hand_positions(positions, normalized_side)
        targets = dict(self._last_hand_targets or _HAND_OPEN)
        targets.update(expanded)
        hand_cmd = self._build_hand_cmd(targets)
        writes = await self._publish_message(self._hand_publisher, hand_cmd, hold_seconds=hold_seconds)
        self._last_hand_targets = targets
        return {
            "ok": True,
            "side": normalized_side,
            "positions": dict(targets),
            "applied_positions": dict(expanded),
            "hold_seconds": self._coerce_hold_seconds(hold_seconds, _DEFAULT_HAND_HOLD_S),
            "write_count": writes,
            "command_topic": _INSPIRE_CMD_TOPIC,
            "state_topic": _INSPIRE_STATE_TOPIC,
        }

    async def hand_preset(
        self,
        config: dict[str, Any],
        preset_name: str,
        *,
        side: str = "both",
        hold_seconds: float = _DEFAULT_HAND_HOLD_S,
    ) -> dict[str, Any]:
        preset_key = str(preset_name or "").strip().lower()
        if preset_key not in _HAND_PRESETS:
            raise ValueError(
                f"Unknown G1 hand preset '{preset_name}'. Available: {', '.join(sorted(_HAND_PRESETS))}."
            )
        result = await self.hand_move(
            config,
            dict(_HAND_PRESETS[preset_key]),
            side=side,
            hold_seconds=hold_seconds,
        )
        result["preset_name"] = preset_key
        return result

    async def gripper_status(self, config: dict[str, Any]) -> dict[str, Any]:
        self._ensure_dex1_gripper_supported(config)
        return {
            "ok": True,
            "robot_variant": self._robot_variant(config),
            "gripper_command_topic": _DEX1_RIGHT_CMD_TOPIC,
            "gripper_state_topic": _DEX1_RIGHT_STATE_TOPIC,
            "received_gripper_state": self._received_gripper_state,
            "gripper_close_amounts": self._current_gripper_targets(),
            "gripper_targets": dict(self._last_gripper_targets),
            "last_gripper_state_timestamp": self._last_gripper_state_ts,
            "connected": self._connected,
        }

    async def gripper_move(
        self,
        config: dict[str, Any],
        close_amount: Any,
        *,
        side: str = "right",
        hold_seconds: float = _DEFAULT_HAND_HOLD_S,
    ) -> dict[str, Any]:
        self._ensure_dex1_gripper_supported(config)
        normalized_side = self._coerce_gripper_side(side)
        normalized_close_amount = self._coerce_close_amount(close_amount)
        left_open_cmd = self._build_gripper_cmd(self._last_gripper_targets.get("left", 0.0))
        gripper_cmd = self._build_gripper_cmd(normalized_close_amount)
        left_writes = await self._publish_message(self._gripper_left_publisher, left_open_cmd, hold_seconds=hold_seconds)
        right_writes = await self._publish_message(self._gripper_publisher, gripper_cmd, hold_seconds=hold_seconds)
        self._last_gripper_targets["left"] = self._coerce_close_amount(self._last_gripper_targets.get("left", 0.0))
        self._last_gripper_targets[normalized_side] = normalized_close_amount
        return {
            "ok": True,
            "side": normalized_side,
            "close_amount": normalized_close_amount,
            "hold_seconds": self._coerce_hold_seconds(hold_seconds, _DEFAULT_HAND_HOLD_S),
            "write_count": right_writes,
            "left_write_count": left_writes,
            "command_topic": _DEX1_RIGHT_CMD_TOPIC,
            "left_command_topic": _DEX1_LEFT_CMD_TOPIC,
            "state_topic": _DEX1_RIGHT_STATE_TOPIC,
        }

    async def gripper_open(
        self,
        config: dict[str, Any],
        *,
        side: str = "right",
        hold_seconds: float = _DEFAULT_HAND_HOLD_S,
    ) -> dict[str, Any]:
        result = await self.gripper_move(config, 0.0, side=side, hold_seconds=hold_seconds)
        result["preset_name"] = "open"
        return result

    async def gripper_close(
        self,
        config: dict[str, Any],
        *,
        side: str = "right",
        hold_seconds: float = _DEFAULT_HAND_HOLD_S,
    ) -> dict[str, Any]:
        result = await self.gripper_move(config, 1.0, side=side, hold_seconds=hold_seconds)
        result["preset_name"] = "close"
        return result

    @property
    def _received_lowstate(self) -> bool:
        return self._last_lowstate is not None

    @property
    def _received_handstate(self) -> bool:
        return self._last_hand_state is not None

    @property
    def _received_gripper_state(self) -> bool:
        return self._last_gripper_state is not None

    def _validate_config(self, config: dict[str, Any]) -> None:
        network_interface = self._network_interface(config)
        if not network_interface:
            raise ValueError("Unitree G1 network_interface is required.")
        if str(config.get("mode") or "sim") != "sim":
            raise ValueError("Unitree G1 controller currently supports only sim mode.")
        robot_variant = self._robot_variant(config)
        if robot_variant not in _G1_VARIANTS:
            raise ValueError(f"Unitree G1 sim requires robot_variant in {sorted(_G1_VARIANTS)}.")
        if str(config.get("motion_source") or "lowcmd") != "lowcmd":
            raise ValueError("Unitree G1 sim requires motion_source 'lowcmd'.")
        domain = self._dds_domain(config)
        if domain < 0:
            raise ValueError("Unitree G1 dds_domain must be >= 0.")

    def _ensure_connected(self) -> None:
        if not self._connected or self._publisher is None:
            raise RuntimeError("Unitree G1 controller is not connected.")

    def _ensure_inspire_hand_supported(self, config: dict[str, Any]) -> None:
        self._ensure_connected()
        if not self._is_inspire_variant(config):
            raise ValueError("Unitree G1 hand control requires robot_variant 'g129_inspire'.")
        if self._hand_publisher is None:
            raise RuntimeError("Unitree G1 Inspire hand publisher is unavailable.")

    def _ensure_dex1_gripper_supported(self, config: dict[str, Any]) -> None:
        self._ensure_connected()
        if not self._is_dex1_variant(config):
            raise ValueError("Unitree G1 gripper control requires robot_variant 'g129_dex1'.")
        if self._gripper_publisher is None or self._gripper_left_publisher is None:
            raise RuntimeError("Unitree G1 Dex1 gripper publisher is unavailable.")

    def _network_interface(self, config: dict[str, Any]) -> str:
        return str(config.get("network_interface") or "").strip()

    def _dds_domain(self, config: dict[str, Any]) -> int:
        value = config.get("dds_domain")
        return _DEFAULT_DDS_DOMAIN if value is None else int(value)

    def _robot_variant(self, config: dict[str, Any]) -> str:
        return str(config.get("robot_variant") or "g129_dex1")

    def _is_inspire_variant(self, config: dict[str, Any]) -> bool:
        return self._robot_variant(config) == "g129_inspire"

    def _is_dex1_variant(self, config: dict[str, Any]) -> bool:
        return self._robot_variant(config) == "g129_dex1"

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

    def _import_hand_message_module(self) -> dict[str, Any]:
        default_mod = importlib.import_module("unitree_sdk2py.idl.default")
        go_types = importlib.import_module("unitree_sdk2py.idl.unitree_go.msg.dds_")
        return {
            "MotorCmdDefault": getattr(default_mod, "unitree_go_msg_dds__MotorCmd_", None),
            "MotorCmds": getattr(go_types, "MotorCmds_", None),
            "MotorStates": getattr(go_types, "MotorStates_", None),
        }

    def _import_gripper_message_module(self) -> dict[str, Any]:
        return self._import_hand_message_module()

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

    def _make_hand_subscriber(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        subscriber = channel_mod.ChannelSubscriber(_INSPIRE_STATE_TOPIC, msg_mod["MotorStates"])
        subscriber.Init(self._hand_state_handler, 10)
        return subscriber

    def _make_hand_publisher(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        publisher = channel_mod.ChannelPublisher(_INSPIRE_CMD_TOPIC, msg_mod["MotorCmds"])
        publisher.Init()
        return publisher

    def _make_gripper_subscriber(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        subscriber = channel_mod.ChannelSubscriber(_DEX1_RIGHT_STATE_TOPIC, msg_mod["MotorStates"])
        subscriber.Init(self._gripper_state_handler, 10)
        return subscriber

    def _make_gripper_publisher(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        publisher = channel_mod.ChannelPublisher(_DEX1_RIGHT_CMD_TOPIC, msg_mod["MotorCmds"])
        publisher.Init()
        return publisher

    def _make_left_gripper_subscriber(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        subscriber = channel_mod.ChannelSubscriber(_DEX1_LEFT_STATE_TOPIC, msg_mod["MotorStates"])
        subscriber.Init(self._left_gripper_state_handler, 10)
        return subscriber

    def _make_left_gripper_publisher(self, channel_mod: Any, msg_mod: dict[str, Any]) -> Any:
        publisher = channel_mod.ChannelPublisher(_DEX1_LEFT_CMD_TOPIC, msg_mod["MotorCmds"])
        publisher.Init()
        return publisher

    def _lowstate_handler(self, msg: Any) -> None:
        self._last_lowstate = msg
        self._last_lowstate_ts = time.time()
        self._lowstate_ready.set()

    def _hand_state_handler(self, msg: Any) -> None:
        self._last_hand_state = msg
        self._last_hand_state_ts = time.time()

    def _gripper_state_handler(self, msg: Any) -> None:
        self._last_gripper_state = msg
        self._last_gripper_state_ts = time.time()

    def _left_gripper_state_handler(self, msg: Any) -> None:
        self._last_left_gripper_state = msg
        self._last_left_gripper_state_ts = time.time()

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
        variant_hint = "Inspire" if self._is_inspire_variant(config) else "Dex1"
        return (
            f"Timed out waiting for the first G1 lowstate frame on interface '{self._network_interface(config)}' "
            f"(domain={self._dds_domain(config)}). For sim mode, verify unitree_sim_isaaclab is already running, "
            f"{variant_hint} DDS is enabled, and the DDS domain/interface match RoboClaw."
        )

    def _current_joint_targets(self) -> dict[str, float]:
        state = self._last_lowstate
        motor_state = getattr(state, "motor_state", []) if state is not None else []
        return {
            name: float(getattr(motor_state[index], "q", 0.0))
            for name, index in _G1_JOINTS.items()
            if index < len(motor_state)
        }

    def _current_hand_targets(self) -> dict[str, float]:
        state = self._last_hand_state
        hand_state = getattr(state, "states", []) if state is not None else []
        if not hand_state:
            return {}
        positions: dict[str, float] = {}
        for name, index in _HAND_JOINTS.items():
            if index >= len(hand_state):
                continue
            positions[name] = self._normalize_hand_q(name, float(getattr(hand_state[index], "q", 0.0)))
        return positions

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

    def _build_hand_cmd(self, targets: dict[str, float]) -> Any:
        msg_mod = self._import_hand_message_module()
        cmds_type = msg_mod.get("MotorCmds")
        cmd_type = msg_mod.get("MotorCmdDefault")
        if cmds_type is None or cmd_type is None:
            raise RuntimeError("Unitree SDK2 hand message module does not expose Inspire command constructors.")
        hand_cmd = cmds_type()
        hand_cmd.cmds = [cmd_type() for _ in range(len(_HAND_JOINT_ORDER))]
        for name, index in _HAND_JOINTS.items():
            command = hand_cmd.cmds[index]
            command.q = self._denormalize_hand_q(name, targets[name])
            if hasattr(command, "dq"):
                command.dq = 0.0
            if hasattr(command, "tau"):
                command.tau = 0.0
            if hasattr(command, "kp"):
                command.kp = 0.0
            if hasattr(command, "kd"):
                command.kd = 0.0
        return hand_cmd

    def _build_gripper_cmd(self, close_amount: float) -> Any:
        msg_mod = self._import_gripper_message_module()
        cmds_type = msg_mod.get("MotorCmds")
        cmd_type = msg_mod.get("MotorCmdDefault")
        if cmds_type is None or cmd_type is None:
            raise RuntimeError("Unitree SDK2 gripper message module does not expose Dex1 command constructors.")
        gripper_cmd = cmds_type()
        gripper_cmd.cmds = [cmd_type()]
        command = gripper_cmd.cmds[0]
        command.q = self._denormalize_gripper_q(close_amount)
        if hasattr(command, "dq"):
            command.dq = 0.0
        if hasattr(command, "tau"):
            command.tau = 0.0
        if hasattr(command, "kp"):
            command.kp = 0.0
        if hasattr(command, "kd"):
            command.kd = 0.0
        return gripper_cmd

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

    async def _publish_message(
        self,
        publisher: Any,
        message: Any,
        *,
        hold_seconds: float,
        publish_hz: float = _DEFAULT_PUBLISH_HZ,
    ) -> int:
        if publisher is None or not hasattr(publisher, "Write"):
            raise RuntimeError("Unitree publisher is unavailable.")
        duration = max(0.02, float(hold_seconds))
        hz = max(1.0, float(publish_hz))
        interval = 1.0 / hz
        repeats = max(1, int(round(duration * hz)))
        count = 0
        for _ in range(repeats):
            publisher.Write(message)
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

    def _coerce_hand_side(self, side: Any) -> str:
        normalized = str(side or "both").strip().lower()
        if normalized not in _HAND_SIDES:
            raise ValueError(f"Unknown G1 hand side '{side}'. Available: {', '.join(sorted(_HAND_SIDES))}.")
        return normalized

    def _coerce_gripper_side(self, side: Any) -> str:
        normalized = str(side or "right").strip().lower()
        if normalized not in _DEX1_GRIPPER_SIDES:
            raise ValueError(
                f"Unknown G1 gripper side '{side}'. Available: {', '.join(sorted(_DEX1_GRIPPER_SIDES))}."
            )
        return normalized

    def _coerce_close_amount(self, value: Any) -> float:
        try:
            close_amount = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"G1 Dex1 close_amount must be numeric: {exc}") from exc
        if not 0.0 <= close_amount <= 1.0:
            raise ValueError("G1 Dex1 close_amount must be within [0, 1].")
        return close_amount

    def _expand_hand_positions(self, positions: dict[str, Any], side: str) -> dict[str, float]:
        if not isinstance(positions, dict) or not positions:
            raise ValueError("G1 hand positions must be a non-empty object.")
        expanded: dict[str, float] = {}
        for key, raw_value in positions.items():
            names = self._resolve_hand_position_keys(str(key), side)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"G1 hand target '{key}' must be numeric: {exc}") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"G1 hand target '{key}' must be within [0, 1].")
            for name in names:
                expanded[name] = value
        return expanded

    def _resolve_hand_position_keys(self, key: str, side: str) -> list[str]:
        normalized = key.strip().lower()
        if normalized in _HAND_JOINTS:
            return [normalized]
        if normalized not in _HAND_FINGER_ORDER:
            raise ValueError(f"Unknown G1 hand joint '{key}'.")
        if side == "both":
            return [f"left_{normalized}", f"right_{normalized}"]
        return [f"{side}_{normalized}"]

    def _denormalize_hand_q(self, joint_name: str, value: float) -> float:
        low, high = _HAND_JOINT_LIMITS[self._hand_joint_kind(joint_name)]
        return high - value * (high - low)

    def _normalize_hand_q(self, joint_name: str, q_value: float) -> float:
        low, high = _HAND_JOINT_LIMITS[self._hand_joint_kind(joint_name)]
        if high == low:
            return 0.0
        normalized = (high - q_value) / (high - low)
        return max(0.0, min(1.0, normalized))

    def _denormalize_gripper_q(self, close_amount: float) -> float:
        return _DEX1_GRIPPER_OPEN_Q + (_DEX1_GRIPPER_CLOSED_Q - _DEX1_GRIPPER_OPEN_Q) * float(close_amount)

    def _normalize_gripper_q(self, q_value: float) -> float:
        span = _DEX1_GRIPPER_OPEN_Q - _DEX1_GRIPPER_CLOSED_Q
        if span <= 0.0:
            return 0.0
        normalized = (_DEX1_GRIPPER_OPEN_Q - float(q_value)) / span
        return max(0.0, min(1.0, normalized))

    def _hand_joint_kind(self, joint_name: str) -> str:
        _, _, kind = joint_name.partition("_")
        if kind.startswith("thumb_"):
            return kind
        return kind

    def _current_gripper_targets(self) -> dict[str, float]:
        positions = dict(self._last_gripper_targets)
        right_state = getattr(self._last_gripper_state, "states", []) if self._last_gripper_state is not None else []
        left_state = getattr(self._last_left_gripper_state, "states", []) if self._last_left_gripper_state is not None else []
        if right_state:
            positions["right"] = self._normalize_gripper_q(float(getattr(right_state[0], "q", _DEX1_GRIPPER_OPEN_Q)))
        if left_state:
            positions["left"] = self._normalize_gripper_q(float(getattr(left_state[0], "q", _DEX1_GRIPPER_OPEN_Q)))
        return positions

    def _debug_details(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "network_interface": self._network_interface(config),
            "dds_domain": self._dds_domain(config),
            "mode": str(config.get("mode") or "sim"),
            "robot_variant": self._robot_variant(config),
            "motion_source": str(config.get("motion_source") or "lowcmd"),
            "sim_runtime": str(config.get("sim_runtime") or "isaaclab"),
            "command_topic": _LOWCMD_TOPIC,
            "received_lowstate": self._received_lowstate,
            "joint_count": self._joint_count(),
            "last_lowstate_timestamp": self._last_lowstate_ts,
            "channel_initialized": self._channel_initialized,
            "connected": self._connected,
            "last_error": self._last_error,
            "hand_command_topic": _INSPIRE_CMD_TOPIC if self._is_inspire_variant(config) else None,
            "hand_state_topic": _INSPIRE_STATE_TOPIC if self._is_inspire_variant(config) else None,
            "received_handstate": self._received_handstate,
            "last_hand_state_timestamp": self._last_hand_state_ts,
            "gripper_command_topic": _DEX1_RIGHT_CMD_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_state_topic": _DEX1_RIGHT_STATE_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_left_command_topic": _DEX1_LEFT_CMD_TOPIC if self._is_dex1_variant(config) else None,
            "gripper_left_state_topic": _DEX1_LEFT_STATE_TOPIC if self._is_dex1_variant(config) else None,
            "received_gripper_state": self._received_gripper_state,
            "last_gripper_state_timestamp": self._last_gripper_state_ts,
        }
