"""Tests for minimal Unitree G1 embodied support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.embodiment.g1 import UnitreeG1Controller
from roboclaw.embodied.setup import clear_unitree_g1, load_setup, save_setup, set_unitree_g1
from roboclaw.embodied.tool import EmbodiedTool


@pytest.fixture()
def setup_file(tmp_path: Path) -> Path:
    p = tmp_path / "setup.json"
    base = {
        "version": 2,
        "arms": [],
        "cameras": {},
        "datasets": {"root": "/data"},
        "policies": {"root": "/policies"},
        "scanned_ports": [],
        "scanned_cameras": [],
    }
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_set_unitree_g1(setup_file: Path) -> None:
    result = set_unitree_g1(network_interface="wlp68s0", dds_domain=1, enabled=True, path=setup_file)
    cfg = result["unitree_g1"]
    assert cfg["enabled"] is True
    assert cfg["connected"] is False
    assert cfg["mode"] == "sim"
    assert cfg["network_interface"] == "wlp68s0"
    assert cfg["dds_domain"] == 1
    assert cfg["robot_variant"] == "g129_dex1"
    assert cfg["motion_source"] == "lowcmd"
    assert cfg["sim_runtime"] == "isaaclab"


def test_set_unitree_g1_accepts_inspire_variant(setup_file: Path) -> None:
    result = set_unitree_g1(
        network_interface="wlp68s0",
        dds_domain=1,
        enabled=True,
        robot_variant="g129_inspire",
        path=setup_file,
    )
    assert result["unitree_g1"]["robot_variant"] == "g129_inspire"


def test_clear_unitree_g1(setup_file: Path) -> None:
    set_unitree_g1(network_interface="wlp68s0", dds_domain=1, enabled=True, connected=True, path=setup_file)
    result = clear_unitree_g1(path=setup_file)
    cfg = result["unitree_g1"]
    assert cfg["enabled"] is False
    assert cfg["connected"] is False
    assert cfg["network_interface"] == ""


def test_validation_rejects_invalid_unitree_g1(setup_file: Path) -> None:
    setup = load_setup(setup_file)
    setup["unitree_g1"] = {
        "enabled": True,
        "connected": False,
        "mode": "real",
        "network_interface": "wlp68s0",
        "dds_domain": 1,
        "robot_variant": "g129_dex1",
        "motion_source": "lowcmd",
        "sim_runtime": "isaaclab",
    }
    with pytest.raises(ValueError, match="mode must be 'sim'"):
        save_setup(setup, setup_file)


def test_validation_rejects_invalid_variant(setup_file: Path) -> None:
    setup = load_setup(setup_file)
    setup["unitree_g1"] = {
        "enabled": True,
        "connected": False,
        "mode": "sim",
        "network_interface": "wlp68s0",
        "dds_domain": 1,
        "robot_variant": "bad_variant",
        "motion_source": "lowcmd",
        "sim_runtime": "isaaclab",
    }
    with pytest.raises(ValueError, match="robot_variant"):
        save_setup(setup, setup_file)


def test_tool_schema_has_g1_actions() -> None:
    tool = EmbodiedTool()
    actions = tool.parameters["properties"]["action"]["enum"]
    assert "g1_setup" in actions
    assert "g1_connect" in actions
    assert "g1_status" in actions
    assert "g1_move_joint" in actions
    assert "g1_go_named_pose" in actions
    assert "g1_hand_status" in actions
    assert "g1_hand_move" in actions
    assert "g1_hand_preset" in actions


@pytest.mark.asyncio
async def test_g1_setup_action_accepts_variant() -> None:
    tool = EmbodiedTool()
    with patch("roboclaw.embodied.setup.set_unitree_g1") as set_g1, patch("roboclaw.embodied.setup.clear_unitree_g1"):
        set_g1.return_value = {
            "unitree_g1": {
                "enabled": True,
                "connected": False,
                "mode": "sim",
                "network_interface": "wlp68s0",
                "dds_domain": 1,
                "robot_variant": "g129_inspire",
                "motion_source": "lowcmd",
                "sim_runtime": "isaaclab",
            }
        }
        result = await tool.execute(
            action="g1_setup",
            network_interface="wlp68s0",
            dds_domain=1,
            robot_variant="g129_inspire",
        )
    assert "Unitree G1 simulation configured" in result
    assert '"g129_inspire"' in result
    assert set_g1.call_args.kwargs["robot_variant"] == "g129_inspire"


@pytest.mark.asyncio
async def test_g1_connect_status_and_hand_actions() -> None:
    tool = EmbodiedTool()
    tool._g1_controller = SimpleNamespace(
        connect=AsyncMock(return_value={"ok": True, "command_topic": "rt/lowcmd", "hand_command_topic": "rt/inspire/cmd"}),
        status=AsyncMock(return_value={"connected": True, "command_topic": "rt/lowcmd"}),
        move_joint=AsyncMock(return_value={"ok": True, "positions": {"left_shoulder_pitch": 1.2}}),
        go_named_pose=AsyncMock(return_value={"ok": True, "named_pose": "folded"}),
        hand_status=AsyncMock(return_value={"ok": True, "received_handstate": False}),
        hand_move=AsyncMock(return_value={"ok": True, "positions": {"left_index": 0.2}}),
        hand_preset=AsyncMock(return_value={"ok": True, "preset_name": "grasp"}),
    )
    setup = {
        "unitree_g1": {
            "enabled": True,
            "connected": False,
            "mode": "sim",
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
            "sim_runtime": "isaaclab",
        }
    }
    with patch("roboclaw.embodied.setup.ensure_setup", return_value=setup), patch("roboclaw.embodied.setup.set_unitree_g1"):
        connect_result = await tool.execute(action="g1_connect")
        status_result = await tool.execute(action="g1_status")
        move_result = await tool.execute(action="g1_move_joint", joint_positions='{"left_shoulder_pitch": 1.2}', hold_seconds=0.5)
        pose_result = await tool.execute(action="g1_go_named_pose", pose_name="folded", hold_seconds=1.0)
        hand_status_result = await tool.execute(action="g1_hand_status")
        hand_move_result = await tool.execute(
            action="g1_hand_move",
            side="left",
            hand_positions='{"index": 0.2}',
            hold_seconds=0.5,
        )
        hand_preset_result = await tool.execute(action="g1_hand_preset", preset_name="grasp", side="both")
    assert "rt/lowcmd" in connect_result
    assert "connected" in status_result
    assert "left_shoulder_pitch" in move_result
    assert "folded" in pose_result
    assert "received_handstate" in hand_status_result
    assert "left_index" in hand_move_result
    assert "grasp" in hand_preset_result


class _FakeMotorCmd:
    def __init__(self) -> None:
        self.mode = 0
        self.tau = None
        self.dq = None
        self.kp = None
        self.kd = None
        self.q = None


class _FakeLowCmd:
    def __init__(self) -> None:
        self.motor_cmd = [_FakeMotorCmd() for _ in range(30)]
        self.crc = None


class _FakeMotorState:
    def __init__(self, q: float = 0.0) -> None:
        self.q = q


class _FakeLowState:
    def __init__(self, count: int = 35) -> None:
        self.motor_state = [_FakeMotorState() for _ in range(count)]


class _FakeHandCmd:
    def __init__(self) -> None:
        self.cmds = []


class _FakeHandState:
    def __init__(self, count: int = 12, q: float = 0.0) -> None:
        self.states = [_FakeMotorState(q=q) for _ in range(count)]


class _FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def Write(self, msg) -> None:
        self.messages.append(msg)


@pytest.mark.asyncio
async def test_g1_connect_success() -> None:
    controller = UnitreeG1Controller(handshake_timeout_s=0.1)

    class FakeSubscriber:
        def __init__(self, topic, msg_type) -> None:
            self.topic = topic
            self.msg_type = msg_type

        def Init(self, callback, queue_size) -> None:
            callback(_FakeLowState())

    class FakeChannelModule:
        @staticmethod
        def ChannelFactoryInitialize(domain, interface=None) -> None:
            return None

        ChannelSubscriber = FakeSubscriber

        class ChannelPublisher(_FakePublisher):
            def __init__(self, topic, msg_type) -> None:
                super().__init__()
                self.topic = topic
                self.msg_type = msg_type

            def Init(self) -> None:
                return None

    controller._import_channel_module = lambda: FakeChannelModule
    controller._import_message_module = lambda: {"LowState": object, "LowCmd": object, "LowCmdDefault": _FakeLowCmd}
    result = await controller.connect({
        "mode": "sim",
        "network_interface": "wlp68s0",
        "dds_domain": 1,
        "robot_variant": "g129_dex1",
        "motion_source": "lowcmd",
        "sim_runtime": "isaaclab",
    })
    assert result["ok"] is True
    assert result["command_topic"] == "rt/lowcmd"


@pytest.mark.asyncio
async def test_g1_connect_inspire_initializes_hand_channels() -> None:
    controller = UnitreeG1Controller(handshake_timeout_s=0.1)

    class FakeSubscriber:
        def __init__(self, topic, msg_type) -> None:
            self.topic = topic
            self.msg_type = msg_type

        def Init(self, callback, queue_size=None) -> None:
            if self.topic == "rt/lowstate":
                callback(_FakeLowState())
            elif callback is not None:
                callback(_FakeHandState())

    class FakeChannelModule:
        @staticmethod
        def ChannelFactoryInitialize(domain, interface=None) -> None:
            return None

        ChannelSubscriber = FakeSubscriber

        class ChannelPublisher(_FakePublisher):
            def __init__(self, topic, msg_type) -> None:
                super().__init__()
                self.topic = topic
                self.msg_type = msg_type

            def Init(self) -> None:
                return None

    controller._import_channel_module = lambda: FakeChannelModule
    controller._import_message_module = lambda: {"LowState": object, "LowCmd": object, "LowCmdDefault": _FakeLowCmd}
    controller._import_hand_message_module = lambda: {
        "MotorStates": object,
        "MotorCmds": _FakeHandCmd,
        "MotorCmdDefault": _FakeMotorCmd,
    }
    result = await controller.connect({
        "mode": "sim",
        "network_interface": "wlp68s0",
        "dds_domain": 1,
        "robot_variant": "g129_inspire",
        "motion_source": "lowcmd",
        "sim_runtime": "isaaclab",
    })
    assert result["robot_variant"] == "g129_inspire"
    assert result["hand_command_topic"] == "rt/inspire/cmd"
    assert controller._hand_publisher is not None


@pytest.mark.asyncio
async def test_g1_move_joint_continuous_publish() -> None:
    controller = UnitreeG1Controller()
    controller._connected = True
    controller._publisher = _FakePublisher()
    controller._last_lowstate = _FakeLowState()
    controller._import_message_module = lambda: {"LowCmdDefault": _FakeLowCmd, "LowCmd": _FakeLowCmd}
    controller._compute_crc = lambda _: 123
    result = await controller.move_joint(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        {"left_shoulder_pitch": 1.2},
        hold_seconds=0.1,
    )
    assert result["write_count"] == 5
    assert len(controller._publisher.messages) == 5
    assert controller._publisher.messages[-1].motor_cmd[15].q == 1.2
    assert controller._publisher.messages[-1].motor_cmd[22].q == 0.0


@pytest.mark.asyncio
async def test_g1_go_named_pose_translates_targets() -> None:
    controller = UnitreeG1Controller()
    controller._connected = True
    controller._publisher = _FakePublisher()
    controller._last_lowstate = _FakeLowState()
    controller._import_message_module = lambda: {"LowCmdDefault": _FakeLowCmd, "LowCmd": _FakeLowCmd}
    controller._compute_crc = lambda _: 123
    result = await controller.go_named_pose(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        "folded",
        hold_seconds=0.1,
    )
    assert result["named_pose"] == "folded"
    assert result["positions"]["left_elbow"] == 1.2
    assert result["positions"]["right_shoulder_roll"] == -0.6


@pytest.mark.asyncio
async def test_g1_hand_move_continuous_publish_and_denormalize() -> None:
    controller = UnitreeG1Controller()
    controller._connected = True
    controller._publisher = _FakePublisher()
    controller._hand_publisher = _FakePublisher()
    controller._import_hand_message_module = lambda: {
        "MotorCmds": _FakeHandCmd,
        "MotorCmdDefault": _FakeMotorCmd,
        "MotorStates": _FakeHandState,
    }
    result = await controller.hand_move(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
        },
        {"index": 0.2, "thumb_rotation": 0.15},
        side="left",
        hold_seconds=0.1,
    )
    assert result["write_count"] == 5
    assert len(controller._hand_publisher.messages) == 5
    msg = controller._hand_publisher.messages[-1]
    assert pytest.approx(msg.cmds[9].q, rel=1e-5) == 1.36
    assert pytest.approx(msg.cmds[11].q, rel=1e-5) == 1.09
    assert result["applied_positions"]["left_index"] == 0.2
    assert result["positions"]["right_index"] == 1.0


@pytest.mark.asyncio
async def test_g1_hand_preset_and_status() -> None:
    controller = UnitreeG1Controller()
    controller._connected = True
    controller._publisher = _FakePublisher()
    controller._hand_publisher = _FakePublisher()
    controller._last_hand_state = _FakeHandState(q=1.7)
    controller._import_hand_message_module = lambda: {
        "MotorCmds": _FakeHandCmd,
        "MotorCmdDefault": _FakeMotorCmd,
        "MotorStates": _FakeHandState,
    }
    result = await controller.hand_preset(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
        },
        "grasp",
        side="right",
        hold_seconds=0.1,
    )
    status = await controller.hand_status(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
        }
    )
    assert result["preset_name"] == "grasp"
    assert result["positions"]["right_index"] == 0.05
    assert result["positions"]["left_index"] == 1.0
    assert status["received_handstate"] is True
    assert status["hand_positions"]["right_index"] == 0.0
