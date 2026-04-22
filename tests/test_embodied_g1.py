"""Tests for minimal Unitree G1 embodied support."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.embodiment.g1_dex1_grasp import G1Dex1GraspRunner
from roboclaw.embodied.embodiment.g1_grasp_test import G1InspireGraspTestRunner
from roboclaw.embodied.embodiment.sim_feedback import SimVerification
from roboclaw.embodied.perception.schemas import DetectedObject, PerceptionFrame
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
    assert "g1_gripper_status" in actions
    assert "g1_gripper_move" in actions
    assert "g1_gripper_open" in actions
    assert "g1_gripper_close" in actions
    assert "g1_hand_status" in actions
    assert "g1_hand_move" in actions
    assert "g1_hand_preset" in actions
    assert "g1_inspire_grasp_test" in actions


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
        gripper_status=AsyncMock(return_value={"ok": True, "received_gripper_state": True}),
        gripper_move=AsyncMock(return_value={"ok": True, "close_amount": 0.7}),
        gripper_open=AsyncMock(return_value={"ok": True, "preset_name": "open"}),
        gripper_close=AsyncMock(return_value={"ok": True, "preset_name": "close"}),
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
        gripper_status_result = await tool.execute(action="g1_gripper_status")
        gripper_move_result = await tool.execute(action="g1_gripper_move", close_amount=0.7)
        gripper_open_result = await tool.execute(action="g1_gripper_open")
        gripper_close_result = await tool.execute(action="g1_gripper_close")
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
    assert "received_gripper_state" in gripper_status_result
    assert "0.7" in gripper_move_result
    assert "open" in gripper_open_result
    assert "close" in gripper_close_result
    assert "received_handstate" in hand_status_result
    assert "left_index" in hand_move_result
    assert "grasp" in hand_preset_result


class _StubPerception:
    def analyze_scene(self, camera_name: str = "head") -> PerceptionFrame:
        return PerceptionFrame(
            frame_id="head_100",
            timestamp_ms=100,
            camera_name=camera_name,
            image_path="/tmp/frame.png",
            depth_path="/tmp/frame_depth.npy",
            overlay_path=None,
            has_depth=True,
            camera_intrinsics={"fx": 243.2, "fy": 182.4, "cx": 319.5, "cy": 239.5, "width": 640, "height": 480},
            objects=[
                DetectedObject(
                    object_id="object_1",
                    raw_class_name="cylinder",
                    task_label="object",
                    track_id=1,
                    class_name="cylinder",
                    confidence=0.97,
                    bbox_xyxy=[10.0, 10.0, 40.0, 80.0],
                    mask_rle=None,
                    center_xy=[25.0, 45.0],
                    stable=True,
                    age=4,
                    visibility=1.0,
                    pickable=True,
                    container_candidate=False,
                    attributes={
                        "centroid_3d": [0.10, 0.55, 0.28],
                        "extent_3d": [0.04, 0.04, 0.08],
                        "principal_axis_3d": [1.0, 0.0, 0.0],
                    },
                )
            ],
        )


class _StubFeedback:
    def __init__(self) -> None:
        self._calls = 0

    def verify_object_lift(self, *, min_height_m: float = 0.5) -> SimVerification:
        self._calls += 1
        if self._calls == 1:
            return SimVerification(
                object_height_before=0.42,
                object_height_after=0.42,
                height_delta=0.0,
                reward_snapshot=[0.0],
                success_boolean=False,
            )
        return SimVerification(
            object_height_before=0.42,
            object_height_after=0.62,
            height_delta=0.20,
            reward_snapshot=[0.0],
            success_boolean=True,
        )


@pytest.mark.asyncio
async def test_g1_inspire_grasp_test_runner_returns_structured_success(tmp_path: Path) -> None:
    controller = SimpleNamespace(
        status=AsyncMock(
            return_value={
                "joint_positions": {
                    "right_shoulder_pitch": 0.0,
                    "right_shoulder_roll": -0.8,
                    "right_shoulder_yaw": 0.0,
                    "right_elbow": 1.2,
                    "right_wrist_roll": 0.0,
                    "right_wrist_pitch": -0.4,
                    "right_wrist_yaw": 0.0,
                }
            }
        ),
        move_joint=AsyncMock(return_value={"ok": True}),
        hand_preset=AsyncMock(return_value={"ok": True}),
    )
    runner = G1InspireGraspTestRunner(
        workspace=tmp_path,
        controller=controller,
        perception=_StubPerception(),
        feedback=_StubFeedback(),
    )

    payload = await runner.run(
        config={
            "enabled": True,
            "mode": "sim",
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
        },
        arm_side="right",
    )

    assert payload["status"] == "success"
    assert payload["target_object_id"] == "object_1"
    assert payload["verification"]["success_boolean"] is True
    assert "right_wrist_pitch" in payload["pregrasp_joint_positions"]
    assert controller.move_joint.await_count >= 3
    assert controller.hand_preset.await_count == 2


@pytest.mark.asyncio
async def test_g1_dex1_grasp_runner_returns_structured_success(tmp_path: Path) -> None:
    controller = SimpleNamespace(
        status=AsyncMock(
            return_value={
                "connected": True,
                "joint_positions": {
                    "right_shoulder_pitch": 0.0,
                    "right_shoulder_roll": -0.8,
                    "right_shoulder_yaw": 0.0,
                    "right_elbow": 1.2,
                    "right_wrist_roll": 0.0,
                    "right_wrist_pitch": -0.4,
                    "right_wrist_yaw": 0.0,
                },
            }
        ),
        move_joint=AsyncMock(return_value={"ok": True}),
        gripper_open=AsyncMock(return_value={"ok": True, "preset_name": "open"}),
        gripper_close=AsyncMock(return_value={"ok": True, "preset_name": "close"}),
    )
    runner = G1Dex1GraspRunner(
        workspace=tmp_path,
        controller=controller,
        feedback=_StubFeedback(),
    )
    runner._solver = SimpleNamespace(
        solve=lambda request: SimpleNamespace(
            joint_positions={
                "right_shoulder_pitch": 0.1,
                "right_shoulder_roll": -0.7,
                "right_shoulder_yaw": 0.0,
                "right_elbow": 1.1,
                "right_wrist_roll": 0.0,
                "right_wrist_pitch": -0.3,
                "right_wrist_yaw": 0.0,
            }
        )
    )

    payload = await runner.run(
        config={
            "enabled": True,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        frame=_StubPerception().analyze_scene("head"),
        target_object_id="object_1",
        selected_grasp={
            "candidate_id": "candidate_1",
            "grasp_pose_cam": {"translation": [0.10, 0.55, 0.24]},
            "approach_vector_cam": [0.0, 0.0, 1.0],
            "pregrasp_offset_m": 0.08,
            "lift_height_m": 0.10,
        },
        arm_side="right",
    )

    assert payload["status"] == "success"
    assert payload["target_object_id"] == "object_1"
    assert payload["verification"]["success_boolean"] is True
    assert controller.move_joint.await_count >= 3
    assert controller.gripper_open.await_count == 1
    assert controller.gripper_close.await_count == 1


@pytest.mark.asyncio
async def test_tool_g1_inspire_grasp_test_uses_runner(tmp_path: Path) -> None:
    tool = EmbodiedTool()
    tool._g1_controller = SimpleNamespace()
    setup = {
        "unitree_g1": {
            "enabled": True,
            "connected": True,
            "mode": "sim",
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "robot_variant": "g129_inspire",
            "motion_source": "lowcmd",
            "sim_runtime": "isaaclab",
        }
    }
    fake_runner = AsyncMock()
    fake_runner.run.return_value = {
        "status": "success",
        "target_object_id": "object_1",
        "verification": {"success_boolean": True},
    }

    with (
        patch("roboclaw.embodied.setup.ensure_setup", return_value=setup),
        patch("roboclaw.embodied.embodiment.g1_grasp_test.G1InspireGraspTestRunner", return_value=fake_runner),
    ):
        result = await tool.execute(action="g1_inspire_grasp_test", arm_side="right")

    payload = json.loads(result)
    assert payload["status"] == "success"
    assert payload["target_object_id"] == "object_1"


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
async def test_g1_connect_dex1_initializes_gripper_channels() -> None:
    controller = UnitreeG1Controller(handshake_timeout_s=0.1)

    class FakeSubscriber:
        def __init__(self, topic, msg_type) -> None:
            self.topic = topic
            self.msg_type = msg_type

        def Init(self, callback, queue_size=None) -> None:
            if self.topic == "rt/lowstate":
                callback(_FakeLowState())
            elif callback is not None:
                callback(_FakeHandState(count=1, q=2.7))

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
    controller._import_gripper_message_module = lambda: {
        "MotorStates": object,
        "MotorCmds": _FakeHandCmd,
        "MotorCmdDefault": _FakeMotorCmd,
    }
    result = await controller.connect({
        "mode": "sim",
        "network_interface": "wlp68s0",
        "dds_domain": 1,
        "robot_variant": "g129_dex1",
        "motion_source": "lowcmd",
        "sim_runtime": "isaaclab",
    })
    assert result["robot_variant"] == "g129_dex1"
    assert result["gripper_command_topic"] == "rt/dex1/right/cmd"
    assert result["gripper_state_topic"] == "rt/dex1/right/state"
    assert controller._gripper_publisher is not None


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


@pytest.mark.asyncio
async def test_g1_gripper_move_open_close_and_status() -> None:
    controller = UnitreeG1Controller()
    controller._connected = True
    controller._publisher = _FakePublisher()
    controller._gripper_publisher = _FakePublisher()
    controller._last_gripper_state = _FakeHandState(count=1, q=2.7)
    controller._import_gripper_message_module = lambda: {
        "MotorCmds": _FakeHandCmd,
        "MotorCmdDefault": _FakeMotorCmd,
        "MotorStates": _FakeHandState,
    }
    move_result = await controller.gripper_move(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        0.75,
        side="right",
        hold_seconds=0.1,
    )
    open_result = await controller.gripper_open(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        hold_seconds=0.1,
    )
    close_result = await controller.gripper_close(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        },
        hold_seconds=0.1,
    )
    status = await controller.gripper_status(
        {
            "network_interface": "wlp68s0",
            "dds_domain": 1,
            "mode": "sim",
            "robot_variant": "g129_dex1",
            "motion_source": "lowcmd",
        }
    )
    assert move_result["write_count"] == 5
    assert pytest.approx(controller._gripper_publisher.messages[0].cmds[0].q, rel=1e-5) == 1.35
    assert open_result["preset_name"] == "open"
    assert close_result["preset_name"] == "close"
    assert status["received_gripper_state"] is True
    assert pytest.approx(status["gripper_close_amounts"]["right"], rel=1e-5) == 0.5
