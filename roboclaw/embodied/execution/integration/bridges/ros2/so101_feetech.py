"""Minimal SO101 Feetech runtime used by the ROS2 control bridge."""

from __future__ import annotations

import json
import importlib.util
import os
import time
from pathlib import Path
from typing import Any

from roboclaw.config.paths import ensure_robot_calibration_file, resolve_active_serial_device_path
from roboclaw.embodied.execution.integration.bridges.ros2.scservo import ServoCalibration

PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 1_000_000
DEFAULT_SERVO_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
ADDR_MAX_TORQUE_LIMIT = 16
ADDR_P_COEFFICIENT = 21
ADDR_D_COEFFICIENT = 22
ADDR_I_COEFFICIENT = 23
ADDR_PROTECTION_CURRENT = 28
ADDR_HOMING_OFFSET = 31
ADDR_OPERATING_MODE = 33
ADDR_OVERLOAD_TORQUE = 36
ADDR_TORQUE_ENABLE = 40
ADDR_ACCELERATION = 41
ADDR_GOAL_POSITION = 42
ADDR_MIN_POSITION_LIMIT = 9
ADDR_MAX_POSITION_LIMIT = 11
ADDR_LOCK = 55
ADDR_PRESENT_POSITION = 56
POSITION_MODE = 0


def _patch_packet_timeout(port_handler: Any, scs: Any) -> None:
    def set_packet_timeout(self: Any, packet_length: int) -> None:
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (self.tx_time_per_byte * packet_length) + (self.tx_time_per_byte * 3.0) + 50

    port_handler.setPacketTimeout = set_packet_timeout.__get__(port_handler, scs.PortHandler)  # type: ignore[attr-defined]


class So101FeetechRuntime:
    """Direct SO101 runtime used by the control bridge."""

    def __init__(
        self,
        *,
        device_by_id: str,
        robot_name: str = "so101",
        calibration_path: str | None = None,
        calibration_id: str = "so101_real",
        baudrate: int = DEFAULT_BAUDRATE,
    ) -> None:
        self.device_by_id = device_by_id
        self.robot_name = robot_name
        self.calibration_path = self._resolve_calibration_path(robot_name, calibration_path, calibration_id)
        self.baudrate = baudrate
        self._calibration = self._load_calibration(self.calibration_path)
        self._port_handler: Any | None = None
        self._packet_handler: Any | None = None
        self._scs: Any | None = None
        self._resolved_device_path: Path | None = None

    @property
    def connected(self) -> bool:
        return self._port_handler is not None and self._packet_handler is not None

    def connect(self) -> None:
        if self.connected:
            return

        self._ensure_scservo_sdk()
        import scservo_sdk as scs

        self._resolved_device_path = resolve_active_serial_device_path(self.device_by_id)
        port_handler = scs.PortHandler(str(self._resolved_device_path))
        _patch_packet_timeout(port_handler, scs)
        if not port_handler.openPort():
            raise RuntimeError(f"Failed to open servo device '{self._resolved_device_path}'.")
        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            raise RuntimeError(f"Failed to set baudrate {self.baudrate} on '{self._resolved_device_path}'.")

        self._port_handler = port_handler
        self._packet_handler = scs.PacketHandler(PROTOCOL_VERSION)
        self._scs = scs
        self._configure_servos()

    def disconnect(self) -> None:
        if self._port_handler is None:
            return
        try:
            self._write1(self._gripper_id, ADDR_TORQUE_ENABLE, 0)
            self._write1(self._gripper_id, ADDR_LOCK, 0)
        finally:
            self._port_handler.closePort()
            self._port_handler = None
            self._packet_handler = None
            self._scs = None
            self._resolved_device_path = None

    def open_gripper(self) -> dict[str, Any]:
        target = self._normalized_to_raw("gripper", 100.0)
        present = self._move_servo(self._gripper_id, target)
        return {"target_raw": target, "present_raw": present, "gripper_percent": self.gripper_percent()}

    def close_gripper(self) -> dict[str, Any]:
        target = self._normalized_to_raw("gripper", 0.0)
        present = self._move_servo(self._gripper_id, target)
        return {"target_raw": target, "present_raw": present, "gripper_percent": self.gripper_percent()}

    def go_home(self) -> dict[str, Any]:
        moved: dict[str, int] = {}
        for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
            calibration = self._calibration.get(joint)
            if calibration is None:
                continue
            midpoint = int((calibration.range_min + calibration.range_max) / 2)
            moved[joint] = self._move_servo(calibration.id, midpoint)
        moved["gripper"] = self._move_servo(self._gripper_id, self._normalized_to_raw("gripper", 100.0))
        return {"home_targets": moved, "gripper_percent": self.gripper_percent()}

    def gripper_percent(self) -> float | None:
        raw = self.read_gripper_position()
        if raw is None:
            return None
        calibration = self._calibration["gripper"]
        return round(calibration.raw_to_normalized(raw), 2)

    def read_gripper_position(self) -> int | None:
        if not self.connected:
            return None
        return self._read2(self._gripper_id, ADDR_PRESENT_POSITION)

    def snapshot(self) -> dict[str, Any]:
        return {
            "device_by_id": self.device_by_id,
            "resolved_device": str(self._resolved_device_path) if self._resolved_device_path is not None else None,
            "calibration_path": str(self.calibration_path),
            "connected": self.connected,
            "gripper_servo_id": self._gripper_id,
            "gripper_present_raw": self.read_gripper_position(),
            "gripper_percent": self.gripper_percent(),
        }

    @property
    def _gripper_id(self) -> int:
        return self._calibration["gripper"].id

    def _configure_gripper(self) -> None:
        self._write1(self._gripper_id, ADDR_TORQUE_ENABLE, 0)
        self._write1(self._gripper_id, ADDR_LOCK, 0)
        self._apply_calibration("gripper")
        self._write1(self._gripper_id, ADDR_OPERATING_MODE, POSITION_MODE)
        self._write1(self._gripper_id, ADDR_P_COEFFICIENT, 16)
        self._write1(self._gripper_id, ADDR_I_COEFFICIENT, 0)
        self._write1(self._gripper_id, ADDR_D_COEFFICIENT, 32)
        self._write2(self._gripper_id, ADDR_MAX_TORQUE_LIMIT, 500)
        self._write2(self._gripper_id, ADDR_PROTECTION_CURRENT, 250)
        self._write1(self._gripper_id, ADDR_OVERLOAD_TORQUE, 25)
        self._write1(self._gripper_id, ADDR_ACCELERATION, 254)
        self._write1(self._gripper_id, ADDR_TORQUE_ENABLE, 1)
        self._write1(self._gripper_id, ADDR_LOCK, 1)

    def _configure_position_servo(self, joint_name: str) -> None:
        calibration = self._calibration[joint_name]
        servo_id = calibration.id
        self._write1(servo_id, ADDR_TORQUE_ENABLE, 0)
        self._write1(servo_id, ADDR_LOCK, 0)
        self._apply_calibration(joint_name)
        self._write1(servo_id, ADDR_OPERATING_MODE, POSITION_MODE)
        self._write1(servo_id, ADDR_P_COEFFICIENT, 16)
        self._write1(servo_id, ADDR_I_COEFFICIENT, 0)
        self._write1(servo_id, ADDR_D_COEFFICIENT, 32)
        self._write1(servo_id, ADDR_TORQUE_ENABLE, 1)
        self._write1(servo_id, ADDR_LOCK, 1)

    def _configure_servos(self) -> None:
        for joint_name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"):
            if joint_name in self._calibration:
                self._configure_position_servo(joint_name)
        self._configure_gripper()

    def _apply_calibration(self, joint_name: str) -> None:
        calibration = self._calibration[joint_name]
        servo_id = calibration.id
        self._write2(servo_id, ADDR_HOMING_OFFSET, self._encode_signed_16(calibration.homing_offset))
        self._write2(servo_id, ADDR_MIN_POSITION_LIMIT, calibration.range_min)
        self._write2(servo_id, ADDR_MAX_POSITION_LIMIT, calibration.range_max)

    def _move_servo(self, servo_id: int, target_raw: int) -> int:
        self._write2(servo_id, ADDR_GOAL_POSITION, target_raw)
        deadline = time.monotonic() + 2.0
        last_value = self._read2(servo_id, ADDR_PRESENT_POSITION)
        while time.monotonic() < deadline:
            if abs(last_value - target_raw) <= 40:
                return last_value
            time.sleep(0.1)
            current = self._read2(servo_id, ADDR_PRESENT_POSITION)
            if abs(current - last_value) <= 5 and abs(current - target_raw) <= 120:
                return current
            last_value = current
        return last_value

    def _normalized_to_raw(self, joint_name: str, value: float) -> int:
        calibration = self._calibration[joint_name]
        return calibration.normalized_to_raw(value)

    def _read2(self, servo_id: int, address: int) -> int:
        if self._packet_handler is None or self._port_handler is None:
            raise RuntimeError("Servo runtime is not connected.")
        value, result, error = self._packet_handler.read2ByteTxRx(self._port_handler, servo_id, address)
        self._raise_if_comm_failed(result, error, f"read2({servo_id}, {address})")
        return int(value)

    def _write1(self, servo_id: int, address: int, data: int) -> None:
        if self._packet_handler is None or self._port_handler is None:
            raise RuntimeError("Servo runtime is not connected.")
        result, error = self._packet_handler.write1ByteTxRx(self._port_handler, servo_id, address, int(data))
        self._raise_if_comm_failed(result, error, f"write1({servo_id}, {address})")

    def _write2(self, servo_id: int, address: int, data: int) -> None:
        if self._packet_handler is None or self._port_handler is None:
            raise RuntimeError("Servo runtime is not connected.")
        result, error = self._packet_handler.write2ByteTxRx(self._port_handler, servo_id, address, int(data))
        self._raise_if_comm_failed(result, error, f"write2({servo_id}, {address})")

    def _raise_if_comm_failed(self, result: int, error: int, operation: str) -> None:
        if self._packet_handler is None or self._scs is None:
            raise RuntimeError("Servo runtime is not connected.")
        if result != self._scs.COMM_SUCCESS:
            raise RuntimeError(f"{operation} failed: {self._packet_handler.getTxRxResult(result)}")
        if error != 0:
            raise RuntimeError(f"{operation} returned servo error: {self._packet_handler.getRxPacketError(error)}")

    @staticmethod
    def _resolve_calibration_path(robot_name: str, path: str | None, calibration_id: str) -> Path:
        if path:
            resolved = Path(path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Calibration file '{resolved}' does not exist.")
            return resolved

        canonical_path = ensure_robot_calibration_file(robot_name, calibration_id)
        if canonical_path.exists():
            return canonical_path.resolve()
        raise FileNotFoundError(
            "Could not auto-discover an SO101 calibration file. "
            f"Expected '{canonical_path}' or a legacy calibration import that could be migrated into it."
        )

    @staticmethod
    def _load_calibration(path: Path) -> dict[str, ServoCalibration]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, ServoCalibration] = {}
        for joint_name, default_id in DEFAULT_SERVO_IDS.items():
            raw = payload.get(joint_name)
            if raw is None:
                continue
            result[joint_name] = ServoCalibration(
                id=int(raw.get("id", default_id)),
                drive_mode=int(raw.get("drive_mode", 0)),
                homing_offset=int(raw.get("homing_offset", 0)),
                range_min=int(raw.get("range_min", 0)),
                range_max=int(raw.get("range_max", 4095)),
            )
        if "gripper" not in result:
            raise ValueError(f"Calibration file '{path}' does not declare a gripper servo.")
        return result

    @staticmethod
    def _encode_signed_16(value: int) -> int:
        return int(value) & 0xFFFF

    @staticmethod
    def _ensure_scservo_sdk() -> None:
        if importlib.util.find_spec("scservo_sdk") is not None:
            return
        raise ModuleNotFoundError(
            "scservo_sdk is unavailable. Install the RoboClaw SO101 Python dependency bundle before launching the ROS2 control bridge."
        )


def build_so101_runtime_from_env() -> So101FeetechRuntime:
    """Construct a runtime from env vars for ad-hoc diagnostics."""

    return So101FeetechRuntime(
        device_by_id=os.environ.get("ROBOCLAW_SO101_DEVICE_BY_ID", "/dev/serial/by-id/unknown"),
        robot_name=os.environ.get("ROBOCLAW_SO101_ROBOT_NAME", "so101"),
        calibration_path=os.environ.get("ROBOCLAW_SO101_CALIBRATION"),
        calibration_id=os.environ.get("ROBOCLAW_SO101_CALIBRATION_ID", "so101_real"),
    )


__all__ = ["So101FeetechRuntime", "build_so101_runtime_from_env"]
