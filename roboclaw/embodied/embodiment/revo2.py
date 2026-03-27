"""Revo2 dexterous hand controller via bc_stark_sdk (Modbus RS-485)."""

from __future__ import annotations

from bc_stark_sdk import main_mod as libstark

libstark.init_logging()

_BAUDRATE = libstark.Baudrate.Baud460800
_SLAVE_LEFT = 0x7e
_SLAVE_RIGHT = 0x7f
_SLAVE_BY_TYPE = {"revo2_left": _SLAVE_LEFT, "revo2_right": _SLAVE_RIGHT}

_OPEN_POSITIONS = [0, 0, 0, 0, 0, 0]
_CLOSE_POSITIONS = [400, 0, 1000, 1000, 1000, 1000]
_SPEEDS = [1000] * 6


def _slave_id(hand_type: str | None) -> int:
    """Return slave ID for the given hand type. Defaults to left."""
    return _SLAVE_BY_TYPE.get(hand_type or "revo2_left", _SLAVE_LEFT)


async def _connect(port: str, slave_id: int) -> libstark.PyDeviceContext:
    """Open Modbus connection, verify device info, and set Normalized unit mode."""
    client: libstark.PyDeviceContext = await libstark.modbus_open(port, _BAUDRATE)
    if not client:
        raise RuntimeError(f"Failed to open serial port: {port}")
    info = await client.get_device_info(slave_id)
    if not info:
        libstark.modbus_close(client)
        raise RuntimeError(f"Failed to get device info. port={port} slave_id=0x{slave_id:02x}")
    await client.set_finger_unit_mode(slave_id, libstark.FingerUnitMode.Normalized)
    return client


class Revo2Controller:
    """Controls BrainCo Revo2 dexterous hand.

    Each method opens a connection, performs the operation, then closes it.
    Left hand slave_id=0x7E, right hand slave_id=0x7F.

    Finger position indices: [THUMB, THUMB_AUX, INDEX, MIDDLE, RING, PINKY], 0–1000.
    """

    async def open_hand(self, port: str, hand_type: str | None = None) -> str:
        """Extend all fingers to fully open position."""
        slave_id = _slave_id(hand_type)
        client = await _connect(port, slave_id)
        try:
            await client.set_finger_positions_and_speeds(slave_id, _OPEN_POSITIONS, _SPEEDS)
        finally:
            libstark.modbus_close(client)
        return "Hand opened."

    async def close_hand(self, port: str, hand_type: str | None = None) -> str:
        """Close all fingers to gripped position."""
        slave_id = _slave_id(hand_type)
        client = await _connect(port, slave_id)
        try:
            await client.set_finger_positions_and_speeds(slave_id, _CLOSE_POSITIONS, _SPEEDS)
        finally:
            libstark.modbus_close(client)
        return "Hand closed."

    async def set_pose(self, port: str, positions: list[int], hand_type: str | None = None) -> str:
        """Set individual finger positions (6 values: thumb/index/middle/ring/pinky/wrist, 0–1000)."""
        if len(positions) != 6:
            raise ValueError(f"Expected 6 finger positions, got {len(positions)}.")
        slave_id = _slave_id(hand_type)
        client = await _connect(port, slave_id)
        try:
            await client.set_finger_positions_and_speeds(slave_id, positions, _SPEEDS)
        finally:
            libstark.modbus_close(client)
        labels = ["thumb", "index", "middle", "ring", "pinky", "wrist"]
        summary = ", ".join(f"{l}={v}" for l, v in zip(labels, positions))
        return f"Pose set: {summary}."

    async def get_status(self, port: str, hand_type: str | None = None) -> str:
        """Read current finger positions, speeds, and currents."""
        slave_id = _slave_id(hand_type)
        client = await _connect(port, slave_id)
        try:
            status = await client.get_motor_status(slave_id)
        finally:
            libstark.modbus_close(client)
        labels = ["thumb", "index", "middle", "ring", "pinky", "wrist"]
        pos = {l: v for l, v in zip(labels, status.positions)}
        spd = {l: v for l, v in zip(labels, status.speeds)}
        cur = {l: v for l, v in zip(labels, status.currents)}
        return f"positions={pos}\nspeeds={spd}\ncurrents={cur}"
