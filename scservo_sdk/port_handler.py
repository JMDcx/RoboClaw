"""Serial port wrapper compatible with the upstream SCServo API."""

from __future__ import annotations

import time

import serial

LATENCY_TIMER = 16
DEFAULT_BAUDRATE = 1_000_000


class PortHandler:
    """Compatibility wrapper around `pyserial`."""

    def __init__(self, port_name: str):
        self.is_open = False
        self.baudrate = DEFAULT_BAUDRATE
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0
        self.is_using = False
        self.port_name = port_name
        self.ser: serial.Serial | None = None

    def openPort(self) -> bool:  # noqa: N802
        return self.setBaudRate(self.baudrate)

    def closePort(self) -> None:  # noqa: N802
        if self.ser is not None:
            self.ser.close()
        self.is_open = False

    def clearPort(self) -> None:  # noqa: N802
        if self.ser is not None:
            self.ser.flush()

    def setPortName(self, port_name: str) -> None:  # noqa: N802
        self.port_name = port_name

    def getPortName(self) -> str:  # noqa: N802
        return self.port_name

    def setBaudRate(self, baudrate: int) -> bool:  # noqa: N802
        baud = self.getCFlagBaud(baudrate)
        if baud <= 0:
            return False
        self.baudrate = baudrate
        return self.setupPort(baud)

    def getBaudRate(self) -> int:  # noqa: N802
        return self.baudrate

    def getBytesAvailable(self) -> int:  # noqa: N802
        return 0 if self.ser is None else self.ser.in_waiting

    def readPort(self, length: int):  # noqa: N802
        if self.ser is None:
            return b""
        return self.ser.read(length)

    def writePort(self, packet):  # noqa: N802
        if self.ser is None:
            return 0
        return self.ser.write(packet)

    def setPacketTimeout(self, packet_length: int) -> None:  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = (self.tx_time_per_byte * packet_length) + (LATENCY_TIMER * 2.0) + 2.0

    def setPacketTimeoutMillis(self, msec: float) -> None:  # noqa: N802
        self.packet_start_time = self.getCurrentTime()
        self.packet_timeout = msec

    def isPacketTimeout(self) -> bool:  # noqa: N802
        if self.getTimeSinceStart() > self.packet_timeout:
            self.packet_timeout = 0
            return True
        return False

    def getCurrentTime(self) -> float:  # noqa: N802
        return round(time.time() * 1_000_000_000) / 1_000_000.0

    def getTimeSinceStart(self) -> float:  # noqa: N802
        time_since = self.getCurrentTime() - self.packet_start_time
        if time_since < 0.0:
            self.packet_start_time = self.getCurrentTime()
        return time_since

    def setupPort(self, cflag_baud: int) -> bool:  # noqa: N802
        del cflag_baud
        if self.is_open:
            self.closePort()
        self.ser = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            timeout=0,
        )
        self.is_open = True
        self.ser.reset_input_buffer()
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def getCFlagBaud(self, baudrate: int) -> int:  # noqa: N802
        if baudrate in [4800, 9600, 14400, 19200, 38400, 57600, 115200, 128000, 250000, 500000, 1000000]:
            return baudrate
        return -1
