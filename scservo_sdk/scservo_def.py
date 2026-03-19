"""Constants and helpers for the SCServo serial protocol."""

BROADCAST_ID = 0xFE
MAX_ID = 0xFC
SCS_END = 0

INST_PING = 1
INST_READ = 2
INST_WRITE = 3
INST_REG_WRITE = 4
INST_ACTION = 5
INST_SYNC_WRITE = 131
INST_SYNC_READ = 130

COMM_SUCCESS = 0
COMM_PORT_BUSY = -1
COMM_TX_FAIL = -2
COMM_RX_FAIL = -3
COMM_TX_ERROR = -4
COMM_RX_WAITING = -5
COMM_RX_TIMEOUT = -6
COMM_RX_CORRUPT = -7
COMM_NOT_AVAILABLE = -9


def SCS_GETEND() -> int:
    return SCS_END


def SCS_SETEND(value: int) -> None:
    global SCS_END
    SCS_END = value


def SCS_TOHOST(value: int, sign_bit: int) -> int:
    if value & (1 << sign_bit):
        return -(value & ~(1 << sign_bit))
    return value


def SCS_TOSCS(value: int, sign_bit: int) -> int:
    if value < 0:
        return (-value | (1 << sign_bit))
    return value


def SCS_MAKEWORD(low: int, high: int) -> int:
    if SCS_END == 0:
        return (low & 0xFF) | ((high & 0xFF) << 8)
    return (high & 0xFF) | ((low & 0xFF) << 8)


def SCS_MAKEDWORD(low: int, high: int) -> int:
    return (low & 0xFFFF) | ((high & 0xFFFF) << 16)


def SCS_LOWORD(value: int) -> int:
    return value & 0xFFFF


def SCS_HIWORD(value: int) -> int:
    return (value >> 16) & 0xFFFF


def SCS_LOBYTE(value: int) -> int:
    if SCS_END == 0:
        return value & 0xFF
    return (value >> 8) & 0xFF


def SCS_HIBYTE(value: int) -> int:
    if SCS_END == 0:
        return (value >> 8) & 0xFF
    return value & 0xFF
