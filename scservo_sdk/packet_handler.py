"""Factory wrapper matching the upstream SCServo API."""

from .protocol_packet_handler import protocol_packet_handler
from .scservo_def import SCS_SETEND


def PacketHandler(protocol_end):  # noqa: N802
    SCS_SETEND(protocol_end)
    return protocol_packet_handler()
