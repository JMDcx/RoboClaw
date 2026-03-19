"""Subset of the upstream SCServo protocol handler used by RoboClaw."""

from __future__ import annotations

from .scservo_def import (
    BROADCAST_ID,
    COMM_NOT_AVAILABLE,
    COMM_PORT_BUSY,
    COMM_RX_CORRUPT,
    COMM_RX_TIMEOUT,
    COMM_RX_WAITING,
    COMM_SUCCESS,
    COMM_TX_ERROR,
    COMM_TX_FAIL,
)
from .scservo_def import (
    INST_READ,
    INST_WRITE,
    SCS_HIBYTE,
    SCS_LOBYTE,
    SCS_MAKEWORD,
)

TXPACKET_MAX_LEN = 250
RXPACKET_MAX_LEN = 250

PKT_HEADER0 = 0
PKT_HEADER1 = 1
PKT_ID = 2
PKT_LENGTH = 3
PKT_INSTRUCTION = 4
PKT_ERROR = 4
PKT_PARAMETER0 = 5

ERRBIT_VOLTAGE = 1
ERRBIT_ANGLE = 2
ERRBIT_OVERHEAT = 4
ERRBIT_OVERELE = 8
ERRBIT_OVERLOAD = 32


class protocol_packet_handler:
    """Protocol 1.0 packet encoder/decoder for SCServo devices."""

    def getProtocolVersion(self):  # noqa: N802
        return 1.0

    def getTxRxResult(self, result):  # noqa: N802
        return {
            COMM_SUCCESS: "[TxRxResult] Communication success!",
            COMM_PORT_BUSY: "[TxRxResult] Port is in use!",
            COMM_TX_FAIL: "[TxRxResult] Failed transmit instruction packet!",
            COMM_RX_FAIL: "[TxRxResult] Failed get status packet from device!",
            COMM_TX_ERROR: "[TxRxResult] Incorrect instruction packet!",
            COMM_RX_WAITING: "[TxRxResult] Now receiving status packet!",
            COMM_RX_TIMEOUT: "[TxRxResult] There is no status packet!",
            COMM_RX_CORRUPT: "[TxRxResult] Incorrect status packet!",
            COMM_NOT_AVAILABLE: "[TxRxResult] Protocol does not support this function!",
        }.get(result, "")

    def getRxPacketError(self, error):  # noqa: N802
        if error & ERRBIT_VOLTAGE:
            return "[RxPacketError] Input voltage error!"
        if error & ERRBIT_ANGLE:
            return "[RxPacketError] Angle sen error!"
        if error & ERRBIT_OVERHEAT:
            return "[RxPacketError] Overheat error!"
        if error & ERRBIT_OVERELE:
            return "[RxPacketError] OverEle error!"
        if error & ERRBIT_OVERLOAD:
            return "[RxPacketError] Overload error!"
        return ""

    def txPacket(self, port, txpacket):  # noqa: N802
        checksum = 0
        total_packet_length = txpacket[PKT_LENGTH] + 4
        if port.is_using:
            return COMM_PORT_BUSY
        port.is_using = True
        if total_packet_length > TXPACKET_MAX_LEN:
            port.is_using = False
            return COMM_TX_ERROR
        txpacket[PKT_HEADER0] = 0xFF
        txpacket[PKT_HEADER1] = 0xFF
        for idx in range(2, total_packet_length - 1):
            checksum += txpacket[idx]
        txpacket[total_packet_length - 1] = ~checksum & 0xFF
        port.clearPort()
        if total_packet_length != port.writePort(txpacket):
            port.is_using = False
            return COMM_TX_FAIL
        return COMM_SUCCESS

    def rxPacket(self, port):  # noqa: N802
        rxpacket = []
        result = COMM_TX_FAIL
        checksum = 0
        rx_length = 0
        wait_length = 6
        while True:
            rxpacket.extend(port.readPort(wait_length - rx_length))
            rx_length = len(rxpacket)
            if rx_length >= wait_length:
                idx = 0
                for idx in range(0, rx_length - 1):
                    if rxpacket[idx] == 0xFF and rxpacket[idx + 1] == 0xFF:
                        break
                if idx == 0:
                    if rxpacket[PKT_ID] > 0xFD or rxpacket[PKT_LENGTH] > RXPACKET_MAX_LEN or rxpacket[PKT_ERROR] > 0x7F:
                        del rxpacket[0]
                        rx_length -= 1
                        continue
                    if wait_length != (rxpacket[PKT_LENGTH] + PKT_LENGTH + 1):
                        wait_length = rxpacket[PKT_LENGTH] + PKT_LENGTH + 1
                        continue
                    if rx_length < wait_length:
                        if port.isPacketTimeout():
                            result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                            break
                        continue
                    for i in range(2, wait_length - 1):
                        checksum += rxpacket[i]
                    checksum = ~checksum & 0xFF
                    result = COMM_SUCCESS if rxpacket[wait_length - 1] == checksum else COMM_RX_CORRUPT
                    break
                del rxpacket[0:idx]
                rx_length -= idx
            else:
                if port.isPacketTimeout():
                    result = COMM_RX_TIMEOUT if rx_length == 0 else COMM_RX_CORRUPT
                    break
        port.is_using = False
        return rxpacket, result

    def txRxPacket(self, port, txpacket):  # noqa: N802
        rxpacket = None
        error = 0
        result = self.txPacket(port, txpacket)
        if result != COMM_SUCCESS:
            return rxpacket, result, error
        if txpacket[PKT_ID] == BROADCAST_ID:
            port.is_using = False
            return rxpacket, result, error
        if txpacket[PKT_INSTRUCTION] == INST_READ:
            port.setPacketTimeout(txpacket[PKT_PARAMETER0 + 1] + 6)
        else:
            port.setPacketTimeout(6)
        while True:
            rxpacket, result = self.rxPacket(port)
            if result != COMM_SUCCESS or txpacket[PKT_ID] == rxpacket[PKT_ID]:
                break
        if result == COMM_SUCCESS and txpacket[PKT_ID] == rxpacket[PKT_ID]:
            error = rxpacket[PKT_ERROR]
        return rxpacket, result, error

    def readTxRx(self, port, scs_id, address, length):  # noqa: N802
        txpacket = [0] * 8
        data = []
        if scs_id >= BROADCAST_ID:
            return data, COMM_NOT_AVAILABLE, 0
        txpacket[PKT_ID] = scs_id
        txpacket[PKT_LENGTH] = 4
        txpacket[PKT_INSTRUCTION] = INST_READ
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1] = length
        rxpacket, result, error = self.txRxPacket(port, txpacket)
        if result == COMM_SUCCESS:
            error = rxpacket[PKT_ERROR]
            data.extend(rxpacket[PKT_PARAMETER0 : PKT_PARAMETER0 + length])
        return data, result, error

    def read2ByteTxRx(self, port, scs_id, address):  # noqa: N802
        data, result, error = self.readTxRx(port, scs_id, address, 2)
        data_read = SCS_MAKEWORD(data[0], data[1]) if result == COMM_SUCCESS else 0
        return data_read, result, error

    def writeTxRx(self, port, scs_id, address, length, data):  # noqa: N802
        txpacket = [0] * (length + 7)
        txpacket[PKT_ID] = scs_id
        txpacket[PKT_LENGTH] = length + 3
        txpacket[PKT_INSTRUCTION] = INST_WRITE
        txpacket[PKT_PARAMETER0] = address
        txpacket[PKT_PARAMETER0 + 1 : PKT_PARAMETER0 + 1 + length] = data[0:length]
        _, result, error = self.txRxPacket(port, txpacket)
        return result, error

    def write1ByteTxRx(self, port, scs_id, address, data):  # noqa: N802
        return self.writeTxRx(port, scs_id, address, 1, [data])

    def write2ByteTxRx(self, port, scs_id, address, data):  # noqa: N802
        return self.writeTxRx(port, scs_id, address, 2, [SCS_LOBYTE(data), SCS_HIBYTE(data)])
