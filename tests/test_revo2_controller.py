"""Integration tests for Revo2Controller — requires real hardware on /dev/ttyUSB0."""

import pytest

from roboclaw.embodied.embodiment.revo2 import Revo2Controller

PORT = "/dev/ttyUSB0"
HAND_TYPE = "revo2_left"

pytestmark = pytest.mark.hardware  # skip with: pytest -m "not hardware"


@pytest.mark.asyncio
async def test_get_status() -> None:
    result = await Revo2Controller().get_status(PORT, HAND_TYPE)
    assert "positions=" in result
    assert "speeds=" in result
    assert "currents=" in result


@pytest.mark.asyncio
async def test_open_hand() -> None:
    result = await Revo2Controller().open_hand(PORT, HAND_TYPE)
    assert result == "Hand opened."


@pytest.mark.asyncio
async def test_close_hand() -> None:
    result = await Revo2Controller().close_hand(PORT, HAND_TYPE)
    assert result == "Hand closed."


@pytest.mark.asyncio
async def test_set_pose() -> None:
    positions = [500, 500, 500, 500, 500, 500]
    result = await Revo2Controller().set_pose(PORT, positions, HAND_TYPE)
    assert "thumb=500" in result


@pytest.mark.asyncio
async def test_set_pose_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 6"):
        await Revo2Controller().set_pose(PORT, [0, 1, 2], HAND_TYPE)
