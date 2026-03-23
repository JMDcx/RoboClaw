"""Tests for the EmbodiedTool integration with the agent."""

from unittest.mock import AsyncMock, patch

import pytest

from roboclaw.embodied.tool import EmbodiedTool


def test_tool_schema() -> None:
    tool = EmbodiedTool()
    assert tool.name == "embodied"
    assert "robot" in tool.description.lower()

    params = tool.parameters
    assert params["type"] == "object"
    assert "action" in params["properties"]
    assert params["required"] == ["action"]

    action_schema = params["properties"]["action"]
    assert action_schema["type"] == "string"
    expected_actions = [
        "doctor", "calibrate", "teleoperate", "record",
        "train", "run_policy", "job_status",
    ]
    assert action_schema["enum"] == expected_actions

    for key in ("port", "calibration_dir", "dataset_name", "task",
                "num_episodes", "fps", "steps", "checkpoint_path",
                "job_id", "device"):
        assert key in params["properties"], f"Missing optional param: {key}"


@pytest.mark.asyncio
async def test_doctor_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (0, "lerobot 0.5.0", "")

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(action="doctor")

    assert "lerobot 0.5.0" in result


@pytest.mark.asyncio
async def test_record_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (0, "Recorded 5 episodes", "")

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(
            action="record",
            dataset_name="my_data",
            task="grasp block",
            num_episodes=5,
            fps=15,
        )

    assert result == "Recorded 5 episodes"
    argv = mock_runner.run.call_args[0][0]
    assert "lerobot-record" in argv
    assert "--robot.type=so101" in argv


@pytest.mark.asyncio
async def test_train_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run_detached.return_value = "job-abc-123"

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(
            action="train",
            dataset_name="my_data",
            steps=5000,
        )

    assert "job-abc-123" in result
    assert "Training started" in result


@pytest.mark.asyncio
async def test_job_status_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.job_status.return_value = {"status": "running", "step": 2500}

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(action="job_status", job_id="job-abc-123")

    assert "running" in result
    assert "2500" in result


@pytest.mark.asyncio
async def test_command_failure_returns_error() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()
    mock_runner.run.return_value = (1, "", "lerobot not found")

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(action="doctor")

    assert "Command failed" in result
    assert "lerobot not found" in result


@pytest.mark.asyncio
async def test_unknown_action() -> None:
    tool = EmbodiedTool()
    mock_runner = AsyncMock()

    with patch("roboclaw.embodied.runner.LocalLeRobotRunner", return_value=mock_runner):
        result = await tool.execute(action="fly_to_moon")

    assert "Unknown action" in result
