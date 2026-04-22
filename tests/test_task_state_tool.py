import json
from pathlib import Path

import pytest

from roboclaw.agent.tools.task_state import TaskStateTool


@pytest.mark.asyncio
async def test_task_state_tool_round_trip_planning(tmp_path: Path) -> None:
    tool = TaskStateTool(workspace=tmp_path)
    tool.set_context("cli", "session-1")

    reset_payload = json.loads(await tool.execute(action="reset_state"))
    assert reset_payload["current_phase"] == "idle"

    goal_payload = json.loads(
        await tool.execute(action="set_goal", user_goal="把桌上的东西放进右边篮子")
    )
    assert goal_payload["user_goal"] == "把桌上的东西放进右边篮子"
    assert goal_payload["current_phase"] == "need_scene"

    perception_payload = {
        "frame_id": "head_1",
        "objects": [
            {
                "object_id": "object_1",
                "raw_class_name": "cup",
                "task_label": "object",
                "confidence": 0.91,
                "bbox_xyxy": [1, 2, 3, 4],
                "center_xy": [2, 3],
                "stable": True,
                "pickable": True,
                "container_candidate": False,
                "attributes": {},
            },
            {
                "object_id": "object_2",
                "raw_class_name": "basket",
                "task_label": "container",
                "confidence": 0.88,
                "bbox_xyxy": [10, 11, 12, 13],
                "center_xy": [11, 12],
                "stable": True,
                "pickable": False,
                "container_candidate": True,
                "attributes": {},
            },
        ],
    }
    ingested = json.loads(
        await tool.execute(
            action="ingest_perception",
            perception_json=json.dumps(perception_payload, ensure_ascii=False),
        )
    )
    assert ingested["scene_frame_id"] == "head_1"
    assert ingested["target_object_id"] == "object_1"
    assert ingested["container_object_id"] == "object_2"
    assert ingested["current_phase"] == "scene_ready"

    planned = json.loads(await tool.execute(action="plan_next_action"))
    assert planned["action_type"] == "pick"
    assert planned["target_object_id"] == "object_1"


@pytest.mark.asyncio
async def test_task_state_tool_records_execution_and_advances_phase(tmp_path: Path) -> None:
    tool = TaskStateTool(workspace=tmp_path)
    tool.set_context("cli", "session-2")
    await tool.execute(action="reset_state")
    await tool.execute(action="set_goal", user_goal="tidy up")

    perception_payload = {
        "frame_id": "head_1",
        "objects": [
            {
                "object_id": "object_1",
                "raw_class_name": "cup",
                "task_label": "object",
                "confidence": 0.91,
                "bbox_xyxy": [1, 2, 3, 4],
                "center_xy": [2, 3],
                "stable": True,
                "pickable": True,
                "container_candidate": False,
                "attributes": {},
            },
        ],
    }
    await tool.execute(
        action="ingest_perception",
        perception_json=json.dumps(perception_payload, ensure_ascii=False),
    )

    picked = json.loads(
        await tool.execute(
            action="record_execution",
            execution_report_json=json.dumps(
                {
                    "action_type": "pick",
                    "status": "success",
                    "reason": "picked",
                    "attempt_count": 1,
                    "observed_effect": {},
                    "next_recommendation": "place",
                },
                ensure_ascii=False,
            ),
        )
    )
    assert picked["current_phase"] == "picked"

    failed = json.loads(
        await tool.execute(
            action="record_execution",
            execution_report_json=json.dumps(
                {
                    "action_type": "place",
                    "status": "retryable_failure",
                    "reason": "blocked",
                    "attempt_count": 2,
                    "observed_effect": {},
                    "next_recommendation": "replan",
                },
                ensure_ascii=False,
            ),
        )
    )
    assert failed["current_phase"] == "needs_replan"
    assert failed["failure_count"] == 1
