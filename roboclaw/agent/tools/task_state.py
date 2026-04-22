"""Shared task-state tool for multi-agent tidyup coordination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool
from roboclaw.embodied.tasking import ExecutionReport, HighLevelAction, SceneObject, TaskStateStore


class TaskStateTool(Tool):
    """Read and update the shared world state used by supervisor and subagents."""

    def __init__(self, workspace: Path):
        self._store = TaskStateStore(workspace)
        self._session_key = "cli:direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "task_state"

    @property
    def description(self) -> str:
        return (
            "Read or update the shared tidyup world state. "
            "Use this to persist scene objects, targets, current phase, and execution reports."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_state",
                        "reset_state",
                        "set_goal",
                        "ingest_perception",
                        "plan_next_action",
                        "record_execution",
                    ],
                    "description": "The task-state action to perform.",
                },
                "user_goal": {"type": "string"},
                "perception_json": {"type": "string"},
                "execution_report_json": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        user_goal: str = "",
        perception_json: str = "",
        execution_report_json: str = "",
        **kwargs: Any,
    ) -> str:
        state = self._store.load(self._session_key)

        if action == "get_state":
            return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        if action == "reset_state":
            state = self._store.reset(self._session_key)
            return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        if action == "set_goal":
            state.user_goal = user_goal.strip()
            state.current_phase = "need_scene"
            self._store.save(self._session_key, state)
            return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        if action == "ingest_perception":
            if not perception_json.strip():
                return "Error: ingest_perception requires perception_json."
            payload = json.loads(perception_json)
            state.scene_frame_id = str(payload.get("frame_id") or "")
            state.objects = [SceneObject.from_dict(item) for item in payload.get("objects", [])]
            state.candidate_grasps = []
            state.selected_grasp = None
            state.grasp_backend = None
            state.grasp_frame_id = ""
            if not state.target_object_id:
                pickables = [obj for obj in state.objects if obj.pickable and obj.stable]
                if pickables:
                    state.target_object_id = pickables[0].object_id
            if not state.container_object_id:
                containers = [obj for obj in state.objects if obj.container_candidate and obj.stable]
                if containers:
                    state.container_object_id = containers[0].object_id
            state.current_phase = "scene_ready" if state.objects else "need_scene"
            self._store.save(self._session_key, state)
            return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        if action == "plan_next_action":
            planned = self._plan_next_action(state)
            return json.dumps(planned.to_dict(), ensure_ascii=False, indent=2)
        if action == "record_execution":
            if not execution_report_json.strip():
                return "Error: record_execution requires execution_report_json."
            report = ExecutionReport(**json.loads(execution_report_json))
            state.last_executor_report = report.to_dict()
            if report.status == "success":
                if report.action_type == "pick":
                    state.current_phase = "picked"
                elif report.action_type == "place":
                    state.current_phase = "placed"
                elif report.action_type == "verify":
                    state.current_phase = "done"
            else:
                state.failure_count += 1
                state.current_phase = "needs_replan"
            self._store.save(self._session_key, state)
            return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        return f"Error: Unsupported task_state action '{action}'."

    def _plan_next_action(self, state) -> HighLevelAction:
        if not state.scene_frame_id or not state.objects:
            return HighLevelAction(action_type="inspect", retry_budget=1)
        if state.current_phase in {"idle", "need_scene", "scene_ready", "needs_replan"}:
            if state.target_object_id:
                return HighLevelAction(
                    action_type="pick",
                    target_object_id=state.target_object_id,
                    constraints={"goal": state.user_goal},
                    retry_budget=2,
                )
            return HighLevelAction(action_type="inspect", retry_budget=1)
        if state.current_phase == "picked":
            return HighLevelAction(
                action_type="place",
                target_object_id=state.target_object_id,
                target_container_id=state.container_object_id,
                constraints={"goal": state.user_goal},
                retry_budget=2,
            )
        if state.current_phase == "placed":
            return HighLevelAction(
                action_type="verify",
                target_object_id=state.target_object_id,
                target_container_id=state.container_object_id,
                retry_budget=1,
            )
        return HighLevelAction(action_type="inspect", retry_budget=1)
