import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from roboclaw.agent.loop import AgentLoop
from roboclaw.agent.tools.base import ToolResult
from roboclaw.agent.tools.libero_skill import (
    LiberoCosmosRouteTool,
    LiberoEnvManager,
    LiberoManipulationTool,
    LiberoObserveTool,
    LiberoPerceptionTool,
    LiberoPlanTool,
    LiberoSkillTool,
    LiberoVerifyTool,
    _extract_router_decision_from_text,
    _model_names,
    _verify_subgoal_from_detections,
)
import roboclaw.agent.tools.libero_skill as libero_skill_module
from roboclaw.bus.queue import MessageBus


async def _inline_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


class _FakeLiberoManager:
    def __init__(self) -> None:
        self._last_obs = None
        self._episode_done = False
        self.ensure_calls = []
        self.reset_calls = []
        self.run_calls = []
        self.skill_summary = {"skill_id": "skill_06", "steps": 42, "success": True, "final_reward": 1.0}

    def ensure_env(self, task: str, task_id: int, episode_length: int) -> None:
        self.ensure_calls.append((task, task_id, episode_length))

    def reset(self, seed: int | None = None) -> None:
        self.reset_calls.append(seed)
        self._last_obs = {"robot_state": {"eef": {"pos": [1, 2, 3], "quat": [0, 0, 0, 1]}, "gripper": {"qpos": [0.5]}}}

    def render(self) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_proprioception(self) -> list[float]:
        return [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.5]

    def run_skill(self, skill_id: str, max_steps: int, show_window: bool = False, *args, **kwargs):
        self.run_calls.append((skill_id, max_steps, show_window, args, kwargs))
        return dict(self.skill_summary, skill_id=skill_id)

    def set_stage(self, label: str) -> None: pass
    def set_memory_summary(self, text: str) -> None: pass
    def set_plan_summary(self, text: str) -> None: pass


def _json_from_tool_result(result: str | ToolResult) -> dict:
    text = result.content if isinstance(result, ToolResult) else result
    return json.loads(text[text.find("{"):])


def test_libero_yolo_class_name_fallback_replaces_generic_names() -> None:
    model = MagicMock()
    model.names = {0: "class_0", 1: "class_1", 2: "class_2", 3: "class_3", 4: "class_4"}

    assert _model_names(model) == {
        0: "white_mug",
        1: "yellow_mug",
        2: "red_mug",
        3: "plate_left",
        4: "plate_right",
    }


def test_cosmos_text_parser_recovers_skill_from_non_json_response() -> None:
    decision = _extract_router_decision_from_text(
        "I choose skill_06 for white_mug_to_right_plate with confidence 0.82."
    )

    assert decision["skill_id"] == "skill_06"
    assert decision["subgoal_id"] == "white_mug_to_right_plate"
    assert decision["confidence"] == 0.82
    assert decision["should_execute"] is True
    assert decision["parser_fallback"] == "text_skill_extraction"


def test_libero_perception_returns_semantic_scene_and_media(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    tool = LiberoPerceptionTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(action="analyze_scene", reset=True, seed=7, run_yolo=False))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["task_id"] == 4
    assert [obj["object_id"] for obj in payload["objects"]] == [
        "white_mug",
        "yellow_mug",
        "red_mug",
        "left_plate",
        "right_plate",
    ]
    assert payload["target_relations"][0]["skill_id"] == "skill_06"
    assert payload["proprioception_8d"][-1] == 0.5
    assert payload["detector"]["status"] == "disabled"
    assert result.media and Path(result.media[0]).is_file()
    assert fake.reset_calls == [7]


def test_libero_perception_returns_optional_yolo_detections(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    def fake_detect(image_path: Path, *, conf: float, imgsz: int = 640):
        return {
            "enabled": True,
            "status": "ok",
            "model_path": "runs/detect/runs/yolo/roboclaw/weights/best.pt",
            "class_names": {0: "white_mug"},
            "image_size": {"width": 4, "height": 4},
            "detections": [
                {
                    "detection_id": "yolo_00",
                    "class_id": 0,
                    "class_name": "white_mug",
                    "confidence": 0.9,
                    "bbox_xyxy": [1.0, 1.0, 3.0, 3.0],
                }
            ],
        }

    monkeypatch.setattr(libero_skill_module, "_run_yolo_detection", fake_detect)

    tool = LiberoPerceptionTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(action="analyze_scene", reset=True, seed=7, run_yolo=True))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["detector"]["status"] == "ok"
    assert payload["detected_objects"][0]["class_name"] == "white_mug"
    assert payload["timing_ms"]["yolo"] >= 0


def test_libero_cv_verifier_satisfied_from_yolo_geometry() -> None:
    report = _verify_subgoal_from_detections(
        "white_mug_to_right_plate",
        [
            {
                "class_name": "white_mug",
                "confidence": 0.92,
                "bbox_xyxy": [80.0, 70.0, 120.0, 130.0],
            },
            {
                "class_name": "plate_right",
                "confidence": 0.88,
                "bbox_xyxy": [70.0, 110.0, 140.0, 150.0],
            },
        ],
    )

    assert report["status"] == "satisfied"
    assert report["satisfied"] is True


def test_libero_verify_tool_returns_local_cv_report(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    def fake_detect(image_source, *, conf: float, imgsz: int = 640):
        return {
            "enabled": True,
            "status": "ok",
            "detections": [
                {"class_name": "yellow_mug", "confidence": 0.91, "bbox_xyxy": [20.0, 70.0, 60.0, 130.0]},
                {"class_name": "plate_left", "confidence": 0.89, "bbox_xyxy": [10.0, 110.0, 80.0, 150.0]},
            ],
        }

    monkeypatch.setattr(libero_skill_module, "_run_yolo_detection", fake_detect)

    tool = LiberoVerifyTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="verify_subgoal",
        subgoal_id="yellow_mug_to_left_plate",
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["action_type"] == "libero_verify"
    assert payload["status"] == "satisfied"
    assert payload["satisfied"] is True
    assert payload["next_recommendation"] == "continue_to_next_subgoal"


def test_libero_plan_maps_goal_to_skill_sequence() -> None:
    tool = LiberoPlanTool()
    perception_json = json.dumps({"frame_id": "libero_1", "objects": [{"object_id": "white_mug"}]})

    payload = json.loads(
        asyncio.run(tool.execute(
            action="plan_task",
            user_goal="Put the white mug on the right plate and the yellow mug on the left plate.",
            perception_json=perception_json,
        ))
    )

    assert payload["ordered_subgoals"][0]["subgoal_id"] == "white_mug_to_right_plate"
    assert payload["ordered_subgoals"][0]["skill_id"] == "skill_06"
    assert payload["ordered_subgoals"][1]["skill_id"] == "skill_07"
    assert payload["next_action"]["skill_id"] == "skill_06"
    assert payload["next_action"]["requires_verification"] is False


def test_libero_plan_can_respect_requested_yellow_first_order() -> None:
    tool = LiberoPlanTool()
    payload = json.loads(
        asyncio.run(tool.execute(
            action="plan_task",
            user_goal="Run skill_07 first, then skill_06.",
            perception_json=json.dumps({"frame_id": "libero_1", "objects": []}),
        ))
    )

    assert payload["ordered_subgoals"][0]["skill_id"] == "skill_07"
    assert payload["ordered_subgoals"][1]["skill_id"] == "skill_06"


def test_libero_plan_uses_memory_preferences_for_order() -> None:
    tool = LiberoPlanTool()
    payload = json.loads(
        asyncio.run(tool.execute(
            action="plan_task",
            user_goal="Put the mugs on their preferred plates.",
            perception_json=json.dumps({"frame_id": "libero_1", "objects": []}),
            memory_preferences="execution_order: yellow_mug_to_left_plate first, then white_mug_to_right_plate",
        ))
    )

    assert payload["ordered_subgoals"][0]["skill_id"] == "skill_07"
    assert payload["ordered_subgoals"][1]["skill_id"] == "skill_06"
    assert "yellow_mug_to_left_plate first" in payload["memory_preferences_used"]


def test_libero_plan_filters_hands_off_subgoals() -> None:
    tool = LiberoPlanTool()
    planning_memory_context = {
        "object_constraints": {"hands_off": ["yellow_mug"]},
        "operational_rules": {
            "hands_off": [{
                "target": "yellow_mug",
                "value": True,
                "source": "explicit_user",
                "confidence": 0.98,
            }]
        },
    }

    payload = json.loads(
        asyncio.run(tool.execute(
            action="plan_task",
            user_goal="Tidy up the mugs.",
            perception_json=json.dumps({"frame_id": "libero_1", "objects": []}),
            planning_memory_context=json.dumps(planning_memory_context),
        ))
    )

    assert [sg["subgoal_id"] for sg in payload["ordered_subgoals"]] == [
        "white_mug_to_right_plate"
    ]
    assert payload["next_action"]["skill_id"] == "skill_06"
    assert payload["personalization_decisions"] == [{
        "decision": "exclude_subgoal",
        "subgoal_id": "yellow_mug_to_left_plate",
        "memory_source": "operational_rule:hands_off:yellow_mug",
        "reason": "User marked yellow_mug as hands-off.",
    }]


def test_libero_cosmos_route_uses_local_vlm_decision(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    async def fake_cosmos_call(**kwargs):
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        assert kwargs["model"] == "nvidia/Cosmos-Reason2-2B"
        assert kwargs["image_data_url"].startswith("data:image/png;base64,")
        return {
            "skill": "skill_06",
            "target": "right_plate",
            "confidence": 0.88,
            "should_execute": True,
            "reason": "white mug route matches planner",
        }

    monkeypatch.setattr(libero_skill_module, "_call_cosmos_reason2_router", fake_cosmos_call)

    plan = {
        "ordered_subgoals": [{
            "subgoal_id": "white_mug_to_right_plate",
            "skill_id": "skill_06",
            "target_container_id": "right_plate",
        }]
    }
    tool = LiberoCosmosRouteTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="route_skill",
        plan_json=json.dumps(plan),
        subgoal_id="white_mug_to_right_plate",
        intended_skill_id="skill_06",
        allowed_skill_ids=["skill_06", "wait", "recover"],
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["action_type"] == "libero_cosmos_route"
    assert payload["status"] == "ok"
    assert payload["router_role"] == "fast_vlm_subagent"
    assert payload["decision"]["skill_id"] == "skill_06"
    assert payload["decision"]["should_execute"] is True
    assert result.media and Path(result.media[0]).is_file()


def test_libero_cosmos_route_can_fallback_to_planner_skill(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    async def failing_cosmos_call(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(libero_skill_module, "_call_cosmos_reason2_router", failing_cosmos_call)

    tool = LiberoCosmosRouteTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="route_skill",
        plan_json=json.dumps({"ordered_subgoals": []}),
        subgoal_id="white_mug_to_right_plate",
        intended_skill_id="skill_06",
        allowed_skill_ids=["skill_06", "wait"],
        allow_fallback=True,
    ))

    payload = _json_from_tool_result(result)
    assert payload["status"] == "fallback"
    assert payload["fallback_used"] is True
    assert payload["decision"]["skill_id"] == "skill_06"
    assert payload["decision"]["should_execute"] is True
    assert "connection refused" in payload["error"]


def test_libero_manipulation_can_delegate_control_loop_to_cosmos(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    decisions = [
        {
            "skill": "skill_06",
            "subgoal_id": "white_mug_to_right_plate",
            "target": "right_plate",
            "confidence": 0.9,
            "should_execute": True,
            "reason": "start with white mug",
        },
        {
            "skill": "done",
            "completed_subgoals": ["white_mug_to_right_plate"],
            "confidence": 0.93,
            "should_execute": False,
            "reason": "white mug is visually complete",
        },
        {
            "skill": "skill_07",
            "subgoal_id": "yellow_mug_to_left_plate",
            "target": "left_plate",
            "confidence": 0.91,
            "should_execute": True,
            "reason": "continue with yellow mug",
        },
        {
            "skill": "done",
            "completed_subgoals": ["yellow_mug_to_left_plate"],
            "confidence": 0.93,
            "should_execute": False,
            "reason": "yellow mug is visually complete",
        },
    ]

    async def fake_cosmos_call(**kwargs):
        assert kwargs["image_data_url"].startswith("data:image/png;base64,")
        return decisions.pop(0)

    monkeypatch.setattr(libero_skill_module, "_call_cosmos_reason2_router", fake_cosmos_call)

    plan = {
        "plan_id": "plan_1",
        "user_goal": "tidy mugs",
        "ordered_subgoals": [
            {
                "subgoal_id": "white_mug_to_right_plate",
                "skill_id": "skill_06",
                "target_container_id": "right_plate",
            },
            {
                "subgoal_id": "yellow_mug_to_left_plate",
                "skill_id": "skill_07",
                "target_container_id": "left_plate",
            },
        ],
    }
    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        plan_json=json.dumps(plan),
        previous_summary="No hands-off objects.",
        use_cosmos_controller=True,
        cosmos_max_decisions=3,
        cosmos_chunk_steps=25,
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["action_type"] == "libero_cosmos_manipulation"
    assert payload["status"] == "success"
    assert payload["controller_role"] == "fast_vlm_manipulation_controller"
    assert payload["completed_subgoals"] == [
        "white_mug_to_right_plate",
        "yellow_mug_to_left_plate",
    ]
    assert fake.run_calls[0][0:2] == ("skill_06", 25)
    assert fake.run_calls[1][0:2] == ("skill_07", 25)
    assert fake.run_calls[0][3][0] is None
    assert fake.run_calls[1][3][0] is None
    assert payload["local_verify"] is False
    assert result.media


def test_cosmos_controller_can_mark_visual_completion_when_reward_is_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    fake.skill_summary = {
        "skill_id": "skill_06",
        "steps": 10,
        "success": False,
        "final_reward": 0.0,
    }
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    decisions = [
        {
            "skill": "skill_06",
            "subgoal_id": "white_mug_to_right_plate",
            "confidence": 0.95,
            "should_execute": True,
            "reason": "white mug still needs placement",
        },
        {
            "skill": "done",
            "completed_subgoals": ["white_mug_to_right_plate"],
            "subgoal_statuses": {"white_mug_to_right_plate": "completed"},
            "confidence": 0.92,
            "should_execute": False,
            "reason": "white mug is visually on the right plate",
        },
    ]

    async def fake_cosmos_call(**kwargs):
        return decisions.pop(0)

    monkeypatch.setattr(libero_skill_module, "_call_cosmos_reason2_router", fake_cosmos_call)

    plan = {
        "plan_id": "plan_white_only",
        "user_goal": "tidy white mug",
        "ordered_subgoals": [{
            "subgoal_id": "white_mug_to_right_plate",
            "skill_id": "skill_06",
            "target_container_id": "right_plate",
        }],
    }
    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        plan_json=json.dumps(plan),
        use_cosmos_controller=True,
        cosmos_chunk_steps=10,
        cosmos_max_decisions=3,
    ))

    payload = _json_from_tool_result(result)
    assert payload["status"] == "success"
    assert payload["stop_reason"] == "all_subgoals_completed"
    assert payload["completed_subgoals"] == ["white_mug_to_right_plate"]
    assert payload["execution_trace"][0]["skill_summary"]["final_reward"] == 0.0
    assert len(fake.run_calls) == 1


def test_libero_manipulation_runs_requested_skill_and_reports(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        skill_id="skill_06",
        subgoal_id="white_mug_to_right_plate",
        max_steps=123,
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["action_type"] == "libero_skill"
    assert payload["status"] == "success"
    assert payload["observed_effect"]["skill_id"] == "skill_06"
    assert payload["observed_effect"]["subgoal_id"] == "white_mug_to_right_plate"
    assert payload["observed_effect"]["steps"] == 42
    assert fake.run_calls[0][0:3] == ("skill_06", 123, False)
    assert fake.run_calls[0][3][0] == "white_mug_to_right_plate"
    assert fake.run_calls[0][3][4] is True  # early_stop_on_verify default flipped
    assert fake.run_calls[0][3][5] == 30  # verifier_settle_steps default
    assert result.media and Path(result.media[0]).is_file()


def test_libero_manipulation_skips_duplicate_subgoal_by_default(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    fake._pipeline_subgoal_attempts = {"white_mug_to_right_plate": 1}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        skill_id="skill_06",
        subgoal_id="white_mug_to_right_plate",
        max_steps=123,
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["status"] == "skipped_duplicate"
    assert payload["observed_effect"]["skill_id"] == "skill_06"
    assert fake.run_calls == []


def test_libero_manipulation_zero_reward_requires_visual_verification(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    fake.skill_summary = {"skill_id": "skill_06", "steps": 150, "success": False, "final_reward": 0.0}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        skill_id="skill_06",
        subgoal_id="white_mug_to_right_plate",
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["status"] == "needs_visual_verification"
    assert payload["reason"] == "reward_zero_visual_verification_required"
    assert payload["observed_effect"]["final_reward"] == 0.0
    assert "zero reward is not enough" in payload["verification_hint"]


def test_libero_manipulation_trusts_local_cv_success(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    fake.skill_summary = {
        "skill_id": "skill_06",
        "steps": 40,
        "success": False,
        "final_reward": 0.0,
        "local_verifier": {
            "enabled": True,
            "subgoal_id": "white_mug_to_right_plate",
            "status": "satisfied",
            "checks": 3,
            "last_report": {"satisfied": True, "status": "satisfied"},
        },
    }
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    tool = LiberoManipulationTool(workspace=tmp_path)
    result = asyncio.run(tool.execute(
        action="execute_skill",
        skill_id="skill_06",
        subgoal_id="white_mug_to_right_plate",
    ))

    assert isinstance(result, ToolResult)
    payload = _json_from_tool_result(result)
    assert payload["status"] == "success"
    assert payload["reason"] == "local_cv_verifier_satisfied"
    assert payload["observed_effect"]["local_verifier"]["checks"] == 3


def test_libero_compat_tools_ignore_mid_episode_reset(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeLiberoManager()
    fake._last_obs = {"already": "reset"}
    monkeypatch.setattr(LiberoEnvManager, "get", classmethod(lambda cls: fake))
    monkeypatch.setattr(libero_skill_module.asyncio, "to_thread", _inline_to_thread)

    skill_tool = LiberoSkillTool(workspace=tmp_path)
    skill_result = asyncio.run(skill_tool.execute(
        skill_id="skill_06",
        reset=True,
    ))
    skill_payload = _json_from_tool_result(skill_result)

    observe_tool = LiberoObserveTool(workspace=tmp_path)
    observe_result = asyncio.run(observe_tool.execute(reset=True))
    observe_payload = _json_from_tool_result(observe_result)

    assert fake.reset_calls == []
    assert skill_payload["reset_requested"] is True
    assert skill_payload["reset_applied"] is False
    assert skill_payload["reset_ignored"] is True
    assert observe_payload["reset_requested"] is True
    assert observe_payload["reset_applied"] is False
    assert observe_payload["reset_ignored"] is True


def test_agent_loop_registers_libero_pipeline_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ROBOCLAW_ENABLE_LIBERO", "1")
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
    )

    names = set(loop.tools.tool_names)
    assert {
        "libero_perception",
        "libero_plan",
        "libero_cosmos_route",
        "libero_verify",
        "libero_manipulation",
    }.issubset(names)
    assert {"libero_skill", "libero_observe"}.issubset(names)
