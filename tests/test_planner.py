import asyncio

import pytest

from roboclaw.embodied.execution.planner import (
    EmbodiedPlanner,
    ExecutionPlan,
    PlanStep,
)

PRIMITIVES = ("gripper_open", "gripper_close", "go_named_pose", "move_joints")


def test_fallback_matches_primitive():
    planner = EmbodiedPlanner()
    plan = asyncio.run(planner.decompose("gripper_open", PRIMITIVES))
    assert plan.feasible is True
    assert len(plan.steps) == 1
    assert plan.steps[0].primitive == "gripper_open"


def test_fallback_unknown_intent():
    planner = EmbodiedPlanner()
    plan = asyncio.run(planner.decompose("fly to the moon", PRIMITIVES))
    assert plan.feasible is False
    assert "No LLM configured" in plan.infeasible_reason


@pytest.mark.asyncio
async def test_decompose_with_mock_llm():
    async def mock_llm(system_prompt: str, user_message: str) -> str:
        return (
            'STEP: gripper_open | {} | Open the gripper\n'
            'STEP: move_joints | {"shoulder": 0.5} | Move to target\n'
            'STEP: gripper_close | {} | Close the gripper'
        )

    planner = EmbodiedPlanner(llm_caller=mock_llm)
    plan = await planner.decompose("pick up the object", PRIMITIVES)
    assert plan.feasible is True
    assert len(plan.steps) == 3
    assert plan.steps[0].primitive == "gripper_open"
    assert plan.steps[1].primitive == "move_joints"
    assert plan.steps[1].args == {"shoulder": 0.5}
    assert plan.steps[2].primitive == "gripper_close"


@pytest.mark.asyncio
async def test_decompose_infeasible():
    async def mock_llm(system_prompt: str, user_message: str) -> str:
        return "INFEASIBLE: Robot does not have wheels for navigation"

    planner = EmbodiedPlanner(llm_caller=mock_llm)
    plan = await planner.decompose("drive forward 1 meter", PRIMITIVES)
    assert plan.feasible is False
    assert "wheels" in plan.infeasible_reason


@pytest.mark.asyncio
async def test_decompose_llm_exception():
    async def broken_llm(system_prompt: str, user_message: str) -> str:
        raise ConnectionError("LLM service unavailable")

    planner = EmbodiedPlanner(llm_caller=broken_llm)
    plan = await planner.decompose("open gripper", PRIMITIVES)
    assert plan.feasible is False
    assert "unavailable" in plan.infeasible_reason


@pytest.mark.asyncio
async def test_parse_skips_unknown_primitives():
    async def mock_llm(system_prompt: str, user_message: str) -> str:
        return (
            "STEP: gripper_open | {} | Open\n"
            "STEP: fly_away | {} | This primitive does not exist\n"
            "STEP: gripper_close | {} | Close"
        )

    planner = EmbodiedPlanner(llm_caller=mock_llm)
    plan = await planner.decompose("do something", PRIMITIVES)
    assert plan.feasible is True
    assert len(plan.steps) == 2
    assert "Skipped unknown primitive: fly_away" in plan.reasoning
