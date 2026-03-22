"""Minimal skill composition for embodied primitive sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from roboclaw.embodied.definition.foundation.schema import CapabilityFamily
from roboclaw.embodied.execution.orchestration.procedures.model import ProcedureKind

if TYPE_CHECKING:
    from roboclaw.embodied.execution.orchestration.runtime.executor import ProcedureExecutionResult

ProgressCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class SkillStep:
    primitive_name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    steps: tuple[SkillStep, ...]
    parameters: tuple[str, ...] = ()
    required_capabilities: tuple[CapabilityFamily, ...] = field(default_factory=tuple)


# Common skill definitions shared across embodiments
PICK_AND_PLACE = SkillSpec(
    name="pick_and_place",
    description="Open, home, close, home, and release.",
    steps=(
        SkillStep("gripper_open"),
        SkillStep("go_named_pose", {"name": "home"}),
        SkillStep("gripper_close"),
        SkillStep("go_named_pose", {"name": "home"}),
        SkillStep("gripper_open"),
    ),
    required_capabilities=(CapabilityFamily.END_EFFECTOR, CapabilityFamily.NAMED_POSE),
)

RESET_ARM = SkillSpec(
    name="reset_arm",
    description="Return to home and open the gripper.",
    steps=(SkillStep("go_named_pose", {"name": "home"}), SkillStep("gripper_open")),
    required_capabilities=(CapabilityFamily.NAMED_POSE,),
)

DEFAULT_ARM_SKILLS = (PICK_AND_PLACE, RESET_ARM)


async def execute_skill(
    executor: Any,
    context: Any,
    skill: SkillSpec,
    skill_args: dict[str, Any] | None = None,
    on_progress: ProgressCallback | None = None,
) -> ProcedureExecutionResult:
    completed_steps: list[dict[str, Any]] = []
    result_type: type[Any] | None = None
    for index, step in enumerate(skill.steps, start=1):
        if on_progress is not None:
            await on_progress(
                f"Running skill `{skill.name}` step {index}/{len(skill.steps)}: `{step.primitive_name}`."
            )
        result = await executor.execute_move(
            context,
            primitive_name=step.primitive_name,
            primitive_args=step.args,
            on_progress=on_progress,
        )
        result_type = type(result)
        if not result.ok:
            return result
        if on_progress is not None:
            changed = "yes" if result.details.get("state_changed") else "no"
            await on_progress(
                f"Completed skill `{skill.name}` step {index}/{len(skill.steps)}: `{step.primitive_name}` (state changed: {changed})."
            )
        completed_steps.append({"primitive_name": step.primitive_name, "args": dict(step.args)})
    if result_type is None:
        result_type = SimpleNamespace
    return result_type(
        procedure=ProcedureKind.MOVE,
        ok=True,
        message=f"Skill `{skill.name}` completed on setup `{context.setup_id}`.",
        details={"skill_name": skill.name, "skill_args": dict(skill_args or {}), "completed_steps": completed_steps},
    )
