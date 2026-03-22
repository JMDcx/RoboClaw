"""LLM-based intent decomposition for embodied skill composition (L2)."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanStep:
    """A single step in an execution plan."""

    primitive: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    requires_perception: bool = False


@dataclass(frozen=True)
class ExecutionPlan:
    """A sequence of steps decomposed from a high-level intent."""

    intent: str
    steps: tuple[PlanStep, ...] = ()
    reasoning: str = ""
    feasible: bool = True
    infeasible_reason: str = ""


class EmbodiedPlanner:
    """Decomposes high-level user intents into primitive sequences using an LLM."""

    def __init__(self, llm_caller=None):
        """
        Args:
            llm_caller: async callable (system_prompt: str, user_message: str) -> str.
                        If None, returns a fallback single-step plan.
        """
        self._llm_caller = llm_caller

    async def decompose(
        self,
        intent: str,
        available_primitives: tuple[str, ...],
        capability_families: tuple[str, ...] = (),
        robot_name: str = "",
    ) -> ExecutionPlan:
        if self._llm_caller is None:
            return self._fallback_plan(intent, available_primitives)

        system_prompt = self._build_system_prompt(
            available_primitives, capability_families, robot_name
        )
        try:
            response = await self._llm_caller(system_prompt, f"User intent: {intent}")
            return self._parse_response(intent, response, available_primitives)
        except Exception as exc:
            return ExecutionPlan(
                intent=intent,
                feasible=False,
                infeasible_reason=str(exc),
                reasoning=f"Planning failed: {exc}",
            )

    def _build_system_prompt(
        self,
        available_primitives: tuple[str, ...],
        capability_families: tuple[str, ...],
        robot_name: str,
    ) -> str:
        primitives_list = "\n".join(f"- {p}" for p in available_primitives)
        capabilities = ", ".join(capability_families) if capability_families else "unknown"
        return (
            f"You are a robot motion planner for {robot_name or 'an embodied robot'}.\n"
            f"Capabilities: {capabilities}\n"
            f"Available primitives:\n{primitives_list}\n\n"
            "Given the user intent, decompose it into a sequence of primitive calls.\n"
            "Respond in this exact format, one step per line:\n"
            "STEP: <primitive_name> | <args_json> | <description>\n"
            "If the intent is not feasible, respond:\n"
            "INFEASIBLE: <reason>\n"
            "Use ONLY the available primitives listed above."
        )

    def _parse_response(
        self, intent: str, response: str, available_primitives: tuple[str, ...]
    ) -> ExecutionPlan:
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

        for line in lines:
            if line.startswith("INFEASIBLE:"):
                reason = line.split(":", 1)[1].strip()
                return ExecutionPlan(intent=intent, feasible=False, infeasible_reason=reason)

        steps: list[PlanStep] = []
        notes: list[str] = []
        for line in lines:
            if not line.startswith("STEP:"):
                notes.append(line)
                continue
            parts = line[5:].split("|")
            primitive = parts[0].strip() if parts else ""
            args: dict[str, Any] = {}
            if len(parts) >= 2:
                try:
                    args = _json.loads(parts[1].strip())
                except (ValueError, _json.JSONDecodeError):
                    pass
            description = parts[2].strip() if len(parts) >= 3 else ""
            if primitive in available_primitives:
                steps.append(PlanStep(primitive=primitive, args=args, description=description))
            else:
                notes.append(f"Skipped unknown primitive: {primitive}")

        return ExecutionPlan(
            intent=intent,
            steps=tuple(steps),
            reasoning="\n".join(notes),
            feasible=len(steps) > 0,
            infeasible_reason="" if steps else "No valid steps parsed from LLM response.",
        )

    def _fallback_plan(
        self, intent: str, available_primitives: tuple[str, ...]
    ) -> ExecutionPlan:
        intent_lower = intent.lower()
        for primitive in available_primitives:
            if primitive.replace("_", " ") in intent_lower or primitive in intent_lower:
                return ExecutionPlan(
                    intent=intent,
                    steps=(PlanStep(primitive=primitive, description=f"Direct match: {primitive}"),),
                    reasoning="Fallback: matched intent to primitive by name.",
                )
        return ExecutionPlan(
            intent=intent,
            feasible=False,
            infeasible_reason="No LLM configured and intent could not be matched to a known primitive.",
            reasoning="Fallback: no LLM available and no primitive name matched.",
        )
