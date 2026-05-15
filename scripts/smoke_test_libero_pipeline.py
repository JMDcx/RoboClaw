#!/usr/bin/env python3
"""Smoke test for the LIBERO perception-plan-manipulation pipeline.

This bypasses the LLM and drives the new structured tools directly:

    libero_perception(reset=false)
      -> libero_plan
      -> libero_manipulation(skill_06, local_verify=true)
      -> libero_verify if needed
      -> libero_manipulation(skill_07, local_verify=true)
      -> libero_verify if needed

Usage:
    conda run -n lerobot312 python scripts/smoke_test_libero_pipeline.py
    conda run -n lerobot312 python scripts/smoke_test_libero_pipeline.py --workspace /tmp/rc_libero_pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

if os.environ.get("ROBOCLAW_ENABLE_LIBERO", "0") != "1":
    os.environ["ROBOCLAW_ENABLE_LIBERO"] = "1"

from roboclaw.agent.tools.base import ToolResult
from roboclaw.agent.tools.libero_skill import (
    LiberoEnvManager,
    LiberoManipulationTool,
    LiberoPerceptionTool,
    LiberoPlanTool,
    LiberoVerifyTool,
)


_DEFAULT_GOAL = "Put the white mug on the right plate and the yellow mug on the left plate."


def _content(result: str | ToolResult) -> str:
    return result.content if isinstance(result, ToolResult) else result


def _json_from_result(result: str | ToolResult) -> dict:
    text = _content(result)
    start = text.find("{")
    if start >= 0:
        text = text[start:]
    return json.loads(text)


def _print_result(label: str, result: str | ToolResult) -> None:
    print(f"\n=== {label} ===")
    print(_content(result))
    if isinstance(result, ToolResult) and result.media:
        print(f"  media: {result.media}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test LIBERO structured pipeline tools.")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/rc_libero_pipeline"))
    parser.add_argument("--goal", default=_DEFAULT_GOAL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=150)
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {args.workspace}")
    print(f"Goal: {args.goal}")

    perception = LiberoPerceptionTool(workspace=args.workspace)
    planner = LiberoPlanTool()
    manipulation = LiberoManipulationTool(workspace=args.workspace)
    verifier = LiberoVerifyTool(workspace=args.workspace)

    perception_result = await perception.execute(action="analyze_scene", reset=False, seed=args.seed)
    _print_result("perception (initial)", perception_result)
    perception_payload = _json_from_result(perception_result)

    plan_result = await planner.execute(
        action="plan_task",
        user_goal=args.goal,
        perception_json=json.dumps(perception_payload, ensure_ascii=False),
    )
    _print_result("plan", plan_result)
    plan_payload = json.loads(plan_result)

    for subgoal in plan_payload["ordered_subgoals"]:
        skill_result = await manipulation.execute(
            action="execute_skill",
            skill_id=subgoal["skill_id"],
            subgoal_id=subgoal["subgoal_id"],
            max_steps=args.max_steps,
            local_verify=True,
            verifier_hz=5,
            early_stop_on_verify=True,
        )
        _print_result(f"manipulation ({subgoal['subgoal_id']})", skill_result)
        skill_payload = _json_from_result(skill_result)
        local_verifier = skill_payload.get("observed_effect", {}).get("local_verifier", {})
        if skill_payload.get("status") != "success" or local_verifier.get("status") == "uncertain":
            verify_result = await verifier.execute(
                action="verify_subgoal",
                subgoal_id=subgoal["subgoal_id"],
            )
            _print_result(f"local verify ({subgoal['subgoal_id']})", verify_result)

    LiberoEnvManager.get().close()
    print("\nDone. Browse pipeline artifacts under:")
    print(f"  {args.workspace / '.roboclaw_tmp' / 'libero'}")


if __name__ == "__main__":
    asyncio.run(main())
