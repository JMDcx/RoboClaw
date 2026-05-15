#!/usr/bin/env python3
"""Smoke test for LIBERO IL skill tools (no LLM, no agent loop).

Drives the two tools directly in the order an LLM should:
    libero_observe (initial)
      → libero_skill(skill_06, reset=True)
      → libero_observe (verify white mug on right plate)
      → libero_skill(skill_07)
      → libero_observe (verify yellow mug on left plate)

Saves keyframes under <workspace>/.roboclaw_tmp/libero/ so you can flip through
them after the run.

Usage:
    conda run -n lerobot312 python scripts/smoke_test_libero_tools.py
    conda run -n lerobot312 python scripts/smoke_test_libero_tools.py --workspace /tmp/rc_libero
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from roboclaw.agent.tools.base import ToolResult
from roboclaw.agent.tools.libero_skill import LiberoEnvManager, LiberoObserveTool, LiberoSkillTool


def _print_result(label: str, result: str | ToolResult) -> None:
    print(f"\n=== {label} ===")
    if isinstance(result, ToolResult):
        print(result.content)
        if result.media:
            print(f"  media: {result.media}")
    else:
        print(result)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/rc_libero_smoke"))
    parser.add_argument("--skill06-steps", type=int, default=150)
    parser.add_argument("--skill07-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.workspace.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {args.workspace}")

    skill_tool = LiberoSkillTool(workspace=args.workspace)
    observe_tool = LiberoObserveTool(workspace=args.workspace)

    # 1. Initial observation (also resets env to a fresh episode)
    res = await observe_tool.execute(reset=True, seed=args.seed)
    _print_result("observe (initial)", res)

    # 2. Skill 06: white mug → right plate
    res = await skill_tool.execute(skill_id="skill_06", max_steps=args.skill06_steps)
    _print_result("skill_06", res)

    # 3. Verify
    res = await observe_tool.execute()
    _print_result("observe (after skill_06)", res)

    # 4. Skill 07: yellow mug → left plate (continues from previous state)
    res = await skill_tool.execute(skill_id="skill_07", max_steps=args.skill07_steps)
    _print_result("skill_07", res)

    # 5. Final verify
    res = await observe_tool.execute()
    _print_result("observe (final)", res)

    # Cleanup
    LiberoEnvManager.get().close()
    print("\nDone. Browse keyframes under:")
    print(f"  {args.workspace / '.roboclaw_tmp' / 'libero'}")


if __name__ == "__main__":
    asyncio.run(main())
