#!/usr/bin/env python3
"""Demo: RoboClaw agent runs the LIBERO perception-plan-manipulation pipeline.

Uses an OpenAI-compatible API (any provider that follows the OpenAI chat format).

The agent receives a natural-language task, then should:
  1. Call libero_perception(action="analyze_scene", reset=false, seed=0)
  2. Call libero_plan(action="plan_task", ...)
  3. Call libero_manipulation(action="execute_skill", skill_id="skill_06", local_verify=true, ...)
  4. Call libero_verify only if the manipulation report is uncertain
  5. Call libero_manipulation(action="execute_skill", skill_id="skill_07", ...)
  6. Call libero_verify only if the manipulation report is uncertain

Usage:
    export OPENAI_API_KEY=sk-...
    export OPENAI_API_BASE=https://...
    export OPENAI_MODEL=claude-opus-4-5
    export ROBOCLAW_ENABLE_LIBERO=1

    cd /home/xinyuan/RoboClaw
    conda run -n lerobot312 python scripts/demo_libero_pipeline_agent.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

if os.environ.get("ROBOCLAW_ENABLE_LIBERO", "0") != "1":
    os.environ["ROBOCLAW_ENABLE_LIBERO"] = "1"

_DEFAULT_TASK = (
    "Run the LIBERO perception-plan-manipulation pipeline. Goal: put the white mug "
    "on the right plate and the yellow mug on the left plate. Start with "
    "libero_perception(action='analyze_scene', reset=false, seed=0); the tool will "
    "initialize a fresh episode only if none exists and will return optional YOLO "
    "detected_objects as auxiliary visual evidence. Then call "
    "libero_plan(action='plan_task') using the perception JSON. Execute each planned "
    "sub-goal with libero_manipulation(action='execute_skill', local_verify=true, "
    "verifier_hz=5, early_stop_on_verify=true). The local CV verifier ends the rollout "
    "as soon as the mug is visibly on the target plate so the next skill can start "
    "without delay. Only call libero_verify or inspect the RGB "
    "image when the local verifier status is uncertain. "
    "Never request reset during the task. Execute each planned sub-goal once "
    "by default; if the mug is visibly on its target plate, mark it complete and move "
    "to the next sub-goal even if the reward flag is false or uncertain. Only retry "
    "with allow_retry=true when visual verification clearly shows the mug did not move. "
    "Treat status='needs_visual_verification' as pending visual evidence, not as failure. "
    "Summarize the perception, plan, manipulation reports, local CV verification, "
    "and any uncertain cases that required LLM visual reasoning."
)


async def run_demo(task: str, workspace: Path, api_key: str, api_base: str, model: str) -> None:
    from roboclaw.agent.loop import AgentLoop
    from roboclaw.bus.queue import MessageBus
    from roboclaw.providers.base import GenerationSettings
    from roboclaw.providers.custom_provider import CustomProvider

    workspace.mkdir(parents=True, exist_ok=True)
    videos_dir = workspace / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Endpoint : {api_base}")
    print(f"Model    : {model}")
    print(f"Workspace: {workspace}")
    print(f"Task     : {task[:100]}...")
    print(f"{'=' * 60}\n")

    provider = CustomProvider(api_key=api_key, api_base=api_base, default_model=model)
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=8192)

    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=40,
        context_window_tokens=65536,
    )

    registered = agent.tools.tool_names
    required = {"libero_perception", "libero_plan", "libero_verify", "libero_manipulation"}
    missing = sorted(required.difference(registered))
    if missing:
        print(f"[error] Missing LIBERO pipeline tools: {missing}")
        sys.exit(1)

    print(f"Registered LIBERO tools: {[t for t in registered if 'libero' in t]}\n")

    from roboclaw.agent.tools.libero_skill import LiberoEnvManager

    mgr = LiberoEnvManager.get()
    mgr.start_recording()
    print("Video recording started (frames collected during all LIBERO calls).\n")

    async def on_progress(text: str, *, tool_hint: bool = False) -> None:
        prefix = "  [tool]" if tool_hint else "  >"
        print(f"{prefix} {text}", flush=True)

    session_key = f"demo:libero-pipeline:{int(time.time())}"
    response = await agent.process_direct(
        content=task,
        session_key=session_key,
        on_progress=on_progress,
    )

    print(f"\n{'=' * 60}")
    print("AGENT FINAL RESPONSE:")
    print(response)
    print(f"{'=' * 60}")

    mgr.stop_recording()
    ts = int(time.time())
    video_path = videos_dir / f"pipeline_session_{ts}.mp4"
    saved = mgr.save_video(video_path, fps=10)
    if saved:
        print(f"\nSession video saved -> {saved} ({len(mgr._session_frames)} frames)")
    else:
        print("\n[info] No frames recorded (pipeline tools may not have run).")

    try:
        mgr.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboClaw LIBERO pipeline agent demo")
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "tmp" / "rc_libero_pipeline_demo")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "claude-opus-4-5"))
    args = parser.parse_args()

    if not args.api_key:
        print("[error] No API key. Pass --api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    asyncio.run(run_demo(args.task, args.workspace, args.api_key, args.api_base, args.model))


if __name__ == "__main__":
    main()
