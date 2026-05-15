#!/usr/bin/env python3
"""Demo: RoboClaw agent schedules LIBERO IL skills end-to-end.

Uses an OpenAI-compatible API (any provider that follows the OpenAI chat format).

The agent receives a natural-language task, then autonomously:
  1. Calls libero_observe (initial scene)
  2. Calls libero_skill(skill_06) — white mug → right plate
  3. Calls libero_observe to verify
  4. Calls libero_skill(skill_07) — yellow mug → left plate
  5. Calls libero_observe to confirm

Usage:
    export OPENAI_API_KEY=sk-...          # your key
    export OPENAI_API_BASE=https://...    # your endpoint (default: https://api.openai.com/v1)
    export OPENAI_MODEL=claude-opus-4-5   # model name your endpoint accepts

    export ROBOCLAW_ENABLE_LIBERO=1
    cd /home/xinyuan/RoboClaw
    python scripts/demo_libero_agent.py

All three env vars can also be passed as CLI flags (--api-key, --api-base, --model).
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
    "Please put the white mug on the right plate and the yellow mug on the left plate "
    "in the LIBERO simulation. Start a fresh episode first (reset=true, seed=0). "
    "Use libero_observe to check the scene first, then run the appropriate skills in order, "
    "and verify each sub-goal with libero_observe before proceeding to the next skill."
)


async def run_demo(task: str, workspace: Path, api_key: str, api_base: str, model: str) -> None:
    from roboclaw.providers.custom_provider import CustomProvider
    from roboclaw.providers.base import GenerationSettings
    from roboclaw.agent.loop import AgentLoop
    from roboclaw.bus.queue import MessageBus

    workspace.mkdir(parents=True, exist_ok=True)
    videos_dir = workspace / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Endpoint : {api_base}")
    print(f"Model    : {model}")
    print(f"Workspace: {workspace}")
    print(f"Task     : {task[:80]}...")
    print(f"{'='*60}\n")

    provider = CustomProvider(api_key=api_key, api_base=api_base, default_model=model)
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=8192)

    bus = MessageBus()
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=30,
        context_window_tokens=65536,
    )

    registered = agent.tools.tool_names
    if "libero_skill" not in registered:
        print("[error] libero_skill tool not registered — is ROBOCLAW_ENABLE_LIBERO=1?")
        sys.exit(1)

    print(f"Registered LIBERO tools: {[t for t in registered if 'libero' in t]}\n")

    # Start session-level video recording before the agent runs
    from roboclaw.agent.tools.libero_skill import LiberoEnvManager
    mgr = LiberoEnvManager.get()
    mgr.start_recording()
    print("Video recording started (frames collected during all skill calls).\n")

    async def on_progress(text: str, *, tool_hint: bool = False) -> None:
        prefix = "  🔧" if tool_hint else "  ▸"
        print(f"{prefix} {text}", flush=True)

    session_key = f"demo:libero:{int(time.time())}"
    response = await agent.process_direct(
        content=task,
        session_key=session_key,
        on_progress=on_progress,
    )

    print(f"\n{'='*60}")
    print("AGENT FINAL RESPONSE:")
    print(response)
    print(f"{'='*60}")

    # Save session video
    mgr.stop_recording()
    ts = int(time.time())
    video_path = videos_dir / f"session_{ts}.mp4"
    saved = mgr.save_video(video_path, fps=10)
    if saved:
        print(f"\nSession video saved → {saved}  ({len(mgr._session_frames)} frames)")
    else:
        print("\n[info] No frames recorded (skills may not have run).")

    try:
        mgr.close()
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="RoboClaw LIBERO IL skill agent demo (OpenAI-compatible API)")
    ap.add_argument("--task", default=_DEFAULT_TASK)
    ap.add_argument("--workspace", type=Path, default=Path.home() / "tmp" / "rc_libero_demo")
    ap.add_argument("--api-key",  default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (or set OPENAI_API_KEY)")
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                    help="API base URL (or set OPENAI_API_BASE)")
    ap.add_argument("--model",    default=os.environ.get("OPENAI_MODEL", "claude-opus-4-5"),
                    help="Model name the endpoint accepts (or set OPENAI_MODEL)")
    args = ap.parse_args()

    if not args.api_key:
        print("[error] No API key. Pass --api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    asyncio.run(run_demo(args.task, args.workspace, args.api_key, args.api_base, args.model))


if __name__ == "__main__":
    main()
