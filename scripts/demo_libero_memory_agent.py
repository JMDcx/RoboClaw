#!/usr/bin/env python3
"""Demo: LIBERO pipeline with personalized memory — task scope controlled by memory.

Two profiles demonstrate how memory determines *what* the robot does, not just *how*:

  white-only    Memory marks yellow_mug as hands-off. Agent plans and executes
                only skill_06 (white mug → right plate). Yellow mug is untouched.

  full-cleanup  No hands-off restriction. Agent plans skill_06 then skill_07
                (white mug first, then yellow mug). Both mugs are tidied.

The user goal is intentionally open-ended ("tidy up the mugs"). The demo first
learns explicit user memory, then gives the planner structured memory context.

Usage:
    # Defaults are read from .env if present:
    #   GLM_API_KEY / GLM_API_BASE / GLM_MODEL       -> slow main controller
    #   QWEN_API_KEY / QWEN_API_BASE / QWEN_MODEL    -> fast skill controller
    export ROBOCLAW_ENABLE_LIBERO=1

    conda run -n lerobot312 python scripts/demo_libero_memory_agent.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if os.environ.get("ROBOCLAW_ENABLE_LIBERO", "0") != "1":
    os.environ["ROBOCLAW_ENABLE_LIBERO"] = "1"

_USER_ID = "user"
_DEFAULT_TASK = (
    "Tidy up the mugs on the table using a two-level VLM architecture. First read the "
    "structured PlanningMemoryContext provided below and explicitly state which objects "
    "you are allowed to handle and which are hands-off. As the slow main agent, call "
    "libero_perception(reset=false, run_yolo=false) to observe the scene without local "
    "YOLO, then call libero_plan and pass "
    "the full structured context in the planning_memory_context argument. Only plan "
    "sub-goals for objects that are NOT in hands-off classes. Then hand the whole "
    "manipulation layer to the fast Qwen skill controller by calling "
    "libero_manipulation(action='execute_skill', use_cosmos_controller=true, "
    "plan_json=<full libero_plan JSON>, previous_summary=<your concise scene/plan/memory "
    "summary>, local_verify=false, "
    "cosmos_confidence_threshold=<configured threshold>, "
    "cosmos_allow_fallback=<configured fallback>, cosmos_chunk_steps=120, "
    "cosmos_max_decisions=30). Qwen should run a YOLO-free RGB perception loop at "
    "about 1Hz and use the previous summary to decide skill routing, mark completed "
    "sub-goals, and switch strategies internally. "
    "Do not call libero_verify or any YOLO/CV verifier in this demo. If uncertain, refresh "
    "RGB perception with run_yolo=false or report uncertainty. Do not call libero_skill, "
    "libero_observe, exec, or any direct low-level fallback tool. Never reset during the task."
)


def _load_dotenv_minimal(path: Path, *, override: bool = False) -> None:
    """Load KEY=value pairs without adding a python-dotenv dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def _seed_base_profile(workspace: Path, profile: str) -> Path:
    """Create a deterministic base semantic memory file for the demo user."""
    user_dir = workspace / "memory" / "users" / _USER_ID
    user_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=timezone.utc).isoformat()

    if profile == "white-only":
        display_name = "LIBERO demo user - white-only"
    else:
        display_name = "LIBERO demo user - full-cleanup"

    preferred_placement = {
        "white_mug": "right_plate",
        "yellow_mug": "left_plate",
    }
    stable_preferences = [
        {
            "preference_id": "pref_demo_verification_policy",
            "scope": "routing",
            "key": "verification_policy",
            "value": "trust Qwen RGB reasoning for skill switching",
            "source": "demo_seed",
            "confidence": 1.0,
            "evidence_count": 1,
            "last_updated_iso": now,
        },
        {
            "preference_id": "pref_demo_retry_policy",
            "scope": "routing",
            "key": "retry_policy",
            "value": "retry a failed mug skill at most once",
            "source": "demo_seed",
            "confidence": 1.0,
            "evidence_count": 1,
            "last_updated_iso": now,
        },
    ]

    payload = {
        "profile": {
            "sender_id": _USER_ID,
            "display_name": display_name,
            "preferred_language": "zh",
            "first_seen_iso": now,
            "last_seen_iso": now,
            "total_tasks_completed": 0,
            "total_tasks_attempted": 0,
        },
        "interaction": {
            "confirm_before_pick": False,
            "confirm_before_place": False,
            "report_each_step": True,
            "report_only_on_completion": False,
            "use_terse_replies": False,
            "preferred_arm_side": "auto",
            "allow_multi_object_sessions": True,
        },
        "kitchen": {
            "preferred_placement": preferred_placement,
            "tidy_dirty_dishes_to": "sink",
            "tidy_food_items_to": "counter",
            "tidy_trash_to": "trash_bin",
            "hands_off_object_classes": [],
            "fragile_object_classes": ["mug"],
            "known_container_labels": ["left_plate", "right_plate"],
        },
        "object_knowledge": {},
        "environment_map": {},
        "failure_patterns": [],
        "preferences": stable_preferences,
        "operational_rules": [],
        "schema_version": 2,
    }

    path = user_dir / "semantic.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _learn_demo_memory(workspace: Path, profile: str) -> dict:
    """Use memory intake to transform user language into semantic memory."""
    from roboclaw.agent.memory.manager import PersonalizedMemoryManager

    manager = PersonalizedMemoryManager(workspace)
    if profile == "white-only":
        learning_message = "以后不要碰 yellow mug，我只想整理 white mug。"
    else:
        learning_message = "以后 white mug 放到 right plate，yellow mug 放到 left plate。"

    intake = manager.ingest_user_message(
        user_id=_USER_ID,
        session_key=f"demo:libero-memory-learning:{profile}",
        user_message=learning_message,
        task_category="tidyup",
    )
    planning_context = manager.get_planning_context(
        user_id=_USER_ID,
        task_category="tidyup",
        scene_objects=["white_mug", "yellow_mug", "left_plate", "right_plate"],
    ).to_dict()
    return {
        "learning_message": learning_message,
        "intake_result": {
            "working_updates": [vars(x) for x in intake.working_updates],
            "episodic_events": [vars(x) for x in intake.episodic_events],
            "semantic_updates": [vars(x) for x in intake.semantic_updates],
            "ignored": intake.ignored,
        },
        "planning_context": planning_context,
    }


async def run_demo(
    task: str,
    workspace: Path,
    api_key: str,
    api_base: str,
    model: str,
    profile: str,
    skill_api_key: str,
    skill_api_base: str,
    skill_model: str,
    skill_confidence_threshold: float,
    skill_allow_fallback: bool,
) -> None:
    from roboclaw.agent.loop import AgentLoop
    from roboclaw.bus.queue import MessageBus
    from roboclaw.providers.base import GenerationSettings
    from roboclaw.providers.custom_provider import CustomProvider

    workspace.mkdir(parents=True, exist_ok=True)
    videos_dir = workspace / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    memory_path = _seed_base_profile(workspace, profile)
    memory_demo = _learn_demo_memory(workspace, profile)
    # The current LIBERO manipulation tool still exposes these compatibility
    # names, but the endpoint/model can be any OpenAI-compatible VLM.
    os.environ["ROBOCLAW_COSMOS_API_KEY"] = skill_api_key
    os.environ["ROBOCLAW_COSMOS_API_BASE"] = skill_api_base
    os.environ["ROBOCLAW_COSMOS_MODEL"] = skill_model
    router_policy = {
        "small_vlm_subagent": "qwen_skill_controller",
        "qwen_api_base": skill_api_base,
        "qwen_model": skill_model,
        "confidence_threshold": skill_confidence_threshold,
        "allow_fallback": skill_allow_fallback,
        "chunk_steps": 120,
        "max_decisions": 30,
        "instruction": (
            "Call libero_manipulation once with use_cosmos_controller=true and the full "
            "libero_plan JSON. Pass cosmos_confidence_threshold and cosmos_allow_fallback "
            "exactly as shown here, plus cosmos_chunk_steps=120 and cosmos_max_decisions=30. "
            "Do not do separate per-subgoal routing in the main agent. Do not call direct "
            "low-level fallback tools if Qwen fails; summarize the manipulation report."
        ),
    }
    task_with_memory = (
        f"{task}\n\nStructured PlanningMemoryContext JSON:\n"
        f"{json.dumps(memory_demo['planning_context'], ensure_ascii=False, indent=2)}"
        "\n\nFast Qwen Skill Controller Config JSON:\n"
        f"{json.dumps(router_policy, ensure_ascii=False, indent=2)}"
    )

    print(f"\n{'=' * 60}")
    print("LIBERO Memory Pipeline Demo")
    print(f"Endpoint : {api_base}")
    print(f"Model    : {model}")
    print(f"Workspace: {workspace}")
    print(f"Memory   : {memory_path}")
    print(f"Profile  : {profile}")
    print(f"Qwen     : {skill_api_base} | {skill_model}")
    print(f"Task     : {task[:120]}...")
    print(f"{'=' * 60}\n")

    print("Memory learning phase:")
    print(f"User says: {memory_demo['learning_message']}")
    print(json.dumps(memory_demo["intake_result"], ensure_ascii=False, indent=2)[:2200])
    print("\nStructured PlanningMemoryContext:")
    print(json.dumps(memory_demo["planning_context"], ensure_ascii=False, indent=2)[:2200])
    print("\nFast Qwen Skill Controller Config:")
    print(json.dumps(router_policy, ensure_ascii=False, indent=2))
    print("\nPersisted semantic memory:")
    print(memory_path.read_text(encoding="utf-8")[:1800])
    print("\n" + "=" * 60 + "\n")

    provider = CustomProvider(api_key=api_key, api_base=api_base, default_model=model)
    provider.generation = GenerationSettings(temperature=0.1, max_tokens=8192)

    agent = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=workspace,
        model=model,
        max_iterations=45,
        context_window_tokens=65536,
    )

    required = {
        "libero_perception",
        "libero_plan",
        "libero_manipulation",
    }
    missing = sorted(required.difference(agent.tools.tool_names))
    if missing:
        print(f"[error] Missing LIBERO pipeline tools: {missing}")
        sys.exit(1)
    for tool_name in list(agent.tools.tool_names):
        if tool_name not in required:
            agent.tools.unregister(tool_name)
    print(f"Demo tool surface: {agent.tools.tool_names}\n")

    from roboclaw.agent.tools.libero_skill import LiberoEnvManager

    mgr = LiberoEnvManager.get()
    mgr.start_recording()
    hands_off_summary = "yellow_mug" if profile == "white-only" else "none"
    mgr.set_memory_summary(f"profile={profile} | hands-off={hands_off_summary}")
    print("Video recording started.\n")

    async def on_progress(text: str, *, tool_hint: bool = False) -> None:
        prefix = "  [tool]" if tool_hint else "  >"
        print(f"{prefix} {text}", flush=True)

    session_key = f"demo:libero-memory:{profile}:{int(time.time())}"
    response = await agent.process_direct(
        content=task_with_memory,
        session_key=session_key,
        channel="cli",
        chat_id="libero-memory-demo",
        on_progress=on_progress,
    )

    print(f"\n{'=' * 60}")
    print("AGENT FINAL RESPONSE:")
    print(response)
    print(f"{'=' * 60}")

    mgr.stop_recording()
    ts = int(time.time())
    video_path = videos_dir / f"memory_pipeline_{profile}_{ts}.mp4"
    saved = mgr.save_video(video_path, fps=10)
    if saved:
        print(f"\nSession video saved -> {saved} ({len(mgr._session_frames)} frames)")
    else:
        print("\n[info] No frames recorded.")

    try:
        mgr.close()
    except Exception:
        pass


def main() -> None:
    repo_dotenv = Path(__file__).resolve().parents[1] / ".env"
    _load_dotenv_minimal(repo_dotenv, override=True)
    _load_dotenv_minimal(Path(".env"))

    parser = argparse.ArgumentParser(description="RoboClaw LIBERO memory pipeline demo")
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "tmp" / "rc_libero_memory_demo")
    parser.add_argument("--profile", choices=("white-only", "full-cleanup"), default="white-only")
    parser.add_argument("--api-key", default=os.environ.get("GLM_API_KEY", ""))
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"))
    parser.add_argument("--model", default=os.environ.get("GLM_MODEL", "glm-5v-turbo"))
    parser.add_argument(
        "--qwen-api-key",
        "--cosmos-api-key",
        dest="qwen_api_key",
        default=os.environ.get("QWEN_API_KEY", os.environ.get("ROBOCLAW_COSMOS_API_KEY", "")),
        help="API key for the fast Qwen skill controller.",
    )
    parser.add_argument(
        "--qwen-api-base",
        "--cosmos-api-base",
        dest="qwen_api_base",
        default=os.environ.get("QWEN_API_BASE", os.environ.get("ROBOCLAW_COSMOS_API_BASE", "https://api.openai.com/v1")),
        help="OpenAI-compatible base URL for the fast Qwen skill controller.",
    )
    parser.add_argument(
        "--qwen-model",
        "--cosmos-model",
        dest="qwen_model",
        default=os.environ.get("QWEN_MODEL", os.environ.get("ROBOCLAW_COSMOS_MODEL", "qwen3.6-27b")),
        help="Model id for the fast Qwen skill controller.",
    )
    parser.add_argument(
        "--qwen-confidence-threshold",
        "--cosmos-confidence-threshold",
        dest="qwen_confidence_threshold",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--qwen-allow-fallback",
        "--cosmos-allow-fallback",
        dest="qwen_allow_fallback",
        action="store_true",
        help="Allow the skill controller to fall back to the main planner skill if Qwen routing fails.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("[error] No GLM API key. Pass --api-key or set GLM_API_KEY in .env.")
        sys.exit(1)
    if not args.qwen_api_key:
        print("[error] No Qwen API key. Pass --qwen-api-key or set QWEN_API_KEY in .env.")
        sys.exit(1)

    asyncio.run(run_demo(
        args.task,
        args.workspace,
        args.api_key,
        args.api_base,
        args.model,
        args.profile,
        args.qwen_api_key,
        args.qwen_api_base,
        args.qwen_model,
        args.qwen_confidence_threshold,
        args.qwen_allow_fallback,
    ))


if __name__ == "__main__":
    main()
