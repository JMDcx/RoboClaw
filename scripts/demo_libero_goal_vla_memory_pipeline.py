#!/usr/bin/env python3
"""Demo: memory-aware LIBERO-goal prompt rewrite for a StarVLA policy.

Pipeline:
  1. Learn or load personalized memory, for example "bowl goes on the stove".
  2. Ask a cloud OpenAI-compatible small model to choose one supported LIBERO-goal
     skill prompt from memory plus the user's natural command.
  3. Pass the rewritten prompt to StarVLA's interactive_prompt_libero.py.

By default this script dry-runs the StarVLA launch and prints the exact command.
Pass --execute-vla to run the rollout.

Example:
    export OPENAI_API_KEY=sk-...
    export OPENAI_API_BASE=https://api.openai.com/v1
    export OPENAI_MODEL=gpt-4.1-mini

    python scripts/demo_libero_goal_vla_memory_pipeline.py \
        --user-goal "tidy up the bowl" \
        --learn-memory "bowl goes on the stove"

    python scripts/demo_libero_goal_vla_memory_pipeline.py \
        --user-goal "tidy up the bowl" \
        --execute-vla
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIBERO_GOAL_PROMPTS: dict[int, str] = {
    0: "open the middle drawer of the cabinet",
    1: "put the bowl on the stove",
    2: "put the wine bottle on top of the cabinet",
    3: "open the top drawer and put the bowl inside",
    4: "put the bowl on top of the cabinet",
    5: "push the plate to the front of the stove",
    6: "put the cream cheese in the bowl",
    7: "turn on the stove",
    8: "put the bowl on the plate",
    9: "put the wine bottle on the rack",
}

LIBERO_GOAL_OBJECTS = [
    "bowl",
    "plate",
    "stove",
    "cabinet",
    "top_of_cabinet",
    "middle_drawer",
    "top_drawer",
    "wine_bottle",
    "rack",
    "cream_cheese",
]

_BOWL_TARGET_TO_TASK = {
    "stove": 1,
    "top_of_cabinet": 4,
    "cabinet": 4,
    "plate": 8,
    "top_drawer": 3,
    "drawer": 3,
}


@dataclass
class PlannerDecision:
    selected_task_id: int
    vla_prompt: str
    confidence: float
    reason: str
    source: str
    raw: dict[str, Any]


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _preferred_bowl_target(planning_context: dict[str, Any]) -> str:
    preferences = planning_context.get("preferences") or {}
    constraints = planning_context.get("object_constraints") or {}
    preferred = constraints.get("preferred_placements") or {}
    raw = (
        preferred.get("bowl")
        or preferences.get("preferred_placement.bowl")
        or preferences.get("bowl")
        or ""
    )
    return str(raw).strip().lower().replace(" ", "_").replace("-", "_")


def _deterministic_fallback(user_goal: str, planning_context: dict[str, Any]) -> PlannerDecision:
    goal = user_goal.lower()
    target = _preferred_bowl_target(planning_context)
    if ("bowl" in goal or "碗" in goal) and target in _BOWL_TARGET_TO_TASK:
        task_id = _BOWL_TARGET_TO_TASK[target]
        return PlannerDecision(
            selected_task_id=task_id,
            vla_prompt=LIBERO_GOAL_PROMPTS[task_id],
            confidence=0.82,
            reason=f"Fallback used memory preferred_placement.bowl={target}.",
            source="deterministic_memory_fallback",
            raw={"preferred_bowl_target": target},
        )

    for task_id, prompt in LIBERO_GOAL_PROMPTS.items():
        if prompt in goal:
            return PlannerDecision(
                selected_task_id=task_id,
                vla_prompt=prompt,
                confidence=0.8,
                reason="Fallback matched an exact supported LIBERO-goal prompt.",
                source="deterministic_exact_prompt_fallback",
                raw={},
            )

    keyword_routes = [
        (("wine", "bottle", "rack"), 9),
        (("wine", "bottle", "cabinet"), 2),
        (("cream", "cheese", "bowl"), 6),
        (("turn", "stove"), 7),
        (("plate", "front", "stove"), 5),
        (("middle", "drawer"), 0),
        (("top", "drawer", "bowl"), 3),
        (("bowl", "stove"), 1),
        (("bowl", "cabinet"), 4),
        (("bowl", "plate"), 8),
    ]
    for keywords, task_id in keyword_routes:
        if all(k in goal for k in keywords):
            return PlannerDecision(
                selected_task_id=task_id,
                vla_prompt=LIBERO_GOAL_PROMPTS[task_id],
                confidence=0.7,
                reason=f"Fallback matched keywords: {', '.join(keywords)}.",
                source="deterministic_keyword_fallback",
                raw={},
            )

    task_id = 1 if target == "stove" else 8
    return PlannerDecision(
        selected_task_id=task_id,
        vla_prompt=LIBERO_GOAL_PROMPTS[task_id],
        confidence=0.35,
        reason="Fallback could not fully disambiguate; chose the closest bowl placement prompt.",
        source="deterministic_low_confidence_fallback",
        raw={"preferred_bowl_target": target},
    )


async def _call_cloud_planner(
    *,
    user_goal: str,
    planning_context: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
) -> PlannerDecision:
    from roboclaw.providers.custom_provider import CustomProvider

    provider = CustomProvider(api_key=api_key, api_base=api_base, default_model=model)
    supported = "\n".join(f"{idx}: {prompt}" for idx, prompt in LIBERO_GOAL_PROMPTS.items())
    messages = [
        {
            "role": "system",
            "content": (
                "You are the prompt planner for a LIBERO-goal StarVLA policy. "
                "The VLA can only execute one of the listed prompts, exactly as written. "
                "Use personalized memory to resolve vague commands. For example, if the "
                "user says to tidy the bowl and memory says preferred_placement.bowl=stove, "
                "choose task 1. If memory says top_of_cabinet choose task 4; if plate choose "
                "task 8; if top_drawer choose task 3. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Supported LIBERO-goal prompts:\n"
                f"{supported}\n\n"
                f"User goal:\n{user_goal}\n\n"
                "PlanningMemoryContext JSON:\n"
                f"{_json_dumps(planning_context)}\n\n"
                "Return this JSON shape exactly:\n"
                "{\n"
                '  "selected_task_id": 1,\n'
                '  "vla_prompt": "put the bowl on the stove",\n'
                '  "memory_facts_used": ["preferred_placement.bowl=stove"],\n'
                '  "reason": "short explanation",\n'
                '  "confidence": 0.0\n'
                "}"
            ),
        },
    ]
    response = await provider.chat(messages, model=model, max_tokens=1024, temperature=0.0)
    raw_text = response.content or ""
    parsed = _extract_json_object(raw_text)
    try:
        task_id = int(parsed.get("selected_task_id"))
    except (TypeError, ValueError):
        fallback = _deterministic_fallback(user_goal, planning_context)
        fallback.raw = {"planner_error": raw_text, "fallback_raw": fallback.raw}
        return fallback

    if task_id not in LIBERO_GOAL_PROMPTS:
        fallback = _deterministic_fallback(user_goal, planning_context)
        fallback.raw = {"planner_invalid_task_id": parsed, "fallback_raw": fallback.raw}
        return fallback

    prompt = str(parsed.get("vla_prompt") or "").strip()
    if prompt != LIBERO_GOAL_PROMPTS[task_id]:
        prompt = LIBERO_GOAL_PROMPTS[task_id]
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return PlannerDecision(
        selected_task_id=task_id,
        vla_prompt=prompt,
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(parsed.get("reason") or ""),
        source="cloud_small_model",
        raw=parsed,
    )


def _build_vla_command(args: argparse.Namespace, decision: PlannerDecision, video_out_path: Path) -> tuple[list[str], dict[str, str]]:
    starvla_dir = args.starvla_dir.expanduser().resolve()
    libero_home = args.libero_home.expanduser().resolve()
    ckpt = args.ckpt.expanduser().resolve()
    python_bin = args.libero_python or sys.executable
    script = starvla_dir / "examples" / "LIBERO" / "eval_files" / "interactive_prompt_libero.py"

    env = os.environ.copy()
    env["LIBERO_HOME"] = str(libero_home)
    env["LIBERO_CONFIG_PATH"] = str(libero_home / "libero")
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["PYOPENGL_PLATFORM"] = env.get("PYOPENGL_PLATFORM", "egl")
    pythonpath = [str(libero_home), str(starvla_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    cmd = [
        python_bin,
        str(script),
        "--args.pretrained-path",
        str(ckpt),
        "--args.host",
        args.vla_host,
        "--args.port",
        str(args.vla_port),
        "--args.task-suite-name",
        args.task_suite_name,
        "--args.task-id",
        str(decision.selected_task_id),
        "--args.init-state-id",
        str(args.init_state_id),
        "--args.rollout-steps",
        str(args.rollout_steps),
        "--args.video-out-path",
        str(video_out_path),
        "--args.prompt",
        decision.vla_prompt,
    ]
    return cmd, env


def _policy_server_command(args: argparse.Namespace) -> str:
    starvla_dir = args.starvla_dir.expanduser().resolve()
    script = starvla_dir / "examples" / "LIBERO" / "eval_files" / "run_policy_server.sh"
    env_parts = {
        "STARVLA_PYTHON": args.starvla_python,
        "CKPT": str(args.ckpt.expanduser().resolve()),
        "BASE_VLM": str(args.base_vlm.expanduser().resolve()),
        "GPU_ID": str(args.policy_gpu_id),
        "PORT": str(args.vla_port),
        "ATTN_IMPLEMENTATION": args.attn_implementation,
    }
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items() if value)
    return f"cd {shlex.quote(str(starvla_dir))} && {prefix} bash {shlex.quote(str(script))}"


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


async def run_pipeline(args: argparse.Namespace) -> int:
    from roboclaw.agent.memory.manager import PersonalizedMemoryManager

    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    manager = PersonalizedMemoryManager(workspace)
    if args.learn_memory.strip():
        intake = manager.ingest_user_message(
            user_id=args.user_id,
            session_key=f"demo:libero-goal-vla-memory-learning:{int(time.time())}",
            user_message=args.learn_memory,
            task_category="libero_goal",
            current_scene={"objects": [{"object_id": obj} for obj in LIBERO_GOAL_OBJECTS]},
        )
    else:
        intake = None

    planning_context = manager.get_planning_context(
        user_id=args.user_id,
        task_category="libero_goal",
        scene_objects=LIBERO_GOAL_OBJECTS,
    ).to_dict()

    decision = await _call_cloud_planner(
        user_goal=args.user_goal,
        planning_context=planning_context,
        api_key=args.planner_api_key,
        api_base=args.planner_api_base,
        model=args.planner_model,
    )

    video_out_path = args.video_out_path.expanduser().resolve() if args.video_out_path else (
        workspace / "vla_rollouts" / args.task_suite_name
    )
    video_out_path.mkdir(parents=True, exist_ok=True)
    cmd, env = _build_vla_command(args, decision, video_out_path)

    print("\n" + "=" * 70)
    print("LIBERO-goal memory -> VLA prompt pipeline")
    print(f"Workspace     : {workspace}")
    print(f"Planner       : {args.planner_api_base} | {args.planner_model}")
    print(f"User goal     : {args.user_goal}")
    if intake is not None:
        print(f"Learn memory  : {args.learn_memory}")
        print("Semantic updates:")
        print(_json_dumps([vars(x) for x in intake.semantic_updates]))
    print("\nPlanningMemoryContext:")
    print(_json_dumps(planning_context))
    print("\nPlanner decision:")
    print(_json_dumps({
        "selected_task_id": decision.selected_task_id,
        "vla_prompt": decision.vla_prompt,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "source": decision.source,
        "raw": decision.raw,
    }))
    print("\nStart policy server first, if it is not already running:")
    print(_policy_server_command(args))
    print("\nStarVLA command:")
    print(_shell_join(cmd))
    print("=" * 70 + "\n")

    if not args.execute_vla:
        print("Dry run only. Pass --execute-vla to launch StarVLA with this rewritten prompt.")
        return 0

    completed = subprocess.run(cmd, cwd=args.starvla_dir, env=env, check=False)
    return int(completed.returncode)


def main() -> None:
    default_starvla = Path(os.environ.get("STARVLA_DIR", "/home/xinyuan/starVLA"))
    default_ckpt = default_starvla / "playground" / "Pretrained_models" / "StarVLA" / "Qwen3-VL-OFT-LIBERO-4in1" / "checkpoints" / "steps_50000_pytorch_model.pt"
    default_base_vlm = default_starvla / "playground" / "Pretrained_models" / "Qwen3-VL-4B-Instruct"

    parser = argparse.ArgumentParser(description="Memory-aware LIBERO-goal VLA prompt pipeline")
    parser.add_argument("--user-goal", default="tidy up the bowl")
    parser.add_argument(
        "--learn-memory",
        default="bowl goes on the stove",
        help="Memory sentence to ingest before planning. Pass an empty string to use existing memory only.",
    )
    parser.add_argument("--user-id", default="user")
    parser.add_argument("--workspace", type=Path, default=Path.home() / "tmp" / "rc_libero_goal_vla_memory_demo")

    parser.add_argument("--planner-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--planner-api-base", default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"))
    parser.add_argument("--planner-model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))

    parser.add_argument("--starvla-dir", type=Path, default=default_starvla)
    parser.add_argument("--starvla-python", default=os.environ.get("STARVLA_PYTHON", sys.executable))
    parser.add_argument("--libero-home", type=Path, default=Path(os.environ.get("LIBERO_HOME", "/home/xinyuan/lerobot/lerobot-libero")))
    parser.add_argument("--libero-python", default=os.environ.get("LIBERO_Python", sys.executable))
    parser.add_argument("--ckpt", type=Path, default=Path(os.environ.get("CKPT", str(default_ckpt))))
    parser.add_argument("--base-vlm", type=Path, default=Path(os.environ.get("BASE_VLM", str(default_base_vlm))))
    parser.add_argument("--policy-gpu-id", default=os.environ.get("GPU_ID", "0"))
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
    parser.add_argument("--vla-host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--vla-port", type=int, default=int(os.environ.get("PORT", "6694")))
    parser.add_argument("--task-suite-name", default=os.environ.get("TASK_SUITE_NAME", "libero_goal"))
    parser.add_argument("--init-state-id", type=int, default=int(os.environ.get("INIT_STATE_ID", "0")))
    parser.add_argument("--rollout-steps", type=int, default=int(os.environ.get("ROLLOUT_STEPS", "150")))
    parser.add_argument("--video-out-path", type=Path, default=None)
    parser.add_argument("--execute-vla", action="store_true")
    args = parser.parse_args()

    if not args.planner_api_key:
        print("[error] No planner API key. Pass --planner-api-key or set OPENAI_API_KEY.")
        sys.exit(1)

    raise SystemExit(asyncio.run(run_pipeline(args)))


if __name__ == "__main__":
    main()
