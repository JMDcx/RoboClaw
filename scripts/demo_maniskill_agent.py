#!/usr/bin/env python3
"""Interactive ManiSkill big-brain / small-brain agent demo.

Big brain  = LLM (GLM or rule-based) — understands natural language, decides
             which policy to invoke, stores user preferences in memory.
Small brain = trained diffusion policy checkpoint — executes the actual
              manipulation inside ManiSkill simulation.

Tested policy registry (success_at_end):
  pick_bowl   -> ReplicaCADLiftBowl-v1   (1.0)
  pick_cup02  -> ReplicaCADLiftCup-v1    (1.0)
  pick_cup03  -> ReplicaCADLiftCup03-v1  (0.6)
  push_plate01-> ReplicaCADPushPlate01-v1(0.33, demo only)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANISKILL_DIR = Path("/home/xinyuan/ManiSkill")
DP_DIR = MANISKILL_DIR / "examples" / "baselines" / "diffusion_policy"

# Make the upstream diffusion_policy package importable.
for _p in (str(MANISKILL_DIR), str(DP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

POLICIES: dict[str, dict[str, Any]] = {
    "pick_bowl": {
        "env_id": "ReplicaCADLiftBowl-v1",
        "checkpoint": str(DP_DIR / "runs/mid-ReplicaCADLiftBowl-joint-delta/checkpoints/best_eval_success_at_end.pt"),
        "max_episode_steps": 400,
        "success_at_end": 1.0,
        "description": "Pick up the bowl",
    },
    "place_bowl_sink": {
        "env_id": "ReplicaCADPlaceBowlSink-v1",
        "checkpoint": str(DP_DIR / "runs/pipeline-ReplicaCADPlaceBowlSink-joint-delta/checkpoints/final.pt"),
        "max_episode_steps": 250,
        "success_at_end": 1.0,
        "description": "Place the bowl in the sink (must follow pick_bowl)",
    },
    "pick_cup02": {
        "env_id": "ReplicaCADLiftCup-v1",
        "checkpoint": str(DP_DIR / "runs/mid-ReplicaCADLiftCup-joint-delta/checkpoints/best_eval_success_at_end.pt"),
        "max_episode_steps": 400,
        "success_at_end": 1.0,
        "description": "Pick up cup02",
    },
    "pick_cup03": {
        "env_id": "ReplicaCADLiftCup03-v1",
        "checkpoint": str(DP_DIR / "runs/dp-ReplicaCADLiftCup03-joint-delta/checkpoints/best_eval_success_at_end.pt"),
        "max_episode_steps": 400,
        "success_at_end": 0.6,
        "description": "Pick up cup03",
    },
    "push_plate01": {
        "env_id": "ReplicaCADPushPlate01-v1",
        "checkpoint": str(DP_DIR / "runs/quick-ReplicaCADPushPlate01-joint-delta/checkpoints/best_eval_success_at_end.pt"),
        "max_episode_steps": 400,
        "success_at_end": 0.333,
        "description": "Push plate01 (weak, demo only)",
    },
}

# A scenario locks the session to one ManiSkill env and a list of policies that
# can be invoked inside it. Order matters: policies must be invoked in the
# listed sequence (handoff state from policy N feeds policy N+1).
SCENARIOS: dict[str, dict[str, Any]] = {
    "pick_bowl_alone": {
        "env_id": "ReplicaCADLiftBowl-v1",
        "policies": ["pick_bowl"],
        "description": "Pick the bowl, single skill.",
    },
    "pick_and_place_bowl_sink": {
        "env_id": "ReplicaCADPlaceBowlSink-v1",
        "policies": ["pick_bowl", "place_bowl_sink"],
        "description": "Pick the bowl, then place it in the sink.",
    },
}

# Maps canonical object names to which policy handles them.
# cup is ambiguous — big brain resolves cup02 vs cup03 via context/memory.
OBJECT_TO_POLICY: dict[str, str] = {
    "bowl": "pick_bowl",
    "cup02": "pick_cup02",
    "cup": "pick_cup02",
    "cup03": "pick_cup03",
    "plate": "push_plate01",
    "plate01": "push_plate01",
}

OBJECT_ALIASES: dict[str, str] = {
    "bowl": "bowl",
    "碗": "bowl",
    "碗具": "bowl",
    "cup": "cup",
    "cup02": "cup02",
    "cup03": "cup03",
    "杯子": "cup",
    "杯": "cup",
    "plate": "plate",
    "plate01": "plate01",
    "盘子": "plate",
    "碟子": "plate",
}

MANIPULATION_MARKERS = (
    "拿", "抓", "拾", "取", "拿起", "抓起", "举起",
    "推", "移", "挪",
    "pick", "grab", "grasp", "lift", "push", "move",
    "整理", "收拾",
)

TALK_MEMORY_MARKERS = (
    "以后", "记住", "我喜欢", "我希望", "偏好", "应该",
    "remember", "prefer", "always", "from now on",
)

EXECUTION_NOW_MARKERS = (
    "现在", "立刻", "马上", "现在执行", "现在帮我",
    "do it now", "execute now", "start now", "right now", "run it",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_dotenv_minimal(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_value(text: Any) -> str:
    return str(text or "").strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Memory paths
# ---------------------------------------------------------------------------

def user_dir(workspace: Path, user_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in user_id)
    return workspace / "memory" / "users" / (safe or "user")


def semantic_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "semantic.json"


def working_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "working.json"


def episodes_path(workspace: Path, user_id: str) -> Path:
    return user_dir(workspace, user_id) / "episodes.jsonl"


def load_json_file(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def save_json_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Semantic memory schema
# ---------------------------------------------------------------------------

def default_semantic_memory(user_id: str) -> dict[str, Any]:
    now = datetime.now(tz=timezone.utc).isoformat()
    return {
        "profile": {
            "user_id": user_id,
            "preferred_language": "zh",
            "first_seen_iso": now,
            "last_seen_iso": now,
        },
        "object_preferences": {
            # e.g. {"cup": "cup03"} means user defaults to cup03 when saying "杯子"
        },
        "grasp_history": {
            # policy_key -> {"attempts": int, "successes": int}
        },
        "schema_version": 1,
    }


def load_semantic_memory(workspace: Path, user_id: str) -> dict[str, Any]:
    path = semantic_path(workspace, user_id)
    data = load_json_file(path, default_semantic_memory(user_id))
    data.setdefault("profile", {})
    data["profile"].setdefault("user_id", user_id)
    data.setdefault("object_preferences", {})
    data.setdefault("grasp_history", {})
    data.setdefault("schema_version", 1)
    return data


def save_semantic_memory(workspace: Path, user_id: str, data: dict[str, Any]) -> None:
    save_json_file(semantic_path(workspace, user_id), data)


def write_working_memory(workspace: Path, user_id: str, payload: dict[str, Any]) -> None:
    save_json_file(working_path(workspace, user_id), payload)


def append_episode(workspace: Path, user_id: str, record: dict[str, Any]) -> None:
    path = episodes_path(workspace, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Object / intent detection
# ---------------------------------------------------------------------------

def detect_objects(text: str) -> list[str]:
    lower = text.lower().replace("-", "_")
    hits: list[str] = []
    for alias, canonical in sorted(OBJECT_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if alias.lower() in lower and canonical not in hits:
            hits.append(canonical)
    return hits


def is_manipulation_request(text: str) -> bool:
    lower = text.lower()
    has_now = any(m in lower for m in EXECUTION_NOW_MARKERS)
    has_memory = any(m in lower for m in TALK_MEMORY_MARKERS)
    has_action = any(m in lower for m in MANIPULATION_MARKERS)
    if has_memory and not has_now:
        return False
    return has_action or has_now


def resolve_cup_policy(semantic: dict[str, Any]) -> str:
    """When user says 'cup' ambiguously, prefer whatever they used last."""
    pref = semantic.get("object_preferences", {}).get("cup")
    if pref in ("cup02", "cup03"):
        return f"pick_{pref}"
    # default to cup02 (higher success rate)
    return "pick_cup02"


def map_object_to_policy(obj: str, semantic: dict[str, Any]) -> str | None:
    if obj == "cup":
        return resolve_cup_policy(semantic)
    return OBJECT_TO_POLICY.get(obj)


def build_rag_context(workspace: Path, user_id: str, user_message: str) -> dict[str, Any]:
    semantic = load_semantic_memory(workspace, user_id)
    working = load_json_file(working_path(workspace, user_id), {})
    objects = detect_objects(user_message)
    resolved_policies = []
    for obj in objects:
        pk = map_object_to_policy(obj, semantic)
        if pk:
            resolved_policies.append(pk)

    recent_episodes: list[dict[str, Any]] = []
    ep_path = episodes_path(workspace, user_id)
    if ep_path.exists():
        for line in ep_path.read_text(encoding="utf-8").splitlines()[-5:]:
            if line.strip():
                try:
                    recent_episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return {
        "detected_objects": objects,
        "resolved_policies": resolved_policies,
        "object_preferences": semantic.get("object_preferences", {}),
        "grasp_history": semantic.get("grasp_history", {}),
        "recent_episodes": recent_episodes,
        "working_memory": working,
        "available_policies": {k: v["description"] for k, v in POLICIES.items()},
    }


# ---------------------------------------------------------------------------
# Rule-based controller (no LLM needed)
# ---------------------------------------------------------------------------

def rule_controller_decision(user_message: str, rag_context: dict[str, Any]) -> dict[str, Any]:
    objects = rag_context.get("detected_objects") or []
    resolved = rag_context.get("resolved_policies") or []
    lower = user_message.lower()

    has_memory = any(m in lower for m in TALK_MEMORY_MARKERS)
    has_now = any(m in lower for m in EXECUTION_NOW_MARKERS)
    has_action = any(m in lower for m in MANIPULATION_MARKERS)

    if has_memory and not has_now and objects:
        # User is teaching a preference, not asking for execution
        semantic_updates = []
        if "cup03" in objects:
            semantic_updates.append({"object": "cup", "preferred_variant": "cup03"})
        elif "cup02" in objects or "cup" in objects:
            semantic_updates.append({"object": "cup", "preferred_variant": "cup02"})
        return {
            "route": "talk",
            "assistant_reply": f"已记住偏好：{objects}。",
            "semantic_updates": semantic_updates,
            "policy_key": None,
            "source": "rule_memory",
        }

    if resolved and (has_action or has_now):
        policy_key = resolved[0]
        policy = POLICIES[policy_key]
        return {
            "route": "manipulation",
            "assistant_reply": f"好，调用策略 [{policy_key}] —— {policy['description']}（成功率 {policy['success_at_end']:.0%}）。",
            "semantic_updates": [],
            "policy_key": policy_key,
            "source": "rule_manipulation",
        }

    if objects and not has_action:
        return {
            "route": "talk",
            "assistant_reply": f"检测到物体：{objects}。请说'拿起碗'或'现在执行'来触发操作。",
            "semantic_updates": [],
            "policy_key": None,
            "source": "rule_clarify",
        }

    return {
        "route": "talk",
        "assistant_reply": (
            "你可以说：\n"
            "  • 拿起碗 / pick bowl\n"
            "  • 拿杯子 / grab the cup\n"
            "  • 推盘子 / push plate\n"
            "  • 以后用 cup03 → 更新杯子偏好"
        ),
        "semantic_updates": [],
        "policy_key": None,
        "source": "rule_general_talk",
    }


# ---------------------------------------------------------------------------
# LLM big-brain controller
# ---------------------------------------------------------------------------

async def chat_json(
    *,
    api_key: str,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 1000,
    temperature: float = 0.0,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return {"_error": str(e)}
    parsed = extract_json_object(text)
    if not parsed:
        return {"_error": f"non_json: {text[:300]}"}
    return parsed


async def call_big_brain(
    *,
    user_message: str,
    rag_context: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
    scenario_policies: list[str],
    expected_next: str,
) -> dict[str, Any]:
    if not api_key:
        return rule_controller_decision(user_message, rag_context)

    policy_list = "\n".join(
        f'  "{k}": {POLICIES[k]["description"]} (success_at_end={POLICIES[k]["success_at_end"]:.0%})'
        for k in scenario_policies
    )
    order_hint = " -> ".join(scenario_policies)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the big-brain controller for a ManiSkill robot demo. "
                "This session is locked to ONE scenario. The available policies, in required "
                f"execution order, are: {order_hint}\n"
                "Policy details:\n"
                + policy_list + "\n\n"
                "Rules:\n"
                "1. If the user is teaching a preference (memory markers: 记住/以后/prefer/remember/always/"
                "from now on), return route=talk and fill semantic_updates. Do NOT execute.\n"
                "2. If the user gives a direct manipulation command, return route=manipulation and "
                "pick the policy_key from the list above. Stages must be invoked in the listed order; "
                f"the next stage expected right now is '{expected_next}'.\n"
                "3. Never invent policy_keys outside the list. If the user asks for something not in "
                "the list, return route=talk and explain.\n"
                "Return JSON only with keys: route, assistant_reply, semantic_updates, policy_key."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User message: {user_message}\n\n"
                f"RAG context:\n{json_dumps(rag_context)}\n\n"
                "Return one JSON object:\n"
                '{"route":"manipulation","assistant_reply":"...","semantic_updates":[],"policy_key":"<one_of_listed>"}\n'
                "or\n"
                '{"route":"talk","assistant_reply":"...","semantic_updates":[],"policy_key":null}'
            ),
        },
    ]
    parsed = await chat_json(
        api_key=api_key,
        api_base=api_base,
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.0,
    )
    if "_error" in parsed:
        fallback = rule_controller_decision(user_message, rag_context)
        fallback["big_brain_error"] = parsed["_error"]
        return fallback
    return _normalize_big_brain(parsed, user_message, rag_context, scenario_policies)


def _normalize_big_brain(
    parsed: dict[str, Any],
    user_message: str,
    rag_context: dict[str, Any],
    scenario_policies: list[str],
) -> dict[str, Any]:
    route = str(parsed.get("route") or "").strip().lower()
    if route not in {"talk", "manipulation"}:
        fallback = rule_controller_decision(user_message, rag_context)
        fallback["big_brain_bad_route"] = route
        return fallback
    parsed.setdefault("assistant_reply", "")
    parsed.setdefault("semantic_updates", [])
    if not isinstance(parsed["semantic_updates"], list):
        parsed["semantic_updates"] = []
    if route == "manipulation":
        pk = str(parsed.get("policy_key") or "").strip()
        if pk not in scenario_policies:
            fallback = rule_controller_decision(user_message, rag_context)
            fallback["big_brain_bad_policy"] = pk
            return fallback
    else:
        parsed["policy_key"] = None
    parsed.setdefault("source", "big_brain_llm")
    return parsed


# ---------------------------------------------------------------------------
# Semantic memory updates
# ---------------------------------------------------------------------------

def apply_semantic_updates(
    workspace: Path,
    user_id: str,
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    semantic = load_semantic_memory(workspace, user_id)
    applied: list[dict[str, Any]] = []
    prefs = semantic.setdefault("object_preferences", {})
    now = datetime.now(tz=timezone.utc).isoformat()

    for update in updates:
        if not isinstance(update, dict):
            continue
        obj = normalize_value(update.get("object"))
        variant = normalize_value(update.get("preferred_variant") or "")
        if obj == "cup" and variant in ("cup02", "cup03"):
            prefs["cup"] = variant
            applied.append({"object": obj, "preferred_variant": variant, "updated_at": now})

    if applied:
        save_semantic_memory(workspace, user_id, semantic)
    return applied


def record_grasp_result(
    workspace: Path,
    user_id: str,
    policy_key: str,
    success: bool,
) -> None:
    semantic = load_semantic_memory(workspace, user_id)
    hist = semantic.setdefault("grasp_history", {})
    entry = hist.setdefault(policy_key, {"attempts": 0, "successes": 0})
    entry["attempts"] += 1
    if success:
        entry["successes"] += 1
    save_semantic_memory(workspace, user_id, semantic)


# ---------------------------------------------------------------------------
# Small brain: persistent in-process diffusion-policy runner
# ---------------------------------------------------------------------------

OBS_HORIZON = 2
ACT_HORIZON = 8
PRED_HORIZON = 16


def _find_base_env(env: Any) -> Any:
    current = env
    while current is not None:
        if hasattr(current, "base_env"):
            return current.base_env
        current = getattr(current, "env", None)
    return None


class PersistentRunner:
    """Owns one ManiSkill env + N diffusion-policy agents for the session.

    The env is reset exactly once at startup. Each ``rollout`` call runs a
    chosen policy for up to its ``max_episode_steps`` step budget (or until
    the env truncates). State carries over between policies, which is how
    chained skills (pick -> place) get their handoff for free.
    """

    def __init__(
        self,
        scenario_key: str,
        sim_backend: str,
        save_video: str | None = None,
    ) -> None:
        import numpy as np
        import torch
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  (registers envs)
        from mani_skill.utils.wrappers import CPUGymWrapper, RecordEpisode
        from train import Agent, Args

        if sim_backend != "cpu":
            raise NotImplementedError("Chained scenarios only support sim_backend=cpu for now.")

        self._np = np
        self._torch = torch
        self._collections_deque = __import__("collections").deque

        scenario = SCENARIOS[scenario_key]
        self.scenario_key = scenario_key
        self.scenario = scenario
        self.policy_keys = list(scenario["policies"])
        self.env_id = scenario["env_id"]
        self.sim_backend = sim_backend
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sum of per-policy budgets + headroom so the env doesn't truncate
        # mid-chain. Each rollout enforces its own per-policy step cap.
        max_steps = sum(POLICIES[k]["max_episode_steps"] for k in self.policy_keys) + 50

        raw_env = gym.make(
            self.env_id,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            reconfiguration_freq=1,
        )
        wrapped = CPUGymWrapper(raw_env, ignore_terminations=True, record_metrics=True)
        if save_video:
            wrapped = RecordEpisode(
                wrapped,
                output_dir=save_video,
                save_trajectory=False,
                info_on_video=True,
                source_type="roboclaw_demo",
                source_desc="chained scenario rollout",
            )
        self.env = wrapped
        self.base_env = _find_base_env(self.env)

        agent_args = Args(
            env_id=self.env_id,
            control_mode="pd_joint_delta_pos",
            sim_backend=sim_backend,
            max_episode_steps=max_steps,
            obs_horizon=OBS_HORIZON,
            act_horizon=ACT_HORIZON,
            pred_horizon=PRED_HORIZON,
            capture_video=False,
            cuda=(self.device.type == "cuda"),
        )
        self._agent_args = agent_args

        # Agent expects an object with `single_observation_space` / `single_action_space`
        # of vectorized shape. Build a tiny shim around the raw env's spaces.
        obs_dim = int(np.prod(raw_env.observation_space.shape))
        act_dim = int(np.prod(raw_env.action_space.shape))
        shim = _AgentSpaceShim(obs_horizon=OBS_HORIZON, obs_dim=obs_dim, act_dim=act_dim)

        self.agents: dict[str, Any] = {}
        for k in self.policy_keys:
            agent = Agent(shim, agent_args).to(self.device)
            ckpt = torch.load(POLICIES[k]["checkpoint"], map_location=self.device)
            ckpt_key = "ema_agent" if "ema_agent" in ckpt else "agent"
            agent.load_state_dict(ckpt[ckpt_key])
            agent.eval()
            self.agents[k] = agent

        self.history = self._collections_deque(maxlen=OBS_HORIZON)
        self.obs, _ = self.env.reset()
        for _ in range(OBS_HORIZON):
            self.history.append(self.obs.copy())
        self.last_invoked: str | None = None

        if self.base_env is not None:
            self.base_env.render_human()

    def has_policy(self, policy_key: str) -> bool:
        return policy_key in self.agents

    def expected_next_policy(self) -> str:
        if self.last_invoked is None:
            return self.policy_keys[0]
        idx = self.policy_keys.index(self.last_invoked)
        return self.policy_keys[min(idx + 1, len(self.policy_keys) - 1)]

    def rollout(self, policy_key: str) -> dict[str, Any]:
        torch = self._torch
        np = self._np
        agent = self.agents[policy_key]
        max_steps = POLICIES[policy_key]["max_episode_steps"]

        # Seed history with the current observation so handoff frames are
        # treated as a steady state for the incoming policy.
        history = self._collections_deque(
            [self.obs.copy() for _ in range(OBS_HORIZON)], maxlen=OBS_HORIZON
        )

        info: dict[str, Any] = {}
        steps = 0
        truncated = False
        with torch.no_grad():
            while steps < max_steps:
                obs_seq = np.stack(history, axis=0)[None]  # (1, obs_horizon, obs_dim)
                obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
                action_seq = agent.get_action(obs_tensor).cpu().numpy()[0]
                for action in action_seq:
                    self.obs, _, _, trunc, info = self.env.step(action)
                    history.append(self.obs.copy())
                    steps += 1
                    if self.base_env is not None:
                        self.base_env.render_human()
                    if bool(np.asarray(trunc).item()):
                        truncated = True
                        break
                    if steps >= max_steps:
                        break
                if truncated:
                    break

        self.history = history
        self.last_invoked = policy_key

        episode = info.get("episode", {}) if isinstance(info, dict) else {}
        metrics = {k: round(float(np.asarray(v).item()), 4) for k, v in episode.items()}
        return {
            "steps": steps,
            "truncated": truncated,
            "metrics": metrics,
        }

    def pump_viewer(self) -> None:
        """Keep the SAPIEN viewer responsive while idle at the prompt."""
        if self.base_env is not None:
            self.base_env.render_human()


class _AgentSpaceShim:
    """Minimal stand-in for the vectorized env that ``Agent.__init__`` inspects."""

    def __init__(self, obs_horizon: int, obs_dim: int, act_dim: int) -> None:
        import numpy as np
        import gymnasium.spaces as spaces

        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_horizon, obs_dim), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )


# ---------------------------------------------------------------------------
# Per-turn processing
# ---------------------------------------------------------------------------

async def process_turn(
    args: argparse.Namespace,
    runner: PersistentRunner,
    user_message: str,
) -> None:
    workspace = args.workspace.expanduser().resolve()
    rag_context = build_rag_context(workspace, args.user_id, user_message)

    decision = await call_big_brain(
        user_message=user_message,
        rag_context=rag_context,
        api_key=args.glm_api_key,
        api_base=args.glm_api_base,
        model=args.glm_model,
        scenario_policies=runner.policy_keys,
        expected_next=runner.expected_next_policy(),
    )

    applied_updates = apply_semantic_updates(
        workspace,
        args.user_id,
        decision.get("semantic_updates") or [],
    )

    base_working = {
        "session_key": args.session_key,
        "last_user_message": user_message,
        "current_route": decision.get("route"),
        "retrieved_memory": rag_context,
        "big_brain_decision": decision,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    print("\n" + "=" * 72)
    print(f"User: {user_message}")
    print("\n[RAG context]")
    print(json_dumps(rag_context))
    print("\n[Big brain decision]")
    print(json_dumps(decision))
    if applied_updates:
        print("\n[Semantic updates applied]")
        print(json_dumps(applied_updates))

    if decision.get("route") != "manipulation" or not decision.get("policy_key"):
        write_working_memory(workspace, args.user_id, {
            **base_working,
            "current_skill": {"status": "no_action"},
        })
        print(f"\nAssistant: {decision.get('assistant_reply') or '收到。'}")
        print("=" * 72 + "\n")
        return

    policy_key = str(decision["policy_key"])
    if not runner.has_policy(policy_key):
        base_working["current_skill"] = {"status": "policy_mismatch", "requested": policy_key}
        write_working_memory(workspace, args.user_id, base_working)
        print(
            f"\n[Refused] scenario '{runner.scenario_key}' only allows {runner.policy_keys}; "
            f"requested '{policy_key}'."
        )
        print("Restart with --scenario to switch.")
        print("=" * 72 + "\n")
        return

    expected = runner.expected_next_policy()
    if policy_key != expected:
        print(
            f"\n[Note] expected next stage was '{expected}' but big-brain chose "
            f"'{policy_key}'. Running anyway — handoff may not be ready."
        )

    base_working["current_skill"] = {
        "policy_key": policy_key,
        "env_id": runner.env_id,
        "status": "running",
    }
    write_working_memory(workspace, args.user_id, base_working)

    print(f"\n[Small brain] policy: {policy_key}  env: {runner.env_id}")
    print(f"\nAssistant: {decision.get('assistant_reply') or '执行中...'}")

    rollout_result = runner.rollout(policy_key)
    metrics_summary = rollout_result["metrics"]
    print(
        f"\n[Rollout] steps={rollout_result['steps']}  truncated={rollout_result['truncated']}"
    )
    print("[Rollout metrics]")
    print(json_dumps(metrics_summary))

    result = ""
    while result not in {"success", "partial", "failed", "skip"}:
        result = input("Result? [success/partial/failed/skip]: ").strip().lower()

    if result != "skip":
        feedback = input("Feedback note (optional): ").strip()
        success = result == "success"
        record_grasp_result(workspace, args.user_id, policy_key, success)
        episode = {
            "episode_id": str(uuid.uuid4()),
            "user_id": args.user_id,
            "session_key": args.session_key,
            "user_message": user_message,
            "policy_key": policy_key,
            "env_id": runner.env_id,
            "result": result,
            "user_feedback": feedback,
            "metrics": metrics_summary,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        append_episode(workspace, args.user_id, episode)
        base_working["current_skill"]["status"] = result
        base_working["last_episode_id"] = episode["episode_id"]
        write_working_memory(workspace, args.user_id, base_working)
        print("\n[Episodic memory appended]")
        print(json_dumps(episode))
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

async def interactive_loop(args: argparse.Namespace, runner: PersistentRunner) -> None:
    workspace = args.workspace.expanduser().resolve()
    user_dir(workspace, args.user_id).mkdir(parents=True, exist_ok=True)
    save_semantic_memory(workspace, args.user_id, load_semantic_memory(workspace, args.user_id))

    print("\nManiSkill big-brain / small-brain agent demo")
    print(f"Workspace  : {workspace}")
    print(f"User       : {args.user_id}")
    print(f"LLM        : {args.glm_api_base} | {args.glm_model}")
    print(f"Scenario   : {runner.scenario_key}  env: {runner.env_id}")
    print(f"Sim backend: {args.sim_backend}  device: {runner.device}")
    print("\nPolicies in this scenario (execution order):")
    for k in runner.policy_keys:
        v = POLICIES[k]
        print(f"  - {k:<18} {v['description']}  (success={v['success_at_end']:.0%})")
    print("\nCommands: /quit  /memory  /working  /policies")
    print()

    while True:
        runner.pump_viewer()
        try:
            user_message = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_message:
            continue
        if user_message in {"/quit", "/exit"}:
            break
        if user_message == "/memory":
            workspace_r = args.workspace.expanduser().resolve()
            print(json_dumps(load_semantic_memory(workspace_r, args.user_id)))
            continue
        if user_message == "/working":
            workspace_r = args.workspace.expanduser().resolve()
            print(json_dumps(load_json_file(working_path(workspace_r, args.user_id), {})))
            continue
        if user_message == "/policies":
            print(json_dumps(POLICIES))
            continue
        await process_turn(args, runner, user_message)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    load_dotenv_minimal(Path(".env"))

    parser = argparse.ArgumentParser(description="ManiSkill big-brain / small-brain agent demo")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.home() / "tmp" / "rc_maniskill_agent",
    )
    parser.add_argument("--user-id", default=os.environ.get("ROBOCLAW_USER_ID", "user"))
    parser.add_argument("--session-key", default=f"maniskill-agent:{int(time.time())}")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Scenario (env + ordered policy list) for the whole session.",
    )
    parser.add_argument("--sim-backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--save-video", default=None, help="Directory to dump rollout videos (optional).")

    parser.add_argument("--glm-api-key", default=os.environ.get("GLM_API_KEY", ""))
    parser.add_argument("--glm-api-base", default=os.environ.get("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"))
    parser.add_argument("--glm-model", default=os.environ.get("GLM_MODEL", "glm-4-flash"))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = PersistentRunner(args.scenario, args.sim_backend, save_video=args.save_video)
    asyncio.run(interactive_loop(args, runner))


if __name__ == "__main__":
    main()
