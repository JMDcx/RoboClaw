---
name: libero_manipulation
description: Schedule learned imitation-learning manipulation skills in the LIBERO simulator.
metadata: {"roboclaw":{"emoji":"🦾","requires":{"env":["ROBOCLAW_ENABLE_LIBERO"]}}}
---

# LIBERO Manipulation (Perception-Plan-Manipulation Pipeline)

You can drive a robot in the LIBERO simulator by calling **learned ACT skills**.
Each skill performs **one** atomic manipulation primitive — they do not understand
language, they just execute a motor program.

## Available skills (LIBERO task 4: two mugs / two plates)

| skill_id | what it does |
|---|---|
| `skill_06` | Pick up the **white** mug and place it on the **right** plate. |
| `skill_07` | Pick up the **yellow** mug and place it on the **left** plate. |

When the user asks for a multi-step task, decompose it into a sequence of
single-skill calls.

## How to drive a task

Prefer the two-level LIBERO pipeline:

1. Start every task with **`libero_perception(action="analyze_scene", reset=false)`**
   to obtain semantic scene JSON plus an RGB image. The tool will initialize
   a fresh episode only if no LIBERO episode exists yet. The response includes
   the task-aware `objects` table. For Cosmos-controlled demos, pass
   `run_yolo=false`; do not use local YOLO detections.
2. Call **`libero_plan(action="plan_task", user_goal=..., perception_json=..., memory_preferences=...)`**
   to produce ordered sub-goals and the next skill action. If Robotic Memory
   contains execution order, placement, verification, retry, fragile, or hands-off
   preferences, pass the relevant memory lines into `memory_preferences`.
3. Call **`libero_manipulation(action="execute_skill", use_cosmos_controller=true, plan_json=<full libero_plan JSON>, previous_summary=..., local_verify=false, cosmos_chunk_steps=10, cosmos_max_decisions=30)`**.
   This hands the manipulation layer to the fast local Cosmos-Reason2 subagent.
   The tool renders live RGB frames, sends base64 data URLs to the local
   OpenAI-compatible vLLM endpoint, lets Cosmos choose `skill_06`, `skill_07`,
   `wait`, `recover`, or `done`, executes the selected learned skill chunk for
   about 10 env steps, then asks Cosmos again with a YOLO-free LIBERO perception
   packet, the updated RGB image, and execution trace. This is approximately a
   1Hz perception/control loop. The main agent should not do separate per-subgoal
   routing.
4. If the Cosmos controller stops with `wait`, `recover`, low confidence, or
   partial completion, refresh perception or report the uncertainty instead of
   manually choosing another skill.
5. Do not call **`libero_verify`** in Cosmos-controlled demos. That tool uses
   local YOLO/CV. If uncertain, call `libero_perception(run_yolo=false)` for a
   fresh RGB frame or report partial/uncertain status.
6. Retry the same skill at most once, and only when the mug is visibly not moved
   or the scene is clearly unchanged/corrupted. To retry, explicitly pass
   `allow_retry=true`; otherwise duplicate sub-goal executions are skipped.
   Never reset for a retry.
7. When all sub-goals are satisfied, summarize the perception, plan,
   manipulation, and verification outcomes to the user.

## Important rules

- The simulation **state persists** across LIBERO calls. Do not request reset
  during the task; each reset loses all completed progress. Pipeline tools
  ignore reset requests once an episode already exists.
- Each skill is short (~120 steps); `max_steps=150` is a safe default.
- Do **not** call `perception`, `executor`, or `embodied` tools — those belong
  to a different (G1 / real-robot) pipeline. Use only `libero_perception`,
  `libero_plan`, `libero_manipulation`, `libero_observe`, and `libero_skill`
  for Cosmos-controlled LIBERO tasks.
- The intended control structure is: the main LLM/VLM agent plans at low
  frequency, then `libero_manipulation(use_cosmos_controller=true)` uses local
  Cosmos-Reason2 for fast 1Hz scene understanding, skill routing, completion
  detection, and strategy switching. Local YOLO/CV verification is disabled; the
  main LLM is consulted only when Cosmos reports uncertainty or partial completion.
- Trust your eyes more than the `success` flag — `success=true` from the env
  signals task-level reward, not sub-goal completion. Always inspect the
  image before deciding the next step.
- Do not rely on `detected_objects` in Cosmos-controlled demos; keep
  `run_yolo=false` and use RGB plus task-aware target relations.
- Treat `status="needs_visual_verification"` as pending, not failed. This
  usually means the low-level reward stayed at zero; the attached image is the
  evidence for sub-goal completion.
- **Hard stop rule**: execute each planned sub-goal once by default. If visual
  verification clearly shows failure, retry once without reset. If visual
  verification shows success, proceed immediately to the next sub-goal.
- **Do NOT reset mid-task**. If the robot arm has visibly collided or the scene
  is clearly corrupted, report that the episode is corrupted instead of
  resetting automatically.

## Example interaction

User: "Put the white mug on the right plate and the yellow mug on the left."

Steps you take:
1. `libero_perception(action="analyze_scene", reset=false)` → semantic scene + RGB image
2. `libero_plan(action="plan_task", user_goal=..., perception_json=...)` → ordered sub-goals
3. `libero_manipulation(action="execute_skill", use_cosmos_controller=true, plan_json=..., previous_summary="white mug -> right plate, yellow mug -> left plate; no hands-off objects", local_verify=false, cosmos_chunk_steps=10, cosmos_max_decisions=30)`
4. If uncertain, call `libero_perception(action="analyze_scene", reset=false, run_yolo=false)` or report partial status.
5. Summarize: "White mug placed on right plate ✓. Yellow mug placed on left plate ✓/✗."
