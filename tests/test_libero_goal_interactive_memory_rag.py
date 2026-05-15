import importlib.util
import json
import sys
from pathlib import Path


def _load_demo_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "demo_libero_goal_interactive_memory_rag.py"
    spec = importlib.util.spec_from_file_location("demo_libero_goal_interactive_memory_rag", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_semantic_update_writes_libero_goal_preference(tmp_path: Path) -> None:
    mod = _load_demo_module()

    applied = mod.apply_semantic_updates(
        tmp_path,
        "user",
        [{"object": "bowl", "field": "preferred_target", "value": "stove", "confidence": 0.95}],
    )

    assert applied == [{"object": "bowl", "field": "preferred_target", "value": "stove", "confidence": 0.95}]
    semantic = json.loads((tmp_path / "memory" / "users" / "user" / "semantic.json").read_text(encoding="utf-8"))
    assert semantic["libero_goal"]["object_preferences"]["bowl"]["preferred_target"] == "stove"
    assert semantic["kitchen"]["preferred_placement"]["bowl"] == "stove"


def test_json_rag_retrieves_object_preference(tmp_path: Path) -> None:
    mod = _load_demo_module()
    mod.apply_semantic_updates(
        tmp_path,
        "user",
        [{"object": "bowl", "field": "preferred_target", "value": "stove", "confidence": 0.95}],
    )

    context = mod.build_rag_context(tmp_path, "user", "整理碗")

    assert context["detected_objects"] == ["bowl"]
    assert context["semantic_preferences"]["bowl"]["preferred_target"] == "stove"


def test_rule_controller_uses_memory_for_manipulation(tmp_path: Path) -> None:
    mod = _load_demo_module()
    mod.apply_semantic_updates(
        tmp_path,
        "user",
        [{"object": "bowl", "field": "preferred_target", "value": "plate", "confidence": 0.95}],
    )
    context = mod.build_rag_context(tmp_path, "user", "整理碗")

    decision = mod.rule_controller_decision("整理碗", context)

    assert decision["route"] == "manipulation"
    assert decision["manipulation_plan"]["target_object"] == "bowl"
    assert decision["manipulation_plan"]["desired_target"] == "plate"


def test_rule_controller_extracts_directional_chinese_preference(tmp_path: Path) -> None:
    mod = _load_demo_module()
    context = mod.build_rag_context(tmp_path, "user", "以后碗放在炉子上")

    decision = mod.rule_controller_decision("以后碗放在炉子上", context)

    assert decision["route"] == "talk"
    assert decision["semantic_updates"][0]["object"] == "bowl"
    assert decision["semantic_updates"][0]["value"] == "stove"


def test_memory_only_english_statement_overrides_manipulation_route(tmp_path: Path) -> None:
    mod = _load_demo_module()
    message = "put the bowl on the stove all the time when i call you to tidy up or move the bowl"
    context = mod.build_rag_context(tmp_path, "user", message)
    model_decision = {
        "route": "manipulation",
        "assistant_reply": "I will execute and remember.",
        "semantic_updates": [
            {"object": "bowl", "field": "preferred_target", "value": "stove", "confidence": 0.95}
        ],
        "manipulation_plan": {
            "target_object": "bowl",
            "desired_target": "stove",
            "memory_facts_used": ["preferred_target.bowl=stove"],
        },
        "source": "glm_main_controller",
    }

    decision = mod.enforce_memory_only_route(message, context, model_decision)

    assert decision["route"] == "talk"
    assert decision["manipulation_plan"] is None
    assert decision["route_override"] == "memory_only_statement_no_execute"


def test_subagent_prompt_validation_rejects_invalid_prompt() -> None:
    mod = _load_demo_module()

    result = mod.normalize_subagent_decision(
        {"selected_task_id": 1, "vla_prompt": "put the bowl somewhere else"},
        {"target_object": "unknown", "desired_target": "unknown"},
        allow_rule_fallback=False,
    )

    assert result["_error"] == "invalid_subagent_prompt"


def test_valid_plan_maps_to_supported_prompt() -> None:
    mod = _load_demo_module()

    mapped = mod.validate_manipulation_plan({"target_object": "wine_bottle", "desired_target": "rack"})

    assert mapped == (9, "put the wine bottle on the rack")
