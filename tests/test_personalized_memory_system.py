import json
from pathlib import Path

from roboclaw.agent.memory.manager import PersonalizedMemoryManager
from roboclaw.agent.memory.semantic import SemanticStore
from roboclaw.agent.memory.working import TaskWorkingMemory


def test_memory_intake_classifies_hands_off_rule(tmp_path: Path) -> None:
    manager = PersonalizedMemoryManager(tmp_path)

    result = manager.ingest_user_message(
        user_id="user",
        session_key="s1",
        user_message="以后不要碰 yellow mug",
        task_category="tidyup",
    )

    assert any(
        update.update_type == "operational_rule"
        and update.key == "hands_off"
        and update.target == "yellow_mug"
        for update in result.semantic_updates
    )

    memory = SemanticStore(tmp_path / "memory" / "users" / "user").load("user")
    assert "yellow_mug" in memory.kitchen.hands_off_object_classes
    assert any(
        rule.rule_type == "hands_off" and rule.target == "yellow_mug"
        for rule in memory.operational_rules
    )


def test_memory_intake_scopes_hands_off_to_negated_clause(tmp_path: Path) -> None:
    manager = PersonalizedMemoryManager(tmp_path)

    result = manager.ingest_user_message(
        user_id="user",
        session_key="s1",
        user_message="以后不要碰 yellow mug，我只想整理 white mug。",
        task_category="tidyup",
    )

    hands_off_targets = {
        update.target
        for update in result.semantic_updates
        if update.update_type == "operational_rule" and update.key == "hands_off"
    }
    assert hands_off_targets == {"yellow_mug"}

    memory = SemanticStore(tmp_path / "memory" / "users" / "user").load("user")
    assert "yellow_mug" in memory.kitchen.hands_off_object_classes
    assert "white_mug" not in memory.kitchen.hands_off_object_classes


def test_memory_intake_classifies_preferred_placement(tmp_path: Path) -> None:
    manager = PersonalizedMemoryManager(tmp_path)

    result = manager.ingest_user_message(
        user_id="user",
        session_key="s1",
        user_message="以后 white mug 放到 right plate",
        task_category="tidyup",
    )

    assert any(
        update.update_type == "preference"
        and update.key == "preferred_placement.white_mug"
        and update.value == "right_plate"
        for update in result.semantic_updates
    )

    memory = SemanticStore(tmp_path / "memory" / "users" / "user").load("user")
    assert memory.kitchen.preferred_placement["white_mug"] == "right_plate"
    assert any(
        rule.rule_type == "placement"
        and rule.target == "white_mug"
        and rule.value == "right_plate"
        for rule in memory.operational_rules
    )


def test_semantic_store_migrates_v1_to_v2(tmp_path: Path) -> None:
    user_dir = tmp_path / "memory" / "users" / "user"
    user_dir.mkdir(parents=True)
    (user_dir / "semantic.json").write_text(
        json.dumps({
            "profile": {"sender_id": "user"},
            "interaction": {},
            "kitchen": {"hands_off_object_classes": ["yellow_mug"]},
            "schema_version": 1,
        }),
        encoding="utf-8",
    )

    store = SemanticStore(user_dir)
    memory = store.load("user")

    assert memory.schema_version == 2
    assert memory.preferences == []
    assert memory.operational_rules == []

    store.save(memory)
    saved = json.loads((user_dir / "semantic.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["preferences"] == []
    assert saved["operational_rules"] == []


def test_retrieval_returns_planning_context(tmp_path: Path) -> None:
    manager = PersonalizedMemoryManager(tmp_path)
    manager.ingest_user_message(
        user_id="user",
        session_key="s1",
        user_message="以后不要碰 yellow mug",
        task_category="tidyup",
    )
    manager.ingest_user_message(
        user_id="user",
        session_key="s1",
        user_message="以后 white mug 放到 right plate",
        task_category="tidyup",
    )

    context = manager.get_planning_context(
        user_id="user",
        task_category="tidyup",
        scene_objects=["white_mug", "yellow_mug", "right_plate"],
    ).to_dict()

    assert "yellow_mug" in context["object_constraints"]["hands_off"]
    assert context["object_constraints"]["preferred_placements"]["white_mug"] == "right_plate"
    assert context["preferences"]["preferred_placement.white_mug"] == "right_plate"
    assert any("yellow_mug" in hint for hint in context["planner_hints"])


def test_episode_records_personalization_decision(tmp_path: Path) -> None:
    manager = PersonalizedMemoryManager(tmp_path)
    working = TaskWorkingMemory(
        task_id="task1",
        user_goal="tidy up the mugs",
        user_id="user",
        session_key="s1",
        retrieved_memory={"object_constraints": {"hands_off": ["yellow_mug"]}},
        active_plan={"ordered_subgoals": [{"subgoal_id": "white_mug_to_right_plate"}]},
        skill_calls=[{"skill_id": "skill_06", "subgoal_id": "white_mug_to_right_plate"}],
        planner_decision_trace=[{
            "decision": "exclude_subgoal",
            "subgoal_id": "yellow_mug_to_left_plate",
            "memory_source": "operational_rule:hands_off:yellow_mug",
        }],
    )

    manager.end_task(working, outcome="success", task_category="tidyup")

    ep_path = tmp_path / "memory" / "users" / "user" / "episodes.jsonl"
    record = json.loads(ep_path.read_text(encoding="utf-8").strip())
    assert record["retrieved_memory_summary"] == "hands_off=yellow_mug"
    assert record["planner_output"]["ordered_subgoals"][0]["subgoal_id"] == "white_mug_to_right_plate"
    assert record["skill_sequence"][0]["skill_id"] == "skill_06"
    assert record["personalization_decisions"][0]["decision"] == "exclude_subgoal"
