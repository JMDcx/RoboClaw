"""PiperX built-in embodiment declaration."""

from roboclaw.embodied.builtins.model import BuiltinEmbodiment
from roboclaw.embodied.builtins.registry import register_builtin_embodiment
from roboclaw.embodied.definition.components.robots import PIPERX_ROBOT
from roboclaw.embodied.definition.foundation.schema import CapabilityFamily
from roboclaw.embodied.execution.integration.adapters.ros2.profiles import PrimitiveAliasSpec, PrimitiveServiceSpec, Ros2EmbodimentProfile
from roboclaw.embodied.execution.orchestration.skills import SkillSpec, SkillStep

PIPERX_ROS2_PROFILE = Ros2EmbodimentProfile(
    id="piperx_ros2_standard",
    robot_id="piperx",
    primitive_aliases=(
        PrimitiveAliasSpec("gripper_open", ("打开夹爪", "张开夹爪", "open gripper", "open the gripper")),
        PrimitiveAliasSpec("gripper_close", ("闭合夹爪", "关闭夹爪", "夹住", "close gripper", "close the gripper")),
        PrimitiveAliasSpec("go_named_pose", ("回到 home", "回到原点", "go home", "go to home"), {"name": "home"}),
    ),
    primitive_services=(
        PrimitiveServiceSpec("gripper_open", "primitive_gripper_open"),
        PrimitiveServiceSpec("gripper_close", "primitive_gripper_close"),
        PrimitiveServiceSpec("go_named_pose", "primitive_go_home"),
    ),
    notes=("Control-surface profile for a ROS2-backed PiperX setup.", "PiperX uses CAN bus on `can0` and ships factory calibrated."),
)

PIPERX_BUILTIN = BuiltinEmbodiment(
    id="piperx",
    robot=PIPERX_ROBOT,
    ros2_profile=PIPERX_ROS2_PROFILE,
    sim_model_path="roboclaw/embodied/simulation/models/piperx.urdf",
    onboarding_aliases=("piperx", "piper x", "agilex piperx"),
    skills=(
        SkillSpec(
            name="pick_and_place",
            description="Open, home, close, home, and release.",
            steps=(SkillStep("gripper_open"), SkillStep("go_named_pose", {"name": "home"}), SkillStep("gripper_close"), SkillStep("go_named_pose", {"name": "home"}), SkillStep("gripper_open")),
            required_capabilities=(CapabilityFamily.END_EFFECTOR, CapabilityFamily.NAMED_POSE),
        ),
        SkillSpec(
            name="reset_arm",
            description="Return to home and open the gripper.",
            steps=(SkillStep("go_named_pose", {"name": "home"}), SkillStep("gripper_open")),
            required_capabilities=(CapabilityFamily.NAMED_POSE,),
        ),
    ),
)

register_builtin_embodiment(PIPERX_BUILTIN)

__all__ = ["PIPERX_BUILTIN", "PIPERX_ROS2_PROFILE"]
