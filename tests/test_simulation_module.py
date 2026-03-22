import importlib
from types import SimpleNamespace

import pytest

from roboclaw.embodied.builtins.model import BuiltinEmbodiment
from roboclaw.embodied.simulation import MujocoRuntime, SimulationLauncher


def _fake_mujoco():
    class _Model:
        njnt = 2
        jnt_qposadr = [0, 1]

        @classmethod
        def from_xml_path(cls, path):
            inst = cls()
            inst.path = path
            return inst

    class _Data:
        def __init__(self, model):
            self.qpos = [0.1, -0.2]
            self.qvel = [1.0, -1.0]
            self.ctrl = [0.0, 0.0]
            self.step_calls = 0

    names = {0: "joint_a", 1: "joint_b"}
    actuators = {"joint_a": 0, "joint_b": 1, "act_b": 1}

    def _step(model, data):
        data.step_calls += 1

    return SimpleNamespace(
        MjModel=_Model,
        MjData=_Data,
        mjtObj=SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_ACTUATOR=2),
        mj_step=_step,
        mj_id2name=lambda model, obj, idx: names[idx],
        mj_name2id=lambda model, obj, name: actuators.get(name, -1),
    )


def _patch_mujoco(monkeypatch: pytest.MonkeyPatch, module):
    original_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "mujoco":
            return module
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)


def test_mujoco_runtime_instantiation():
    runtime = MujocoRuntime("robot.xml")
    assert runtime.model_path == "robot.xml"
    assert runtime.joint_mapping == {}
    assert runtime.is_running is False


def test_mujoco_runtime_start_step_reset_and_stop(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")

    runtime = MujocoRuntime(str(model_path))
    runtime.start()

    assert runtime.is_running is True
    assert runtime.get_joint_positions() == {"joint_a": 0.1, "joint_b": -0.2}

    runtime.set_joint_targets({"joint_a": 0.25, "joint_b": -0.5})
    assert runtime._data.ctrl == [0.25, -0.5]

    runtime.step()
    assert runtime._data.step_calls == 1

    runtime.reset()
    assert runtime._data.qpos == [0.0, 0.0]
    assert runtime._data.qvel == [0.0, 0.0]
    assert runtime._data.ctrl == [0.0, 0.0]
    assert runtime._data.step_calls == 2

    runtime.stop()
    assert runtime.is_running is False


def test_mujoco_runtime_joint_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")

    runtime = MujocoRuntime(str(model_path), joint_mapping={"logical_b": "joint_b"})
    runtime.start()

    assert runtime.get_joint_positions() == {"joint_a": 0.1, "logical_b": -0.2}

    runtime.set_joint_targets({"logical_b": 0.5})
    assert runtime._data.ctrl == [0.0, 0.5]


def test_mujoco_runtime_start_without_mujoco(monkeypatch: pytest.MonkeyPatch, tmp_path):
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")
    original_import_module = importlib.import_module

    def _import_module(name, package=None):
        if name == "mujoco":
            raise ModuleNotFoundError()
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import_module)

    with pytest.raises(ModuleNotFoundError, match="mujoco"):
        MujocoRuntime(str(model_path)).start()


def test_simulation_launcher_launch_and_shutdown(monkeypatch: pytest.MonkeyPatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")

    launcher = SimulationLauncher(str(model_path), joint_mapping={"logical_b": "joint_b"})
    runtime = launcher.launch()

    assert isinstance(runtime, MujocoRuntime)
    assert runtime.is_running is True
    assert runtime.get_joint_positions() == {"joint_a": 0.1, "logical_b": -0.2}

    launcher.shutdown(runtime)
    assert runtime.is_running is False


def test_simulation_launcher_launch_missing_model(tmp_path):
    missing_model_path = tmp_path / "missing.xml"

    with pytest.raises(FileNotFoundError, match="Simulation model file not found"):
        SimulationLauncher(str(missing_model_path)).launch()


def test_simulation_launcher_from_builtin():
    builtin = BuiltinEmbodiment(
        id="test_embodiment",
        robot=object(),
        sim_model_path="robot.xml",
        sim_joint_mapping={"logical_b": "joint_b"},
    )

    launcher = SimulationLauncher.from_builtin(builtin)

    assert launcher.model_path == "robot.xml"
    assert launcher.joint_mapping == {"logical_b": "joint_b"}


def test_simulation_launcher_from_builtin_without_sim_model_path():
    builtin = BuiltinEmbodiment(id="test_embodiment", robot=object())

    with pytest.raises(ValueError, match="sim_model_path"):
        SimulationLauncher.from_builtin(builtin)
