"""Tests for MujocoControlRuntime — the bridge between MuJoCo and ROS2 control surface."""

import importlib
from types import SimpleNamespace

import pytest

from roboclaw.embodied.simulation.mujoco_control_runtime import MujocoControlRuntime


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
            self.qpos = [0.0, 0.0]
            self.qvel = [0.0, 0.0]
            self.ctrl = [0.0, 0.0]

    names = {0: "shoulder", 1: "gripper"}
    actuators = {"shoulder": 0, "gripper": 1}
    return SimpleNamespace(
        MjModel=_Model,
        MjData=_Data,
        mjtObj=SimpleNamespace(mjOBJ_JOINT=1, mjOBJ_ACTUATOR=2),
        mj_step=lambda model, data: None,
        mj_id2name=lambda model, obj, idx: names[idx],
        mj_name2id=lambda model, obj, name: actuators.get(name, -1),
    )


def _patch_mujoco(monkeypatch, fake):
    original = importlib.import_module

    def _import(name, package=None):
        if name == "mujoco":
            return fake
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", _import)


def test_connect_disconnect(monkeypatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")

    rt = MujocoControlRuntime(model_path=str(model))
    assert rt.connected is False
    rt.connect()
    assert rt.connected is True
    rt.disconnect()
    assert rt.connected is False


def test_open_close_gripper(monkeypatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")

    rt = MujocoControlRuntime(model_path=str(model))
    rt.connect()

    result = rt.open_gripper()
    assert result["connected"] is True
    assert "gripper_percent" in result

    result = rt.close_gripper()
    assert result["connected"] is True


def test_go_home(monkeypatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")

    rt = MujocoControlRuntime(model_path=str(model))
    rt.connect()
    result = rt.go_home()
    assert result["connected"] is True
    assert result["simulator"] == "mujoco"


def test_snapshot(monkeypatch, tmp_path):
    _patch_mujoco(monkeypatch, _fake_mujoco())
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")

    rt = MujocoControlRuntime(model_path=str(model))
    rt.connect()
    snap = rt.snapshot()
    assert snap["connected"] is True
    assert "joint_positions" in snap
    assert snap["simulator"] == "mujoco"
