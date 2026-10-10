"""四处同步回复守卫：异常首包立即失败，正常首包只发送一次请求。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robomme_ood_eval.models import framesamp_modul as FM


class _Client:
    """只记录请求与关闭次数，不连接服务端、不创建环境。"""

    def __init__(self, flag=None, reply=None):
        self.flag = flag
        self.reply = reply
        self.calls = []
        self.closed = 0
        self._ws = SimpleNamespace(close=self._close)

    def _close(self):
        self.closed += 1

    def reset(self):
        self.calls.append("reset")
        return self.reply if self.flag == "reset_finished" else {"reset_finished": True}

    def add_buffer(self, buffer):
        self.calls.append("add_buffer")
        assert buffer["add_buffer"] is True
        return self.reply if self.flag == "add_buffer_finished" else {"add_buffer_finished": True}

    def infer(self, observation):
        self.calls.append("infer")
        return {"actions": np.zeros((20, 8), np.float32)}


def _pre_traj():
    image = np.zeros((2, 2, 3), np.uint8)
    return {"images": [image], "wrist_images": [image], "states": [np.zeros(8, np.float32)],
            "task_goal": "test"}


def _step(action):
    obs = {"front_rgb_list": [np.zeros((2, 2, 3), np.uint8)],
           "wrist_rgb_list": [np.zeros((2, 2, 3), np.uint8)],
           "joint_state_list": [np.zeros(7, np.float32)],
           "gripper_state_list": [np.zeros(2, np.float32)]}
    return obs, 0.0, True, False, {"status": "success"}


def _invoke(route, client):
    if route == "run_loop":
        return FM.run_loop(client, FM.EnvRunnerShim(_step), _pre_traj, FM._Progress())
    return FM.warmup_server("unused", 1, frames=1, hw=(2, 2), client_factory=lambda: client)


@pytest.fixture(autouse=True)
def _no_polling(monkeypatch):
    """旧循环一旦进入等待便使测试立即失败，避免挂起测试进程。"""
    def sleep(seconds):
        pytest.fail(f"同步回复不应轮询等待：{seconds}")

    monkeypatch.setattr(FM.time, "sleep", sleep)


def test_all_four_guards_and_normal_paths():
    """逐处覆盖缺标志、假标志、非字典，并检查正常请求顺序与预热关闭。"""
    raised = 0
    for route in ("run_loop", "warmup_server"):
        for flag in ("reset_finished", "add_buffer_finished"):
            for reply in ({"z": 1, "a": 2}, {flag: False}, ["invalid"]):
                client = _Client(flag, reply)
                with pytest.raises(FM.ProtocolError) as exc:
                    _invoke(route, client)
                assert flag in str(exc.value)
                assert isinstance(exc.value, RuntimeError)
                if isinstance(reply, dict):
                    assert f"keys={sorted(reply.keys())}" in str(exc.value)
                else:
                    assert "reply_type=list; expected dict" in str(exc.value)
                assert client.calls == (["reset"] if flag == "reset_finished" else ["reset", "add_buffer"])
                assert client.closed == int(route == "warmup_server")
                raised += 1

    normal = 0
    for route in ("run_loop", "warmup_server"):
        client = _Client()
        result = _invoke(route, client)
        if route == "run_loop":
            assert result == "success"
        else:
            assert result["actions_shape"] == [20, 8]
        assert client.calls == ["reset", "add_buffer", "infer"]
        assert client.closed == int(route == "warmup_server")
        normal += 1
    assert raised == 12 and normal == 2
    print(f"ADD_BUFFER_GUARD=PASS sites=4 raised={raised} normal={normal}")


@pytest.mark.parametrize("route", ["run_loop", "warmup_server"])
@pytest.mark.parametrize("flag", ["reset_finished", "add_buffer_finished"])
@pytest.mark.parametrize("reply", [None, "invalid", 1, False, []])
def test_non_dict_reply_is_protocol_error(route, flag, reply):
    """各种非字典回复都归为协议异常，避免属性错误泄漏。"""
    with pytest.raises(FM.ProtocolError, match="expected dict"):
        _invoke(route, _Client(flag, reply))


@pytest.mark.parametrize("route", ["run_loop", "warmup_server"])
@pytest.mark.parametrize("flag", ["reset_finished", "add_buffer_finished"])
@pytest.mark.parametrize("value", [None, 0, "", 1, "true"])
def test_completion_requires_boolean_true(route, flag, value):
    """服务端完成标志必须明确为布尔真值，不能接受字符串或数字替代。"""
    with pytest.raises(FM.ProtocolError, match=flag):
        _invoke(route, _Client(flag, {flag: value}))


@pytest.mark.parametrize("flag", ["reset_finished", "add_buffer_finished"])
@pytest.mark.parametrize("kind", ["missing", "false", "non_dict"])
def test_episode_protocol_failure_is_infra(flag, kind):
    """正式局首包错误保留失败状态，单独归为可重试的基础设施协议错误。"""
    reply = {"missing": {}, "false": {flag: False}, "non_dict": ["invalid"]}[kind]
    client = _Client(flag, reply)
    result = FM.evaluate_one(lambda: client, _step, _pre_traj)
    assert result["status"] == "error" and result["task_success"] is False
    assert result["infra"] is True and result["infra_reason"] == "server_protocol"
    assert result["steps"] == 0 and result["decisions"] == 0
    assert result["error"].startswith("ProtocolError:") and flag in result["error"]
    assert client.calls == (["reset"] if flag == "reset_finished" else ["reset", "add_buffer"])
    assert client.closed == 1


def test_normal_episode_has_no_infra_failure():
    """正常正式局的动作数、状态、请求顺序和关闭行为保持原样。"""
    client = _Client()
    result = FM.evaluate_one(lambda: client, _step, _pre_traj)
    assert result["status"] == "success" and result["task_success"] is True
    assert result["steps"] == 1 and result["decisions"] == 1
    assert result["infra"] is False and result["infra_reason"] is None and result["error"] is None
    assert client.calls == ["reset", "add_buffer", "infer"] and client.closed == 1
