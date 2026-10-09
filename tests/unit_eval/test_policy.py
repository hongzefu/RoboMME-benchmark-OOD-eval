"""``Policy`` 基类计数器、``load_policy`` 工厂、模型注册表与 ``ServerProcess``（起停、就绪、attach 核对、按元数据停）。

服务端用本机回环上的一个小 HTTP 进程充当（不碰 GPU、不出网）。"""
from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest

from robomme_ood_eval import models
from robomme_ood_eval.policy import (Policy, Ready, ServerDead, ServerMismatch, ServerProcess, load_policy, pick_port,
                                      port_busy)
from tests.unit_eval import fakes

SERVER_CODE = r"""
import http.server, sys, signal
port = int(sys.argv[1])
if len(sys.argv) > 2 and sys.argv[2] == "ignore-term":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", port), H)
print("history_config='fake.yaml' READY", flush=True)
srv.serve_forever()
"""


@pytest.fixture(autouse=True)
def _registry(monkeypatch):
    fakes.EVENTS.clear()
    monkeypatch.setitem(models.REGISTRY, "fake", ("tests.unit_eval.fakes", "FakePolicy"))


def test_registry_lists_six_models_lazily():
    assert models.MODELS == ("dummy", "perceptual-framesamp-modul", "groundsg", "smvla", "pp", "astra")
    assert models.REGISTRY["groundsg"] == ("robomme_ood_eval.models.groundsg", "GroundSGPolicy")
    assert models.REGISTRY["perceptual-framesamp-modul"][1] == "FrameSampModulPolicy"
    assert models.REGISTRY["smvla"][1] == "SmvlaPolicy" and models.REGISTRY["pp"][1] == "PPPolicy"
    assert models.REGISTRY["astra"][1] == "AstraPolicy"
    assert models.resolve("dummy").__name__ == "DummyPolicy"
    with pytest.raises(KeyError):
        models.resolve("nope")


def test_counters_count_outermost_call_only():
    class Sub(Policy):
        model = "sub"

        def reset(self, spec):
            super().reset(spec)  # 基类 reset 不重复计数

        def play(self, session, spec, recorder):
            return {}

    p = Sub(3)
    p.load()
    p.reset(None)
    p.reset(None)
    p.play(None, None, None)
    p.close()
    p.close()  # 幂等，但调用计数照记
    assert p.calls == {"load": 1, "reset": 2, "play": 1, "close": 2}
    assert p.label == "sub" and p.policy_seed == 3 and p.episodes_run == 0


def test_reset_contract_in_docstring():
    doc = Policy.__doc__ + (Policy.reset.__doc__ or "")
    assert "send no message to the server" in doc and "never touch the environment" in doc and "ServerDead" in doc


def test_load_policy_calls_load_once_and_context_closes():
    with load_policy("fake", 7, behavior="success") as p:
        assert p.calls["load"] == 1 and p.model == "fake" and p.label == "fake" and p.cfg == {}
        assert p.behavior == "success"
    assert p.calls["close"] == 1 and fakes.EVENTS == ["policy.load", "policy.close"]


def test_load_failure_closes(monkeypatch):
    class Boom(fakes.FakePolicy):
        def load(self):
            raise RuntimeError("加载失败")

    monkeypatch.setattr(fakes, "Boom", Boom, raising=False)
    monkeypatch.setitem(models.REGISTRY, "boom", ("tests.unit_eval.fakes", "Boom"))
    with pytest.raises(RuntimeError):
        load_policy("boom", 1)
    assert fakes.EVENTS == ["policy.close"]


def test_server_seed_from_metadata():
    p = load_policy("fake", 7, with_server=True)
    assert p.server_seed() == 7
    assert load_policy("fake", 7).server_seed() is None


def test_pick_port_free_and_skips_busy():
    p = pick_port()
    assert not port_busy(p)
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    try:
        got = pick_port(busy, span=2, tries=50)
        assert got != busy and got != busy - 1
    finally:
        s.close()


def _server(tmp_path, port, *, seed=7, extra=(), ready=None):
    return ServerProcess([sys.executable, "-c", SERVER_CODE, str(port), *extra], env={"PYTHONUNBUFFERED": "1"},
                         cwd=tmp_path, ready=ready or [Ready.port(), Ready.health(), Ready.log("history_config='fake.yaml'")],
                         port=port, metadata_dir=tmp_path, policy_seed=seed, ckpt="/ckpt/a", ready_timeout_s=30,
                         ready_poll_s=0.1)


def test_server_start_metadata_attach_mismatch_stop(tmp_path, capsys):
    port = pick_port()
    srv = _server(tmp_path, port).start()
    try:
        meta = json.loads(srv.metadata_path.read_text())
        assert meta["pid"] == srv.pid and meta["port"] == port and meta["policy_seed"] == 7 and meta["ckpt"] == "/ckpt/a"
        assert meta["argv"] == srv.argv and os.getpgid(srv.pid) == srv.pid  # setsid 起进程组
        srv.check()
        # 同配置 attach：不另起
        again = _server(tmp_path, port)
        assert again.attach() is True and again.pid == srv.pid
        # 种子不同：拒接
        with pytest.raises(ServerMismatch):
            _server(tmp_path, port, seed=8).start()
        assert "RUN_BLOCKED reason=server_mismatch" in capsys.readouterr().out
        # argv 不同（端口参数相同、多一个参数）：拒接
        with pytest.raises(ServerMismatch):
            _server(tmp_path, port, extra=("x",)).attach()
        assert "SERVER_LEFT" in srv.left_line() and "--stop-server" in srv.left_line()
    finally:
        how = srv.stop(grace_s=10, check_gpu=False)
    assert how == "term" and not srv.metadata_path.exists()
    time.sleep(0.1)
    assert not port_busy(port)


def test_server_dead_detected(tmp_path):
    port = pick_port()
    srv = _server(tmp_path, port).start()
    os.killpg(srv.pid, signal.SIGKILL)
    srv.proc.wait(timeout=10)
    with pytest.raises(ServerDead):
        srv.check()
    srv.stop(grace_s=1, check_gpu=False)


def test_server_dies_before_ready(tmp_path):
    srv = ServerProcess([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], cwd=tmp_path,
                        ready=[Ready.port()], port=pick_port(), metadata_dir=tmp_path, ready_timeout_s=30,
                        ready_poll_s=0.1)
    with pytest.raises(ServerDead):
        srv.start()


def test_stop_escalates_to_kill(tmp_path):
    port = pick_port()
    srv = _server(tmp_path, port, extra=("ignore-term",)).start()
    assert srv.stop(grace_s=1, check_gpu=False) == "kill"


def test_stop_by_metadata_and_attach_stale(tmp_path):
    port = pick_port()
    srv = _server(tmp_path, port).start()
    path = srv.metadata_path
    # 模拟看门狗 os._exit 后另一个进程按元数据停服务端
    assert ServerProcess.stop_by_metadata(path, grace_s=10, check_gpu=False) == "term"
    srv.proc.wait(timeout=10)
    assert not path.exists()
    # 元数据残留但进程已不在：attach 返回 False，start 新起
    path.write_text(json.dumps({"pid": srv.pid, "argv": srv.argv, "port": port, "policy_seed": 7, "ckpt": "/ckpt/a"}))
    fresh = _server(tmp_path, port)
    assert fresh.attach() is False
    assert ServerProcess.stop_by_metadata(path, check_gpu=False) == "gone"


def test_stop_by_metadata_refuses_pid_reuse(tmp_path):
    path = tmp_path / "server-metadata-1.json"
    path.write_text(json.dumps({"pid": os.getpid(), "argv": ["not", "me"], "port": 1}))
    assert ServerProcess.stop_by_metadata(path, check_gpu=False) == "mismatch"
    assert path.exists()


def test_policy_close_stops_server(tmp_path):
    port = pick_port()

    class WithServer(Policy):
        model = "ws"

        def load(self):
            self.server = _server(tmp_path, port).start()

        def play(self, session, spec, recorder):
            return {}

    p = WithServer(7)
    p.load()
    p.server.stop = lambda **k: ServerProcess.stop(p.server, grace_s=10, check_gpu=False)
    p.reset(None)  # 探活通过
    p.close()
    assert not port_busy(port)
