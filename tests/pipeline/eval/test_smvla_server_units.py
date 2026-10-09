"""C13-17 ``smvla_server`` 的进程内单元契约（不起端口、不读权重、不初始化 CUDA）：

- 指纹函数（``sha256_file``／``array_sha``／``frame_sha``）；
- ``make_closures`` 的相机键映射与状态归一化；
- ``reseed``／``rng_digest``／``warmup``：每局重设后随机状态回到纯净参照；
- ``_handler`` 的逐消息分派：用内存里的假 websocket 喂真实 msgpack 字节，核对回包、``req_sha``、决策序号与出错即关；
- ``enable_det``／``main``／``cmd_serve`` 的参数与环境闸门。

``SMVLAPolicyHost`` 用 ``object.__new__`` 构造，模型三件换成 CPU 替身，``new_episode``／``observe``／``infer``／``warmup``
都是真实实现（同 ``test_smvla_server_protocol.py`` 的做法，但不走网络、不标 slow）。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import types

import numpy as np
import pytest

import eval_fakes as F

torch = pytest.importorskip("torch", reason="未验证：torch 未安装")
msgpack_numpy = pytest.importorskip("openpi_client.msgpack_numpy", reason="未验证：openpi_client 未安装")


@pytest.fixture(autouse=True)
def _keep_global_rng():
    """被测函数会重设 numpy／torch 全局种子；用完还原，免得影响同进程其他用例。"""
    np_state = np.random.get_state()
    t_state = torch.get_rng_state()
    yield
    np.random.set_state(np_state)
    torch.set_rng_state(t_state)


class _Buf:
    image_keys = ["observation.images.front", "observation.images.wrist"]

    def __init__(self):
        self.frames = []
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.frames.clear()

    def observe(self, full):
        self.frames.append(full)

    def _prepare_inputs(self, instruction):
        return {"n": len(self.frames), "instruction": instruction}


class _Batched:
    device = "cpu"

    def __init__(self, *, consume_rng=False, fail=False):
        self.calls = []
        self.consume_rng = consume_rng
        self.fail = fail

    def generate_batch(self, processed, states):
        self.calls.append((processed[0], states[0]))
        if self.fail:
            raise RuntimeError("替身推理失败")
        if self.consume_rng:
            torch.randn(3)  # 模拟 DiT 采样消耗 torch 全局随机数
        a = np.arange(F.CHUNK_ROWS * 8, dtype=np.float32).reshape(F.CHUNK_ROWS, 8) + processed[0]["n"]
        return [(a, f"sub{processed[0]['n']}")]


def _host(**kw):
    srv = F.smvla_server()
    h = object.__new__(srv.SMVLAPolicyHost)
    h.buffer_factory = _Buf
    h.batched = _Batched(**kw)
    h.normalize_state = None
    h.to_full, h.state_norm = srv.make_closures(h.buffer_factory, None, h.batched)
    h.load_s = 0.0
    srv.reseed()
    h.rng_ref = srv.rng_digest()
    h.metadata = {"policy": "smvla", "fake_host": True, "rng_ref": h.rng_ref, "warmup": None}
    return h


# ---------------------------------------------------------------- 指纹


def test_fingerprints_hand_computed(tmp_path):
    srv = F.smvla_server()
    data = bytes(range(256)) * 5000  # 1.28 MB，跨过 1 MiB 分块
    p = tmp_path / "config.json"
    p.write_bytes(data)
    assert srv.sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert srv.sha256_bytes(b"x") == hashlib.sha256(b"x").hexdigest()
    fr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert srv.frame_sha(fr[:, ::-1]) == hashlib.sha256(fr[:, ::-1].copy().tobytes()).hexdigest()
    # array_sha 把 dtype 与 shape 一起算进去：同字节、不同 dtype 或形状必须给不同指纹（负例）
    a = np.zeros(8, dtype=np.float32)
    assert srv.array_sha(a) != srv.array_sha(a.view(np.int32))
    assert srv.array_sha(a) != srv.array_sha(a.reshape(2, 4))
    assert srv.array_sha(a) == srv.array_sha(np.zeros(8, dtype=np.float32))
    # frame_sha 只看字节：同字节不同形状指纹相同（与 recorder.frame_sha256 同一定义）
    assert srv.frame_sha(fr) == srv.frame_sha(fr.reshape(-1))


def test_git_head_and_setup_paths(tmp_path, monkeypatch):
    """子模块提交号：git 仓库给 40 位 sha，非仓库目录给 None（metadata 里如实记缺失）；路径只补一次。"""
    import re
    import sys

    from tests._support.loaders import REPO

    srv = F.smvla_server()
    assert re.fullmatch(r"[0-9a-f]{40}", srv._git_head(REPO) or "")
    assert srv._git_head(tmp_path) is None
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in (str(srv.SUBMODULE_ROOT),
                                                                        str(srv.OPENPI_CLIENT_SRC))])
    srv._setup_paths()
    srv._setup_paths()
    assert sys.path[:2] == [str(srv.OPENPI_CLIENT_SRC), str(srv.SUBMODULE_ROOT)]
    assert sys.path.count(str(srv.SUBMODULE_ROOT)) == 1


# ---------------------------------------------------------------- 闭包与随机状态


def test_make_closures_maps_camera_keys_and_normalizes_state():
    srv = F.smvla_server()

    class _Norm:
        def __init__(self):
            self.got = None

        def normalize(self, t):
            self.got = t
            return t * 2

    norm = _Norm()
    to_full, state_norm = srv.make_closures(_Buf, norm, _Batched())
    out = to_full({"front": [[1, 2]], "wrist": np.zeros((1, 2)), "depth": np.ones((1, 1))})
    assert sorted(out) == ["depth", "observation.images.front", "observation.images.wrist"]  # 未知键原样
    assert all(v.dtype == np.uint8 for v in out.values())
    s = state_norm(np.arange(8, dtype=np.float64))
    assert norm.got.dtype == torch.float32 and tuple(s.shape) == (1, 8)
    assert s[0].tolist() == [2.0 * i for i in range(8)] and str(s.device) == "cpu"
    _, none_norm = srv.make_closures(_Buf, None, _Batched())
    assert none_norm(np.zeros(8)) is None


def test_reseed_restores_reference_digest():
    srv = F.smvla_server()
    srv.reseed()
    ref = srv.rng_digest()
    assert set(ref) == {"torch_cpu", "numpy"}  # 资源守卫下 CUDA 不可用，不出 torch_cuda
    np.random.random()
    assert srv.rng_digest()["numpy"] != ref["numpy"]
    torch.randn(2)
    assert srv.rng_digest()["torch_cpu"] != ref["torch_cpu"]
    srv.reseed()
    assert srv.rng_digest() == ref


@pytest.mark.parametrize("consume", [True, False])
def test_warmup_reports_consumption_and_restoration(consume):
    h = _host(consume_rng=consume)
    w = h.warmup()
    assert w["rng_consumed_by_warmup"] is consume  # 消耗了才有区分力；不消耗时如实报 False
    assert w["rng_restored"] is True and w["rng"] == h.rng_ref
    assert h.metadata["warmup"] is w
    (processed, state), = h.batched.calls
    assert processed == {"n": 1, "instruction": "warmup"} and state is None  # 一帧全零图 + 零状态（无归一化器）


def test_observe_and_infer_on_real_host_methods():
    srv = F.smvla_server()
    h = _host()
    buf = h.new_episode()
    assert buf.resets == 1
    frames = [{"wrist": F.frame(2), "front": F.frame(1)}, {"front": F.frame(3), "wrist": F.frame(4)}]
    shas = h.observe(buf, frames)
    assert shas == [{"front": F.sha_bytes(F.frame(1)), "wrist": F.sha_bytes(F.frame(2))},
                    {"front": F.sha_bytes(F.frame(3)), "wrist": F.sha_bytes(F.frame(4))}]
    assert list(buf.frames[0]) == ["observation.images.wrist", "observation.images.front"]
    st = np.full(8, 0.5, dtype=np.float32)
    out = h.infer(buf, "抓起方块", st)
    full = np.arange(F.CHUNK_ROWS * 8, dtype=np.float32).reshape(F.CHUNK_ROWS, 8) + 2
    assert np.array_equal(out["actions_full"], full)
    assert np.array_equal(out["actions"], full[:srv.EXECUTE_HORIZON]) and srv.EXECUTE_HORIZON < F.CHUNK_ROWS
    assert out["subtask"] == "sub2" and out["infer_ms"] >= 0
    assert out["recv_instruction_sha"] == hashlib.sha256("抓起方块".encode("utf-8")).hexdigest()
    assert out["recv_state_sha"] == srv.array_sha(st) != srv.array_sha(st.astype(np.float64))


# ---------------------------------------------------------------- _handler 分派（内存假 websocket）


class _FakeWS:
    """按顺序吐出预置的原始消息，吐完抛 ConnectionClosed；记录 send 与 close。``fail_send_at`` 让第 n 次 send 断开。"""

    def __init__(self, raws, *, fail_send_at=None):
        self.raws = list(raws)
        self.sent = []
        self.closed = False
        self.fail_send_at = fail_send_at

    async def recv(self):
        if not self.raws:
            raise F.connection_closed()
        return self.raws.pop(0)

    async def send(self, data):
        if self.fail_send_at is not None and len(self.sent) == self.fail_send_at:
            raise F.connection_closed()
        self.sent.append(data)

    async def close(self):
        self.closed = True


def _drive(host, msgs, **kw):
    srv = F.smvla_server()
    packer = msgpack_numpy.Packer()
    raws = [packer.pack(m) for m in msgs]
    ws = _FakeWS(raws, **kw)
    asyncio.run(srv._handler(host, ws))
    return ws, raws, [msgpack_numpy.unpackb(x) for x in ws.sent]


def test_handler_dispatch_and_req_sha():
    h = _host()
    st = np.zeros(8, dtype=np.float32)
    msgs = [{"reset": {"episode_key": "T/3/7"}},
            {"observe": {"frames": [{"front": F.frame(1), "wrist": F.frame(2)}]}},
            {"infer": {"instruction": "g", "state": st}},
            {"infer": {"instruction": "g", "state": st}},
            {"reset": {"episode_key": "T/4/8"}},
            {"infer": {"instruction": "g", "state": st}}]
    ws, raws, reps = _drive(h, msgs)
    assert reps[0] == h.metadata  # 连接建立先发 metadata
    reps = reps[1:]
    assert [r["req_sha"] for r in reps] == [hashlib.sha256(r).hexdigest() for r in raws]
    assert reps[0]["reset_finished"] is True and reps[0]["episode_key"] == "T/3/7" and reps[0]["rng_matches_ref"] is True
    assert reps[1]["observe_finished"] is True and reps[1]["n"] == 1
    assert reps[1]["frame_sha"] == [{"front": F.sha_bytes(F.frame(1)), "wrist": F.sha_bytes(F.frame(2))}]
    assert [reps[2]["decision"], reps[3]["decision"]] == [0, 1]
    # 第二局 reset 后缓冲清空、决策序号归零
    assert reps[4]["episode_key"] == "T/4/8" and reps[5]["decision"] == 0
    assert [c[0]["n"] for c in h.batched.calls] == [1, 1, 0]
    assert ws.closed is False  # 正常结束只因对端关闭


@pytest.mark.parametrize("msgs,needle", [
    ([{"bogus": {}}], "unknown message keys ['bogus']"),
    ([{"observe": {"frames": []}}], "observe before reset"),
    ([{"infer": {"instruction": "x", "state": np.zeros(8, np.float32)}}], "infer before reset"),
    ([{"reset": {"episode_key": "k"}}, {"infer": {"instruction": "x", "state": np.zeros(8, np.float32)}}],
     "RuntimeError: 替身推理失败"),
])
def test_handler_error_reply_then_close(msgs, needle):
    h = _host(fail=True)
    ws, _, reps = _drive(h, msgs + [{"reset": {"episode_key": "不应处理"}}])
    assert needle in reps[-1]["error"] and ws.closed is True
    assert len(reps) == 1 + len(msgs)  # 出错后不再处理后续消息
    assert all("不应处理" != r.get("episode_key") for r in reps)


def test_handler_returns_quietly_when_peer_closes_during_send():
    h = _host()
    ws, _, reps = _drive(h, [{"reset": {"episode_key": "k"}}], fail_send_at=1)
    assert len(reps) == 1 and ws.closed is False  # 只发出 metadata；回 reset 时对端已断开，不回 error


# ---------------------------------------------------------------- 闸门：确定性模式、固定环境变量、serve 参数


def test_enable_det_requires_cublas_env(monkeypatch):
    srv = F.smvla_server()
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        srv.enable_det()
    assert torch.are_deterministic_algorithms_enabled() is False
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", srv.DET_CUBLAS_WORKSPACE_CONFIG)
    try:
        info = srv.enable_det()
        assert info == {"det": True, "cublas_workspace_config": srv.DET_CUBLAS_WORKSPACE_CONFIG,
                        "deterministic_algorithms": True}
    finally:
        torch.use_deterministic_algorithms(False)


def test_main_refuses_without_fixed_env(monkeypatch):
    srv = F.smvla_server()
    for k in srv._FIXED_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="fixed environment variables not in effect"):
        srv.main(["serve", "--port", "1"])


def test_main_parses_serve_args(monkeypatch):
    srv = F.smvla_server()
    for k, v in srv._FIXED_ENV.items():
        monkeypatch.setenv(k, v)
    got = {}
    monkeypatch.setattr(srv, "cmd_serve", lambda a: got.setdefault("a", a) and 7)
    assert srv.main(["serve", "--port", "4321", "--warmup", "--det", "--policy-seed", "42", "--ckpt", "/ckpt/x"]) == 7
    a = got["a"]
    assert (a.port, a.host, a.warmup, a.det, a.expect_config_sha, a.metadata_out, a.policy_seed) == \
        (4321, "127.0.0.1", True, True, None, None, 42)
    assert a.ckpt == "/ckpt/x"
    with pytest.raises(SystemExit) as ei:  # --ckpt is required (no default checkpoint)
        srv.main(["serve", "--port", "4321", "--policy-seed", "42"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit):  # --port 必填
        srv.main(["serve", "--policy-seed", "7", "--ckpt", "/ckpt/x"])
    with pytest.raises(SystemExit):  # 第三阶段：--policy-seed 必填，不回落旧常量 0
        srv.main(["serve", "--port", "4321", "--ckpt", "/ckpt/x"])


class _HostStub:
    """cmd_serve 用的 host 替身：metadata 字段与真实 host 同名。"""

    instances: list = []

    def __init__(self, ckpt, *, policy_seed):
        self.ckpt = ckpt
        self.policy_seed = policy_seed
        self.load_s = 1.25
        self.warmups = 0
        self.metadata = {"ckpt_config_sha256": "ab" * 32, "versions": {"torch": "x"}, "gpu_name": None}
        _HostStub.instances.append(self)

    def warmup(self):
        self.warmups += 1
        return {"warmup_s": 0.1, "infer_ms": 2.0, "rng_consumed_by_warmup": True, "rng_restored": True}


def _serve_args(tmp_path, **kw):
    base = dict(ckpt=str(tmp_path / "ckpt"), host="127.0.0.1", port=1, warmup=False, expect_config_sha=None,
                metadata_out=None, det=False, policy_seed=7)
    base.update(kw)
    return argparse.Namespace(**base)


def test_cmd_serve_config_sha_gate_and_metadata(monkeypatch, tmp_path, capsys):
    srv = F.smvla_server()
    served = []
    monkeypatch.setattr(srv, "SMVLAPolicyHost", _HostStub)
    monkeypatch.setattr(srv, "asyncio", types.SimpleNamespace(run=lambda coro: (served.append(coro), coro.close())))
    # 负例：config.json 指纹不符 → 退出 2，不起 server
    rc = srv.cmd_serve(_serve_args(tmp_path, expect_config_sha="cd" * 32))
    out = capsys.readouterr().out
    assert rc == 2 and served == []
    assert "SMVLA_DET det=off" in out and f"SMVLA_CONFIG_SHA=FAIL got={'ab' * 32} want={'cd' * 32}" in out
    # 正例：指纹相符 + 预热 + 落 metadata → 起 server（被替换的 asyncio.run 收到 _serve 协程）
    meta = tmp_path / "out" / "meta.json"
    rc = srv.cmd_serve(_serve_args(tmp_path, expect_config_sha="ab" * 32, warmup=True, metadata_out=str(meta)))
    out = capsys.readouterr().out
    host = _HostStub.instances[-1]
    assert rc == 0 and len(served) == 1 and host.warmups == 1
    assert "SMVLA_WARMUP warmup_s=0.1 infer_ms=2.0 rng_consumed=True rng_restored=True" in out
    written = json.loads(meta.read_text())
    assert written["det"] is False and written["ckpt_config_sha256"] == "ab" * 32
    assert "startup_s" in host.metadata and "startup_s" not in written  # 启动耗时在落盘之后才补
    # 第三阶段：服务元数据带 policy_seed／argv／pid／port（客户端据此反查结果行 server_seed）
    assert written["policy_seed"] == 7 and host.policy_seed == 7 and isinstance(written["argv"], list)
    assert written["port"] == 1 and isinstance(written["pid"], int)
