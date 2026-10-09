"""四个普通模型的 Policy 子类（FrameSamp+Modulation、GroundSG、SimpleMemVLA、PonderPounce）：模型侧 4 个方法的契约。

纯 CPU：服务端换成只记参数的 ``FakeSrv``（替换 ``robomme_ood_eval.servers.CleanServerProcess``），websocket／vla-eval
连接换成进程内替身，环境走 ``tests/unit_eval/fakes.py`` 的假 builder；外层用真实 ``episode.run_episode``。期望值一律手写。

核对：

* ``load()`` 进程寿命内只起一次服务端，命令照抄旧 ``run_seat.sh::build_server_cmd``（种子、端口、ckpt、cwd、环境变量），
  就绪判据按模型（MME-VLA 端口 + 日志 ``history_config``、SimpleMemVLA 端口、PonderPounce ``/health``）；
* ``reset(spec)`` 不向服务端发任何消息、不碰环境；服务端死了抛 ``ServerDead``；
* ``play`` 里服务端 reset／start_episode 与 ``session.reset()`` 的次序与旧代码相同（FrameSamp、GroundSG 先服务端，
  SimpleMemVLA、PonderPounce 先环境）；轨迹写在本局 raw 目录；
* ``close()`` 停服务端进程组恰一次；
* GroundSG：``label`` 带变体名、两个评估器（1300、1800）共用一个预测器、按 ``spec.max_steps`` 切换；
* PonderPounce：同一 sid 在本服务进程里第二次出现（``attempt > 1``）才重起服务端，``pp_sid_use_index`` 恒为该 sid 在
  当前服务进程里的使用序号。
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from robomme_ood_eval import episode as E
from robomme_ood_eval import servers as S
from robomme_ood_eval.models import framesamp_modul as FM
from robomme_ood_eval.models import groundsg as GS
from robomme_ood_eval.models import pp as PP
from robomme_ood_eval.models import smvla as SM
from robomme_ood_eval.policy import Policy, ServerDead, ServerMismatch, load_policy

from . import fakes

TASK = "PickXtimes"
#: 真实的 CleanServerProcess（夹具会把模块属性换成 FakeSrv）
REAL_CLEAN = S.CleanServerProcess


# ── 替身 ─────────────────────────────────────────────────────────────────────


class FakeSrv:
    """``CleanServerProcess`` 的替身：只记构造参数；``start()`` 模拟服务端写日志与外壳元数据。"""

    instances: list["FakeSrv"] = []
    log_text = "history_config='perceptual-framesamp-modul.yaml'\n"
    wrap_seed: int | None = None  # None：外壳元数据写本次种子；给值即写该值（模拟种子不符）

    def __init__(self, argv, env=None, cwd=None, gpu=None, ready=None, *, port, metadata_dir, policy_seed=None,
                 ckpt=None, log_path=None, name="server", ready_timeout_s=0.0):
        self.argv, self.env, self.cwd, self.gpu = [str(x) for x in argv], dict(env or {}), cwd, gpu
        self.ready = list(ready) if isinstance(ready, (list, tuple)) else [ready]
        self.port, self.metadata_dir = int(port), Path(metadata_dir)
        self.policy_seed, self.ckpt, self.name = policy_seed, ckpt, name
        self.log_path = self.metadata_dir / f"{name}-{self.port}.log"
        self.pid, self.metadata, self.attached = None, None, False
        self.alive_flag = True
        self.starts = self.stops = 0
        FakeSrv.instances.append(self)

    @property
    def metadata_path(self) -> Path:
        return self.metadata_dir / f"server-metadata-{self.port}.json"

    def start(self):
        self.starts += 1
        fakes.EVENTS.append("server.start")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(type(self).log_text)
        meta = None
        for i, a in enumerate(self.argv):
            if a.startswith("--sgeval-metadata-out="):
                meta = a.split("=", 1)[1]
            elif a == "--metadata_out":
                meta = self.argv[i + 1]
        if meta:
            seed = self.policy_seed if type(self).wrap_seed is None else type(self).wrap_seed
            Path(meta).write_text(json.dumps({"policy_seed": seed}))
        self.pid = 4000 + len(FakeSrv.instances)
        self.metadata = {"policy_seed": self.policy_seed, "pid": self.pid}
        return self

    def check(self):
        if not self.alive_flag:
            raise ServerDead(f"{self.name} 已退出")

    def stop(self, **k):
        self.stops += 1
        fakes.EVENTS.append("server.stop")
        self.pid = None
        return "term"

    def left_line(self):
        return f"SERVER_LEFT pid={self.pid} port={self.port}"


class FakeMMEVLAWebsocketClient:
    """MME-VLA websocket 客户端替身（reset／add_buffer／infer）：每条消息记进 ``EVENTS``。"""

    log: list = []

    def __init__(self, *a, **k):
        self._ws = type("WS", (), {"close": lambda self: None})()

    def reset(self):
        fakes.EVENTS.append("server.reset")
        FakeMMEVLAWebsocketClient.log.append(("reset", None))
        return {"reset_finished": True}

    def add_buffer(self, buf):
        FakeMMEVLAWebsocketClient.log.append(("add_buffer", np.asarray(buf["images"]).shape))
        return {"add_buffer_finished": True}

    def infer(self, obs):
        FakeMMEVLAWebsocketClient.log.append(("infer", {k: v for k, v in obs.items() if isinstance(v, str)}))
        return {"actions": np.zeros((20, 8), np.float32)}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeSmvlaConn:
    """smvla 服务连接替身：按 ``smvla.run_episode`` 的校验口径回包（req_sha、帧指纹、状态／指令指纹）。"""

    def __init__(self, host, port, *a, **k):
        self.metadata = {"fake": True}

    def call(self, msg):
        raw = pickle.dumps(msg)
        kind = next(iter(msg))
        body = msg[kind]
        reply = {"req_sha": _sha(raw)}
        if kind == "reset":
            fakes.EVENTS.append("server.reset")
            reply["rng"] = "r"
        elif kind == "observe":
            fr = body["frames"]
            reply.update(n=len(fr), observe_time_ms=0.1,
                         frame_sha=[{"front": SM.frame_sha(f["front"]), "wrist": SM.frame_sha(f["wrist"])} for f in fr])
        else:
            acts = np.zeros((16, 8), np.float32)
            reply.update(recv_state_sha=SM.array_sha(np.asarray(body["state"])),
                         recv_instruction_sha=_sha(body["instruction"].encode("utf-8")), infer_ms=1.0,
                         actions=acts, actions_full=acts, subtask="grasp")
        return reply, raw, pickle.dumps(reply)

    def close(self):
        pass


class FakePPConn:
    """vla-eval 连接替身（外壳打开：回包带 ``subgoal``）。"""

    def __init__(self, url, timeout):
        self.url = url

    async def connect(self, benchmark=None):
        fakes.EVENTS.append("server.connect")

    async def start_episode(self, cfg):
        fakes.EVENTS.append("server.episode_start")
        FakePPConn.starts.append(cfg)

    async def act(self, obs):
        return {"actions": np.zeros((1, 8), np.float32), "subgoal": None}

    async def end_episode(self, result):
        pass

    async def close(self):
        pass

    async def reconnect(self):
        pass


FakePPConn.starts = []


class GSEnv(fakes.FakeEnv):
    """GroundSG 用的假环境：info 带 Oracle 子目标；有 ``unwrapped.difficulty``。"""

    difficulty = "hard"

    @property
    def unwrapped(self):
        return self

    def reset(self):
        obs, info = super().reset()
        info.update(grounded_subgoal_online="pick up the cube at <10, 20>", simple_subgoal_online="pick")
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        info.update(grounded_subgoal_online="pick up the cube at <10, 20>", simple_subgoal_online="pick")
        return obs, r, term, trunc, info


class GSBuilder(fakes.FakeBuilder):
    def make_env_for_episode(self, episode: int):
        fakes.EVENTS.append("builder.make_env")
        env = GSEnv(episode, self.finish_at, self.finish_status)
        self.envs.append(env)
        return env


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    fakes.EVENTS.clear()
    fakes.FakeBuilder.instances.clear()
    FakeSrv.instances.clear()
    FakeSrv.log_text = "history_config='perceptual-framesamp-modul.yaml'\n"
    FakeSrv.wrap_seed = None
    FakeMMEVLAWebsocketClient.log.clear()
    FakePPConn.starts.clear()
    E.clear_builders()
    monkeypatch.setattr(E, "BUILDER_FACTORY", fakes.FakeBuilder)
    monkeypatch.setattr(S, "CleanServerProcess", FakeSrv)
    monkeypatch.setattr(FM, "make_recording_client", FakeMMEVLAWebsocketClient)
    monkeypatch.setattr(SM, "WSPolicyConn", FakeSmvlaConn)
    monkeypatch.delenv("SGEVAL_PP_SERVER_WRAP", raising=False)
    yield
    E.clear_builders()


def _cfg(tmp_path: Path, **kw) -> dict:
    ck = tmp_path / "ckpt"
    ck.mkdir(exist_ok=True)
    cfg = dict(preflight=False, warmup=False, ckpt_fingerprint=False, server_dir=str(tmp_path / "srv"), port=19001,
               ckpt=str(ck), repo_root=str(tmp_path / "repo"))
    cfg.update(kw)
    return cfg


def _run(policy, tmp_path, episode=0, *, dataset="hard-verify", attempt=1):
    return E.run_episode(policy, dataset, TASK, episode, tmp_path / "out", recorder_factory=fakes.FakeRecorder,
                         render=False, attempt=attempt)


def _spec(policy, tmp_path, dataset="hard-verify", episode=0, attempt=1):
    return E.make_spec(policy, dataset, TASK, episode, tmp_path / "out", attempt=attempt)[0]


# ── 公共件 ───────────────────────────────────────────────────────────────────


def test_clean_server_process_drops_determinism_vars(monkeypatch, tmp_path):
    """起服务端前去掉 XLA_FLAGS 等（旧 CLEAN_ENV）与代理变量；本模型显式给的同名变量保留。"""
    monkeypatch.setenv("XLA_FLAGS", "--bad")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/old")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("http_proxy", "http://proxy")
    srv = REAL_CLEAN(["x"], {"JAX_COMPILATION_CACHE_DIR": "/new"}, port=1, metadata_dir=tmp_path, gpu=3)
    env = srv._child_env()
    assert "XLA_FLAGS" not in env and "CUBLAS_WORKSPACE_CONFIG" not in env and "http_proxy" not in env
    assert env["JAX_COMPILATION_CACHE_DIR"] == "/new" and env["NO_PROXY"] == "127.0.0.1,localhost"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_policy_seed_must_be_non_negative(tmp_path):
    for name in ("perceptual-framesamp-modul", "groundsg", "smvla", "pp"):
        with pytest.raises(S.PreflightError, match="RUN_BLOCKED reason=policy_seed"):
            load_policy(name, -1, **_cfg(tmp_path, groundsg_variant="ground-sg-oracle"))
    assert FakeSrv.instances == []


def test_choose_port_reuses_left_server_metadata(tmp_path):
    d = tmp_path / "srv"
    d.mkdir()
    (d / "server-metadata-18010.json").write_text("{}")
    assert S.choose_port({}, "perceptual-framesamp-modul", d) == 18010  # 留下的服务端：沿用端口去 attach／拒接
    assert S.choose_port({"port": 5}, "pp", d) == 5
    assert S.DEFAULT_PORT_BASE == {"smvla": 18000, "perceptual-framesamp-modul": 18010, "groundsg": 18020,
                                   "pp": 18030}


def test_variant_pairing_and_gates(tmp_path):
    assert S.variant_pairing("ground-sg-oracle", None, None).startswith("VARIANT_PAIRING=PASS")
    for args, why in ((("ground-sg-oracle", "/a", None), "adapter_without_qwenvl"),
                      (("ground-sg-qwenvl", None, None), "qwenvl_needs_adapter"),
                      (("ground-sg-memer", None, None), "memer_needs_adapter"),
                      ((None, None, None), "groundsg_needs_variant"), (("x", None, None), "unknown_variant")):
        with pytest.raises(S.PreflightError, match=f"reason=variant_pairing.*{why}"):
            S.variant_pairing(*args)
    tok = tmp_path / "openpi" / S.TOKENIZER_REL
    tok.parent.mkdir(parents=True)
    tok.write_bytes(b"tokenizer")
    want = _sha(b"tokenizer")
    assert S.tokenizer_gate(tmp_path / "openpi", want.upper()).startswith("TOKENIZER_SHA=PASS")
    with pytest.raises(S.PreflightError, match="reason=tokenizer_sha expected="):
        S.tokenizer_gate(tmp_path / "openpi", "0" * 64)
    with pytest.raises(S.PreflightError, match="missing_args"):
        S.tokenizer_gate(None, want)
    with pytest.raises(S.PreflightError, match="pp_submodule_missing"):
        S.preflight_pp(tmp_path, None, Path("/bin/sh"), server_wrap=True, seed=0)
    (tmp_path / "third_party" / "PonderPounce").mkdir(parents=True)
    with pytest.raises(S.PreflightError, match="pp_ckpt_missing"):
        S.preflight_pp(tmp_path, None, Path("/bin/sh"), server_wrap=True, seed=0)
    ck = tmp_path / "ppck"
    ck.mkdir()
    with pytest.raises(S.PreflightError, match="pp_ckpt_layout"):
        S.preflight_pp(tmp_path, ck, Path("/bin/sh"), server_wrap=True, seed=0)
    (ck / "norm_stats.json").write_text("{}")
    assert S.preflight_pp(tmp_path, ck, Path("/bin/sh"), server_wrap=True, seed=0).startswith("PP_PREFLIGHT=PASS")
    with pytest.raises(S.PreflightError, match="pp_venv_missing"):
        S.preflight_pp(tmp_path, ck, tmp_path / "nope", server_wrap=True, seed=0)
    line = S.ckpt_fingerprint(tmp_path / "openpi")
    assert line.startswith("CKPT_FINGERPRINT ") and "files=1 bytes=9" in line


# ── FrameSamp+Modulation ─────────────────────────────────────────────────────


def test_framesamp_load_once_server_argv_and_close(tmp_path):
    cfg = _cfg(tmp_path, openpi_data_home="/openpi", mme_vla_py="/py/mme-vla", gpus=[1])
    p = load_policy("perceptual-framesamp-modul", 7, **cfg)
    (srv,) = FakeSrv.instances
    meta = Path(cfg["server_dir"]) / "server-wrap-metadata-19001.json"
    assert srv.argv == ["/py/mme-vla", str(S.SERVERS_DIR / "policy_server_wrap.py"), f"--sgeval-metadata-out={meta}",
                        "--seed=7", "--port=19001", "policy:checkpoint", "--policy.config=mme_vla_suite",
                        f"--policy.dir={cfg['ckpt']}"]
    assert Path(srv.cwd) == Path(cfg["repo_root"]) / "third_party" / "mme-vla" and srv.gpu == "1"
    assert srv.env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.75" and srv.env["OPENPI_DATA_HOME"] == "/openpi"
    assert "JAX_COMPILATION_CACHE_DIR" not in srv.env and "XLA_FLAGS" not in srv.env
    assert [r.kind for r in srv.ready] == ["port"] and srv.policy_seed == 7 and srv.ckpt == cfg["ckpt"]
    assert p.load_info["server_config"].startswith("SERVER_CONFIG=PASS")
    assert p.server_seed() == 7
    p.close()
    p.close()  # 幂等
    assert srv.stops == 1 and p.calls == {"load": 1, "reset": 0, "play": 0, "close": 2}


def test_framesamp_compile_cache_and_det_env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "gpu_slug", lambda g: "RTX_6000")
    p = load_policy("perceptual-framesamp-modul", 0, **_cfg(tmp_path, compile_cache="on", det=True,
                                                            jax_cache_root="/jc"))
    srv = FakeSrv.instances[0]
    assert srv.env["JAX_COMPILATION_CACHE_DIR"] == "/jc/RTX_6000"
    assert srv.env["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
    p.close()


def test_framesamp_server_config_missing_blocks_and_stops(tmp_path):
    FakeSrv.log_text = "history_config='symbolic-grounded-subgoal.yaml'\n"
    with pytest.raises(S.PreflightError, match="RUN_BLOCKED reason=server_config"):
        load_policy("perceptual-framesamp-modul", 7, **_cfg(tmp_path, server_config_grace_s=0))
    assert FakeSrv.instances[0].stops >= 1


def test_wrap_metadata_seed_mismatch_is_server_mismatch(tmp_path):
    FakeSrv.wrap_seed = 42
    with pytest.raises(ServerMismatch):
        load_policy("perceptual-framesamp-modul", 7, **_cfg(tmp_path))
    assert FakeSrv.instances[0].stops == 1  # load 失败即 close


def test_framesamp_reset_sends_nothing_and_play_order(tmp_path):
    p = load_policy("perceptual-framesamp-modul", 7, **_cfg(tmp_path))
    spec = _spec(p, tmp_path)
    fakes.EVENTS.clear()
    p.reset(spec)
    assert fakes.EVENTS == [] and FakeMMEVLAWebsocketClient.log == []  # reset 不发消息、不碰环境
    fakes.EVENTS.clear()
    res = _run(p, tmp_path)
    ev = fakes.EVENTS
    assert ev.index("builder.make_env") < ev.index("server.reset") < ev.index("env.reset")  # 先服务端 reset 再环境
    assert res.status == "success" and res.task_success == 1 and res.exec_steps == 4 and res.decisions == 1
    assert [k for k, _ in FakeMMEVLAWebsocketClient.log] == ["reset", "add_buffer", "infer"]
    trace = tmp_path / "out" / "rollouts" / "perceptual-framesamp-modul" / "hard-verify" / "seed7" / res.raw_dir
    rows = [json.loads(x) for x in (trace / "trace.jsonl").read_text().splitlines()]
    assert rows[0]["route"] == "perceptual-framesamp-modul/new" and rows[0]["identity"]["attempt"] == 1
    assert rows[0]["policy_seed"] == 7 and rows[-1]["status"] == "success"
    assert res.timing["policy"]["policy_load"]["port"] == 19001
    p.close()
    assert p.calls == {"load": 1, "reset": 2, "play": 1, "close": 1}


def test_framesamp_reset_raises_server_dead(tmp_path):
    p = load_policy("perceptual-framesamp-modul", 7, **_cfg(tmp_path))
    FakeSrv.instances[0].alive_flag = False
    with pytest.raises(ServerDead):
        _run(p, tmp_path)
    assert "builder.make_env" not in fakes.EVENTS  # 局前拒绝：环境还没建
    p.close()


def test_warmup_reset_add_buffer_infer(tmp_path):
    out = FM.warmup_server("127.0.0.1", 1, subgoal="pick up the cube at <1, 2>", client_factory=FakeMMEVLAWebsocketClient)
    assert [k for k, _ in FakeMMEVLAWebsocketClient.log] == ["reset", "add_buffer", "infer"]
    assert FakeMMEVLAWebsocketClient.log[1][1] == (16, 1, 256, 256, 3)
    assert FakeMMEVLAWebsocketClient.log[2][1] == {"prompt": "warm up", "simple_subgoal": "pick up the cube at <1, 2>",
                                       "grounded_subgoal": "pick up the cube at <1, 2>"}
    assert out["frames"] == 16 and out["actions_shape"] == [20, 8]


def test_framesamp_load_runs_warmup_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(FM, "warmup_server", lambda host, port, **k: calls.append((host, port, k)) or {"ok": 1})
    p = load_policy("perceptual-framesamp-modul", 7, **_cfg(tmp_path, warmup=True))
    assert calls == [("127.0.0.1", 19001, {"frames": 16})] and p.load_info["warmup"] == {"ok": 1}
    _run(p, tmp_path)
    _run(p, tmp_path, episode=1)
    assert len(calls) == 1 and len(FakeSrv.instances) == 1
    p.close()


# ── SimpleMemVLA ─────────────────────────────────────────────────────────────


def test_smvla_server_argv_and_play_order(tmp_path):
    cfg = _cfg(tmp_path, smvla_py="/py/smvla")
    p = load_policy("smvla", 42, **cfg)
    (srv,) = FakeSrv.instances
    meta = Path(cfg["server_dir"]) / "server-wrap-metadata-19001.json"
    assert srv.argv == ["/py/smvla", str(S.SERVERS_DIR / "smvla_server.py"), "serve", "--port", "19001", "--ckpt",
                        cfg["ckpt"], "--warmup", "--policy-seed", "42", "--metadata_out", str(meta)]
    assert Path(srv.cwd) == Path(cfg["repo_root"]) and srv.env["OMP_NUM_THREADS"] == "1"
    assert "CUBLAS_WORKSPACE_CONFIG" not in srv.env and [r.kind for r in srv.ready] == ["port"]
    fakes.EVENTS.clear()
    res = _run(p, tmp_path)
    ev = fakes.EVENTS
    assert ev.index("env.reset") < ev.index("server.reset")  # SimpleMemVLA：环境 reset 之后才发服务端 reset
    assert res.status == "success" and res.exec_steps == 4
    p.close()
    assert srv.stops == 1


def test_smvla_det_flag(tmp_path):
    p = load_policy("smvla", 1, **_cfg(tmp_path, det="on"))
    srv = FakeSrv.instances[0]
    assert "--det" in srv.argv and srv.env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    p.close()


def test_session_no_close_blocks_env_close():
    class Sess:
        steps, closed = 3, 0

        def close(self):
            self.closed += 1

    s = Sess()
    proxy = S.SessionNoClose(s)
    proxy.close()
    assert s.closed == 0 and proxy.close_calls == 1 and proxy.steps == 3


# ── PonderPounce ─────────────────────────────────────────────────────────────


def test_pp_requires_ckpt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.pop("ckpt")
    with pytest.raises(S.PreflightError, match="pp_ckpt_missing"):
        load_policy("pp", 0, **cfg)


def test_pp_server_argv_wrap_and_plain(tmp_path):
    cfg = _cfg(tmp_path, pp_py="/py/pp", connection_factory=FakePPConn)
    p = load_policy("pp", 7, **cfg)
    srv = FakeSrv.instances[0]
    meta = Path(cfg["server_dir"]) / "server-wrap-metadata-19001.json"
    assert srv.argv == ["/py/pp", str(S.SERVERS_DIR / "pp_server_wrap.py"), f"--sgeval-metadata-out={meta}",
                        "--args.checkpoint_path", cfg["ckpt"], "--args.seed", "7", "--args.device", "cuda:0",
                        "--port", "19001"]
    assert Path(srv.cwd) == Path(cfg["repo_root"]) / "third_party" / "PonderPounce"
    assert srv.env["HF_HUB_OFFLINE"] == "1" and [r.kind for r in srv.ready] == ["health"]
    p.close()
    q = load_policy("pp", 7, **_cfg(tmp_path, pp_py="/py/pp", pp_server_wrap=False, port=19003))
    assert FakeSrv.instances[1].argv[:3] == ["/py/pp", "-m", "ponderpounce.eval.robomme_server"]
    q.close()


def test_pp_play_order_and_sid(tmp_path):
    p = load_policy("pp", 7, **_cfg(tmp_path, connection_factory=FakePPConn))
    fakes.EVENTS.clear()
    res = _run(p, tmp_path)
    ev = fakes.EVENTS
    assert ev.index("server.connect") < ev.index("env.reset") < ev.index("server.episode_start")
    sid = f"{TASK}|3|500"  # hard-verify：<task>|<source_episode>|<seed>（fakes.FakeBuilder：source_episode=3+ep、seed=500+ep）
    assert FakePPConn.starts[0]["recording"]["sid"] == sid
    d = res.to_dict()
    assert (d["pp_sid"], d["pp_sid_use_index"], d["pp_server_restarts"]) == (sid, 0, 0)
    assert res.status == "success"
    p.close()


def test_pp_restarts_server_only_when_sid_reused(tmp_path):
    p = load_policy("pp", 7, **_cfg(tmp_path, connection_factory=FakePPConn))
    r1 = _run(p, tmp_path, episode=0)
    r2 = _run(p, tmp_path, episode=1)  # 不同身份：不重起
    assert len(FakeSrv.instances) == 1 and p.server_restarts == 0
    fakes.EVENTS.clear()
    r3 = _run(p, tmp_path, episode=0, attempt=2)  # 同一身份重跑：本服务进程已用过该 sid → reset 里重起
    ev = fakes.EVENTS
    assert ev[:2] == ["server.stop", "server.start"] and ev.index("server.start") < ev.index("builder.make_env")
    assert len(FakeSrv.instances) == 2 and FakeSrv.instances[0].stops == 1 and FakeSrv.instances[1].starts == 1
    r4 = _run(p, tmp_path, episode=1)  # 重起后的新服务进程里 episode 1 的 sid 没用过：不重起
    assert len(FakeSrv.instances) == 2
    assert [r.to_dict()["pp_sid_use_index"] for r in (r1, r2, r3, r4)] == [0, 0, 0, 0]
    assert p.server_restarts == 1 and r3.to_dict()["pp_server_restarts"] == 1 and r3.attempt == 2
    p.close()
    assert FakeSrv.instances[1].stops == 1


# ── GroundSG ─────────────────────────────────────────────────────────────────


@pytest.fixture
def gs_env(monkeypatch):
    import os
    import subprocess

    rel = Path("mme-vla") / "examples" / "robomme" / "eval.py"
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "third_party" / rel).is_file() and not os.environ.get("SGEVAL_THIRD_PARTY"):
        # worktree 里子模块目录为空：只读借主检出的同一固定 sha（git common dir 的上一级即主检出）
        common = subprocess.run(["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=30).stdout.strip()
        main_tp = Path(common).parent / "third_party" if common else None
        assert main_tp is not None and (main_tp / rel).is_file(), "官方源码不在场（设 SGEVAL_THIRD_PARTY 指向主检出）"
        monkeypatch.setenv("SGEVAL_THIRD_PARTY", str(main_tp))
    FakeSrv.log_text = "history_config='symbolic-grounded-subgoal.yaml'\n"
    monkeypatch.setattr(E, "BUILDER_FACTORY", GSBuilder)
    for k in ("IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "FPS_MAX_FRAMES", "USE_HF", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)


def _gs(tmp_path, **kw):
    return load_policy("groundsg", 7, **_cfg(tmp_path, groundsg_variant="ground-sg-oracle",
                                             client_factory=lambda h, p, ep: FakeMMEVLAWebsocketClient(),
                                             work_dir=str(tmp_path / "work"), **kw))


def test_groundsg_needs_variant(tmp_path):
    with pytest.raises(S.PreflightError, match="groundsg_needs_variant"):
        load_policy("groundsg", 7, **_cfg(tmp_path))
    assert FakeSrv.instances == []


def test_groundsg_label_two_evaluators_shared_predictor(tmp_path, gs_env):
    p = _gs(tmp_path)
    assert p.label == "groundsg-ground-sg-oracle" and p.model == "groundsg"
    assert sorted(p.evaluators) == [1300, 1800]
    (a13, e13), (a18, e18) = p.evaluators[1300], p.evaluators[1800]
    assert (a13.max_steps, a18.max_steps) == (1300, 1800) and e13 is not e18
    assert a13.model_seed == a18.model_seed == 7 and a13.use_oracle and a18.use_oracle
    assert len(FakeSrv.instances) == 1 and FakeSrv.instances[0].argv[3] == "--seed=7"
    assert "--policy.dir=" + _cfg(tmp_path)["ckpt"] in FakeSrv.instances[0].argv
    seen = []

    def fake_run(session, identity, conn, recorder):
        ctx = conn["policy_context"]
        seen.append((conn["max_steps"], ctx["args"].max_steps, ctx["evaluator"], id(ctx["predictor"]),
                     conn["trace_path"], conn["groundsg_variant"], Path(conn["trace_dir"]).parent))
        session.reset()
        session.step(np.zeros(8, np.float32))
        return {"status": "fail", "task_success": False, "steps": 1, "error": None, "infra": False,
                "infra_reason": None, "decisions": 1}

    import robomme_ood_eval.models.groundsg as gmod

    orig = gmod.run_episode
    gmod.run_episode = fake_run
    try:
        r1 = _run(p, tmp_path, dataset="hard-verify")
        r2 = _run(p, tmp_path, dataset="ood")
    finally:
        gmod.run_episode = orig
    assert [(s[0], s[1]) for s in seen] == [(1300, 1300), (1800, 1800)]
    assert seen[0][2] is e13 and seen[1][2] is e18 and seen[0][3] == seen[1][3]  # 同一个预测器
    assert seen[0][4].endswith("/rollouts/groundsg-ground-sg-oracle/hard-verify/seed7/raw/"
                               f"{TASK}_ep3_xhard0/trace.jsonl")
    assert seen[0][5] == "ground-sg-oracle" and seen[0][6] == tmp_path / "work"
    assert not any((tmp_path / "work").glob("groundsg-*"))  # 本局临时区局末整删
    assert r1.policy_label == "groundsg-ground-sg-oracle" and r2.status == "fail"
    p.close()
    assert p.ctx is None and p.evaluators == {} and FakeSrv.instances[0].stops == 1


def test_groundsg_end_to_end_order_and_official_video(tmp_path, gs_env):
    p = _gs(tmp_path)
    fakes.EVENTS.clear()
    res = _run(p, tmp_path)
    ev = fakes.EVENTS
    assert ev.index("builder.make_env") < ev.index("server.reset") < ev.index("env.reset")
    assert res.status == "success" and res.exec_steps == 4
    d = res.to_dict()
    assert d["policy_variant"] == "ground-sg-oracle" and d["official_source"] == "official"
    raw = tmp_path / "out" / "rollouts" / "groundsg-ground-sg-oracle" / "hard-verify" / "seed7" / res.raw_dir
    assert (raw / "trace.jsonl").is_file() and len(list((raw / "official").glob("*.mp4"))) == 1
    infers = [x for k, x in FakeMMEVLAWebsocketClient.log if k == "infer"]
    assert infers and infers[0]["grounded_subgoal"] == "pick up the cube at <10, 20>"
    p.close()


def test_policy_subclasses_are_registered():
    from robomme_ood_eval.models import resolve

    for name, cls in (("perceptual-framesamp-modul", FM.FrameSampModulPolicy), ("groundsg", GS.GroundSGPolicy),
                      ("smvla", SM.SmvlaPolicy), ("pp", PP.PPPolicy)):
        assert resolve(name) is cls and issubclass(cls, Policy) and cls.model == name
