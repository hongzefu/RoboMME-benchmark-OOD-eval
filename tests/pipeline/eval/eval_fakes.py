"""评估流水线测试（C13）的公共替身：假环境、混合 builder、假录制器、假策略 server 与两种协议的假连接。

计划细则 4.5.4 原定放 ``tests/_support/eval_fakes.py``；``tests/_support/`` 归主会话，本块按分配表放在
``tests/pipeline/eval/`` 内。

设计口径：
- 生产模块一律经 ``tests._support.loaders.load_script`` 按路径加载（与生产入口的加载方式相同），不往 sys.modules 注入替身。
- 身份解析用真实 ``robomme_hard`` 的 ``BenchmarkEnvBuilder``（包内规格），只有 ``make_env_for_episode`` 换成 CPU 假环境，
  所以身份核对（``check_identity``）走的是真实解析结果；不构建任何仿真场景。
- 假策略 server 的动作由「本局 reset 之后收到的全部帧指纹 + 当前状态 + 指令」的摘要确定性生成：某局若没有 reset，
  就会带着上一局的缓冲算出不同动作——跨局隔离（A→B→A）因此可观测。
- smvla 协议的指纹函数取真实 ``smvla_server.py``（server 侧的对应实现），不在测试里另写一份。
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import pickle
import types
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tests._support.loaders import REPO, load_script

HW = 4  # 假帧边长（像素）；形状不参与被测逻辑
#: 步数上限（拆分方案已定口径第 3 条），按约定手写，不读被测代码：ood 为 1800 且严格截断，hard-verify 为 1300、不截断
V9_MAX_STEPS = 1800
HARD0_MAX_STEPS = 1300
N_RESET_FRAMES = 3  # 假环境 reset 返回的帧数（2 帧演示 + 1 帧初始）
CHUNK_ROWS = 20  # 假 server 每次推理回的动作行数（多于执行段，用来核「只执行前若干行」）


# ---------------------------------------------------------------- 生产模块


def env_client():
    return load_script("eval-official/env_client.py")


def env_session():
    """拆仓后 ``EnvSession``、``NullRecorder``、``StepCapReached``、``RecorderError``、``ResetBudgetExhausted`` 在评估包
    ``robomme_hard_eval.session``（原在 env_client.py；``env_client()`` 现指向只剩席位层的 ``dev-scripts/gl/seat.py``）。"""
    from robomme_hard_eval import session

    return session


def framesamp_modul_client():
    return load_script("eval-official/framesamp_modul_client.py")


def smvla_client():
    return load_script("eval-official/smvla_client.py")


def smvla_server():
    return load_script("eval-official/smvla_server.py")


def eval_report():
    return load_script("eval-official/eval_report.py")


def eval_manifest():
    return load_script("eval-official/eval_manifest.py")


def hard_specs():
    from robomme_hard.env_record_wrapper import hard_specs as hs

    return hs


def real_builder(task: str, max_steps: int | None = None, dataset: str = "ood"):
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    kw = {} if max_steps is None else {"max_steps": int(max_steps)}
    return BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="joint_angle", **kw)


# ---------------------------------------------------------------- 观测与假环境


def frame(v: int) -> np.ndarray:
    return np.full((HW, HW, 3), int(v) % 256, dtype=np.uint8)


def sha_bytes(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def obs_of(vals: list[int]) -> dict:
    """与真实环境同键的观测：五个列表等长；关节 7 维、夹爪 2 维、末端 6 维。"""
    return {
        "front_rgb_list": [frame(v) for v in vals],
        "wrist_rgb_list": [frame(v + 1) for v in vals],
        "joint_state_list": [np.full(7, v / 10.0, dtype=np.float64) for v in vals],
        "gripper_state_list": [np.array([v / 100.0, v / 100.0], dtype=np.float64) for v in vals],
        "eef_state_list": [np.zeros(6, dtype=np.float64) for _ in vals],
    }


def reset_values(ep: int) -> list[int]:
    base = (ep * 7) % 150
    return [base + i for i in range(N_RESET_FRAMES)]


@dataclasses.dataclass
class Plan:
    """一次尝试里假环境的行为：第 n 次 step 报 success／fail 终态，或抛异常；都为空则永不终止。"""

    success_at: int | None = None
    fail_at: int | None = None
    raise_at: int | None = None
    raise_exc: Callable[[], BaseException] | None = None


class FakeEnv:
    def __init__(self, task: str, ep: int, plan: Plan):
        self.task, self.ep, self.plan = task, ep, plan
        self.n = 0
        self.actions: list[np.ndarray] = []
        self.resets = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        return obs_of(reset_values(self.ep)), {"task_goal": [f"goal-{self.task}-{self.ep}", "alt"], "status": "ongoing"}

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.n += 1
        if self.plan.raise_at == self.n:
            raise self.plan.raise_exc()
        status = "ongoing"
        if self.plan.success_at == self.n:
            status = "success"
        elif self.plan.fail_at == self.n:
            status = "fail"
        terminated = status in ("success", "fail")
        return obs_of([100 + (self.ep * 3 + self.n) % 100]), 0.0, terminated, False, {"status": status}

    def close(self):
        self.closed = True


class World:
    """一组测试共享的假世界：按 (task, builder_episode) 给每次尝试的 Plan，记下全部假环境与 builder 调用。"""

    def __init__(self, plans: dict[tuple[str, int], list[Plan]] | None = None, default: Plan | None = None):
        self.plans = {k: list(v) for k, v in (plans or {}).items()}
        self.default = default or Plan(success_at=3)
        self.envs: list[FakeEnv] = []
        self.make_calls: list[tuple] = []
        self.builders: list[Any] = []
        self.recorders: list["FakeRecorder"] = []

    def new_env(self, task: str, ep: int) -> FakeEnv:
        lst = self.plans.get((task, ep))
        plan = (lst.pop(0) if len(lst) > 1 else lst[0]) if lst else self.default
        env = FakeEnv(task, ep, plan)
        self.envs.append(env)
        return env

    def envs_of(self, task: str, ep: int) -> list[FakeEnv]:
        return [e for e in self.envs if (e.task, e.ep) == (task, ep)]


class HybridBuilder:
    """真实 builder 的身份解析 + CPU 假环境（不构建仿真场景）。``dataset`` 原样交给真实 builder。"""

    def __init__(self, task: str, max_steps: int | None, world: World, dataset: str = "ood"):
        self.task, self.max_steps, self.world, self.dataset = task, max_steps, world, dataset
        self.real = real_builder(task, max_steps, dataset)
        world.builders.append(self)

    def resolve_identity(self, ep):
        return self.real.resolve_identity(ep)

    def make_env_for_episode(self, ep, max_steps=None):
        self.world.make_calls.append((self.task, int(ep), max_steps, self.max_steps))
        return self.world.new_env(self.task, int(ep))


class FakeRecorder:
    """接口同 ``recorder.EpisodeRecorder``；close 时写出报告核对所需的三件媒体占位文件。"""

    def __init__(self, rec_dir, meta, world: World | None = None, *, fail_on: str | None = None):
        self.rec_dir = Path(rec_dir)
        self.meta = dict(meta)
        self.phases: list[str] = []
        self.frames: Counter = Counter()
        self.arrays: Counter = Counter()
        self.events: list[dict] = []
        self.fail_on = fail_on
        self.closed_with = None
        if world is not None:
            world.recorders.append(self)

    def _maybe_fail(self, name):
        if self.fail_on == name:
            raise OSError(28, "No space left on device")

    def set_phase(self, phase):
        self._maybe_fail("set_phase")
        self.phases.append(phase)

    def add_frames(self, stream, frames, *, tag=""):
        self._maybe_fail("add_frames")
        n = int(np.asarray(frames).shape[0])
        self.frames[stream] += n
        return list(range(n))

    def add_array(self, name, arr, *, step=None):
        self._maybe_fail("add_array")
        self.arrays[name] += 1

    def add_event(self, event):
        self.events.append(dict(event))

    def close(self, summary):
        self._maybe_fail("close")
        self.closed_with = dict(summary)
        self.rec_dir.mkdir(parents=True, exist_ok=True)
        for name in ("front.mkv", "wrist.mkv"):
            (self.rec_dir / name).write_bytes(b"placeholder")
        (self.rec_dir / "summary.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")
        return {"RECORDER_VERIFY": "PASS"}


# ---------------------------------------------------------------- 假策略 server 与两种协议的连接


def connection_closed():
    from websockets.exceptions import ConnectionClosed

    return ConnectionClosed(None, None)


class FakePolicyServer:
    """跨连接共享的策略状态。``fail_on``：None／"infer_disconnect"（推理时连接断开）／"server_error"（回 error）。"""

    def __init__(self, *, fail_on: str | None = None):
        self.fail_on = fail_on
        self.buffer: list[str] = []
        self.log: list[tuple[str, Any]] = []

    def reset(self, key=None):
        self.buffer = []
        self.log.append(("reset", key))

    def observe(self, shas: list[str], extra=None):
        self.buffer.extend(shas)
        self.log.append(("observe", {"frames": list(shas), **(extra or {})}))

    def actions(self, state: np.ndarray, prompt: str) -> np.ndarray:
        self.log.append(("infer", {"state": np.array(state, copy=True), "prompt": prompt,
                                   "buffer_len": len(self.buffer)}))
        if self.fail_on == "infer_disconnect":
            raise connection_closed()
        h = hashlib.sha256("|".join(self.buffer).encode() + np.asarray(state, np.float32).tobytes()
                           + prompt.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        a = rng.uniform(-1.0, 1.0, size=(CHUNK_ROWS, 8)).astype(np.float32)
        self.log[-1][1]["actions"] = a.copy()
        return a

    def kinds(self) -> list[str]:
        return [k for k, _ in self.log]


class _FakeWS:
    def __init__(self, server):
        self.server = server

    def close(self):
        self.server.log.append(("close", None))


class FakeMMEVLAWebsocketClient:
    """``MMEVLAWebsocketClientPolicy`` 的协议替身：reset／add_buffer／infer 三种消息。"""

    def __init__(self, server: FakePolicyServer):
        self.server = server
        self._ws = _FakeWS(server)

    def reset(self):
        self.server.reset()
        return {"reset_finished": True}

    def add_buffer(self, buf):
        imgs = np.asarray(buf["images"])
        self.server.observe([sha_bytes(imgs[i, 0]) for i in range(imgs.shape[0])],
                            {"exec_start_idx": int(buf["exec_start_idx"]), "shape": list(imgs.shape),
                             "states": [np.array(s, copy=True) for s in buf["state"]]})
        return {"add_buffer_finished": True}

    def infer(self, element):
        a = self.server.actions(np.asarray(element["observation/state"]), str(element["prompt"]))
        self.server.log[-1][1].update(image=sha_bytes(element["observation/image"]),
                                      wrist=sha_bytes(element["observation/wrist_image"]))
        return {"actions": a}


class FakeSmvlaConn:
    """smvla 协议的假连接（接口同 ``smvla_client.WSPolicyConn``）；回包指纹用真实 ``smvla_server`` 的函数。"""

    def __init__(self, server: FakePolicyServer, *, tamper: str | None = None):
        self.server = server
        self.metadata = {"policy": "smvla", "fake": True}
        self.tamper = tamper  # None／"req_sha"／"frame_sha"
        self.n = 0
        self.closed = False

    def call(self, msg: dict):
        srv = smvla_server()
        raw = pickle.dumps((self.n, msg), protocol=4)
        self.n += 1
        kind = next(iter(msg))
        if kind == "reset":
            self.server.reset(msg["reset"]["episode_key"])
            rep = {"reset_finished": True, "rng": {"fake": True}}
        elif kind == "observe":
            frs = msg["observe"]["frames"]
            shas = [{k: srv.frame_sha(v) for k, v in sorted(f.items())} for f in frs]
            self.server.observe([d["front"] for d in shas])
            if self.tamper == "frame_sha" and shas:
                shas = shas[:-1]
            rep = {"observe_finished": True, "n": len(frs), "frame_sha": shas}
        elif kind == "infer":
            if self.server.fail_on == "server_error":
                rep = {"error": "Traceback: RuntimeError: 假 server 内部错误"}
                return rep, raw, b"e" + raw
            p = msg["infer"]
            full = self.server.actions(np.asarray(p["state"]), str(p["instruction"]))
            rep = {"actions": full[:srv.EXECUTE_HORIZON], "actions_full": full, "subtask": "s", "infer_ms": 1.0,
                   "recv_state_sha": srv.array_sha(np.asarray(p["state"])),
                   "recv_instruction_sha": srv.sha256_bytes(str(p["instruction"]).encode("utf-8"))}
        else:
            raise AssertionError(f"未知消息 {kind}")
        rep["req_sha"] = srv.sha256_bytes(raw)
        if self.tamper == "req_sha":
            rep["req_sha"] = "0" * 64
        return rep, raw, b"r" + raw

    def close(self):
        self.closed = True


def framesamp_modul_policy(monkeypatch, server: FakePolicyServer):
    """真 framesamp_modul_client 模块，只把建 websocket 客户端的工厂换成假客户端（每局一个新客户端，同真实行为）。"""
    mc = framesamp_modul_client()
    monkeypatch.setattr(mc, "make_recording_client", lambda host, port, recorder, timing: FakeMMEVLAWebsocketClient(server))
    return mc


def smvla_policy(server: FakePolicyServer, **conn_kw):
    """真 smvla_client.run_episode，注入假连接；保留其关键字签名（SeatRunner 据此传 max_steps／reset_retries）。"""
    sm = smvla_client()
    return types.SimpleNamespace(run_episode=functools.partial(sm.run_episode, conn=FakeSmvlaConn(server, **conn_kw)))


def policy_module(name: str, monkeypatch, server: FakePolicyServer):
    return framesamp_modul_policy(monkeypatch, server) if name == "perceptual-framesamp-modul" else smvla_policy(server)


# ---------------------------------------------------------------- 身份


def tier_cap(tier: str) -> int:
    """该档按启动约定的步数上限（手写常量）：xhard0 走 hard-verify 的 1300，其余档走 ood 的 1600。"""
    return HARD0_MAX_STEPS if tier == "xhard0" else V9_MAX_STEPS


@functools.lru_cache(maxsize=None)
def _resolved(task: str) -> tuple[tuple[int, dict], ...]:
    """ood 的真实 builder 逐局解析（拆仓后 ood 不含 xhard0，局号 0～49）。"""
    b = real_builder(task)
    return tuple((ep, b.resolve_identity(ep)) for ep in range(b.get_episode_num()))


def packaged_identity(task: str, tier: str, k: int = 0) -> dict:
    """包内真实身份（真实 builder 在 ood 里第 k 个该档局）→ 执行身份行（字段契约 C1 + dataset，key 按契约手写）。"""
    hits = [(ep, ident) for ep, ident in _resolved(task) if ident["tier"] == tier]
    ep, ident = hits[k]
    return {"dataset": "ood", "task": task, "tier": tier, "seed": int(ident["seed"]), "candidate": ident["candidate"],
            "builder_episode": ep, "source_episode": None, "spec_sha256": ident["spec_sha256"],
            "key": f"{task}_{tier}_{int(ident['seed'])}"}


@functools.lru_cache(maxsize=None)
def _resolved_hard0(task: str) -> tuple[tuple[int, dict], ...]:
    b = real_builder(task, dataset="hard-verify")
    return tuple((ep, b.resolve_identity(ep)) for ep in range(b.get_episode_num()))


def hard0_identity(task: str, k: int = 0) -> dict:
    """hard-verify 里第 k 局的执行身份行（字段契约同 C1；candidate／spec_sha256 为 null，key 按契约手写）。"""
    ep, ident = _resolved_hard0(task)[k]
    return {"dataset": "hard-verify", "task": task, "tier": "xhard0", "seed": int(ident["seed"]), "candidate": None,
            "builder_episode": ep, "source_episode": int(ident["source_episode"]), "spec_sha256": None,
            "key": f"{task}_xhard0_{int(ident['seed'])}"}


def v9_cells_sorted() -> list[tuple[str, str]]:
    return sorted(hard_specs().V9_CELLS)


# ---------------------------------------------------------------- 席位客户端（拆仓后：常驻 Policy + 动态队列）
#
# 新 ``seat.py`` 的 ``SeatRunner`` 不再接收策略模块，而是经 ``policy_factory`` 拿一个常驻 ``Policy``，逐局调评估包的
# ``episode.run_episode``。这里给两种替身：
# - ``FakeSeatPolicy``：新接口的假模型，动作由 ``FakePolicyServer`` 按「本局 reset 后收到的帧 + 状态 + 指令」确定性生成；
#   环境 step 抛含 ``svulkan2`` 的异常、或推理连接断开，记基础设施错误（infra=True）；其余异常记普通错误；
# - ``ModulePolicy``：把旧的模块级客户端（``run_episode(session, identity, conn_info, recorder)``）包成 Policy，
#   旧用例的 ``framesamp_modul_policy``／``smvla_policy`` 照旧能喂给 ``make_runner``。
# builder 经 ``episode.BUILDER_FACTORY`` 换成 ``HybridBuilder``（真实身份解析 + CPU 假环境），``run_rows`` 收尾复原。

#: 假模型每次推理执行的动作行数
EXEC_ROWS = 5


def _policy_base():
    from robomme_hard_eval.policy import Policy

    return Policy


def _make_policy_classes():
    Policy = _policy_base()

    class FakeSeatPolicy(Policy):
        model = "fake-seat"

        def __init__(self, policy_seed: int = 7, server: "FakePolicyServer | None" = None, **cfg):
            super().__init__(policy_seed, **cfg)
            self.server = None  # 无真实服务端进程（ServerProcess）
            self.fake = server or FakePolicyServer()
            self.specs: list = []

        def reset(self, spec) -> None:
            super().reset(spec)
            self.specs.append(spec)
            self.fake.reset(spec.key)

        def play(self, session, spec, recorder) -> dict:
            obs, info = session.reset()
            goal = info.get("task_goal") if isinstance(info, dict) else None
            prompt = str(goal[0] if isinstance(goal, list) else goal)
            self.fake.observe([sha_bytes(f) for f in obs["front_rgb_list"]])
            steps = 0
            try:
                while True:
                    state = np.concatenate([obs["joint_state_list"][-1], obs["gripper_state_list"][-1][:1]])
                    chunk = self.fake.actions(state, prompt)
                    for row in chunk[:EXEC_ROWS]:
                        if not spec.strict_cap and steps > spec.max_steps:  # 官方循环 count > max_steps 才停
                            return {"status": "timeout", "task_success": 0, "steps": steps, "error": None,
                                    "infra": False, "infra_reason": None, "decisions": steps}
                        obs, _r, terminated, truncated, info = session.step(row)
                        steps += 1
                        self.fake.observe([sha_bytes(obs["front_rgb_list"][-1])])
                        if terminated or truncated:
                            st = info.get("status")
                            status = st if st in ("success", "fail") else "timeout"
                            return {"status": status, "task_success": int(status == "success"), "steps": steps,
                                    "error": None, "infra": False, "infra_reason": None, "decisions": steps}
            except Exception as e:  # noqa: BLE001
                from websockets.exceptions import ConnectionClosed

                from robomme_hard_eval.session import StepCapReached

                if isinstance(e, StepCapReached):
                    raise
                infra = isinstance(e, ConnectionClosed) or "svulkan2" in str(e)
                return {"status": "error", "task_success": 0, "steps": steps, "error": f"{type(e).__name__}: {e}",
                        "infra": infra, "infra_reason": "env_exception" if infra else None}

    class ModulePolicy(Policy):
        model = "module"

        def __init__(self, policy_seed: int = 7, mod=None, policy_name: str = "module", **cfg):
            super().__init__(policy_seed, **cfg)
            self.mod, self.policy_name = mod, policy_name
            self.ctx = None
            # 模块可带 conn_extra（如 GroundSG 的变体与 adapter），并入每局 conn_info 与 make_policy_context 的入参
            self.conn_extra = dict(getattr(mod, "conn_extra", None) or {})

        def play(self, session, spec, recorder) -> dict:
            make = getattr(self.mod, "make_policy_context", None)
            if self.ctx is None:
                self.ctx = make({"policy": self.policy_name, "max_steps": spec.max_steps, "host": "127.0.0.1",
                                 "port": 1, "policy_seed": self.policy_seed, "dataset": spec.dataset,
                                 **self.conn_extra}) if callable(make) else {}
            ident = dict(spec.identity())
            # 与 servers.ServedPolicy._conn_info 同口径：effective_cap 只在 strict 时给；轨迹落在本局 raw 目录
            conn_info = {"host": "127.0.0.1", "port": 1, "max_steps": spec.max_steps, "policy": self.policy_name,
                         "dataset": spec.dataset, "strict_cap": spec.strict_cap, "policy_seed": self.policy_seed,
                         "effective_cap": spec.max_steps if spec.strict_cap else None,
                         "trace_dir": spec.out_dir, "episode_tag": Path(spec.out_dir).name, "rec_dir": spec.out_dir,
                         "policy_context": self.ctx, **self.conn_extra}
            # 旧 SeatRunner 的口径：模块的 run_episode 接受 max_steps／reset_retries 关键字时显式传本局上限与 0 次 reset 重试
            # （SimpleMemVLA 的 hard_bound 由 max_steps 算）；只读 conn_info 的模块不传
            kw = {k: v for k, v in (("max_steps", spec.max_steps), ("reset_retries", 0))
                  if _accepts(self.mod.run_episode, k)}
            res = dict(self.mod.run_episode(session, ident, conn_info, recorder, **kw))
            res.setdefault("error", None)
            res.setdefault("infra", False)
            res.setdefault("infra_reason", None)
            res["task_success"] = int(bool(res.get("task_success")))
            res.setdefault("steps", 0)
            return res

    return FakeSeatPolicy, ModulePolicy


def _accepts(fn, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def fake_seat_policy(server: "FakePolicyServer | None" = None, seed: int = 7):
    return _make_policy_classes()[0](seed, server=server)


def as_policy(policy_obj, policy_name: str, seed: int = 7):
    """``make_runner`` 的策略参数：Policy 实例原样返回；旧模块（或带 ``run_episode`` 的命名空间）包成 ``ModulePolicy``。"""
    if isinstance(policy_obj, _policy_base()):
        return policy_obj
    return _make_policy_classes()[1](seed, mod=policy_obj, policy_name=policy_name)


def episode_mod():
    from robomme_hard_eval import episode as E

    return E


def seat_args(out: Path, policy: str, *, seat: str = "s00", reset_budget: int | None = 100, infra_retries: int = 1,
              policy_seed: int = 7, dataset: str | None = None, budget_ledger=None, **kw) -> argparse.Namespace:
    """新 ``seat.py run`` 的参数（按其 parser 的缺省值补齐）；``out`` 为产物根。测试缺省给 1 次基础设施重试（每身份至多 2
    次尝试），以便覆盖重试路径；生产缺省为 0。"""
    ec = env_client()
    argv = ["run", "--policy", "dummy", "--identities", "x", "--out", str(out)]
    args, _ = ec.build_parser().parse_known_args(argv)
    args.policy = policy
    args.extra_cfg = {}
    args.seat = seat
    args.reset_budget = reset_budget
    args.infra_retries = infra_retries
    args.policy_seed = policy_seed
    args.dataset = dataset
    args.budget_ledger = budget_ledger if isinstance(budget_ledger, (str, Path)) else None
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def make_runner(stage: Path, policy: str, policy_obj, world: World, *, seat_dir: str = "s00",
                budget_ledger=None, recorder_factory=None, now=None, hard_exit=None, **kw):
    """新席位：产物根 ``<stage>``（其下 rollouts/、queue/、seats/）；builder 换成 ``HybridBuilder``，录制器换成
    ``FakeRecorder``，不出网站视频。``budget_ledger``：路径或已打开的账本对象（共享模式），None 即关闭。"""
    ec, E = env_client(), episode_mod()
    prev = E.BUILDER_FACTORY
    E.clear_builders()
    E.BUILDER_FACTORY = lambda task, ds, ms: HybridBuilder(task, ms, world, ds)
    args = seat_args(Path(stage), policy, seat=seat_dir, budget_ledger=budget_ledger, **kw)
    pol = as_policy(policy_obj, policy, seed=args.policy_seed)
    shared = budget_ledger if budget_ledger is not None else None
    runner = ec.SeatRunner(args, policy_factory=lambda model, seed, **cfg: pol,
                           episode_kwargs={"recorder_factory": recorder_factory
                                           or (lambda raw, meta: FakeRecorder(raw, meta, world)), "render": False},
                           hard_exit=hard_exit or (lambda code: None), shared=shared,
                           **({"now": now} if now is not None else {}))
    runner._fake_restore = prev
    runner.fake_policy = pol
    return runner


def run_rows(runner, rows: list[dict]) -> int:
    """返回生产退出码：``SeatRunner.run`` 的返回值（0／3／5／6）；收尾复原 ``BUILDER_FACTORY``。"""
    E = episode_mod()
    try:
        return int(runner.run(rows))
    finally:
        E.BUILDER_FACTORY = getattr(runner, "_fake_restore", None)
        E.clear_builders()


def seat_dir(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> Path:
    return Path(stage) / "seats" / label / f"seed{seed}" / seat


def seat_ledger(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> list[dict]:
    return read_jsonl(seat_dir(stage, label, seat, seed) / f"{label}.ledger.jsonl")


def seat_results(stage: Path, label: str, seat: str = "s00", seed: int = 7) -> list[dict]:
    return read_jsonl(seat_dir(stage, label, seat, seed) / "seat-results.jsonl")


def queue_dir(stage: Path, label: str, seed: int = 7) -> Path:
    return Path(stage) / "queue" / label / f"seed{seed}"


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- 清单与报告


def write_manifest(path: Path, rows: list[dict], *, shard: str = "00", **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = Counter(f"{r['task']}@{r['tier']}" for r in rows)
    doc = {"schema": eval_manifest().SCHEMA, "total": len(rows), "cells": dict(cells),
           "rows": [dict(r, shard=shard) for r in rows], **extra}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_report(capsys, manifest: Path, stage: Path, policies: list[str], out: Path, *extra: str) -> tuple[int, list[str], dict]:
    """进程内调用真实 ``eval_report.main``；返回 (退出码, 判定行, report.json)。"""
    er = eval_report()
    capsys.readouterr()
    rc = er.main(["--manifest", str(manifest), "--stage", str(stage), "--policies", ",".join(policies),
                  "--out", str(out), *extra])
    lines = [x for x in capsys.readouterr().out.splitlines() if "=" in x.split(" ")[0]]
    return rc, lines, json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))


def verdict(lines: list[str], name: str) -> dict[str, str]:
    """把 ``NAME=PASS k=v ...`` 解析成 {"": "PASS", k: v}。"""
    for line in lines:
        head, *rest = line.split()
        if head.startswith(name + "="):
            d = {"": head.split("=", 1)[1]}
            for kv in rest:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    d[k] = v
            return d
    raise AssertionError(f"没有判定行 {name}：{lines}")


# ---------------------------------------------------------------- 手写运行根


class Stage:
    """按生产布局手写一个席位的结果行、账本行与录像目录。"""

    def __init__(self, root: Path, policy: str = "perceptual-framesamp-modul", seat: str = "s00", dirname: str | None = None):
        self.root, self.policy = Path(root), policy
        self.dir = self.root / seat / (dirname or policy)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, row: dict):
        with open(self.dir / name, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def result(self, ident: dict, aid: str, no: int, status: str, *, infra: bool = False, media: bool = True,
               **kw) -> dict:
        row = {"v8": True, "key": ident["key"], "task": ident["task"], "tier": ident["tier"], "seed": ident["seed"],
               "candidate": ident["candidate"], "spec_sha256": ident["spec_sha256"],
               "source_episode": ident.get("source_episode"),
               "identity": {k: ident[k] for k in ("tier", "seed", "candidate", "spec_sha256")},
               "policy": self.policy, "attempt_id": aid, "attempt_no": no, "status": status,
               "task_success": status == "success", "infra": infra, "exec_steps": 5,
               "rec_dir": str(self.dir / "rec" / f"{ident['key']}.a{no}"), "recorder_verify": "PASS"}
        row.update(kw)
        self._append("results.jsonl", row)
        if media:
            d = self.dir / "rec" / f"{ident['key']}.a{no}"
            d.mkdir(parents=True, exist_ok=True)
            for f in ("front.mkv", "wrist.mkv", "summary.json"):
                (d / f).write_text("x", encoding="utf-8")
        return row

    def ledger(self, kind: str, ident_or_key, aid: str, **kw):
        key = ident_or_key if isinstance(ident_or_key, str) else ident_or_key["key"]
        row = {"kind": kind, "key": key, "attempt_id": aid, "policy": self.policy, **kw}
        if kind == "accept":
            row.setdefault("accepted_attempt_id", aid)
        self._append(f"{self.policy}.ledger.jsonl", row)

    def accepted(self, ident: dict, aid: str, status: str, no: int = 1, **kw) -> dict:
        self.ledger("attempt_start", ident, aid, attempt_no=no)
        row = self.result(ident, aid, no, status, **kw)
        self.ledger("attempt_end", ident, aid, attempt_no=no, status=status)
        self.ledger("accept", ident, aid, status=status)
        return row


__all__ = [n for n in dir() if not n.startswith("_")] + ["REPO"]
