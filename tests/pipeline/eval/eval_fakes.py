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


def env_session():
    """拆仓后 ``EnvSession``、``NullRecorder``、``StepCapReached``、``RecorderError``、``ResetBudgetExhausted`` 在评估包
    ``robomme_ood_eval.session``（原在 env_client.py）。"""
    from robomme_ood_eval import session

    return session


def framesamp_modul_client():
    return load_script("eval-official/framesamp_modul_client.py")


def smvla_client():
    return load_script("eval-official/smvla_client.py")


def smvla_server():
    return load_script("eval-official/smvla_server.py")


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


def episode_mod():
    from robomme_ood_eval import episode as E

    return E


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


__all__ = [n for n in dir() if not n.startswith("_")] + ["REPO"]
