"""GroundSG（S3）测试的公共替身：假环境、两种 builder、假 MME-VLA 服务与连接、不加载权重的 swift 替身、两侧驱动。

设计口径：
- 官方源码只读引用 ``SGEVAL_THIRD_PARTY``，未设时取当前检出的 ``third_party``（worktree 里子模块目录为空，须显式指向主检出）；缺官方源码即失败，不跳过。
- 生产模块一律经 ``tests._support.loaders.load_script`` 按路径加载，不往 sys.modules 注入替身。
- 假环境的下一帧由「执行的动作字节 + 步号」确定性生成，假服务的动作块由「本局 reset 之后收到的全部请求指纹」
  确定性生成：两侧任何一个请求或动作不同，后面的帧、请求、动作都会随之不同（差异可观测，等式非平凡）。
- 假服务的指纹函数在本文件手写，不调用被测代码（``official_defs.canonical_bytes``）。
- swift 替身：``PtEngine`` 构造只记参数、不读任何权重；``infer`` 的回复由请求文本与所附图片字节确定性生成
  （MemER 请求——system prompt 含 ``keyframe_positions``——回 JSON；也可给 ``script`` 逐次指定回复原文）。
- 第三阶段（1006 计划八.3）：三个变体都带 ``policy_seed``（缺省 7），MemER 走同一套两侧装配。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tests._support.loaders import REPO, load_script

#: 假帧形状 H×W：宽度取真实相机的 256，使官方叠字录像的文字区行数与真实运行同为固定高度（窄帧会让每个词各占一行、
#: 帧高随文字变化，官方 mimsave 报「All images in a movie should have same size」）；高度取 8 以省内存
H, W = 8, 256
N_RESET_FRAMES = 3  # 2 帧演示 + 1 帧初始
CHUNK_ROWS = 20  # 假服务每次回的动作行数（官方只执行前 16 行）
EXEC_HORIZON = 16  # 官方 Args.obs_horizon（手写）
THIRD_PARTY_ENV = "SGEVAL_THIRD_PARTY"
ORACLE = "ground-sg-oracle"
QWENVL = "ground-sg-qwenvl"
MEMER = "ground-sg-memer"
VARIANTS = (ORACLE, QWENVL, MEMER)
DATASET = "hard-verify"
POLICY_SEED = 7  # 本版真实运行只传 7（计划〇.1）


def official_dir() -> Path:
    # 未设时取当前检出的 third_party（主检出日常门禁走此路）；worktree 里子模块为空，须显式指向主检出，否则下面断言失败（不 skip）。
    tp = os.environ.get(THIRD_PARTY_ENV) or str(REPO / "third_party")
    d = Path(tp) / "mme-vla" / "examples" / "robomme"
    assert (d / "eval.py").is_file(), f"官方源码不在 {d}"
    return d


def official_sha256() -> dict[str, str]:
    d = official_dir()
    rels = ["eval.py", "env_runner.py", "utils.py", "subgoal_predictor.py", "subgoal_prediction/qwenvl/api.py"]
    return {r: hashlib.sha256((d / r).read_bytes()).hexdigest() for r in rels}


def print_official_sha() -> None:
    for rel, s in official_sha256().items():
        print(f"OFFICIAL_SOURCE {rel} sha256={s}")


# ---------------------------------------------------------------- 生产模块


def env_client():
    return load_script("eval-official/env_client.py")


def groundsg_client():
    return load_script("eval-official/groundsg_client.py")


def official_defs():
    return load_script("eval-official/official_defs.py")


def official_hard_runner():
    return load_script("eval-official/official_hard_runner.py")


# ---------------------------------------------------------------- 假环境


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frame(v: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[...] = int(v) % 256
    img[0, 0] = [(int(v) * 7) % 256, (int(v) * 13) % 256, 1]
    return img


def obs_of(vals: list[int]) -> dict:
    return {
        "front_rgb_list": [frame(v) for v in vals],
        "wrist_rgb_list": [frame(v + 1) for v in vals],
        "joint_state_list": [np.full(7, v / 10.0, dtype=np.float64) for v in vals],
        "gripper_state_list": [np.array([v / 100.0, v / 100.0], dtype=np.float64) for v in vals],
        "eef_state_list": [np.zeros(6, dtype=np.float64) for _ in vals],
    }


def subgoal_at(n: int) -> str:
    """Oracle 的 grounded 子目标：每 25 步换一次（带坐标，官方叠字会画点）。"""
    k = n // 25
    return f"pick up the cube at <{(k * 37) % 250}, {(k * 91) % 250}>"


def goal_of(task: str, ep: int) -> str:
    return f"goal {task} {ep}"


@dataclasses.dataclass
class Plan:
    """假环境行为：第 n 次 step 报 success／fail、报终止却不给 status（官方得 unknown）、或抛异常；都为空则永不终止。"""

    success_at: int | None = None
    fail_at: int | None = None
    unknown_at: int | None = None
    raise_at: int | None = None


class FakeEnv:
    def __init__(self, task: str, ep: int, plan: Plan):
        self.task, self.ep, self.plan = task, int(ep), plan
        self.n = 0
        self.actions: list[np.ndarray] = []
        self.resets = 0
        self.closed = False
        self.difficulty = "hard"
        self.unwrapped = self

    def _info(self, status: str | None) -> dict:
        info = {"grounded_subgoal_online": subgoal_at(self.n), "simple_subgoal_online": f"simple {self.n // 25}"}
        if status is not None:
            info["status"] = status
        return info

    def reset(self):
        self.resets += 1
        base = (self.ep * 7) % 150
        info = self._info("ongoing")
        info["task_goal"] = [goal_of(self.task, self.ep), "alt"]
        return obs_of([base + i for i in range(N_RESET_FRAMES)]), info

    def step(self, action):
        a = np.array(action, copy=True)
        self.actions.append(a)
        self.n += 1
        if self.plan.raise_at == self.n:
            raise RuntimeError(f"fake env step failure at {self.n}")
        v = int(sha(np.ascontiguousarray(a).tobytes() + str(self.n).encode())[:6], 16) % 250
        status: str | None = "ongoing"
        terminated = False
        if self.plan.success_at == self.n:
            status, terminated = "success", True
        elif self.plan.fail_at == self.n:
            status, terminated = "fail", True
        elif self.plan.unknown_at == self.n:
            status, terminated = None, True
        return obs_of([v]), 0.0, terminated, False, self._info(status)

    def close(self):
        self.closed = True


class World:
    """一组假环境：按 source_episode 给 Plan，记下全部假环境与官方 builder 的构造参数。"""

    def __init__(self, plans: dict[int, Plan] | None = None, default: Plan | None = None):
        self.plans = dict(plans or {})
        self.default = default or Plan(success_at=40)
        self.envs: list[FakeEnv] = []
        self.builder_kwargs: list[dict] = []

    def new_env(self, task: str, source_episode: int) -> FakeEnv:
        env = FakeEnv(task, source_episode, self.plans.get(int(source_episode), self.default))
        self.envs.append(env)
        return env

    def official_builder_cls(self):
        """官方 ``env_runner.py`` 的 ``BenchmarkEnvBuilder`` 替身（构造参数照官方关键字记下）。"""
        world = self

        class FakeOfficialBuilder:
            def __init__(self, env_id, dataset, action_space, gui_render, max_steps):
                self.env_id = env_id
                world.builder_kwargs.append(dict(env_id=env_id, dataset=dataset, action_space=action_space,
                                                 gui_render=gui_render, max_steps=max_steps))

            def get_episode_num(self):
                return 50

            def make_env_for_episode(self, episode_id):
                return world.new_env(self.env_id, int(episode_id))

        return FakeOfficialBuilder


class NewSideBuilder:
    """新侧 ``EnvSession`` 用的 builder 替身：builder_episode → source_episode 的映射由调用方给出（手写）。"""

    def __init__(self, task: str, world: World, ep_map: dict[int, int]):
        self.task, self.world, self.ep_map = task, world, dict(ep_map)

    def make_env_for_episode(self, ep, max_steps=None):
        return self.world.new_env(self.task, self.ep_map[int(ep)])


# ---------------------------------------------------------------- 假 MME-VLA 服务


def _arr_fp(a: Any) -> bytes:
    a = np.ascontiguousarray(np.asarray(a))
    return f"{a.dtype.str}{a.shape}".encode() + a.tobytes()


def request_fp(obj: dict) -> tuple[str, str]:
    """（手写）请求指纹：返回 (种类, sha256)。"""
    if obj.get("reset"):
        return "reset", sha(b"reset")
    if obj.get("add_buffer"):
        return "add_buffer", sha(_arr_fp(obj["images"]) + _arr_fp(obj["state"]) + str(obj["exec_start_idx"]).encode())
    parts = [_arr_fp(obj["observation/image"]), _arr_fp(obj["observation/wrist_image"]),
             _arr_fp(obj["observation/state"]), str(obj.get("prompt")).encode(),
             str(obj.get("grounded_subgoal")).encode(), str(obj.get("simple_subgoal")).encode()]
    return "infer", sha(b"|".join(parts))


class FakeServer:
    """动作块由本局 reset 之后的全部请求指纹确定性生成；记下全部请求与发出的动作块。"""

    def __init__(self):
        self.log: list[tuple[str, str, dict]] = []
        self.chunks: list[np.ndarray] = []
        self._h = hashlib.sha256()

    def handle(self, obj: dict) -> dict:
        kind, fp = request_fp(obj)
        self.log.append((kind, fp, {k: obj[k] for k in obj if k in ("prompt", "grounded_subgoal", "simple_subgoal",
                                                                      "exec_start_idx")}))
        if kind == "reset":
            self._h = hashlib.sha256()
            return {"reset_finished": True}
        self._h.update(fp.encode())
        if kind == "add_buffer":
            self.log[-1][2]["n_frames"] = int(np.asarray(obj["images"]).shape[0])
            return {"add_buffer_finished": True}
        seed = int(self._h.hexdigest()[:8], 16)
        acts = np.random.default_rng(seed).standard_normal((CHUNK_ROWS, 8)).astype(np.float32)
        self.chunks.append(acts.copy())
        return {"actions": acts}


class FakeClient:
    """``MMEVLAWebsocketClientPolicy`` 的进程内替身（接口：reset／add_buffer／infer）。"""

    def __init__(self, server: FakeServer):
        self.server = server
        self._ws = None

    def reset(self):
        return self.server.handle({"reset": True})

    def add_buffer(self, buf):
        return self.server.handle(buf)

    def infer(self, obs):
        return self.server.handle(obs)


# ---------------------------------------------------------------- swift 替身（不加载权重）


def memer_reply(h: int, n_images: int) -> str:
    """（手写）MemER 替身回复：合法 JSON；当前输入图多于 1 张且 h%3==0 时挑第 1+h%n 张为关键帧，否则空列表。"""
    pos = [1 + h % n_images] if n_images > 1 and h % 3 == 0 else []
    return json.dumps({"current_subtask": f"pick up the cube at <|box_start|>({h % 1000},{(h // 1000) % 1000})<|box_end|>",
                       "keyframe_positions": pos})


class FakeSwift:
    """``swift.llm`` 三个名字的替身。``PtEngine`` 构造只记参数；``infer`` 的回复由请求确定性生成。

    ``script``：逐次指定回复原文的列表（用完后回落到确定性生成）；``InferRequest`` 与真实 swift 一样把
    ``messages``／``images``／``videos``／``objects`` 放在同名属性上（另留 ``kw`` 原样）。"""

    def __init__(self, script: list[str] | None = None):
        self.engines: list[dict] = []
        self.requests: list[dict] = []
        self.script = list(script or [])
        outer = self

        class InferRequest:
            def __init__(self, **kw):
                self.kw = kw
                self.messages = kw.get("messages")
                self.images = list(kw.get("images") or [])
                self.videos = list(kw.get("videos") or [])
                self.objects = kw.get("objects") or {}

        class RequestConfig:
            def __init__(self, **kw):
                self.kw = kw

        class PtEngine:
            def __init__(self, model_id_or_path, adapters, attn_impl):
                outer.engines.append(dict(model_id_or_path=model_id_or_path, adapters=list(adapters),
                                          attn_impl=attn_impl))

            def infer(self, reqs, request_config=None):
                (req,) = reqs
                kw = req.kw
                imgs = [Path(x).read_bytes() for x in kw["images"]]
                img = imgs[0]
                text = json.dumps(kw["messages"], sort_keys=True)
                rec = {"messages": kw["messages"], "image_sha": sha(img), "has_video": "videos" in kw,
                       "objects": kw.get("objects"), "config": dict(request_config.kw),
                       "image_shas": [sha(b) for b in imgs]}
                outer.requests.append(rec)
                h = int(sha(text.encode() + b"".join(imgs))[:6], 16)
                if outer.script:
                    content = outer.script.pop(0)
                elif "keyframe_positions" in kw["messages"][0]["content"]:
                    content = memer_reply(h, len(imgs))
                else:
                    content = f"pick up the cube at <|box_start|>({h % 1000},{(h // 1000) % 1000})<|box_end|>"
                msg = type("M", (), {"content": content})()
                choice = type("C", (), {"message": msg})()
                return [type("R", (), {"choices": [choice]})()]

        self.names = {"PtEngine": PtEngine, "InferRequest": InferRequest, "RequestConfig": RequestConfig}


# ---------------------------------------------------------------- 两侧驱动


ADAPTER = "/fake/qwenvl/grounded_subgoal/checkpoint-1200"
MEMER_ADAPTER = "/fake/memer/grounded_subgoal/checkpoint-1300"


def adapters_of(variant: str) -> dict:
    """变体对应的 adapter 关键字（手写配对：QwenVL 只给 QwenVL 的，MemER 只给 MemER 的，Oracle 都不给）。"""
    return {"qwenvl_groundSG_adapter_path": ADAPTER if variant == QWENVL else None,
            "memer_adapter_path": MEMER_ADAPTER if variant == MEMER else None}


def identity(task: str = "PickXtimes", source_episode: int = 3, builder_episode: int = 0, seed: int = 510300) -> dict:
    return {"task": task, "tier": "xhard0", "seed": seed, "candidate": None, "builder_episode": builder_episode,
            "source_episode": source_episode, "spec_sha256": None, "key": f"{task}_xhard0_{seed}"}


def seat_info(variant: str, max_steps: int, tmp: Path, port: int = 18120, policy_seed: int = POLICY_SEED) -> dict:
    return {"policy": "groundsg", "seat": "00", "host": "127.0.0.1", "port": port, "dataset": DATASET,
            "max_steps": max_steps, "strict_cap": False, "groundsg_variant": variant, **adapters_of(variant),
            "policy_seed": policy_seed, "trace_root": str(tmp / "trace"), "out": str(tmp)}


class NewSide:
    """新侧：真实 ``EnvSession`` + ``groundsg_client``（假环境、假服务、swift 替身）。"""

    def __init__(self, variant: str, max_steps: int, tmp: Path, world: World, *, strict_cap: bool = False,
                 port: int = 18120, real_client: bool = False, policy_seed: int = POLICY_SEED,
                 swift: FakeSwift | None = None, server: FakeServer | None = None):
        """``real_client=True`` 时用生产默认的客户端工厂（真实 websocket 客户端，连 ``port`` 上的回环假服务）。"""
        self.variant, self.max_steps, self.tmp, self.world, self.strict_cap = variant, max_steps, tmp, world, strict_cap
        self.port = port
        self.server = server or FakeServer()
        self.swift = swift or FakeSwift()
        self.mc = groundsg_client()
        factory = None if real_client else (lambda h, p, ep: FakeClient(self.server))
        self.ctx = self.mc.make_policy_context(seat_info(variant, max_steps, tmp, port, policy_seed),
                                               client_factory=factory, qwen_extra=self.swift.names)
        self.sessions: list[Any] = []

    def run(self, ident: dict, *, attempt: int = 1) -> dict:
        ec = env_client()
        sess = ec.EnvSession(ident["task"], ident["builder_episode"], max_steps=self.max_steps,
                             builder=NewSideBuilder(ident["task"], self.world,
                                                    {ident["builder_episode"]: ident["source_episode"]}),
                             step_cap=self.max_steps if self.strict_cap else None, dataset=DATASET)
        sess.build()
        tag = f"{ident['key']}.a{attempt}"
        conn = {"host": "127.0.0.1", "port": self.port, "max_steps": self.max_steps, "policy": "groundsg", "seat": "00",
                "dataset": DATASET, "strict_cap": self.strict_cap, "groundsg_variant": self.variant,
                **adapters_of(self.variant),
                "trace_root": str(self.tmp / "trace"), "trace_dir": str(self.tmp / "trace" / tag),
                "episode_tag": tag, "rec_dir": str(self.tmp / "rec" / tag), "policy_context": self.ctx}
        res = self.mc.run_episode(sess, ident, conn, ec.NullRecorder())
        sess.close()
        self.sessions.append(sess)
        res["_session_steps"] = sess.steps
        res["_cap_hit"] = sess.cap_hit
        return res


class OrigSide:
    """原侧：``official_hard_runner``（官方 ``EnvRunner`` 摘取原文 + 假 builder、假服务、swift 替身）。"""

    def __init__(self, variant: str, max_steps: int, tmp: Path, world: World, *, port: int = 18120,
                 real_client: bool = False, policy_seed: int = POLICY_SEED, swift: FakeSwift | None = None,
                 server: FakeServer | None = None):
        self.variant, self.max_steps, self.tmp, self.world = variant, max_steps, tmp, world
        self.server = server or FakeServer()
        self.swift = swift or FakeSwift()
        self.ohr = official_hard_runner()
        factory = None if real_client else (lambda h, p, ep: FakeClient(self.server))
        self.ctx = self.ohr.make_context(variant, host="127.0.0.1", port=port, max_steps=max_steps,
                                         policy_seed=policy_seed, adapter=ADAPTER if variant == QWENVL else None,
                                         memer_adapter=MEMER_ADAPTER if variant == MEMER else None,
                                         builder_cls=world.official_builder_cls(), scratch_root=tmp,
                                         client_factory=factory, qwen_extra=self.swift.names)

    def run(self, ident: dict, *, attempt: int = 1) -> dict:
        return self.ohr.run_identity(self.ctx, ident, out=self.tmp, attempt=attempt)


def read_trace(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


#: 语言账本打开时 step 行的关联字段（call_id 是各侧账本自己的编号，两侧逐项比较时不比）
LANG_LINK_FIELDS = ("source_call_id", "chunk_index")


def trace_parts(rows: list[dict]) -> dict:
    """轨迹按种类拆开；去掉两侧必然不同的字段（header.route、end.side、step 的语言账本关联编号）。"""
    out: dict[str, list] = {"request": [], "response": [], "step": [], "demo": [], "history": [], "end": []}
    for r in rows:
        k = r["kind"]
        if k in out:
            r = dict(r)
            r.pop("side", None)
            for f in LANG_LINK_FIELDS:
                r.pop(f, None)
            out[k].append(r)
    return out


_BASE_TRACE_PARTS = trace_parts  # 供 legacy_trace_parts 调用（被 monkeypatch 换掉 trace_parts 后仍指向原函数）

#: 第二阶段 S1 新侧 end 行只增的字段（原侧 R1 不写）
NEW_ONLY_END = ("steps_attempted", "steps_observed", "frames_recorded", "omitted_timeout_frames", "no_frame",
                "official_source", "official_videos")


def legacy_trace_parts(rows: list[dict]) -> dict:
    """新侧契约字段还原成 BASE 口径后再按种类拆开（原侧行原样；1005 计划 S1）。

    只剥离新侧新增的记录字段，且是可逆映射：end 行官方原返回值 ``success_flag`` 还回 ``terminal_reason``、
    删 ``NEW_ONLY_END``；缺观测步（``observed=false``）还回旧写法。``status``、动作、画面、请求一律不动，差异照报。
    两侧逐项比较时用 ``monkeypatch.setattr(F, "trace_parts", F.legacy_trace_parts)`` 局部替换。"""
    conv = []
    for r in rows:
        r = dict(r)
        if r.get("kind") == "end" and "success_flag" in r:
            r["terminal_reason"] = r.pop("success_flag")
            for k in NEW_ONLY_END:
                r.pop(k, None)
        if r.get("kind") == "step" and r.get("observed") is False:
            assert r.pop("missing_reason")
            r.pop("observed")
            r.update(terminated=False, truncated=False, status="error")
        conv.append(r)
    return _BASE_TRACE_PARTS(conv)


# ---------------------------------------------------------------- 两侧逐项比较


def seq_diff(a: list, b: list) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def diffs(new_side, orig_side) -> dict:
    (new, wn, rn), (orig, wo, ro) = new_side, orig_side
    tn = trace_parts(read_trace(rn["trace_path"]))
    to = trace_parts(read_trace(ro["trace_path"]))
    payload = (seq_diff([(r["name"], r["sha256"], r["step"]) for r in tn["request"]],
                        [(r["name"], r["sha256"], r["step"]) for r in to["request"]])
               + seq_diff([x[:2] for x in new.server.log], [x[:2] for x in orig.server.log])
               + seq_diff(tn["response"], to["response"]) + seq_diff(tn["demo"], to["demo"])
               + seq_diff(tn["history"], to["history"]) + seq_diff(new.swift.requests, orig.swift.requests))
    acts_n = [a.tobytes() for e in wn.envs for a in e.actions]
    acts_o = [a.tobytes() for e in wo.envs for a in e.actions]
    exec_ = seq_diff(acts_n, acts_o) + seq_diff(tn["step"], to["step"])
    term = int((rn["status"], rn["steps"], rn["error"], rn["success_flag"])
               != (ro["status"], ro["exec_steps"], ro["error"], ro["success_flag"])) + seq_diff(tn["end"], to["end"])
    if new.variant == MEMER:  # MemER 的子目标模型请求另比附图逐张字节（关键帧 + 最近帧）
        payload += seq_diff([r["image_shas"] for r in new.swift.requests], [r["image_shas"] for r in orig.swift.requests])
    return {"payload": payload, "exec": exec_, "terminal": term, "n_req": len(tn["request"]), "n_exec": len(acts_n)}


# ---------------------------------------------------------------- 回环 websocket 假服务（slow 用例）


class LoopbackServer:
    """``127.0.0.1`` 上的 websocket 假服务：握手发元数据，之后逐条 msgpack 解包交给 ``FakeServer``、回包。"""

    def __init__(self, server: FakeServer):
        import threading

        import websockets.sync.server as wss
        from openpi_client import msgpack_numpy

        self.fake = server
        packer = msgpack_numpy.Packer()

        def handler(ws):
            ws.send(packer.pack({"server": "fake"}))
            try:
                for msg in ws:
                    ws.send(packer.pack(self.fake.handle(msgpack_numpy.unpackb(msg))))
            except Exception:  # noqa: BLE001 客户端断开
                pass

        self._srv = wss.serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
        self.port = self._srv.socket.getsockname()[1]
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()

    def close(self):
        self._srv.shutdown()
        self._t.join(timeout=10)
