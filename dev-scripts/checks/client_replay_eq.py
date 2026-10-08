"""客户端回放等价核对 ``CLIENT_REPLAY_EQ``（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节「主会话」、三节闸门；
1006-rename-official-names-and-stage3-eval-plan.md 八.10 第 2 条改造：两侧检出目录 + sha 核对、两侧接口分别适配、
同 crash／零事件一律 FAIL、自检强制）。

目的：证明候选检出的客户端改动（如 R1 只改名）不改变策略行为。固定观测（确定性假环境）与固定服务回包（按调用
序号确定、与请求内容无关）下，分别用 ``--base`` 与 ``--candidate`` 两个检出目录里的客户端各跑同一组局，比较：

- ``request``：发给模型服务的每条消息。FrameSamp+Modulation、GroundSG 在 websocket 字节层截获
  （``websockets.sync.client.connect`` 换成进程内假连接，比的是 msgpack 打包后的原始字节 sha256）；SimpleMemVLA
  注入假连接，比消息对象的规范化字节；PonderPounce 注入假 vla-eval 连接，比协议帧载荷的规范化字节（EPISODE_END 的
  墙钟 ``elapsed_sec`` 剔除）。
- ``action``：交给环境 ``step`` 的每个动作的 dtype／shape／原始字节。
- ``control``：环境与服务调用的先后顺序与次数（reset、step、各消息类型）。
- ``terminal``：返回字典的 ``status``、``task_success``、``steps``。

新增的记录与媒体字段（轨迹文件、录像事件、返回字典里的其他键）列为允许差异，不参与判定。

每个检出在独立子进程里运行（``--_worker``），模块一律从该检出的 ``scripts/eval-official`` 按路径加载、``src`` 置于
``sys.path`` 最前，避免两个检出互相串味；子进程先核对 ``robomme_hard.__file__`` 落在该检出的 ``src`` 下。子进程不写
字节码缓存（``PYTHONDONTWRITEBYTECODE=1``），被比较的检出目录保持只读。

**两侧接口**：每个检出按文件是否存在判定接口——官方名接口（``framesamp_modul_client``／``groundsg_client``、配置键
``groundsg_variant``、数据集 ``hard-verify``）或改名前接口（模块名、配置键、数据集名取自 ``official_defs.py`` 的
``LEGACY_*`` 别名表）；同一路线在两侧各用本侧的名字驱动，比较的是行为而不是名字。

**检出身份**：``--base``／``--candidate`` 必须是检出目录，``--base-sha``／``--candidate-sha`` 必填；工具以
``git -C <目录> rev-parse HEAD`` 核对，不等即 ``CLIENT_REPLAY_SHA=FAIL``、不跑比较、退出 1。

路线（官方名）：``groundsg-oracle``（GroundSG oracle 变体；qwenvl 与 oracle 只差子目标预测器，客户端代码同一份）、
``smvla``（SimpleMemVLA）、``perceptual-framesamp-modul``（FrameSamp+Modulation）、``pp``（PonderPounce）。每条路线三个
场景：第 37 步成功（跨多次决策）、第 5 步环境异常、步数上限 40 触发超时。任一场景缺失、任一侧崩溃（**两侧同样崩溃
也算 FAIL**）、任一侧零环境 step 事件或零服务事件，都计 ``control_diff``。3-tier Astra（代码 ID astra）不在本工具覆盖范围（零外联夹具测试
覆盖）。

用法：
  uv run --no-sync python scripts/eval-official/client_replay_eq.py \
      --base <检出目录> --base-sha <40 位 sha> --candidate <检出目录> --candidate-sha <40 位 sha> \
      [--routes groundsg-oracle,smvla,perceptual-framesamp-modul,pp] [--third-party <含 mme-vla 的 third_party 目录>]
判定行（每路线一行，另有一条汇总行）：
  CLIENT_REPLAY_EQ=PASS|FAIL base=<sha> candidate=<sha> route=<r> request_diff=<n> action_diff=<n> control_diff=<n>
  terminal_diff=<n>
  CLIENT_REPLAY_EQ_SUMMARY=PASS|FAIL base=<sha> candidate=<sha> routes=<n> scenarios=<n> route_pass=<n> selftest_pass=<n>
自检**强制执行**（``--self-test`` 旗标保留以兼容旧命令，给不给都跑）：以「故意改动」的候选（在子进程内对动作、
请求、调用顺序各做一处篡改）跑一遍，三类都必须被抓到，输出 ``CLIENT_REPLAY_SELFTEST=PASS|FAIL``；自检不过整体 FAIL。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

ROUTES = ("groundsg-oracle", "smvla", "perceptual-framesamp-modul", "pp")
SCENARIOS = ({"name": "success", "done_at": 37, "raise_at": None, "max_steps": 200},
             {"name": "env_error", "done_at": None, "raise_at": 5, "max_steps": 200},
             {"name": "timeout", "done_at": None, "raise_at": None, "max_steps": 40})
TAMPERS = ("action", "request", "order")
H = W = 256  # 与真实环境同尺寸（官方录像器在小图上拼字条会尺寸不一致）
DEMO = 3
CHUNK = 20
#: groundsg 席位信息的模型种子（第三阶段 make_policy_context 必填，接口冻结说明 2.2／2.5；本轮真实运行只传 7）。
#: 两侧都给：改名前／第二阶段的 make_policy_context 不读该键，行为不受影响
POLICY_SEED = 7


# ── 公共：规范化与哈希 ────────────────────────────────────────────────────────


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canon(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.ndarray, np.generic)):
        a = np.ascontiguousarray(np.asarray(obj))
        return {"__array__": [a.dtype.str, list(a.shape), _sha(a.tobytes())]}
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"__bytes__": _sha(bytes(obj))}
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        return {"__float__": float.hex(obj)}
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return {"__repr__": type(obj).__name__}


def canon_sha(obj: Any) -> str:
    return _sha(json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


# ── 子进程：假环境、假服务、驱动 ──────────────────────────────────────────────


class Log:
    def __init__(self, tamper: str | None):
        self.events: list[list[Any]] = []
        self.tamper = tamper
        self.n_step = 0

    def add(self, *ev: Any) -> None:
        self.events.append(list(ev))


class FakeSession:
    """EnvSession 替身：``reset() -> (obs, info)``，``step(a) -> (obs, r, terminated, truncated, info)``。"""

    def __init__(self, log: Log, scen: dict, task: str = "PickXtimes", ep: int = 3):
        self.log, self.scen, self.task = log, scen, task
        self.seed = zlib.crc32(f"{task}:{ep}".encode())
        self.t = 0
        self.steps = 0
        self.recorder = None
        self.timing = {"demo_frames": DEMO}
        self.env = None

    def _frame(self, k, cam):
        import numpy as np
        return np.random.default_rng([self.seed, k, cam]).integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    def _obs(self, ks):
        import numpy as np
        return {"front_rgb_list": [self._frame(k, 0) for k in ks],
                "wrist_rgb_list": [self._frame(k, 1) for k in ks],
                "joint_state_list": [np.random.default_rng([self.seed, k, 7]).normal(size=7) for k in ks],
                "gripper_state_list": [np.random.default_rng([self.seed, k, 9]).uniform(size=2) for k in ks],
                "eef_state_list": [np.zeros(6) for _ in ks]}

    def reset(self):
        self.log.add("env", "reset")
        self.t = 0
        info = {"task_goal": ["pick up the cube 3 times"], "status": "ongoing",
                "simple_subgoal_online": "sg0", "grounded_subgoal_online": "gsg0"}
        return self._obs(range(DEMO + 1)), info

    def step(self, action):
        import numpy as np
        a = np.ascontiguousarray(np.asarray(action))
        if self.log.tamper == "action" and self.t == 2:
            a = a.copy(); a.reshape(-1)[0] += 1  # 自检用的故意改动
        self.log.add("env", "step", a.dtype.str, list(a.shape), _sha(a.tobytes()))
        self.t += 1
        self.steps += 1
        if self.scen["raise_at"] == self.t:
            raise RuntimeError("假环境第 %d 步异常" % self.t)
        done = self.scen["done_at"] is not None and self.t >= self.scen["done_at"]
        info = {"status": "success" if done else "ongoing", "simple_subgoal_online": f"sg{self.t}",
                "grounded_subgoal_online": f"gsg{self.t}"}
        return self._obs([DEMO + self.t]), 0.0, done, False, info


class NullRecorder:
    """录像器替身：记录类调用一律吞掉（记录与媒体字段属于允许差异）。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def __getattr__(self, name):
        return lambda *a, **k: None


def _actions(n: int):
    import numpy as np
    return (np.random.default_rng([n, 11]).uniform(-1, 1, size=(CHUNK, 8))).astype(np.float32)


class FakeWS:
    """``websockets.sync.client.connect`` 的返回值替身：按 msgpack 解包请求类型回确定性回包。"""

    def __init__(self, log: Log, counters: dict):
        from openpi_client import msgpack_numpy
        self.mp = msgpack_numpy
        self.log, self.c = log, counters
        self._pending = [self.mp.packb({"fake_server": True})]

    def send(self, data):
        obj = self.mp.unpackb(data)
        kind = "reset" if isinstance(obj, dict) and "reset" in obj else (
            "add_buffer" if isinstance(obj, dict) and "images" in obj else "infer")
        payload = bytes(data)
        if self.log.tamper == "request" and kind == "infer" and self.c["infer"] == 0:
            payload = payload + b"x"
        self.log.add("srv", kind, _sha(payload))
        if kind == "reset":
            rep = {"reset_finished": True}
        elif kind == "add_buffer":
            rep = {"add_buffer_finished": True}
        else:
            rep = {"actions": _actions(self.c["infer"])}
            self.c["infer"] += 1
        self._pending.append(self.mp.packb(rep))

    def recv(self, *a, **k):
        return self._pending.pop(0)

    def close(self, *a, **k):
        pass


def _load(root: Path, name: str):
    p = root / "scripts" / "eval-official" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"replay_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


#: 本工具自用的 official_defs 模块名：**不**占用 ``sys.modules["official_defs"]``——被比较检出里的客户端按
#: ``load_sibling("official_defs")`` 取「已加载则复用」，占了会让 base 侧用上本工具（candidate）的 official_defs
#: （第三阶段 make_args 必填 model_seed，base 侧随之崩溃），两侧就不再是各用各的代码
_TOOL_DEFS = "_client_replay_tool_official_defs"


def official_defs():
    """本工具同目录的 ``official_defs.py``（官方名与 ``LEGACY_*`` 别名表；不从被比较的检出里取）。"""
    mod = sys.modules.get(_TOOL_DEFS)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_TOOL_DEFS, Path(__file__).resolve().parent / "official_defs.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_TOOL_DEFS] = mod
        spec.loader.exec_module(mod)
    return mod


def side_interface(root: Path) -> dict:
    """按检出里实际存在的文件判定该侧接口：官方名优先，其次改名前的名字（取自别名表）；都不全即 ValueError。"""
    defs = official_defs()
    d = Path(root) / "scripts" / "eval-official"
    official = {"name": "official", "fsm_module": "framesamp_modul_client", "gsg_module": "groundsg_client",
                "variant_key": "groundsg_variant", "dataset": defs.DATASET_HARD_VERIFY}
    rev_mod = {new: old for old, new in defs.LEGACY_MODULE_ALIASES.items()}
    rev_key = {new: old for old, new in defs.LEGACY_CONFIG_KEY_ALIASES.items()}
    rev_ds = {new: old for old, new in defs.LEGACY_DATASET_ALIASES.items()}
    legacy = {"name": "legacy", "fsm_module": rev_mod[official["fsm_module"]],
              "gsg_module": rev_mod[official["gsg_module"]], "variant_key": rev_key[official["variant_key"]],
              "dataset": rev_ds[official["dataset"]]}
    for iface in (official, legacy):
        if (d / f"{iface['fsm_module']}.py").is_file() and (d / f"{iface['gsg_module']}.py").is_file():
            return iface
    raise ValueError(f"{d} 既不是官方名接口也不是改名前接口（客户端模块缺失）")


def _identity(scen, iface: dict):
    return {"task": "PickXtimes", "tier": "xhard0", "seed": 1234, "source_episode": 3, "builder_episode": 0,
            "key": "PickXtimes_xhard0_1234", "dataset": iface["dataset"], "attempt": 1}


def _patch_ws(log, counters):
    import websockets.sync.client as wsc
    wsc.connect = lambda *a, **k: FakeWS(log, counters)


def run_route(root: Path, route: str, scen: dict, tamper: str | None, tmp: Path, iface: dict | None = None) -> dict:
    iface = iface or side_interface(root)
    log = Log(tamper)
    sess = FakeSession(log, scen)
    ident = _identity(scen, iface)
    ep_dir = tmp / f"{ident['key']}.a1"
    ep_dir.mkdir(parents=True, exist_ok=True)
    rec = NullRecorder(ep_dir)
    conn_info = {"host": "127.0.0.1", "port": 1, "max_steps": scen["max_steps"], "dataset": iface["dataset"],
                 "trace_dir": str(ep_dir), "episode_tag": f"{ident['key']}.a1"}
    counters = {"infer": 0}
    if route == "perceptual-framesamp-modul":
        _patch_ws(log, counters)
        mod = _load(root, iface["fsm_module"])
        res = mod.run_episode(sess, ident, conn_info, rec)
    elif route == "groundsg-oracle":
        _patch_ws(log, counters)
        mod = _load(root, iface["gsg_module"])
        vkey = iface["variant_key"]
        ctx = mod.make_policy_context({vkey: "ground-sg-oracle", "port": 1, "max_steps": scen["max_steps"],
                                       "out": str(tmp), "policy_seed": POLICY_SEED})
        conn_info.update({"policy_context": ctx, vkey: "ground-sg-oracle"})
        res = mod.run_episode(sess, ident, conn_info, rec)
    elif route == "smvla":
        srv = _load(root, "smvla_server")
        mod = _load(root, "smvla_client")
        res = mod.run_episode(sess, ident, conn_info, rec, conn=SmvlaConn(log, srv, counters),
                              max_steps=scen["max_steps"])
    elif route == "pp":
        mod = _load(root, "pp_client")
        res = mod.run_episode(sess, ident, conn_info, rec, connection_factory=lambda url, t: PPConn(log, counters))
    else:
        raise ValueError(route)
    if tamper == "order":
        ev = log.events
        i = next((k for k in range(len(ev) - 1) if ev[k][0] != ev[k + 1][0]), None)
        if i is not None:
            ev[i], ev[i + 1] = ev[i + 1], ev[i]
    return {"events": log.events, "terminal": {k: res.get(k) for k in ("status", "task_success", "steps")}}


class SmvlaConn:
    """smvla 协议假连接（接口同 ``smvla_client.WSPolicyConn``）；回包指纹用该检出 ``smvla_server`` 的函数。"""

    def __init__(self, log: Log, srv, counters: dict):
        self.log, self.srv, self.c = log, srv, counters
        self.metadata = {"policy": "smvla", "fake": True}
        self.n = 0

    def call(self, msg: dict):
        import numpy as np
        raw = pickle.dumps((self.n, msg), protocol=4)
        self.n += 1
        kind = next(iter(msg))
        h = canon_sha(msg)
        if self.log.tamper == "request" and kind == "infer" and self.c["infer"] == 0:
            h = _sha(h.encode())
        self.log.add("srv", kind, h)
        if kind == "reset":
            rep = {"reset_finished": True, "rng": {"fake": True}}
        elif kind == "observe":
            frs = msg["observe"]["frames"]
            rep = {"observe_finished": True, "n": len(frs),
                   "frame_sha": [{k: self.srv.frame_sha(v) for k, v in sorted(f.items())} for f in frs]}
        else:
            p = msg["infer"]
            full = _actions(self.c["infer"])
            self.c["infer"] += 1
            rep = {"actions": full[:self.srv.EXECUTE_HORIZON], "actions_full": full, "subtask": f"sub{self.c['infer']}",
                   "infer_ms": 1.0, "recv_state_sha": self.srv.array_sha(np.asarray(p["state"])),
                   "recv_instruction_sha": self.srv.sha256_bytes(str(p["instruction"]).encode("utf-8"))}
        rep["req_sha"] = self.srv.sha256_bytes(raw)
        return rep, raw, b"r" + raw

    def close(self):
        pass


class PPConn:
    """vla-eval ``Connection`` 的用到部分；动作块按调用序号确定。"""

    def __init__(self, log: Log, counters: dict):
        self.log, self.c = log, counters

    def _frame(self, t, p):
        if t == "episode_end" and isinstance(p, dict):
            p = {k: v for k, v in p.items() if k != "elapsed_sec"}
        h = canon_sha(p)
        if self.log.tamper == "request" and t == "observation" and self.c["infer"] == 0:
            h = _sha(h.encode())
        self.log.add("srv", t, h)

    async def connect(self, *, benchmark=None):
        self._frame("hello", {"benchmark": benchmark})

    async def reconnect(self):
        self._frame("hello", {"benchmark": "<reconnect>"})

    async def start_episode(self, config):
        self._frame("episode_start", config)

    async def act(self, obs):
        import numpy as np
        self._frame("observation", obs)
        n = self.c["infer"]
        self.c["infer"] += 1
        return {"actions": (np.arange(10, dtype=np.float32).reshape(1, 10) * np.float32(0.001)
                            + np.float32(n) * np.float32(0.01))}

    async def end_episode(self, result):
        self._frame("episode_end", result)

    async def close(self):
        pass


def worker(args) -> int:
    sys.dont_write_bytecode = True  # 被比较的检出只读：不在其中留 __pycache__
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root / "src"))
    tp = Path(args.third_party).resolve()
    os.environ["SGEVAL_THIRD_PARTY"] = str(tp)
    sys.path.insert(1, str(tp / "mme-vla" / "packages" / "openpi-client" / "src"))
    import robomme_hard
    out = {"root": str(root), "robomme_hard": robomme_hard.__file__, "runs": {}}
    if not str(Path(robomme_hard.__file__).resolve()).startswith(str(root / "src")):
        out["import_error"] = f"robomme_hard 来自 {robomme_hard.__file__}，不在 {root}/src"
    try:
        iface = side_interface(root)
    except ValueError as e:
        out["import_error"] = str(e)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return 0
    out["iface"] = iface["name"]
    with tempfile.TemporaryDirectory(prefix="replay-") as td:
        for scen in SCENARIOS:
            try:
                out["runs"][scen["name"]] = run_route(root, args.route, scen, args.tamper, Path(td) / scen["name"],
                                                      iface)
            except Exception as e:  # 驱动本身崩溃也是一种可比较的结果
                out["runs"][scen["name"]] = {"crash": f"{type(e).__name__}: {e}"[:400]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


# ── 主进程：两侧各跑、比较 ────────────────────────────────────────────────────


def _run_side(root: Path, route: str, third_party: Path, tamper: str | None, py: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    cmd = [py, str(Path(__file__).resolve()), "--_worker", "--root", str(root), "--route", route,
           "--third-party", str(third_party), "--out", out]
    if tamper:
        cmd += ["--tamper", tamper]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # 被比较的检出只读：不在其中留 __pycache__
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    try:
        data = json.loads(Path(out).read_text(encoding="utf-8"))
    except Exception:
        data = {"worker_failed": f"rc={p.returncode} {p.stderr[-800:]}"}
    finally:
        Path(out).unlink(missing_ok=True)
    return data


def _count(events: list, kind: str, name: str | None = None) -> int:
    return sum(1 for e in events if e[0] == kind and (name is None or e[1] == name))


def compare(a: dict, b: dict) -> dict:
    """两侧一条路线的比较。场景集合以 ``SCENARIOS`` 为准；缺场景、任一侧崩溃（两侧同样崩溃也算）、任一侧零环境
    step 事件或零服务事件都计 ``control_diff``，不跳过。"""
    d = {"request_diff": 0, "action_diff": 0, "control_diff": 0, "terminal_diff": 0, "notes": []}
    for tag, side in (("base", a), ("cand", b)):
        if "worker_failed" in side or "import_error" in side:
            d["control_diff"] += 1
            d["notes"].append(f"{tag}: {side.get('worker_failed') or side.get('import_error')}")
    if d["notes"]:
        return d
    for name in [sc["name"] for sc in SCENARIOS]:
        ra, rb = a.get("runs", {}).get(name), b.get("runs", {}).get(name)
        if ra is None or rb is None:
            d["control_diff"] += 1
            d["notes"].append(f"{name}: 场景缺失 base={ra is not None} cand={rb is not None}")
            continue
        if "crash" in ra or "crash" in rb:
            d["control_diff"] += 1
            same = ra.get("crash") == rb.get("crash")
            d["notes"].append(f"{name}: 驱动崩溃（{'两侧相同' if same else '两侧不同'}）base={ra.get('crash')} "
                              f"cand={rb.get('crash')}")
            continue
        ea, eb = ra["events"], rb["events"]
        empty = [tag for tag, ev in (("base", ea), ("cand", eb))
                 if _count(ev, "env", "step") == 0 or _count(ev, "srv") == 0]
        if empty:
            d["control_diff"] += 1
            d["notes"].append(f"{name}: 零事件 sides={','.join(empty)}（环境 step 或服务调用为 0）")
            continue
        if [e[:2] for e in ea] != [e[:2] for e in eb]:
            d["control_diff"] += 1
            d["notes"].append(f"{name}: 调用序列不同（{len(ea)} vs {len(eb)} 个事件）")
        for x, y in zip(ea, eb):
            if x[:2] != y[:2]:
                continue
            if x[0] == "env" and x[1] == "step" and x[2:] != y[2:]:
                d["action_diff"] += 1
            elif x[0] == "srv" and x[2:] != y[2:]:
                d["request_diff"] += 1
        if ra["terminal"] != rb["terminal"]:
            d["terminal_diff"] += 1
            d["notes"].append(f"{name}: 终态 {ra['terminal']} vs {rb['terminal']}")
    return d


def is_clean(d: dict) -> bool:
    return not any(d[k] for k in ("request_diff", "action_diff", "control_diff", "terminal_diff"))


def _git_sha(root: Path) -> str:
    """检出目录的完整 HEAD sha；取不到返回 ``unknown``。"""
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_checkouts(pairs: list[tuple[str, Path, str]]) -> list[str]:
    """[(侧名, 目录, 期望 sha)] → 不符说明列表（空表示全部相符）。期望 sha 须为 40 位、目录须存在且 HEAD 相等。"""
    bad = []
    for side, root, want in pairs:
        if not root.is_dir():
            bad.append(f"side={side} dir_missing={root}")
            continue
        got = _git_sha(root)
        if len(want or "") != 40 or got != want:
            bad.append(f"side={side} want={want} got={got} dir={root}")
    return bad


def _default_third_party() -> str:
    env = os.environ.get("SGEVAL_THIRD_PARTY")
    return env or str(Path(__file__).resolve().parents[2] / "third_party")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", help="base 检出目录")
    ap.add_argument("--base-sha", help="base 检出应有的 HEAD（40 位）")
    ap.add_argument("--candidate", help="candidate 检出目录")
    ap.add_argument("--candidate-sha", help="candidate 检出应有的 HEAD（40 位）")
    ap.add_argument("--routes", default=",".join(ROUTES))
    ap.add_argument("--third-party", default=_default_third_party())
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--self-test", action="store_true", help="兼容旧命令；自检一律强制执行")
    ap.add_argument("--_worker", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--route")
    ap.add_argument("--tamper", choices=TAMPERS)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args._worker:
        return worker(args)
    missing = [f for f in ("base", "base_sha", "candidate", "candidate_sha") if not getattr(args, f)]
    if missing:
        ap.error(f"缺少 {', '.join('--' + m.replace('_', '-') for m in missing)}")
    base, cand = Path(args.base).resolve(), Path(args.candidate).resolve()
    tp = Path(args.third_party).resolve()
    routes = [r for r in args.routes.split(",") if r]
    bad = [r for r in routes if r not in ROUTES]
    if bad or not routes:
        print(f"未知路线 {bad}；可选 {ROUTES}")
        return 2
    sha_bad = check_checkouts([("base", base, args.base_sha), ("candidate", cand, args.candidate_sha)])
    for b in sha_bad:
        print(f"CLIENT_REPLAY_SHA=FAIL {b}", flush=True)
    bs, cs = args.base_sha, args.candidate_sha
    if sha_bad:
        print(f"CLIENT_REPLAY_EQ_SUMMARY=FAIL base={bs} candidate={cs} routes={len(routes)} scenarios=0 route_pass=0 "
              f"selftest_pass=0 reason=sha_mismatch", flush=True)
        return 1
    print(f"CLIENT_REPLAY_SHA=PASS base={bs} candidate={cs}", flush=True)
    route_pass = selftest_pass = 0
    for r in routes:
        a = _run_side(base, r, tp, None, args.python)
        b = _run_side(cand, r, tp, None, args.python)
        print(f"CLIENT_REPLAY_IFACE route={r} base={a.get('iface')} candidate={b.get('iface')}", flush=True)
        d = compare(a, b)
        ok = is_clean(d)
        route_pass += ok
        for n in d["notes"][:8]:
            print(f"CLIENT_REPLAY_NOTE route={r} {n}")
        print(f"CLIENT_REPLAY_EQ={'PASS' if ok else 'FAIL'} base={bs} candidate={cs} route={r} "
              f"request_diff={d['request_diff']} action_diff={d['action_diff']} control_diff={d['control_diff']} "
              f"terminal_diff={d['terminal_diff']}", flush=True)
        caught = {}
        for t in TAMPERS:
            caught[t] = not is_clean(compare(a, _run_side(cand, r, tp, t, args.python)))
        st = ok and all(caught.values())  # 基线本身不等时自检无意义，一并判 FAIL
        selftest_pass += st
        print(f"CLIENT_REPLAY_SELFTEST={'PASS' if st else 'FAIL'} route={r} "
              + " ".join(f"{t}={'caught' if v else 'missed'}" for t, v in caught.items()), flush=True)
    all_ok = route_pass == len(routes) and selftest_pass == len(routes)
    print(f"CLIENT_REPLAY_EQ_SUMMARY={'PASS' if all_ok else 'FAIL'} base={bs} candidate={cs} routes={len(routes)} "
          f"scenarios={len(routes) * len(SCENARIOS)} route_pass={route_pass} selftest_pass={selftest_pass}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
