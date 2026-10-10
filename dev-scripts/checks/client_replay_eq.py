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

每个检出在独立子进程里运行（``--_worker``），避免两个检出互相串味；子进程不写字节码缓存（``PYTHONDONTWRITEBYTECODE=1``），
被比较的检出目录保持只读。

**两种布局、三种接口**（按检出里实际存在的文件判定）：

* 拆仓后的评估仓（``pkg``）：四个模型客户端在 ``src/robomme_ood_eval/models/{framesamp_modul,groundsg,smvla,pp}.py``、
  SimpleMemVLA 服务在 ``src/robomme_ood_eval/servers/smvla_server.py``，按包名导入（原模块级函数 ``run_episode``／
  ``make_policy_context`` 保留，本工具调的就是它们）；``src`` 与 benchmark 子模块的 ``src`` 置于 ``sys.path`` 最前，
  子进程先核对 ``robomme_ood_eval.__file__`` 落在该检出的 ``src`` 下（子模块未检出时用 ``--bench-src``）；
* 旧仓（``scripts``）：模块从该检出的旧评估目录按路径加载（官方名接口或改名前接口，后者的模块名、配置键、数据集名
  取自别名表 ``LEGACY_*``），子进程核对 ``robomme_ood.__file__`` 落在该检出的 ``src`` 下。

同一路线在两侧各用本侧的名字驱动，比较的是行为而不是名字。

**常驻两局**：每个用例在同一个子进程里、用同一份模块与策略上下文（GroundSG 的 ``make_policy_context`` 只建一次）连跑
两局——成功后再一局、环境异常后再一局、超时后再一局；两局的事件中间插局界 ``["ep", i]``，终态逐局比较。

**检出身份**：``--base``／``--candidate`` 必须是检出目录，``--base-sha``／``--candidate-sha`` 必填；工具以
``git -C <目录> rev-parse HEAD`` 核对，不等即 ``CLIENT_REPLAY_SHA=FAIL``、不跑比较、退出 1。

路线（官方名）：``groundsg-oracle``（GroundSG oracle 变体；qwenvl 与 oracle 只差子目标预测器，客户端代码同一份）、
``smvla``（SimpleMemVLA）、``perceptual-framesamp-modul``（FrameSamp+Modulation）、``pp``（PonderPounce），各三个用例
（见 ``CASES``：成功两局、环境异常后下一局、超时后下一局；单局场景：第 37 步成功、第 5 步环境异常、步数上限 40 超时）。
任一用例缺失、任一侧崩溃（**两侧同样崩溃也算 FAIL**）、任一侧零环境 step 事件或零服务事件，都计 ``control_diff``。

``astra``（3-tier Astra）是零外联探针：两侧各自在禁网（``socket.connect``／``create_connection`` 一律记录并拒绝）的子进程里
导入 Astra 驱动模块（``pkg``：``robomme_ood_eval.models.astra``；``scripts``：旧 ``astra_hard_runner``），以旧的模块级
函数为准调 ``check_pairing``（两数据集 × 两步数）与 ``check_policy_seed``（合法与非法输入），比较结果；任一侧有外联尝试
即 FAIL。Astra 的完整一局（规划／监视／VLA）需要真实服务与计费接口，不在本工具里跑。

用法：
  uv run --no-sync python dev-scripts/checks/client_replay_eq.py \
      --base <旧仓检出> --base-sha <40 位 sha> --candidate <评估仓检出> --candidate-sha <40 位 sha> \
      [--routes groundsg-oracle,smvla,perceptual-framesamp-modul,pp,astra] [--third-party <含 mme-vla 的目录>] \
      [--bench-src <benchmark 子模块的 src，检出里子模块为空时给>]
判定行：每路线 ``CLIENT_REPLAY_ROUTE=PASS|FAIL route=<r> request_diff=<n> action_diff=<n> control_diff=<n>
terminal_diff=<n>`` 与 ``CLIENT_REPLAY_SELFTEST=PASS|FAIL route=<r> action=caught|missed request=… order=…``；
末行 ``CLIENT_REPLAY_EQ=PASS|FAIL routes=<n> cases=<n> tamper_detected=<0|1> base=<sha> candidate=<sha>``
（``tamper_detected=1`` 当且仅当每条路线的每类篡改都被抓到）。
自检**强制执行**（``--self-test`` 旗标保留以兼容旧命令，给不给都跑）：以「故意改动」的候选（在子进程内对动作、
请求、调用顺序各做一处篡改）跑一遍，三类都必须被抓到；自检不过整体 FAIL。
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

MODEL_ROUTES = ("groundsg-oracle", "smvla", "perceptual-framesamp-modul", "pp")
ROUTES = (*MODEL_ROUTES, "astra")
SCENARIOS = ({"name": "success", "done_at": 37, "raise_at": None, "max_steps": 200},
             {"name": "env_error", "done_at": None, "raise_at": 5, "max_steps": 200},
             {"name": "timeout", "done_at": None, "raise_at": None, "max_steps": 40})
_SCEN = {sc["name"]: sc for sc in SCENARIOS}
#: 常驻两局的用例：(用例名, 两局各自的单局场景)；同一子进程、同一模块与上下文连跑
CASES = (("success", ("success", "success")), ("env_error", ("env_error", "success")),
         ("timeout", ("timeout", "success")))
#: Astra 零外联探针的用例名
ASTRA_CASES = ("probe",)
#: 拆仓后评估仓的模块（按包名导入）
PKG_MODULES = {"fsm": "robomme_ood_eval.models.framesamp_modul", "gsg": "robomme_ood_eval.models.groundsg",
               "smvla_client": "robomme_ood_eval.models.smvla", "smvla_server": "robomme_ood_eval.servers.smvla_server",
               "pp_client": "robomme_ood_eval.models.pp", "astra": "robomme_ood_eval.models.astra"}
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

    def __init__(self, log: Log, scen: dict, task: str = "PickXtimes", ep: int = 3, first_step: int = 0):
        self.log, self.scen, self.task = log, scen, task
        self.first_step = first_step  # 篡改动作的位置按全用例的累计步数算（只篡改第一局第 3 步）
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
        if self.log.tamper == "action" and self.first_step + self.t == 2:
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


def _old_dir(root: Path) -> Path:
    """旧仓的评估目录（拆仓前布局）。"""
    return Path(root) / "scripts" / ("eval" + "-official")


def _load(root: Path, name: str, iface: dict | None = None):
    """按本侧布局加载模块：``pkg`` 按包名导入（``name`` 为 ``PKG_MODULES`` 的键，或旧模块名映射到它）；``scripts``
    按旧评估目录里的文件路径加载。"""
    iface = iface or {"layout": "scripts"}
    if iface.get("layout") == "pkg":
        from importlib import import_module

        key = {"framesamp_modul_client": "fsm", "groundsg_client": "gsg", "astra_hard_runner": "astra"}.get(name, name)
        return import_module(PKG_MODULES[key])
    p = _old_dir(root) / f"{name}.py"
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
    """本工具所在评估仓的别名表 ``src/robomme_ood_eval/models/_official_defs.py``（官方名与 ``LEGACY_*`` 别名表；
    不从被比较的检出里取，模块名 ``_TOOL_DEFS``）。"""
    mod = sys.modules.get(_TOOL_DEFS)
    if mod is None:
        path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location(_TOOL_DEFS, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_TOOL_DEFS] = mod
        spec.loader.exec_module(mod)
    return mod


def side_interface(root: Path) -> dict:
    """按检出里实际存在的文件判定该侧接口：拆仓后的包布局优先，其次旧评估目录的官方名、改名前的名字（取自别名表）；
    都不全即 ValueError。"""
    defs = official_defs()
    models = Path(root) / "src" / "robomme_ood_eval" / "models"
    if (models / "framesamp_modul.py").is_file() and (models / "groundsg.py").is_file():
        return {"name": "pkg", "layout": "pkg", "fsm_module": "fsm", "gsg_module": "gsg",
                "variant_key": "groundsg_variant", "dataset": defs.DATASET_HARD_VERIFY}
    d = _old_dir(root)
    official = {"name": "official", "layout": "scripts", "fsm_module": "framesamp_modul_client",
                "gsg_module": "groundsg_client", "variant_key": "groundsg_variant", "dataset": defs.DATASET_HARD_VERIFY}
    rev_mod = {new: old for old, new in defs.LEGACY_MODULE_ALIASES.items()}
    rev_key = {new: old for old, new in defs.LEGACY_CONFIG_KEY_ALIASES.items()}
    rev_ds = {new: old for old, new in defs.LEGACY_DATASET_ALIASES.items()}
    legacy = {"name": "legacy", "layout": "scripts", "fsm_module": rev_mod[official["fsm_module"]],
              "gsg_module": rev_mod[official["gsg_module"]], "variant_key": rev_key[official["variant_key"]],
              "dataset": rev_ds[official["dataset"]]}
    for iface in (official, legacy):
        if (d / f"{iface['fsm_module']}.py").is_file() and (d / f"{iface['gsg_module']}.py").is_file():
            return iface
    raise ValueError(f"{d} 既不是官方名接口也不是改名前接口（客户端模块缺失）")


def _identity(scen, iface: dict, i: int = 0):
    """第 i 局（常驻两局）的身份：同一任务、不同局号与种子。"""
    seed = 1234 + i
    return {"task": "PickXtimes", "tier": "xhard0", "seed": seed, "source_episode": 3 + 4 * i, "builder_episode": i,
            "key": f"PickXtimes_xhard0_{seed}", "dataset": iface["dataset"], "attempt": 1}


def _patch_ws(log, counters):
    import websockets.sync.client as wsc
    wsc.connect = lambda *a, **k: FakeWS(log, counters)


def run_case(root: Path, route: str, scen_names, tamper: str | None, tmp: Path, iface: dict | None = None) -> dict:
    """常驻两局：同一模块（与 GroundSG 的策略上下文）连跑 ``scen_names`` 里的各局；事件间插局界 ``["ep", i]``，
    终态逐局记。某局驱动抛异常即整个用例记为崩溃。"""
    iface = iface or side_interface(root)
    log = Log(tamper)
    counters = {"infer": 0}
    vkey = iface["variant_key"]
    # 常驻的策略上下文按进程定一个步数上限（GroundSG 的 policy_context 与每局 conn_info 必须一致）：取用例内最小者，
    # 超时局照样在第 40 步触发；其后的成功局第 37 步成功，不受影响
    case_max = min(_SCEN[n]["max_steps"] for n in scen_names)
    if route in ("perceptual-framesamp-modul", "groundsg-oracle"):
        _patch_ws(log, counters)
    if route == "perceptual-framesamp-modul":
        mod = _load(root, iface["fsm_module"], iface)
    elif route == "groundsg-oracle":
        mod = _load(root, iface["gsg_module"], iface)
        ctx = mod.make_policy_context({vkey: "ground-sg-oracle", "port": 1,
                                       "max_steps": case_max,
                                       "out": str(tmp), "policy_seed": POLICY_SEED})
    elif route == "smvla":
        srv = _load(root, "smvla_server", iface)
        mod = _load(root, "smvla_client", iface)
        conn = SmvlaConn(log, srv, counters)
    elif route == "pp":
        mod = _load(root, "pp_client", iface)
    else:
        raise ValueError(route)
    terminals = []
    steps_before = 0
    for i, name in enumerate(scen_names):
        scen = dict(_SCEN[name], max_steps=case_max)
        log.add("ep", i)
        sess = FakeSession(log, scen, ep=3 + 4 * i, first_step=steps_before)
        ident = _identity(scen, iface, i)
        ep_dir = tmp / f"ep{i}" / f"{ident['key']}.a1"
        ep_dir.mkdir(parents=True, exist_ok=True)
        rec = NullRecorder(ep_dir)
        conn_info = {"host": "127.0.0.1", "port": 1, "max_steps": scen["max_steps"], "dataset": iface["dataset"],
                     "trace_dir": str(ep_dir), "episode_tag": f"{ident['key']}.a1"}
        if route == "perceptual-framesamp-modul":
            res = mod.run_episode(sess, ident, conn_info, rec)
        elif route == "groundsg-oracle":
            conn_info.update({"policy_context": ctx, vkey: "ground-sg-oracle"})
            res = mod.run_episode(sess, ident, conn_info, rec)
        elif route == "smvla":
            res = mod.run_episode(sess, ident, conn_info, rec, conn=conn, max_steps=scen["max_steps"])
        else:
            res = mod.run_episode(sess, ident, conn_info, rec,
                                  connection_factory=lambda url, t: PPConn(log, counters))
        steps_before += sess.t
        terminals.append({k: res.get(k) for k in ("status", "task_success", "steps")})
    if tamper == "order":
        ev = log.events
        i = next((k for k in range(len(ev) - 1) if ev[k][0] != ev[k + 1][0] and "ep" not in (ev[k][0], ev[k + 1][0])),
                 None)
        if i is not None:
            ev[i], ev[i + 1] = ev[i + 1], ev[i]
    return {"events": log.events, "terminal": terminals}


def run_route(root: Path, route: str, scen: dict, tamper: str | None, tmp: Path, iface: dict | None = None) -> dict:
    """单局（兼容旧调用）：等价于只含一局的 ``run_case``，终态为该局的字典。"""
    out = run_case(root, route, (scen["name"],), tamper, tmp, iface)
    return {"events": [e for e in out["events"] if e[0] != "ep"], "terminal": out["terminal"][0]}


class _NoNet:
    """禁网：``socket.socket.connect``／``connect_ex``／``socket.create_connection`` 一律记录并拒绝。"""

    def __init__(self, log: Log):
        self.log = log

    def __enter__(self):
        import socket

        self._saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
        log = self.log

        def deny(*a, **k):
            addr = a[1] if len(a) > 1 else k.get("address")
            log.add("net", "connect", repr(addr))
            raise ConnectionRefusedError("client_replay_eq 禁网")

        socket.socket.connect = lambda self_, *a, **k: deny(self_, *a, **k)
        socket.socket.connect_ex = lambda self_, *a, **k: deny(self_, *a, **k)
        socket.create_connection = lambda *a, **k: deny(None, *a, **k)
        return self

    def __exit__(self, *exc):
        import socket

        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = self._saved
        return False


def run_astra_probe(root: Path, tamper: str | None, iface: dict) -> dict:
    """Astra 零外联探针（见模块文档）：导入驱动模块并调旧的模块级核对函数，全程禁网。"""
    log = Log(tamper)
    with _NoNet(log):
        mod = _load(root, "astra_hard_runner", iface)
        probes = []
        for ds in ("hard-verify", "ood"):
            for ms in (1300, 1800):
                try:
                    mod.check_pairing(ds, ms)
                    probes.append(["pairing", ds, ms, "ok"])
                except ValueError as e:
                    probes.append(["pairing", ds, ms, str(e).split(" ")[1] if " " in str(e) else "err"])
        for v in (7, "7", -1, None, True):
            try:
                probes.append(["seed", repr(v), mod.check_policy_seed(v)])
            except ValueError:
                probes.append(["seed", repr(v), "blocked"])
    if tamper == "action":
        probes[0] = probes[0][:-1] + ["tampered"]
    if tamper == "request":
        log.add("net", "connect", "('tamper', 0)")
    if tamper == "order":
        probes[0], probes[1] = probes[1], probes[0]
    for p in probes:
        log.add("probe", *map(str, p))
    return {"events": log.events, "terminal": [{"outbound": sum(e[0] == "net" for e in log.events)}]}


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


def _bench_src(root: Path, explicit: str | None) -> Path:
    """benchmark 子模块的 ``src``：检出里已检出的优先，否则 ``--bench-src``。"""
    own = Path(root) / "third_party" / "robomme_benchmark" / "src"
    if (own / "robomme_ood").is_dir():
        return own
    return Path(explicit).resolve() if explicit else own


def worker(args) -> int:
    sys.dont_write_bytecode = True  # 被比较的检出只读：不在其中留 __pycache__
    root = Path(args.root).resolve()
    out = {"root": str(root), "runs": {}}
    try:
        iface = side_interface(root)
    except ValueError as e:
        out["import_error"] = str(e)
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return 0
    out["iface"] = iface["name"]
    tp = Path(args.third_party).resolve()
    os.environ["SGEVAL_THIRD_PARTY"] = str(tp)
    if iface["layout"] == "pkg":
        sys.path[:0] = [str(root / "src"), str(_bench_src(root, args.bench_src))]
    else:
        sys.path.insert(0, str(root / "src"))
    sys.path.insert(2, str(tp / "mme-vla" / "packages" / "openpi-client" / "src"))
    try:
        if iface["layout"] == "pkg":
            import robomme_ood_eval
            out["robomme_ood_eval"] = robomme_ood_eval.__file__
            if not str(Path(robomme_ood_eval.__file__).resolve()).startswith(str(root / "src")):
                out["import_error"] = f"robomme_ood_eval 来自 {robomme_ood_eval.__file__}，不在 {root}/src"
        else:
            import robomme_ood
            out["robomme_ood"] = robomme_ood.__file__
            if not str(Path(robomme_ood.__file__).resolve()).startswith(str(root / "src")):
                out["import_error"] = f"robomme_ood 来自 {robomme_ood.__file__}，不在 {root}/src"
    except ImportError as e:
        out["import_error"] = f"{type(e).__name__}: {e}"
    with tempfile.TemporaryDirectory(prefix="replay-") as td:
        if args.route == "astra":
            for name in ASTRA_CASES:
                try:
                    out["runs"][name] = run_astra_probe(root, args.tamper, iface)
                except Exception as e:  # noqa: BLE001
                    out["runs"][name] = {"crash": f"{type(e).__name__}: {e}"[:400]}
        else:
            for name, scen_names in CASES:
                try:
                    out["runs"][name] = run_case(root, args.route, scen_names, args.tamper, Path(td) / name, iface)
                except Exception as e:  # 驱动本身崩溃也是一种可比较的结果
                    out["runs"][name] = {"crash": f"{type(e).__name__}: {e}"[:400]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


# ── 主进程：两侧各跑、比较 ────────────────────────────────────────────────────


def _run_side(root: Path, route: str, third_party: Path, tamper: str | None, py: str,
              bench_src: str | None = None) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    cmd = [py, str(Path(__file__).resolve()), "--_worker", "--root", str(root), "--route", route,
           "--third-party", str(third_party), "--out", out]
    if bench_src:
        cmd += ["--bench-src", str(bench_src)]
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


def case_names(route: str | None = None) -> tuple[str, ...]:
    return ASTRA_CASES if route == "astra" else tuple(n for n, _ in CASES)


def compare(a: dict, b: dict, route: str | None = None) -> dict:
    """两侧一条路线的比较。用例集合以 ``CASES``（Astra 为 ``ASTRA_CASES``）为准；缺用例、任一侧崩溃（两侧同样崩溃
    也算）、任一侧零环境 step 事件或零服务事件都计 ``control_diff``，不跳过。Astra 探针：任一侧有外联尝试计
    ``request_diff``，探针结果不同计 ``action_diff``。"""
    d = {"request_diff": 0, "action_diff": 0, "control_diff": 0, "terminal_diff": 0, "notes": []}
    for tag, side in (("base", a), ("cand", b)):
        if "worker_failed" in side or "import_error" in side:
            d["control_diff"] += 1
            d["notes"].append(f"{tag}: {side.get('worker_failed') or side.get('import_error')}")
    if d["notes"]:
        return d
    for name in case_names(route):
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
        if route == "astra":
            for tag, ev in (("base", ea), ("cand", eb)):
                n_net = _count(ev, "net")
                if n_net:
                    d["request_diff"] += n_net
                    d["notes"].append(f"{name}: {tag} 有 {n_net} 次外联尝试（应为 0）")
            pa, pb = [e for e in ea if e[0] == "probe"], [e for e in eb if e[0] == "probe"]
            if not pa or not pb:
                d["control_diff"] += 1
                d["notes"].append(f"{name}: 零探针事件")
            elif len(pa) != len(pb):
                d["control_diff"] += 1
            else:
                d["action_diff"] += sum(x != y for x, y in zip(pa, pb))
            continue
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
    ap.add_argument("--bench-src", default=None, help="benchmark 子模块的 src（评估仓检出里子模块为空时给）")
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
        print(f"CLIENT_REPLAY_EQ=FAIL routes={len(routes)} cases=0 tamper_detected=0 base={bs} candidate={cs} "
              f"reason=sha_mismatch", flush=True)
        return 1
    print(f"CLIENT_REPLAY_SHA=PASS base={bs} candidate={cs}", flush=True)
    route_pass = selftest_pass = 0
    tamper_all = True
    for r in routes:
        a = _run_side(base, r, tp, None, args.python, args.bench_src)
        b = _run_side(cand, r, tp, None, args.python, args.bench_src)
        print(f"CLIENT_REPLAY_IFACE route={r} base={a.get('iface')} candidate={b.get('iface')}", flush=True)
        d = compare(a, b, r)
        ok = is_clean(d)
        route_pass += ok
        for n in d["notes"][:8]:
            print(f"CLIENT_REPLAY_NOTE route={r} {n}")
        print(f"CLIENT_REPLAY_ROUTE={'PASS' if ok else 'FAIL'} base={bs} candidate={cs} route={r} "
              f"cases={len(case_names(r))} request_diff={d['request_diff']} action_diff={d['action_diff']} "
              f"control_diff={d['control_diff']} terminal_diff={d['terminal_diff']}", flush=True)
        caught = {}
        for t in TAMPERS:
            caught[t] = not is_clean(compare(a, _run_side(cand, r, tp, t, args.python, args.bench_src), r))
        tamper_all = tamper_all and all(caught.values())
        st = ok and all(caught.values())  # 基线本身不等时自检无意义，一并判 FAIL
        selftest_pass += st
        print(f"CLIENT_REPLAY_SELFTEST={'PASS' if st else 'FAIL'} route={r} "
              + " ".join(f"{t}={'caught' if v else 'missed'}" for t, v in caught.items()), flush=True)
    all_ok = route_pass == len(routes) and selftest_pass == len(routes)
    n_cases = sum(len(case_names(r)) for r in routes)
    print(f"CLIENT_REPLAY_EQ={'PASS' if all_ok else 'FAIL'} routes={len(routes)} cases={n_cases} "
          f"tamper_detected={int(tamper_all)} base={bs} candidate={cs} route_pass={route_pass} "
          f"selftest_pass={selftest_pass}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
