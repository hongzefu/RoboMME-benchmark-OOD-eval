"""模型侧 4 个方法的基类 ``Policy``、外层工厂 ``load_policy`` 与服务端子进程管理 ``ServerProcess``（拆分方案 §三）。

调用频率：``load`` 与 ``close`` 进程寿命内各一次，``reset`` 与 ``play`` 每局各一次；``run_episode`` 只编排一局、
绝不调 ``load()``。基类对四个方法各带一个调用计数器 ``calls``（子类覆写的方法同样计数，``super()`` 回调不重复计），
供 ``EVAL_EPISODE`` 判定行核 loads／resets／closes。

``ServerProcess`` 抽自旧 ``run_seat.sh`` 的 ``build_server_cmd``／``start_server``／``stop_server``／``kill_group``／
``pick_port``／``server_ready``：``setsid`` 起进程组、写 ``server-metadata-<port>.json``、三种就绪探测、``attach()``
四项核对（M3）、``stop()`` 先 TERM 进程组等 60 s 再 KILL、可选核显存释放；``stop_by_metadata(path)`` 供
``scripts/evaluate.py --stop-server`` 只按元数据停掉看门狗退出后留下的服务端（S2）。
"""
from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

#: 服务端起进程时一律去掉的代理变量（GL 计算节点有 HTTP 代理，本机回环连接会被拒 403）
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")
#: 缺省就绪等待（秒）
DEFAULT_READY_TIMEOUT_S = 1800.0
#: stop() 发 TERM 后等待进程组退出的秒数，超时再 KILL
STOP_GRACE_S = 60.0
METHODS = ("load", "reset", "play", "close")


class ServerDead(RuntimeError):
    """服务端进程已死（``reset()`` 探活或就绪等待期间发现）。本轮重起次数为 0：直接停下交用户。"""


class AstraStop(RuntimeError):
    """Astra 报停或费用守卫拒发：整批停，账本保留。"""


class ServerMismatch(RuntimeError):
    """``attach()`` 到的服务端与本次配置不符（M3：``RUN_BLOCKED reason=server_mismatch``）。"""


class ServerNotReady(RuntimeError):
    """就绪等待超时。"""


# ── Policy 基类 ──────────────────────────────────────────────────────────────


def _counted(name: str, fn):
    """包一层调用计数：同一实例同一方法只在最外层调用计一次（子类 ``super().reset(spec)`` 不重复计）。"""

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        depth = self.__dict__.setdefault("_call_depth", {})
        calls = self.__dict__.setdefault("calls", {m: 0 for m in METHODS})
        if depth.get(name, 0) == 0:
            calls[name] += 1
        depth[name] = depth.get(name, 0) + 1
        try:
            return fn(self, *a, **k)
        finally:
            depth[name] -= 1

    wrapper._policy_counted = True
    return wrapper


class Policy:
    """五个模型 + dummy 的共同基类。子类覆写 ``load``／``reset``／``play``／``close`` 四个方法。

    ``reset(spec)`` 契约（S1，单测钉死）：**不向服务端发任何消息、不碰环境**；只清客户端状态、按本局身份准备会话
    标识、探一次服务端健康（死了抛 ``ServerDead``）、做局前拒绝（如 Astra 的局数登记与 STOP 检查，抛 ``AstraStop``）。
    服务端的 reset／start_episode 消息由 ``play`` 在每局新连接上发。

    ``play(session, spec, recorder) -> dict`` 返回 ``PlayOutput``：必有 ``status``（success／fail／timeout／error）、
    ``task_success``（0/1）、``steps``、``error``、``infra``、``infra_reason``；可有 ``decisions``、``timing`` 与模型专属
    字段。环境只经 ``session.reset()``／``session.step()`` 碰，不得调 ``session.close()``。

    属性：``model``（注册名）、``label``（产物树里的模型目录名，缺省等于 ``model``；GroundSG 可带变体）、
    ``policy_seed``、``cfg``（``load_policy`` 收到的模型专属参数）、``server``（``ServerProcess`` 或 None）、
    ``episode_wall_s``（单局墙钟缺省，None 时按 ``episode.DEFAULT_WALL_S`` 取）、``calls``（四方法调用计数）、
    ``episodes_run``（本进程已跑局数，外层维护，首局放宽墙钟用）。
    """

    model: str = "base"
    episode_wall_s: float | None = None

    def __new__(cls, *a, **k):
        obj = super().__new__(cls)
        obj.calls = {m: 0 for m in METHODS}
        obj._call_depth = {}
        obj.episodes_run = 0
        return obj

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for name in METHODS:
            fn = cls.__dict__.get(name)
            if fn is not None and not getattr(fn, "_policy_counted", False):
                setattr(cls, name, _counted(name, fn))

    def __init__(self, policy_seed: int, **cfg: Any):
        self.policy_seed = int(policy_seed)
        self.cfg = dict(cfg)
        self.server: ServerProcess | None = None
        self.closed = False
        if not hasattr(self, "label") or self.label is None:
            self.label = type(self).model

    label: str | None = None

    # 四个方法（基类缺省实现）
    def load(self) -> None:
        """进程级一次：起服务端（或 attach 已在跑的那个）、加载客户端侧模型、编译缓存、预热。基类什么都不做。"""

    def reset(self, spec) -> None:
        """每局第一句：基类只探一次服务端健康（有服务端时）。"""
        self.check_server()

    def play(self, session, spec, recorder) -> dict:
        raise NotImplementedError(f"{type(self).__name__}.play 未实现")

    def close(self) -> None:
        """进程级一次：停服务端进程组。幂等。"""
        if self.closed:
            return
        self.closed = True
        for srv in self.servers():
            srv.stop()

    # 辅助
    def servers(self) -> list["ServerProcess"]:
        """本 Policy 持有的服务端（看门狗退出前打印 ``SERVER_LEFT`` 用）；子类有多个时覆写。"""
        return [self.server] if self.server is not None else []

    def check_server(self) -> None:
        for srv in self.servers():
            srv.check()

    def server_seed(self) -> int | None:
        """结果行 ``server_seed``：服务端元数据里的 ``policy_seed``；没有服务端为 None（不伪造）。"""
        for srv in self.servers():
            v = (srv.metadata or {}).get("policy_seed")
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    def __enter__(self) -> "Policy":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


for _m in METHODS:
    setattr(Policy, _m, _counted(_m, Policy.__dict__[_m]))


def load_policy(model: str, policy_seed: int, **cfg: Any) -> Policy:
    """按 ``models`` 注册表构造子类并调 ``load()``（进程级一次）。可作上下文管理器，``with`` 退出即 ``close()``。

    ``load()`` 抛异常时先 ``close()``（停掉可能已起的服务端）再原样上抛。"""
    from robomme_ood_eval import models

    cls = models.resolve(model)
    policy = cls(policy_seed=policy_seed, **cfg)
    policy.model = model
    if getattr(policy, "label", None) in (None, "base"):
        policy.label = model
    try:
        policy.load()
    except BaseException:
        policy.close()
        raise
    return policy


# ── 端口与就绪探测 ───────────────────────────────────────────────────────────


def port_busy(port: int, host: str = "127.0.0.1") -> bool:
    """端口是否已有进程在听（与旧 ``run_seat.sh::port_busy`` 同义：能连上即占用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, int(port)))
            return True
        except OSError:
            return False


def pick_port(base: int | None = None, *, span: int = 2, tries: int = 5) -> int:
    """取一个服务端端口：``base`` 给出时从它起每次加 ``span``，要求该端口与其 +1（中继）都空闲，至多试 ``tries`` 次；
    ``base`` 为 None 时向系统要一个临时空闲端口。都失败抛 RuntimeError。"""
    if base is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
    p = int(base)
    for _ in range(int(tries)):
        if not port_busy(p) and not port_busy(p + 1):
            return p
        p += int(span)
    raise RuntimeError(f"从 {base} 起 {tries} 次都没有空闲端口")


def http_health(port: int, path: str = "/health", host: str = "127.0.0.1", timeout: float = 2.0) -> bool:
    """``GET http://host:port/path`` 返回 200 即健康（不走代理）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{int(port)}{path}", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


@dataclasses.dataclass(frozen=True)
class Ready:
    """就绪判据一条：``port``（端口在听）、``health``（``GET value`` 返回 200，value 缺省 ``/health``）、
    ``log``（服务日志含 value 字符串）。``ServerProcess`` 要求全部判据同时成立。"""

    kind: str
    value: str | None = None

    def __post_init__(self):
        if self.kind not in ("port", "health", "log"):
            raise ValueError(f"Ready.kind 只能是 port／health／log：{self.kind!r}")
        if self.kind == "log" and not self.value:
            raise ValueError("Ready('log') 必须给要匹配的字符串")

    @classmethod
    def port(cls) -> "Ready":
        return cls("port")

    @classmethod
    def health(cls, path: str = "/health") -> "Ready":
        return cls("health", path)

    @classmethod
    def log(cls, text: str) -> "Ready":
        return cls("log", text)


# ── 进程工具 ─────────────────────────────────────────────────────────────────


def pid_alive(pid: int | None) -> bool:
    """进程存在且不是僵尸。"""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def proc_cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except OSError:
        return None
    return [x.decode(errors="replace") for x in raw.split(b"\0") if x]


def _group_alive(pgid: int) -> bool:
    """进程组里还有非僵尸进程（僵尸已退出、只等父进程回收，不算活着）。"""
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    proc = Path("/proc")
    if not proc.is_dir():  # pragma: no cover 非 Linux
        return True
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(int(pgid)) and fields[0] != "Z":
            return True
    return False


def kill_group(pgid: int, *, grace_s: float = STOP_GRACE_S, child: subprocess.Popen | None = None,
               poll_s: float = 0.2) -> str:
    """先 TERM 进程组，等 ``grace_s`` 秒，仍在再 KILL；返回 ``term``／``kill``／``gone``。"""
    try:
        os.killpg(int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        if child is not None:
            child.poll()
        return "gone"
    deadline = time.monotonic() + float(grace_s)
    how = "term"
    while True:
        if child is not None:
            child.poll()  # 回收自己的子进程，避免僵尸让判定恒真
        leader_alive = pid_alive(pgid) if child is None else child.returncode is None
        if not leader_alive and not _group_alive(pgid):
            break
        if time.monotonic() >= deadline:
            how = "kill"
            try:
                os.killpg(int(pgid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            if child is not None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            break
        time.sleep(poll_s)
    return how


def wait_gpu_free(pid: int, *, timeout_s: float = 30.0, poll_s: float = 1.0) -> bool:
    """轮询 ``nvidia-smi`` 计算进程列表直到 ``pid`` 不在其中（显存已释放）；nvidia-smi 不可用时返回 True。"""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return True
        if str(int(pid)) not in {x.strip() for x in out.splitlines()}:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


# ── ServerProcess ────────────────────────────────────────────────────────────


class ServerProcess:
    """一个服务端子进程（在模型自己的 venv 里跑）。

    ``argv``：完整命令（照抄旧 ``run_seat.sh::build_server_cmd`` 对应分支）；``env``：在当前环境基础上追加／覆盖的
    变量（代理变量一律去掉）；``cwd``；``gpu``：设 ``CUDA_VISIBLE_DEVICES``（None 不动）；``ready``：``Ready`` 判据
    列表（全部成立才算就绪）；``port``：服务端口（元数据文件名与端口探测都用它）；``metadata_dir``：写
    ``server-metadata-<port>.json`` 的目录；``policy_seed``、``ckpt``：写进元数据，``attach()`` 逐项核对。
    """

    def __init__(self, argv: list[str], env: dict | None = None, cwd: str | Path | None = None,
                 gpu: str | int | None = None, ready: Iterable[Ready] | Ready | None = None, *, port: int,
                 metadata_dir: str | Path, policy_seed: int | None = None, ckpt: str | None = None,
                 log_path: str | Path | None = None, name: str = "server",
                 ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S, ready_poll_s: float = 0.5):
        self.argv = [str(x) for x in argv]
        self.env = dict(env or {})
        self.cwd = None if cwd is None else str(cwd)
        self.gpu = None if gpu is None else str(gpu)
        if ready is None:
            ready = [Ready.port()]
        elif isinstance(ready, Ready):
            ready = [ready]
        self.ready = list(ready)
        self.port = int(port)
        self.metadata_dir = Path(metadata_dir)
        self.policy_seed = policy_seed
        self.ckpt = None if ckpt is None else str(ckpt)
        self.name = name
        self.log_path = Path(log_path) if log_path else self.metadata_dir / f"{name}-{self.port}.log"
        self.ready_timeout_s = float(ready_timeout_s)
        self.ready_poll_s = float(ready_poll_s)
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.metadata: dict | None = None
        self.attached = False

    @property
    def metadata_path(self) -> Path:
        return self.metadata_dir / f"server-metadata-{self.port}.json"

    # 启动
    def _child_env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
        env.update(self.env)
        for k in PROXY_VARS:
            env.pop(k, None)
        env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
        return env

    def start(self) -> "ServerProcess":
        """有本端口元数据且进程还活着 → ``attach()``（不符即拒接）；否则 setsid 新起、写元数据、等就绪。"""
        if self.metadata_path.is_file():
            if self.attach():
                return self
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(self.argv, env=self._child_env(), cwd=self.cwd, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        self.pid = self.proc.pid
        self.metadata = {"pid": self.pid, "pgid": self.pid, "port": self.port, "argv": self.argv,
                         "policy_seed": self.policy_seed, "ckpt": self.ckpt, "cwd": self.cwd, "gpu": self.gpu,
                         "log": str(self.log_path), "name": self.name, "host": socket.gethostname(),
                         "started": time.time()}
        tmp = self.metadata_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.metadata, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.metadata_path)
        print(f"SERVER_START name={self.name} port={self.port} pid={self.pid} log={self.log_path}", flush=True)
        self.wait_ready()
        return self

    def _ready_now(self) -> bool:
        for r in self.ready:
            if r.kind == "port" and not port_busy(self.port):
                return False
            if r.kind == "health" and not http_health(self.port, r.value or "/health"):
                return False
            if r.kind == "log":
                try:
                    text = self.log_path.read_text(errors="replace")
                except OSError:
                    return False
                if r.value not in text:
                    return False
        return True

    def log_tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def wait_ready(self, timeout_s: float | None = None) -> float:
        """轮询就绪判据；期间服务端死了抛 ``ServerDead``（附日志尾），超时停服务端并抛 ``ServerNotReady``。"""
        limit = self.ready_timeout_s if timeout_s is None else float(timeout_s)
        t0 = time.monotonic()
        while True:
            if not self.alive():
                tail = self.log_tail()
                print(f"SERVER_DIED_BEFORE_READY name={self.name} port={self.port}\n{tail}", flush=True)
                raise ServerDead(f"{self.name} 服务端就绪前退出 port={self.port}：{tail[-800:]}")
            if self._ready_now():
                dt = time.monotonic() - t0
                print(f"SERVER_READY name={self.name} port={self.port} ready_s={dt:.1f}", flush=True)
                return dt
            if time.monotonic() - t0 > limit:
                print(f"SERVER_READY_TIMEOUT name={self.name} port={self.port} limit_s={limit:.0f}", flush=True)
                self.stop()
                raise ServerNotReady(f"{self.name} 就绪等待超过 {limit:.0f} s")
            time.sleep(self.ready_poll_s)

    # attach（M3）
    def attach(self) -> bool:
        """读 ``server-metadata-<port>.json``：进程已不在 → 返回 False（调用方新起）；进程在则核四项——元数据的
        ``policy_seed``、``argv``、``port``、``ckpt`` 与本次一致，且 ``/proc/<pid>/cmdline`` 恰为该 argv。任一不符打印
        ``RUN_BLOCKED reason=server_mismatch`` 并抛 ``ServerMismatch``（拒接）。"""
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        pid = meta.get("pid")
        if not isinstance(pid, int) or not pid_alive(pid):
            return False
        bad = []
        for key, want in (("policy_seed", self.policy_seed), ("argv", self.argv), ("port", self.port),
                          ("ckpt", self.ckpt)):
            if meta.get(key) != want:
                bad.append(f"{key} metadata={meta.get(key)!r} want={want!r}")
        cmd = proc_cmdline(pid)
        if cmd != meta.get("argv"):
            bad.append(f"cmdline pid={pid} 实际={cmd!r} 与元数据 argv 不符")
        if bad:
            detail = "; ".join(bad)
            print(f"RUN_BLOCKED reason=server_mismatch port={self.port} metadata={self.metadata_path} detail={detail}",
                  flush=True)
            raise ServerMismatch(detail)
        self.pid, self.metadata, self.attached = pid, meta, True
        print(f"SERVER_ATTACH name={self.name} port={self.port} pid={pid}", flush=True)
        if not self._ready_now():
            self.wait_ready()
        return True

    # 状态与停止
    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def check(self) -> None:
        """探活：死了抛 ``ServerDead``。"""
        if not self.alive():
            raise ServerDead(f"{self.name} 服务端已退出 port={self.port} pid={self.pid}：{self.log_tail()[-800:]}")

    def stop(self, *, grace_s: float = STOP_GRACE_S, check_gpu: bool = True) -> str:
        """TERM 进程组，等 ``grace_s`` 秒，再 KILL；``check_gpu`` 时轮询 nvidia-smi 核显存释放；删元数据。"""
        if self.pid is None:
            return "none"
        pgid = int((self.metadata or {}).get("pgid") or self.pid)
        how = kill_group(pgid, grace_s=grace_s, child=self.proc)
        gpu_ok = wait_gpu_free(self.pid) if check_gpu else None
        print(f"SERVER_STOPPED name={self.name} port={self.port} pid={self.pid} how={how} gpu_free={gpu_ok}",
              flush=True)
        try:
            self.metadata_path.unlink()
        except OSError:
            pass
        self.pid = None
        self.proc = None
        return how

    def left_line(self) -> str:
        """看门狗硬退出前打印的一行（S2）。"""
        return (f"SERVER_LEFT pid={self.pid} port={self.port} metadata={self.metadata_path} "
                f"stop=\"scripts/evaluate.py --stop-server {self.metadata_path}\"")

    @staticmethod
    def stop_by_metadata(path: str | Path, *, grace_s: float = STOP_GRACE_S, check_gpu: bool = True) -> str:
        """只按元数据文件停服务端：进程还在且 ``/proc/<pid>/cmdline`` 与元数据 argv 一致才停（防止 pid 复用误杀），
        停后删元数据。返回 ``term``／``kill``／``gone``／``mismatch``。"""
        path = Path(path)
        meta = json.loads(path.read_text(encoding="utf-8"))
        pid = meta.get("pid")
        if not isinstance(pid, int) or not pid_alive(pid):
            print(f"SERVER_STOP_BY_METADATA result=gone pid={pid} metadata={path}", flush=True)
            path.unlink(missing_ok=True)
            return "gone"
        if proc_cmdline(pid) != meta.get("argv"):
            print(f"SERVER_STOP_BY_METADATA result=mismatch pid={pid} metadata={path}（cmdline 与元数据不符，不动）",
                  flush=True)
            return "mismatch"
        how = kill_group(int(meta.get("pgid") or pid), grace_s=grace_s)
        if check_gpu:
            wait_gpu_free(pid)
        path.unlink(missing_ok=True)
        print(f"SERVER_STOP_BY_METADATA result={how} pid={pid} metadata={path}", flush=True)
        return how


def import_attr(module: str, attr: str):
    """按模块路径与属性名惰性取对象（注册表用）。"""
    return getattr(importlib.import_module(module), attr)


__all__ = ["Policy", "load_policy", "ServerProcess", "Ready", "ServerDead", "AstraStop", "ServerMismatch",
           "ServerNotReady", "pick_port", "port_busy", "http_health", "kill_group", "pid_alive", "proc_cmdline",
           "wait_gpu_free", "METHODS"]

if sys.version_info < (3, 10):  # pragma: no cover
    raise RuntimeError("robomme_ood_eval.policy 需要 Python ≥ 3.10")
