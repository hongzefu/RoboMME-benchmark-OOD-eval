"""原侧只读观测器的公共小工具：按路径加载本仓库模块、传输日志写入、清单读取、身份与局目录、导入后挂钩。

文件名刻意带下划线前缀 ``_obs_common``：本目录会追加到原版客户端的 PYTHONPATH 末尾，不能与原版
``examples/robomme`` 下的 ``utils``、``env_runner`` 等模块重名。

S7（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S7）相对 v7.5eval ``_v75_obs_common.py`` 的改动：
- 局目录改为与 GroundSG 原侧 ``official_hard_runner.run_identity`` 同布局的 ``<root>/<key>.a<N>/``，``key`` 与上一轮
  xhard0 分片相同（``<task>_xhard0_<seed>``），``N`` 按该身份在本录制根下的开局顺序编号（续跑接着编）；
- 新增 ``load_eval_module`` 按路径加载评估包里的 ``groundsg_client``（拆仓后在 ``src/robomme_ood_eval/models/groundsg.py``；
  复用其 ``RawFrameWriter`` 与已加载的 ``trace_writer``），旧模块名到新文件的映射见 ``EVAL_MODULE_FILES``；
- 新增 ``HookErrors``：钩子异常只打一行 ``OBSERVER_HOOK_ERROR`` 并累加计数，同时追加到
  ``<root>/hook-errors.jsonl``（启动器据此出 ``OBSERVER_COMPLETE`` 的 ``hook_errors=``）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

OBS_DIR = Path(__file__).resolve().parent
#: 评估仓根（本文件在 ``dev-scripts/orig/orig_observer/``）
REPO = OBS_DIR.parents[2]
#: 评估包源码根（``src/robomme_ood_eval/``）
EVAL_PKG = REPO / "src" / "robomme_ood_eval"
#: 旧仓评估目录下 ``<名>.py`` 的模块名 → 拆仓后的文件（别名照旧，同一模块只加载一份）
EVAL_MODULE_FILES = {
    "official_defs": EVAL_PKG / "models" / "_official_defs.py",
    "groundsg_client": EVAL_PKG / "models" / "groundsg.py",
    "framesamp_modul_client": EVAL_PKG / "models" / "framesamp_modul.py",
    "smvla_client": EVAL_PKG / "models" / "smvla.py",
    "pp_client": EVAL_PKG / "models" / "pp.py",
    "trace_writer": EVAL_PKG / "record" / "trace_writer.py",
    "recorder": EVAL_PKG / "record" / "recorder.py",
    "budget_ledger": REPO / "dev-scripts" / "gl" / "budget_ledger.py",
}
RECORDER_PATH = EVAL_MODULE_FILES["recorder"]

XHARD0 = "xhard0"
DATASET = "hard-verify"  # 与 official_hard_runner.DATASET 相同（第二档两侧身份口径）
EP_DIR_RE = re.compile(r"^(?P<key>.+)\.a(?P<attempt>\d+)$")
FINAL_STATUSES = ("success", "fail", "timeout")
TERMINALS = ("success", "fail", "timeout", "error")


def load_recorder():
    """按路径加载评估包的 ``record/recorder.py``（模块名 v75_recorder，不改 sys.path；代理记账仍用它）。"""
    name = "v75_recorder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RECORDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_eval_module(name: str):
    """按路径加载旧名 ``name`` 对应的评估包模块（``EVAL_MODULE_FILES``；模块名即 ``name``，已加载则复用；与模型客户端
    的 ``load_sibling`` 同一别名约定，故 ``trace_writer`` 只有一份）。评估包 ``src`` 追加到 ``sys.path`` 末尾，供拆仓后
    模块里的 ``robomme_ood_eval.*`` 包内导入使用（放末尾，不遮蔽原版客户端的同名模块）。"""
    if name in sys.modules:
        return sys.modules[name]
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.append(src)
    if name.endswith("_client"):  # 客户端按旧名 load_sibling 取同用模块：先按新位置登记，免得它到 models/ 下找旧文件名
        for dep in ("official_defs", "trace_writer"):
            load_eval_module(dep)
    spec = importlib.util.spec_from_file_location(name, EVAL_MODULE_FILES[name])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr)


def payload_sha(msg: Any) -> tuple[str, int, str]:
    """websocket 消息的 (帧类型, 字节长度, sha256)；文本按 UTF-8 编码后计算。"""
    if isinstance(msg, str):
        b = msg.encode("utf-8")
        kind = "text"
    else:
        b = bytes(msg)
        kind = "binary"
    return kind, len(b), hashlib.sha256(b).hexdigest()


def payload_bytes(msg: Any) -> bytes:
    """websocket 消息的原始字节（文本按 UTF-8）。"""
    return msg.encode("utf-8") if isinstance(msg, str) else bytes(msg)


class JsonlLog:
    """线程安全的追加式 jsonl（每行立即 flush）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self._lock:
            self._fh.write(dumps(rec) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class HookErrors:
    """钩子异常计数（C11）：打印一行 ``OBSERVER_HOOK_ERROR``、累加计数、追加 ``<root>/hook-errors.jsonl``。

    本类自身绝不抛异常（写日志失败也只吞掉），保证钩子异常不外抛。"""

    def __init__(self) -> None:
        self.count = 0
        self.root: Path | None = None
        self._lock = threading.Lock()

    def note(self, where: str, episode: str | None = None) -> None:
        try:
            tb = traceback.format_exc(limit=3)
        except Exception:  # noqa: BLE001
            tb = "<traceback 不可用>"
        with self._lock:
            self.count += 1
        try:
            print(f"OBSERVER_HOOK_ERROR where={where} episode={episode} {tb!r}", flush=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.root is not None:
                with open(self.root / "hook-errors.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(dumps({"where": where, "episode": episode, "t": time.time(), "pid": os.getpid(),
                                    "traceback": tb[-2000:]}) + "\n")
        except Exception:  # noqa: BLE001
            pass


def rec_root() -> Path:
    """录制根目录（环境变量 REC_ROOT，必填）。"""
    r = os.environ.get("REC_ROOT")
    if not r:
        raise RuntimeError("未设 REC_ROOT")
    p = Path(r)
    p.mkdir(parents=True, exist_ok=True)
    return p


def identity_key(task: str, seed: int) -> str:
    """与上一轮 xhard0 分片（``eval_manifest.v8_key``）相同的身份键 ``<task>_xhard0_<seed>``。"""
    return f"{task}_{XHARD0}_{int(seed)}"


def existing_attempts(root: Path, key: str) -> list[int]:
    """录制根下该身份已有的局目录尝试号（升序）。"""
    out = []
    if Path(root).is_dir():
        for p in Path(root).iterdir():
            m = EP_DIR_RE.match(p.name)
            if m and m["key"] == key and p.is_dir():
                out.append(int(m["attempt"]))
    return sorted(out)


def claim_episode_dir(root: Path, key: str) -> tuple[Path, int]:
    """新建 ``<root>/<key>.a<N>``（N = 已有最大号 + 1，原子 mkdir，撞号即顺延）；返回 (目录, N)。"""
    n = (existing_attempts(root, key) or [0])[-1] + 1
    while True:
        d = Path(root) / f"{key}.a{n}"
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d, n
        except FileExistsError:
            n += 1


def now() -> float:
    return time.time()


class PostImportHooks:
    """模块首次被导入、执行完毕后立即调用钩子（不提前导入任何模块，保持原版代码的导入顺序不变）。"""

    def __init__(self, hooks: dict):
        self.hooks = dict(hooks)  # 模块名 → fn(module)

    def install(self) -> None:
        for name in list(self.hooks):
            if name in sys.modules:  # 已导入的直接挂
                self.hooks.pop(name)(sys.modules[name])
        if self.hooks:
            sys.meta_path.insert(0, self)

    def find_spec(self, name, path, target=None):
        if name not in self.hooks:
            return None
        spec = None
        for f in sys.meta_path:
            if f is self or not hasattr(f, "find_spec"):
                continue
            spec = f.find_spec(name, path, target)
            if spec is not None:
                break
        if spec is None or spec.loader is None:
            return None
        hook = self.hooks.pop(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _hook=hook):
            _orig(module)
            _hook(module)

        spec.loader.exec_module = exec_module
        return spec


def parse_flag(argv: list[str], names: tuple[str, ...]) -> str | None:
    """从命令行取 ``--x=v`` 或 ``--x v`` 形式的值（tyro 与 argparse 两种写法都认）。"""
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a[len(n) + 1:]
    return None


def manifest_seeds(path: str | None) -> dict:
    """清单 (task, source_episode) → seed。"""
    out = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[(r["task"], int(r["source_episode"]))] = int(r["seed"])
    return out
