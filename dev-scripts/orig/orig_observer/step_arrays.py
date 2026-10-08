"""逐步原数组 ``arrays.npz``（计划第二部分一节 S7、审计第 12 条；契约 C4、C5）。

在 ``InProcSimPool.step``（SimpleMemVLA）／``EnvRunner.step``（FrameSamp+Modulation）钩子里，把每步**实际交给环境**的动作行按原 dtype、
shape 复制一份（只复制主机端 numpy，不调随机函数、不做 GPU 运算、不改参数与返回值），局末写到局目录
``arrays.npz``（经 ``trace_writer.merge_write_npz`` 合并写；该函数尚不存在时退回 ``np.savez``），键 ``exec_action__%05d``（0 起的步序号 = ``trace`` 的 ``step - 1``）。

SimpleMemVLA 一次派发一个动作块，``SimEnvService.step`` 逐行执行到 ``consumed`` 行为止；实际交给环境的第 j 行是
``np.asarray(np.asarray(chunk, float64)[j], float64).reshape(-1)[:8]``（``SimEnvService.step`` 先整体转 float64，
``RoboMMESimEnv.step_one`` 再 ``reshape(-1)`` 后取前 8 维交给 ``env.step``），见 ``smvla_exec_rows``。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

KEY_FMT = "exec_action__{:05d}"


def smvla_exec_rows(chunk: Any, consumed: int) -> list[np.ndarray]:
    """SimpleMemVLA 动作块里实际执行的前 ``consumed`` 行（float64、8 维），换算与原版 ``SimEnvService.step`` →
    ``RoboMMESimEnv.step_one`` 逐字相同；返回新数组，不与入参共享内存。"""
    n = int(consumed or 0)
    if n <= 0:
        return []
    rows = np.asarray(chunk, dtype=np.float64)
    out = []
    for a in rows[:n]:
        act = np.asarray(a, dtype=np.float64).reshape(-1)
        out.append(np.array(act[:8], copy=True))
    return out


class StepArrays:
    """一局的逐步动作原数组；``add`` 复制一份，``save`` 写 ``arrays.npz``（非压缩 ``np.savez``）。"""

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}

    def add(self, index: int, action: Any) -> str:
        key = KEY_FMT.format(int(index))
        if key in self._arrays:
            raise ValueError(f"重复的步键 {key}")
        self._arrays[key] = np.array(np.asarray(action), copy=True)
        return key

    def __len__(self) -> int:
        return len(self._arrays)

    def keys(self) -> list[str]:
        return sorted(self._arrays)

    def save(self, path: str | Path) -> Path | None:
        """写 ``arrays.npz``；没有任何步时不写文件，返回 None。

        接口冻结说明四.4：``arrays.npz`` 唯一允许的写法是 ``trace_writer.merge_write_npz(path, mapping)``（读已有文件、
        同键核 dtype／shape／sha256、合并后原子替换）。该函数存在时委托给它；R6 尚未合入（函数不存在）时保持旧写法
        （整份 ``np.savez`` 到临时文件再改名）。"""
        if not self._arrays:
            return None
        path = Path(path)
        merge = getattr(_trace_writer(), "merge_write_npz", None)
        if callable(merge):
            merge(path, dict(self._arrays))
            return path
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **self._arrays)
        tmp.replace(path)
        return path


def _trace_writer():
    """评估包 ``src/robomme_hard_eval/record/trace_writer.py``（模块名 ``trace_writer``，已加载则复用，与 ``_obs_common.load_eval_module``
    同一别名约定）；加载失败返回 None（只影响是否走合并写，不影响旧写法）。"""
    mod = sys.modules.get("trace_writer")
    if mod is not None:
        return mod
    path = Path(__file__).resolve().parents[3] / "src" / "robomme_hard_eval" / "record" / "trace_writer.py"
    try:
        spec = importlib.util.spec_from_file_location("trace_writer", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["trace_writer"] = mod
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        sys.modules.pop("trace_writer", None)
        return None
    return mod
