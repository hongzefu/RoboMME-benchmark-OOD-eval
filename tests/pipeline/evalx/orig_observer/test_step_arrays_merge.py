"""``orig_observer/step_arrays.StepArrays.save`` 改走 ``trace_writer.merge_write_npz``（接口冻结说明四.4；R2）。

R6 的 ``merge_write_npz`` 尚未合入：用替身模块装到 ``sys.modules["trace_writer"]``（被测代码按此别名取 trace_writer，
与 ``_obs_common.load_eval_module`` 同一约定），核「有就委托、没有就保持旧写法」两条路径；旧写法的字节由 numpy 读回手核。
"""
from __future__ import annotations

import sys
import types

import numpy as np

from tests._support.dev_loaders import load_script


def _arrays(sa):
    a = sa.StepArrays()
    a.add(0, np.arange(8, dtype=np.float64))
    a.add(1, np.arange(8, dtype=np.float32) + 1)
    return a


def test_save_delegates_to_merge_write_npz(tmp_path, monkeypatch):
    calls = []

    def merge_write_npz(path, mapping):
        calls.append((path, {k: (v.dtype.str, v.tobytes()) for k, v in mapping.items()}))

    monkeypatch.setitem(sys.modules, "trace_writer", types.SimpleNamespace(merge_write_npz=merge_write_npz))
    sa = load_script("eval-official/orig_observer/step_arrays.py", fresh=True)
    out = _arrays(sa).save(tmp_path / "arrays.npz")
    assert out == tmp_path / "arrays.npz" and not out.exists()  # 写盘交给 merge_write_npz（替身不写）
    ((path, mapping),) = calls
    assert path == tmp_path / "arrays.npz"
    assert mapping == {"exec_action__00000": ("<f8", np.arange(8, dtype=np.float64).tobytes()),
                       "exec_action__00001": ("<f4", (np.arange(8, dtype=np.float32) + 1).tobytes())}
    assert sa.StepArrays().save(tmp_path / "empty.npz") is None and len(calls) == 1


def test_save_keeps_old_write_when_merge_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "trace_writer", types.SimpleNamespace())
    sa = load_script("eval-official/orig_observer/step_arrays.py", fresh=True)
    out = _arrays(sa).save(tmp_path / "arrays.npz")
    with np.load(out) as z:
        assert sorted(z.files) == ["exec_action__00000", "exec_action__00001"]
        assert z["exec_action__00000"].dtype == np.float64 and z["exec_action__00001"].dtype == np.float32
        assert z["exec_action__00001"].tobytes() == (np.arange(8, dtype=np.float32) + 1).tobytes()
    assert not (tmp_path / "arrays.npz.tmp.npz").exists()
