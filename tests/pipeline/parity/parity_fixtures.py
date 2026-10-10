"""对拍块（tests/pipeline/parity/）的公共夹具：微型 h5、生成输出根、噪声基线小样本。

全部只写 ``tmp_path``；不读 ``artifacts/``、不起仿真、不碰 GPU。夹具只「造输入」，期望值在各用例里手写，
不调用被测模块的私有夹具（``noise_gate._fx_*`` 属于生产自检，不作为本块的期望来源）。

被测脚本一律经 ``tests._support.loaders.load_script`` 按路径加载（真实生产模块）。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from tests._support.dev_loaders import load_script

REPO = Path(__file__).resolve().parents[3]
#: 对拍配置（评估仓 dev-scripts/parity/configs，原 scripts/configs 的五份）
CONFIGS = REPO / "dev-scripts" / "parity" / "configs"
LAUNCH_SCHEMA = "hard-parity-launch/1"  # hard_parity generate 写的启动记录 schema（读者与写者两侧同名契约）


def noise_gate():
    return load_script("parity/noise_gate.py")


def hard_parity():
    return load_script("parity/hard_parity.py")


def noise_run():
    return load_script("parity/noise_run.py")


def hard_pull():
    return load_script("parity/hard_pull.py")


def hard_regression():
    return load_script("parity/hard_regression.py")


def bench_root() -> Path:
    """benchmark 子模块源码树根（子模块未检出时退回当前解释器能找到的 robomme_ood 所在树，只查找不导入）。"""
    return hard_parity()._common.bench_root()


# ── 微型 h5（与录像器同形：episode_<n>/setup + timestep_<i>/{action,obs,info}）─────────────


def write_h5(path: Path, *, seed: int = 1, frames: int = 4, setup_seed: int | None = None,
             joint_offset_from: int | None = None, state_offset_from: int | None = None,
             gripper_offset_from: int | None = None, offset: float = 0.5,
             drop: tuple[int, str] | None = None, joint_dtype: str = "float64",
             frame_ids: Iterable[int] | None = None, frame0_pixel: bool = False,
             extra_field: dict[int, float] | None = None, root_attrs: dict[str, Any] | None = None,
             episode_attrs: dict[str, Any] | None = None, setup_attrs: dict[str, Any] | None = None,
             waypoint: dict[int, str] | None = None, subgoals: list[str] | None = None) -> Path:
    """写一份小 h5。各开关只改一处，便于「同一批变体」逐个喂不同比较器：

    - ``joint_offset_from``／``state_offset_from``／``gripper_offset_from``：从该帧起对应数据集加 ``offset``；
    - ``drop=(帧, 数据集)``：删一帧的一个字段；``joint_dtype``：全部帧的 joint_action dtype；
    - ``frame_ids``：时间步编号（默认 0..frames-1，可造不连续）；``frame0_pixel``：改第 0 帧一个像素；
    - ``extra_field={帧: 值}``：每帧都新增 ``obs/extra``（float32 (3,)），值取字典（缺省 0.0）；
    - ``root_attrs``／``episode_attrs``／``setup_attrs``：写属性；
    - ``waypoint={帧: "real"|"nan"|"int"}``：``action/waypoint_action`` 的三种签名（float64 实值／float32 NaN／int64）；
    - ``subgoals``：逐帧 ``info/simple_subgoal`` 文本（缺省 ``sg<帧>``）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = list(frame_ids) if frame_ids is not None else list(range(frames))
    with h5py.File(path, "w") as f:
        for k, v in (root_attrs or {}).items():
            f.attrs[k] = v
        ep = f.create_group(f"episode_{seed}")
        for k, v in (episode_attrs or {}).items():
            ep.attrs[k] = v
        st = ep.create_group("setup")
        for k, v in (setup_attrs or {}).items():
            st.attrs[k] = v
        st["seed"] = np.int64(setup_seed if setup_seed is not None else seed)
        st["difficulty"] = "xhard1"
        st["task_goal"] = "pick the cube"
        st["front_camera_intrinsic"] = np.eye(3, dtype=np.float32)
        for n, i in enumerate(ids):
            g = ep.create_group(f"timestep_{i}")
            j_off = offset if joint_offset_from is not None and n >= joint_offset_from else 0.0
            s_off = offset if state_offset_from is not None and n >= state_offset_from else 0.0
            p_off = offset if gripper_offset_from is not None and n >= gripper_offset_from else 0.0
            g["action/joint_action"] = (np.arange(8) * 0.01 + n * 0.1 + j_off + seed).astype(joint_dtype)
            g["obs/joint_state"] = (np.arange(7) * 0.02 + n * 0.2 + s_off).astype(np.float32)
            g["obs/gripper_state"] = np.full(2, 0.04 + p_off, dtype=np.float32)
            img = np.full((4, 4, 3), (n * 7 + seed) % 251, dtype=np.uint8)
            if frame0_pixel and n == 0:
                img[0, 0, 0] = (int(img[0, 0, 0]) + 9) % 256
            g["obs/front_rgb"] = img
            g["obs/wrist_rgb"] = img[::-1].copy()
            g["info/is_completed"] = np.bool_(n == len(ids) - 1)
            g["info/simple_subgoal"] = (subgoals[n] if subgoals else f"sg{n}")
            if extra_field is not None:
                g["obs/extra"] = np.full(3, extra_field.get(n, 0.0), dtype=np.float32)
            kind = (waypoint or {}).get(n)
            if kind == "real":
                g["action/waypoint_action"] = np.full(7, 0.3, dtype=np.float64)
            elif kind == "nan":
                g["action/waypoint_action"] = np.full(7, np.nan, dtype=np.float32)
            elif kind == "int":
                g["action/waypoint_action"] = np.zeros(7, dtype=np.int64)
            if drop is not None and drop[0] == n:
                del g[drop[1]]
    return path


#: 失败局占位文件：录像器失败时留下的小文件，每次失败字节相同，h5py 打不开
PLACEHOLDER = b"FAILED-EPISODE-PLACEHOLDER\n" * 30


def write_placeholder(path: Path, payload: bytes = PLACEHOLDER) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── 生成输出根（hard_parity generate 的 identities.jsonl + launch-*.json）─────────────


def id_line(task: str, tier: str, seed: int, *, ok: bool = True, sha: str | None = None, path: str | None = None,
            error_type: str | None = None, **extra: Any) -> dict[str, Any]:
    line = {"task": task, "tier": tier, "seed": seed, "success": ok, "path": path, "sha256": sha,
            "error_type": error_type if error_type is not None else (None if ok else "DatasetGenerationError")}
    if sha is not None and path is None:
        line["path"] = f"episodes/{tier}/{task}_episode_{seed}/hdf5_files/x.h5"
    line.update(extra)
    return line


def write_run(root: Path, lines: list[dict[str, Any]], *, workers: int, gpu: str, host: str = "gl1001",
              driver: str = "595.71.05", launch: bool = True) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "identities.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines),
                                          encoding="utf-8")
    if launch:
        (root / "launch-1.json").write_text(json.dumps({"schema": LAUNCH_SCHEMA, "workers": workers,
                                                        "gpu_model": gpu, "driver": driver, "host": host}),
                                            encoding="utf-8")
    return root


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


# ── 噪声基线小样本：a、b 两遍 + gen-compare 比对记录 + build-ref 参照 ────────────────


def build_baseline(root: Path, layouts: dict[str, list[tuple[str, str, int, str]]], *,
                   run_names: dict[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """按布局造基线。``layouts[集合] = [(task, tier, seed, kind)]``，kind ∈ stable／known_fail／jitter：

    - stable：a、b 两遍都成功，h5 字节相同（b 遍是 a 遍的拷贝）；
    - known_fail：两遍都失败（DatasetGenerationError），留下相同的占位文件；
    - jitter：a 遍成功、b 遍失败。

    比对记录由真实 ``noise_gate.py gen-compare`` 写出，参照由真实 ``gen-regress build-ref`` 写出。
    返回 {ref, runs: {集合: (a 根, b 根)}, ids: {集合: 清单}, h5: {(集合, seed): a 遍 h5}, gen_root, lines}。
    """
    ng = noise_gate()
    root = Path(root)
    gen_root = root / "gen"
    rec = root / "records"
    out: dict[str, Any] = {"runs": {}, "ids": {}, "h5": {}, "gen_root": gen_root, "layouts": layouts}
    argv = ["gen-regress", "build-ref", "--records", str(rec), "--out", str(root / "noise-ref.json")]
    for set_name, layout in layouts.items():
        a_name, b_name = (run_names or {}).get(set_name, (f"{set_name}-a", f"{set_name}-b"))
        ids = write_jsonl(root / f"ids-{set_name}.jsonl", [{"task": t, "tier": tier, "seed": s}
                                                             for t, tier, s, _k in layout])
        lines = {a_name: [], b_name: []}
        for t, tier, s, kind in layout:
            rel = f"episodes/{tier}/{t}_episode_{s}/hdf5_files/{t}_seed{s}.h5"
            a_path, b_path = gen_root / a_name / rel, gen_root / b_name / rel
            if kind == "known_fail":
                write_placeholder(a_path)
                write_placeholder(b_path)
            else:
                write_h5(a_path, seed=s)
                if kind == "stable":
                    b_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(a_path, b_path)
                else:
                    write_placeholder(b_path)
            ok_a = kind != "known_fail"
            ok_b = kind == "stable"
            lines[a_name].append(id_line(t, tier, s, ok=ok_a, sha=sha256(a_path), path=rel))
            lines[b_name].append(id_line(t, tier, s, ok=ok_b, sha=sha256(b_path), path=rel))
            out["h5"][(set_name, s)] = a_path
        for name in (a_name, b_name):
            write_run(gen_root / name, lines[name], workers=4, gpu="NVIDIA A40", host=f"host-{name}")
        a_root, b_root = gen_root / a_name, gen_root / b_name
        assert ng.main(["gen-compare", "--ref", str(a_root), "--new", str(b_root), "--identities", str(ids),
                        "--out", str(rec / f"{set_name}-ab.jsonl")]) == 0
        out["runs"][set_name] = (a_root, b_root)
        out["ids"][set_name] = ids
        argv += ["--runs", f"{set_name}={a_root},{b_root}", f"--identities-{set_name}", str(ids)]
    out["build_argv"] = argv
    out["ref"] = root / "noise-ref.json"
    return out


def ref_episodes(ref_path: Path, set_name: str) -> list[dict[str, Any]]:
    return json.loads(Path(ref_path).read_text(encoding="utf-8"))["sets"][set_name]["episodes"]


def first_run_lines(ref_path: Path, set_name: str, overrides: dict[int, dict[str, Any] | None] | None = None,
                    only: Iterable[int] | None = None) -> list[dict[str, Any]]:
    """按参照造一遍「新跑」的 identities 行：默认每局复现基线 a 遍（稳定局成功同 sha；确定性失败局失败、占位 sha 相同、
    h5 打不开；抖动局成功同 a 遍）。``overrides[seed]`` 覆盖字段，值为 None 表示这一局缺行。"""
    overrides = overrides or {}
    keep = set(only) if only is not None else None
    lines = []
    for e in ref_episodes(ref_path, set_name):
        s = e["seed"]
        if keep is not None and s not in keep:
            continue
        if s in overrides and overrides[s] is None:
            continue
        if e["class"] == "known_fail":
            line = id_line(e["task"], e["tier"], s, ok=False, sha=e["shas"][0], h5_error="OSError: unable to open")
        else:
            line = id_line(e["task"], e["tier"], s, ok=True, sha=e["shas"][0])
        line.update(overrides.get(s) or {})
        lines.append(line)
    return lines
