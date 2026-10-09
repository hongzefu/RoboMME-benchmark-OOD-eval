"""专家演示渲染 ``dev-scripts/media/render_expert_demos.py`` 的夹具测试：合成 h5（字段名照录制器 ``RecordWrapper`` 的
写法：``episode_<N>/timestep_<K>/{obs/front_rgb, obs/wrist_rgb, obs/joint_state, obs/gripper_state,
action/joint_action, info/is_video_demo}``、``setup/task_goal``），不读真实 V9 h5、不跑仿真。

期望全部手写：timestep 按 K 数值排序（timestep_10 在 timestep_9 之后）、演示段红框只在 ``is_video_demo`` 帧、
两版帧数都等于 h5 帧数、文件名 ``<Task>_ep<N>_<task_goal>_<tier>[_annotation].mp4``、续跑跳过、缺 h5 记失败。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests._support.dev_loaders import load_script

GOAL = "pick up the red cube"
N_FRAMES = 12  # 含 timestep_10、timestep_11：字符串排序会把它们排到 timestep_1 之后
DEMO = 3


def _mod():
    return load_script("media/render_expert_demos.py")


def _frame(k: int, cam: int) -> np.ndarray:
    f = np.zeros((256, 256, 3), np.uint8)
    f[..., 0] = (40 + 17 * k) % 256
    f[..., 1] = 30 + 100 * cam
    return f


def write_h5(path: Path, *, episode: int = 7, n: int = N_FRAMES, demo: int = DEMO, dup_at: int | None = None) -> Path:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        ep = h5.create_group(f"episode_{episode}")
        setup = ep.create_group("setup")
        setup.create_dataset("task_goal", data=np.array([GOAL.encode(), b"alt goal"]))
        setup.create_dataset("seed", data=123)
        order = list(range(n))
        np.random.default_rng(0).shuffle(order)  # 打乱创建顺序：读取端必须自己按数值排
        for k in order:
            ts = ep.create_group(f"timestep_{k}")
            obs = ts.create_group("obs")
            obs.create_dataset("front_rgb", data=_frame(k, 0))
            obs.create_dataset("wrist_rgb", data=_frame(k, 1))
            obs.create_dataset("joint_state", data=np.full(7, k / 10.0))
            obs.create_dataset("gripper_state", data=np.array([0.04, 0.04]))
            act = ts.create_group("action")
            if k < demo:
                act.create_dataset("joint_action", data="None", dtype=h5py.special_dtype(vlen=str))
            else:
                act.create_dataset("joint_action", data=np.full(8, k / 100.0))
            info = ts.create_group("info")
            info.create_dataset("is_video_demo", data=k < demo)
        if dup_at is not None:
            ts = ep.create_group(f"timestep_{dup_at}_dup1")
            obs = ts.create_group("obs")
            obs.create_dataset("front_rgb", data=_frame(99, 0))
            obs.create_dataset("wrist_rgb", data=_frame(99, 1))
            obs.create_dataset("joint_state", data=np.zeros(7))
            obs.create_dataset("gripper_state", data=np.zeros(2))
            ts.create_group("info").create_dataset("is_video_demo", data=False)
    return path


def test_timestep_numeric_order_and_meta(tmp_path):
    import h5py

    m = _mod()
    p = write_h5(tmp_path / "a.h5", dup_at=4)
    with h5py.File(p, "r") as h5:
        ep = m.pick_episode(h5, 7)
        meta = m.read_meta(ep)
    want = [f"timestep_{k}" for k in range(5)] + ["timestep_4_dup1"] + [f"timestep_{k}" for k in range(5, N_FRAMES)]
    assert meta["keys"] == want
    assert meta["demo"] == [k < DEMO for k in range(5)] + [False] + [False] * (N_FRAMES - 5)
    assert meta["task_goal"] == GOAL


def test_state_action_and_names(tmp_path):
    import h5py

    m = _mod()
    p = write_h5(tmp_path / "a.h5")
    with h5py.File(p, "r") as h5:
        ep = m.pick_episode(h5, None)
        assert m.action_of(ep["timestep_0"]) is None  # 演示段 "None" 字符串
        assert np.allclose(m.action_of(ep["timestep_5"]), np.full(8, 0.05))
        st = m.state_of(ep["timestep_5"])
        assert st.dtype == np.float32 and st.shape == (8,) and np.allclose(st, [0.5] * 7 + [0.04])
    plain, annot = m.names_for("VideoUnmask", 3, GOAL, "xhard2")
    assert plain == f"VideoUnmask_ep3_{GOAL}_xhard2.mp4" and annot == f"VideoUnmask_ep3_{GOAL}_xhard2_annotation.mp4"


def test_official_video_recorder_border_only_on_demo():
    m = _mod()
    VR = m.official_video_recorder()
    f, w = _frame(1, 0), _frame(1, 1)
    plain = VR._make_frame(f, w, is_video_demo=False)
    demo = VR._make_frame(f, w, is_video_demo=True)
    assert plain.shape == demo.shape == (256, 512, 3)
    assert np.array_equal(plain, np.hstack([f, w]))
    assert tuple(demo[0, 0]) == (255, 0, 0) and tuple(demo[128, 256]) == tuple(plain[128, 256])


class _Builder:
    """ood builder 替身：局 0、1 的身份，h5 由交付清单按 (task, tier, seed) 找。"""

    def __init__(self, task):
        self.task = task

    def get_episode_num(self):
        return 2

    def resolve_identity(self, ep):
        return {"episode": ep, "tier": "xhard2", "seed": 9000 + ep, "candidate": ep, "spec_sha256": "0" * 64}


def _official_root(repo_root: Path) -> Path:
    import os

    if (repo_root / "third_party/mme-vla/examples/robomme/utils.py").is_file():
        return repo_root
    return Path(os.environ["SGEVAL_THIRD_PARTY"]).parent


@pytest.mark.slow
def test_end_to_end_two_versions_frames_match_and_resume(tmp_path, repo_root, capsys):
    if not shutil.which("ffprobe"):
        pytest.skip("未验证：缺 ffprobe")
    m = _mod()
    rows = []
    for ep in range(2):
        h5 = write_h5(tmp_path / "h5" / f"VideoUnmask_ep{ep}.h5", episode=ep)
        rows.append({"task": "VideoUnmask", "tier": "xhard2", "seed": 9000 + ep, "episode": ep, "h5": str(h5)})
    delivery = tmp_path / "delivery.json"
    delivery.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["--delivery", str(delivery), "--out", str(out), "--tasks", "VideoUnmask", "--workers", "2",
            "--official-root", str(_official_root(repo_root))]
    assert m.main(argv, builder_factory=_Builder) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("EXPERT_DEMOS=PASS episodes=2 files=4 frames_match=2 "), line
    for ep in range(2):
        for suffix in ("", "_annotation"):
            p = out / f"VideoUnmask_ep{ep}_{GOAL}_xhard2{suffix}.mp4"
            info = m.R.probe_video(m.R.find_ffprobe(m.R.find_ffmpeg()), p)
            assert info["frames"] == N_FRAMES and info["codec"] == "av1" and info["pix_fmt"] == "yuv420p"
            assert info["fps"] == "30" and info["width"] == 512
    man = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["status"] for r in man] == ["rendered", "rendered"] and all(r["demo_frames"] == DEMO for r in man)
    # 续跑：两版都在且帧数对得上 → 跳过
    assert m.main(argv, builder_factory=_Builder) == 0
    assert "skipped=2 fail=0" in capsys.readouterr().out


def test_missing_h5_fails(tmp_path, capsys):
    m = _mod()
    delivery = tmp_path / "delivery.json"
    delivery.write_text(json.dumps({"rows": []}), encoding="utf-8")
    rc = m.main(["--delivery", str(delivery), "--out", str(tmp_path / "out"), "--tasks", "VideoUnmask"],
                builder_factory=_Builder)
    assert rc == 1
    assert "EXPERT_DEMOS=FAIL episodes=2 files=0 frames_match=0 skipped=0 fail=2" in capsys.readouterr().out
