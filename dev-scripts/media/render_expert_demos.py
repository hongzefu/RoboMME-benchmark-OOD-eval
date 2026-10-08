#!/usr/bin/env python3
"""专家演示视频：从 V9 生成 h5 离线出 ood 每局两版视频（1008 拆分方案 §四「专家演示」、第二部分 §二 media 行、S4）。

只读 h5、不跑仿真、不碰录制器。ood 的局号与身份取子模块 ``robomme_hard`` 的 ``BenchmarkEnvBuilder(task, "ood")``
（``resolve_identity`` 给 tier／seed），按 (task, tier, seed) 在交付清单 ``--delivery``（V9 的
``delivery.local.json``，每行带 ``h5`` 绝对路径与相对 ``path``）里找该局的 h5。每局输出到 ``--out``（缺省
``<repo>/artifacts/expert_demos/ood/``）两个文件::

    <Task>_ep<N>_<task_goal>_<tier>.mp4              无字、只有演示段红框：官方 scripts/evaluation.py 的 VideoRecorder 版式
    <Task>_ep<N>_<task_goal>_<tier>_annotation.mp4   官方 RolloutRecorder 版式（帧号、任务目标、动作、状态文字区）

文件名经 ``official_render.safe_filename``（超 255 字节截断加哈希）。读 h5 的口径（字段名与旧仓
``scripts/dataset_replay.py``、旧站点工具 ``render_xhard0.py``（现 ``dev-scripts/site/``）交叉核对过）：

* 每个 h5 只读一个 ``episode_<N>`` 组（有多个时取与交付行 ``episode`` 同号者），其下 ``timestep_<K>``（以及录制器
  去重用的 ``timestep_<K>_dup<M>``）**按 K、M 的数值排序**，不按字符串排；
* 每个 timestep 一帧：``obs/front_rgb`` 与 ``obs/wrist_rgb``（256×256×3 uint8）左右拼接；不补没有来源的 reset 帧；
* 演示段边界取 ``info/is_video_demo``（红框只画在演示段）；任务目标取 ``setup/task_goal`` 的第一项；
* 文字版的状态为 ``obs/joint_state``（7）+ ``obs/gripper_state`` 第一维（与官方 ``pack_state`` 同式），动作为
  ``action/joint_action``（演示段写成字符串 ``"None"`` 时按无动作显示）；不画子目标。官方文字区高度随换行可能
  逐帧不同，统一在底部补黑到全局最高（编码要求逐帧同尺寸）。

两版都是 30 fps、AV1 4:2:0 封装 MP4（``-movflags +faststart``，参数同 ``official_render.WEB_ENCODE_ARGS``，尺寸不是 16 的
倍数时与官方 ``save_video`` 一样缩放到取整尺寸）；逐帧流式喂给编码器，不把整局帧堆在内存里。编码后用 ffprobe 完整
数帧，两版帧数都等于 h5 帧数才算这一局通过（``frames_match``）；已有输出且帧数对得上即跳过（可续跑）。

用法::

    python dev-scripts/media/render_expert_demos.py --delivery <V9 delivery.local.json> --workers 16 \\
        [--out <目录>] [--tasks A,B] [--episodes a:b] [--official-root <含 third_party/mme-vla 的仓根>]

逐局写 ``<out>/manifest.jsonl``；末行 ``EXPERT_DEMOS=PASS|FAIL episodes=<n> files=<n> frames_match=<n>
[skipped=<n> fail=<n>]``（``files`` 为现存两版视频数，``frames_match`` 为两版帧数都等于 h5 帧数的局数）。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from robomme_hard_eval.record import official_render as R  # noqa: E402

DATASET = "ood"
FPS = 30
DEFAULT_OUT = REPO / "artifacts" / "expert_demos" / DATASET
TS_RE = re.compile(r"^timestep_(?P<k>\d+)(?:_dup(?P<m>\d+))?$")


# ── h5 读取 ──────────────────────────────────────────────────────────────────


def timestep_keys(episode) -> list[str]:
    """``timestep_<K>``／``timestep_<K>_dup<M>`` 按 (K, M) 数值排序（不按字符串排：timestep_10 在 timestep_9 之后）。"""
    keys = []
    for k in episode.keys():
        m = TS_RE.match(k)
        if m:
            keys.append((int(m["k"]), int(m["m"] or 0), k))
    return [k for _, _, k in sorted(keys)]


def _scalar_bool(ds) -> bool:
    return bool(np.reshape(np.asarray(ds[()]), -1)[0])


def _text(raw) -> str:
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")
    return str(raw)


def pick_episode(h5, episode: int | None):
    groups = sorted((k for k in h5.keys() if k.startswith("episode_")), key=lambda k: int(k.split("_")[1]))
    if not groups:
        raise ValueError("h5 里没有 episode_<N> 组")
    if len(groups) == 1:
        return h5[groups[0]]
    want = f"episode_{episode}"
    if episode is None or want not in h5:
        raise ValueError(f"h5 里有多个 episode 组 {groups[:5]}，无法按 episode={episode} 选定")
    return h5[want]


def read_meta(episode) -> dict:
    """任务目标与每帧的演示段标志（只读小数组，不读画面）。"""
    keys = timestep_keys(episode)
    if not keys:
        raise ValueError("episode 组里没有 timestep_<K>")
    demo = []
    for k in keys:
        info = episode[k].get("info")
        demo.append(bool(info is not None and "is_video_demo" in info and _scalar_bool(info["is_video_demo"])))
    return {"keys": keys, "demo": demo, "task_goal": _text(episode["setup"]["task_goal"][()])}


def frame_of(ts) -> tuple[np.ndarray, np.ndarray]:
    obs = ts["obs"]
    return np.asarray(obs["front_rgb"][()], dtype=np.uint8), np.asarray(obs["wrist_rgb"][()], dtype=np.uint8)


def state_of(ts) -> np.ndarray:
    obs = ts["obs"]
    joint = np.asarray(obs["joint_state"][()]).reshape(-1)
    grip = np.asarray(obs["gripper_state"][()]).reshape(-1)
    return np.concatenate([joint, grip[:1]], axis=0).astype(np.float32)


def action_of(ts):
    act = ts.get("action")
    if act is None or "joint_action" not in act:
        return None
    raw = act["joint_action"][()]
    if isinstance(raw, (bytes, str, np.bytes_)) or (isinstance(raw, np.ndarray) and raw.dtype.kind in "OSU"):
        return None
    return np.asarray(raw, dtype=np.float64).reshape(-1)


# ── 官方两种版式 ─────────────────────────────────────────────────────────────


def official_eval_script() -> Path:
    """官方 ``scripts/evaluation.py``：benchmark 子模块里那份；子模块未检出时按已安装 ``robomme`` 包的位置找。"""
    p = REPO / "third_party" / "robomme_benchmark" / "scripts" / "evaluation.py"
    if p.is_file():
        return p
    spec = importlib.util.find_spec("robomme")
    if spec is not None and spec.submodule_search_locations:
        cand = Path(list(spec.submodule_search_locations)[0]).parents[1] / "scripts" / "evaluation.py"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"找不到官方 scripts/evaluation.py（{p}）")


def official_video_recorder():
    """从官方 ``scripts/evaluation.py`` 原文只取 ``VideoRecorder`` 类执行（不导入整模块，不拉起 torch／仿真）。"""
    import cv2

    from robomme_hard_eval.models import _official_defs as D

    torch_stub = types.SimpleNamespace(Tensor=type("Tensor", (), {}))
    ns = D.extract_defs(official_eval_script(), ["VideoRecorder"], {"cv2": cv2, "torch": torch_stub})
    return ns["VideoRecorder"]


class _Encoder:
    """流式 AV1 4:2:0 MP4 编码（参数同 ``official_render.WEB_ENCODE_ARGS``）。"""

    def __init__(self, out: Path, w: int, h: int, ffmpeg: str):
        ow, oh = R.web_size(w, h)
        vf = ["-vf", f"scale={ow}:{oh}"] if (ow, oh) != (w, h) else []
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0", "-an",
                                      *vf, *R.WEB_ENCODE_ARGS, "-f", "mp4", str(out)],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)
        self.shape = (h, w, 3)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"帧尺寸 {frame.shape} 与首帧 {self.shape} 不一致")
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        if code:
            self.err.seek(0)
            raise RuntimeError(f"编码失败 rc={code}：{self.err.read().decode(errors='replace')[-600:]}")


def _pad(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    out = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    out[: img.shape[0]] = img
    return out


def render_plain(episode, meta: dict, out: Path, ffmpeg: str) -> None:
    VR = official_video_recorder()
    enc = None
    for k, demo in zip(meta["keys"], meta["demo"]):
        front, wrist = frame_of(episode[k])
        frame = VR._make_frame(front, wrist, is_video_demo=demo)
        if enc is None:
            enc = _Encoder(out, frame.shape[1], frame.shape[0], ffmpeg)
        enc.write(frame)
    enc.close()


def render_annotation(episode, meta: dict, out: Path, ffmpeg: str, official_root: Path | None) -> None:
    Recorder, _demo_tasks = R.load_official(official_root)
    with tempfile.TemporaryDirectory(prefix=".expert-rec-") as td:
        rec = Recorder(Path(td), meta["task_goal"], fps=FPS)

        def draw(k: str, demo: bool) -> np.ndarray:
            ts = episode[k]
            front, wrist = frame_of(ts)
            rec.record(front, wrist, state_of(ts), action=action_of(ts), is_video_demo=demo, subgoal=None)
            return rec.total_images.pop()

        # 第一遍只量高度（官方文字区高度随换行变化），第二遍画并流式编码；帧号文字依赖已录帧数，两遍各自从 0 计
        heights = []
        for k, demo in zip(meta["keys"], meta["demo"]):
            img = draw(k, demo)
            rec.total_images.append(None)  # 占位：保持官方「Frame: <已录帧数>」计数
            heights.append(img.shape[0])
        rec.total_images.clear()
        hmax = max(heights)
        enc = None
        for k, demo in zip(meta["keys"], meta["demo"]):
            img = _pad(draw(k, demo), hmax)
            rec.total_images.append(None)
            if enc is None:
                enc = _Encoder(out, img.shape[1], hmax, ffmpeg)
            enc.write(img)
        enc.close()


# ── 一局 ─────────────────────────────────────────────────────────────────────


def probe_frames(path: Path, ffmpeg: str) -> int:
    try:
        return int(R.probe_video(R.find_ffprobe(ffmpeg), path)["frames"])
    except Exception:  # noqa: BLE001
        return -1


def names_for(task: str, ep: int, goal: str, tier: str) -> tuple[str, str]:
    base = f"{task}_ep{ep}_{goal}_{tier}"
    return R.safe_filename(base + ".mp4"), R.safe_filename(base + "_annotation.mp4")


def render_one(job: dict) -> dict:
    import h5py

    out_dir, ffmpeg = Path(job["out"]), job["ffmpeg"]
    official_root = Path(job["official_root"]) if job.get("official_root") else None
    rec = {k: job[k] for k in ("task", "episode", "tier", "seed", "h5")}
    try:
        with h5py.File(job["h5"], "r") as h5:
            ep = pick_episode(h5, job.get("h5_episode"))
            meta = read_meta(ep)
            n = len(meta["keys"])
            plain, annot = names_for(job["task"], job["episode"], meta["task_goal"], job["tier"])
            rec.update(frames=n, demo_frames=int(sum(meta["demo"])), task_goal=meta["task_goal"], plain=plain,
                       annotation=annot)
            targets = {"plain": out_dir / plain, "annotation": out_dir / annot}
            got = {name: probe_frames(p, ffmpeg) if p.is_file() else -1 for name, p in targets.items()}
            if all(v == n for v in got.values()):
                rec.update(status="skipped", plain_frames=got["plain"], annotation_frames=got["annotation"])
                return rec
            for name, path in targets.items():
                if got[name] == n:
                    continue
                tmp = path.with_name(f".{path.stem}.part.mp4")
                if name == "plain":
                    render_plain(ep, meta, tmp, ffmpeg)
                else:
                    render_annotation(ep, meta, tmp, ffmpeg, official_root)
                tmp.replace(path)
            rec.update(status="rendered", plain_frames=probe_frames(targets["plain"], ffmpeg),
                       annotation_frames=probe_frames(targets["annotation"], ffmpeg))
    except Exception as e:  # noqa: BLE001 逐局失败如实记录
        rec.update(status="fail", error=f"{type(e).__name__}: {e}"[:600])
    return rec


# ── 局表 ─────────────────────────────────────────────────────────────────────


def read_delivery(path: Path) -> dict[tuple[str, str, int], dict]:
    """交付清单 → {(task, tier, seed): 行}；h5 取绝对 ``h5``，不存在时按清单所在目录解析相对 ``path``。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in doc["rows"]:
        h5 = Path(r.get("h5") or "")
        if not h5.is_file() and r.get("path"):
            h5 = (Path(path).parent / r["path"]).resolve()
        out[(r["task"], r["tier"], int(r["seed"]))] = dict(r, h5=str(h5))
    return out


def make_builder(task: str):
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder

    return BenchmarkEnvBuilder(task, dataset=DATASET)


def plan_jobs(tasks: list[str], lo: int, hi: int | None, delivery: dict, builder_factory=make_builder) -> list[dict]:
    jobs = []
    for task in tasks:
        b = builder_factory(task)
        n = int(b.get_episode_num())
        for ep in range(lo, n if hi is None else min(hi, n)):
            ident = b.resolve_identity(ep)
            row = delivery.get((task, ident["tier"], int(ident["seed"])))
            jobs.append({"task": task, "episode": ep, "tier": ident["tier"], "seed": int(ident["seed"]),
                         "h5": None if row is None else row["h5"],
                         "h5_episode": None if row is None else row.get("episode")})
    return jobs


def parse_episodes(text: str | None) -> tuple[int, int | None]:
    if not text:
        return 0, None
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"--episodes 须为 a:b 半开区间：{text!r}")
    lo, hi = int(a or 0), (int(b) if b else None)
    if lo < 0 or (hi is not None and hi <= lo):
        raise argparse.ArgumentTypeError(f"--episodes 区间为空或非法：{text!r}")
    return lo, hi


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delivery", type=Path, required=True, help="V9 交付清单 delivery.local.json（每行带 h5 路径）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tasks", default=None, help="逗号分隔任务名；缺省官方 16 任务")
    ap.add_argument("--episodes", type=parse_episodes, default=(0, None), help="ood builder 局号半开区间 a:b")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--official-root", type=Path, default=None,
                    help=f"官方 RolloutRecorder 所在仓根（缺省取环境变量 {R.ENV_OFFICIAL_ROOT} 或评估仓根）")
    ap.add_argument("--ffmpeg", default=None, help="支持 libaom-av1 的 ffmpeg（缺省自动找）")
    return ap


def main(argv: list[str] | None = None, *, builder_factory=make_builder) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers 至少 1")
    from robomme_hard.env_record_wrapper import hard_specs as hs

    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else list(hs.ALL_TASKS)
    bad = [t for t in tasks if t not in hs.ALL_TASKS]
    if bad:
        raise SystemExit(f"未知任务 {bad}")
    lo, hi = args.episodes
    ffmpeg = args.ffmpeg or R.find_ffmpeg()
    jobs = plan_jobs(tasks, lo, hi, read_delivery(args.delivery), builder_factory)
    args.out.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        j.update(out=str(args.out), ffmpeg=ffmpeg, official_root=str(args.official_root) if args.official_root else None)
    rows = []
    todo = [j for j in jobs if j["h5"] and Path(j["h5"]).is_file()]
    for j in jobs:
        if j not in todo:
            rows.append({k: j[k] for k in ("task", "episode", "tier", "seed", "h5")} | {"status": "fail",
                                                                                          "error": "h5_missing"})
    if args.workers == 1:
        results = map(render_one, todo)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = pool.map(render_one, todo)
    try:
        for i, r in enumerate(results, 1):
            rows.append(r)
            if r["status"] == "fail":
                print(f"EXPERT_DEMO_FAIL task={r['task']} ep={r['episode']} error={r.get('error')}", flush=True)
            if i % 16 == 0 or i == len(todo):
                print(f"EXPERT_DEMOS_PROGRESS done={i}/{len(todo)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    rows.sort(key=lambda r: (tasks.index(r["task"]), r["episode"]))
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                     for r in rows), encoding="utf-8")
    files = sum((args.out / r[k]).is_file() for r in rows for k in ("plain", "annotation") if r.get(k))
    match = sum(r.get("status") != "fail" and r.get("plain_frames") == r.get("frames") == r.get("annotation_frames")
                for r in rows)
    fail = sum(r.get("status") == "fail" for r in rows)
    skipped = sum(r.get("status") == "skipped" for r in rows)
    ok = fail == 0 and match == len(rows) and files == 2 * len(rows) and rows
    print(f"EXPERT_DEMOS={'PASS' if ok else 'FAIL'} episodes={len(rows)} files={files} frames_match={match} "
          f"skipped={skipped} fail={fail} out={args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
