#!/usr/bin/env python3
"""PonderPounce 原侧驱动（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.4、1.7；子任务 S4）。

独立进程，只导入官方 ``robomme``（benchmark 子模块 ``third_party/robomme_benchmark/src/robomme``）与 vla-eval 0.7.0
（客户端扩展环境 client-env），不导入 ``robomme_hard``：启动时断言 ``robomme.__file__`` 在子模块 ``src/robomme/`` 下、``"robomme_hard" not in
sys.modules``，每局结束再查一次。

流程（照 vla-eval ``orchestrator.py`` 的单分片循环，只在外围加记录）：

1. ``RoboMMEBenchmark.configure_render("gpu")``（不改环境变量）；
2. 整个分片一条 ``Connection(url, timeout=300.0)``，``connect(benchmark=PP_BENCHMARK)``；
3. 每个任务一个 ``RoboMMEBenchmark(tasks=[task], action_space="joint_angle", max_steps=1300)``（实际为它的记录子类，
   见下）；每个分片行调用 ``SyncEpisodeRunner().run_episode(bench, {"name","env_id","episode_idx": source_episode},
   conn, max_steps=1300, recorder=<固定 sid 的记录器>)``；
4. ``TimeoutError``／``ConnectionClosed``／``RuntimeError``：本局记 ``error``（基础设施），``conn.reconnect()`` 后
   继续下一局；``ConnectionError``（服务不可达、重试耗尽）：本局记 error 并停止整个分片（退出码 2）；其他异常记
   error、继续。

固定 sid 记录器：vla-eval ``NullEpisodeRecorder`` 的子类，``is_active=True``、``sid``／``eid`` 为
``pp_client.fixed_sid`` 的结果、``eval_id``／``db_path`` 为空串；画面与逐步记录全部为空操作（不开录制库、
不出官方视频）。于是 EPISODE_START 载荷与新侧 ``pp_client`` 完全相同。

外围记录（不碰官方 ``robomme`` 的任何方法、不 monkeypatch）：``RoboMMEBenchmark`` 的子类只覆写 ``reset``——调用父类
``reset`` 后，把 ``self._env`` 换成一个委托代理，代理的 ``step`` 原样转发给官方环境并把返回的五元组交给记录器；
``close`` 等其余属性透传。记录内容：

- ``trace.jsonl``（``trace_writer``，route ``pp-orig``）：演示行、每个协议帧的规范化字节哈希、模型动作块、每步画面／
  状态／8 维动作／终态（字段口径与新侧 ``pp_client`` 共用同一批辅助函数）；
- 原始帧（供 S6 启动器转码，格式见下）。

原始帧落盘格式（每局一个目录，``--no-frames`` 时不写）::

    <out>/<key>.a<attempt>/frames/front.rgb24   # 逐帧 H×W×3 uint8 原始字节顺序拼接（ffmpeg rawvideo rgb24）
    <out>/<key>.a<attempt>/frames/wrist.rgb24   # 同上，腕部相机
    <out>/<key>.a<attempt>/frames/frames.json   # {"pix_fmt":"rgb24","streams":{"front":{"width","height","count"},
                                                #  "wrist":{…}},"demo_frames","init_frames","exec_steps",
                                                #  "missing_steps","order"}

帧顺序：``reset`` 返回的 ``front_rgb_list`` 全部帧（``demo_frames`` 个演示帧 + 1 个初始帧），之后每执行一步追加
该步观测的 ``front_rgb_list[-1]``（腕部同理）；某步观测为空（``obs`` 为 None 或无画面）时不追加，步号记入
``missing_steps``。所以 ``count = demo_frames + 1 + exec_steps - len(missing_steps)``。转码示例::

    ffmpeg -f rawvideo -pix_fmt rgb24 -s <W>x<H> -r 30 -i front.rgb24 \\
           -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart front.mp4

结果行（``<out>/results.jsonl``，每局一行，追加并 fsync）：分片行的身份字段（``task``、``tier``、``seed``、
``source_episode``、``builder_episode``、``key`` 等）+ ``side="orig"``、``policy="pp"``、``dataset="hard-verify"``、
``attempt``、``status``、``task_success``、``exec_steps``、``steps``、``demo_frames``、``max_steps``、
``effective_max_steps``、``sid``、``eid``、``episode_idx``、``error``、``infra``、``infra_reason``、``ep_dir``、
``trace_path``、``frames_dir``、``video_frames``、``wall_s``。

用法::

    python pp_official_runner.py --shard <shard-NN.json> --out <目录> --port <端口> \\
        [--host 127.0.0.1] [--max-steps 1300] [--attempt 1] [--only <key,…>] [--no-frames]
    python pp_official_runner.py --check-imports     # 只做导入断言，打印 OFFICIAL_IMPORTS=PASS …

分片文件为 ``eval_manifest.py --mode hard0`` 产出的 ``shard-NN.json``（执行身份行数组）；本驱动只接受
``tier == "xhard0"`` 的行。退出码：0 分片跑完（含 error 局）；2 服务不可达中止；3 分片或导入断言不合格。
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p and str(_Path(p).resolve()) != _HERE] + [_HERE]

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
#: benchmark 子模块源码根（官方 ``robomme``）；评估包源码根（``pp_client``／``trace_writer`` 拆仓后所在）
BENCH_SRC = REPO / "third_party" / "robomme_benchmark" / "src"
EVAL_SRC = REPO / "src"
for _p in (EVAL_SRC, BENCH_SRC):  # 子模块 src 在最前
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_eval(name: str, path: Path):
    """按文件路径加载评估包模块，模块名用旧名（与模型客户端的 ``load_sibling`` 同一别名，已加载则复用）。"""
    import importlib.util

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


trace_writer = _load_eval("trace_writer", EVAL_SRC / "robomme_hard_eval" / "record" / "trace_writer.py")
TraceWriter = trace_writer.TraceWriter
pp_client = _load_eval("pp_client", EVAL_SRC / "robomme_hard_eval" / "models" / "pp.py")

MAX_STEPS = 1300
DATASET = "hard-verify"
EXIT_UNREACHABLE = 2
EXIT_BAD_INPUT = 3


# ── 导入断言 ───────────────────────────────────────────────────────────────


def assert_official_only() -> str:
    """``robomme`` 必须来自 benchmark 子模块 ``src/robomme/``，且进程内从未导入 ``robomme_hard``。返回 ``robomme.__file__``。"""
    import robomme

    path = Path(robomme.__file__).resolve()
    want = (BENCH_SRC / "robomme").resolve()
    if want not in path.parents:
        raise AssertionError(f"robomme 不在 benchmark 子模块 src/robomme 下：{path}")
    if "robomme_hard" in sys.modules:
        raise AssertionError("原侧进程导入了 robomme_hard")
    return str(path)


# ── 分片 ───────────────────────────────────────────────────────────────────


def load_shard(path: Path, only: str | None = None) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"分片应为 JSON 数组：{path}")
    keys = set()
    for i, r in enumerate(rows):
        for k in ("task", "tier", "seed", "source_episode", "key"):
            if k not in r:
                raise ValueError(f"分片第 {i} 行缺 {k}")
        if r["tier"] != pp_client.XHARD0:
            raise ValueError(f"分片第 {i} 行 tier={r['tier']!r}，原侧只跑 {pp_client.XHARD0}")
        if r["key"] in keys:
            raise ValueError(f"分片 key 重复：{r['key']}")
        keys.add(r["key"])
        pp_client.fixed_sid(r, DATASET)
    if only:
        want = set(only.split(","))
        rows = [r for r in rows if r["key"] in want]
    return rows


# ── 原始帧 ─────────────────────────────────────────────────────────────────


class RawFrameWriter:
    """每局一个目录：``front.rgb24``／``wrist.rgb24`` 逐帧拼接原始字节，``frames.json`` 记尺寸与帧数。"""

    STREAMS = ("front", "wrist")

    def __init__(self, frames_dir: Path) -> None:
        self.dir = Path(frames_dir)
        self.dir.mkdir(parents=True, exist_ok=False)
        self._fh = {s: (self.dir / f"{s}.rgb24").open("wb") for s in self.STREAMS}
        self.shape: dict[str, tuple | None] = {s: None for s in self.STREAMS}
        self.count = {s: 0 for s in self.STREAMS}
        self.meta: dict[str, Any] = {"demo_frames": None, "init_frames": 0, "exec_steps": 0, "missing_steps": []}
        self._closed = False

    def write(self, stream: str, frame: Any) -> None:
        a = np.ascontiguousarray(np.asarray(frame))
        if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError(f"{stream} 帧应为 H×W×3 uint8，实际 {a.dtype} {a.shape}")
        if self.shape[stream] is None:
            self.shape[stream] = a.shape
        elif self.shape[stream] != a.shape:
            raise ValueError(f"{stream} 帧尺寸变化：{self.shape[stream]} → {a.shape}")
        self._fh[stream].write(a.tobytes())
        self.count[stream] += 1

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        for fh in self._fh.values():
            fh.close()
        self._closed = True
        summ = self.summary()
        (self.dir / "frames.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                              encoding="utf-8")
        return summ

    def summary(self) -> dict:
        streams = {}
        for s in self.STREAMS:
            shp = self.shape[s]
            streams[s] = {"width": None if shp is None else int(shp[1]), "height": None if shp is None else int(shp[0]),
                          "count": self.count[s]}
        return {"pix_fmt": "rgb24", "streams": streams, "order": "reset_all_then_per_step_last", **self.meta}


# ── 外围记录：环境代理与 benchmark 子类 ─────────────────────────────────────


class EpisodeCapture:
    """一局的外围记录：reset 帧与每步五元组 → 轨迹 + 原始帧。"""

    def __init__(self, trace, frames: RawFrameWriter | None) -> None:
        self.trace = trace
        self.frames = frames
        self.demo_frames: int | None = None
        self.exec_steps = 0
        self.last_info: dict = {}
        self.terminated = False
        self.truncated = False
        self.done = False

    def on_reset(self, raw_obs: dict, task_description: str) -> None:
        self.demo_frames = pp_client.trace_reset(self.trace, raw_obs, task_description)
        if self.frames is not None:
            fronts = list(raw_obs["front_rgb_list"])
            wrists = list(raw_obs.get("wrist_rgb_list", []) or [])
            for f in fronts:
                self.frames.write("front", f)
            for w in wrists:
                self.frames.write("wrist", w)
            self.frames.meta.update(demo_frames=self.demo_frames, init_frames=1 if fronts else 0)

    def on_step(self, raw_action: Any, out: tuple) -> None:
        a8 = [float(x) for x in list(raw_action)[: pp_client.PP_ACTION_DIMS]]
        self.exec_steps += 1
        pp_client.trace_step(self.trace, self.exec_steps, out, a8)
        obs, _reward, terminated, truncated, info = out
        self.last_info = info if isinstance(info, dict) else {}
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.done = pp_client.step_done(terminated, truncated, self.last_info)
        if self.frames is not None:
            self.frames.meta["exec_steps"] = self.exec_steps
            if isinstance(obs, dict) and obs.get("front_rgb_list"):
                self.frames.write("front", obs["front_rgb_list"][-1])
                if obs.get("wrist_rgb_list"):
                    self.frames.write("wrist", obs["wrist_rgb_list"][-1])
            else:
                self.frames.meta["missing_steps"].append(self.exec_steps)


class EnvProxy:
    """官方环境的委托代理：``step`` 原样转发并把五元组交给 ``EpisodeCapture``；其余属性透传。"""

    def __init__(self, env: Any, capture: EpisodeCapture) -> None:
        self._pp_env = env
        self._pp_capture = capture

    def step(self, action: Any):
        out = self._pp_env.step(action)
        self._pp_capture.on_step(action, out)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pp_env, name)


def make_tracing_bench_class(base: type) -> type:
    """``RoboMMEBenchmark``（或同接口替身）的记录子类：只覆写 ``reset``，父类逻辑一行不改。"""

    class TracingRoboMMEBenchmark(base):  # type: ignore[misc, valid-type]
        pp_capture: EpisodeCapture | None = None

        def reset(self, task):
            raw = super().reset(task)
            cap = self.pp_capture
            if cap is not None:
                cap.on_reset(raw, self._task_description)
                self._env = EnvProxy(self._env, cap)
            return raw

    TracingRoboMMEBenchmark.__name__ = f"Tracing{base.__name__}"
    return TracingRoboMMEBenchmark


def make_fixed_sid_recorder_class(base: type) -> type:
    """vla-eval ``NullEpisodeRecorder`` 的子类：is_active 为真、sid/eid 固定、eval_id/db_path 为空串。"""

    class FixedSidRecorder(base):  # type: ignore[misc, valid-type]
        def __init__(self, sid: str, eid: str) -> None:
            super().__init__()
            self._pp_sid, self._pp_eid = sid, eid

        @property
        def is_active(self) -> bool:
            return True

        @property
        def sid(self) -> str:
            return self._pp_sid

        @property
        def eid(self) -> str:
            return self._pp_eid

        @property
        def eval_id(self) -> str:
            return ""

        @property
        def db_path(self) -> str:
            return ""

    return FixedSidRecorder


# ── 分片循环 ───────────────────────────────────────────────────────────────


def append_result(path: Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


async def run_shard(rows: list[dict], *, out_dir: Path, url: str, bench_cls: type, runner: Any,
                    conn_factory: Callable[[str, float], Any], recorder_cls: type, max_steps: int = MAX_STEPS,
                    attempt: int = 1, write_frames: bool = True,
                    import_check: Callable[[], Any] | None = None) -> dict:
    """跑一个分片；返回 ``{"episodes","errors","aborted"}``。所有依赖注入，单测用替身。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    conn = conn_factory(url, pp_client.PP_TIMEOUT_S)
    await conn.connect(benchmark=pp_client.PP_BENCHMARK)
    benches: dict[str, Any] = {}
    summary = {"episodes": 0, "errors": 0, "aborted": False}
    try:
        for row in rows:
            task = row["task"]
            bench = benches.get(task)
            if bench is None:
                for b in benches.values():
                    b.cleanup()
                benches.clear()
                bench = benches[task] = bench_cls(tasks=[task], action_space="joint_angle", max_steps=max_steps)
            sid = pp_client.fixed_sid(row, DATASET)
            ep_idx = int(row["source_episode"])
            ep_dir = out_dir / f"{row['key']}.a{attempt}"
            ep_dir.mkdir(parents=True, exist_ok=False)
            trace_path = ep_dir / "trace.jsonl"
            trace = TraceWriter(trace_path, route=pp_client.TRACE_SCHEMA_ROUTE_ORIG, max_steps=max_steps,
                                identity={"task": task, "tier": row.get("tier"), "seed": row.get("seed"),
                                          "source_episode": row.get("source_episode"),
                                          "builder_episode": row.get("builder_episode"), "key": row.get("key"),
                                          "dataset": DATASET, "sid": sid, "episode_idx": ep_idx, "side": "orig"})
            frames = RawFrameWriter(ep_dir / "frames") if write_frames else None
            capture = EpisodeCapture(trace, frames)
            bench.pp_capture = capture
            tconn = pp_client.TracedConnection(conn, trace)
            recorder = recorder_cls(sid, sid)
            vtask = {"name": task, "env_id": task, "episode_idx": ep_idx}
            status, error, infra, infra_reason, steps, reason = "error", None, False, None, None, None
            reconnect_needed = False
            t0 = time.perf_counter()
            try:
                res = await runner.run_episode(bench, vtask, tconn, max_steps=max_steps, recorder=recorder)
                steps = int(res.get("steps", capture.exec_steps)) if isinstance(res, dict) else capture.exec_steps
                status, error = pp_client.terminal_status(capture.done, capture.truncated, capture.last_info)
                reason = "env_done" if capture.done else "loop_exit"
            except ConnectionError as e:
                error, infra, infra_reason = f"ConnectionError: {e}"[:800], True, "pp_unreachable"
                summary["aborted"] = True
                reason = "exception:ConnectionError"
            except Exception as e:  # noqa: BLE001
                status, infra_reason, infra = pp_client.classify_exception(e)
                error = f"{type(e).__name__}: {e}"[:800]
                reason = f"exception:{type(e).__name__}"
                reconnect_needed = (isinstance(e, (TimeoutError, RuntimeError)) or pp_client.is_connection_closed(e))
            finally:
                bench.pp_capture = None
            wall = time.perf_counter() - t0
            trace.close(status=status, terminal_reason=reason, sid=sid, frames_sent=tconn.frames_sent)
            fsum = frames.close() if frames is not None else None
            row_out = dict(row)
            row_out.update(side="orig", policy="pp", dataset=DATASET, route=pp_client.TRACE_SCHEMA_ROUTE_ORIG,
                           attempt=int(attempt), status=status, task_success=status == "success",
                           exec_steps=capture.exec_steps, steps=steps, demo_frames=capture.demo_frames,
                           max_steps=int(max_steps), effective_max_steps=int(max_steps), sid=sid, eid=sid,
                           episode_idx=ep_idx, error=error, infra=bool(infra), infra_reason=infra_reason,
                           frames_sent=tconn.frames_sent, decisions=tconn.actions_received, ep_dir=str(ep_dir),
                           trace_path=str(trace_path), frames_dir=None if frames is None else str(frames.dir),
                           video_frames=None if fsum is None else {s: v["count"] for s, v in fsum["streams"].items()},
                           wall_s=round(wall, 3))
            append_result(results, row_out)
            summary["episodes"] += 1
            summary["errors"] += int(status == "error")
            print(f"EPISODE_DONE side=orig policy=pp key={row['key']} sid={sid} status={status} "
                  f"exec_steps={capture.exec_steps} demo_frames={capture.demo_frames} infra={bool(infra)} "
                  f"wall_s={wall:.1f}", flush=True)
            if import_check is not None:
                import_check()
            if summary["aborted"]:
                break
            if reconnect_needed:
                try:
                    await conn.reconnect()
                except Exception as e:  # noqa: BLE001
                    print(f"PP_ORIG_RECONNECT_FAILED error={type(e).__name__}: {e}", flush=True)
                    summary["aborted"] = True
                    break
    finally:
        for b in benches.values():
            try:
                b.cleanup()
            except Exception:  # noqa: BLE001
                pass
        try:
            await conn.close()
        except Exception:  # noqa: BLE001
            pass
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PonderPounce 原侧驱动（只用官方 robomme + vla-eval SyncEpisodeRunner）")
    ap.add_argument("--check-imports", action="store_true", help="只做导入断言")
    ap.add_argument("--shard", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--only", default=None, help="逗号分隔的 key，只跑这些")
    ap.add_argument("--no-frames", action="store_true", help="不写原始帧")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        robomme_path = assert_official_only()
    except AssertionError as e:
        print(f"OFFICIAL_IMPORTS=FAIL reason={e}", flush=True)
        return EXIT_BAD_INPUT
    if args.check_imports:
        print(f"OFFICIAL_IMPORTS=PASS robomme={robomme_path} robomme_hard_imported=0", flush=True)
        return 0
    if args.shard is None or args.out is None or args.port is None:
        print("用法错误：需要 --shard、--out、--port", flush=True)
        return EXIT_BAD_INPUT
    try:
        rows = load_shard(args.shard, args.only)
    except ValueError as e:
        print(f"PP_ORIG_BLOCKED reason=shard detail={e}", flush=True)
        return EXIT_BAD_INPUT

    import anyio
    from vla_eval.benchmarks.robomme.benchmark import RoboMMEBenchmark
    from vla_eval.recording import NullEpisodeRecorder
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    RoboMMEBenchmark.configure_render("gpu")
    url = f"ws://{args.host}:{args.port}"
    print(f"PP_ORIG_START shard={args.shard} episodes={len(rows)} url={url} max_steps={args.max_steps} "
          f"robomme={robomme_path}", flush=True)
    summary = anyio.run(lambda: run_shard(
        rows, out_dir=args.out, url=url, bench_cls=make_tracing_bench_class(RoboMMEBenchmark),
        runner=SyncEpisodeRunner(), conn_factory=pp_client.default_connection_factory,
        recorder_cls=make_fixed_sid_recorder_class(NullEpisodeRecorder), max_steps=args.max_steps,
        attempt=args.attempt, write_frames=not args.no_frames, import_check=assert_official_only))
    print(f"PP_ORIG_DONE shard={args.shard} episodes={summary['episodes']} errors={summary['errors']} "
          f"aborted={int(summary['aborted'])}", flush=True)
    return EXIT_UNREACHABLE if summary["aborted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
