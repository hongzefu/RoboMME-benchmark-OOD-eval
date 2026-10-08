"""预检资源探针（计划第二部分 1.7、第四节）：一个常驻进程，每秒采样给定进程树的内存与显存，进程退出后出一行判定。

两种用法：

- 附着：``--pid server=<PID> --pid client=<PID>``（可多个，标签任意），``--main client`` 指定以哪个进程退出为准
  （默认第一个）；退出码不可知，只要主进程退出且至少采到一次样即 PASS。
- 托管：``-- <命令...>``（放在参数末尾），由本探针起子进程并以它为主进程；子进程退出码非 0 判 FAIL。
  托管与附着可同时用（附着的进程另作标签，如模型服务）。

采样（节奏 ``--interval``，默认 1 s）：

- 每个标签的进程树（自身及全部子孙，按 ``/proc/*/stat`` 的 ppid 建树）逐进程读 ``/proc/<pid>/status`` 的
  ``VmRSS``、``VmHWM``，读 ``/proc/<pid>/stat`` 的 ``utime+stime``；
- 显存：一个持久的 ``nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i <卡>
  -lms <间隔毫秒>`` 子进程（每张卡一个，不做高频全卡查询），读线程逐行解析；某进程的显存取最近
  ``1.5 × 间隔`` 内见到的值，只计入属于被测进程树的 PID。

输出（``rss`` 与 ``vram`` 都是「同一时刻各进程之和」的峰值；``hwm_sum_gb`` 为见过的每个进程 ``VmHWM`` 之和，
是内存峰值的上界）::

    PREFLIGHT=PASS|FAIL route=<路线> rss_peak_gb=<x> vram_peak_mb=<x> episode_s=<x> cpu_cores_used=<x>
        hwm_sum_gb=<x> cpu_cores_peak=<x> samples=<n> episodes=<n> rc=<退出码|na>
        [<标签>_rss_peak_gb=<x> <标签>_vram_peak_mb=<x> ...]

``episode_s`` = 主进程存活墙钟 / ``--episodes``（默认 1）；``cpu_cores_used`` = 全部被测进程累计 CPU 秒 / 墙钟。
``--json-out`` 另写逐秒样本与汇总。测试用 ``--proc-root`` 指向假 ``/proc``、``--nvidia-cmd`` 换成替身命令。
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# ── /proc 读取 ───────────────────────────────────────────────────────────────


class ProcFS:
    def __init__(self, root: str | Path = "/proc"):
        self.root = Path(root)

    def alive(self, pid: int) -> bool:
        st = self.root / str(pid) / "stat"
        if not st.exists():
            return False
        try:
            fields = self._stat_fields(pid)
        except (OSError, ValueError):
            return False
        return fields is not None and fields[0] != "Z"

    def _stat_fields(self, pid: int) -> list[str] | None:
        text = (self.root / str(pid) / "stat").read_text()
        # comm 可能含空格与括号：取最后一个 ')' 之后的字段（第 3 个字段 state 起）
        rest = text[text.rindex(")") + 2:].split()
        return rest

    def ppid_map(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for d in self.root.iterdir():
            if not d.name.isdigit():
                continue
            try:
                f = self._stat_fields(int(d.name))
            except (OSError, ValueError):
                continue
            if f:
                out[int(d.name)] = int(f[1])
        return out

    def tree(self, root_pid: int, ppids: dict[int, int] | None = None) -> set[int]:
        ppids = self.ppid_map() if ppids is None else ppids
        kids: dict[int, list[int]] = defaultdict(list)
        for p, pp in ppids.items():
            kids[pp].append(p)
        out, stack = set(), [root_pid]
        while stack:
            p = stack.pop()
            if p in out or p not in ppids:
                continue
            out.add(p)
            stack.extend(kids.get(p, []))
        return out

    def mem_kb(self, pid: int) -> tuple[int, int]:
        """(VmRSS, VmHWM)，单位 kB；读不到返回 (0, 0)。"""
        rss = hwm = 0
        try:
            for line in (self.root / str(pid) / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
        return rss, hwm

    def cpu_ticks(self, pid: int) -> int:
        try:
            f = self._stat_fields(pid)
            return int(f[11]) + int(f[12])  # utime、stime（stat 第 14、15 字段）
        except (OSError, ValueError, IndexError, TypeError):
            return 0


# ── 显存流 ───────────────────────────────────────────────────────────────────


class GpuStream:
    """持久 nvidia-smi 子进程的读线程；``latest(now, window)`` 返回窗口内各 PID 的最近显存（MiB）。"""

    def __init__(self, cmd: list[str], clock: Callable[[], float] = time.monotonic):
        self.cmd = cmd
        self.clock = clock
        self.seen: dict[int, tuple[float, int]] = {}
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.lines = 0

    def feed(self, line: str, now: float | None = None) -> None:
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            return
        try:
            mib = int(float(parts[1].split()[0]))
        except (ValueError, IndexError):
            return
        with self.lock:
            self.seen[int(parts[0])] = (self.clock() if now is None else now, mib)
            self.lines += 1

    def latest(self, now: float, window: float) -> dict[int, int]:
        with self.lock:
            return {p: v for p, (t, v) in self.seen.items() if now - t <= window}

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

        def _reader():
            assert self.proc is not None and self.proc.stdout is not None
            for line in self.proc.stdout:
                self.feed(line)

        self.thread = threading.Thread(target=_reader, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# ── 采样器 ───────────────────────────────────────────────────────────────────


class Sampler:
    def __init__(self, procfs: ProcFS, roots: dict[str, int], gpus: Iterable[GpuStream] = (), *, window: float = 1.5):
        self.fs = procfs
        self.roots = dict(roots)
        self.gpus = list(gpus)
        self.window = window
        self.samples: list[dict] = []
        self.hwm_kb: dict[int, int] = {}
        self.ticks_first: dict[int, int] = {}
        self.ticks_last: dict[int, int] = {}
        self.rss_peak_kb = 0
        self.vram_peak_mib = 0
        self.label_rss_peak: dict[str, int] = defaultdict(int)
        self.label_vram_peak: dict[str, int] = defaultdict(int)
        self.cpu_peak = 0.0
        self._prev: tuple[float, int] | None = None
        self.t0: float | None = None

    def sample(self, now: float) -> dict:
        if self.t0 is None:
            self.t0 = now
        ppids = self.fs.ppid_map()
        vram_by_pid: dict[int, int] = {}
        for g in self.gpus:
            for p, v in g.latest(now, self.window).items():
                vram_by_pid[p] = vram_by_pid.get(p, 0) + v
        rec = {"t": round(now - self.t0, 3), "labels": {}}
        all_pids: set[int] = set()
        for label, root in self.roots.items():
            pids = self.fs.tree(root, ppids)
            rss = vram = 0
            for p in pids:
                r, h = self.fs.mem_kb(p)
                rss += r
                self.hwm_kb[p] = max(self.hwm_kb.get(p, 0), h)
                vram += vram_by_pid.get(p, 0)
                if p not in all_pids:
                    t = self.fs.cpu_ticks(p)
                    self.ticks_first.setdefault(p, t)
                    self.ticks_last[p] = max(self.ticks_last.get(p, 0), t)
            all_pids |= pids
            self.label_rss_peak[label] = max(self.label_rss_peak[label], rss)
            self.label_vram_peak[label] = max(self.label_vram_peak[label], vram)
            rec["labels"][label] = {"pids": len(pids), "rss_kb": rss, "vram_mib": vram}
        tot_rss = sum(self.fs.mem_kb(p)[0] for p in all_pids)
        tot_vram = sum(vram_by_pid.get(p, 0) for p in all_pids)
        self.rss_peak_kb = max(self.rss_peak_kb, tot_rss)
        self.vram_peak_mib = max(self.vram_peak_mib, tot_vram)
        ticks = self.total_ticks()
        if self._prev is not None and now > self._prev[0]:
            self.cpu_peak = max(self.cpu_peak, (ticks - self._prev[1]) / CLK_TCK / (now - self._prev[0]))
        self._prev = (now, ticks)
        rec.update(rss_kb=tot_rss, vram_mib=tot_vram, pids=len(all_pids))
        self.samples.append(rec)
        return rec

    def total_ticks(self) -> int:
        return sum(self.ticks_last[p] - self.ticks_first[p] for p in self.ticks_last)


def summarize(sm: Sampler, *, route: str, wall_s: float, episodes: int, rc: int | None, ok: bool) -> dict:
    cpu_s = sm.total_ticks() / CLK_TCK
    res = {
        "verdict": "PASS" if ok and sm.samples else "FAIL", "route": route,
        "rss_peak_gb": round(sm.rss_peak_kb / 1024 / 1024, 3), "vram_peak_mb": int(sm.vram_peak_mib),
        "episode_s": round(wall_s / max(1, episodes), 1), "cpu_cores_used": round(cpu_s / wall_s, 2) if wall_s > 0 else 0.0,
        "hwm_sum_gb": round(sum(sm.hwm_kb.values()) / 1024 / 1024, 3), "cpu_cores_peak": round(sm.cpu_peak, 2),
        "samples": len(sm.samples), "episodes": episodes, "rc": "na" if rc is None else rc, "wall_s": round(wall_s, 1),
        "labels": {k: {"rss_peak_gb": round(sm.label_rss_peak[k] / 1024 / 1024, 3), "vram_peak_mb": int(sm.label_vram_peak[k])}
                   for k in sm.roots},
    }
    return res


def verdict_line(res: dict) -> str:
    parts = [f"PREFLIGHT={res['verdict']}", f"route={res['route']}", f"rss_peak_gb={res['rss_peak_gb']}",
             f"vram_peak_mb={res['vram_peak_mb']}", f"episode_s={res['episode_s']}", f"cpu_cores_used={res['cpu_cores_used']}",
             f"hwm_sum_gb={res['hwm_sum_gb']}", f"cpu_cores_peak={res['cpu_cores_peak']}", f"samples={res['samples']}",
             f"episodes={res['episodes']}", f"rc={res['rc']}"]
    for k, v in res["labels"].items():
        parts += [f"{k}_rss_peak_gb={v['rss_peak_gb']}", f"{k}_vram_peak_mb={v['vram_peak_mb']}"]
    return " ".join(parts)


# ── 主流程 ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="预检资源探针（进程树内存、显存、耗时、CPU 核数）")
    ap.add_argument("--route", required=True)
    ap.add_argument("--pid", action="append", default=[], help="标签=PID，可多个")
    ap.add_argument("--main", default=None, help="以哪个标签的进程退出为准（托管命令时恒为 cmd）")
    ap.add_argument("--gpu", action="append", default=[], help="卡号，可多个；每张卡一个持久 nvidia-smi")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-wall", type=float, default=None, help="超过即停止采样并判 FAIL（不杀被测进程）")
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--nvidia-cmd", default="nvidia-smi", help="测试替身入口；其后自动拼查询参数")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 之后为托管命令")
    return ap


def gpu_cmd(base: str, gpu: str, interval: float) -> list[str]:
    return [*shlex.split(base), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
            "-i", str(gpu), "-lms", str(max(100, int(interval * 1000)))]


def run(args, *, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict:
    roots: dict[str, int] = {}
    for item in args.pid:
        label, _, pid = item.partition("=")
        roots[label] = int(pid)
    cmd = [c for c in args.cmd if c != "--"] if args.cmd else []
    child = None
    if cmd:
        child = subprocess.Popen(cmd)
        roots = {"cmd": child.pid, **roots}
        main_label = "cmd"
    else:
        if not roots:
            raise SystemExit("需要 --pid 或 -- <命令>")
        main_label = args.main or next(iter(roots))
    fs = ProcFS(args.proc_root)
    gpus = [GpuStream(gpu_cmd(args.nvidia_cmd, g, args.interval), clock) for g in args.gpu]
    for g in gpus:
        g.start()
    sm = Sampler(fs, roots, gpus, window=1.5 * args.interval)
    t0 = clock()
    timed_out = False
    try:
        while True:
            sm.sample(clock())
            if child is not None:
                if child.poll() is not None:
                    break
            elif not fs.alive(roots[main_label]):
                break
            if args.max_wall is not None and clock() - t0 > args.max_wall:
                timed_out = True
                break
            sleep(args.interval)
    finally:
        for g in gpus:
            g.stop()
    wall = clock() - t0
    rc = child.wait() if child is not None and not timed_out else None
    ok = not timed_out and (rc is None or rc == 0)
    res = summarize(sm, route=args.route, wall_s=wall, episodes=args.episodes, rc=rc, ok=ok)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({"summary": res, "samples": sm.samples}, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    res = run(args)
    print(verdict_line(res), flush=True)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
