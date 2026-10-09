"""C13-SG-RESOURCE-PROBE：预检资源探针。``/proc`` 与 ``nvidia-smi`` 一律用替身（资源守卫拦真实 GPU）。期望值手写。"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from tests._support.dev_loaders import load_script


def R():
    return load_script("eval-official/resource_probe.py")


def _proc(root: Path, pid: int, ppid: int, rss_kb: int, hwm_kb: int, ticks: int = 0, state: str = "S",
          comm: str = "py (x) y") -> None:
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    # stat：pid (comm) state ppid ... 第 14、15 字段为 utime、stime；comm 故意含空格与括号
    fields = [state, str(ppid)] + ["0"] * 9 + [str(ticks), "0"] + ["0"] * 10
    (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
    (d / "status").write_text(f"Name:\tpy\nVmHWM:\t{hwm_kb} kB\nVmRSS:\t{rss_kb} kB\n")


def _fake_proc(tmp_path: Path) -> Path:
    root = tmp_path / "proc"
    _proc(root, 1, 0, 10, 10)
    _proc(root, 100, 1, 1024 * 1024, 2 * 1024 * 1024, ticks=100)  # 服务：1 GiB，HWM 2 GiB
    _proc(root, 101, 100, 512 * 1024, 512 * 1024, ticks=50)       # 服务子进程
    _proc(root, 200, 1, 256 * 1024, 256 * 1024)                    # 客户端
    _proc(root, 300, 1, 9 * 1024 * 1024, 9 * 1024 * 1024)          # 无关进程，不得计入
    (root / "self").mkdir()
    return root


def test_tree_mem_vram_and_peaks(tmp_path):
    r = R()
    fs = r.ProcFS(_fake_proc(tmp_path))
    assert fs.tree(100) == {100, 101} and fs.tree(200) == {200} and fs.tree(999) == set()
    assert fs.mem_kb(100) == (1024 * 1024, 2 * 1024 * 1024) and fs.cpu_ticks(100) == 100
    g = r.GpuStream(["unused"], clock=lambda: 0.0)
    for line in ("100, 30000", "101, 2000", "300, 40000", "garbage", "No running processes found"):
        g.feed(line, now=10.0)
    g.feed("200, 1000", now=7.0)  # 过期（窗口 1.5 s 外），不计
    sm = r.Sampler(fs, {"server": 100, "client": 200}, [g], window=1.5)
    rec = sm.sample(10.0)
    assert rec["vram_mib"] == 32000 and rec["rss_kb"] == 1536 * 1024 + 256 * 1024
    assert rec["labels"]["server"] == {"pids": 2, "rss_kb": 1536 * 1024, "vram_mib": 32000}
    # 第二次采样：服务 RSS 降下来、子进程消失；峰值保持
    _proc(fs.root, 100, 1, 100 * 1024, 2 * 1024 * 1024, ticks=300)
    import shutil

    shutil.rmtree(fs.root / "101")
    sm.sample(12.0)
    res = r.summarize(sm, route="groundsg-oracle-new", wall_s=4.0, episodes=2, rc=None, ok=True)
    assert res["rss_peak_gb"] == 1.75 and res["vram_peak_mb"] == 32000
    assert res["hwm_sum_gb"] == round((2 * 1024 * 1024 + 512 * 1024 + 256 * 1024) / 1024 / 1024, 3)
    assert res["episode_s"] == 2.0 and res["samples"] == 2 and res["rc"] == "na"
    assert res["cpu_cores_used"] == round(200 / r.CLK_TCK / 4.0, 2)  # 服务 ticks 100→300
    line = r.verdict_line(res)
    assert line.startswith("PREFLIGHT=PASS route=groundsg-oracle-new rss_peak_gb=1.75 vram_peak_mb=32000 episode_s=2.0 "
                           "cpu_cores_used=")
    assert "server_vram_peak_mb=32000" in line and "client_rss_peak_gb=0.25" in line


def test_gpu_cmd_is_single_persistent_query():
    cmd = R().gpu_cmd("nvidia-smi", "1", 1.0)
    assert cmd == ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", "1",
                   "-lms", "1000"]


def test_attach_mode_ends_when_main_exits(tmp_path, capsys):
    root = _fake_proc(tmp_path)
    fake = tmp_path / "fake_smi.py"
    fake.write_text("import sys, time\nfor _ in range(200):\n    print('100, 1234', flush=True)\n    time.sleep(0.02)\n")

    def _kill():
        time.sleep(0.4)
        import shutil

        shutil.rmtree(root / "200")

    th = threading.Thread(target=_kill)
    th.start()
    jo = tmp_path / "o" / "probe.json"
    rc = R().main(["--route", "pp-new", "--pid", "client=200", "--pid", "server=100", "--gpu", "0",
                   "--interval", "0.05", "--proc-root", str(root), "--nvidia-cmd", f"{sys.executable} {fake}",
                   "--json-out", str(jo)])
    th.join()
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and line.startswith("PREFLIGHT=PASS route=pp-new ")
    assert "vram_peak_mb=1234" in line and "server_vram_peak_mb=1234" in line and "client_vram_peak_mb=0" in line
    assert jo.exists()


def test_managed_command_rc(tmp_path, capsys):
    r = R()
    rc = r.main(["--route", "cmd-ok", "--interval", "0.05", "--", sys.executable, "-c", "import time; time.sleep(0.2)"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 0 and line.startswith("PREFLIGHT=PASS route=cmd-ok ") and " rc=0" in line
    rc = r.main(["--route", "cmd-bad", "--interval", "0.05", "--", sys.executable, "-c", "raise SystemExit(3)"])
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert rc == 1 and line.startswith("PREFLIGHT=FAIL route=cmd-bad ") and " rc=3" in line
