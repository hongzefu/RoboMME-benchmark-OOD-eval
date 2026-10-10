"""客户端回放等价核对 ``dev-scripts/checks/client_replay_eq.py`` 的自测：同一检出两侧比较必须 PASS，三类故意改动必须
被抓到；检出 sha 不符、两侧同样崩溃、零事件、缺用例一律 FAIL；两侧接口（拆仓后的包布局／旧仓官方名／改名前）按文件判定。

子进程只跑 smvla、perceptual-framesamp-modul 两条路线（各三个「常驻两局」用例、确定性假环境与假服务，纯 CPU、无网络、
无真实仿真）与 Astra 零外联探针；groundsg-oracle、pp 路线由主会话对真实 base（旧仓）／candidate（评估仓）检出跑全量。
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from tests._support.dev_loaders import REPO, load_script

SCRIPT = REPO / "dev-scripts" / "checks" / "client_replay_eq.py"


def _bench_src() -> str:
    """benchmark 子模块的 src（worktree 里子模块为空时取已安装 robomme_ood 包所在的 src）。"""
    import importlib.util

    own = REPO / "third_party" / "robomme_benchmark" / "src"
    if (own / "robomme_ood").is_dir():
        return str(own)
    spec = importlib.util.find_spec("robomme_ood")
    return str(next(iter(spec.submodule_search_locations)).rsplit("/robomme_ood", 1)[0])


@pytest.fixture(scope="module")
def cre():
    return load_script("eval-official/client_replay_eq.py")


def _head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def _run(*extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *extra], capture_output=True, text=True, timeout=600)


def test_same_checkout_passes_and_tampers_are_caught():
    sha = _head()
    p = _run("--base", str(REPO), "--base-sha", sha, "--candidate", str(REPO), "--candidate-sha", sha,
             "--routes", "smvla,perceptual-framesamp-modul,astra", "--bench-src", _bench_src())  # 自检强制执行
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("CLIENT_REPLAY_")]
    rt = [ln for ln in lines if ln.startswith("CLIENT_REPLAY_ROUTE=")]
    st = [ln for ln in lines if ln.startswith("CLIENT_REPLAY_SELFTEST=")]
    assert len(rt) == 3 and all(ln.startswith("CLIENT_REPLAY_ROUTE=PASS") for ln in rt), p.stdout + p.stderr
    assert all(f"base={sha} candidate={sha}" in ln for ln in rt)
    assert all("request_diff=0 action_diff=0 control_diff=0 terminal_diff=0" in ln for ln in rt)
    assert [ln.split("cases=")[1].split()[0] for ln in rt] == ["3", "3", "1"]
    assert len(st) == 3 and all("action=caught request=caught order=caught" in ln for ln in st), p.stdout
    assert lines[-1] == (f"CLIENT_REPLAY_EQ=PASS routes=3 cases=7 tamper_detected=1 base={sha} candidate={sha} "
                         f"route_pass=3 selftest_pass=3"), p.stdout
    assert "CLIENT_REPLAY_IFACE route=smvla base=pkg candidate=pkg" in p.stdout
    assert p.returncode == 0


def test_sha_mismatch_fails_without_running():
    sha = _head()
    wrong = "0" * 40
    p = _run("--base", str(REPO), "--base-sha", wrong, "--candidate", str(REPO), "--candidate-sha", sha,
             "--routes", "smvla")
    assert p.returncode == 1, p.stdout + p.stderr
    assert f"CLIENT_REPLAY_SHA=FAIL side=base want={wrong} got={sha}" in p.stdout
    assert "CLIENT_REPLAY_ROUTE=" not in p.stdout
    last = p.stdout.strip().splitlines()[-1]
    assert last.startswith("CLIENT_REPLAY_EQ=FAIL routes=1 cases=0 tamper_detected=0") and "reason=sha_mismatch" in last


def test_sha_args_are_required():
    p = _run("--base", str(REPO), "--candidate", str(REPO))
    assert p.returncode == 2 and "--base-sha" in p.stderr


def test_check_checkouts_rejects_short_sha_and_missing_dir(cre, tmp_path):
    sha = _head()
    assert cre.check_checkouts([("base", REPO, sha)]) == []
    assert cre.check_checkouts([("base", REPO, sha[:12])])  # 短 sha 不认
    assert cre.check_checkouts([("base", tmp_path / "nope", sha)])


def _side(runs: dict) -> dict:
    return {"runs": runs, "iface": "official"}


def _ok_run():
    return {"events": [["env", "reset"], ["srv", "infer", "h"], ["env", "step", "<f4", [8], "a"]],
            "terminal": {"status": "success", "task_success": True, "steps": 1}}


SCENARIO_NAMES = ("success", "env_error", "timeout")


def test_scenario_names_match_tool(cre):
    """用例名与单局场景名同名；每个用例常驻两局：第一局为同名场景、第二局一律成功（异常／超时后下一局）。"""
    assert tuple(sc["name"] for sc in cre.SCENARIOS) == SCENARIO_NAMES
    assert cre.case_names() == SCENARIO_NAMES
    assert [scs for _, scs in cre.CASES] == [(n, "success") if n != "success" else ("success", "success")
                                              for n in SCENARIO_NAMES]
    assert cre.case_names("astra") == ("probe",) and "astra" in cre.ROUTES


def test_compare_clean_baseline(cre):
    runs = {n: _ok_run() for n in SCENARIO_NAMES}
    d = cre.compare(_side(runs), _side({n: _ok_run() for n in SCENARIO_NAMES}))
    assert cre.is_clean(d), d


def test_compare_same_crash_on_both_sides_fails(cre):
    a = {n: _ok_run() for n in SCENARIO_NAMES}
    b = {n: _ok_run() for n in SCENARIO_NAMES}
    a["timeout"] = b["timeout"] = {"crash": "RuntimeError: boom"}
    d = cre.compare(_side(a), _side(b))
    assert d["control_diff"] == 1 and "两侧相同" in d["notes"][0], d


def test_compare_zero_events_fails(cre):
    empty = {"events": [["env", "reset"]], "terminal": {"status": "error", "task_success": False, "steps": 0}}
    a = {n: _ok_run() for n in SCENARIO_NAMES}
    b = {n: _ok_run() for n in SCENARIO_NAMES}
    a["success"] = dict(empty)
    b["success"] = dict(empty)
    d = cre.compare(_side(a), _side(b))
    assert d["control_diff"] == 1 and "零事件" in d["notes"][0], d


def test_compare_missing_scenario_fails(cre):
    a = {n: _ok_run() for n in SCENARIO_NAMES}
    b = {n: _ok_run() for n in SCENARIO_NAMES if n != "env_error"}
    d = cre.compare(_side(a), _side(b))
    assert d["control_diff"] == 1 and "场景缺失" in d["notes"][0], d


def test_side_interface_pkg_official_and_legacy(cre, tmp_path):
    """评估仓检出判 pkg（包布局）；旧仓官方名检出判 official；只有改名前模块名的旧检出判 legacy，且模块名／配置键／
    数据集名都取自别名表。"""
    defs = load_script("eval-official/official_defs.py")
    assert cre.side_interface(REPO)["name"] == "pkg" and cre.side_interface(REPO)["layout"] == "pkg"
    off = cre._old_dir(tmp_path / "off")
    off.mkdir(parents=True)
    for name in ("framesamp_modul_client", "groundsg_client"):
        (off / f"{name}.py").write_text("", encoding="utf-8")
    assert cre.side_interface(tmp_path / "off")["name"] == "official"
    d = cre._old_dir(tmp_path / "old")
    d.mkdir(parents=True)
    for old in defs.LEGACY_MODULE_ALIASES:
        (d / f"{old}.py").write_text("", encoding="utf-8")
    iface = cre.side_interface(tmp_path / "old")
    assert iface["name"] == "legacy"
    assert {iface["fsm_module"], iface["gsg_module"]} == set(defs.LEGACY_MODULE_ALIASES)
    assert iface["variant_key"] in defs.LEGACY_CONFIG_KEY_ALIASES
    assert defs.LEGACY_DATASET_ALIASES[iface["dataset"]] == defs.DATASET_HARD_VERIFY
    with pytest.raises(ValueError):
        cre.side_interface(tmp_path / "empty")


def test_workers_do_not_write_bytecode_into_checkouts(cre, monkeypatch, tmp_path):
    """被比较的检出只读：子进程环境带 PYTHONDONTWRITEBYTECODE=1、去掉 PYTHONPATH。"""
    seen = {}

    def fake_run(cmd, env=None, **kw):
        seen["env"] = env
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(cre.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    out = cre._run_side(tmp_path, "smvla", tmp_path, None, sys.executable)
    assert "worker_failed" in out
    assert seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1" and "PYTHONPATH" not in seen["env"]


def test_tool_official_defs_does_not_shadow_checkout_module(cre, monkeypatch):
    """回放工具自用的 official_defs 不占 ``sys.modules["official_defs"]``：被比较检出里的客户端按 ``load_sibling``
    「已加载则复用」取 official_defs，占了 base 侧就会用上本工具（candidate）的版本（MERGE-1 实测：第三阶段
    make_args 必填 model_seed，base 侧 groundsg-oracle 三个场景全部崩溃）。"""
    monkeypatch.delitem(sys.modules, "official_defs", raising=False)
    monkeypatch.delitem(sys.modules, cre._TOOL_DEFS, raising=False)
    mod = cre.official_defs()
    assert mod.__name__ == cre._TOOL_DEFS and "official_defs" not in sys.modules
    assert cre.side_interface(REPO)["name"] == "pkg" and "official_defs" not in sys.modules
