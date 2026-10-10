"""C12 ``_rollout.run_batch``：一批身份交给 runner 子进程、读回逐条结果。``subprocess.run`` 由 FakeRunner 冒充
（三种输出：写 ``results.json``／只写 partial／什么都不写），其余（jobs 落盘、命令行、环境变量、结果解析、
回注绑定计数、h5 事实、局目录挪开留证）全部是真实代码。"""
from __future__ import annotations


def test_runner_specs_are_bound_to_selected_namespace(tmp_path, monkeypatch):
    from tests._support.dev_loaders import load_script
    runner = load_script("parity/train_split_runner.py", fresh=True)
    for package in ("robomme_hard", "robomme_ood"):
        path = tmp_path / "src" / package / "env_record_wrapper/hard_specs.py"
        path.parent.mkdir(parents=True)
        path.write_text(f"ALL_TASKS = ({package!r},)\n")
    for package in ("robomme_hard", "robomme_ood"):
        monkeypatch.delenv("ROBOMME_SPECS_PACKAGE", raising=False)
        monkeypatch.setenv("ROBOMME_ENV_PACKAGE", package)
        assert runner._all_tasks(tmp_path) == (package,)
    monkeypatch.setenv("ROBOMME_SPECS_PACKAGE", "robomme_hard")
    monkeypatch.setenv("ROBOMME_ENV_PACKAGE", "robomme_hard")
    (tmp_path / "src/robomme_hard/env_record_wrapper/hard_specs.py").unlink()
    import pytest
    with pytest.raises(FileNotFoundError):
        runner._all_tasks(tmp_path)

import hashlib
import json
from pathlib import Path

import pytest

from gen_world import R, FakeRunner, freeze_file, install_run

S = "StopCube"


@pytest.fixture
def batch(tmp_path):
    header, rows = freeze_file(tmp_path / "frozen" / "xhard1" / "specs.jsonl", "xhard1", {S: (2, 3), "SwingXtimes": (1, 1)})
    picked = [dict(r) for r in rows if r["selected"]]
    picked[1]["_role"] = "backfill"
    return header, picked


def _run(batch, out, **kw):
    header, rows = batch
    return R.run_batch(rows, header, out, 0, src_root=Path("/src-root"), workers=3, gpu="1", pkg="robomme_ood", **kw)


def test_results_json_mode(monkeypatch, batch, tmp_path):
    fake = FakeRunner({(S, 1): ["fail"]}).install(monkeypatch)
    out = tmp_path / "out"
    recs = _run(batch, out, runner_env={"PYTHONPATH": "/leak", "KEEP": "1"})
    header, rows = batch
    work = out / "_rounds" / "xhard1_round_00"
    argv = fake.calls[0]
    for flag, value in (("--jobs-json", work / "jobs.json"), ("--results-json", work / "results.json"),
                        ("--workers", "3"), ("--gpu", "1"), ("--identity-source", "formula"), ("--src-root", "/src-root")):
        assert argv[argv.index(flag) + 1] == str(value)
    assert "--no-recovery" in argv and "--resume" not in argv
    env = fake.envs[0]
    assert env["ROBOMME_ENV_PACKAGE"] == "robomme_ood" and env["KEEP"] == "1" and "PYTHONPATH" not in env
    assert env["OMP_NUM_THREADS"] == "1" and env["PYTHONUNBUFFERED"] == "1"
    jobs = json.loads((work / "jobs.json").read_text())
    assert [(j["task"], j["episode"], j["seed"], j["difficulty"]) for j in jobs] == \
        [(r["task"], r["episode"], r["seed"], "xhard1") for r in rows]
    assert all(j["seed_rule"] == header["seed_rule"] for j in jobs)
    assert set(json.loads((work / "sampling.json").read_text())["tasks"]) == {S, "SwingXtimes"}
    assert set(json.loads((work / "specs.json").read_text())["specs"]) == {f"{r['task']}/{r['episode']}" for r in rows}
    by = {(r["task"], r["candidate"]): r for r in recs}
    ok = by[(S, 0)]
    h5 = Path(ok["h5"])
    assert ok["ok"] and ok["h5_sha256"] == hashlib.sha256(h5.read_bytes()).hexdigest()
    assert (ok["frames"], ok["exec_steps"], ok["bytes"]) == (5, 4, h5.stat().st_size)
    assert ok["spec_binding"] == {"mismatch": 0, "unattributed_mismatch": 0, "unused": 0, "value_points": 3,
                                  "layout_hit": 1, "layout_drift": 0, "layout_overridden": 0}
    assert ok["env_module"] == f"robomme_ood.robomme_env.{S}" and ok["role"] == "selected"
    bad = by[(S, 1)]
    assert (bad["ok"], bad["error_type"], bad["h5"], bad["role"]) == (False, "TaskFailed", None, "backfill")
    assert "h5_sha256" not in bad
    assert (work / "runner.log").read_text().startswith("RUNNER_DONE")


def test_partial_only_mode_is_read(monkeypatch, batch, tmp_path):
    FakeRunner(output="partial").install(monkeypatch)
    recs = _run(batch, tmp_path / "out")
    assert [r["ok"] for r in recs] == [True, True, True]
    assert not (tmp_path / "out" / "_rounds" / "xhard1_round_00" / "results.json").exists()


def test_nothing_written_is_runner_crash(monkeypatch, batch, tmp_path):
    FakeRunner(output="nothing").install(monkeypatch)
    recs = _run(batch, tmp_path / "out")
    assert [(r["ok"], r["error_type"]) for r in recs] == [(False, "RunnerCrash")] * 3
    assert all("runner exit=0" in r["error"] for r in recs)
    assert all(R.is_infra(r) for r in recs)


def test_binding_counts_unattributed_mismatches(monkeypatch, batch, tmp_path):
    fake = FakeRunner()

    def run(argv, **kw):
        result = fake.run(argv, **kw)
        for job in json.loads(Path(argv[argv.index("--jobs-json") + 1]).read_text()):
            (Path(job["worker_dir"]) / "spec_replay.json").write_text(json.dumps({
                "mismatches": [{"decision_key": "n"}, {"path": "x"}, {"decision_key": None}],
                "unused": ["a", "b"], "value_points": 7, "layout_drift": 2}))
        return result

    install_run(monkeypatch, run)
    rec = _run(batch, tmp_path / "out")[0]
    assert rec["spec_binding"] == {"mismatch": 3, "unattributed_mismatch": 2, "unused": 2, "value_points": 7,
                                   "layout_hit": 0, "layout_drift": 2, "layout_overridden": 0}


def test_existing_episode_dir_moved_aside_not_deleted(monkeypatch, batch, tmp_path):
    out = tmp_path / "out"
    stale = out / "episodes" / "xhard1" / f"{S}_episode_0"
    stale.mkdir(parents=True)
    (stale / "evidence.txt").write_text("上一次中断的残留")
    FakeRunner().install(monkeypatch)
    _run(batch, out)
    asides = list(stale.parent.glob(f"{S}_episode_0.aside*"))
    assert len(asides) == 1 and (asides[0] / "evidence.txt").read_text() == "上一次中断的残留"
    assert list((stale / "hdf5_files").glob("*.h5"))


def test_resume_keeps_done_dirs_and_passes_flag(monkeypatch, batch, tmp_path):
    out = tmp_path / "out"
    FakeRunner(crash_on_call=1, crash_after_jobs=1).install(monkeypatch)
    from gen_world import Crash
    with pytest.raises(Crash):
        _run(batch, out)
    fake = FakeRunner().install(monkeypatch)
    recs = _run(batch, out, resume=True)
    assert "--resume" in fake.calls[0]
    assert [e[:2] for e in fake.executions] == [(S, 1), ("SwingXtimes", 0)]
    assert [r["ok"] for r in recs] == [True, True, True]
    assert not list((out / "episodes" / "xhard1").glob("*.aside*"))  # 已完成局目录不挪开


@pytest.mark.parametrize("mutate, needle", [
    (lambda rows, header: rows[0].__setitem__("tier", "xhard2"), "同一档"),
    (lambda rows, header: header.__setitem__("difficulty", "xhard2"), "同一档"),
])
def test_batch_must_be_single_tier_matching_header(monkeypatch, batch, tmp_path, mutate, needle):
    header, rows = dict(batch[0]), [dict(r) for r in batch[1]]
    mutate(rows, header)
    fake = FakeRunner().install(monkeypatch)
    with pytest.raises(R.RolloutError, match=needle):
        R.run_batch(rows, header, tmp_path / "out", 0, src_root=tmp_path, workers=1, gpu="0", pkg="robomme_ood")
    assert fake.calls == []
