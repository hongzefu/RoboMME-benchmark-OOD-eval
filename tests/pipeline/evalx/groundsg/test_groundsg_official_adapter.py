"""GroundSG 新侧适配与官方循环的逐项一致（1003 评估计划 1.3，子任务 S3；判定行 OFFICIAL_ADAPTER）。

两侧都跑官方 ``EpisodeEvaluator.eval_each_episode`` 原文：新侧经 ``groundsg_client.SessionRunner`` 委托到真实
``EnvSession``，原侧经 ``official_hard_runner`` 用官方 ``EnvRunner`` 原文 + 官方 builder 替身。两侧给同一组假环境
（下一帧由所执行动作决定）、各自一个同实现的假服务（动作块由本局请求指纹决定）、各自一份 swift 替身——任何
一个请求、动作或子目标不同都会向后传播。比较三类：请求（轨迹 request 行、假服务收到的指纹、Qwen 请求）、
执行（环境收到的动作字节、轨迹 step 行）、终态（status／步数／error 与轨迹 end 行）。

另有手写期望（不调用被测代码推出）：消息顺序、首批帧数与 exec_start_idx、提示词与 Oracle 子目标、执行动作 = 各
动作块前 16 行；``count > max_steps`` 才 timeout（max_steps=1300 时执行 1301 步、原侧帧数 3 + 1301）；``--strict-cap``
下第 max_steps+1 步不进环境；官方 ``unknown`` 记 error 且不中止后续局。

第二阶段 S1（1005 计划第二部分一节「S1」、八节 3.5）：新侧保留官方叠字视频到 ``<局目录>/official/`` 并写
``provenance.json``；新侧轨迹按共享契约收尾（C3 终态、C6 attempt、C8 三分计数与缺观测步）。两侧逐项比较前用
``legacy_trace_parts`` 把契约字段还原成旧口径——还原是可逆映射（官方原返回值 ``success_flag`` 还回
``terminal_reason``、缺观测步还回旧写法），不放过动作、画面、请求的任何差异。

第三阶段（1006 计划八.3、八.10 第 8 条；R2）：原侧也传 ``keep_official=True``、identity 补 ``attempt``／
``policy_seed``、end 补三分计数，两侧 end 行同一套字段；三个变体（含 MemER）两侧逐项相同。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import groundsg_fakes_dev as F
from tests.pipeline.evalx.report import trace_contract as TC

NEW_ONLY_END = F.NEW_ONLY_END


@pytest.fixture(autouse=True)
def _legacy_view(monkeypatch):
    monkeypatch.setattr(F, "trace_parts", F.legacy_trace_parts)


def _rewrite_end(path: str, **changes) -> None:
    rows = F.read_trace(path)
    rows[-1].update(changes)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                          encoding="utf-8")


def test_legacy_view_still_reports_status_and_action_diffs(tmp_path):
    """反例：还原只剥新增记录字段——新侧终态 status、步行动作被改动后 diffs 仍报出差异。"""
    n, o = run_pair(F.ORACLE, F.Plan(success_at=37), 60, tmp_path)
    assert F.diffs(n, o)["terminal"] == 0
    _rewrite_end(n[2]["trace_path"], status="fail")
    assert F.diffs(n, o)["terminal"] >= 1
    _rewrite_end(n[2]["trace_path"], status="success", success_flag="fail")  # 官方原值被改也报
    assert F.diffs(n, o)["terminal"] >= 1
    _rewrite_end(n[2]["trace_path"], success_flag="success")
    assert F.diffs(n, o)["terminal"] == 0
    rows = F.read_trace(n[2]["trace_path"])
    step = next(r for r in rows if r["kind"] == "step")
    step["action"] = dict(step["action"], sha256="0" * 64)
    Path(n[2]["trace_path"]).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                                for r in rows), encoding="utf-8")
    assert F.diffs(n, o)["exec"] >= 1

SCENARIOS = {
    "success": (F.Plan(success_at=37), 60),
    "fail": (F.Plan(fail_at=20), 60),
    "timeout": (F.Plan(), 40),
    "unknown": (F.Plan(unknown_at=10), 60),
    "env_raise": (F.Plan(raise_at=5), 60),
}


def run_pair(variant: str, plan: F.Plan, max_steps: int, tmp: Path, ident: dict | None = None):
    ident = ident or F.identity()
    wn, wo = F.World(default=plan), F.World(default=plan)
    new, orig = F.NewSide(variant, max_steps, tmp / "new", wn), F.OrigSide(variant, max_steps, tmp / "orig", wo)
    rn, ro = new.run(ident), orig.run(ident)
    return (new, wn, rn), (orig, wo, ro)


def test_official_adapter_matches_official_loop(tmp_path):
    F.print_official_sha()
    total = {"payload": 0, "exec": 0, "terminal": 0}
    seen = {}
    for variant in F.VARIANTS:
        for name, (plan, max_steps) in SCENARIOS.items():
            n, o = run_pair(variant, plan, max_steps, tmp_path / variant / name)
            d = F.diffs(n, o)
            assert d["n_req"] > 0 and d["n_exec"] > 0, (variant, name, d)
            for k in total:
                total[k] += d[k]
            seen[(variant, name)] = (n[2]["status"], n[2]["steps"], n[2]["error"])
            assert d["payload"] == d["exec"] == d["terminal"] == 0, (variant, name, d)
            # 还原只针对新侧确实写了的契约字段：新侧局目录整份过契约（含 error 局）
            assert TC.contract_problems(Path(n[2]["trace_path"]).parent) == [], (variant, name)
            assert "success_flag" in F.read_trace(n[2]["trace_path"])[-1]
    # 终态手写：每种情形两变体相同
    for variant in F.VARIANTS:
        assert seen[(variant, "success")][:2] == ("success", 37)
        assert seen[(variant, "fail")][:2] == ("fail", 20)
        assert seen[(variant, "timeout")][:2] == ("timeout", 41)  # count > 40 才 timeout：执行 41 步
        assert seen[(variant, "unknown")] == ("error", 10, "success_flag=unknown")
        st, steps, err = seen[(variant, "env_raise")]
        assert st == "error" and steps == 5 and err.startswith("AttributeError"), seen[(variant, "env_raise")]
    print(f"OFFICIAL_ADAPTER=PASS variants={len(F.VARIANTS)} payload_diff={total['payload']} "
          f"exec_diff={total['exec']} terminal_diff={total['terminal']}")


def test_comparison_is_not_trivial(tmp_path):
    """换一局（source_episode 不同）后两侧请求与动作必须不同：等式非平凡。"""
    (new, wn, rn), _ = run_pair(F.ORACLE, F.Plan(success_at=37), 60, tmp_path / "a")
    (new2, wn2, rn2), _ = run_pair(F.ORACLE, F.Plan(success_at=37), 60, tmp_path / "b",
                                   ident=F.identity(source_episode=7, builder_episode=1, seed=510700))
    assert [x[1] for x in new.server.log] != [x[1] for x in new2.server.log]
    assert wn.envs[0].actions[0].tobytes() != wn2.envs[0].actions[0].tobytes()


def test_oracle_messages_hand_expected(tmp_path):
    """手写期望：消息顺序、首批 reset 帧、提示词、Oracle 子目标、执行动作 = 每块前 16 行。"""
    (new, wn, rn), _ = run_pair(F.ORACLE, F.Plan(success_at=37), 60, tmp_path)
    kinds = [k for k, _, _ in new.server.log]
    assert kinds == ["reset", "add_buffer", "infer", "add_buffer", "infer", "add_buffer", "infer"]
    bufs = [m for k, _, m in new.server.log if k == "add_buffer"]
    assert (bufs[0]["n_frames"], bufs[0]["exec_start_idx"]) == (F.N_RESET_FRAMES, F.N_RESET_FRAMES - 1)
    assert [(b["n_frames"], b["exec_start_idx"]) for b in bufs[1:]] == [(16, 0), (16, 0)]
    infers = [m for k, _, m in new.server.log if k == "infer"]
    goal = F.goal_of("PickXtimes", 3)
    # 决策发生在第 0、16、32 步之后：Oracle 取当时 info 的 grounded_subgoal_online
    assert [m["prompt"] for m in infers] == [goal] * 3
    assert [m["grounded_subgoal"] for m in infers] == [F.subgoal_at(0), F.subgoal_at(16), F.subgoal_at(32)]
    assert [m["simple_subgoal"] for m in infers] == [m["grounded_subgoal"] for m in infers]
    want = np.concatenate([c[:F.EXEC_HORIZON] for c in new.server.chunks])[:37]
    got = np.stack(wn.envs[0].actions)
    assert got.shape == (37, 8) and got.tobytes() == want.tobytes()
    assert rn["decisions"] == 3 and rn["side"] == "new" and rn["demo_frames"] == F.N_RESET_FRAMES - 1


def test_mme_vla_loop_times_out_only_after_count_exceeds_1300(tmp_path):
    """官方 ``count > max_steps``：max_steps=1300 时两侧都真实执行 1301 步再记 timeout；原侧外围帧含第 1301 步。"""
    (new, wn, rn), (orig, wo, ro) = run_pair(F.ORACLE, F.Plan(), 1300, tmp_path)
    assert (rn["status"], rn["steps"], rn["_session_steps"]) == ("timeout", 1301, 1301)
    assert (ro["status"], ro["exec_steps"]) == ("timeout", 1301)
    assert len(wn.envs[0].actions) == len(wo.envs[0].actions) == 1301
    assert ro["video_frames"] == {"front": F.N_RESET_FRAMES + 1301, "wrist": F.N_RESET_FRAMES + 1301}
    fj = json.loads(Path(ro["frames_dir"], "frames.json").read_text())
    assert (fj["demo_frames"], fj["init_frames"], fj["exec_steps"], fj["missing_steps"]) == (2, 1, 1301, [])
    assert Path(ro["frames_dir"], "front.rgb24").stat().st_size == (F.N_RESET_FRAMES + 1301) * F.H * F.W * 3
    d = F.diffs((new, wn, rn), (orig, wo, ro))
    assert d["payload"] == d["exec"] == d["terminal"] == 0, d


def test_strict_cap_stops_before_env_and_reports_timeout(tmp_path):
    """``--strict-cap``（V9 约定）：第 max_steps+1 次 step 不进环境，适配入口收住 StepCapReached、记 timeout。"""
    world = F.World(default=F.Plan())
    side = F.NewSide(F.ORACLE, 30, tmp_path, world, strict_cap=True)
    res = side.run(F.identity())
    assert res["status"] == "timeout" and res["exception"] == "StepCapReached"
    assert res["_session_steps"] == 30 and res["_cap_hit"] is True and len(world.envs[0].actions) == 30
    end = F.read_trace(res["trace_path"])[-1]
    assert end["kind"] == "end" and end["status"] == "timeout" and end["exec_steps"] == 30


@pytest.mark.parametrize("variant", F.VARIANTS)
def test_unknown_is_error_and_does_not_stop_seat(tmp_path, variant):
    """官方 ``unknown`` → status=error、error=success_flag=unknown、非基础设施；同一上下文的下一局照常跑。
    Qwen／MemER 临时目录与官方叠字视频临时目录局末都不留，子目标模型日志归档到轨迹目录。"""
    world = F.World(plans={3: F.Plan(unknown_at=10), 7: F.Plan(success_at=20)})
    side = F.NewSide(variant, 60, tmp_path, world)
    r1 = side.run(F.identity())
    r2 = side.run(F.identity(source_episode=7, builder_episode=1, seed=510700))
    assert (r1["status"], r1["error"], r1["infra"], r1["steps"]) == ("error", "success_flag=unknown", False, 10)
    assert (r2["status"], r2["steps"]) == ("success", 20)
    # 官方 episode_id 用短编号 <source_episode>a<attempt>（12.453：长 episode_tag 使叠字视频文件名超 255 字节）
    for r, eid in ((r1, "3a1"), (r2, "7a1")):
        tdir = Path(r["trace_path"]).parent
        assert not (tdir / "qwen-tmp").exists() and not (tdir / "memer-tmp").exists()
        assert not (tdir / "official-video").exists()
        logs = sorted(p.name for p in tdir.glob("*_log.jsonl"))
        if variant == F.QWENVL:
            assert logs == [f"ep{eid}_QwenVL_log.jsonl"] and r["qwen_log"] == str(tdir / logs[0])
            assert all(json.loads(x)["messages"] for x in (tdir / logs[0]).read_text().splitlines())
        elif variant == F.MEMER:
            assert logs == [f"ep{eid}_MemER_log.jsonl"] and r["qwen_log"] == str(tdir / logs[0])
            lines = [json.loads(x) for x in (tdir / logs[0]).read_text().splitlines()]
            assert lines and all(("messages" in x) != ("response" in x) for x in lines)  # 请求、回复交替
            assert r["memer_compat_sha256"] == side.mc.official_defs.MEMER_COMPAT_SHA256
        else:
            assert logs == [] and r["qwen_log"] is None
    # 原侧：同一分片两行，unknown 不中止
    orig_world = F.World(plans={3: F.Plan(unknown_at=10), 7: F.Plan(success_at=20)})
    orig = F.OrigSide(variant, 60, tmp_path / "orig", orig_world)
    rows = [F.identity(), F.identity(source_episode=7, builder_episode=1, seed=510700)]
    summary = orig.ohr.run_shard(orig.ctx, rows, out=tmp_path / "orig")
    assert summary == {"episodes": 2, "errors": 1, "aborted": False, "budget_blocked": False, "retry_denied": 0,
                       "success": 1, "fail": 0, "timeout": 0}
    got = [json.loads(x) for x in (tmp_path / "orig" / "results.jsonl").read_text().splitlines()]
    assert [(g["status"], g["error"]) for g in got] == [("error", "success_flag=unknown"), ("success", None)]
    assert all(g["side"] == "orig" and g["policy"] == "groundsg" and g["policy_variant"] == variant
               and g["dataset"] == "hard-verify" and g["policy_seed"] == F.POLICY_SEED for g in got)


def test_trace_location_fallbacks(tmp_path):
    """轨迹位置：trace_dir → recorder.out_dir → 不写（此时临时区用后即删）。"""
    ec = F.env_session()
    world = F.World(default=F.Plan(success_at=5))
    side = F.NewSide(F.ORACLE, 60, tmp_path, world)
    ident = F.identity()

    class Rec(ec.NullRecorder):
        out_dir = tmp_path / "recdir"

    for rec, conn_extra, want in ((Rec(), {"trace_dir": None}, tmp_path / "recdir" / "trace.jsonl"),
                                  (ec.NullRecorder(), {"trace_dir": None}, None)):
        sess = ec.EnvSession(ident["task"], 0, max_steps=60, builder=F.NewSideBuilder(ident["task"], world, {0: 3}),
                             dataset=F.DATASET)
        sess.build()
        conn = {"max_steps": 60, "dataset": F.DATASET, "groundsg_variant": F.ORACLE, "episode_tag": "x.a1",
                "policy_context": side.ctx, **conn_extra}
        res = side.mc.run_episode(sess, ident, conn, rec)
        sess.close()
        assert res["status"] == "success"
        assert res["trace_path"] == (None if want is None else str(want))
        if want is not None:
            assert F.read_trace(want)[0]["kind"] == "header"


# ---------------------------------------------------------------- S1：新侧保留官方叠字视频


def _decoded_frames(path: Path) -> int:
    """独立于被测代码的解码计数（imageio 逐帧读完）。"""
    import imageio.v2 as iio

    with iio.get_reader(str(path), format="FFMPEG") as rd:
        return sum(1 for _ in rd)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _official(res: dict) -> tuple[Path, list[Path], dict]:
    tdir = Path(res["trace_path"]).parent
    odir = tdir / "official"
    mp4s = sorted(odir.glob("*.mp4"))
    pj = odir / "provenance.json"
    prov = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
    return tdir, mp4s, prov


def _check_kept(res: dict, *, source: str, label: str, frames: int) -> dict:
    tdir, mp4s, prov = _official(res)
    assert res["official_source"] == source and res["official_save_error"] is None, res
    assert len(mp4s) == 1 and res["official_videos"] == [str(mp4s[0])]
    # 官方文件名格式 {env_id}_ep{episode_id}_{flag}_{task_goal}_{difficulty}.mp4，短编号 3a1
    assert mp4s[0].name == f"PickXtimes_ep3a1_{label}_{F.goal_of('PickXtimes', 3)}_hard.mp4"
    assert "ep3a1_" in mp4s[0].name
    assert res["frames_recorded"] == frames == _decoded_frames(mp4s[0])
    # provenance：身份、路线、数据集、attempt、终态、官方源码 sha、视频 sha、帧数依据、来源
    ident = prov["identity"]
    assert (ident["key"], ident["attempt"], ident["dataset"], ident["source_episode"]) == \
        ("PickXtimes_xhard0_510300", 1, F.DATASET, 3)
    assert prov["route"] == f"groundsg/{F.ORACLE}/new" and prov["dataset"] == F.DATASET and prov["attempt"] == 1
    assert prov["terminal"]["status"] == res["status"] and prov["terminal"]["success_flag"] == res["success_flag"]
    assert prov["official_sha256"] == res["official_sha256"] and {"eval.py", "utils.py"} <= set(prov["official_sha256"])
    assert prov["video"] == {"name": mp4s[0].name, "sha256": _sha(mp4s[0]), "bytes": mp4s[0].stat().st_size}
    assert prov["frames"]["decoded"] == prov["frames"]["expected"] == prov["frames"]["frames_recorded"] == frames
    assert prov["frames"]["basis"] == "demo_frames + 1 + steps_observed - omitted_timeout_frames"
    assert prov["official_source"] == source
    # 轨迹 end 与结果行一致；video_dir 已删
    end = F.read_trace(res["trace_path"])[-1]
    assert (end["official_source"], end["official_videos"]) == (source, [mp4s[0].name])
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == \
        (res["steps_attempted"], res["steps_observed"], res["frames_recorded"])
    assert not (tdir / "official-video").exists()
    return end


def _side(tmp: Path, plan: F.Plan, max_steps: int = 60, **kw) -> tuple[F.NewSide, F.World]:
    world = F.World(default=plan)
    return F.NewSide(F.ORACLE, max_steps, tmp, world, **kw), world


def test_normal_episode_keeps_one_official_video_with_provenance(tmp_path, capsys):
    side, _ = _side(tmp_path, F.Plan(success_at=37))
    res = side.run(F.identity())
    assert res["status"] == "success"
    end = _check_kept(res, source="official", label="success", frames=F.N_RESET_FRAMES + 37)
    assert (end["status"], end["terminal_reason"], end["success_flag"]) == ("success", "success", "success")
    assert F.read_trace(res["trace_path"])[0]["identity"]["attempt"] == 1  # C6
    assert "OFFICIAL_VIDEO=KEPT source=official" in capsys.readouterr().out
    TC.assert_renderable(Path(res["trace_path"]).parent)
    TC.assert_counts_consistent(Path(res["trace_path"]).parent,
                                {"exec_steps": res["_session_steps"], "status": res["status"]})
    # evaluator 的 init_episode 包装已还原（实例上不留属性）
    assert "init_episode" not in vars(side.ctx["evaluator"])


def test_natural_timeout_omits_last_frame(tmp_path):
    """官方 count > max_steps 先 break 后不 record：执行 41 步、官方视频 3 + 40 帧、omitted_timeout_frames=1。"""
    side, _ = _side(tmp_path, F.Plan(), max_steps=40)
    res = side.run(F.identity())
    assert (res["status"], res["steps"]) == ("timeout", 41)
    end = _check_kept(res, source="official", label="timeout", frames=F.N_RESET_FRAMES + 40)
    assert (end["omitted_timeout_frames"], end["steps_attempted"], end["steps_observed"]) == (1, 41, 41)
    TC.assert_renderable(Path(res["trace_path"]).parent)


def test_strict_cap_salvages_official_video(tmp_path):
    side, world = _side(tmp_path, F.Plan(), max_steps=30, strict_cap=True)
    res = side.run(F.identity())
    assert res["status"] == "timeout" and res["exception"] == "StepCapReached" and len(world.envs[0].actions) == 30
    end = _check_kept(res, source="official-salvaged", label="timeout", frames=F.N_RESET_FRAMES + 30)
    assert (end["status"], end["terminal_reason"], end["success_flag"]) == ("timeout", "timeout", "error")
    assert end["omitted_timeout_frames"] == 0
    TC.assert_renderable(Path(res["trace_path"]).parent)


def test_exception_salvages_official_video_and_keeps_missing_step(tmp_path):
    """环境第 5 步抛异常：该步缺观测（log_missing_step），官方循环随后抛 AttributeError；补存 3 + 4 帧。"""
    side, _ = _side(tmp_path, F.Plan(raise_at=5))
    res = side.run(F.identity())
    assert res["status"] == "error" and res["error"].startswith("AttributeError")
    end = _check_kept(res, source="official-salvaged", label="error", frames=F.N_RESET_FRAMES + 4)
    assert (end["steps_attempted"], end["steps_observed"], end["terminal_reason"]) == (5, 4, "error")
    last = [r for r in F.read_trace(res["trace_path"]) if r["kind"] == "step"][-1]
    assert last["step"] == 5 and last["observed"] is False and last["front_sha256"] is None
    assert last["missing_reason"].startswith("RuntimeError: fake env step failure at 5")
    assert last["action"] is not None and last["terminated"] == "NOT_OBSERVED"
    # error 局：重绘器放行由 S2a 负责，这里只断言契约部分
    assert TC.contract_problems(Path(res["trace_path"]).parent) == []


def test_unknown_salvages_official_video(tmp_path):
    side, _ = _side(tmp_path, F.Plan(unknown_at=10))
    res = side.run(F.identity())
    assert (res["status"], res["error"]) == ("error", "success_flag=unknown")
    end = _check_kept(res, source="official-salvaged", label="unknown", frames=F.N_RESET_FRAMES + 10)
    assert (end["status"], end["terminal_reason"], end["success_flag"]) == ("error", "error", "unknown")
    assert TC.contract_problems(Path(res["trace_path"]).parent) == []


def test_failure_before_init_episode_is_none(tmp_path):
    """策略 reset（init_episode 之前）就失败：一帧未录，official_source=none，轨迹 no_frame 局过契约。"""
    side, _ = _side(tmp_path, F.Plan(success_at=5))

    def boom(obj):
        raise ConnectionRefusedError("fake server down")

    side.server.handle = boom
    res = side.run(F.identity())
    assert res["status"] == "error" and res["official_source"] == "none" and res["official_videos"] == []
    assert "一帧未录" in res["official_save_error"]
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "official").exists() and not (tdir / "official-video").exists()
    end = F.read_trace(res["trace_path"])[-1]
    assert (end["no_frame"], end["frames_recorded"], end["demo_frames"], end["steps_attempted"]) == (True, 0, 0, 0)
    assert TC.contract_problems(tdir) == []
    assert "init_episode" not in vars(side.ctx["evaluator"])


def test_get_init_obs_failure_is_none(tmp_path, monkeypatch):
    """init_episode 里取初始观测就失败（录像器尚未建）：none。"""
    side, _ = _side(tmp_path, F.Plan(success_at=5))

    def bad_reset(self):
        raise RuntimeError("reset boom")

    monkeypatch.setattr(F.FakeEnv, "reset", bad_reset)
    res = side.run(F.identity())
    assert res["status"] == "error" and res["official_source"] == "none" and res["no_frame"] is True
    assert not (Path(res["trace_path"]).parent / "official").exists()


def test_init_episode_midway_failure_is_partial(tmp_path, monkeypatch):
    """取到 reset 帧后、官方录像器交出前失败：partial，只记原因，不出视频（不冒充完整）。"""
    side, _ = _side(tmp_path, F.Plan(success_at=5))
    g = type(side.ctx["evaluator"]).init_episode.__globals__
    real = g["RolloutRecorder"]

    class BrokenRecorder(real):
        def record(self, *a, **kw):
            if len(self.total_images) == 1:
                raise RuntimeError("recorder boom")
            return super().record(*a, **kw)

    monkeypatch.setitem(g, "RolloutRecorder", BrokenRecorder)
    res = side.run(F.identity())
    assert res["status"] == "error" and res["official_source"] == "partial" and res["official_videos"] == []
    assert "init_episode 中途失败" in res["official_save_error"]
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "official").exists()
    end = F.read_trace(res["trace_path"])[-1]
    assert end["official_source"] == "partial" and "no_frame" not in end
    assert TC.contract_problems(tdir) == []


def _make_mp4(path: Path, n: int) -> Path:
    import imageio.v2 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.full((16, 16, 3), i * 9 % 256, dtype=np.uint8) for i in range(n)]
    iio.mimsave(str(path), frames, fps=30)
    assert _decoded_frames(path) == n
    return path


def test_keep_official_videos_accepts_exactly_one_full_video(tmp_path, capsys):
    mc = F.groundsg_client()
    src = _make_mp4(tmp_path / "vd" / "a.mp4", 12)
    sha = _sha(src)
    got = mc.keep_official_videos(tmp_path / "vd", tmp_path / "ep" / "official", expected_frames=12,
                                  provenance={"official_source": "official", "route": "groundsg/x/new"})
    assert got == [str(tmp_path / "ep" / "official" / "a.mp4")] and not src.exists()
    prov = json.loads((tmp_path / "ep" / "official" / "provenance.json").read_text())
    assert prov["video"]["sha256"] == sha and prov["frames"]["decoded"] == 12 and prov["route"] == "groundsg/x/new"
    assert "OFFICIAL_VIDEO=KEPT" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["truncated", "wrong_frames", "two_files", "no_file", "dst_not_empty"])
def test_keep_official_videos_rejects(tmp_path, capsys, case):
    """截断 mp4、错帧数、多片、无片、目标非空：一律拒绝，不搬、不写 provenance、不打印 KEPT。"""
    mc = F.groundsg_client()
    vd, dst = tmp_path / "vd", tmp_path / "ep" / "official"
    src = _make_mp4(vd / "a.mp4", 12)
    expected = 12
    if case == "truncated":
        data = src.read_bytes()
        src.write_bytes(data[: len(data) // 2])
    elif case == "wrong_frames":
        expected = 13
    elif case == "two_files":
        shutil.copy(src, vd / "b.mp4")
    elif case == "no_file":
        src.unlink()
    elif case == "dst_not_empty":
        dst.mkdir(parents=True)
        (dst / "old.mp4").write_bytes(b"x")
    before = sorted(p.name for p in vd.iterdir())
    with pytest.raises(mc.OfficialVideoRejected):
        mc.keep_official_videos(vd, dst, expected_frames=expected, provenance={"official_source": "official"})
    assert sorted(p.name for p in vd.iterdir()) == before
    assert not (dst / "provenance.json").exists() and not (dst / "a.mp4").exists()
    assert "OFFICIAL_VIDEO=KEPT" not in capsys.readouterr().out


def test_rejected_official_video_keeps_raw_and_records_error(tmp_path, monkeypatch):
    """核帧失败（这里令期望帧数 +1）时局照常收尾：official_source=none、记 official_save_error、无 official/。"""
    mc = F.groundsg_client()
    real = mc.episode_counts

    def off_by_one(*a, **kw):
        c = real(*a, **kw)
        return dict(c, frames_recorded=c["frames_recorded"] + 1)

    monkeypatch.setattr(mc, "episode_counts", off_by_one)
    side, _ = _side(tmp_path, F.Plan(success_at=20))
    res = side.run(F.identity())
    assert res["status"] == "success" and res["official_source"] == "none" and res["official_videos"] == []
    assert res["official_save_error"].startswith("OfficialVideoRejected: 帧数不符")
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "official").exists() and not (tdir / "official-video").exists()


#: BASE（b869a3df）run_official_episode 返回字典的键与顺序（手写自 BASE 源码）
BASE_RESULT_KEYS = ["status", "task_success", "steps", "error", "success_flag", "decisions", "infra", "infra_reason",
                    "env_exception", "exception", "qwen_log", "timing", "official_sha256"]


@pytest.mark.parametrize("name", ["success", "unknown", "env_raise"])
def test_keep_official_unset_is_base_behavior(tmp_path, name):
    """不传 keep_official、上下文不带 policy_seed（旧调用方）：返回键与 BASE 相同、不包 init_episode、不建 official/、
    叠字 mp4 照删。直接调 ``run_official_episode``（第三阶段起原侧驱动本身已改传 keep_official，见下一用例）。"""
    plan, max_steps = SCENARIOS[name]
    orig = F.OrigSide(F.ORACLE, max_steps, tmp_path, F.World(default=plan))
    mc, ohr, ctx = orig.ohr.groundsg, orig.ohr, orig.ctx
    assert mc.run_official_episode.__kwdefaults__["keep_official"] is mc.trace_writer.UNSET
    ident = F.identity()
    runner = ohr.runner_for(ctx, ident["task"], tmp_path / "official-video")
    runner.make_env(ident["source_episode"])
    tap = mc.EpisodeTap(None)
    seed = ctx.pop("policy_seed")
    try:
        res = mc.run_official_episode(ctx, ohr.TapRunner(runner, tap), tap, dataset=F.DATASET, episode_tag="x.a1",
                                      scratch=tmp_path, archive_dir=tmp_path)
    finally:
        ctx["policy_seed"] = seed
        runner.close_env()
    assert list(res) == BASE_RESULT_KEYS
    assert "init_episode" not in vars(ctx["evaluator"])
    assert not (tmp_path / "official").exists() and not (tmp_path / "official-video").exists()


@pytest.mark.parametrize("name", ["success", "unknown", "env_raise", "timeout"])
def test_orig_side_keeps_official_video_and_counts(tmp_path, name):
    """第三阶段原侧（八.10 第 8 条）：传 keep_official=True + provenance，官方视频核验后进 official/（归档先于清理）；
    identity 补 attempt 与 policy_seed，end 补三分计数，缺观测步按 C8 记；结果行带 policy_seed 与官方来源。"""
    plan, max_steps = SCENARIOS[name]
    orig = F.OrigSide(F.ORACLE, max_steps, tmp_path, F.World(default=plan), policy_seed=42)
    row = orig.run(F.identity(), attempt=2)
    ep = Path(row["ep_dir"])
    assert ep.name == "PickXtimes_xhard0_510300.a2"
    mp4s = sorted((ep / "official").glob("*.mp4"))
    assert len(mp4s) == 1 and row["official_videos"] == [str(mp4s[0])] and row["official_save_error"] is None
    assert row["official_source"] == ("official" if name in ("success", "timeout") else "official-salvaged")
    prov = json.loads((ep / "official" / "provenance.json").read_text())
    assert prov["route"] == f"groundsg/{F.ORACLE}/orig" and prov["side"] == "orig" and prov["policy_seed"] == 42
    assert prov["identity"]["attempt"] == 2 and prov["frames"]["decoded"] == row["frames_recorded"]
    assert not (ep / "official-video").exists()
    rows = F.read_trace(row["trace_path"])
    assert (rows[0]["identity"]["attempt"], rows[0]["identity"]["policy_seed"]) == (2, 42)
    end = rows[-1]
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == \
        (row["steps_attempted"], row["steps_observed"], row["frames_recorded"])
    assert end["policy_seed"] == 42 and end["terminal_reason"] == row["status"]
    assert row["policy_seed"] == 42 and row["effective_cap"] == max_steps
    if name == "env_raise":
        last = [r for r in rows if r["kind"] == "step"][-1]
        assert last["observed"] is False and (end["steps_attempted"], end["steps_observed"]) == (5, 4)
    assert TC.contract_problems(ep) == []
