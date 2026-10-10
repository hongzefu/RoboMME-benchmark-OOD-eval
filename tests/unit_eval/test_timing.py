"""用假时钟与独立手算覆盖十五类计时契约；不使用模型或仿真。"""
import copy
import json
from pathlib import Path

import pytest

from robomme_ood_eval.timing import ChunkTimer, attach_policy_timing, merge_summaries, summarize_chunks


def record(timer, wall=100, rtt=80, lang=0, **extra):
    if lang:
        timer.add_lang(lang, kind="s2", env_step=len(timer.chunks))
    return timer.close(decision=len(timer.chunks), env_step=len(timer.chunks), action_rtt_ms=rtt,
                       server_infer_ms=60, decision_wall_ms=wall, extra=extra)


def attach(timer, kind="serial"):
    return attach_policy_timing({}, timer, gpu_name="CPU-test", model_kind=kind)


def test_counted_and_detail():
    t = ChunkTimer(clock=lambda: 0)
    t.start()
    t.add_lang(30, kind="subgoal", env_step=0, reuse=False, omitted=None)
    t.add_lang(50, kind="ask", env_step=0)
    c = t.close(decision=0, env_step=0, action_rtt_ms=60, server_infer_ms=40, t=.1)
    assert c["lang_ms"] == 30 and c["n_lang"] == 1
    assert len(t.lang_calls) == 2 and "omitted" not in t.lang_calls[0]
    assert attach(t)["lang_summary"]["total_ms_counted"] == 30


def test_restart_explicit_end():
    now = [0]
    t = ChunkTimer(clock=lambda: now[0])
    t.start()
    now[0] = 1
    t.start()
    now[0] = 99
    assert t.close(decision=0, env_step=0, action_rtt_ms=5, server_infer_ms=4, t=1.01)["decision_wall_ms"] == 10


def test_missing_start():
    with pytest.raises(RuntimeError, match="without start"):
        ChunkTimer().close(decision=0, env_step=0, action_rtt_ms=1, server_infer_ms=1)


def test_explicit_zero_json():
    t = ChunkTimer()
    c = record(t)
    assert json.loads(json.dumps(c))["lang_ms"] == 0.0 and c["n_lang"] == 0


def test_invalid_values_and_reserved():
    t = ChunkTimer()
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            t.add_lang(value, kind="s2", env_step=0)
    with pytest.raises(ValueError):
        t.add_lang(1, kind="wrong", env_step=0)
    for key in ("decision_wall_ms", "action_rtt_ms", "server_infer_ms", "lang_ms", "decision", "env_step", "n_lang", "phase"):
        with pytest.raises(ValueError):
            record(t, **{key: 1})
    with pytest.raises(ValueError):
        attach(t, "wrong")


def test_orphan_absorption():
    t = ChunkTimer()
    t.add_lang(20, kind="s2", env_step=0)
    t.start()
    t.add_lang(30, kind="s2", env_step=0)
    t.absorb_orphan(40, 1)
    assert t.orphan() == {"orphan_lang_ms": 90, "n": 3}
    assert t.orphan() == {"orphan_lang_ms": 0.0, "n": 0}
    assert t._t_obs is not None
    for ms, n in ((-1, 1), (1, -1), (float("nan"), 1)):
        with pytest.raises(ValueError):
            t.absorb_orphan(ms, n)


def test_phases_before_filter_and_linear_percentile():
    t = ChunkTimer()
    for i in range(10):
        record(t, wall=10 * (i + 1), rtt=None if i == 0 else i)
    s = summarize_chunks(t.chunks)
    assert s["action_rtt_ms"]["first"] is None
    assert s["action_rtt_ms"]["second"] == 1
    assert s["decision_wall_ms"]["steady_mean"] == 70
    assert s["decision_wall_ms"]["steady_p50"] == 70
    assert s["decision_wall_ms"]["steady_p95"] == 97
    short = summarize_chunks(t.chunks[:3])
    assert short["steady_short"] and short["decision_wall_ms"]["steady_mean"] == 20
    sub = summarize_chunks([t.chunks[5]])
    assert not sub["steady_short"] and sub["decision_wall_ms"]["first"] is None


def test_serial_conservation():
    t = ChunkTimer()
    for wall in (100, 100, 60, 100, 100):
        record(t, wall, rtt=60, lang=20)
    c = attach(t)["conservation"]
    assert c["violations"] == 1 and c["missing_fields"] == 0
    assert c["overhead_pct_p50"] == pytest.approx(.2)
    assert c["expected"] == c["checked"] == 5


def test_pp_direct_wall_and_noise():
    t = ChunkTimer()
    record(t, 100.5, rtt=80, lang=20)
    record(t, 102, rtt=80, lang=20)
    c = attach(t, "pp")["conservation"]
    assert c["violations"] == 1 and c["missing_fields"] == 0


def test_smvla_inequalities_and_slack():
    t = ChunkTimer()
    for wall, rtt, obs, lang, infer in ((100, 80, 20, 40, 60), (90, 80, 20, 40, 60),
                                        (100, 80, 20, 70, 60), (100, 80, 20, 40, 90), (100, 80, None, 40, 60)):
        t.add_lang(lang, kind="lang_in_forward", env_step=0)
        t.close(decision=len(t.chunks), env_step=0, action_rtt_ms=rtt, server_infer_ms=infer,
                decision_wall_ms=wall, extra={"observe_rtt_ms": obs})
    out = attach(t, "smvla")
    c = out["conservation"]
    assert c["violations"] == 4 and c["missing_fields"] == 1
    assert c["checked"] == 4 and c["expected"] == 5
    assert c["lang_split_available"]
    assert out["chunk_summary"]["slack_ms"]["n"] == 4
    assert out["chunk_summary"]["slack_ms"]["steady_mean"] == 0


def test_astra_layers_and_merge():
    t = ChunkTimer()
    for planner, review, lang in ((True, True, 10), (True, False, 10), (False, True, 10),
                                 (False, False, 10), (False, False, 0)):
        record(t, lang=lang, has_planner=planner, has_review=review)
    p = attach(t)
    layers = p["chunk_summary"]["by_planner_review"]
    assert all(s["n_decisions"] == 1 for s in layers.values())
    assert layers["neither"]["decision_wall_ms"]["first"] is None
    assert layers["neither"]["steady_short"] is False
    merged = merge_summaries([p, p])
    assert all(s["decisions"] == 2 for s in merged["by_planner_review"].values())
    assert all(s["expected"] == s["checked"] == 2 for s in merged["by_planner_review"].values())
    assert merged["by_planner_review"]["neither"]["decision_wall_ms"]["first_mean"] is None
    t2 = ChunkTimer()
    record(t2, has_planner=True, has_review=False, lang=10)
    assert attach(t2)["chunk_summary"]["by_planner_review"]["review_only"] == {"n": 0}


def test_attach_idempotent_preserves_existing():
    t = ChunkTimer()
    record(t)
    t.add_lang(40, kind="monitor", env_step=0)
    t.absorb_orphan(10, 1)
    timing = {"policy_load": {"load_s": 1}, "infer_ms": [60]}
    attach_policy_timing(timing, t, gpu_name=None, model_kind="serial")
    first = copy.deepcopy(timing)
    timing["chunks"][0]["decision_wall_ms"] = -1
    attach_policy_timing(timing, t, gpu_name=None, model_kind="serial")
    assert timing == first and timing["orphan_lang_ms"] == 50 and timing["orphan_n"] == 2
    assert timing["policy_load"] == {"load_s": 1} and timing["infer_ms"] == [60]


def test_merge_weights_and_raw_percentiles():
    a, b = ChunkTimer(), ChunkTimer()
    for w in (1, 1, 1, 10, 20):
        record(a, wall=w, rtt=0)
    for w in (1, 1, 1, 100):
        record(b, wall=w, rtt=0)
    pa, pb = attach(a), attach(b)
    out = merge_summaries([pa, pb])
    assert out["decision_wall_ms"]["steady_mean"] == pytest.approx(130 / 3)
    assert out["decision_wall_ms"]["steady_p95"] == pytest.approx(92)
    assert not out["approx"]
    pa.pop("chunks")
    assert merge_summaries([pa, pb])["approx"]
    pb["gpu_name"] = None
    with pytest.raises(ValueError):
        merge_summaries([pa, pb])


def test_source_field_name():
    import robomme_ood_eval.timing as module

    assert "chunk_rtt_ms" not in Path(module.__file__).read_text()
    print(f"TIMING_SOURCE={module.__file__}")


def test_missing_values_never_pass():
    for kind in ("serial", "pp", "smvla"):
        t = ChunkTimer()
        record(t, rtt=None, observe_rtt_ms=0)
        out = attach(t, kind)
        c = out["conservation"]
        assert c["missing_fields"] == 1 and c["violations"] == 0
        assert c["checked"] < c["expected"]
        assert out["chunk_summary"]["action_rtt_ms"] == {"n": 0}
        if kind == "smvla":
            assert not c["lang_split_available"]
    print("TIMING_HELPER=PASS cases=15 kinds=serial,pp,smvla")
