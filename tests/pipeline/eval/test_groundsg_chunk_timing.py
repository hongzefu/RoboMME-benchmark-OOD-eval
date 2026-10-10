"""GroundSG 动作块计时：真实抽取的官方循环与模型，全部使用 CPU 替身。"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robomme_ood_eval import timing
from robomme_ood_eval.models import groundsg as M
from robomme_ood_eval.session import StepCapReached
from tests.pipeline.evalx.groundsg import groundsg_fakes as F

GOOD = '{"current_subtask": "move cube", "keyframe_positions": []}'
BAD = "invalid json"
ASK, BUFFER, RTT, SERVER = .030, .015, .020, 10.0


class Ledger:
    """仅记录语言账本调用，不写磁盘，契约与 LanguageLog 相同。"""

    def __init__(self):
        self.calls, self.messages, self.closed, self.reused = [], [], [], []

    def open_call(self, *args, **kwargs):
        cid = f"call-{len(self.calls)}"
        self.calls.append((cid, args, kwargs))
        return cid

    def message(self, cid, **kwargs):
        self.messages.append((cid, kwargs))

    def close_call(self, cid, **kwargs):
        self.closed.append((cid, kwargs))

    def reuse(self, *args):
        self.reused.append(args)


class Transport:
    """具有实际等待和接收时刻的客户端；解码等待发生在接收之后。"""

    def __init__(self, *, bad=None, missing=False, decode_delay=0):
        self.n = 0
        self.bad, self.missing, self.decode_delay = bad, missing, decode_delay
        self._last_recv_t = None
        self._ws = None
        if not missing:
            self._last_rtt, self._last_server_timing = {}, None

    def reset(self):
        return {"reset_finished": True}

    def add_buffer(self, obj):
        time.sleep(BUFFER)
        return {"add_buffer_finished": True}

    def infer(self, obj):
        t0 = time.perf_counter()
        time.sleep(RTT)
        self._last_recv_t = time.perf_counter()
        if not self.missing:
            self._last_rtt = {"infer": (self._last_recv_t - t0) * 1000}
            self._last_server_timing = {"infer_ms": SERVER}
        self.n += 1
        if self.bad is not None and self.n == 2:
            if self.bad == "raise":
                raise RuntimeError("transport failure")
            return {"foo": 1} if self.bad == "dict" else None
        time.sleep(self.decode_delay)
        return {"actions": np.zeros((16, 7)), M.AUDIT_KEY: {"channels": []}}


class Runner:
    """不调用仿真的官方 EnvRunner 接口；执行次数由夹具固定。"""

    episode_id, task_goal, difficulty = "0", "move cube", "hard"
    grounded_subgoal_oracle = "move cube"

    def __init__(self, tap, *, task, steps, cap=None):
        self.tap, self.env_id, self.steps, self.cap = tap, task, steps, cap

    def step(self, action):
        if self.cap is not None and self.tap.steps >= self.cap:
            raise StepCapReached("fake strict cap")
        obs = (F.frame(1), F.frame(2), np.zeros(8))
        stop = self.tap.steps + 1 >= self.steps
        flag = "success" if stop else "ongoing"
        self.tap.on_step(action, obs, stop, flag, stop, False)
        return obs, stop, flag


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    for key in ("IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "FPS_MAX_FRAMES", "USE_HF", "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(M, "_timing_cuda", lambda: None)


def run_case(tmp, variant, *, decisions=1, task="PickXtimes", script=None, ledger=True,
             bad=None, missing=False, cap=None, decode_delay=0):
    sw = F.FakeSwift(script=script)
    base = sw.names["PtEngine"]

    class SlowEngine(base):
        def infer(self, *args, **kwargs):
            time.sleep(ASK)
            return super().infer(*args, **kwargs)

    sw.names["PtEngine"] = SlowEngine
    clients = []

    def factory(h, p, ep):
        client = Transport(bad=bad, missing=missing, decode_delay=decode_delay)
        clients.append(client)
        return client

    ctx = M.make_policy_context(F.seat_info(variant, decisions * 16 + 1, tmp), client_factory=factory,
                                qwen_extra=sw.names)
    original = ctx["predictor"]
    engine = getattr(getattr(original, "api", None), "engine", None)
    tap = M.EpisodeTap(None)
    runner = Runner(tap, task=task, steps=decisions * 16, cap=cap)

    def init_episode(r, epstate, video_dir):
        pre = {"images": [F.frame(0)], "wrist_images": [F.frame(1)], "states": [np.zeros(8)],
               "task_goal": r.task_goal}
        tap.on_reset(pre)
        epstate.image_buffer.extend(pre["images"])
        epstate.wrist_image_buffer.extend(pre["wrist_images"])
        epstate.state_buffer.extend(pre["states"])
        return r.task_goal, SimpleNamespace(record=lambda **kw: None, save_video=lambda name: None)

    ctx["evaluator"].init_episode = init_episode
    log = Ledger() if ledger else None
    res = M.run_official_episode(ctx, runner, tap, dataset="hard-verify", episode_tag="timing.a1",
                                 scratch=tmp, archive_dir=None, language_log=log)
    assert ctx["predictor"] is original and ctx["episode"] is None
    assert "get_subgoal" not in vars(original)
    if engine is not None:
        assert original.api.engine is engine
    return res, log, sw, clients


def split(res):
    t = res["timing"]
    return ([x for x in t["lang_calls"] if x["kind"] == "subgoal"],
            [x for x in t["lang_calls"] if x["kind"] == "ask"])


def check_chunks(res, variant, expected):
    t = res["timing"]
    chunks = t["chunks"]
    assert res["status"] in {"success", "fail", "timeout"}
    assert len(chunks) == res["decisions"] == math.ceil(res["steps"] / 16) == expected
    assert t["timing_missing"] == 0
    assert [c["decision"] for c in chunks] == list(range(expected))
    for c in chunks:
        assert c["action_rtt_ms"] == pytest.approx(RTT * 1000, abs=5)
        assert c["server_infer_ms"] == SERVER
        assert c["decision_wall_ms"] >= c["action_rtt_ms"] + c["lang_ms"] - 1
    assert t["conservation"]["violations"] == 0
    gap = [c["decision_wall_ms"] - c["action_rtt_ms"] - c["lang_ms"] for c in chunks]
    assert min(gap) >= BUFFER * 1000 - 5
    model = variant.replace("ground-sg", "groundsg")
    print(f"CHUNK_TIMING=PASS model={model} chunks={expected} decisions={res['decisions']} missing=0 "
          f"rtt_ok={expected} wall_ge_rtt={expected}")
    print(f"WALL_DECOMP=PASS model={model} kind=serial chunks={expected} violations=0 "
          f"overhead_pct_p50={t['conservation']['overhead_pct_p50']} min_gap_ms={min(gap):.3f} "
          f"max_gap_ms={max(gap):.3f} lang_ms={','.join(str(c['lang_ms']) for c in chunks)}")


def check_split(res, model):
    subs, asks = split(res)
    total = sum(x["ms"] for x in subs)
    counted = sum(x["lang_ms"] for x in res["timing"]["chunks"]) + res["timing"]["orphan_lang_ms"]
    delta = abs(counted - total) / total * 100 if total else 0
    assert delta < 1
    assert res["language"]["subgoal_calls"] == len(asks)
    assert res["language"]["reuses"] == sum(x["reuse"] for x in subs)
    print(f"LANG_SPLIT=PASS model={model} calls={len(subs)} delta_pct={delta:.4f} asks={len(asks)} "
          f"reuse={sum(x['reuse'] for x in subs)} retry={','.join(str(x['retry']) for x in asks)}")
    return subs, asks


def test_oracle_three_chunks(tmp_path):
    res, _, sw, _ = run_case(tmp_path, F.ORACLE, decisions=3)
    check_chunks(res, F.ORACLE, 3)
    assert res["timing"]["lang_calls"] == [] and sw.requests == []


@pytest.mark.parametrize("task, asks, reuses", [("ButtonUnmask", 1, 3), ("PickXtimes", 4, 0)])
def test_qwenvl_reuse_and_normal(tmp_path, task, asks, reuses):
    res, log, _, _ = run_case(tmp_path, F.QWENVL, decisions=4, task=task)
    check_chunks(res, F.QWENVL, 4)
    subs, detail = check_split(res, "groundsg-qwenvl")
    assert len(detail) == asks and sum(x["reuse"] for x in subs) == reuses
    assert subs[0]["ms"] >= ASK * 1000 - 5
    assert all(x["ms"] < 5 for x in subs if x["reuse"])
    assert len(log.reused) == reuses
    assert all(c["n_lang"] == 1 for c in res["timing"]["chunks"])


def test_memer_retry_and_fallback(tmp_path):
    retry, _, _, _ = run_case(tmp_path / "retry", F.MEMER, script=[BAD, GOOD])
    subs, asks = check_split(retry, "groundsg-memer")
    assert subs[0]["tries"] == 2 and [a["retry"] for a in asks] == [0, 1]
    assert all(a["ms"] == pytest.approx(30, abs=5) for a in asks)
    assert subs[0]["ms"] >= 60 - 5
    fallback, _, _, _ = run_case(tmp_path / "fallback", F.MEMER, decisions=2, script=[GOOD, BAD, BAD, BAD])
    check_chunks(fallback, F.MEMER, 2)
    subs, asks = check_split(fallback, "groundsg-memer")
    assert [s["tries"] for s in subs] == [1, 3]
    assert subs[1]["fallback"] == "last_valid" and not any(s["reuse"] for s in subs)
    assert [a["retry"] for a in asks] == [0, 0, 1, 2]
    assert subs[1]["ms"] >= 90 - 5


def test_memer_no_history_orphan(tmp_path):
    res, _, _, clients = run_case(tmp_path, F.MEMER, script=[BAD] * 3)
    assert res["status"] == "error" and res["error_kind"] == "model_response_error"
    subs, asks = check_split(res, "groundsg-memer")
    assert res["timing"]["chunks"] == [] and clients[0].n == 0
    assert len(subs) == 1 and subs[0]["tries"] == 3 and len(asks) == 3
    assert subs[0]["error"] == "MemERResponseError"
    assert res["timing"]["orphan_lang_ms"] == pytest.approx(subs[0]["ms"], abs=.001)
    assert res["timing"]["orphan_lang_ms"] >= 85


@pytest.mark.parametrize("variant, task", [(F.QWENVL, "ButtonUnmask"), (F.MEMER, "PickXtimes")])
def test_ledger_off_preserves_wall_and_subgoal(tmp_path, variant, task):
    opened, _, _, _ = run_case(tmp_path / "on", variant, decisions=2, task=task)
    closed, _, _, _ = run_case(tmp_path / "off", variant, decisions=2, task=task, ledger=False)
    subs_on, asks_on = split(opened)
    subs_off, asks_off = split(closed)
    assert len(subs_on) == len(subs_off) == 2 and asks_on and asks_off == []
    assert "language" not in closed
    assert [s["tries"] for s in subs_on] == [s["tries"] for s in subs_off]
    assert [s["reuse"] for s in subs_on] == [s["reuse"] for s in subs_off]
    for res in (opened, closed):
        assert len(res["timing"]["chunks"]) == 2
        assert res["timing"]["conservation"]["violations"] == 0


@pytest.mark.parametrize("bad", ["dict", "non-dict", "raise"])
def test_invalid_action_reply_no_chunk(tmp_path, bad):
    res, _, _, _ = run_case(tmp_path, F.QWENVL, decisions=2, bad=bad)
    assert res["status"] == "error" and len(res["timing"]["chunks"]) == 1
    assert res["timing"]["timing_missing"] == 0
    assert res["timing"]["orphan_n"] == 1 and res["timing"]["orphan_lang_ms"] >= 25
    print(f"BAD_REPLY=PASS kind={bad} bad_reply_chunks=0 chunks=1 orphan_n=1")


def test_missing_transport_fields_tolerated(tmp_path):
    res, _, _, _ = run_case(tmp_path, F.QWENVL, missing=True)
    assert res["status"] == "success" and res["timing"]["timing_missing"] == 1
    chunk = res["timing"]["chunks"][0]
    assert chunk["action_rtt_ms"] is None and chunk["server_infer_ms"] is None
    assert chunk["decision_wall_ms"] >= 60


def test_receive_timestamp_excludes_decode(tmp_path):
    res, _, _, clients = run_case(tmp_path, F.ORACLE, decode_delay=.040)
    chunk = res["timing"]["chunks"][0]
    assert clients[0]._last_recv_t is not None
    assert 30 <= chunk["decision_wall_ms"] < 60
    assert res["timing"]["episode_s"] * 1000 >= chunk["decision_wall_ms"] + 35


def test_step_cap_retains_received_chunk(tmp_path):
    res, _, _, _ = run_case(tmp_path, F.MEMER, decisions=2, cap=17)
    assert res["status"] == "timeout" and res["exception"] == "StepCapReached"
    assert res["steps"] == 17
    check_chunks(res, F.MEMER, 2)


def test_engine_error_skips_ask_and_keeps_subgoal_orphan():
    timer = timing.ChunkTimer()
    tap = M.EpisodeTap(None)

    class Broken:
        def infer(self, *args, **kwargs):
            raise RuntimeError("engine failed")

    api = SimpleNamespace(engine=Broken())
    cls = type("QwenVLSubgoalPredictor", (), {"get_subgoal": lambda self, *args: self.api.engine.infer(
        [SimpleNamespace(kw={"messages": [], "images": []})])})
    predictor = cls()
    predictor.api = api
    lang = M.LangTap(Ledger(), tap, variant=F.QWENVL, api=api, timer=timer)
    undo = M.install_language(predictor, lang)
    counter = M._CountingEngine(api.engine)
    api.engine = counter
    with pytest.raises(RuntimeError, match="engine failed"):
        M._TimedPredictor(predictor, timer, variant=F.QWENVL, counter=counter, tap=tap).get_subgoal(0, None, None)
    assert [x["kind"] for x in timer.lang_calls] == ["subgoal"]
    assert timer.lang_calls[0]["tries"] == 1 and lang._open_t == {}
    assert timer.orphan()["n"] == 1
    api.engine = counter._inner
    undo()


def test_two_argument_client_preserves_legacy_path():
    tap = M.EpisodeTap(None)
    inner = Transport()
    out = M.TracingClient(inner, tap).infer({})
    assert "actions" in out and M.AUDIT_KEY not in out and tap.decisions == 1


def test_stale_transport_timing_cleared_and_fallback_timestamp():
    timer = timing.ChunkTimer()
    inner = SimpleNamespace(_last_rtt={"infer": 999}, _last_server_timing={"infer_ms": 999},
                            _last_recv_t=1, infer=lambda obs: {"actions": np.zeros((16, 7))})
    summary = {}
    timer.start()
    M.TracingClient(inner, M.EpisodeTap(None), timer, summary).infer({})
    assert summary["timing_missing"] == 1
    assert timer.chunks[0]["action_rtt_ms"] is None and timer.chunks[0]["server_infer_ms"] is None
    assert timer.chunks[0]["decision_wall_ms"] >= 0 and inner._last_recv_t is None


def test_present_server_timing_none_is_not_missing():
    timer = timing.ChunkTimer()

    class Client:
        _last_rtt, _last_server_timing = {}, None

        def infer(self, obs):
            self._last_rtt = {"infer": 0}
            return {"actions": np.zeros((16, 7))}

    summary = {}
    timer.start()
    M.TracingClient(Client(), M.EpisodeTap(None), timer, summary).infer({})
    assert summary == {} and timer.chunks[0]["server_infer_ms"] is None


def test_sync_only_language_calls(monkeypatch):
    events = []
    cuda = SimpleNamespace(synchronize=lambda: events.append("sync"))
    monkeypatch.setattr(M, "_timing_cuda", lambda: cuda)
    inner = SimpleNamespace(get_subgoal=lambda *args: events.append("subgoal") or ("move cube", False))
    timer = timing.ChunkTimer()
    M._TimedPredictor(inner, timer, variant=F.ORACLE).get_subgoal(0, None, None)
    assert events == ["subgoal"] and timer.lang_calls == []
    events.clear()
    M._TimedPredictor(inner, timer, variant=F.QWENVL).get_subgoal(0, None, None)
    assert events == ["sync", "subgoal", "sync"]
    assert timer.lang_calls[0]["kind"] == "subgoal"
    assert "tries" not in timer.lang_calls[0] and "reuse" not in timer.lang_calls[0]
