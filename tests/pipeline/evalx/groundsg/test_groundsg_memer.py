"""MemER wiring and compat layer (1006-rename-official-names-and-stage3-eval-plan.md part two, 8.3; verdict lines
MEMER_COMPAT, MEMER_WIRING).

Everything uses the **really extracted official class** ``subgoal_prediction/qwenvl/api_memer.py::Qwen3VLModelMemER``
(``official_defs.load_memer_model`` with the compat layer applied; ``compat=False`` takes the official original as a
regression reference) paired with a fake ``PtEngine`` (``groundsg_fakes.FakeSwift``, reads no weights). All
expectations are hand-written: reply texts, the reminder sentence, temperatures, log rows, keyframe merge results and
every-other-frame indices are fixed in this file; expectations are never derived by calling the code under test.

Eight cases (plan 8.3 acceptance): A first query with empty keyframes; B non-empty memory and valid reply (byte-for-byte
identical to the official function); C1 "bad, bad, good" with a previous valid subgoal; C2 "bad, bad, bad" with a
previous one (reuse it); D1 "bad, bad, good" without a previous one; D2 "bad, bad, bad" without a previous one (named
exception); E missing ``keyframe_positions`` key; F fewer than 15 execution frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import groundsg_fakes as F

NOTE = "Your previous reply was not valid JSON. Reply with the JSON object only."  # hand-written (verbatim from plan 8.3)
GOOD = '{"current_subtask": "move cube", "keyframe_positions": []}'
BAD = "this is not json"


def od():
    return F.official_defs()


def memer_cls(swift: F.FakeSwift, *, compat: bool = True):
    return od().load_memer_model(swift_names=swift.names, compat=compat)["Qwen3VLModelMemER"]


def new_model(tmp: Path, swift: F.FakeSwift, *, compat: bool = True, frames: int = 1, name: str = "ep0"):
    m = memer_cls(swift, compat=compat)(adapter_path=F.MEMER_ADAPTER)
    m.start_new_episode(str(tmp / "PickXtimes" / name), [], "goal PickXtimes 3")
    for i in range(frames):
        m.add_execution_frame(F.frame(10 + i))
    return m


def log_rows(m) -> list[dict]:
    return [json.loads(x) for x in Path(m.save_json_path).read_text().splitlines()]


def user_text(req: dict) -> str:
    return next(x["content"] for x in req["messages"] if x["role"] == "user")


def frame_ids(paths) -> list[int]:
    return [int(Path(p).name.split("_")[1]) for p in paths]


# ---------------------------------------------------------------- eight cases


def case_a(tmp):
    """A: first query with empty memory, model replies with empty keyframes -- the official original lets IndexError
    escape call; the compat layer extracts the subgoal and the episode continues."""
    sw_o = F.FakeSwift(script=[GOOD])
    off = new_model(tmp / "off", sw_o, compat=False)
    with pytest.raises(IndexError):
        off.call()
    sw = F.FakeSwift(script=[GOOD])
    m = new_model(tmp / "new", sw)
    assert m.call() == "move cube"
    assert (m.subgoals, m.key_frame_paths, m._memer_fallback) == (["move cube"], {}, None)
    assert [r["config"] for r in sw.requests] == [{"max_tokens": 128, "temperature": 0}]
    # The first query text matches official verbatim: keyframe field is a literal [], execution-frame field has 1 image
    assert user_text(sw.requests[0]) == user_text(sw_o.requests[0])
    assert "importance:[]\nHere is current input image list from the front-view camera: [<image>]" in \
        user_text(sw.requests[0])


def case_b(tmp):
    """B: non-empty memory and a valid reply (including keyframe picks) -- the compat layer matches the official
    function in return value, memory and log bytes."""
    reply = '{"current_subtask": "pick up the cube at <|box_start|>(500,250)<|box_end|>", "keyframe_positions": [2, 4]}'
    outs = []
    for compat in (False, True):
        sw = F.FakeSwift(script=[reply])
        m = new_model(tmp / f"b{int(compat)}", sw, compat=compat, frames=20)
        m.key_frame_paths = {3: m.execution_frame_paths[3]}
        sub = m.call()
        outs.append((sub, {k: Path(v).name for k, v in m.key_frame_paths.items()},
                     Path(m.save_json_path).read_bytes().replace(str(tmp / f"b{int(compat)}").encode(), b"<T>"),
                     sw.requests))
    assert outs[0] == outs[1]
    # Hand-written: from 20 frames take 19,17,...,5, 8 in total (ascending odd 5..19); position 2->7, 4->11; merged with
    # the existing 3: 3,7,11 are pairwise <=8 apart, so one group keeps the median 7
    assert outs[1][0] == "pick up the cube at <128, 64>" and outs[1][1] == {7: "step_7_image.png"}


def _with_prior(tmp, script, name):
    """First one valid query (subgoal "first"), then add 4 frames; the second query is answered per script."""
    sw = F.FakeSwift(script=['{"current_subtask": "first", "keyframe_positions": []}', *script])
    m = new_model(tmp / name, sw)
    assert m.call() == "first"
    for i in range(4):
        m.add_execution_frame(F.frame(50 + i))
    return m, sw


def _check_retry_requests(sw, first_idx):
    """Second and third requests: reminder sentence appended to the user prompt, temperature=0.7; system prompt and
    attached images match the first query."""
    a, b, c = sw.requests[first_idx:first_idx + 3]
    assert a["config"] == {"max_tokens": 128, "temperature": 0}
    for r in (b, c):
        assert r["config"] == {"max_tokens": 128, "temperature": 0.7}
        assert user_text(r) == user_text(a) + "\n" + NOTE and r["messages"][0] == a["messages"][0]
        assert r["image_shas"] == a["image_shas"]


def _retry_log(m) -> list:
    return [(r.get("retry"), "response" in r) for r in log_rows(m) if r.get("retry")]


def case_c1(tmp):
    m, sw = _with_prior(tmp, [BAD, "{}", '{"current_subtask": "third", "keyframe_positions": [1]}'], "c1")
    assert m.call() == "third" and m.subgoals == ["first", "third"] and m._memer_fallback is None
    _check_retry_requests(sw, 1)
    assert _retry_log(m) == [(1, False), (1, True), (2, False), (2, True)]
    assert len(m.execution_frame_paths) == 5  # retries do not re-append execution frames


def case_c2(tmp):
    m, sw = _with_prior(tmp, [BAD, BAD, '{"current_subtask": ""}'], "c2")
    assert m.call() == "first" and m.subgoals == ["first"] and m._memer_fallback == "last_valid"
    _check_retry_requests(sw, 1)
    assert _retry_log(m) == [(1, False), (1, True), (2, False), (2, True)]
    last = log_rows(m)[-1]
    assert last["fallback_used"] == 1 and last["fallback"] == "last_valid" and last["subgoal"] == "first"


def case_d1(tmp):
    sw = F.FakeSwift(script=[BAD, BAD, GOOD])
    m = new_model(tmp / "d1", sw)
    assert m.call() == "move cube" and m.subgoals == ["move cube"]
    _check_retry_requests(sw, 0)
    assert _retry_log(m) == [(1, False), (1, True), (2, False), (2, True)]


def case_d2(tmp):
    sw = F.FakeSwift(script=[BAD, BAD, BAD])
    m = new_model(tmp / "d2", sw)
    with pytest.raises(od().MemERResponseError) as ei:
        m.call()
    assert type(ei.value).__name__ == "MemERResponseError" and ei.value.error_kind == "model_response_error"
    assert m.subgoals == [] and m.key_frame_paths == {} and m._memer_fallback == "model_response_error"
    _check_retry_requests(sw, 0)
    assert _retry_log(m) == [(1, False), (1, True), (2, False), (2, True)]
    assert log_rows(m)[-1]["fallback"] == "model_response_error"


def case_e(tmp):
    """E: missing keyframe_positions key -- official raises KeyError, then IndexError in its fallback; the compat
    layer retries."""
    sw_o = F.FakeSwift(script=['{"current_subtask": "x"}'])
    off = new_model(tmp / "eo", sw_o, compat=False)
    with pytest.raises(IndexError):
        off.call()
    m, sw = _with_prior(tmp, ['{"current_subtask": "x"}', '{"current_subtask": "y", "keyframe_positions": []}'], "e")
    assert m.call() == "y" and m.subgoals == ["first", "y"]
    assert [r["config"]["temperature"] for r in sw.requests] == [0, 0, 0.7]


def case_f(tmp):
    """F: 1 / 5 / 14 / 15 / 40 execution frames -- with fewer than 15, take every other frame down to the first
    (ascending); 1 frame and >=15 frames match the official function."""
    want = {1: [0], 5: [0, 2, 4], 14: [1, 3, 5, 7, 9, 11, 13], 15: [0, 2, 4, 6, 8, 10, 12, 14],
            40: [25, 27, 29, 31, 33, 35, 37, 39]}  # hand-written (frame index from 0 = frame n+1)
    sw = F.FakeSwift()
    for n, ids in want.items():
        m = new_model(tmp / "f", sw, frames=n, name=f"f{n}")
        assert frame_ids(m._get_current_execution_frame_paths()) == ids, n
        off = new_model(tmp / "fo", sw, compat=False, frames=n, name=f"f{n}")
        if n == 1 or n >= 15:
            assert m._get_current_execution_frame_paths() == m._official_get_current_execution_frame_paths()
            assert frame_ids(off._get_current_execution_frame_paths()) == ids
        else:
            with pytest.raises(IndexError):
                off._get_current_execution_frame_paths()
    # Only 5 frames at the second query: the query proceeds as usual, attached images = the 3 most recent frames
    sw2 = F.FakeSwift(script=[GOOD, GOOD])
    m = new_model(tmp / "f2", sw2)
    m.call()
    for i in range(4):
        m.add_execution_frame(F.frame(70 + i))
    assert m.call() == "move cube"
    assert len(sw2.requests[1]["image_shas"]) == 3 and user_text(sw2.requests[1]).endswith(
        "[<image>, <image>, <image>]\n\nWhat subtask should the robot execute and what is the keyframe position?")


CASES = {"A": case_a, "B": case_b, "C1": case_c1, "C2": case_c2, "D1": case_d1, "D2": case_d2, "E": case_e,
         "F": case_f}


def test_memer_compat_cases(tmp_path):
    F.print_official_sha()
    mod = od()
    for name, fn in CASES.items():
        fn(tmp_path / name)
    fp = mod.MEMER_COMPAT_SHA256
    import hashlib

    assert fp == hashlib.sha256(mod.MEMER_COMPAT_SOURCE.encode("utf-8")).hexdigest()
    # Official file is untouched: the extracted namespace records the sha256 of the whole official file
    src = F.official_dir() / "subgoal_prediction" / "qwenvl" / "api_memer.py"
    ns = mod.load_memer_model(swift_names=F.FakeSwift().names)
    assert ns["__source_sha256__"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert ns["__memer_compat_sha256__"] == fp
    print(f"MEMER_COMPAT=PASS cases={len(CASES)} fingerprint={fp}")


# ---------------------------------------------------------------- atomic validation counterexamples


BAD_REPLIES = {
    "out_of_range": '{"current_subtask": "a", "keyframe_positions": [1, 2]}',
    "zero": '{"current_subtask": "a", "keyframe_positions": [0]}',
    "negative": '{"current_subtask": "a", "keyframe_positions": [-1]}',
    "bool": '{"current_subtask": "a", "keyframe_positions": [true]}',
    "float": '{"current_subtask": "a", "keyframe_positions": [1.0]}',
    "missing_subtask": '{"keyframe_positions": []}',
    "non_string": '{"current_subtask": 123, "keyframe_positions": []}',
    "empty_string": '{"current_subtask": "  ", "keyframe_positions": []}',
    "not_list": '{"current_subtask": "a", "keyframe_positions": 1}',
    "not_object": '["a"]',
}


@pytest.mark.parametrize("name", sorted(BAD_REPLIES))
def test_atomic_validation_rejects_without_touching_state(tmp_path, name):
    """Bad reply: ``update_history_subgoals`` raises and keyframes, history and execution frames are all unchanged
    (for [1,2] the official original writes the first image before going out of range)."""
    sw = F.FakeSwift()
    m = new_model(tmp_path, sw, frames=1)
    m.current_execution_frame_paths = m._get_current_execution_frame_paths()
    m.key_frame_paths, m.subgoals = {}, ["prev"]
    before = (dict(m.key_frame_paths), list(m.subgoals), list(m.execution_frame_paths))
    with pytest.raises(Exception):
        m.update_history_subgoals(BAD_REPLIES[name])
    assert (m.key_frame_paths, m.subgoals, m.execution_frame_paths) == before
    if name == "out_of_range":  # official original: image 1 is written to memory before going out of range -- state is polluted
        off = new_model(tmp_path / "o", sw, compat=False, frames=1)
        off.current_execution_frame_paths = off._get_current_execution_frame_paths()
        with pytest.raises(IndexError):
            off.update_history_subgoals(BAD_REPLIES[name])
        assert list(off.key_frame_paths) == [0]


def test_conversion_failure_is_atomic(tmp_path):
    sw = F.FakeSwift()
    m = new_model(tmp_path, sw, frames=3)
    m.current_execution_frame_paths = m._get_current_execution_frame_paths()

    def boom(*a, **kw):
        raise ValueError("conversion failed")

    m._parse_box_patterns = boom
    with pytest.raises(ValueError):
        m.update_history_subgoals('{"current_subtask": "a", "keyframe_positions": [1]}')
    assert m.key_frame_paths == {} and m.subgoals == []


def test_bad_then_good_retry_keeps_first_request_memory_and_images(tmp_path):
    """First reply [1,2] (only 1 image, out of range), then a valid reply: the second request has the same memory and
    attached images as the first; memory is not polluted."""
    sw = F.FakeSwift(script=[BAD_REPLIES["out_of_range"], '{"current_subtask": "ok", "keyframe_positions": [1]}'])
    m = new_model(tmp_path, sw, frames=1)
    assert m.call() == "ok"
    a, b = sw.requests
    assert a["image_shas"] == b["image_shas"] and len(a["image_shas"]) == 1
    assert user_text(b) == user_text(a) + "\n" + NOTE
    assert list(m.key_frame_paths) == [0] and m.subgoals == ["ok"]


def test_merge_empty_memory_and_regression(tmp_path):
    """One call with empty memory does not raise; with frames 3, 5, 20 it matches the official function (3 and 5 merge
    into one group, keeping only 5)."""
    sw = F.FakeSwift()
    m = new_model(tmp_path, sw)
    off = new_model(tmp_path / "o", sw, compat=False)
    m.key_frame_paths = {}
    m.merge_key_frame_paths()
    assert m.key_frame_paths == {}
    off.key_frame_paths = {}
    with pytest.raises(IndexError):
        off.merge_key_frame_paths()
    for obj in (m, off):
        obj.key_frame_paths = {20: "p20", 3: "p3", 5: "p5"}
        obj.merge_key_frame_paths()
    assert m.key_frame_paths == off.key_frame_paths == {5: "p5", 20: "p20"}


def test_patch_refuses_changed_upstream(tmp_path):
    """If the upstream class lacks a patched method, the patch refuses to apply (KeyError) instead of wiring silently."""
    import ast

    mod = od()
    node = ast.parse("class Qwen3VLModelMemER:\n    def call(self):\n        return 1\n").body[0]
    with pytest.raises(KeyError):
        mod.patch_memer_class(node)


# ---------------------------------------------------------------- wiring and seed


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("IMAGE_MAX_TOKEN_NUM", "VIDEO_MAX_TOKEN_NUM", "FPS_MAX_FRAMES", "USE_HF", "HF_HUB_OFFLINE",
              "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_memer_wiring(tmp_path, clean_env):
    """MemER variant: extracts only MemERSubgoalPredictor + Qwen3VLModelMemER with the compat layer; engine args come
    verbatim from official code; Args has exactly use_memer; one episode succeeds, the log is archived and the temp
    area is cleared."""
    import os
    import sys

    side = F.NewSide(F.MEMER, 60, tmp_path, F.World(default=F.Plan(success_at=37)))
    defs, args, pred = side.ctx["defs"], side.ctx["args"], side.ctx["predictor"]
    names = set(defs["predictor"])
    assert "MemERSubgoalPredictor" in names
    assert not names & {"QwenVLSubgoalPredictor", "OracleSubgoalPredictor", "GeminiSubgoalPredictor",
                        "Qwen3VLModel"}
    assert type(pred).__name__ == "MemERSubgoalPredictor" and type(pred.api).__name__ == "Qwen3VLModelMemER"
    assert hasattr(type(pred.api), "_official_call") and defs["memer_compat_sha256"] == od().MEMER_COMPAT_SHA256
    assert side.swift.engines == [{"model_id_or_path": "Qwen/Qwen3-VL-4B-Instruct", "adapters": [F.MEMER_ADAPTER],
                                   "attn_impl": "flash_attention_2"}]
    assert (args.use_oracle, args.use_qwenvl, args.use_memer, args.use_gemini) == (False, False, True, False)
    assert args.subgoal_type == "grounded_subgoal" and args.memer_adapter_path == F.MEMER_ADAPTER
    assert args.model_seed == F.POLICY_SEED
    assert os.environ["USE_HF"] == "1" and os.environ["IMAGE_MAX_TOKEN_NUM"] == "256"
    assert "swift" not in sys.modules and "google.generativeai" not in sys.modules
    src = "subgoal_prediction/qwenvl/api_memer.py"
    assert src in defs["sha256"] and "subgoal_prediction/qwenvl/api.py" not in defs["sha256"]
    res = side.run(F.identity())
    assert (res["status"], res["steps"], res["policy_variant"], res["policy_seed"]) == \
        ("success", 37, F.MEMER, F.POLICY_SEED)
    assert res["memer_compat_sha256"] == od().MEMER_COMPAT_SHA256 and res["error_kind"] is None
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "memer-tmp").exists() and (tdir / "ep3a1_MemER_log.jsonl").is_file()
    assert len(side.swift.requests) == 3  # decisions at steps 0, 16, 32
    print(f"MEMER_WIRING=PASS predictor={type(pred).__name__}")


def test_memer_model_response_error_is_named_error_and_cleans_up(tmp_path, clean_env):
    """D2 at episode level: three bad replies and no previous one -> status=error, error_kind=model_response_error,
    not infra, no step run; temp area cleared, log archived with the two retry rows, official video salvaged as
    error; the next episode in the same context runs normally."""
    swift = F.FakeSwift(script=[BAD, BAD, BAD])
    side = F.NewSide(F.MEMER, 60, tmp_path, F.World(default=F.Plan(success_at=20)), swift=swift)
    res = side.run(F.identity())
    assert (res["status"], res["infra"], res["error_kind"], res["exception"], res["steps"]) == \
        ("error", False, "model_response_error", "MemERResponseError", 0)
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "memer-tmp").exists() and not (tdir / "official-video").exists()
    rows = [json.loads(x) for x in (tdir / "ep3a1_MemER_log.jsonl").read_text().splitlines()]
    assert [r.get("retry") for r in rows if "response" in r] == [None, 1, 2]
    assert rows[-1]["fallback"] == "model_response_error"
    end = F.read_trace(res["trace_path"])[-1]
    assert (end["status"], end["terminal_reason"]) == ("error", "error")
    assert res["official_source"] == "official-salvaged"
    r2 = side.run(F.identity(source_episode=7, builder_episode=1, seed=510700))
    assert r2["status"] == "success" and r2["error_kind"] is None


def test_engine_exception_cleans_memer_tmp(tmp_path, clean_env):
    swift = F.FakeSwift()
    side = F.NewSide(F.MEMER, 60, tmp_path, F.World(default=F.Plan(success_at=20)), swift=swift)

    def boom(*a, **kw):
        raise RuntimeError("engine exploded")

    side.ctx["predictor"].api.engine.infer = boom
    res = side.run(F.identity())
    assert res["status"] == "error" and res["error"].startswith("RuntimeError: engine exploded")
    assert res["error_kind"] is None
    tdir = Path(res["trace_path"]).parent
    assert not (tdir / "memer-tmp").exists()
