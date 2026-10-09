"""C13 两个策略客户端（``framesamp_modul_client``、``smvla_client``）的协议与动作转换，以及席位层「普通失败不重试、唯一接受、
必填身份」。

假 server 只记录收到的消息并按输入确定性地回动作；期望（帧序列、exec_start_idx、执行的动作行、步数）都由本文件
手写的假环境推出，不调用被测函数生成。
"""
from __future__ import annotations

import collections
import hashlib
import json
import threading

import numpy as np
import pytest

import eval_fakes as F
from tests._support.loaders import load_script
from tests.pipeline.evalx.report import trace_contract as tc


# ---------------------------------------------------------------- FrameSamp+Modulation：逐行照抄旧官方的循环语义


class _Session:
    """最小会话：reset 返回假环境 reset 观测；step 交给假环境。"""

    def __init__(self, plan, ep=2):
        self.env = F.FakeEnv("T", ep, plan)
        self.steps = 0

    def reset(self):
        return self.env.reset()

    def step(self, a):
        self.steps += 1
        return self.env.step(a)


def _framesamp_modul_run(plan, *, max_steps=None, server=None, client_factory=None):
    mc = F.framesamp_modul_client()
    server = server or F.FakePolicyServer()
    sess = _Session(plan)
    kw = {} if max_steps is None else {"max_steps": max_steps}
    res = mc.evaluate_one(client_factory or (lambda: F.FakeMMEVLAWebsocketClient(server)), sess.step,
                          lambda: mc.pre_traj_from_reset(*sess.reset()), **kw)
    return res, sess, server


def test_pack_state_is_joint7_plus_first_gripper_float32():
    mc = F.framesamp_modul_client()
    s = mc.pack_state(np.arange(7, dtype=np.float64), np.array([0.25, 0.75]))
    assert s.dtype == np.float32 and s.tolist() == [0, 1, 2, 3, 4, 5, 6, 0.25]


def test_framesamp_modul_first_inference_sends_all_reset_frames_then_only_new_frames():
    mc = F.framesamp_modul_client()
    res, sess, server = _framesamp_modul_run(F.Plan(success_at=20))
    assert res["status"] == "success" and res["steps"] == 20 and res["infra"] is False
    obs_msgs = [p for k, p in server.log if k == "observe"]
    infer_msgs = [p for k, p in server.log if k == "infer"]
    # 第一次决策：reset 的全部帧（演示 + 初始），exec_start_idx 指向初始帧
    reset_shas = [F.sha_bytes(F.frame(v)) for v in F.reset_values(2)]
    assert obs_msgs[0]["frames"] == reset_shas and obs_msgs[0]["exec_start_idx"] == F.N_RESET_FRAMES - 1
    assert obs_msgs[0]["shape"] == [F.N_RESET_FRAMES, 1, F.HW, F.HW, 3]
    assert infer_msgs[0]["prompt"] == "goal-T-2"  # 取 task_goal 的第一条
    assert infer_msgs[0]["image"] == reset_shas[-1]
    assert np.array_equal(infer_msgs[0]["state"], mc.pack_state(np.full(7, F.reset_values(2)[-1] / 10.0),
                                                                np.full(2, F.reset_values(2)[-1] / 100.0)))
    # 第二次决策：只带执行段新出现的帧，exec_start_idx 归零
    h = mc.OBS_HORIZON
    assert len(obs_msgs[1]["frames"]) == h and obs_msgs[1]["exec_start_idx"] == 0
    assert obs_msgs[1]["frames"][0] == F.sha_bytes(F.frame(100 + (2 * 3 + 1) % 100))
    # 每次推理回 CHUNK_ROWS 行，只执行前 OBS_HORIZON 行
    assert F.CHUNK_ROWS > h and res["decisions"] == 2
    executed = np.stack(sess.env.actions)
    assert executed.shape == (20, 8)
    assert np.array_equal(executed[:h], infer_msgs[0]["actions"][:h])
    assert np.array_equal(executed[h:], infer_msgs[1]["actions"][:20 - h])


def test_framesamp_modul_timeout_counts_to_max_plus_one():
    """count > max_steps 即 timeout，步数记 max_steps+1（该判断先于终态判断）。"""
    res, sess, _ = _framesamp_modul_run(F.Plan(), max_steps=5)
    assert res["status"] == "timeout" and res["steps"] == 6 and sess.env.n == 6


def test_framesamp_modul_env_exception_becomes_error_without_infra():
    res, sess, _ = _framesamp_modul_run(F.Plan(raise_at=3, raise_exc=lambda: RuntimeError("IK 失败")))
    assert res["status"] == "error" and res["infra"] is False
    assert res["env_exception"] == "RuntimeError: IK 失败" and res["steps"] == 3


@pytest.mark.parametrize("message", ["svulkan2 boom", "CUDA_ERROR_LAUNCH_FAILED", "out of memory"])
def test_framesamp_modul_infra_marker_sets_infra(message):
    res, _, _ = _framesamp_modul_run(F.Plan(raise_at=1, raise_exc=lambda: RuntimeError(message)))
    assert res["status"] == "error" and res["infra"] is True


def test_framesamp_modul_unknown_terminal_status_is_error():
    class _Weird(F.FakeEnv):
        def step(self, a):
            obs, r, _, tr, _ = super().step(a)
            return obs, r, True, tr, {"status": "ongoing"}

    mc = F.framesamp_modul_client()
    env = _Weird("T", 2, F.Plan())
    res = mc.evaluate_one(lambda: F.FakeMMEVLAWebsocketClient(F.FakePolicyServer()), env.step,
                          lambda: mc.pre_traj_from_reset(*env.reset()))
    assert res["status"] == "error" and res["error"] == "success_flag=ongoing"


def test_framesamp_modul_connection_refused_is_infra():
    def factory():
        raise ConnectionRefusedError("拒绝连接")

    res, sess, _ = _framesamp_modul_run(F.Plan(success_at=1), client_factory=factory)
    assert res["status"] == "error" and res["infra"] is True and sess.env.n == 0


# ---------------------------------------------------------------- SMVLA：SimEnvService 的打包语义


def test_step_chunk_converts_rows_to_float64_first8_and_stops_on_done():
    sm = F.smvla_client()
    sess = _Session(F.Plan(success_at=2))
    sess.reset()
    chunk = np.arange(40, dtype=np.float32).reshape(4, 10)
    out = sm.step_chunk(sess, chunk)
    assert out["consumed"] == 2 and out["done"] is True and out["success"] is True
    assert [a.dtype for a in sess.env.actions] == [np.float64] * 2
    assert [a.tolist() for a in sess.env.actions] == [list(range(0, 8)), list(range(10, 18))]
    assert len(out["frames"]) == 2 and out["states"][0].dtype == np.float32


def test_step_chunk_rejects_short_action():
    sm = F.smvla_client()
    sess = _Session(F.Plan())
    sess.reset()
    with pytest.raises(ValueError):
        sm.step_chunk(sess, np.zeros((2, 7)))
    assert sess.env.n == 0


def _smvla_run(plan, **kw):
    sm = F.smvla_client()
    server = F.FakePolicyServer(fail_on=kw.pop("fail_on", None))
    conn = F.FakeSmvlaConn(server, tamper=kw.pop("tamper", None))
    sess = _Session(plan)
    res = sm.run_episode(sess, {"task": "T", "source_episode": None, "seed": 1}, {}, None, conn=conn, **kw)
    return res, sess, server


def test_smvla_first_inference_and_messages():
    res, sess, server = _smvla_run(F.Plan(success_at=3))
    assert res["status"] == "success" and res["steps"] == 3 and res["decisions"] == 1
    assert server.kinds()[:3] == ["reset", "observe", "infer"]
    assert server.log[0][1] == "T/None/1"
    reset_shas = [F.sha_bytes(F.frame(v)) for v in F.reset_values(2)]
    assert server.log[1][1]["frames"] == reset_shas
    inf = server.log[2][1]
    assert inf["prompt"] == "goal-T-2" and inf["state"].dtype == np.float32
    v = F.reset_values(2)[-1]
    assert inf["state"].tolist() == pytest.approx([v / 10.0] * 7 + [v / 100.0])
    assert res["protocol"]["sha_mismatch"] == 0 and res["protocol"]["frames_sent"] == F.N_RESET_FRAMES + 3


def test_smvla_decision_budget_hand_computed():
    """max_steps=32、执行段 16：决策上限 32/16+2 = 4 次（手算），永不终止的环境走满 4×16 = 64 步后记 timeout。"""
    res, sess, _ = _smvla_run(F.Plan(), max_steps=32, execute_horizon=16)
    assert res["hard_bound"] == 4 and res["decisions"] == 4
    assert res["status"] == "timeout" and res["steps"] == 64 and sess.env.n == 64
    assert "hard_bound=4" in res["error"]


@pytest.mark.parametrize("fail_on,tamper,reason", [("server_error", None, "server_error"),
                                                    (None, "req_sha", "protocol"),
                                                    (None, "frame_sha", "protocol"),
                                                    ("infer_disconnect", None, "connection:ConnectionClosed")])
def test_smvla_server_side_failures_are_infra(fail_on, tamper, reason):
    res, sess, _ = _smvla_run(F.Plan(success_at=3), fail_on=fail_on, tamper=tamper)
    assert res["status"] == "error" and res["infra"] is True and res["infra_reason"] == reason
    assert sess.env.n == 0


def test_smvla_step_exception_without_marker_is_final_error():
    res, sess, _ = _smvla_run(F.Plan(raise_at=2, raise_exc=lambda: RuntimeError("IK 失败")))
    assert res["status"] == "error" and res["infra"] is False and res["error"].startswith("step_exc: ")


def test_smvla_reset_failure_with_vulkan_marker_is_infra_and_retries_count():
    class _Bad(_Session):
        def __init__(self):
            super().__init__(F.Plan())
            self.n_reset = 0
            self.closed = 0

        def reset(self):
            self.n_reset += 1
            raise RuntimeError("vk::DeviceLost")

        def close(self):
            self.closed += 1

    sm = F.smvla_client()
    sess = _Bad()
    res = sm.run_episode(sess, {"task": "T", "source_episode": 0, "seed": 1}, {}, None,
                         conn=F.FakeSmvlaConn(F.FakePolicyServer()), reset_retries=2)
    assert sess.n_reset == 3 and sess.closed == 3  # 2 次重试 = 共 3 次
    assert res["status"] == "error" and res["infra"] is True and res["infra_reason"] == "env_reset:vk::"


# ---------------------------------------------------------------- 席位层：普通失败不重试、必填身份
#
# 拆仓后席位层是 ``dev-scripts/gl/seat.py``（常驻 Policy + 动态队列）：账本在 ``seats/<标签>/seed<n>/<席位>/``，
# 身份清单坏行抛 ``SeatStop(EXIT_BLOCKED)``（由 ``cmd_run`` 转成退出码 3），入口拦截改为 ``entry_blockers``。


@pytest.mark.parametrize("plan", [F.Plan(fail_at=1), F.Plan(raise_at=1, raise_exc=lambda: RuntimeError("IK"))],
                         ids=["fail", "ordinary_error"])
@pytest.mark.parametrize("policy", ("perceptual-framesamp-modul", "smvla"))
def test_ordinary_failure_is_not_retried(tmp_path, monkeypatch, policy, plan):
    task, tier = F.v9_cells_sorted()[0]
    ident = F.packaged_identity(task, tier, 0)
    world = F.World({(task, ident["builder_episode"]): [plan]})
    runner = F.make_runner(tmp_path, policy, F.policy_module(policy, monkeypatch, F.FakePolicyServer()), world)
    assert F.run_rows(runner, [ident]) == 0
    assert len(world.envs) == 1
    ledger = F.seat_ledger(tmp_path, policy)
    assert [x["kind"] for x in ledger].count("attempt_start") == 1
    assert [x["kind"] for x in ledger].count("accept") == 1
    # 队列里该身份恰有一次领取、一份 accepted 标记（跨席位的唯一接受）
    q = F.queue_dir(tmp_path, policy)
    assert len(list((q / "claims").glob("*.json"))) == 1 and len(list((q / "accepted").glob("*.json"))) == 1


def _args_for_identities(tmp_path, rows, **kw):
    p = tmp_path / "shard-00.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return F.seat_args(tmp_path / "out", "perceptual-framesamp-modul", identities=str(p), **kw)


@pytest.mark.parametrize("mutate", ["drop_spec", "dup_key", "float_seed"])
def test_load_identities_blocks_on_bad_rows(tmp_path, capsys, mutate):
    ec = F.env_client()
    task, tier = F.v9_cells_sorted()[0]
    a, b = F.packaged_identity(task, tier, 0), F.packaged_identity(task, tier, 1)
    rows = [a, b]
    if mutate == "drop_spec":
        rows = [a, {k: v for k, v in b.items() if k != "spec_sha256"}]
    elif mutate == "dup_key":
        rows = [a, dict(a)]
    else:
        rows = [a, dict(b, seed=float(b["seed"]))]
    with pytest.raises(ec.SeatStop) as ei:
        ec.load_identities(_args_for_identities(tmp_path, rows))
    assert ei.value.code == ec.EXIT_BLOCKED
    assert "RUN_BLOCKED reason=identities" in capsys.readouterr().out


def test_load_identities_accepts_clean_rows_and_only_filter(tmp_path):
    ec = F.env_client()
    task, tier = F.v9_cells_sorted()[0]
    a, b = F.packaged_identity(task, tier, 0), F.packaged_identity(task, tier, 1)
    args = _args_for_identities(tmp_path, [a, b])
    assert [r["key"] for r in ec.load_identities(args)] == [a["key"], b["key"]]
    args.only = b["key"]
    assert [r["key"] for r in ec.load_identities(args)] == [b["key"]]


def test_dataset_has_no_default_and_max_steps_follow_dataset(tmp_path, capsys):
    """数据集无缺省：身份行不带 ``dataset`` 且不给 ``--dataset`` 即 RUN_BLOCKED（不会默默落到某一档）；给了 ``--dataset``
    才补进行内。步数上限不再是 CLI 参数，由数据集决定（hard-verify 1300 不截断、ood 1800 严格截断，手写常量）。"""
    ec = F.env_client()
    task, tier = F.v9_cells_sorted()[0]
    a = {k: v for k, v in F.packaged_identity(task, tier, 0).items() if k != "dataset"}
    with pytest.raises(ec.SeatStop) as ei:
        ec.load_identities(_args_for_identities(tmp_path, [a]))
    assert ei.value.code == ec.EXIT_BLOCKED and "RUN_BLOCKED reason=identities" in capsys.readouterr().out
    assert [r["dataset"] for r in ec.load_identities(_args_for_identities(tmp_path, [a], dataset="ood"))] == ["ood"]
    E = F.episode_mod()
    assert E.DATASET_MAX_STEPS == {"hard-verify": F.HARD0_MAX_STEPS, "ood": F.V9_MAX_STEPS}
    with pytest.raises(SystemExit) as ei:  # --dataset 只认 ood／hard-verify 两个名字（旧名 test 已退场）
        ec.build_parser().parse_args(["run", "--policy", "dummy", "--identities", "x", "--out", "o", "--dataset", "test"])
    assert ei.value.code == 2


_BUDGET = ["--budget-ledger", "l.jsonl", "--trajectory-cap", "10", "--reset-cap", "20", "--shared-infra-cap", "1",
           "--expired-cap", "1", "--planned-first-tries", "10"]
_BASE = ["run", "--identities", "x.json", "--seat", "s", "--out", "o", "--policy-seed", "0"]


@pytest.mark.parametrize("extra,reason,why", [
    (["--policy", "perceptual-framesamp-modul"], "budget_args", "--budget-ledger"),
    (["--policy", "groundsg", *_BUDGET], "variant_pairing", "--groundsg-variant"),
    (["--policy", "groundsg", "--groundsg-variant", "ground-sg-qwenvl", *_BUDGET], "variant_pairing",
     "--qwenvl-groundsg-adapter"),
    (["--policy", "perceptual-framesamp-modul", "--groundsg-variant", "ground-sg-oracle", *_BUDGET], "variant_pairing",
     "只能与 --policy groundsg"),
    (["--policy", "groundsg", "--groundsg-variant", "ground-sg-oracle", "--qwenvl-groundsg-adapter", "a", *_BUDGET],
     "variant_pairing", "只能与 --groundsg-variant ground-sg-qwenvl"),
    (["--policy", "pp", "--infra-retries", "-1", *_BUDGET], "args", "--infra-retries 须为非负整数"),
], ids=["no_ledger", "groundsg_no_variant", "qwenvl_no_adapter", "variant_on_framesamp_modul", "adapter_on_oracle",
        "negative_retries"])
def test_run_args_blocked(capsys, extra, reason, why):
    ec = F.env_client()
    assert ec.main(_BASE + extra) == ec.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert f"RUN_BLOCKED reason={reason}" in out and why in out


def test_run_args_accept_all_four_policies(tmp_path):
    ec = F.env_client()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    for pol, extra in (("perceptual-framesamp-modul", []), ("smvla", []), ("pp", []),
                       ("groundsg", ["--groundsg-variant", "ground-sg-oracle"]),
                       ("groundsg", ["--groundsg-variant", "ground-sg-qwenvl", "--qwenvl-groundsg-adapter",
                                     str(adapter)])):
        args = ec.build_parser().parse_args(_BASE + ["--policy", pol, *_BUDGET, *extra])
        assert ec.entry_blockers(args) is None, pol
        assert ec.variant_problems(args, check_dirs=True) == [], pol


# ---------------------------------------------------------------- S4：两条新侧路线的逐步轨迹（契约 C1～C11）
#
# 期望一律由假环境的计划手算（步数、帧数、终态）；契约判据交 S0 的 trace_contract 助手。
# 环境会话用真实 robomme_ood_eval.session.EnvSession（假 builder），步数口径（异常步计步、strict-cap 不进环境）与生产相同。

IDENT = {"task": "T", "tier": "xhard0", "seed": 1, "source_episode": 2, "builder_episode": 2, "key": "T_xhard0_1"}


class _B:
    def __init__(self, plan):
        self.plan = plan
        self.env = None

    def make_env_for_episode(self, ep, max_steps=None):
        self.env = F.FakeEnv("T", ep, self.plan)
        return self.env


def _env_session(plan, *, cap=None, recorder=None):
    b = _B(plan)
    return F.env_session().EnvSession("T", 2, recorder=recorder, builder=b, step_cap=cap), b


def _conn(tmp_path, attempt=1, **kw):
    tag = f"{IDENT['key']}.a{attempt}"
    return dict({"trace_dir": str(tmp_path / tag), "episode_tag": tag, "dataset": "hard-verify"}, **kw), tmp_path / tag


def _rows(ep):
    return [json.loads(x) for x in (ep / "trace.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _kind(rows, k):
    return [r for r in rows if r["kind"] == k]


def _tw():
    return load_script("eval-official/trace_writer.py")


def _norm(x):
    """把消息载荷里的数组换成 (dtype, shape, bytes)，便于逐项比较两次运行。"""
    if isinstance(x, np.ndarray):
        return ("nd", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    return x


def _strip(res):
    return {k: v for k, v in res.items() if k not in ("timing", "trace_path")}


def _smvla_traced(tmp_path, plan, *, cap=None, traced=True, recorder=None, **kw):
    sm = F.smvla_client()
    server = F.FakePolicyServer()
    sess, b = _env_session(plan, cap=cap, recorder=recorder)
    conn_info, ep = _conn(tmp_path) if traced else ({}, None)
    res = sm.run_episode(sess, dict(IDENT), conn_info, recorder, conn=F.FakeSmvlaConn(server), **kw)
    return res, sess, b, server, ep


def test_smvla_trace_success_is_renderable_with_subgoal_and_logical_requests(tmp_path):
    """success_at=20、执行段 16：决策 2 次，20 步全有观测；动作 float64 → 轨迹目录写 arrays.npz。"""
    res, sess, b, server, ep = _smvla_traced(tmp_path, F.Plan(success_at=20))
    assert res["status"] == "success" and sess.steps == 20 and res["trace_path"] == str(ep / "trace.jsonl")
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": res["status"]})
    rows = _rows(ep)
    header, end, demo = rows[0], rows[-1], _kind(rows, "demo")[0]
    assert header["route"] == "smvla/new"
    assert {k: header["identity"][k] for k in IDENT} == IDENT
    assert header["identity"]["attempt"] == 1 and header["identity"]["dataset"] == "hard-verify"
    assert header["max_steps"] == res["hard_bound"] * 16  # 客户端循环的真实步数上界
    steps = _kind(rows, "step")
    assert len(steps) == sess.steps == 20  # 步数行 = 执行步数
    assert all(st["subgoal"] == "s" for st in steps)  # 子目标取回包 subtask，非空
    assert demo["frames"] == F.N_RESET_FRAMES and end["demo_frames"] == F.N_RESET_FRAMES - 1
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (20, 20, F.N_RESET_FRAMES + 20)
    # 第三阶段（冻结说明四.3）：end.arrays 为摘要，另收观测步状态 exec_state__%05d
    assert end["arrays"] == {"path": "arrays.npz", "action_keys": 20, "state_keys": 20, "missing_state_steps": []}
    assert end["observer_hook_errors"] == 0 and end["request_encoding"] == "logical"
    # 动作按实际交给环境的原 dtype 记录，arrays.npz 原值与环境收到的逐字节相同
    with np.load(ep / "arrays.npz") as arr:
        assert sorted(k for k in arr.files if k.startswith("exec_action__")) == \
            [f"exec_action__{i:05d}" for i in range(20)]
        assert sorted(k for k in arr.files if k.startswith("exec_state__")) == \
            [f"exec_state__{i:05d}" for i in range(20)]
        for i, a in enumerate(b.env.actions):
            assert arr[f"exec_action__{i:05d}"].dtype == np.float64
            assert arr[f"exec_action__{i:05d}"].tobytes() == a.tobytes()
    # C10：请求行 = 逻辑输入（指令、状态、帧哈希序列）；响应行 = 完整动作块
    tw = _tw()
    reqs, reps = _kind(rows, "request"), _kind(rows, "response")
    assert [r["name"] for r in reqs] == ["infer", "infer"] and [r["step"] for r in reqs] == [0, 16]
    vals = F.reset_values(2)
    v = vals[-1]
    state0 = np.array([v / 10.0] * 7 + [v / 100.0], dtype=np.float32)
    frames0 = [[tw.image_sha256(F.frame(x)), tw.image_sha256(F.frame(x + 1))] for x in vals]
    expect0 = tw.bytes_record(tw.canonical_bytes({"instruction": "goal-T-2", "state": state0, "frames": frames0}))
    assert (reqs[0]["sha256"], reqs[0]["nbytes"]) == (expect0["sha256"], expect0["nbytes"])
    infer_actions = [p["actions"] for k, p in server.log if k == "infer"]
    assert [r["actions"]["sha256"] for r in reps] == [tw.array_record(a)["sha256"] for a in infer_actions]


def test_smvla_trace_does_not_change_requests_or_actions(tmp_path):
    """有无轨迹两次运行：发给服务的消息、交给环境的动作、结果字段逐项相同。"""
    a = _smvla_traced(tmp_path, F.Plan(success_at=40), traced=True)
    b = _smvla_traced(tmp_path, F.Plan(success_at=40), traced=False)
    assert _norm(a[3].log) == _norm(b[3].log)
    assert [x.tobytes() for x in a[2].env.actions] == [x.tobytes() for x in b[2].env.actions]
    assert _strip(a[0]) == _strip(b[0]) and "trace_path" not in b[0]


def test_smvla_hard_bound_timeout_is_renderable_without_omitted_frame(tmp_path):
    """max_steps=32、执行段 16：决策上限 4 次、64 步 timeout；每步都录，omitted=0。"""
    res, sess, _, _, ep = _smvla_traced(tmp_path, F.Plan(), max_steps=32, execute_horizon=16)
    assert res["status"] == "timeout" and sess.steps == 64
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "timeout"})
    end = _rows(ep)[-1]
    assert end["omitted_timeout_frames"] == 0 and end["frames_recorded"] == F.N_RESET_FRAMES + 64
    assert end["episode_max_steps"] == 32 and end["step_bound"] == 64


def test_smvla_strict_cap_trace_closes_as_timeout(tmp_path):
    """strict-cap 20：第 21 步不进环境（客户端记 step_exc error、env_client 按 cap_hit 改记 timeout）；轨迹 20 步、timeout。"""
    res, sess, b, _, ep = _smvla_traced(tmp_path, F.Plan(), cap=20, max_steps=20, execute_horizon=16)
    assert sess.cap_hit is True and sess.steps == 20 and b.env.n == 20
    assert res["status"] == "error" and res["error"].startswith("step_exc: ")  # 客户端行为不变
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "timeout"})
    end = _rows(ep)[-1]
    assert (end["status"], end["terminal_reason"], end["cap_hit"]) == ("timeout", "timeout", True)
    assert len(_kind(_rows(ep), "step")) == 20 and end["omitted_timeout_frames"] == 0


def test_smvla_step_exception_is_missing_step_with_consistent_counts(tmp_path):
    """第 5 步环境抛异常：环境侧计 5 步，轨迹 5 行（末行缺观测、保留动作与原因），三分计数 5／4／demo+1+4。"""
    res, sess, b, _, ep = _smvla_traced(tmp_path, F.Plan(raise_at=5, raise_exc=lambda: RuntimeError("IK 失败")))
    assert res["status"] == "error" and sess.steps == 5
    assert tc.contract_problems(ep) == []  # error 局的重绘放行由 S2a 负责，这里只核契约
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "error"})
    steps = _kind(_rows(ep), "step")
    assert [st.get("observed", True) for st in steps] == [True] * 4 + [False]
    assert "IK 失败" in steps[-1]["missing_reason"] and steps[-1]["subgoal"] == "s"
    assert steps[-1]["terminated"] == steps[-1]["truncated"] == "NOT_OBSERVED"
    with np.load(ep / "arrays.npz") as arr:
        assert arr["exec_action__00004"].tobytes() == b.env.actions[4].tobytes()
    end = _rows(ep)[-1]
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (5, 4, F.N_RESET_FRAMES + 4)


def test_smvla_reset_failure_is_no_frame_error(tmp_path):
    class _BadReset(_Session):
        def reset(self):
            raise RuntimeError("reset 爆了")

        def close(self):
            pass

    sm = F.smvla_client()
    conn_info, ep = _conn(tmp_path)
    res = sm.run_episode(_BadReset(F.Plan()), dict(IDENT), conn_info, None,
                         conn=F.FakeSmvlaConn(F.FakePolicyServer()), reset_retries=0)
    assert res["status"] == "error"
    assert tc.contract_problems(ep) == []
    end = _rows(ep)[-1]
    assert (end["no_frame"], end["status"], end["demo_frames"], end["frames_recorded"]) == (True, "error", 0, 0)
    assert end["steps_attempted"] == 0
    tc.assert_counts_consistent(ep, {"exec_steps": 0, "status": "error"})


def _bare_recorder(out_dir):
    """真实 recorder.EpisodeRecorder 的 add_array／_write_arrays（不起 ffmpeg 写线程），其余录制调用置空。"""
    R = load_script("eval-official/recorder.py")
    rec = object.__new__(R.EpisodeRecorder)
    out_dir.mkdir(parents=True)
    rec.out_dir = out_dir
    rec._lock = threading.RLock()
    rec._arrays = []
    rec._array_counts = collections.defaultdict(int)
    rec._writer_error = None
    rec._writer = None
    rec._closed = False
    rec.set_phase = lambda phase: None
    rec.add_frames = lambda stream, frames, tag="": list(range(len(frames)))
    rec.add_event = lambda ev: None
    return rec


def test_smvla_recorder_arrays_keys_match_trace_contract(tmp_path):
    """生产形态：EnvSession 与客户端共用录像器、轨迹落在录像目录（无 trace_dir）。轨迹不另写 arrays.npz，
    录像器写出的 exec_action__%05d 与契约键名一致，合起来可渲染。"""
    ep = tmp_path / f"{IDENT['key']}.a2"
    rec = _bare_recorder(ep)
    sm = F.smvla_client()
    sess, _ = _env_session(F.Plan(success_at=18), recorder=rec)
    res = sm.run_episode(sess, dict(IDENT), {"episode_tag": ep.name, "dataset": "hard-verify"}, rec,
                         conn=F.FakeSmvlaConn(F.FakePolicyServer()))
    assert res["status"] == "success"
    assert res["trace_path"] == str(ep / "trace.jsonl")  # trace_location 退回 recorder.out_dir
    # 第三阶段：轨迹先经 merge_write_npz 写同目录 arrays.npz，录像器收尾再合并同名键，先后不覆盖
    assert (ep / "arrays.npz").exists()
    assert _rows(ep)[-1]["arrays"]["action_keys"] == sess.steps and _rows(ep)[0]["identity"]["attempt"] == 2
    rec._write_arrays()  # 录像器收尾写 arrays.npz（与轨迹的 exec_action 同键同值，不冲突）
    with np.load(ep / "arrays.npz") as arr:
        files = set(arr.files)
    assert {f"exec_action__{i:05d}" for i in range(sess.steps)} <= files
    assert {f"exec_state__{i:05d}" for i in range(sess.steps)} <= files and "model_action__00000" in files
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "success"})


def _framesamp_modul_traced(tmp_path, monkeypatch, plan, *, cap=None, max_steps=1300, traced=True, client_wrap=None):
    mc = F.framesamp_modul_client()
    server = F.FakePolicyServer()

    def factory(host, port, recorder, timing):
        c = F.FakeMMEVLAWebsocketClient(server)
        return client_wrap(c) if client_wrap else c

    monkeypatch.setattr(mc, "make_recording_client", factory)
    sess, b = _env_session(plan, cap=cap)
    if traced:
        conn_info, ep = _conn(tmp_path, port=1, max_steps=max_steps)
    else:
        conn_info, ep = {"port": 1, "max_steps": max_steps}, None
    res = mc.run_episode(sess, dict(IDENT), conn_info, None)
    return res, sess, b, server, ep


def test_framesamp_modul_trace_success_is_renderable_with_null_subgoals(tmp_path, monkeypatch):
    res, sess, b, server, ep = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(success_at=20))
    assert res["status"] == "success" and sess.steps == 20
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "success"})
    rows = _rows(ep)
    assert rows[0]["route"] == "perceptual-framesamp-modul/new" and rows[0]["identity"]["key"] == IDENT["key"]
    steps = _kind(rows, "step")
    assert len(steps) == 20 and all(st["subgoal"] is None for st in steps)  # FrameSamp+Modulation 无子目标功能（C7）
    assert [r["name"] for r in _kind(rows, "request")] == ["reset", "add_buffer", "infer", "add_buffer", "infer"]
    assert [(h["start"], h["end"]) for h in _kind(rows, "history")] == [(0, 0), (0, 16)]
    tw = _tw()
    infer_actions = [p["actions"] for k, p in server.log if k == "infer"]
    assert [r["actions"]["sha256"] for r in _kind(rows, "response")] == \
        [tw.array_record(a)["sha256"] for a in infer_actions]
    end = rows[-1]
    # 第三阶段：float32 动作也逐步收进 arrays.npz（完整数值），end.arrays 为摘要
    assert end["arrays"] == {"path": "arrays.npz", "action_keys": 20, "state_keys": 20, "missing_state_steps": []}
    assert end["request_encoding"] == "canonical" and end["observer_hook_errors"] == 0  # 替身客户端没有原始字节钩子
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (20, 20, F.N_RESET_FRAMES + 20)


def test_framesamp_modul_trace_does_not_change_messages_or_actions(tmp_path, monkeypatch):
    a = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(success_at=40))
    b = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(success_at=40), traced=False)
    assert _norm(a[3].log) == _norm(b[3].log)
    assert [x.tobytes() for x in a[2].env.actions] == [x.tobytes() for x in b[2].env.actions]
    assert _strip(a[0]) == _strip(b[0]) and "trace_path" not in b[0]


def test_framesamp_modul_natural_timeout_omits_last_frame(tmp_path, monkeypatch):
    """不带 strict-cap、max_steps=5：第 6 步照常执行并记录，官方录像不录最后一步（omitted=1）。"""
    res, sess, _, _, ep = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(), max_steps=5)
    assert res["status"] == "timeout" and sess.steps == 6
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "timeout"})
    end = _rows(ep)[-1]
    assert (end["omitted_timeout_frames"], end["frames_recorded"]) == (1, F.N_RESET_FRAMES + 6 - 1)


def test_framesamp_modul_strict_cap_trace_closes_as_timeout(tmp_path, monkeypatch):
    """strict-cap 5：第 6 步不进环境、不记步；轨迹 5 步、timeout、omitted=0。"""
    res, sess, b, _, ep = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(), cap=5, max_steps=5)
    assert sess.cap_hit is True and sess.steps == 5 and b.env.n == 5
    tc.assert_renderable(ep)
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "timeout"})
    end = _rows(ep)[-1]
    assert (end["status"], end["cap_hit"], end["omitted_timeout_frames"]) == ("timeout", True, 0)
    assert len(_kind(_rows(ep), "step")) == 5


def test_framesamp_modul_env_exception_is_missing_step_with_consistent_counts(tmp_path, monkeypatch):
    """第 3 步抛异常：EnvRunnerShim 返回 (None,)*3 的这一步记缺观测步；三分计数 3／2／demo+1+2。"""
    res, sess, b, _, ep = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(raise_at=3, raise_exc=lambda: RuntimeError("IK")))
    assert res["status"] == "error" and sess.steps == 3 and res["steps"] == 3
    assert tc.contract_problems(ep) == []
    tc.assert_counts_consistent(ep, {"exec_steps": sess.steps, "status": "error"})
    steps = _kind(_rows(ep), "step")
    assert [st.get("observed", True) for st in steps] == [True, True, False]
    assert "IK" in steps[-1]["missing_reason"] and steps[-1]["subgoal"] is None
    assert steps[-1]["action"]["sha256"] == hashlib.sha256(b.env.actions[2].tobytes()).hexdigest()
    end = _rows(ep)[-1]
    assert (end["steps_attempted"], end["steps_observed"], end["frames_recorded"]) == (3, 2, F.N_RESET_FRAMES + 2)


def test_framesamp_modul_non_float32_actions_write_arrays_npz(tmp_path, monkeypatch):
    """服务回 float64 动作块：逐步原值进 arrays.npz（键 exec_action__%05d），可渲染。"""

    class _F64:
        def __init__(self, inner):
            self._inner = inner
            self._ws = inner._ws

        def reset(self):
            return self._inner.reset()

        def add_buffer(self, buf):
            return self._inner.add_buffer(buf)

        def infer(self, element):
            out = self._inner.infer(element)
            return {"actions": np.asarray(out["actions"], dtype=np.float64)}

    res, sess, b, _, ep = _framesamp_modul_traced(tmp_path, monkeypatch, F.Plan(success_at=7), client_wrap=_F64)
    assert res["status"] == "success"
    tc.assert_renderable(ep)
    assert _rows(ep)[-1]["arrays"]["action_keys"] == 7  # 第三阶段：end.arrays 为摘要
    with np.load(ep / "arrays.npz") as arr:
        assert sorted(k for k in arr.files if k.startswith("exec_action__")) == \
            [f"exec_action__{i:05d}" for i in range(7)]
        assert all(arr[f"exec_action__{i:05d}"].tobytes() == b.env.actions[i].tobytes() for i in range(7))


def test_framesamp_modul_real_recording_client_logs_raw_msgpack_hashes(tmp_path):
    """真实 RecordingClient（回环假 server）：请求行记实际发出的 msgpack 字节 sha256。"""
    pytest.importorskip("openpi_client", reason="未验证：openpi_client 未安装")
    pytest.importorskip("websockets", reason="未验证：websockets 未安装")
    from openpi_client import msgpack_numpy
    from test_framesamp_modul_transport import _FakeServer

    mc = F.framesamp_modul_client()
    sess, _ = _env_session(F.Plan(success_at=20))
    with _FakeServer() as srv:
        conn_info, ep = _conn(tmp_path, port=srv.port, host="127.0.0.1", max_steps=1300)
        res = mc.run_episode(sess, dict(IDENT), conn_info, None)
    assert res["status"] == "success"
    tc.assert_renderable(ep)
    rows = _rows(ep)
    reqs = _kind(rows, "request")
    assert [r["name"] for r in reqs] == ["reset", "add_buffer", "infer", "add_buffer", "infer"]
    raw_reset = msgpack_numpy.Packer().pack({"reset": True})
    assert reqs[0]["sha256"] == hashlib.sha256(raw_reset).hexdigest() and reqs[0]["nbytes"] == len(raw_reset)
    assert rows[-1]["request_encoding"] == "msgpack"
