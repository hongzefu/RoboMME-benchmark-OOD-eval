"""第二档对比：原侧（官方原版入口）对新侧（本仓库改后入口）的逐身份差异表（计划第二部分 1.7）。

输入：两侧的结果行（jsonl，可多份分片文件）与轨迹根目录（递归找 ``trace.jsonl``，格式见 ``trace_writer.py``）。

配对：按 ``(task, source_episode, seed)``。每侧每个身份取「最终结果行」：``canary``、``infra``、``late`` 为真的行不算；
同一侧同一身份有多于一条最终行记 ``duplicate``。轨迹先用结果行的 ``trace_path``（若有）；否则在轨迹根下按轨迹
header 的 ``identity`` 三元组找，同一身份多份时按 ``identity.attempt`` 对上结果行的 ``attempt``，仍不唯一时取
所在目录名以 ``.a<attempt>`` 结尾的那份。

逐身份输出：两侧终态（结果行 ``status``）、是否同终态、两侧执行步数、是否逐项相同、首个分叉步与分叉类型。
分叉类型按时间线依次比较：演示（画面／状态／文本）→ 每步之前的请求与历史边界（``request``）→ 模型返回的
动作块（``action``）→ 第 k 步行：子目标文本（``text``）、执行动作（``action``）、执行后画面（``obs``）、
状态（``state``）、``terminated``／``truncated``／``status``（``stop``）；公共步都相同而步数不同记 ``stop``。
``identical_trace`` 的覆盖范围只限这些字段。

判定行（``GATE2`` 恒为 INFO 或 INCOMPLETE，不阻塞任何流程）::

    GATE2=INFO policy=<p> compared=<n> same_terminal=<n> identical_trace=<n> first_diverge_obs=<n>
          first_diverge_state=<n> first_diverge_text=<n> first_diverge_action=<n> missing=0
          [first_diverge_stop=<n> first_diverge_request=<n> duplicate=0 missing_trace=0 bad_trace=0]
          [first_episode_identical=<n>/<服务启动次数> first_episode_aligned=<n>] [site=local]

两侧身份不齐（``missing``）、重复（``duplicate``）、完整模式下缺轨迹或轨迹结构不合规、``--expect-total`` 与
配上的身份数不符时，判定为 ``GATE2=INCOMPLETE``。

``--mode astra``：规划来自在线模型，两侧不会逐步相同；只比终态与子任务序列（结果行 ``subtasks`` 字段优先，
否则取轨迹逐步 ``subgoal`` 去重后的序列），另报两侧请求次数（按请求名）。判定行
``GATE2=INFO policy=… mode=astra compared=… same_terminal=… same_subtasks=… missing=0 …``。

``--groundsg``：GroundSG 服务端随机数跨局累积，只有每次服务启动后的第一局能逐步对拍。服务启动边界取结果行
字段 **``server_epoch``**（整数，同一结果文件、同一席位内每次（重新）起服务加 1；由席位脚本或驱动写入）；
同一 ``(结果文件, seat, server_epoch)`` 内按文件行序的第一行即该次启动的第一局。缺该字段时每个结果文件
（分片）视为一次启动，取文件第一行。分母为新侧的启动次数；分子为「该局在原侧也是某次启动的第一局，且两侧
轨迹逐项相同」的个数；``first_episode_aligned`` 为两侧都是启动第一局的个数。

用法::

    python dev-scripts/checks/gate2_compare.py --policy groundsg-oracle \
        --orig-results <原侧结果 jsonl>... --new-results <新侧结果 jsonl>... \
        --orig-traces <原侧轨迹根> --new-traces <新侧轨迹根> [--groundsg] [--mode astra] \
        [--expect-total 192] [--site local] [--out-json gate2.json] [--out-md gate2.md]

第二阶段扩展（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分一节 S6；**只在传了任一新参数时启用**，
不传新参数时上面的默认格式输出与此前逐字节相同）：

- 输入绑定：``--manifest <冻结清单>``（jsonl 每行一个身份，或 JSON 列表／``{"identities": [...]}``）、
  ``--new-ledger <AttemptLedger jsonl>...``（新侧每身份只取账本 ``accept`` 行 ``accepted_attempt_id`` 对应的结果行）、
  ``--orig-attempts <orig-attempts.json>``（原侧每身份只取 S7 适配器给出的唯一 attempt；``--new-attempts`` 同格式，供
  「新侧」位置放的是原侧重跑时用）。轨迹经 ``TraceIndex.lookup_bound`` 核 header ``identity`` 三元组与 ``attempt``
  等于结果行（header 无 ``attempt`` 时取局目录名 ``.a<N>``）。输出::

      GATE2_INPUTS=PASS|FAIL expected=<n> missing=<n> extra=<n> unaccepted=<n> ambiguous=<n> trace_binding_mismatch=<n>

  ``missing``：清单身份在任一侧没有被接受的结果行；``extra``：被接受的身份不在清单；``unaccepted``：账本侧为
  最终结果行（非 canary／infra／late）不是账本接受的那一次（含该键无 accept），attempt 映射侧为映射里没有该身份的
  最终行（映射里有该身份时，非权威 attempt 的行是适配器已裁定的旧尝试，不计；结果行无 ``attempt`` 视作 1）；
  ``ambiguous``：同侧同身份多于一条被接受的行、账本或映射给出多个 attempt、清单身份重复。
- 节点来源（R8）：被比较的结果行读 ``node``（缺则读 ``host``）；短主机名形如 ``glNNNN`` 为 GL，其余计
  ``local_rows``，都没有计 ``unknown_rows``；``--orig-format v75`` 时原侧（E0 历史文件）不核来源。输出
  ``GATE2_PROVENANCE=PASS|FAIL local_rows=<n> unknown_rows=<n>``。``GATE2_INPUTS`` 或 ``GATE2_PROVENANCE`` 任一
  FAIL，判定行为 ``GATE2=INVALID reason=inputs|cross_machine``。
- 来源主机模式（1006 计划八.10 第 6 条、冻结说明第九节）：``--expect-host <name>`` 时不再按 GL 判，逐行核两侧
  ``node``（缺则 ``host``）的短主机名都等于给定主机（不区分大小写，``sled-vail.x.y`` 与 ``sled-vail`` 等同），异主机计
  ``foreign_rows``、缺字段计 ``unknown_rows``，输出 ``GATE2_PROVENANCE=PASS|FAIL mode=host expect_host=<name>
  foreign_rows=<n> unknown_rows=<n>``，FAIL 时 ``GATE2=INVALID reason=cross_machine``；不给时保持上面的 GL 模式原检查。
  给 ``--expect-host`` 即启用扩展模式。
- 语言账本逐次比（1006 计划八.11）：扩展模式下读两侧轨迹同目录的 ``language.jsonl``（``trace_writer.LanguageLog`` 行：
  ``call_open``／``message``／``call_close``／``reuse``），调用序号按 ``call_open`` 出现顺序（0 起），按
  ``(调用序号, message_index)`` 对齐有序消息全集（不按四字段去重），逐次比 ``text``（另比 ``dir``／``role``）；不同或只一侧有的
  消息按 ``dir`` 计入 ``prompt_diff``（``in``）或 ``reply_diff``（``out``）。``GATE2`` 行在 ``other_terminal`` 后追加
  ``prompt_diff=<n> reply_diff=<n>``（没有任何一对两侧都有 ``language.jsonl`` 时为 ``NA``；只一侧有的对数 >0 时另追加
  ``lang_missing=<n>``），并在 ``GATE2`` 行之前多一行 ``GATE2_LANG_FIRST_DIFF``，给出第一处不同的身份、调用序号、消息序号与
  两侧文字片段（以第一个不同字符为中心截取）。
- 补集合核对：``--orig-supplement <本轮补跑结果 jsonl>...`` 与 ``--manifest`` 同给时，旧原侧（``--orig-results``）的
  最终身份集合与补跑身份集合按身份比：补跑必须恰等于「清单 − 旧原侧」，不凭条数；输出
  ``GATE2_SUPPLEMENT=PASS|FAIL old=<n> supplement=<n> expected=<n> missing=<n> extra=<n> overlap=<n>``，FAIL 时
  ``GATE2=INVALID reason=supplement``。比较时原侧 = 旧原侧 + 补跑。
- 跨格式投影（C9、C10）：每对轨迹按维度 ``action``（执行动作）、``obs``（画面）、``state``（状态）、
  ``logic``（逻辑输入：请求／动作块／历史边界）、``text``（展示文本：演示文本与逐步子目标）、``stop``（终止标志与
  步数）逐项计 ``same/diff/not_observed``，不在首个差异处停止。任一侧为 ``NOT_OBSERVED`` 或两侧都为 ``None``
  （都没观测到）计 ``not_observed``，不算相同。动作按值比：有 ``arrays.npz`` 原值时按原值比（两侧都有则按 float64
  比数值；一侧有原值则转成另一侧 dtype 须无损往返且字节哈希相等），都没有原值时 dtype 相同比字节哈希、dtype 不同且
  元素数 ≤8 比 ``f32hex``，否则计 ``not_observed``。请求只比两侧都出现过的请求名（只一侧有的请求名计
  ``not_observed``）；动作块、历史边界只一侧记录时同样计 ``not_observed``。
- 统计：``s2f``（原侧成功→新侧非成功）、``f2s``、``sr_orig``、``sr_new``（成功率，0～1）、``sr_diff_pp``（新−原，
  百分点）、终态 3×3 矩阵（success／fail／timeout，其余终态计 ``other_terminal``）、McNemar 精确二项检验 p 值
  （双侧，自写，不加依赖）。
- ``--orig-format v75``：原侧读 v7.5eval 的 E0 逐局文件（``episodes*.jsonl``，``steps`` 视作执行步数），读前按
  ``--v75-input-manifest``（默认主检出 ``artifacts/v7.5eval/input-manifest.json``）核每个文件的 sha256，文件不在
  清单或 sha 不符即报错（退出码 2）；E0 无轨迹，只比终态与统计，另出
  ``ORIG_RERUN_VS_E0=INFO policy=<p> compared=<n> flips=<n> s2f=<n> f2s=<n>``。``--noise-runs <O1> <O2>``（文件或目录）
  输出 E0／O1／O2 三对两方向翻转 ``GATE2_NOISE=INFO policy=<p> pair=<a>-<b> compared=<n> s2f=<n> f2s=<n> flips=<n> …``，
  只作描述性历史观测。
- ``--groundsg`` 在扩展模式下不再只判首局：192 局照常全比，表里加诊断列 ``server_epoch_first``（新侧该局是否某次
  服务启动后的第一局）与 ``server_epoch_first_orig``，判定行附 ``server_epoch_first=<新侧首局数>``。

扩展模式判定行::

    GATE2=INFO|INCOMPLETE|INVALID policy=<p> compared=<n> same_terminal=<n> s2f=<n> f2s=<n> sr_orig=<x> sr_new=<y>
          sr_diff_pp=<d> mcnemar_p=<p> not_observed=<n> identical_trace=<n> missing=<n> duplicate=<n>
          missing_trace=<n> bad_trace=<n> matrix=ss:..,sf:..,…,tt:.. other_terminal=<n> prompt_diff=<n|NA>
          reply_diff=<n|NA> [lang_missing=<n>] [server_epoch_first=<n>] [expect_total=<n>] [reason=…] [site=…]

另有一行 ``GATE2_PROJ policy=<p> action=<same>/<diff>/<not_observed> obs=… state=… logic=… text=… stop=…
episodes_diff=action:<n>,obs:<n>,…``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

def _load_trace_writer():
    """评估包的 ``trace_writer``：模块名 ``trace_writer``（与原侧观测器、模型客户端同一别名，已加载则复用同一份）。"""
    import importlib.util

    mod = sys.modules.get("trace_writer")
    if mod is not None and hasattr(mod, "TraceWriter"):
        return mod
    path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "record" / "trace_writer.py"
    spec = importlib.util.spec_from_file_location("trace_writer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trace_writer"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("trace_writer", None)
        raise
    return mod


tw = _load_trace_writer()  # 评估包 robomme_ood_eval/record/trace_writer.py（按路径加载，模块名 trace_writer）

DIVERGE_KINDS = ("obs", "state", "text", "action", "stop", "request")
EPOCH_FIELD = "server_epoch"


# ── 结果行 ───────────────────────────────────────────────────────────────────


def ident_of(row: dict) -> tuple:
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    task = row.get("task", ident.get("task"))
    se = row.get("source_episode", ident.get("source_episode"))
    seed = row.get("seed", ident.get("seed"))
    return (task, None if se is None else int(se), None if seed is None else int(seed))


def is_final(row: dict) -> bool:
    return not (row.get("canary") or row.get("infra") or row.get("late"))


def official_defs():
    """评估包里的别名表 ``robomme_ood_eval.models._official_defs``（旧名别名表的唯一来源；模块名 ``official_defs``，
    已加载则复用同一模块）。"""
    import importlib.util

    mod = sys.modules.get("official_defs")
    if mod is None or not hasattr(mod, "canonical_row"):
        path = Path(__file__).resolve().parents[2] / "src" / "robomme_ood_eval" / "models" / "_official_defs.py"
        spec = importlib.util.spec_from_file_location("official_defs", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["official_defs"] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop("official_defs", None)
            raise
    return mod


def read_rows(paths: list[str | Path]) -> list[dict]:
    """读多份结果 jsonl；每行附加 ``_file``（来源文件）与 ``_line``（文件内行序）。历史行的旧标签映射成官方名。"""
    canon = official_defs().canonical_row
    rows: list[dict] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for i, line in enumerate(x for x in fh if x.strip()):
                r = canon(json.loads(line))
                r["_file"] = str(p)
                r["_line"] = i
                rows.append(r)
    return rows


def final_rows(rows: list[dict]) -> tuple[dict[tuple, dict], Counter]:
    """每身份唯一最终行；返回 (身份→行, 身份→最终行数)。"""
    finals: dict[tuple, dict] = {}
    count: Counter = Counter()
    for r in rows:
        if not is_final(r):
            continue
        k = ident_of(r)
        count[k] += 1
        finals.setdefault(k, r)
    return finals, count


def epoch_firsts(rows: list[dict]) -> tuple[set[tuple], int]:
    """每次服务启动的第一局身份集合与启动次数（见模块说明的 ``server_epoch`` 约定）。"""
    seen: dict[tuple, tuple] = {}
    for r in sorted(rows, key=lambda x: (x["_file"], x["_line"])):
        if r.get("canary"):
            continue
        ep = (r["_file"], r.get("seat"), r.get(EPOCH_FIELD, 0))
        seen.setdefault(ep, ident_of(r))
    return set(seen.values()), len(seen)


# ── 轨迹索引 ─────────────────────────────────────────────────────────────────


class TraceIndex:
    def __init__(self, root: str | Path | None):
        self.by_ident: dict[tuple, list[tuple[Path, dict]]] = defaultdict(list)
        if root is None or not Path(root).exists():
            return
        for p in tw.find_traces(root):
            try:
                with p.open(encoding="utf-8") as fh:
                    head = json.loads(fh.readline())
            except Exception:  # noqa: BLE001 坏文件不进索引，按缺轨迹计
                continue
            ident = head.get("identity") or {}
            self.by_ident[ident_of(ident)].append((p, ident))

    def lookup(self, row: dict) -> Path | None:
        if row.get("trace_path"):
            p = Path(row["trace_path"])
            if p.exists():
                return p
            # 结果行记的是节点本地路径（如 /tmp/<run>/…），轨迹发布到 NFS／本机 media 后本地副本已删：退回按身份索引查
        cands = self.by_ident.get(ident_of(row), [])
        if len(cands) <= 1:
            return cands[0][0] if cands else None
        att = row.get("attempt")
        hit = [p for p, ident in cands if ident.get("attempt") is not None and ident.get("attempt") == att]
        if len(hit) == 1:
            return hit[0]
        hit = [p for p, _ in cands if p.parent.name.endswith(f".a{att}")]
        return hit[0] if len(hit) == 1 else None

    def lookup_bound(self, row: dict) -> tuple[Path | None, str | None]:
        """扩展模式（审计第 14 条）：先按 ``lookup`` 找轨迹，再核 header 的 identity 三元组与 attempt 等于结果行。

        返回 ``(路径, 不符原因)``；找不到轨迹为 ``(None, None)``（按缺轨迹计，不算绑定不符）。header 无 ``attempt``
        时取局目录名 ``.a<N>``；两处都没有记 ``attempt_unknown``。"""
        p = self.lookup(row)
        if p is None:
            return None, None
        try:
            with p.open(encoding="utf-8") as fh:
                head = json.loads(fh.readline())
        except Exception:  # noqa: BLE001
            return p, "header_unreadable"
        ident = head.get("identity") or {}
        if ident_of(ident) != ident_of(row):
            return p, "identity"
        att_h = ident.get("attempt")
        if att_h is None:
            m = re.search(r"\.a(\d+)$", p.parent.name)
            att_h = int(m.group(1)) if m else None
        if att_h is None:
            return p, "attempt_unknown"
        att_r = row.get("attempt")
        if att_r is None or int(att_r) != int(att_h):
            return p, "attempt"
        return p, None


# ── 轨迹逐项比较 ─────────────────────────────────────────────────────────────


def _sha(rec: Any) -> Any:
    """array_record → (dtype, shape, sha256)；其他原样。"""
    if isinstance(rec, dict) and "sha256" in rec:
        return (rec.get("dtype"), tuple(rec.get("shape") or ()), rec["sha256"])
    return rec


def _pre_step(rows: list[dict]) -> tuple[dict[int, list], dict[int, list]]:
    """按 step 分组的请求（含历史边界）与动作块。"""
    req: dict[int, list] = defaultdict(list)
    resp: dict[int, list] = defaultdict(list)
    for r in rows:
        k = r.get("kind")
        if k == "request":
            req[int(r["step"])].append(("request", r.get("name"), r.get("sha256"), r.get("nbytes")))
        elif k == "history":
            req[int(r.get("end", r.get("start", 0)))].append(("history", r.get("start"), r.get("end")))
        elif k == "response":
            resp[int(r["step"])].append(_sha(r.get("actions")))
    return req, resp


def compare_traces(a: list[dict], b: list[dict]) -> dict:
    """返回 ``{"identical": bool, "diverge_step": int|None, "diverge_kind": str|None, "field": str|None}``。

    ``diverge_step`` 为 0 表示演示阶段或第一步之前的请求就分叉。"""
    def out(step, kind, field):
        return {"identical": False, "diverge_step": step, "diverge_kind": kind, "field": field}

    da = next((r for r in a if r.get("kind") == "demo"), {}) or {}
    db = next((r for r in b if r.get("kind") == "demo"), {}) or {}
    if da.get("frames") != db.get("frames") or da.get("front_sha256") != db.get("front_sha256") \
            or da.get("wrist_sha256") != db.get("wrist_sha256"):
        return out(0, "obs", "demo.frames")
    if [_sha(x) for x in da.get("states") or []] != [_sha(x) for x in db.get("states") or []]:
        return out(0, "state", "demo.states")
    if (da.get("texts") or []) != (db.get("texts") or []):
        return out(0, "text", "demo.texts")

    req_a, resp_a = _pre_step(a)
    req_b, resp_b = _pre_step(b)
    steps_a = [r for r in a if r.get("kind") == "step"]
    steps_b = [r for r in b if r.get("kind") == "step"]
    n = max(len(steps_a), len(steps_b))
    for i in range(n + 1):
        # 第 i 步执行后、第 i+1 步执行前发出的请求与收到的动作块（i=0 即首个请求）
        if req_a.get(i, []) != req_b.get(i, []):
            return out(i, "request", "request")
        if resp_a.get(i, []) != resp_b.get(i, []):
            return out(i, "action", "response")
        if i == n:
            break
        if i >= len(steps_a) or i >= len(steps_b):
            return out(i + 1, "stop", "exec_steps")
        sa, sb = steps_a[i], steps_b[i]
        step = i + 1
        if sa.get("subgoal") != sb.get("subgoal"):
            return out(step, "text", "subgoal")
        if _sha(sa.get("action")) != _sha(sb.get("action")):
            return out(step, "action", "action")
        if sa.get("front_sha256") != sb.get("front_sha256") or sa.get("wrist_sha256") != sb.get("wrist_sha256"):
            return out(step, "obs", "front_sha256" if sa.get("front_sha256") != sb.get("front_sha256") else "wrist_sha256")
        if _sha(sa.get("state")) != _sha(sb.get("state")):
            return out(step, "state", "state")
        for f in ("terminated", "truncated", "status"):
            if sa.get(f) != sb.get(f):
                return out(step, "stop", f)
    ea = next((r for r in reversed(a) if r.get("kind") == "end"), {}) or {}
    eb = next((r for r in reversed(b) if r.get("kind") == "end"), {}) or {}
    if ea.get("status") != eb.get("status") or ea.get("exec_steps") != eb.get("exec_steps"):
        return out(n, "stop", "end")
    return {"identical": True, "diverge_step": None, "diverge_kind": None, "field": None}


# ── 主流程 ───────────────────────────────────────────────────────────────────


def _subtasks(row: dict, trace: list[dict] | None) -> list | None:
    if isinstance(row.get("subtasks"), list):
        return list(row["subtasks"])
    return tw.subgoal_sequence(trace) if trace is not None else None


def _request_counts(trace: list[dict] | None) -> dict:
    if trace is None:
        return {}
    return dict(Counter(r.get("name") for r in trace if r.get("kind") == "request"))


def compare(orig_rows: list[dict], new_rows: list[dict], orig_traces: str | Path | None, new_traces: str | Path | None,
            *, mode: str = "full", groundsg: bool = False, expect_total: int | None = None) -> dict:
    of, oc = final_rows(orig_rows)
    nf, nc = final_rows(new_rows)
    dup = sorted({k for k, v in oc.items() if v > 1} | {k for k, v in nc.items() if v > 1}, key=str)
    only_o = sorted(set(of) - set(nf), key=str)
    only_n = sorted(set(nf) - set(of), key=str)
    both = sorted(set(of) & set(nf), key=str)
    oi, ni = TraceIndex(orig_traces), TraceIndex(new_traces)

    table: list[dict] = []
    kinds: Counter = Counter()
    missing_trace = bad_trace = identical = same_term = same_sub = 0
    traces: dict[tuple, tuple] = {}
    for k in both:
        ro, rn = of[k], nf[k]
        po, pn = oi.lookup(ro), ni.lookup(rn)
        to = tw.read_trace(po) if po else None
        tn = tw.read_trace(pn) if pn else None
        row = {"task": k[0], "source_episode": k[1], "seed": k[2],
               "orig_status": ro.get("status"), "new_status": rn.get("status"),
               "orig_exec_steps": ro.get("exec_steps"), "new_exec_steps": rn.get("exec_steps"),
               "orig_trace": str(po) if po else None, "new_trace": str(pn) if pn else None}
        row["same_terminal"] = ro.get("status") == rn.get("status")
        same_term += row["same_terminal"]
        if mode == "astra":
            so, sn = _subtasks(ro, to), _subtasks(rn, tn)
            row.update(orig_subtasks=so, new_subtasks=sn, same_subtasks=(so is not None and so == sn),
                       orig_requests=_request_counts(to), new_requests=_request_counts(tn))
            same_sub += row["same_subtasks"]
        else:
            if to is None or tn is None:
                missing_trace += 1
                row.update(identical_trace=False, diverge_step=None, diverge_kind="missing_trace")
            else:
                probs = tw.validate_trace(to) + tw.validate_trace(tn)
                if probs:
                    bad_trace += 1
                    row["trace_problems"] = probs
                c = compare_traces(to, tn)
                row.update(identical_trace=c["identical"] and not probs, diverge_step=c["diverge_step"],
                           diverge_kind=c["diverge_kind"], diverge_field=c["field"])
                identical += row["identical_trace"]
                if c["diverge_kind"]:
                    kinds[c["diverge_kind"]] += 1
                traces[k] = (to, tn)
        table.append(row)

    missing = len(only_o) + len(only_n)
    summary: dict[str, Any] = {"mode": mode, "compared": len(both), "same_terminal": same_term, "missing": missing,
                               "missing_orig": [list(x) for x in only_n], "missing_new": [list(x) for x in only_o],
                               "duplicate": len(dup), "duplicates": [list(x) for x in dup]}
    if mode == "astra":
        summary["same_subtasks"] = same_sub
    else:
        summary.update(identical_trace=identical, missing_trace=missing_trace, bad_trace=bad_trace,
                       **{f"first_diverge_{x}": kinds.get(x, 0) for x in DIVERGE_KINDS})
    if groundsg:
        firsts_n, epochs_n = epoch_firsts(new_rows)
        firsts_o, _ = epoch_firsts(orig_rows)
        aligned = firsts_n & firsts_o
        ident_ok = {(r["task"], r["source_episode"], r["seed"]): bool(r.get("identical_trace")) for r in table}
        ok = sum(1 for k in aligned if ident_ok.get(k))
        summary.update(first_episode_identical=ok, server_epochs=epochs_n, first_episode_aligned=len(aligned))
        for r in table:
            k = (r["task"], r["source_episode"], r["seed"])
            r["epoch_first_new"] = k in firsts_n
            r["epoch_first_orig"] = k in firsts_o
    incomplete = missing > 0 or len(dup) > 0 or (expect_total is not None and len(both) != expect_total)
    if mode != "astra":
        incomplete = incomplete or missing_trace > 0 or bad_trace > 0
    summary["verdict"] = "INCOMPLETE" if incomplete else "INFO"
    summary["expect_total"] = expect_total
    return {"summary": summary, "table": table}


def verdict_line(res: dict, policy: str, site: str | None = None) -> str:
    s = res["summary"]
    parts = [f"GATE2={s['verdict']}", f"policy={policy}"]
    if s["mode"] == "astra":
        parts += ["mode=astra", f"compared={s['compared']}", f"same_terminal={s['same_terminal']}",
                  f"same_subtasks={s['same_subtasks']}", f"missing={s['missing']}", f"duplicate={s['duplicate']}"]
    else:
        parts += [f"compared={s['compared']}", f"same_terminal={s['same_terminal']}",
                  f"identical_trace={s['identical_trace']}"]
        parts += [f"first_diverge_{x}={s[f'first_diverge_{x}']}" for x in ("obs", "state", "text", "action")]
        parts += [f"missing={s['missing']}", f"first_diverge_stop={s['first_diverge_stop']}",
                  f"first_diverge_request={s['first_diverge_request']}", f"duplicate={s['duplicate']}",
                  f"missing_trace={s['missing_trace']}", f"bad_trace={s['bad_trace']}"]
    if "first_episode_identical" in s:
        parts += [f"first_episode_identical={s['first_episode_identical']}/{s['server_epochs']}",
                  f"first_episode_aligned={s['first_episode_aligned']}"]
    if s.get("expect_total") is not None:
        parts.append(f"expect_total={s['expect_total']}")
    if site:
        parts.append(f"site={site}")
    return " ".join(parts)


def to_markdown(res: dict, policy: str) -> str:
    s = res["summary"]
    lines = [f"# 第二档差异表：{policy}", "", f"- 判定：`{s['verdict']}`；配对 {s['compared']} 个身份，同终态 {s['same_terminal']}。"]
    if s["mode"] == "astra":
        lines += ["", "| 任务 | 原 episode | seed | 原侧终态 | 新侧终态 | 同终态 | 子任务序列相同 |", "|---|---|---|---|---|---|---|"]
        for r in res["table"]:
            lines.append(f"| {r['task']} | {r['source_episode']} | {r['seed']} | {r['orig_status']} | {r['new_status']} | "
                         f"{int(r['same_terminal'])} | {int(r['same_subtasks'])} |")
    else:
        lines.append("- `identical_trace` 只覆盖轨迹字段：演示画面哈希／状态／文本、逐请求规范化字节哈希、历史边界、"
                     "动作块、逐步画面哈希／状态／动作／子目标／终止标志与终态；不覆盖视频字节。")
        lines += ["", "| 任务 | 原 episode | seed | 原侧终态 | 新侧终态 | 同终态 | 逐项相同 | 首个分叉步 | 分叉类型 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in res["table"]:
            lines.append(f"| {r['task']} | {r['source_episode']} | {r['seed']} | {r['orig_status']} | {r['new_status']} | "
                         f"{int(r['same_terminal'])} | {int(bool(r.get('identical_trace')))} | {r.get('diverge_step')} | "
                         f"{r.get('diverge_kind')} |")
    return "\n".join(lines) + "\n"


# ── 第二阶段扩展：输入绑定、节点来源、补集合 ─────────────────────────────────

GL_NODE_RE = re.compile(r"^gl\d{4}$")
TERMINALS = ("success", "fail", "timeout")
PROJ_DIMS = ("action", "obs", "state", "logic", "text", "stop")
DEFAULT_V75_MANIFEST = Path(__file__).resolve().parents[2] / "artifacts" / "v7.5eval" / "input-manifest.json"


class InputShaMismatch(RuntimeError):
    """``--orig-format v75`` 读前 sha256 核对失败（文件不在清单或哈希不符）。"""


def _load_json_or_jsonl(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(x) for x in text.splitlines() if x.strip()]


def read_manifest(path: str | Path) -> list[tuple]:
    """冻结清单 → 身份三元组列表（保留重复，供 ambiguous 计数）。"""
    data = _load_json_or_jsonl(path)
    if isinstance(data, dict):
        data = data.get("identities") or data.get("rows") or data.get("episodes") or []
    if isinstance(data, dict):
        data = [data]
    return [ident_of(r) for r in data]


def node_of(row: dict) -> str | None:
    n = row.get("node") or row.get("host")
    return str(n) if n else None


def is_gl_node(name: str) -> bool:
    return bool(GL_NODE_RE.match(name.strip().lower().split(".")[0]))


def short_host(name: str) -> str:
    return str(name).strip().lower().split(".")[0]


def host_matches(name: str, expect: str) -> bool:
    """``--expect-host``：短主机名相同即同一台机器（``sled-vail.x.y`` 与 ``sled-vail`` 等同）。"""
    return bool(short_host(expect)) and short_host(name) == short_host(expect)


# ── 语言账本逐次比（1006 计划八.11）──────────────────────────────────────────

LANG_FILE = "language.jsonl"
SNIPPET = 40


def read_language(trace_path: Path | None) -> list[dict] | None:
    """轨迹同目录的 ``language.jsonl``；不存在返回 None，坏行记 ``{"kind": "_bad"}``。"""
    if trace_path is None:
        return None
    p = Path(trace_path).parent / LANG_FILE
    if not p.is_file():
        return None
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            rows.append({"kind": "_bad"})
            continue
        rows.append(r if isinstance(r, dict) else {"kind": "_bad"})
    return rows


def language_messages(rows: list[dict]) -> dict[tuple, dict]:
    """``{(调用序号, message_index): message 行}``；调用序号按 ``call_open`` 出现顺序（0 起），悬空消息的序号为 -1。"""
    order: dict[Any, int] = {}
    for r in rows:
        if r.get("kind") == "call_open" and r.get("call_id") not in order:
            order[r.get("call_id")] = len(order)
    out: dict[tuple, dict] = {}
    for r in rows:
        if r.get("kind") == "message":
            mi = r.get("message_index")
            key = (order.get(r.get("call_id"), -1), mi if isinstance(mi, int) else -1)
            while key in out:  # 同键重复消息不去重：顺延到同调用内的新序号，按缺一侧计差异
                key = (key[0], key[1] + 100000)
            out[key] = r
    return out


def _text_str(t: Any) -> str:
    return t if isinstance(t, str) else json.dumps(t, ensure_ascii=False, sort_keys=True)


def text_snippets(a: Any, b: Any, width: int = SNIPPET) -> tuple[str, str, int]:
    """两段文字以第一个不同字符为中心各截一段；返回 (原侧片段, 新侧片段, 第一个不同字符下标)。"""
    sa, sb = ("" if a is None else _text_str(a)), ("" if b is None else _text_str(b))
    i = 0
    while i < min(len(sa), len(sb)) and sa[i] == sb[i]:
        i += 1
    lo = max(0, i - width // 2)
    return sa[lo:lo + width], sb[lo:lo + width], i


def compare_language(la: list[dict], lb: list[dict]) -> dict:
    """两侧语言账本按 ``(调用序号, message_index)`` 对齐逐次比 ``text``（另比 ``dir``／``role``）。"""
    ma, mb = language_messages(la), language_messages(lb)
    prompt = reply = 0
    first = None
    for k in sorted(set(ma) | set(mb)):
        a, b = ma.get(k), mb.get(k)
        if a is not None and b is not None and a.get("text") == b.get("text") and a.get("dir") == b.get("dir") \
                and a.get("role") == b.get("role"):
            continue
        d = (a or b).get("dir")
        if d == "out":
            reply += 1
        else:
            prompt += 1
        if first is None:
            oa, ob, at = text_snippets(a.get("text") if a else None, b.get("text") if b else None)
            first = {"call": k[0], "message_index": k[1], "dir": d, "role": (a or b).get("role"),
                     "orig": oa if a else None, "new": ob if b else None, "at": at,
                     "orig_present": a is not None, "new_present": b is not None}
    return {"prompt_diff": prompt, "reply_diff": reply, "messages": len(set(ma) | set(mb)), "first": first}


def _key_of(row: dict) -> str | None:
    """结果行的账本键：有 ``key`` 用它，否则 ``<task>_<tier>_<seed>``（tier 可在 identity 里）；无法确定返回 None。"""
    if row.get("key"):
        return str(row["key"])
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    tier = row.get("tier", ident.get("tier"))
    task, _, seed = ident_of(row)
    if task is None or tier is None or seed is None:
        return None
    return f"{task}_{tier}_{int(seed)}"


def read_ledgers(paths: list[str | Path]) -> dict[str, set[tuple]]:
    """读一份或多份 ``AttemptLedger``：键 → {(accepted_attempt_id, attempt_no)}。

    同一账本只认每键第一条 ``accept``（与 ``AttemptLedger._apply`` 一致）；多份账本对同一键给出不同接受即多于一个元素。"""
    out: dict[str, set[tuple]] = defaultdict(set)
    for p in paths:
        first: dict[str, tuple] = {}
        for r in read_rows([p]):
            if r.get("kind") == "accept" and r.get("key") is not None:
                first.setdefault(str(r["key"]), (r.get("accepted_attempt_id"), r.get("attempt_no")))
        for k, v in first.items():
            out[k].add(v)
    return dict(out)


def read_attempt_map(path: str | Path) -> dict[tuple, set[int]]:
    """``orig-attempts.json``（S7 ``orig_results_adapter.py`` 产出）→ {身份: {attempt}}。

    兼容格式：条目列表，或 ``{"attempts"|"entries"|"identities": [...]}``，或 ``{"<task>|<source_episode>[|<seed>]": attempt}``；
    条目字段 ``task``、``source_episode``、可选 ``seed``、``attempt``；``orphan`` 为真或 ``status``／``kind`` 为 ``orphan`` 的
    条目不进比较。身份键为 ``(task, source_episode, seed|None)``。"""
    data = _load_json_or_jsonl(path)
    entries: list[dict] = []
    if isinstance(data, dict):
        lst = data.get("attempts") or data.get("entries") or data.get("identities")
        if isinstance(lst, list):
            entries = lst
        else:
            for k, v in data.items():
                parts = str(k).split("|")
                e = {"task": parts[0], "source_episode": int(parts[1]) if len(parts) > 1 else None,
                     "seed": int(parts[2]) if len(parts) > 2 and parts[2] not in ("", "None") else None}
                e.update(v if isinstance(v, dict) else {"attempt": v})
                entries.append(e)
    elif isinstance(data, list):
        entries = data
    out: dict[tuple, set[int]] = defaultdict(set)
    for e in entries:
        if e.get("orphan") or e.get("status") == "orphan" or e.get("kind") == "orphan" or e.get("attempt") is None:
            continue
        t, se, seed = ident_of(e)
        out[(t, se, seed)].add(int(e["attempt"]))
    return dict(out)


def _map_attempts(amap: dict[tuple, set[int]], ident: tuple) -> set[int] | None:
    if ident in amap:
        return amap[ident]
    return amap.get((ident[0], ident[1], None))


def bind_side(rows: list[dict], *, ledger: dict[str, set[tuple]] | None = None,
              attempts: dict[tuple, set[int]] | None = None) -> dict:
    """一侧每身份选出唯一被接受的结果行。

    返回 ``{"finals": {身份: 行}, "unaccepted": n, "ambiguous": [身份...], "dup": [身份...]}``；没有账本与映射时
    退回默认的 ``final_rows``（同身份多条最终行记 ``dup`` 与 ``ambiguous``）。"""
    if ledger is None and attempts is None:
        finals, count = final_rows(rows)
        dup = sorted((k for k, v in count.items() if v > 1), key=str)
        return {"finals": finals, "unaccepted": 0, "ambiguous": dup, "dup": dup}
    accepted: dict[tuple, list[dict]] = defaultdict(list)
    amb: set[tuple] = set()
    unaccepted = 0
    for r in rows:
        if r.get("canary"):
            continue
        k = ident_of(r)
        ok = False
        if ledger is not None:
            key = _key_of(r)
            acc = ledger.get(key) if key is not None else None
            if acc:
                if len(acc) > 1:
                    amb.add(k)
                ids = {a for a, _ in acc}
                if r.get("attempt_id") is not None:
                    ok = r["attempt_id"] in ids
                else:
                    ok = r.get("attempt") is not None and int(r["attempt"]) in {int(n) for _, n in acc if n is not None}
        else:
            want = _map_attempts(attempts, k)
            if want:
                if len(want) > 1:
                    amb.add(k)
                # 结果行无 attempt 时按单次运行约定视作第 1 次（官方入口默认 --attempt 1）
                ok = int(r.get("attempt") if r.get("attempt") is not None else 1) in want
            elif is_final(r):
                unaccepted += 1  # 映射里没有该身份：最终行无从绑定
            if ok:
                accepted[k].append(r)
            # 映射里有该身份但不是权威 attempt 的行：适配器已按「最后一条终态行」裁定，属预期，不计未接受
            continue
        if ok:
            accepted[k].append(r)
        elif is_final(r):
            unaccepted += 1
    finals = {}
    for k, lst in accepted.items():
        if len(lst) > 1:
            amb.add(k)
        finals[k] = lst[0]
    return {"finals": finals, "unaccepted": unaccepted, "ambiguous": sorted(amb, key=str), "dup": []}


def supplement_check(old_rows: list[dict], supp_rows: list[dict], manifest: list[tuple]) -> dict:
    """补集合核对：补跑身份集合必须恰等于「清单 − 旧原侧最终身份」。"""
    old = set(final_rows(old_rows)[0])
    supp = set(final_rows(supp_rows)[0])
    man = set(manifest)
    expected = man - old
    res = {"old": len(old), "supplement": len(supp), "expected": len(expected),
           "missing": len(expected - supp), "extra": len(supp - expected), "overlap": len(old & supp),
           "missing_ids": [list(x) for x in sorted(expected - supp, key=str)],
           "extra_ids": [list(x) for x in sorted(supp - expected, key=str)]}
    res["verdict"] = "PASS" if res["missing"] == 0 and res["extra"] == 0 and res["overlap"] == 0 else "FAIL"
    return res


# ── v7.5eval（E0／O1／O2）读取 ────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_v75_inputs(paths: list[str | Path], manifest_path: str | Path) -> list[dict]:
    """按 ``input-manifest.json`` 的 ``files`` 核每个文件的 sha256；不在清单或不符即抛 ``InputShaMismatch``。"""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    by_path = {str(Path(f["path"]).resolve()): f for f in man.get("files", [])}
    checked = []
    for p in paths:
        rp = Path(p).resolve()
        ent = by_path.get(str(rp))
        if ent is None:
            raise InputShaMismatch(f"path={rp} reason=not_in_manifest manifest={manifest_path}")
        actual = _sha256_file(rp)
        if actual != ent.get("sha256"):
            raise InputShaMismatch(f"path={rp} reason=sha256 expected={ent.get('sha256')} actual={actual}")
        checked.append({"path": str(rp), "sha256": actual})
    return checked


def expand_run(path: str | Path) -> list[Path]:
    """``--noise-runs`` 的一项：文件原样；目录递归找 ``episodes*.jsonl``，没有再找 ``*.jsonl``（排序后返回）。"""
    p = Path(path)
    if p.is_file():
        return [p]
    found = sorted(p.rglob("episodes*.jsonl")) or sorted(p.rglob("*.jsonl"))
    if not found:
        raise FileNotFoundError(f"{p} 下没有 jsonl")
    return found


def _v75_normalize(rows: list[dict]) -> list[dict]:
    """v7.5eval 逐局行：补 ``exec_steps``；同一身份在文件顺序上之后还有正常终态（success／fail／timeout）的
    ``status=error`` 行标 ``infra=True``——这些是原版启动器 3 遍重评前的基础设施失败（如 Vulkan
    ``createDeviceUnique`` 致 reset 失败），v7.5eval 口径取每身份最后一条正常终态，不把它们当终态。"""
    last_ok: dict[tuple, int] = {}
    for i, r in enumerate(rows):
        r.setdefault("exec_steps", r.get("steps"))
        if r.get("status") in ("success", "fail", "timeout"):
            last_ok[ident_of(r)] = i
    for i, r in enumerate(rows):
        if r.get("status") == "error" and last_ok.get(ident_of(r), -1) > i:
            r.setdefault("infra", True)
    return rows


# ── 跨格式投影（C9、C10）─────────────────────────────────────────────────────


def _not_obs(x: Any) -> bool:
    return isinstance(x, str) and x == tw.NOT_OBSERVED


class _Tally:
    def __init__(self):
        self.d = {k: {"same": 0, "diff": 0, "not_observed": 0, "first_diff_step": None} for k in PROJ_DIMS}

    def add(self, dim: str, verdict: str, step: int | None = None, n: int = 1) -> None:
        if n <= 0:
            return
        self.d[dim][verdict] += n
        if verdict == "diff" and self.d[dim]["first_diff_step"] is None:
            self.d[dim]["first_diff_step"] = step


def _load_arrays(trace_path: Path | None) -> dict:
    if trace_path is None:
        return {}
    p = Path(trace_path).parent / "arrays.npz"
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as z:
            return {k: np.array(z[k]) for k in z.files}
    except Exception:  # noqa: BLE001 读不了按无原值处理
        return {}


def _rec_size(rec: dict) -> int:
    return int(math.prod(rec.get("shape") or [1])) if rec.get("shape") is not None else 1


def _value_vs_record(v: np.ndarray, rec: dict) -> bool:
    """一侧有原值、另一侧只有记录：原值转成记录的 dtype 须无损往返，且连续字节 sha256 等于记录。"""
    try:
        dt = np.dtype(rec["dtype"])
        conv = np.ascontiguousarray(np.asarray(v).astype(dt))
        if not np.array_equal(conv.astype(np.asarray(v).dtype), np.asarray(v)):
            return False
        return conv.size == _rec_size(rec) and hashlib.sha256(conv.tobytes()).hexdigest() == rec.get("sha256")
    except Exception:  # noqa: BLE001
        return False


def _cmp_array(ra: Any, rb: Any, va: np.ndarray | None = None, vb: np.ndarray | None = None) -> str:
    """数组记录按值比较 → ``same``／``diff``／``not_observed``（规则见模块说明）。"""
    if _not_obs(ra) or _not_obs(rb):
        return "not_observed"
    if ra is None and rb is None and va is None and vb is None:
        return "not_observed"
    if va is not None and vb is not None:
        a, b = np.asarray(va), np.asarray(vb)
        if a.size != b.size:
            return "diff"
        try:
            return "same" if np.array_equal(a.astype(np.float64).ravel(), b.astype(np.float64).ravel()) else "diff"
        except (TypeError, ValueError):
            return "same" if a.tobytes() == b.tobytes() else "diff"
    if ra is None or rb is None:
        return "diff"
    if not isinstance(ra, dict) or not isinstance(rb, dict):
        return "same" if ra == rb else "diff"
    if va is not None:
        return "same" if _value_vs_record(va, rb) else "diff"
    if vb is not None:
        return "same" if _value_vs_record(vb, ra) else "diff"
    if _rec_size(ra) != _rec_size(rb):
        return "diff"
    if ra.get("dtype") == rb.get("dtype"):
        return "same" if ra.get("sha256") == rb.get("sha256") else "diff"
    if _rec_size(ra) <= 8 and ra.get("f32hex") is not None and rb.get("f32hex") is not None:
        return "same" if ra["f32hex"] == rb["f32hex"] else "diff"
    return "not_observed"


def _cmp_scalar(a: Any, b: Any, *, none_is_unobserved: bool = True) -> str:
    if _not_obs(a) or _not_obs(b):
        return "not_observed"
    if none_is_unobserved and a is None and b is None:
        return "not_observed"
    return "same" if a == b else "diff"


def _group(rows: list[dict], kind: str, keyf, valf) -> dict:
    out: dict = defaultdict(list)
    for r in rows:
        if r.get("kind") == kind:
            out[keyf(r)].append(valf(r))
    return out


def project_traces(a: list[dict], b: list[dict], arrays_a: dict | None = None, arrays_b: dict | None = None) -> dict:
    """两份轨迹按维度逐项比较，不在首个差异处停止；返回 ``{维度: {same, diff, not_observed, first_diff_step}}``。"""
    arrays_a, arrays_b = arrays_a or {}, arrays_b or {}
    t = _Tally()
    da = next((r for r in a if r.get("kind") == "demo"), {}) or {}
    db = next((r for r in b if r.get("kind") == "demo"), {}) or {}
    # 演示段：画面逐帧（front、wrist 各一项）、状态逐帧、文本整体一项；帧数不同计一项画面差异
    if da.get("frames") != db.get("frames"):
        t.add("obs", "diff", 0)
    for cam in ("front_sha256", "wrist_sha256"):
        la, lb = da.get(cam) or [], db.get(cam) or []
        for i in range(min(len(la), len(lb))):
            t.add("obs", _cmp_scalar(la[i], lb[i]), 0)
    sa_, sb_ = da.get("states") or [], db.get("states") or []
    for i in range(min(len(sa_), len(sb_))):
        t.add("state", _cmp_array(sa_[i], sb_[i]), 0)
    if len(sa_) != len(sb_):
        t.add("state", "diff", 0)
    if da or db:
        t.add("text", _cmp_scalar(da.get("texts") or [], db.get("texts") or [], none_is_unobserved=False), 0)

    # 逻辑输入：请求只比两侧都出现过的请求名；动作块、历史边界只一侧有记录时计 not_observed
    names_a = {r.get("name") for r in a if r.get("kind") == "request"}
    names_b = {r.get("name") for r in b if r.get("kind") == "request"}
    common = names_a & names_b
    t.add("logic", "not_observed", None, sum(1 for r in a + b if r.get("kind") == "request" and r.get("name") not in common))
    req_a = _group([r for r in a if r.get("name") in common], "request", lambda r: (int(r["step"]), r.get("name")),
                   lambda r: (r.get("sha256"), r.get("nbytes")))
    req_b = _group([r for r in b if r.get("name") in common], "request", lambda r: (int(r["step"]), r.get("name")),
                   lambda r: (r.get("sha256"), r.get("nbytes")))
    for k in sorted(set(req_a) | set(req_b), key=str):
        t.add("logic", "same" if req_a.get(k, []) == req_b.get(k, []) else "diff", k[0])
    for kind, keyf, valf in (("response", lambda r: int(r["step"]), lambda r: r.get("actions")),
                             ("history", lambda r: int(r.get("end", r.get("start", 0))),
                              lambda r: (r.get("start"), r.get("end")))):
        ga, gb = _group(a, kind, keyf, valf), _group(b, kind, keyf, valf)
        if not ga or not gb:
            t.add("logic", "not_observed", None, sum(len(v) for v in ga.values()) + sum(len(v) for v in gb.values()))
            continue
        for k in sorted(set(ga) | set(gb)):
            xa, xb = ga.get(k, []), gb.get(k, [])
            if len(xa) != len(xb):
                t.add("logic", "diff", k)
            elif kind == "response":
                vs = [_cmp_array(p, q) for p, q in zip(xa, xb)]
                t.add("logic", "diff" if "diff" in vs else ("not_observed" if "not_observed" in vs else "same"), k)
            else:
                t.add("logic", "same" if xa == xb else "diff", k)

    # 逐步：按步号对齐，公共步逐项比；多出的步计一项 stop 差异
    steps_a = {int(r["step"]): r for r in a if r.get("kind") == "step"}
    steps_b = {int(r["step"]): r for r in b if r.get("kind") == "step"}
    for s in sorted(set(steps_a) & set(steps_b)):
        ra, rb = steps_a[s], steps_b[s]
        key = f"exec_action__{s - 1:05d}"
        t.add("action", _cmp_array(ra.get("action"), rb.get("action"), arrays_a.get(key), arrays_b.get(key)), s)
        for cam in ("front_sha256", "wrist_sha256"):
            t.add("obs", _cmp_scalar(ra.get(cam), rb.get(cam)), s)
        t.add("state", _cmp_array(ra.get("state"), rb.get("state")), s)
        t.add("text", _cmp_scalar(ra.get("subgoal"), rb.get("subgoal"), none_is_unobserved=False), s)
        for f in ("terminated", "truncated", "status"):
            t.add("stop", _cmp_scalar(ra.get(f), rb.get(f)), s)
    if set(steps_a) != set(steps_b):
        extra = sorted(set(steps_a) ^ set(steps_b))
        t.add("stop", "diff", extra[0])
    ea = next((r for r in reversed(a) if r.get("kind") == "end"), {}) or {}
    eb = next((r for r in reversed(b) if r.get("kind") == "end"), {}) or {}
    t.add("stop", _cmp_scalar(ea.get("status"), eb.get("status")), None)
    return t.d


# ── 统计 ─────────────────────────────────────────────────────────────────────


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）：不一致对 n=b+c 下 min(b,c) 的二项尾概率乘 2，上限 1。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))


def terminal_stats(pairs: list[tuple[Any, Any]]) -> dict:
    """``pairs``：(原侧终态, 新侧终态) 列表。"""
    n = len(pairs)
    s2f = sum(1 for o, w in pairs if o == "success" and w != "success")
    f2s = sum(1 for o, w in pairs if o != "success" and w == "success")
    so = sum(1 for o, _ in pairs if o == "success")
    sn = sum(1 for _, w in pairs if w == "success")
    matrix = {o: {w: 0 for w in TERMINALS} for o in TERMINALS}
    other = 0
    for o, w in pairs:
        if o in TERMINALS and w in TERMINALS:
            matrix[o][w] += 1
        else:
            other += 1
    sr_o = so / n if n else 0.0
    sr_n = sn / n if n else 0.0
    return {"compared": n, "s2f": s2f, "f2s": f2s, "flips": s2f + f2s, "sr_orig": sr_o, "sr_new": sr_n,
            "sr_diff_pp": (sr_n - sr_o) * 100.0, "matrix": matrix, "other_terminal": other,
            "mcnemar_p": mcnemar_exact(s2f, f2s)}


def _matrix_str(m: dict) -> str:
    return ",".join(f"{o[0]}{w[0]}:{m[o][w]}" for o in TERMINALS for w in TERMINALS)


def _pair_flips(rows_a: list[dict], rows_b: list[dict]) -> dict:
    fa, fb = final_rows(rows_a)[0], final_rows(rows_b)[0]
    both = sorted(set(fa) & set(fb), key=str)
    st = terminal_stats([(fa[k].get("status"), fb[k].get("status")) for k in both])
    st["missing"] = len(set(fa) ^ set(fb))
    return st


# ── 扩展模式主流程 ───────────────────────────────────────────────────────────


def compare_ext(orig_rows: list[dict], new_rows: list[dict], orig_traces: str | Path | None,
                new_traces: str | Path | None, *, mode: str = "full", groundsg: bool = False,
                expect_total: int | None = None, manifest: list[tuple] | None = None,
                new_ledger: dict | None = None, new_attempts: dict | None = None, orig_attempts: dict | None = None,
                orig_supplement: list[dict] | None = None, orig_format: str = "default",
                noise_runs: list[list[dict]] | None = None, expect_host: str | None = None) -> dict:
    """第二阶段扩展比较（见模块说明）；返回 ``{"summary", "table", "inputs", "provenance", "supplement", "noise"}``。"""
    v75 = orig_format == "v75"
    old_orig = list(orig_rows)
    if orig_supplement:
        orig_rows = list(orig_rows) + list(orig_supplement)
    ob = bind_side(orig_rows, attempts=orig_attempts)
    nb = bind_side(new_rows, ledger=new_ledger, attempts=new_attempts)
    of, nf = ob["finals"], nb["finals"]
    dup = sorted(set(ob["dup"]) | set(nb["dup"]), key=str)
    only_o = sorted(set(of) - set(nf), key=str)
    only_n = sorted(set(nf) - set(of), key=str)
    both = sorted(set(of) & set(nf), key=str)
    oi = TraceIndex(None if v75 else orig_traces)
    ni = TraceIndex(new_traces)
    use_traces = mode != "astra" and not v75

    table: list[dict] = []
    proj_tot = {d: {"same": 0, "diff": 0, "not_observed": 0} for d in PROJ_DIMS}
    ep_diff = Counter()
    binding_mismatch = missing_trace = bad_trace = identical = same_term = same_sub = 0
    local_rows = unknown_rows = foreign_rows = 0
    lang_pairs = lang_missing = prompt_diff = reply_diff = 0
    lang_first = None
    for k in both:
        ro, rn = of[k], nf[k]
        row = {"task": k[0], "source_episode": k[1], "seed": k[2],
               "orig_status": ro.get("status"), "new_status": rn.get("status"),
               "orig_exec_steps": ro.get("exec_steps", ro.get("steps")),
               "new_exec_steps": rn.get("exec_steps", rn.get("steps")),
               "orig_node": None if v75 else node_of(ro), "new_node": node_of(rn)}
        row["same_terminal"] = ro.get("status") == rn.get("status")
        same_term += row["same_terminal"]
        for side_row in ([rn] if v75 else [ro, rn]):
            nd = node_of(side_row)
            if nd is None:
                unknown_rows += 1
            elif expect_host is not None:
                foreign_rows += not host_matches(nd, expect_host)
            elif not is_gl_node(nd):
                local_rows += 1
        po = pn = None
        bo = bn = None
        if not v75:
            po, bo = oi.lookup_bound(ro)
        pn, bn = ni.lookup_bound(rn)
        row.update(orig_trace=str(po) if po else None, new_trace=str(pn) if pn else None)
        for side, b_ in (("orig", bo), ("new", bn)):
            if b_ is not None:
                binding_mismatch += 1
                row[f"{side}_trace_binding"] = b_
        to = tw.read_trace(po) if po and bo is None else None
        tn = tw.read_trace(pn) if pn and bn is None else None
        lo = read_language(po) if po and bo is None else None
        ln = read_language(pn) if pn and bn is None else None
        if lo is not None and ln is not None:
            lc = compare_language(lo, ln)
            lang_pairs += 1
            prompt_diff += lc["prompt_diff"]
            reply_diff += lc["reply_diff"]
            row.update(prompt_diff=lc["prompt_diff"], reply_diff=lc["reply_diff"], lang_first_diff=lc["first"])
            if lc["first"] is not None and lang_first is None:
                lang_first = {"task": k[0], "source_episode": k[1], "seed": k[2], **lc["first"]}
        elif (lo is None) != (ln is None):
            lang_missing += 1
            row["lang_missing"] = "orig" if lo is None else "new"
        if mode == "astra":
            so, sn = _subtasks(ro, to), _subtasks(rn, tn)
            row.update(orig_subtasks=so, new_subtasks=sn, same_subtasks=(so is not None and so == sn))
            same_sub += row["same_subtasks"]
        elif use_traces:
            if to is None or tn is None:
                missing_trace += 1
                row.update(identical_trace=False, diverge_step=None, diverge_kind="missing_trace")
            else:
                probs = tw.validate_trace(to) + tw.validate_trace(tn)
                if probs:
                    bad_trace += 1
                    row["trace_problems"] = probs
                c = compare_traces(to, tn)
                row.update(identical_trace=c["identical"] and not probs, diverge_step=c["diverge_step"],
                           diverge_kind=c["diverge_kind"], diverge_field=c["field"])
                identical += row["identical_trace"]
                pj = project_traces(to, tn, _load_arrays(po), _load_arrays(pn))
                row["projection"] = pj
                for d in PROJ_DIMS:
                    for v in ("same", "diff", "not_observed"):
                        proj_tot[d][v] += pj[d][v]
                    if pj[d]["diff"]:
                        ep_diff[d] += 1
        table.append(row)

    stats = terminal_stats([(of[k].get("status"), nf[k].get("status")) for k in both])
    missing = len(only_o) + len(only_n)
    summary: dict[str, Any] = {"mode": mode, "extended": True, "orig_format": orig_format, "compared": len(both),
                               "same_terminal": same_term, "missing": missing,
                               "missing_orig": [list(x) for x in only_n], "missing_new": [list(x) for x in only_o],
                               "duplicate": len(dup), "duplicates": [list(x) for x in dup],
                               "identical_trace": identical, "missing_trace": missing_trace, "bad_trace": bad_trace,
                               "projection": proj_tot, "episodes_diff": {d: ep_diff.get(d, 0) for d in PROJ_DIMS},
                               "not_observed": sum(proj_tot[d]["not_observed"] for d in PROJ_DIMS), **stats,
                               "lang_pairs": lang_pairs, "lang_missing": lang_missing,
                               "prompt_diff": prompt_diff if lang_pairs else None,
                               "reply_diff": reply_diff if lang_pairs else None, "lang_first_diff": lang_first}
    if mode == "astra":
        summary["same_subtasks"] = same_sub
    if groundsg:
        firsts_n, epochs_n = epoch_firsts(new_rows)
        firsts_o, _ = epoch_firsts(orig_rows)
        for r in table:
            kk = (r["task"], r["source_episode"], r["seed"])
            r["server_epoch_first"] = kk in firsts_n
            r["server_epoch_first_orig"] = kk in firsts_o
        summary.update(server_epoch_first=sum(1 for r in table if r["server_epoch_first"]), server_epochs=epochs_n)

    inputs = None
    if manifest is not None:
        man_set = set(manifest)
        accepted_both = set(of) & set(nf)
        inputs = {"expected": len(man_set), "missing": len(man_set - accepted_both),
                  "extra": len((set(of) | set(nf)) - man_set), "unaccepted": ob["unaccepted"] + nb["unaccepted"],
                  "ambiguous": len(set(ob["ambiguous"]) | set(nb["ambiguous"])) + (len(manifest) - len(man_set)),
                  "trace_binding_mismatch": binding_mismatch}
        inputs["verdict"] = "PASS" if all(inputs[x] == 0 for x in ("missing", "extra", "unaccepted", "ambiguous",
                                                                    "trace_binding_mismatch")) else "FAIL"
    if expect_host is not None:
        provenance = {"mode": "host", "expect_host": expect_host, "foreign_rows": foreign_rows,
                      "unknown_rows": unknown_rows,
                      "verdict": "PASS" if foreign_rows == 0 and unknown_rows == 0 else "FAIL"}
    else:
        provenance = {"local_rows": local_rows, "unknown_rows": unknown_rows,
                      "verdict": "PASS" if local_rows == 0 and unknown_rows == 0 else "FAIL"}
    supplement = None
    if orig_supplement is not None and manifest is not None:
        supplement = supplement_check(old_orig, orig_supplement, manifest)
    noise = []
    if noise_runs:
        runs = [("E0", orig_rows)] + [(f"O{i + 1}", r) for i, r in enumerate(noise_runs)]
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                st = _pair_flips(runs[i][1], runs[j][1])
                st["pair"] = f"{runs[i][0]}-{runs[j][0]}"
                noise.append(st)

    reasons = []
    if inputs is not None and inputs["verdict"] == "FAIL":
        reasons.append("inputs")
    if provenance["verdict"] == "FAIL":
        reasons.append("cross_machine")
    if supplement is not None and supplement["verdict"] == "FAIL":
        reasons.append("supplement")
    incomplete = missing > 0 or len(dup) > 0 or (expect_total is not None and len(both) != expect_total)
    if use_traces:
        incomplete = incomplete or missing_trace > 0 or bad_trace > 0
    summary["verdict"] = "INVALID" if reasons else ("INCOMPLETE" if incomplete else "INFO")
    summary["reason"] = ",".join(reasons) or None
    summary["expect_total"] = expect_total
    summary["unaccepted"] = ob["unaccepted"] + nb["unaccepted"]
    return {"summary": summary, "table": table, "inputs": inputs, "provenance": provenance,
            "supplement": supplement, "noise": noise}


def _f4(x: float) -> str:
    return f"{x:.4f}"


def ext_lines(res: dict, policy: str, site: str | None = None) -> list[str]:
    """扩展模式输出行（顺序：输入、来源、补集合、投影、v75、噪声、GATE2）。"""
    s = res["summary"]
    out = []
    i = res.get("inputs")
    if i is not None:
        out.append(f"GATE2_INPUTS={i['verdict']} expected={i['expected']} missing={i['missing']} extra={i['extra']} "
                   f"unaccepted={i['unaccepted']} ambiguous={i['ambiguous']} "
                   f"trace_binding_mismatch={i['trace_binding_mismatch']}")
    p = res["provenance"]
    if p.get("mode") == "host":
        out.append(f"GATE2_PROVENANCE={p['verdict']} mode=host expect_host={p['expect_host']} "
                   f"foreign_rows={p['foreign_rows']} unknown_rows={p['unknown_rows']}")
    else:
        out.append(f"GATE2_PROVENANCE={p['verdict']} local_rows={p['local_rows']} unknown_rows={p['unknown_rows']}")
    sp = res.get("supplement")
    if sp is not None:
        out.append(f"GATE2_SUPPLEMENT={sp['verdict']} old={sp['old']} supplement={sp['supplement']} "
                   f"expected={sp['expected']} missing={sp['missing']} extra={sp['extra']} overlap={sp['overlap']}")
    pj = s["projection"]
    out.append(f"GATE2_PROJ policy={policy} " + " ".join(
        f"{d}={pj[d]['same']}/{pj[d]['diff']}/{pj[d]['not_observed']}" for d in PROJ_DIMS)
        + " episodes_diff=" + ",".join(f"{d}:{s['episodes_diff'][d]}" for d in PROJ_DIMS))
    if s["orig_format"] == "v75":
        out.append(f"ORIG_RERUN_VS_E0=INFO policy={policy} compared={s['compared']} flips={s['flips']} "
                   f"s2f={s['s2f']} f2s={s['f2s']}")
    for n in res.get("noise") or []:
        out.append(f"GATE2_NOISE=INFO policy={policy} pair={n['pair']} compared={n['compared']} s2f={n['s2f']} "
                   f"f2s={n['f2s']} flips={n['flips']} sr_a={_f4(n['sr_orig'])} sr_b={_f4(n['sr_new'])} "
                   f"missing={n['missing']} note=descriptive_history")
    lf = s.get("lang_first_diff")
    if lf is not None:
        out.append(f"GATE2_LANG_FIRST_DIFF policy={policy} task={lf['task']} source_episode={lf['source_episode']} "
                   f"seed={lf['seed']} call={lf['call']} message_index={lf['message_index']} dir={lf['dir']} "
                   f"role={lf['role']} at={lf['at']} orig={json.dumps(lf['orig'], ensure_ascii=False)} "
                   f"new={json.dumps(lf['new'], ensure_ascii=False)}")
    parts = [f"GATE2={s['verdict']}", f"policy={policy}", f"compared={s['compared']}",
             f"same_terminal={s['same_terminal']}", f"s2f={s['s2f']}", f"f2s={s['f2s']}",
             f"sr_orig={_f4(s['sr_orig'])}", f"sr_new={_f4(s['sr_new'])}", f"sr_diff_pp={s['sr_diff_pp']:.2f}",
             f"mcnemar_p={s['mcnemar_p']:.4g}", f"not_observed={s['not_observed']}",
             f"identical_trace={s['identical_trace']}", f"missing={s['missing']}", f"duplicate={s['duplicate']}",
             f"missing_trace={s['missing_trace']}", f"bad_trace={s['bad_trace']}",
             f"matrix={_matrix_str(s['matrix'])}", f"other_terminal={s['other_terminal']}",
             f"prompt_diff={'NA' if s.get('prompt_diff') is None else s['prompt_diff']}",
             f"reply_diff={'NA' if s.get('reply_diff') is None else s['reply_diff']}"]
    if s.get("lang_missing"):
        parts.append(f"lang_missing={s['lang_missing']}")
    if s["mode"] == "astra":
        parts.append(f"mode=astra same_subtasks={s['same_subtasks']}")
    if "server_epoch_first" in s:
        parts.append(f"server_epoch_first={s['server_epoch_first']}")
    if s.get("expect_total") is not None:
        parts.append(f"expect_total={s['expect_total']}")
    if s.get("reason"):
        parts.append(f"reason={s['reason']}")
    if site:
        parts.append(f"site={site}")
    out.append(" ".join(parts))
    return out


def to_markdown_ext(res: dict, policy: str) -> str:
    s = res["summary"]
    lines = [f"# 第二档差异表（扩展）：{policy}", "",
             f"- 判定：`{s['verdict']}`{'（' + s['reason'] + '）' if s.get('reason') else ''}；配对 {s['compared']} 个身份，"
             f"同终态 {s['same_terminal']}；s2f {s['s2f']}、f2s {s['f2s']}；成功率 原 {_f4(s['sr_orig'])} → 新 "
             f"{_f4(s['sr_new'])}（{s['sr_diff_pp']:.2f} pp）；McNemar p = {s['mcnemar_p']:.4g}。",
             "- 投影各维度为「相同/不同/不可观察」项数；`not_observed` 不算相同（C9）。", "",
             "| 任务 | 原 episode | seed | 原侧终态 | 新侧终态 | 同终态 | " + " | ".join(PROJ_DIMS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(PROJ_DIMS)]
    for r in res["table"]:
        pj = r.get("projection")
        cells = [f"{pj[d]['same']}/{pj[d]['diff']}/{pj[d]['not_observed']}" if pj else "—" for d in PROJ_DIMS]
        lines.append(f"| {r['task']} | {r['source_episode']} | {r['seed']} | {r['orig_status']} | {r['new_status']} | "
                     f"{int(r['same_terminal'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="第二档：原侧对新侧逐身份差异表")
    ap.add_argument("--policy", required=True, help="报告用的模型标签，如 groundsg-oracle、pp、astra")
    ap.add_argument("--orig-results", nargs="+", required=True)
    ap.add_argument("--new-results", nargs="+", required=True)
    ap.add_argument("--orig-traces", default=None)
    ap.add_argument("--new-traces", default=None)
    ap.add_argument("--mode", choices=["full", "astra"], default="full")
    ap.add_argument("--groundsg", action="store_true", help="另出 first_episode_identical（服务启动边界见 server_epoch）")
    ap.add_argument("--expect-total", type=int, default=None)
    ap.add_argument("--site", default=None, help="如 local：判定行附 site=local")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    g = ap.add_argument_group("第二阶段扩展（传任一项即启用扩展模式，见模块说明）")
    g.add_argument("--manifest", default=None, help="冻结身份清单（jsonl 或 JSON 列表）")
    g.add_argument("--new-ledger", nargs="+", default=None, help="新侧 AttemptLedger jsonl（可多份）")
    g.add_argument("--new-attempts", default=None, help="新侧 attempt 映射（与 --orig-attempts 同格式；新侧位置放原侧重跑时用）")
    g.add_argument("--orig-attempts", default=None, help="原侧 attempt 映射 orig-attempts.json（S7 适配器产出）")
    g.add_argument("--orig-supplement", nargs="+", default=None, help="本轮原侧补跑结果 jsonl（补集合核对，需 --manifest）")
    g.add_argument("--orig-format", choices=["default", "v75"], default=None, help="v75：原侧读 E0 逐局文件")
    g.add_argument("--v75-input-manifest", default=None, help=f"E0 sha256 清单，默认 {DEFAULT_V75_MANIFEST}")
    g.add_argument("--noise-runs", nargs="+", default=None, help="O1 O2（文件或目录）：输出三对两方向翻转（需 --orig-format v75）")
    g.add_argument("--expect-host", default=None,
                   help="来源主机模式：两侧每行 node／host 的短主机名都必须等于它（如 sled-vail 本机对拍）；不给时按 GL 判")
    return ap


def _extended(args) -> bool:
    return any(getattr(args, a) is not None for a in ("manifest", "new_ledger", "new_attempts", "orig_attempts",
                                                       "orig_supplement", "orig_format", "v75_input_manifest",
                                                       "noise_runs", "expect_host"))


def main_ext(args) -> int:
    v75 = args.orig_format == "v75"
    if args.noise_runs and not v75:
        print("GATE2=ERROR reason=noise_runs_need_orig_format_v75", flush=True)
        return 2
    sha_checked: list[dict] = []
    if v75:
        man = args.v75_input_manifest or DEFAULT_V75_MANIFEST
        try:
            sha_checked = verify_v75_inputs(args.orig_results, man)
        except (InputShaMismatch, FileNotFoundError) as e:
            print(f"GATE2_INPUT_SHA=FAIL {e}", flush=True)
            print(f"GATE2=ERROR policy={args.policy} reason=input_sha", flush=True)
            return 2
        print(f"GATE2_INPUT_SHA=PASS files={len(sha_checked)}", flush=True)
    orig_rows = read_rows(args.orig_results)
    if v75:
        _v75_normalize(orig_rows)
    noise = None
    if args.noise_runs:
        noise = [_v75_normalize(read_rows(expand_run(p))) for p in args.noise_runs]
    res = compare_ext(
        orig_rows, read_rows(args.new_results), args.orig_traces, args.new_traces, mode=args.mode,
        groundsg=args.groundsg, expect_total=args.expect_total,
        manifest=read_manifest(args.manifest) if args.manifest else None,
        new_ledger=read_ledgers(args.new_ledger) if args.new_ledger else None,
        new_attempts=read_attempt_map(args.new_attempts) if args.new_attempts else None,
        orig_attempts=read_attempt_map(args.orig_attempts) if args.orig_attempts else None,
        orig_supplement=read_rows(args.orig_supplement) if args.orig_supplement else None,
        orig_format="v75" if v75 else "default", noise_runs=noise, expect_host=args.expect_host)
    lines = ext_lines(res, args.policy, args.site)
    res["summary"].update(policy=args.policy, site=args.site, line=lines[-1], lines=lines, input_sha=sha_checked)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(to_markdown_ext(res, args.policy), encoding="utf-8")
    for ln in lines:
        print(ln, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _extended(args):
        return main_ext(args)
    res = compare(read_rows(args.orig_results), read_rows(args.new_results), args.orig_traces, args.new_traces,
                  mode=args.mode, groundsg=args.groundsg, expect_total=args.expect_total)
    res["summary"].update(policy=args.policy, site=args.site)
    line = verdict_line(res, args.policy, args.site)
    res["summary"]["line"] = line
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(to_markdown(res, args.policy), encoding="utf-8")
    print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
