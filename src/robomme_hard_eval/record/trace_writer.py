"""每局轨迹记录 ``trace.jsonl``（1003-oracle-subgoal-groundsg-eval-plan.md 第二部分 1.7）。

本文件先由主会话以「接口桩」形式提交（计划 1.8），S3～S5 的原侧／新侧驱动按这里的签名与字段约定调用；
S7 合入时补全实现与测试，签名与字段名不改。第二档对比工具 ``gate2_compare.py`` 只读本文件写出的字段。

文件格式：一局一个 ``trace.jsonl``，逐行一个 JSON 对象，``kind`` 区分行类型：

- ``header``（恰好一行，第一行）：``route``、``identity``（task / source_episode / seed / tier / dataset 等，
  由调用方给出）、``max_steps``、``schema``。
- ``demo``（恰好一行，在首个 step 之前；其前只允许 header／request／response／history——GroundSG 官方循环先 reset 策略再取初始观测，
  故 request 行可先于 demo；无演示时 ``frames=0``）：演示帧数、逐帧 front／wrist 画面 sha256、
  演示阶段的状态与文本。
- ``request``：发给模型的每次请求（GroundSG 的 ``reset``／``add_buffer``／``infer``，PonderPounce 的每个协议帧，
  Astra 的规划／监视请求）规范化字节的 sha256 与字节数、``step``（发生在第几步之前）。
- ``response``：模型返回的完整动作块（``array_record``）。
- ``step``：每执行一步一行：``step``（从 1 计）、``front_sha256``、``wrist_sha256``、``state``、``action``
  （均为 ``array_record``：原始 dtype、shape、sha256，另附 float32×8 的 hex 便于人读）、``subgoal``、
  ``terminated``、``truncated``、``status``。画面哈希在画面进入视频编码器之前计算。
- ``history``：历史缓冲边界（如 FrameSamp+Modulation ``add_buffer`` 覆盖的步号区间）。
- ``end``（恰好一行，最后一行）：``status``、``exec_steps``、``terminal_reason`` 与调用方附加字段。

``identical_trace`` 的含义只限于上述字段逐项相等（报告里照此写明覆盖范围）。

S7 补充（只加不改）：

- ``canonical_bytes(obj)``：把请求对象（dict／list／数组／标量混合）规范化成确定的字节，供调用方在没有现成
  序列化字节时传给 ``log_request``；数组按原始 dtype、shape 与 sha256 表示，不先转 float32。
- ``validate_trace(rows)``：行序与结构自检（header 首行、demo 唯一且在首个 step 之前、end 末行且唯一、step 从 1 连续、
  ``end.exec_steps`` 等于最后一步），返回问题列表，空列表即合规。
- ``find_traces(root)``、``subgoal_sequence(rows)``：供 ``gate2_compare.py`` 使用。
- ``TraceWriter`` 在 ``close`` 之后再写任何行会抛 ``RuntimeError``（避免收尾后的迟到写入悄悄丢失）。
- 身份建议字段：``identity`` 里放 ``task``、``source_episode``、``seed``、``tier``、``dataset``，以及 ``attempt``
  （第几次尝试）；``gate2_compare`` 先按 ``(task, source_episode, seed)`` 配对，同一身份多份轨迹时再按 ``attempt``
  对上结果行。

第二阶段共享契约 C1～C11（1005-eval-video-phase2-all-models-rerun-plan.md 第二部分〇节；S0 由主会话写入，
各路线子任务按此落地，测试助手 ``tests/pipeline/evalx/report/trace_contract.py`` 逐条核对）：

- C1 ``route``：新侧一律 ``<模型>/new``（``groundsg/<variant>/new``、``pp/new``、``astra/new``、``smvla/new``、
  ``perceptual-framesamp-modul/new``），原侧 ``<模型>/orig``（``groundsg/<variant>/orig``、``pp/orig``、``smvla/orig``、``perceptual-framesamp-modul/orig``）。
- C2 演示段记全部 reset 帧（含最后一帧初始画面），``demo.frames == len(demo.states)``；收尾
  ``close(..., demo_frames=<演示帧数，不含初始帧>)``，满足 ``demo.frames == end.demo_frames + 1``。
- C3 ``end.status`` 与 ``end.terminal_reason`` 取 ``success``／``fail``／``timeout``／``error``；strict-cap 命中一律
  ``timeout``；``error`` 局允许无帧（``end.no_frame=true``，此时演示段可为空、``demo_frames`` 记 0），重绘器对
  无帧局记原因不出视频。
- C4 每步记执行后的画面、状态、动作与当步子目标；动作按实际交给环境的原 dtype／shape／bytes 记录，不为契约
  转换；非 float32 动作的原值写同目录 ``arrays.npz``，键 ``exec_action__%05d``（按 0 起的步序号，即 ``step - 1``），
  一旦写 ``arrays.npz`` 则每个执行步都要有键。
- C5 ``trace.jsonl``、``arrays.npz``、原始帧放同一局目录。
- C6 ``identity`` 含 ``task``、``tier``、``seed``、``dataset``、``source_episode`` 或 ``builder_episode``、与局目录名
  ``<key>.a<N>`` 一致的 ``key``，以及 ``attempt``（= 账本 ``accepted_attempt_id`` 对应的尝试号 N）。
- C7 子目标为 ``None`` 表示模型等待中（官方录像以 ``[initializing...]`` 占位），轨迹里保留原始 ``None``；没有
  子目标功能的路线（FrameSamp+Modulation）全程 ``None``。
- C8 计数三分，写进 ``end``：``steps_attempted``（交给环境的步数，含异常步，= 结果行 ``exec_steps``）、
  ``steps_observed``（返回有效观测的步数）、``frames_recorded``（官方实际录制帧数
  = ``demo_frames + 1 + steps_observed - omitted_timeout_frames``）；缺观测步用 ``log_missing_step`` 保留步号、
  动作与原因（``observed=false``），不删不补。
- C9 两侧不可同时观察的字段写 ``NOT_OBSERVED``（如原侧的 ``terminated``／``truncated``）；比较器不把双
  ``NOT_OBSERVED`` 算相同，按维度报 ``not_observed=<n>``。
- C10 请求／响应／历史边界按模型定义「共同逻辑输入」：GroundSG、PonderPounce、FrameSamp+Modulation 两侧同协议，比原始哈希；
  SimpleMemVLA 原侧内嵌、新侧 websocket，只比逻辑输入（指令、状态、帧哈希序列）与完整动作块。
- C11 ``end.observer_hook_errors=<n>``（只读观测器路线必写）；大于 0 时该局 ``TRACE_COMPLETE`` 计失败，不影响
  任务成绩。

为落地 C3、C8、C9，S0 只增不改地加了 ``NOT_OBSERVED``、``UNSET`` 两个常量，``log_step`` 的 ``terminated``／
``truncated`` 接受 ``NOT_OBSERVED`` 原样写出并接受附加字段 ``**extra``，以及 ``log_missing_step``；不传这些新值时
写出字节与此前完全相同。共享函数新增的可选参数一律用 ``UNSET`` 作「未提供」哨兵，``None`` 只表示「模型等待中」。

第三阶段（``docs/plans/1006-stage3-interface-freeze.md`` 第四、五节，R6）只增不改：

- ``header`` 可选新增 ``policy_seed``、``effective_cap``（构造参数缺省 ``None`` 时不写，旧字节不变）；给了
  ``policy_seed`` 时 ``end`` 行也带 ``policy_seed``。
- ``log_step``／``log_missing_step`` 可选新增 ``source_call_id``、``chunk_index``（语言账本关联；不传时 step 行
  仍是旧 9 键 + 旧附加键）。
- 完整数组：构造参数 ``arrays_path``（缺省 ``<trace 同目录>/arrays.npz``）与开关 ``collect_arrays``（缺省 True）。
  每个 attempted 步收 ``exec_action__%05d``（键号 = step−1），有状态的观测步另收 ``exec_state__%05d``；缺观测步
  不补零，步号记进 ``end.arrays.missing_state_steps``。``close()`` 经 ``merge_write_npz`` 合并写盘（与录像器等
  其他写者先后任意都不互相覆盖），``end`` 行写 ``arrays`` 摘要；合并冲突不抛出，记 ``arrays.error``。
- ``merge_write_npz(path, mapping)``：``arrays.npz`` 唯一允许的写法（同键 dtype／shape／sha256 须全同，否则
  ``ArraysConflict``；目录级排他锁 + ``path.tmp`` + ``os.replace`` 原子写）。
- ``LanguageLog(path)``：每局 ``language.jsonl`` 语言账本（call_open／message／call_close／reuse 四种行）。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "sgeval-trace/1"
NOT_OBSERVED = "NOT_OBSERVED"  # C9：该侧不可观察的字段


class _Unset:
    """「未提供」哨兵（C7／R2）：与 ``None``（模型等待中）区分。"""

    _inst = None

    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


def _flag(x: Any) -> Any:
    return NOT_OBSERVED if isinstance(x, str) and x == NOT_OBSERVED else bool(x)


def image_sha256(img: Any) -> str | None:
    """画面的 sha256：按原始 dtype、shape 的连续字节计算；``None`` 原样返回 ``None``。"""
    if img is None:
        return None
    arr = np.ascontiguousarray(np.asarray(img))
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def array_record(x: Any) -> dict | None:
    """数组的身份记录：``{"dtype","shape","sha256","f32hex"}``；不先转 float32 再哈希。

    ``f32hex`` 只为人读：展平后转 float32 的前 8 个值的 hex，不参与相等判定。
    """
    if x is None:
        return None
    arr = np.ascontiguousarray(np.asarray(x))
    h = hashlib.sha256()
    h.update(arr.tobytes())
    flat = arr.reshape(-1)
    head = flat[:8].astype(np.float32) if flat.dtype.kind in "fiub" else np.zeros(0, np.float32)
    return {"dtype": arr.dtype.str, "shape": list(arr.shape), "sha256": h.hexdigest(), "f32hex": head.tobytes().hex()}


def bytes_record(payload: bytes) -> dict:
    """请求的规范化字节记录：``{"sha256","nbytes"}``。"""
    return {"sha256": hashlib.sha256(payload).hexdigest(), "nbytes": len(payload)}


ACTION_KEY = "exec_action__%05d"  # 键号 = step − 1（C4）
STATE_KEY = "exec_state__%05d"    # 键号 = step − 1；只有有状态的观测步才有


class ArraysConflict(ValueError):
    """``merge_write_npz`` 发现同键数组的 dtype／shape／sha256 与已有文件不一致。"""


def _array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


#: ``merge_write_npz`` 等目录锁的上限（秒）；超时抛 ``TimeoutError``（``TraceWriter.close`` 记进 ``end.arrays.error``）
MERGE_LOCK_TIMEOUT_S = 600.0
_MERGE_LOCK_POLL_S = 0.05


def _lock_dir(directory: Path, timeout_s: float | None = None):
    """目录级排他锁（``fcntl.flock`` 锁目录本身，不在局目录里留锁文件）；没有 fcntl 时返回 None（不加锁）。

    **只用于本地盘**：正式运行的逐局产物在节点本地写（不写 NFS），``flock`` 在 NFS 上的语义不可靠，不要把局目录
    放到网络盘上再依赖这把锁。非阻塞轮询，等满 ``timeout_s``（缺省 ``MERGE_LOCK_TIMEOUT_S``）仍拿不到即抛
    ``TimeoutError``，不无限挂住收尾。"""
    try:
        import fcntl
    except ImportError:  # pragma: no cover 非 POSIX
        return None
    timeout_s = MERGE_LOCK_TIMEOUT_S if timeout_s is None else float(timeout_s)
    deadline = time.monotonic() + timeout_s
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"merge_write_npz 等目录锁超时 {timeout_s:g}s：{directory}") from None
                time.sleep(_MERGE_LOCK_POLL_S)
    except BaseException:
        os.close(fd)
        raise


def _unlock_dir(fd) -> None:
    if fd is None:
        return
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def merge_write_npz(path: str | Path, mapping: dict) -> None:
    """``arrays.npz`` 唯一允许的写法：读已有文件 → 同键须 dtype／shape／sha256 全同（否则抛 ``ArraysConflict``，
    不写任何字节）→ 合并 → 写 ``<path>.tmp`` 后 ``os.replace``。

    同一目录的并发写者经目录级 ``flock`` 串行（只用于本地盘；等锁超过 ``MERGE_LOCK_TIMEOUT_S`` 抛 ``TimeoutError``）；
    ``mapping`` 为空且文件不存在时不建文件；没有新键时不重写。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = {str(k): np.array(v, copy=True) for k, v in dict(mapping).items()}
    for k, a in new.items():
        if a.dtype.hasobject:
            raise TypeError(f"merge_write_npz 拒绝 object 数组：{k}")
    fd = _lock_dir(path.parent)
    try:
        existing: dict[str, np.ndarray] = {}
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                existing = {k: z[k] for k in z.files}
        conflicts = []
        for k, a in new.items():
            old = existing.get(k)
            if old is None:
                continue
            if old.dtype.str != a.dtype.str or tuple(old.shape) != tuple(a.shape) or \
                    _array_sha256(old) != _array_sha256(a):
                conflicts.append(f"{k}: 已有 {old.dtype.str}{list(old.shape)} 新 {a.dtype.str}{list(a.shape)}")
        if conflicts:
            raise ArraysConflict(f"{path} 同键不一致 {len(conflicts)} 个：" + "；".join(conflicts[:5]))
        added = [k for k in new if k not in existing]
        if not added:
            return
        merged = {**existing, **{k: new[k] for k in added}}
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as fh:  # 传文件对象：np.savez 不会自作主张补 .npz 后缀
                np.savez(fh, **merged)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:  # 写到一半失败：旧文件原样保留，不留半截临时文件
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
    finally:
        _unlock_dir(fd)


class TraceWriter:
    """一局一个实例；按调用顺序逐行追加，``close`` 写 ``end`` 行。上下文管理器退出时未 close 则以 ``status="error"`` 收尾。"""

    def __init__(self, path: str | Path, *, route: str, identity: dict, max_steps: int,
                 policy_seed: int | None = None, effective_cap: int | None = None,
                 arrays_path: str | Path | None = None, collect_arrays: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False
        self.exec_steps = 0
        self.policy_seed = None if policy_seed is None else int(policy_seed)
        self.arrays_path = Path(arrays_path) if arrays_path is not None else self.path.parent / "arrays.npz"
        self.collect_arrays = bool(collect_arrays)
        self._actions: dict[int, np.ndarray] = {}
        self._states: dict[int, np.ndarray] = {}
        self._missing_state: set[int] = set()
        header = {"kind": "header", "schema": SCHEMA, "route": route, "identity": identity, "max_steps": int(max_steps)}
        if policy_seed is not None:
            header["policy_seed"] = int(policy_seed)
        if effective_cap is not None:
            header["effective_cap"] = int(effective_cap)
        self._write(header)
        self._demo_written = False

    def _write(self, row: dict) -> None:
        if self._closed:
            raise RuntimeError(f"trace 已收尾，拒绝追加 {row.get('kind')} 行：{self.path}")
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def log_demo(self, fronts: list, wrists: list, states: list | None = None, texts: list | None = None) -> None:
        """演示阶段（只调一次）：逐帧画面哈希、状态与文本。"""
        assert not self._demo_written, "log_demo 只能调用一次"
        self._demo_written = True
        self._write({
            "kind": "demo",
            "frames": len(fronts),
            "front_sha256": [image_sha256(f) for f in fronts],
            "wrist_sha256": [image_sha256(w) for w in wrists],
            "states": [array_record(s) for s in (states or [])],
            "texts": list(texts or []),
        })

    def log_request(self, name: str, payload: bytes, *, step: int) -> None:
        """发给模型的一次请求；``payload`` 为调用方规范化后的字节（如 msgpack 序列化结果）。"""
        self._write({"kind": "request", "name": name, "step": int(step), **bytes_record(payload)})

    def log_response(self, actions: Any, *, step: int) -> None:
        """模型返回的完整动作块。"""
        self._write({"kind": "response", "step": int(step), "actions": array_record(actions)})

    def log_history(self, start_step: int, end_step: int, *, note: str = "") -> None:
        """历史缓冲边界（闭区间步号）。"""
        self._write({"kind": "history", "start": int(start_step), "end": int(end_step), "note": note})

    def log_step(self, *, step: int, front: Any, wrist: Any, state: Any, action: Any, subgoal: str | None,
                 terminated: bool, truncated: bool, status: str | None, source_call_id: str | None = None,
                 chunk_index: int | None = None, **extra: Any) -> None:
        """执行完第 ``step`` 步（从 1 计）后的一行；画面为执行后的观测。

        ``terminated``／``truncated`` 可传 ``NOT_OBSERVED``（C9）；``extra`` 原样并入该行（不传时与旧格式逐字节相同）。
        ``source_call_id``／``chunk_index``（第三阶段）：本步动作来自语言账本里哪次调用、动作块内第几个；``None`` 不写。
        ``collect_arrays`` 打开时另收本步动作原值（``exec_action__%05d``）与状态原值（``exec_state__%05d``，无状态不补零）。
        """
        if not self._demo_written:
            self.log_demo([], [])
        self.exec_steps = max(self.exec_steps, int(step))
        row = {
            "kind": "step", "step": int(step),
            "front_sha256": image_sha256(front), "wrist_sha256": image_sha256(wrist),
            "state": array_record(state), "action": array_record(action),
            "subgoal": subgoal, "terminated": _flag(terminated), "truncated": _flag(truncated), "status": status,
            **extra,
        }
        if source_call_id is not None:
            row["source_call_id"] = source_call_id
        if chunk_index is not None:
            row["chunk_index"] = int(chunk_index)
        self._write(row)
        if self.collect_arrays:
            k = int(step) - 1
            if action is not None:
                self._actions[k] = np.array(action, copy=True)
            if state is not None:
                self._states[k] = np.array(state, copy=True)
                self._missing_state.discard(int(step))
            else:
                self._missing_state.add(int(step))

    def log_missing_step(self, *, step: int, action: Any, reason: str, subgoal: str | None = None,
                         source_call_id: str | None = None, chunk_index: int | None = None, **extra: Any) -> None:
        """C8：动作已交给环境但没有返回有效观测的一步；保留步号、动作与原因，画面与状态记 ``None``。"""
        self.log_step(step=step, front=None, wrist=None, state=None, action=action, subgoal=subgoal,
                      terminated=NOT_OBSERVED, truncated=NOT_OBSERVED, status=None,
                      source_call_id=source_call_id, chunk_index=chunk_index,
                      observed=False, missing_reason=str(reason), **extra)

    def _arrays_summary(self) -> dict:
        """收尾写 ``arrays.npz`` 并返回 ``end.arrays`` 摘要；写盘失败（含 ``ArraysConflict``）记 ``error``、不抛出。"""
        try:
            rel = os.path.relpath(self.arrays_path, self.path.parent)
        except ValueError:  # pragma: no cover 跨盘符
            rel = str(self.arrays_path)
        summary: dict[str, Any] = {"path": rel.replace(os.sep, "/"), "action_keys": len(self._actions),
                                   "state_keys": len(self._states),
                                   "missing_state_steps": sorted(self._missing_state)}
        mapping = {ACTION_KEY % k: a for k, a in sorted(self._actions.items())}
        mapping.update({STATE_KEY % k: a for k, a in sorted(self._states.items())})
        if mapping:
            try:
                merge_write_npz(self.arrays_path, mapping)
            except Exception as e:  # noqa: BLE001 收尾不因数组写盘失败丢 end 行；检查器据 error 判 FAIL
                summary["error"] = f"{type(e).__name__}: {e}"[:600]
        return summary

    def close(self, *, status: str, terminal_reason: str | None = None, **extra: Any) -> None:
        if self._closed:
            return
        if not self._demo_written:
            self.log_demo([], [])
        row = {"kind": "end", "status": status, "exec_steps": self.exec_steps,
               "terminal_reason": terminal_reason, **extra}
        if self.policy_seed is not None and "policy_seed" not in extra:
            row["policy_seed"] = self.policy_seed
        if self.collect_arrays:
            if "arrays" in extra:  # 调用方旧口径的 arrays 字符串让位给冻结说明的摘要，原值另存
                row["arrays_caller"] = extra["arrays"]
            row["arrays"] = self._arrays_summary()
        self._write(row)
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close(status="error", terminal_reason=f"exception:{exc_type.__name__}" if exc_type else "unclosed")


# ── 第三阶段：语言账本 language.jsonl（冻结说明第五节）───────────────────────────

LANG_MODELS = ("subgoal_model", "action_model", "planner", "monitor")
LANG_DIRS = ("in", "out")
LANG_ROLES = ("system", "user", "assistant", "fields")
LANG_STATUSES = ("reply", "error", "cancelled")
LANG_FALLBACKS = (None, "last_valid", "model_response_error", "continue_last")
#: 服务外壳回包里的审计键（R3 写入；客户端在把动作交给环境前 pop 掉）
AUDIT_KEY = "_sgeval_audit"


def _jsonable(x: Any) -> Any:
    """numpy 标量／数组、bytes、tuple、Path 转成 JSON 可写的值（文字原样，不截断）。"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (bytes, bytearray, memoryview)):
        b = bytes(x)
        return {"__bytes_sha256__": hashlib.sha256(b).hexdigest(), "nbytes": len(b)}
    if isinstance(x, Path):
        return str(x)
    return x


class LanguageLog:
    """每局一份 ``language.jsonl``：按真实模型调用记账（冻结说明第五节）。

    每行 ``ensure_ascii=False``、写后立即 ``flush``；``dir="in"`` 的消息调用方须在真实发送前写入（写入即落盘）。
    ``close()`` 给仍未关闭的调用补 ``call_close status=cancelled`` 再关文件；收尾后再写抛 ``RuntimeError``。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False
        self._seq = 0
        self._open: dict[str, int] = {}  # 未关闭的调用 → 下一条消息序号
        self._known: set[str] = set()

    @staticmethod
    def _ts() -> float:
        return round(time.time(), 6)

    def _write(self, row: dict) -> None:
        if self._closed:
            raise RuntimeError(f"language 账本已收尾，拒绝追加 {row.get('kind')} 行：{self.path}")
        self._fh.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        self._fh.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def open_calls(self) -> list[str]:
        return list(self._open)

    def open_call(self, model: str, step: int, *, params: dict | None = None, transport_attempt: int = 0,
                  retry: int = 0) -> str:
        if model not in LANG_MODELS:
            raise ValueError(f"未知 model={model!r}，应为 {LANG_MODELS}")
        call_id = f"c{self._seq + 1:05d}"
        self._write({"kind": "call_open", "call_id": call_id, "model": model, "step": int(step), "params": params,
                     "transport_attempt": int(transport_attempt), "retry": int(retry), "ts": self._ts()})
        self._seq += 1
        self._open[call_id] = 0
        self._known.add(call_id)
        return call_id

    def message(self, call_id: str, *, dir: str, role: str, text: Any, images: list | None = None,  # noqa: A002
                channel: str | None = None, token_ids: Any = None, mask: Any = None, tokenizer: Any = None,
                truncated: Any = None, demo_video: Any = None) -> int:
        if call_id not in self._open:
            raise ValueError(f"调用 {call_id!r} 未打开或已关闭")
        if dir not in LANG_DIRS:
            raise ValueError(f"未知 dir={dir!r}")
        if role not in LANG_ROLES:
            raise ValueError(f"未知 role={role!r}")
        idx = self._open[call_id]
        self._write({"kind": "message", "call_id": call_id, "message_index": idx, "dir": dir, "role": role,
                     "text": text, "images": images, "channel": channel, "token_ids": token_ids, "mask": mask,
                     "tokenizer": tokenizer, "truncated": truncated, "demo_video": demo_video, "ts": self._ts()})
        self._open[call_id] = idx + 1
        return idx

    def close_call(self, call_id: str, *, status: str, parsed: Any = None, fallback: str | None = None,
                   server_final_text: Any = None, server_truncated: Any = None) -> None:
        if call_id not in self._open:
            raise ValueError(f"调用 {call_id!r} 未打开或已关闭")
        if status not in LANG_STATUSES:
            raise ValueError(f"未知 status={status!r}")
        if fallback not in LANG_FALLBACKS:
            raise ValueError(f"未知 fallback={fallback!r}")
        self._write({"kind": "call_close", "call_id": call_id, "status": status, "parsed": parsed,
                     "fallback": fallback, "server_final_text": server_final_text,
                     "server_truncated": server_truncated, "ts": self._ts()})
        del self._open[call_id]

    def reuse(self, step: int, reused_call_id: str) -> None:
        if reused_call_id not in self._known:
            raise ValueError(f"复用的调用 {reused_call_id!r} 不存在")
        self._write({"kind": "reuse", "step": int(step), "reused_call_id": reused_call_id, "reused_previous": True})

    def close(self) -> None:
        if self._closed:
            return
        try:
            for cid in list(self._open):
                self.close_call(cid, status="cancelled")
        finally:
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "LanguageLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_language(path: str | Path) -> list[dict]:
    """读回一局语言账本。"""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def audit_channel_messages(lang: "LanguageLog", call_id: str, audit: Any) -> tuple[Any, Any]:
    """把服务外壳审计块 ``{"channels":[…], "server_final_text", …}`` 的逐通道分词写成 ``dir=in role=fields`` 消息；
    返回 ``(server_final_text, server_truncated)``。``audit`` 为 None 或不是 dict 时不写、返回 ``(None, None)``。"""
    if not isinstance(audit, dict):
        return None, None
    truncs = []
    for ch in audit.get("channels") or []:
        if not isinstance(ch, dict):
            continue
        truncs.append(ch.get("truncated"))
        lang.message(call_id, dir="in", role="fields", text=ch.get("text"), channel=ch.get("channel"),
                     token_ids=ch.get("token_ids"), mask=ch.get("mask"), tokenizer=ch.get("tokenizer"),
                     truncated=ch.get("truncated"))
    known = [t for t in truncs if t is not None]
    server_truncated = audit.get("server_truncated", any(bool(t) for t in known) if known else None)
    return audit.get("server_final_text"), server_truncated


def read_trace(path: str | Path) -> list[dict]:
    """读回一局轨迹（供 gate2_compare 与测试使用）。"""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── S7 补充：规范化字节、结构自检、检索 ────────────────────────────────────────


def _canon(obj: Any) -> Any:
    """递归规范化：数组 → 原始 dtype/shape/sha256 记录；bytes → sha256；dict 键转字符串（json 再按键排序）。"""
    if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
        rec = array_record(obj)
        return {"__array__": [rec["dtype"], rec["shape"], rec["sha256"]]}
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"__bytes__": hashlib.sha256(bytes(obj)).hexdigest()}
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        # float 用 repr 精确往返，避免 json 实现差异
        return {"__float__": float.hex(obj)}
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return {"__repr__": repr(obj)}


def canonical_bytes(obj: Any) -> bytes:
    """请求对象的规范化字节：同一内容恒得同一字节；数组按原始 dtype 与 shape 记，不转 float32。"""
    return json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_trace(rows: list[dict]) -> list[str]:
    """结构自检，返回问题列表（空即合规）。"""
    problems: list[str] = []
    if not rows:
        return ["空轨迹"]
    if rows[0].get("kind") != "header":
        problems.append("首行不是 header")
    if rows[0].get("schema") != SCHEMA:
        problems.append(f"schema 不是 {SCHEMA}")
    kinds = [r.get("kind") for r in rows]
    if kinds.count("header") != 1:
        problems.append(f"header 行数 {kinds.count('header')}")
    if kinds.count("demo") != 1:
        problems.append(f"demo 行数 {kinds.count('demo')}")
    else:
        i = kinds.index("demo")
        if any(k not in ("header", "request", "response", "history") for k in kinds[1:i]):
            problems.append("demo 不在首个 step 之前")
    if kinds.count("end") != 1 or kinds[-1] != "end":
        problems.append("end 不是唯一末行")
    steps = [int(r["step"]) for r in rows if r.get("kind") == "step"]
    if steps != list(range(1, len(steps) + 1)):
        problems.append("step 不是从 1 连续递增")
    if rows[-1].get("kind") == "end" and int(rows[-1].get("exec_steps", -1)) != (steps[-1] if steps else 0):
        problems.append("end.exec_steps 与最后一步不符")
    return problems


def find_traces(root: str | Path) -> list[Path]:
    """递归找 ``trace.jsonl``（排序后返回，结果确定）。"""
    return sorted(Path(root).rglob("trace.jsonl"))


def subgoal_sequence(rows: list[dict]) -> list[str]:
    """逐步 ``subgoal`` 去掉连续重复与空值后的子任务序列（Astra 模式比较用）。"""
    seq: list[str] = []
    for r in rows:
        if r.get("kind") != "step":
            continue
        s = r.get("subgoal")
        if s is None or s == "":
            continue
        if not seq or seq[-1] != s:
            seq.append(s)
    return seq
