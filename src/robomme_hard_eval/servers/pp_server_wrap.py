#!/usr/bin/env python3
"""PonderPounce 服务外壳：在每个 ACTION 回包里附上模型自己的子目标（1005-eval-video-phase2-all-models-rerun-plan.md
第二部分一节「S5 PonderPounce 新侧」、八节 2.2）。

上游 ``ponderpounce.eval.robomme_server.PonderPounceRoboMMEServer``（子模块锁定 ``723df357``）把 System 2 产出的
子目标文本只写进服务端日志，协议里不回传。本外壳子类化原类（原文件一行不改），只覆写两个方法：

- ``_fire_s1``：调父类之前，从本局 ``ep.cognitions`` 里取「到 ``now`` 为止已可见（``ready_at_ns <= now``）且子目标
  非空」的最新一条，更新本局「最近一个已可见且非空的子目标」（没有新的就沿用旧值，本局开头为 ``None``）；
  父类生成 ``ep.chunk`` 之后把这个值挂到 chunk 上。默认节拍（Ponder／Pounce 各 1000 ms）下每次 Pounce 触发之间恰
  有一条 cognition 变为可见，与只看 ``_visible_cognition(ep, now)`` 等价；节拍不等时也不会漏掉夹在两次触发之间
  变为可见的子目标。
- ``_dispense``：父类结果原样保留，另加 ``"subgoal": <当前 chunk 上的子目标>``；没有 chunk（手臂 hold）时为
  ``None``。chunk 用尽后父类重复最后一行动作，子目标同样不变。

外壳不读写随机数发生器、不改 ``cursor``／触发计数／动作，只在 ``Episode``、``Chunk`` 实例上各加一个私有属性
（``_sgeval_subgoal``）。等价性由 ``tests/pipeline/evalx/pp/test_pp_server_wrap.py``（``PP_SERVER_ACTION_EQ``）钉死。

第三阶段（1006 计划八.11／八.12 第 5、7 条；接口冻结说明 2.2、五节「服务外壳回包审计键」）：

- 每个 ACTION 回包另加 ``_sgeval_audit``：``channels`` 为 System 1 本局实际用的 prompt（``Task: …;\nAction: ``
  原文、父类算出的 token id 与 mask、tokenizer 名、是否填满／截断；``text_reconstructed=true``：原文是按同一
  task_text 重新拼出的），``server_final_text`` 同该原文，``pp_generation`` 为自上次回包以来 System 2 每次 ``fire``
  一块的列表（reasoning、子目标原文与 token、kind、gate 分数、是否提交、是否回滚、上下文长度增量；键名对齐客户端
  语言账本，见 ``generation_blocks``；没有 fire 为 None）。System 2 的观察挂在本局上下文实例的 ``fire``／``_restore`` 上（调用原方法、原样
  返回）；第一次 Ponder 时父类才新建上下文，外壳在父类 ``_fire_s2`` 期间把模块里的上下文类临时换成「建好即挂观察」
  的工厂，返回后立即还原。外壳只观察真实结果：不多推理、不多抽随机数、不改动作。环境变量 ``SGEVAL_AUDIT=0`` 时
  不挂观察、不加审计键（``OBS_EQ`` 对照）。
- S2 输入（FIX-3，计划八.11 PonderPounce 行）：另挂在上下文实例的 ``_append`` 上只读镜像进上下文的 token id，每次
  ``fire`` 把「上次 fire 输入之后新增的段 + 本次观测段」用 S2 分词器解码成 ``input_text``（视觉段为 ``<image:k>``），
  附图描述放 ``input_images``；详见 ``_S2Watch`` 与 ``generation_blocks``。
- 外壳自用参数 ``--sgeval-metadata-out <path>``：启动时从 argv 摘掉（其余原样交给 vla_eval），写服务元数据
  ``server-metadata-<port>.json``（``policy_seed`` 即 ``--args.seed``、``argv``、``pid``、``port``）。

启动（参数与原服务完全相同，cwd 在第三方 PonderPounce 目录，脚本用绝对路径）::

    python /abs/path/scripts/eval-official/pp_server_wrap.py --args.seed 0 --args.checkpoint_path ... --port 8000

``python -m`` 会把 cwd 放在 ``sys.path`` 首位，而按路径运行脚本放的是脚本目录；为了与原命令的导入环境一致，
入口先把本目录移出 ``sys.path``、把 cwd 放到首位，再导入 ``ponderpounce``。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

#: 回包里新增的键（客户端 ``pp_client.TracedConnection.act`` 按此读取）
SUBGOAL_KEY = "subgoal"
#: 挂在 Episode／Chunk 实例上的私有属性名
_ATTR = "_sgeval_subgoal"


def _prepare_sys_path() -> None:
    """与 ``python -m ponderpounce.eval.robomme_server`` 的 ``sys.path`` 对齐：cwd 在首位，本目录不在路径里。"""
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p and str(Path(p).resolve()) != here]
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


if __name__ == "__main__":  # pragma: no cover - 只在作为脚本启动时调整
    _prepare_sys_path()

from ponderpounce.eval.robomme_server import PonderPounceRoboMMEServer  # noqa: E402

#: 服务回包审计键（接口冻结说明五节）；SGEVAL_AUDIT=0 时不加
AUDIT_KEY = "_sgeval_audit"
ENV_AUDIT = "SGEVAL_AUDIT"
#: 外壳自用参数（启动时从 argv 里摘掉，其余原样交给 vla_eval 的 run_server）
METADATA_FLAG = "--sgeval-metadata-out"
#: 挂在 Episode 实例上的私有属性：本次回包之前累积的 System 2 生成块（每次回包后清空）
_GEN_ATTR = "_sgeval_generations"


def audit_enabled() -> bool:
    """``SGEVAL_AUDIT`` 缺省开；为 ``0`` 时外壳不挂观察、不加审计键（``OBS_EQ`` 对照）。"""
    return os.environ.get(ENV_AUDIT, "1") != "0"


def latest_visible_subgoal(cognitions: Any, now: int, previous: str | None) -> str | None:
    """``cognitions`` 中 ``ready_at_ns <= now`` 且 ``subgoal`` 非空的最新一条的文本；没有则返回 ``previous``。"""
    for cog in reversed(list(cognitions)):
        if cog.ready_at_ns <= now and cog.subgoal:
            return str(cog.subgoal)
    return previous


def _tolist(x: Any) -> list | None:
    if x is None:
        return None
    try:
        return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else list(x))]
    except Exception:  # noqa: BLE001 观察失败不影响服务
        return None


def _pixel_sha256(img: Any) -> str | None:
    """送进 S2 的一帧（PIL 或数组）的像素哈希，算法与 ``trace_writer.image_sha256`` 相同（``dtype.str|shape|`` + 连续
    字节）；RGB 的 PIL 转回 ``uint8 (H, W, 3)`` 与客户端发出的原始帧逐字节相同，客户端据此核对帧号。失败返回 None。"""
    try:
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(img))
        h = hashlib.sha256()
        h.update(f"{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
        return h.hexdigest()
    except Exception:  # noqa: BLE001 观察失败不影响服务
        return None


class _S2Watch:
    """System 2 上下文（``SoftS2SessionContext``）的只读观察：包住实例的 ``fire`` 与 ``_restore``，原样调用、原样返回，
    记下本次真实生成结果（reasoning、子目标、token、gate 分数）、上下文长度增量与回滚次数。

    FIX-3（1006 计划八.11 PonderPounce 行「S2 完整上下文含回灌历史、增量图文片段、附图引用」）：上游
    ``ponderpounce/inference/append_context.py::SoftS2SessionContext`` 只保留 KV cache（``self.cache``）与累计的
    cognition mask／query id，**不保留上下文的 token id**；所有进上下文的 token 都经实例方法 ``_append(segment)``
    （``_Segment.input_ids``／``mm_token_type_ids`` 是 CPU 张量），任务前缀 + 演示图（``self._prefix``）在构造时的
    ``reset()`` 里就已追加。所以输入观察挂在实例的 ``_append`` 上：原方法照常调用、原样返回，成功后只读地把这段
    token id 记进一份镜像（按追加起点 ``_context_len`` 截断，``_restore`` 回滚／``reset`` 由此体现）；构造时已追加的
    前缀从 ``ctx._prefix`` 只读补进镜像。每次 ``fire`` 的输入 = 镜像里「上次 fire 输入之后」到「本次观测段」为止的
    全部段（第一次含任务前缀与演示图；之后含上次 fire 提交的生成 token 即回灌的上一子目标、cognition 占位 token、
    本次观测图；回滚掉的生成段不在其中），用 S2 自己的分词器（``ctx.compiler.tokenizer``，即
    ``session.s2_processor.tokenizer``）``decode(skip_special_tokens=False)`` 成文字，视觉 token 段
    （``<|vision_start|>`` + ``<|image_pad|>``×n + ``<|vision_end|>``）换成 ``<image:k>`` 占位，不解码像素。只读：
    不调模型、不抽随机数、不改上下文、不多做前向；``ctx`` 没有 ``_append``／``compiler.tokenizer`` 时（测试桩）不观察
    输入，生成块与之前逐字节相同。"""

    def __init__(self, ctx: Any, sink: list, cam_keys: Any = None):
        self.ctx, self.sink = ctx, sink
        self.restores = 0
        self.fire_index = 0
        self.cam_keys = tuple(cam_keys) if cam_keys else None
        self.input_errors = 0
        self._segs: list[dict] = []      # 上下文镜像：每段 {start, end, ids, mm, origin, images}
        self._consumed = 0               # 已作为某次 fire 输入报告过的上下文长度
        self._await_obs = False          # 本次 fire 的观测段还没追加
        self._fire_pils: Any = None
        self._restored = False
        self._pending: dict | None = None
        orig_fire, orig_restore = ctx.fire, getattr(ctx, "_restore", None)
        compiler = getattr(ctx, "compiler", None)
        self._tok = getattr(compiler, "tokenizer", None)
        self._vs = getattr(compiler, "vision_start_id", None)
        self._ve = getattr(compiler, "vision_end_id", None)
        orig_append = getattr(ctx, "_append", None)
        self.observe_input = callable(orig_append) and callable(getattr(self._tok, "decode", None))
        if self.observe_input:
            try:
                self._seed_prefix()
            except Exception:  # noqa: BLE001 观察失败不影响服务
                self.input_errors += 1

        def fire(*a, **k):
            before = getattr(ctx, "_context_len", None)
            prefix_restore = getattr(ctx, "_prefix_snapshot", None) is not None
            self.restores = 0
            self._await_obs, self._pending, self._restored = True, None, False
            self._fire_pils = a[0] if a else k.get("obs_pils")
            try:
                result = orig_fire(*a, **k)
            finally:
                self._await_obs, self._fire_pils = False, None
            pending, self._pending = self._pending, None
            after = getattr(ctx, "_context_len", None)
            rollbacks = max(0, self.restores - (1 if prefix_restore else 0))
            sg_tokens = getattr(result, "subgoal_tokens", None)
            self.sink.append({
                "fire_index": self.fire_index,
                "subgoal_text": getattr(result, "subgoal_text", None),
                "reasoning_text": getattr(result, "reasoning_text", None),
                "subgoal_tokens": _tolist(sg_tokens),
                "kind": getattr(result, "kind", None),
                "gate_score": None if getattr(result, "gate_score", None) is None else float(result.gate_score),
                "n_input_frames": getattr(result, "n_input_frames", None),
                "committed": sg_tokens is not None,
                "rolled_back": rollbacks > 0,
                "context_len_before": before,
                "context_len_after": after,
                "context_delta": (after - before) if isinstance(before, int) and isinstance(after, int) else None,
                **(pending or {}),
            })
            self.fire_index += 1
            return result

        ctx.fire = fire
        if callable(orig_restore):
            def restore(*a, **k):
                self.restores += 1
                return orig_restore(*a, **k)
            ctx._restore = restore
        if self.observe_input:
            def append(segment, *a, **k):
                start = getattr(ctx, "_context_len", None)
                out = orig_append(segment, *a, **k)
                try:
                    self._on_append(segment, start)
                except Exception:  # noqa: BLE001 观察失败不影响服务
                    self.input_errors += 1
                return out
            ctx._append = append

    # -- S2 输入观察（只读：只读 segment 的 CPU token id，不碰 cache、不调模型） --
    def _demo_images(self) -> list:
        demo = getattr(getattr(self.ctx, "session", None), "demo_images", None) or []
        return [{"source": "demo", "demo_pos": i, "pixel_sha256": _pixel_sha256(im)} for i, im in enumerate(demo)]

    def _obs_images(self) -> list:
        pils = list(self._fire_pils or [])
        keys = self.cam_keys or ()
        return [{"source": "obs", "cam_key": keys[j] if j < len(keys) else None, "cam_slot": j,
                 "pixel_sha256": _pixel_sha256(im)} for j, im in enumerate(pils)]

    def _seed_prefix(self) -> None:
        """构造时 ``reset()`` 已追加的任务前缀 + 演示图（``ctx._prefix``）补进镜像（只读）。"""
        prefix = getattr(self.ctx, "_prefix", None)
        clen = getattr(self.ctx, "_context_len", None)
        ids = _tolist(getattr(prefix, "input_ids", None))
        if not ids or not isinstance(clen, int) or clen < len(ids):
            return
        self._segs.append({"start": 0, "end": len(ids), "ids": ids,
                           "mm": _tolist(getattr(prefix, "mm_token_type_ids", None)), "origin": "prefix",
                           "images": self._demo_images()})

    def _on_append(self, segment: Any, start: Any) -> None:
        if not isinstance(start, int):
            return
        ids = _tolist(getattr(segment, "input_ids", None)) or []
        # 追加起点及之后的镜像段已被 _restore／reset 丢弃（段整体追加，不会跨起点）
        self._segs = [s for s in self._segs if s["start"] < start]
        if self._consumed > start:  # 上下文回到了已报告位置之前（current_obs_only 每次回到前缀、或 reset）
            self._consumed, self._restored = start, True
        if segment is getattr(self.ctx, "_prefix", None):
            origin, images = "prefix", self._demo_images()
        elif self._await_obs:
            origin, images = "obs", self._obs_images()
        else:
            cm = _tolist(getattr(segment, "cognition_mask", None)) or []
            origin, images = ("cog" if any(cm) else "generated"), []
        end = start + len(ids)
        if ids:
            self._segs.append({"start": start, "end": end, "ids": ids,
                               "mm": _tolist(getattr(segment, "mm_token_type_ids", None)), "origin": origin,
                               "images": images})
        if origin == "obs":
            self._await_obs = False
            segs = [s for s in self._segs if s["start"] >= self._consumed and s["end"] <= end]
            self._pending = self._decode_input(segs, base=self._consumed)
            self._consumed = end

    def _decode_input(self, segs: list, *, base: int) -> dict:
        """镜像段 → 文字（S2 分词器 decode，视觉段换 ``<image:k>``）+ 逐图描述（来源、相机、像素哈希）。"""
        tok = self._tok
        parts: list[str] = []
        buf: list[int] = []
        images: list[dict] = []

        def flush() -> None:
            if buf:
                parts.append(str(tok.decode(list(buf), skip_special_tokens=False)))
                buf.clear()

        for s in segs:
            ids = s["ids"]
            mm = s["mm"] if s["mm"] is not None and len(s["mm"]) == len(ids) else [0] * len(ids)
            pending_imgs = list(s["images"])
            i = 0
            while i < len(ids):
                if mm[i] == 1:  # 一张图的视觉 token 段（<|image_pad|>×n）
                    j = i
                    while j < len(ids) and mm[j] == 1:
                        j += 1
                    if buf and self._vs is not None and buf[-1] == self._vs:
                        buf.pop()
                    flush()
                    k = len(images)
                    parts.append(f"<image:{k}>")
                    d = dict(pending_imgs.pop(0)) if pending_imgs else {"source": s["origin"]}
                    d.update(index=k, n_tokens=j - i)
                    images.append(d)
                    if j < len(ids) and self._ve is not None and ids[j] == self._ve:
                        j += 1
                    i = j
                    continue
                buf.append(int(ids[i]))
                i += 1
        flush()
        return {"input_text": "".join(parts), "input_token_count": sum(len(s["ids"]) for s in segs),
                "input_images": images, "input_decoded_from_tokens": True, "input_base_len": base,
                "input_context_restored": self._restored,
                "input_segments": [{"origin": s["origin"], "tokens": len(s["ids"])} for s in segs]}


class SubgoalReportingServer(PonderPounceRoboMMEServer):
    """原类 + ACTION 回包附 ``subgoal``（与审计键）；动作、随机数、游标与触发节拍与原类逐项相同。"""

    def _fire_s2(self, ep, obs, now: int) -> None:
        if not audit_enabled():
            return super()._fire_s2(ep, obs, now)
        sink = getattr(ep, _GEN_ATTR, None)
        if sink is None:
            sink = []
            setattr(ep, _GEN_ATTR, sink)
        if ep.s2_context is not None:
            return super()._fire_s2(ep, obs, now)
        # 本局第一次 Ponder：父类在 _fire_s2 里新建上下文后立即 fire。临时把模块里的上下文类换成「建好即挂观察」的
        # 工厂（构造参数原样转交，返回原类实例），父类返回后立即还原；不多建、不多调。
        import ponderpounce.eval.robomme_server as rs

        orig_cls = rs.SoftS2SessionContext

        def factory(*a, **k):
            ctx = orig_cls(*a, **k)
            _S2Watch(ctx, sink, cam_keys=getattr(self, "_camera_keys", None))
            return ctx

        rs.SoftS2SessionContext = factory
        try:
            return super()._fire_s2(ep, obs, now)
        finally:
            rs.SoftS2SessionContext = orig_cls

    def _fire_s1(self, ep, obs, now: int) -> None:
        current = latest_visible_subgoal(ep.cognitions, now, getattr(ep, _ATTR, None))
        setattr(ep, _ATTR, current)
        super()._fire_s1(ep, obs, now)
        if ep.chunk is not None:
            setattr(ep.chunk, _ATTR, current)

    def _dispense(self, ep, obs):
        action = super()._dispense(ep, obs)
        subgoal = getattr(ep.chunk, _ATTR, None) if ep.chunk is not None else None
        out = dict(action)
        out[SUBGOAL_KEY] = subgoal
        if audit_enabled():
            out[AUDIT_KEY] = self._audit_block(ep)
        return out

    def _audit_block(self, ep) -> dict:
        """S1 prompt（本局 session 上父类真实算出的 token id／mask）+ 自上次回包以来的 S2 完整生成块。"""
        s = ep.session
        ids, mask = getattr(s, "s1_prompt_ids", None), getattr(s, "s1_prompt_mask", None)
        text = None
        if ids is not None:
            try:
                from ponderpounce.pi05.prompt import build_pi0_prompt

                text = build_pi0_prompt(getattr(s, "task_text", None) or "", None)
            except Exception:  # noqa: BLE001 观察失败不影响服务
                text = None
        mask_l = _tolist(mask)
        tok = getattr(self, "_s1_tokenizer", None)
        channels = [] if ids is None else [{
            "channel": "task", "text": text, "token_ids": _tolist(ids), "mask": mask_l,
            "tokenizer": getattr(tok, "name_or_path", None) or (type(tok).__name__ if tok is not None else None),
            # 右侧补齐到 max_token_len 且 truncation=True：mask 全为 1 表示填满（可能被截断）
            "truncated": None if mask_l is None else bool(mask_l) and all(mask_l),
            # text 是用 build_pi0_prompt 按同一 task_text 重新拼出的（token_ids／mask 才是真实截获）
            "text_reconstructed": True}]
        gens = getattr(ep, _GEN_ATTR, None)
        fires = list(gens) if gens else []
        if gens:
            gens.clear()
        return {"channels": channels, "server_final_text": text,
                "pp_generation": generation_blocks(fires, task_text=getattr(s, "task_text", None),
                                                   active_subgoal=getattr(ep, "active_subgoal", None),
                                                   n_s2_fires=getattr(ep, "n_s2_fires", None),
                                                   n_s1_fires=getattr(ep, "n_s1_fires", None))}


def generation_blocks(fires: list, *, task_text: Any = None, active_subgoal: Any = None, n_s2_fires: Any = None,
                      n_s1_fires: Any = None) -> list | None:
    """``pp_generation``：自上次回包以来每次 System 2 ``fire`` 一块（列表；没有 fire 为 ``None``，客户端不开调用）。

    每块保留 ``_S2Watch`` 记下的原字段，另按客户端 ``pp_client.TracedConnection._log_generation`` 读的键补齐：
    ``subgoal_raw``（= ``subgoal_text``，``at [x, y]`` 原文）、``reasoning``（= ``reasoning_text``）、``kind``／``committed``
    （原字段）、``params``（fire 序号、gate 分数、输入帧数、上下文长度增量、是否回滚）。``text`` 不给（上游只保留拆开后的
    reasoning 与子目标，不保留整块解码原文；客户端缺 ``text`` 时取 ``subgoal_raw``）；``context``／``prompt``／``images``
    不给。S2 输入（FIX-3）：上下文实例可观察时每块另带 ``input_text``（本次 fire 前新增进上下文的 token 用 S2 分词器
    解码，视觉段为 ``<image:k>``）、``input_token_count``、``input_images``（第 k 个占位对应的图：``source=demo`` 带
    ``demo_pos``，``source=obs`` 带 ``cam_key``／``cam_slot``，均带 ``pixel_sha256``）、``input_decoded_from_tokens=True``、
    ``input_base_len``（这段追加在多长的上下文之后）、``input_context_restored``（上下文先回退到已报告位置之前，如
    ``current_obs_only`` 每次回到前缀）、``input_segments``（各段来源 prefix／generated／cog／obs 与 token 数）；不可观察
    时这些键不出现。``s2_task_text`` 与回包时刻的 ``active_subgoal``／``n_s2_fires``／``n_s1_fires`` 随每块附上。"""
    if not fires:
        return None
    out = []
    for f in fires:
        b = dict(f)
        b.update(subgoal_raw=f.get("subgoal_text"), reasoning=f.get("reasoning_text"),
                 params={k: f.get(k) for k in ("fire_index", "gate_score", "n_input_frames", "context_len_before",
                                               "context_len_after", "context_delta", "rolled_back")},
                 s2_task_text=task_text, active_subgoal=active_subgoal, n_s2_fires=n_s2_fires, n_s1_fires=n_s1_fires)
        out.append(b)
    return out


def split_wrapper_args(argv: list[str]) -> tuple[str | None, list[str]]:
    """从 argv 摘掉外壳自用的 ``--sgeval-metadata-out[=]<path>``，返回 (路径, 其余参数)。"""
    meta, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith(METADATA_FLAG + "="):
            meta = a.split("=", 1)[1]
        elif a == METADATA_FLAG and i + 1 < len(argv):
            meta = argv[i + 1]
            i += 1
        else:
            rest.append(a)
        i += 1
    return meta, rest


def _flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def write_metadata(path: str, argv_full: list[str], argv_rest: list[str]) -> dict:
    """服务元数据 ``server-metadata-<port>.json``：policy_seed（即 --args.seed）、argv、pid、port、audit。"""
    seed = _flag_value(argv_rest, "--args.seed")
    port = _flag_value(argv_rest, "--port")
    doc = {"policy": "pp", "policy_seed": int(seed) if seed is not None and seed.lstrip("-").isdigit() else None,
           "argv": list(argv_full), "pid": os.getpid(), "port": int(port) if port and port.isdigit() else port,
           "audit": audit_enabled(), "wrapper": "pp_server_wrap.py"}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return doc


def main() -> None:
    from vla_eval.model_servers.serve import run_server

    full = list(sys.argv)
    meta, rest = split_wrapper_args(full[1:])
    if meta:
        write_metadata(meta, full, rest)
    sys.argv = [full[0], *rest]
    run_server(SubgoalReportingServer)


if __name__ == "__main__":  # pragma: no cover
    main()
