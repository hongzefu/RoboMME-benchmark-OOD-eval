"""植入配方补充表：mutants.json 只给了文字描述的条目，在这里补成可机检的具体做法。

键为 ``<块>:<编号>``，块 = mutants.json 相对 ``tests/`` 的目录（如 ``pipeline/eval``）。三种形态：

- ``{"kind": "text", "patch": [{"file", "old", "new"}, ...]}``：A 类，在隔离副本里逐字替换，每个 old 必须恰好命中 1 次；
- ``{"kind": "transform", "func": <可调用>, "files": [...]}``：A 类，在隔离副本里按函数改写数据文件（拆仓后本表已无此类）；
  ``files`` 列出会被改写的相对路径，执行器据此备份与还原；
- 进程内植入（B 类）不在本表，见 ``plugins/mut_inproc.py`` 的 ``SUPPORTED`` 与 ``tests/unit/hard/mutants_plugin.py``。

每条配方的语义照抄对应 mutants.json 的 ``method``，只把文字落成精确的替换点；不改各块的 expect_fail。
"""
from __future__ import annotations



def _t(*patches: tuple[str, str, str]) -> dict:
    return {"kind": "text", "patch": [{"file": f, "old": o, "new": n} for f, o, n in patches]}


# 拆仓后（评估仓）：挑战接口块（tests/pipeline/challenge）与契约块（tests/contract 的规格 jsonl／builder 植入）随 benchmark
# 仓走，配方已删；评估块（tests/pipeline/eval）的 mutants.json 每条自带内联 patch（指向 dev-scripts／src/robomme_hard_eval
# 新位置），不再依赖本表，旧评估目录（eval-official）的配方一并删去。

# ─────────────────────────────── 生成块（tests/pipeline/gen）与站点块（tests/pipeline/site） ───────────────────────────────
# 拆仓后各文件的新位置（见 tests/_support/loaders.py::LEGACY_PATHS）；_freeze 的两条（T7-F1／F2）已不在 gen 的
# mutants.json 里（gen 只留 _freeze_min 的测试），对应配方删去
_R = "dev-scripts/parity/_rollout.py"
_SBD = "dev-scripts/site/site_build.py"
_EX = "dev-scripts/gl/export_eval_identities.py"
_SV = "dev-scripts/site/site_server.py"
_CT = "dev-scripts/site/site_catalog.py"
GEN_SITE = {
    "pipeline/gen:M17a": _t((_R, '            "expected": expected,\n',
                             '            "expected": sum(hard_specs.delivered(r) for r in mine),\n')),
    "pipeline/gen:M17b": _t((_R, '"counts": totals, "cells": cell_out',
                             '"counts": {k: v for k, v in totals.items() if v}, "cells": cell_out')),
    "pipeline/gen:M18a": _t((_R, 'if entries and not any(r["tried"] for r in rows):', "if False:")),
    "pipeline/gen:M18b": _t((_R, '    if resume:\n        command.append("--resume")\n', "")),
    "pipeline/gen:T7-G1": _t((_R, 'key=lambda r: r["candidate"])\n            if row["task"] in same_way_tasks:',
                              'key=lambda r: -r["candidate"])\n            if row["task"] in same_way_tasks:')),
    "pipeline/gen:T7-G2": _t((_R, "infra_retries.get(key, 0) < max_infra_retries",
                              "infra_retries.get(key, 0) <= max_infra_retries")),
    "pipeline/gen:T7-G3": _t((_R, "if int(steps) > int(exec_cap):", "if int(steps) >= int(exec_cap):")),
    "pipeline/gen:T7-G4": _t((_R, 'if row["task"] in same_way_tasks:', "if False:")),
    "pipeline/gen:T7-S1": _t((_SBD, 'prev.get("sha256s") == [sha256_file(p) for p in paths]', "True")),
    "pipeline/gen:T7-S2": _t((_SBD, 'if old.get("fingerprint") != runner.fingerprint:', "if False:")),
    "pipeline/gen:T7-E1": _t((_EX, 'and facts.get("delivery_mismatch", 0) == 0)', ")")),
    "pipeline/site:M20a": _t((_SV, "os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)",
                              "os.O_RDONLY | os.O_DIRECTORY, dir_fd=directory)")),
    "pipeline/site:M20b": _t((_SV, "path = Path(filename).resolve(strict=True)", "path = Path(filename).absolute()")),
    "pipeline/site:M20c": _t((_SV, "    return 206, start, end - start + 1", "    return 200, 0, size")),
    "pipeline/site:T7-C1": _t((_CT, "if not table1_ok(task, dim, tier, value):", "if False:")),
}

RECIPES: dict[str, dict] = {**GEN_SITE}
