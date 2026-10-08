"""对拍块的钉值文件（红线 R8 允许的「钉值文件」：业务常量字面值只出现在这里与 tests/contract/test_constants.py）。

每项注明用户口径出处（1003-code-test-maintenance-todo.md 第二部分「对拍细则」）。改这些值等于改闸门口径，
须先改计划与用户确认；``test_parity_pins.py`` 断言 ``dev-scripts/parity/noise_gate.py`` 里的同名常量与这里逐个相等。
"""
from __future__ import annotations

#: 确认为噪声的翻转上限：V9 ≤ 2 局、xhard0 ≤ 1 局（对拍细则 3.3 第 2 条，用户 2026-10-03 选定）
NOISE_MAX = {"v9": 2, "xhard0": 1}
#: 出问题的局超过 10 个就不进第二次跑、直接交用户（对拍细则 3.4「预算」）
FLIP_RERUN_MAX = 10
#: 第二次跑每席 4 worker，出问题的局不足 4 个用陪跑局凑满（对拍细则 3.4「陪跑局凑满并发」）
RERUN_MIN = 4
#: 跑法前提：GL A40、--workers 4（对拍细则 3.3 第 4 条）
PRECOND_WORKERS = 4
PRECOND_GPU = "A40"

#: canonical sha 的固定向量（对拍细则 3.2／G 块「顶层 sha256 为剔除自身后 canonical JSON 的 sha256」）：
#: sort_keys、紧凑分隔 (",", ":")、不转义非 ASCII；含中文、非 ASCII 拉丁字母、浮点、null、布尔与嵌套。
#: digest 由独立脚本预先算好写死，任何序列化口径漂移（ensure_ascii、分隔符、是否剔除 sha256 键）都会变。
CANONICAL_VECTOR_OBJ = {
    "schema": "noise-ref/1",
    "sets": {"v9": {"episodes": [{"id": "T|xhard1|1", "shas": ["ab"], "class": "stable"}]}},
    "note": "中文 ünï",
    "n": [1, 2.5, None, True],
    "sha256": "不参与",
}
CANONICAL_VECTOR_TEXT = ('{"n":[1,2.5,null,true],"note":"中文 ünï","schema":"noise-ref/1",'
                         '"sets":{"v9":{"episodes":[{"class":"stable","id":"T|xhard1|1","shas":["ab"]}]}}}')
CANONICAL_VECTOR_SHA256 = "1f479ca3f96c2067d564f363f2d8f49e65e0e2d1c6644698256695e4d7ac056e"
