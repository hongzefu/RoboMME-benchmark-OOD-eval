"""模型注册表：名字字符串 → (模块路径, Policy 子类名)，惰性导入（拆分方案 §三「五个模型各自怎么落」）。

E1 一次登记全部六个模型；E2／E3 只在各自模块末尾加对应的 Policy 子类，不再改本文件。
"""
from __future__ import annotations

import importlib

#: 注册名 → (模块路径, 类名)；注册名即 ``scripts/evaluate.py --model`` 的取值与产物树里的 ``<模型>`` 目录名
REGISTRY: dict[str, tuple[str, str]] = {
    "dummy": ("robomme_hard_eval.models.dummy", "DummyPolicy"),
    "perceptual-framesamp-modul": ("robomme_hard_eval.models.framesamp_modul", "FrameSampModulPolicy"),
    "groundsg": ("robomme_hard_eval.models.groundsg", "GroundSGPolicy"),
    "smvla": ("robomme_hard_eval.models.smvla", "SmvlaPolicy"),
    "pp": ("robomme_hard_eval.models.pp", "PPPolicy"),
    "astra": ("robomme_hard_eval.models.astra", "AstraPolicy"),
}
MODELS = tuple(REGISTRY)


def resolve(name: str) -> type:
    """按注册名取 Policy 子类（此时才导入模块）。未登记的名字抛 ``KeyError``，模块里没有该类抛 ``ImportError``。"""
    if name not in REGISTRY:
        raise KeyError(f"未登记的模型 {name!r}，可选：{', '.join(MODELS)}")
    mod_name, cls_name = REGISTRY[name]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name, None)
    if cls is None:
        raise ImportError(f"{mod_name} 里没有 {cls_name}（模型 {name!r} 的 Policy 子类尚未实现）")
    return cls
