#!/usr/bin/env python3
"""Extract definitions by name from the official source.

The official MME-VLA client ``third_party/mme-vla/examples/robomme/eval.py`` unconditionally imports gemini
(``google.generativeai``) and memer modules via ``subgoal_predictor.py``, none of which are installed in the
evaluation environment. So neither the GroundSG new side (``groundsg_client.py``) nor the original side
(``official_hard_runner.py``) **imports the official files as whole modules**; instead ``extract_defs`` takes the
**verbatim source** of the requested top-level functions, classes and single-target assignments from the source
file and executes it; the rest of the module (import lines, module-level side effects) is not executed, and
dependencies are injected by the caller via ``extra``. The returned namespace is the globals of these definitions, so
names added to it later (e.g. a lazily imported ``PtEngine``) also take effect for the extracted functions.

The official module headers contain three environment settings (at the top of ``eval.py`` and
``subgoal_prediction/qwenvl/api.py``) that extraction does not carry over; ``apply_official_env`` applies them. The
QwenVL offline runtime constraints are applied by ``apply_qwen_runtime_env``.

``load_groundsg`` is the assembly shared by both sides: per variant it only extracts the classes it needs (Oracle
does not read qwenvl/api.py and does not import swift; QwenVL only takes ``QwenVLSubgoalPredictor`` and
``Qwen3VLModel``, not Gemini / MemER; MemER only takes ``MemERSubgoalPredictor`` and
``qwenvl/api_memer.py::Qwen3VLModelMemER``, not Gemini / QwenVL).

MemER compatibility layer: after extracting the verbatim ``Qwen3VLModelMemER``, patch it **at the AST level** (only
the extracted copy is touched; ``third_party`` and the gitlink are not) -- the official ``merge_key_frame_paths`` /
``_get_current_execution_frame_paths`` / ``update_history_subgoals`` / ``call`` are renamed unchanged to
``_official_<name>`` (``official_method_name``) and kept in the class, and the same-named methods from
``MEMER_COMPAT_SOURCE`` are appended:

1. ``merge_key_frame_paths``: return immediately when memory is empty; otherwise call the official original function
   (byte-identical);
2. ``call``: every valid parse stores the converted subgoal handed to the action model in ``self.subgoals``; on a
   parse failure re-ask at most twice (three times in total); the second and third user prompts get
   ``MEMER_RETRY_NOTE`` appended and use ``RequestConfig(max_tokens=128, temperature=0.7)``, with request and reply
   appended to ``ep*_MemER_log.jsonl`` with ``retry=<n>``; if all three fail and there is a previous valid subgoal it
   is reused (``fallback=last_valid``), otherwise ``MemERResponseError`` is raised (the client records
   ``error_kind=model_response_error`` and does not rerun);
3. ``_get_current_execution_frame_paths``: starting from the last frame take every other frame, stopping before
   frame 1 (with fewer than 15 frames take as many as there are); with 1 or >= 15 frames call the official original
   function;
4. ``update_history_subgoals``: atomic validation -- first check on a temporary copy the JSON structure, non-empty
   string subtask, integer key-frame positions (rejecting bool, 0, negatives and out-of-range values), coordinate
   conversion and candidate memory merge, and commit once only if everything passes; a bad reply changes neither
   key frames, history nor execution frames.

The sha256 of the patch source text is the implementation fingerprint ``MEMER_COMPAT_SHA256`` (written into verdict
lines, result rows and media provenance). The question template, system prompt, ``prepare_infer_request``, key-frame
selection and (when non-empty) merge rules are unchanged word for word.

Location of the official source: environment variable ``SGEVAL_THIRD_PARTY`` (pointing at some checkout's
``third_party``; e.g. tests use it to reference a populated checkout read-only when the submodule directories are
empty), otherwise this repo's ``third_party``.
"""
from __future__ import annotations

import ast
import collections
import copy
import dataclasses
import hashlib
import json
import os
import pprint
import random
import re
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

#: evaluation repo root (this file lives in src/robomme_ood_eval/models/)
REPO = Path(__file__).resolve().parents[3]
#: the three variants
VARIANT_ORACLE = "ground-sg-oracle"
VARIANT_QWENVL = "ground-sg-qwenvl"
VARIANT_MEMER = "ground-sg-memer"
VARIANTS = (VARIANT_ORACLE, VARIANT_QWENVL, VARIANT_MEMER)
_SEQ = 0
#: the three environment settings at the top of the official modules (verbatim from the top of eval.py and qwenvl/api.py)
OFFICIAL_ENV = {"IMAGE_MAX_TOKEN_NUM": "256", "VIDEO_MAX_TOKEN_NUM": "64", "FPS_MAX_FRAMES": "10"}
#: QwenVL runtime constraints (ms-swift defaults to ModelScope; here everything runs offline from the HF cache)
QWEN_RUNTIME_ENV = {"USE_HF": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# -- official names and legacy-name compatibility (the only alias table in the repo) --
# Output always uses official names and the CLI only accepts official names; canonical_* mappings are only used when
# reading historical per-episode rows, budget ledgers, trace headers and archived records.
#: policy labels (official names): FrameSamp+Modulation and GroundSG
POLICY_FRAMESAMP_MODUL = "perceptual-framesamp-modul"
POLICY_GROUNDSG = "groundsg"
#: dataset interfaces (official names): OOD and hard-verify (the 12 official hard episodes)
DATASET_OOD = "ood"
DATASET_HARD_VERIFY = "hard-verify"
# >>> LEGACY_NAMES (the OFFICIAL_NAMES leftover check exempts only this block)
#: legacy policy label -> official label (policy, first route segment and directory names in archived records)
LEGACY_POLICY_ALIASES = {"mme": POLICY_FRAMESAMP_MODUL, "mmevla": POLICY_FRAMESAMP_MODUL, "mmesg": POLICY_GROUNDSG}
#: legacy dataset name -> official dataset name
LEGACY_DATASET_ALIASES = {"test-hard": DATASET_OOD, "test-hard0": DATASET_HARD_VERIFY}
#: client module names and config keys of a pre-rename checkout (e.g. the base side of the replay gate) -> official
#: names (used by client_replay_eq.py when driving an old checkout)
LEGACY_MODULE_ALIASES = {"mme_client": "framesamp_modul_client", "mmesg_client": "groundsg_client"}
LEGACY_CONFIG_KEY_ALIASES = {"mme_variant": "groundsg_variant"}
# <<< LEGACY_NAMES


def canonical_policy(name: Any) -> Any:
    """Policy label: map legacy names to official names; labels with a variant (legacy GroundSG prefix +
    ``-<variant>``) get their prefix replaced too; everything else is returned unchanged."""
    if not isinstance(name, str):
        return name
    if name in LEGACY_POLICY_ALIASES:
        return LEGACY_POLICY_ALIASES[name]
    for old, new in LEGACY_POLICY_ALIASES.items():
        if new == POLICY_GROUNDSG and name.startswith(old + "-"):
            return new + name[len(old):]
    return name


def canonical_dataset(name: Any) -> Any:
    """Dataset name: map legacy names to official names; everything else is returned unchanged."""
    return LEGACY_DATASET_ALIASES.get(name, name) if isinstance(name, str) else name


def canonical_route(route: Any) -> Any:
    """Strings segmented by ``/`` such as routes / media keys: map each segment by policy label and dataset name
    (e.g. a legacy GroundSG route ``<legacy>/<variant>/orig`` -> ``groundsg/<variant>/orig``)."""
    if not isinstance(route, str) or not route:
        return route
    return "/".join(canonical_dataset(canonical_policy(seg)) for seg in route.split("/"))


#: fields mapped by canonical_row
_POLICY_FIELDS = ("policy", "label", "policy_label")
_DATASET_FIELDS = ("dataset",)
_ROUTE_FIELDS = ("route",)


def canonical_row(row: Any) -> Any:
    """Per-episode rows / ledger rows / trace headers: map ``policy`` / ``label``, ``dataset`` and ``route`` by
    legacy name, and ``dataset`` inside ``identity`` as well; returns a new dict without modifying the input.
    Non-dicts are returned unchanged."""
    if not isinstance(row, dict):
        return row
    out = dict(row)
    for k in _POLICY_FIELDS:
        if k in out:
            out[k] = canonical_policy(out[k])
    for k in _DATASET_FIELDS:
        if k in out:
            out[k] = canonical_dataset(out[k])
    for k in _ROUTE_FIELDS:
        if k in out:
            out[k] = canonical_route(out[k])
    if isinstance(out.get("identity"), dict) and "dataset" in out["identity"]:
        out["identity"] = {**out["identity"], "dataset": canonical_dataset(out["identity"]["dataset"])}
    return out


def legacy_labels(label: str) -> list[str]:
    """Legacy labels corresponding to an official label (for reading historical directories; excluding the official
    label itself), e.g. ``groundsg-<variant>`` -> the legacy GroundSG-prefixed version."""
    out = []
    for old, new in LEGACY_POLICY_ALIASES.items():
        if label == new:
            out.append(old)
        elif new == POLICY_GROUNDSG and label.startswith(new + "-"):
            out.append(old + label[len(new):])
    return out


def third_party_root() -> Path:
    """Official third-party source root: ``SGEVAL_THIRD_PARTY`` first, otherwise this repo's ``third_party``."""
    env = os.environ.get("SGEVAL_THIRD_PARTY")
    return Path(env).resolve() if env else REPO / "third_party"


def official_robomme_dir() -> Path:
    """The official ``examples/robomme`` directory; raises if ``eval.py`` is missing (never skips silently)."""
    d = third_party_root() / "mme-vla" / "examples" / "robomme"
    if not (d / "eval.py").is_file():
        raise FileNotFoundError(f"official MME-VLA source not found: {d}/eval.py (submodule not initialized? you can "
                                f"set SGEVAL_THIRD_PARTY)")
    return d


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_defs(path: str | Path, names: list[str], extra: dict | None = None, *,
                 transform: dict | None = None) -> dict:
    """Extract the verbatim source of the given top-level functions / classes / single-target assignments from a
    source file with ast, execute it, and return the namespace.

    * only the top-level ``def`` / ``async def`` / ``class`` (with decorators) and single-target assignments
      ``X = ...`` listed in ``names`` are taken; a name defined several times is taken every time in source order
      (the last one wins, as when executing the whole module);
    * a name that cannot be found raises ``KeyError``;
    * the rest of the module (import lines, module-level side effects) is not executed; dependencies are injected via
      ``extra`` (basic names such as ``np`` and ``Any`` are preset);
    * the namespace records ``__source_path__`` and ``__source_sha256__`` (sha256 of the whole file's bytes);
    * ``transform``: ``{name: f(ast node) -> ast node}``, an AST-level rewrite of that extracted definition before
      execution (only used for the MemER compatibility layer; ``__source_sha256__`` is still the sha256 of the
      official original file, and the rewrite has its own fingerprint).
    """
    path = Path(path)
    raw = path.read_bytes()
    src = raw.decode("utf-8")
    tree = ast.parse(src, filename=str(path))
    body = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                body.append(node)
                found.add(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            if node.target.id in names:
                body.append(node)
                found.add(node.target.id)
    missing = sorted(set(names) - found)
    if missing:
        raise KeyError(f"{path} does not contain {missing}")
    if transform:
        for i, node in enumerate(body):
            name = node.name if hasattr(node, "name") else None
            if name in transform:
                body[i] = transform[name](node)
    # the namespace lives on a separate module object registered in sys.modules (dataclass etc. look the module up
    # via __module__); the module name has an _official_ prefix and a sequence number so it never clashes with real
    # modules (eval, utils, ...)
    global _SEQ
    _SEQ += 1
    mod = types.ModuleType(f"_official_{path.stem}_{_SEQ}")
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    ns: dict[str, Any] = mod.__dict__
    ns.update({
        "__builtins__": __builtins__,
        "np": np, "Any": Any, "Optional": Optional, "Tuple": Tuple, "List": List, "Path": Path,
        "os": os, "re": re, "json": json, "time": time, "shutil": shutil, "collections": collections,
        "dataclasses": dataclasses, "pprint": pprint,
    })
    ns.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), ns)  # noqa: S102 only executes the verbatim official definitions
    ns["__source_path__"] = str(path)
    ns["__source_sha256__"] = hashlib.sha256(raw).hexdigest()
    return ns


def apply_official_env() -> dict:
    """The three environment settings at the top of the official eval.py / qwenvl/api.py (overwritten verbatim).
    Returns the keys and values written."""
    for k, v in OFFICIAL_ENV.items():
        os.environ[k] = v
    return dict(OFFICIAL_ENV)


def apply_qwen_runtime_env() -> dict:
    """QwenVL runtime constraints: ``USE_HF=1``, ``HF_HUB_OFFLINE=1``, ``TRANSFORMERS_OFFLINE=1`` (call before
    importing swift)."""
    for k, v in QWEN_RUNTIME_ENV.items():
        os.environ[k] = v
    return dict(QWEN_RUNTIME_ENV)


def import_swift_names() -> dict:
    """The three real ``swift.llm`` names (only called when building the predictor for the QwenVL / MemER variants;
    sets the runtime constraints before importing)."""
    apply_qwen_runtime_env()
    apply_official_env()
    from swift.llm import InferRequest, PtEngine, RequestConfig

    return {"PtEngine": PtEngine, "InferRequest": InferRequest, "RequestConfig": RequestConfig}


def seed_everything(seed: int) -> dict:
    """Model seed: set ``random``, ``numpy`` and ``torch`` (when importable) together.

    Called before building the QwenVL / MemER predictor (``build_predictor``); only calls ``torch.manual_seed`` (which
    registers lazily for CUDA and does not initialize the GPU here). Returns the random sources actually seeded, for
    results and tests to check."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"policy_seed={seed!r} must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    out = {"seed": int(seed), "random": True, "numpy": True, "torch": False}
    try:
        import torch
    except ImportError:
        return out
    torch.manual_seed(seed)
    out["torch"] = True
    return out


def check_policy_seed(value: Any) -> int:
    """Check a ``policy_seed`` value: a non-negative integer (rejecting bool, None and negatives); decimal integers
    in string form are accepted too."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r} (required, non-negative integer)")
    try:
        n = int(str(value).strip()) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r} (required, non-negative integer)") from None
    if isinstance(value, float) and value != n:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r} (required, non-negative integer)")
    if n < 0:
        raise ValueError(f"RUN_BLOCKED reason=policy_seed value={value!r} (required, non-negative integer)")
    return n


#: names to extract from each official file
UTILS_NAMES = ["TASK_WITH_VIDEO_DEMO", "TASK_NAME_LIST", "SUBGOAL_TYPES", "pack_buffer", "check_args", "EpisodeState",
               "RolloutRecorder"]
ENV_RUNNER_NAMES = ["pack_state", "EnvRunner"]
EVAL_NAMES = ["Args", "EpisodeEvaluator"]
PREDICTOR_NAMES = {
    VARIANT_ORACLE: ["SubgoalPredictorBase", "OracleSubgoalPredictor", "build_subgoal_predictor"],
    VARIANT_QWENVL: ["SubgoalPredictorBase", "QwenVLSubgoalPredictor", "build_subgoal_predictor"],
    VARIANT_MEMER: ["SubgoalPredictorBase", "MemERSubgoalPredictor", "build_subgoal_predictor"],
}
#: subgoal model source file (relative to examples/robomme) and class name per variant; none for Oracle
SUBGOAL_MODEL_SOURCES = {
    VARIANT_QWENVL: ("subgoal_prediction/qwenvl/api.py", "Qwen3VLModel"),
    VARIANT_MEMER: ("subgoal_prediction/qwenvl/api_memer.py", "Qwen3VLModelMemER"),
}


# -- MemER compatibility layer (changes 1-4) -------------------------------------------


class MemERResponseError(RuntimeError):
    """All three MemER replies were invalid and the episode has no earlier valid subgoal (case D): the client
    records ``status=error, terminal_reason=error, error_kind=model_response_error``, never fabricates a subgoal and
    never reruns."""

    error_kind = "model_response_error"


#: reminder appended to the end of the user prompt on the second and third asks
MEMER_RETRY_NOTE = "Your previous reply was not valid JSON. Reply with the JSON object only."
#: total number of asks (first ask + two re-asks) and the sampling temperature of re-asks
MEMER_MAX_TRIES = 3
MEMER_RETRY_TEMPERATURE = 0.7
#: official methods changed by the compatibility layer (the verbatim originals are kept in the class renamed to
#: ``_official_<name>``)
MEMER_PATCHED_METHODS = ("merge_key_frame_paths", "_get_current_execution_frame_paths", "update_history_subgoals",
                         "call")

#: patch source text: appended to the body of the extracted ``Qwen3VLModelMemER`` class; the sha256 of its UTF-8 bytes
#: is the implementation fingerprint ``MEMER_COMPAT_SHA256``.
#: Dependency names (``json``, ``re``, ``copy``, ``RequestConfig``, ``InferRequest``, ``MemERResponseError``,
#: ``MEMER_*``) are injected into the extraction namespace by ``load_groundsg``. Changing this text changes the
#: fingerprint, which must be noted wherever results are compared.
MEMER_COMPAT_SOURCE = '''\
def merge_key_frame_paths(self, dist: int = 8):
    # compat 1: return immediately when the key-frame memory is empty (the official cur = [nums[0]] would index out of range); otherwise call the official original function, byte-identical
    if not self.key_frame_paths:
        return
    return self._official_merge_key_frame_paths(dist)


def _get_current_execution_frame_paths(self) -> list:
    # compat 3: starting from the last frame take every other frame, at most 8, stopping before frame 1; with 1 or >=15 frames call the official original function (verbatim)
    n = len(self.execution_frame_paths)
    if n == 1 or n >= 15:
        return self._official_get_current_execution_frame_paths()
    paths = []
    idx = n - 1
    while idx >= 0 and len(paths) < 8:
        paths.insert(0, self.execution_frame_paths[idx])
        idx -= 2
    return paths


def _memer_validate(self, subgoal: str):
    # compat 4: atomic validation -- everything is done on temporary objects without changing any state of self; any invalid item raises
    response = json.loads(subgoal)
    if not isinstance(response, dict):
        raise ValueError("reply is not a JSON object")
    current_subtask = response["current_subtask"]
    keyframe_positions = response["keyframe_positions"]
    if not isinstance(current_subtask, str) or not current_subtask.strip():
        raise ValueError(f"current_subtask must be a non-empty string: {current_subtask!r}")
    if not isinstance(keyframe_positions, list):
        raise ValueError(f"keyframe_positions must be a list: {keyframe_positions!r}")
    n_frames = len(self.current_execution_frame_paths)
    for key_id in keyframe_positions:
        if type(key_id) is not int or key_id < 1 or key_id > n_frames:
            raise ValueError(f"keyframe position {key_id!r} out of range 1..{n_frames}")
    vla_subgoal = self._parse_box_patterns(current_subtask, replacement="scaled_coords", return_bbox=False)
    if not isinstance(vla_subgoal, str) or not vla_subgoal.strip():
        raise ValueError(f"converted subgoal is empty: {vla_subgoal!r}")
    candidate = dict(self.key_frame_paths)
    for key_id in keyframe_positions:
        path_str = self.current_execution_frame_paths[key_id - 1]
        int_idx = int(re.search(r"step_(\\d+)_image.png", path_str).group(1))
        candidate[int_idx] = path_str
    saved = self.key_frame_paths
    try:
        self.key_frame_paths = candidate
        self.merge_key_frame_paths()
        merged = self.key_frame_paths
    finally:
        self.key_frame_paths = saved
    return current_subtask, vla_subgoal, merged, list(keyframe_positions)


def update_history_subgoals(self, subgoal: str):
    # compat 4: commit the key-frame memory once, only after all checks pass (return value same as official: the raw current_subtask)
    current_subtask, _vla, merged, _pos = self._memer_validate(subgoal)
    self.key_frame_paths = merged
    return current_subtask


def _memer_request_fields(self, infer_request):
    # copy of the first ask's request fields (taken before sending; re-asks only change the end of the user prompt on top of it)
    fields = {"messages": copy.deepcopy(list(infer_request.messages)), "images": list(infer_request.images)}
    videos = getattr(infer_request, "videos", None)
    if videos:
        fields["videos"] = list(videos)
    return fields


def _memer_log(self, row):
    with open(self.save_json_path, "a") as f:
        json.dump(row, f)
        f.write("\\n")


def call(self) -> str:
    # compat 2: valid subgoals are stored in self.subgoals; a bad reply is re-asked at most twice (reminder appended +
    # temperature=0.7); if all three are bad reuse the previous valid subgoal, otherwise raise MemERResponseError.
    # The request and reply log lines of the first ask are verbatim official; re-ask lines carry retry
    infer_request = self.prepare_infer_request()
    base_fields = self._memer_request_fields(infer_request)
    self._memer_fallback = None
    self._memer_errors = []
    self._memer_last_positions = None
    for retry in range(MEMER_MAX_TRIES):
        self._memer_retry = retry
        if retry == 0:
            request = infer_request
            config = RequestConfig(max_tokens=128, temperature=0)
        else:
            fields = copy.deepcopy(base_fields)
            for message in fields["messages"]:
                if message.get("role") == "user":
                    message["content"] = message["content"] + "\\n" + MEMER_RETRY_NOTE
            self._memer_log({**fields, "retry": retry})
            request = InferRequest(**fields)
            config = RequestConfig(max_tokens=128, temperature=MEMER_RETRY_TEMPERATURE)
        response = self.engine.infer([request], request_config=config)
        response = response[0].choices[0].message.content
        print("Response: ", response)
        self._memer_log({"response": response} if retry == 0 else {"response": response, "retry": retry})
        try:
            _subtask, vla_subgoal, merged, positions = self._memer_validate(response)
        except Exception as e:
            print(f"Error updating history subgoals: {e}")
            self._memer_errors.append(f"{type(e).__name__}: {e}")
            continue
        self.key_frame_paths = merged
        self.subgoals.append(vla_subgoal)
        self._memer_last_positions = positions
        return vla_subgoal
    if self.subgoals:
        self._memer_fallback = "last_valid"
        self._memer_log({"fallback_used": 1, "fallback": "last_valid", "subgoal": self.subgoals[-1],
                         "errors": self._memer_errors})
        return self.subgoals[-1]
    self._memer_fallback = "model_response_error"
    self._memer_log({"fallback_used": 0, "fallback": "model_response_error", "errors": self._memer_errors})
    raise MemERResponseError(f"MemER replies invalid {MEMER_MAX_TRIES} times and no previous valid subgoal: "
                             + " | ".join(self._memer_errors)[:600])
'''
MEMER_COMPAT_SHA256 = hashlib.sha256(MEMER_COMPAT_SOURCE.encode("utf-8")).hexdigest()


def official_method_name(name: str) -> str:
    """Name of a patched official method after renaming: ``call`` -> ``_official_call``, ``_get_x`` ->
    ``_official_get_x``."""
    return "_official" + ("" if name.startswith("_") else "_") + name


def patch_memer_class(cls_node: ast.ClassDef) -> ast.ClassDef:
    """AST patch: rename the official ``MEMER_PATCHED_METHODS`` to ``_official_<name>`` (function bodies untouched),
    then append the methods of ``MEMER_COMPAT_SOURCE`` at the end of the class body. A missing patched method in the
    official class raises ``KeyError`` (upstream changed; never applied silently)."""
    have = {n.name for n in cls_node.body if isinstance(n, ast.FunctionDef)}
    missing = sorted(set(MEMER_PATCHED_METHODS) - have)
    if missing:
        raise KeyError(f"Qwen3VLModelMemER is missing {missing}; the compatibility layer does not apply")
    for n in cls_node.body:
        if isinstance(n, ast.FunctionDef) and n.name in MEMER_PATCHED_METHODS:
            n.name = official_method_name(n.name)
    compat = ast.parse(MEMER_COMPAT_SOURCE, filename="<MEMER_COMPAT_SOURCE>")
    cls_node.body.extend(compat.body)
    return cls_node


def memer_compat_extra(swift_names: dict | None) -> dict:
    """Names used by the compatibility-layer methods (injected into the extraction namespace)."""
    extra = {"copy": copy, "MemERResponseError": MemERResponseError, "MEMER_RETRY_NOTE": MEMER_RETRY_NOTE,
             "MEMER_MAX_TRIES": MEMER_MAX_TRIES, "MEMER_RETRY_TEMPERATURE": MEMER_RETRY_TEMPERATURE,
             "MEMER_COMPAT_SHA256": MEMER_COMPAT_SHA256}
    extra.update(swift_names or {})
    return extra


def load_memer_model(d: str | Path | None = None, *, swift_names: dict | None = None, compat: bool = True) -> dict:
    """Extract ``subgoal_prediction/qwenvl/api_memer.py::Qwen3VLModelMemER``; ``compat=True`` (the default and the
    only use on both sides) applies the compatibility layer, ``compat=False`` is only for tests to get the verbatim
    official class for regression comparison. Applies the three environment variables from the official module
    header (same as ``api.py``)."""
    d = Path(d) if d is not None else official_robomme_dir()
    apply_official_env()
    import imageio

    transform = {"Qwen3VLModelMemER": patch_memer_class} if compat else None
    ns = extract_defs(d / "subgoal_prediction" / "qwenvl" / "api_memer.py", ["Qwen3VLModelMemER"],
                      {"imageio": imageio, **memer_compat_extra(swift_names)}, transform=transform)
    ns["__memer_compat_sha256__"] = MEMER_COMPAT_SHA256 if compat else None
    return ns


def load_groundsg(variant: str, *, env_runner_extra: dict | None = None, ws_module: Any = None,
                  qwen_extra: dict | None = None, with_env_runner: bool = True) -> dict:
    """Assembly of official definitions shared by both sides; returns
    ``{"utils","env_runner","predictor","qwen","eval","sha256","EnvRunner",...}``.

    * ``env_runner_extra``: dependencies for ``env_runner.py`` (the original side passes the real
      ``BenchmarkEnvBuilder``; tests pass stand-ins). With ``with_env_runner=False`` ``EnvRunner`` is not extracted
      (the new side does not use it and only takes ``pack_state``), and ``EnvRunner`` annotations in the subgoal
      predictor and evaluator are replaced by a placeholder class (annotations are evaluated at definition time and
      do not affect behavior).
    * ``ws_module``: injected as ``eval.py``'s ``_websocket_client_policy`` (must have
      ``MMEVLAWebsocketClientPolicy``); ``None`` uses the real ``openpi_client.websocket_client_policy``.
    * ``qwen_extra``: for the QwenVL / MemER variants, injects ``PtEngine`` / ``InferRequest`` / ``RequestConfig`` of
      the subgoal model source file; with ``None`` nothing is injected and the caller must add them before building
      the predictor (``import_swift_names``). The Oracle variant does not read the two qwenvl source files.
    * MemER: the ``qwen`` key is the ``Qwen3VLModelMemER`` namespace with the compatibility layer applied,
      ``memer_compat_sha256`` is the implementation fingerprint; ``sha256`` records the whole-file sha256 of the
      official ``api_memer.py``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant={variant!r} is not one of {VARIANTS}")
    d = official_robomme_dir()
    apply_official_env()
    import cv2
    import imageio

    utils = extract_defs(d / "utils.py", UTILS_NAMES, {"cv2": cv2, "imageio": imageio})
    er_names = ENV_RUNNER_NAMES if with_env_runner else ["pack_state"]
    er_extra = {"TASK_NAME_LIST": utils["TASK_NAME_LIST"]}
    er_extra.update(env_runner_extra or {})
    env_runner = extract_defs(d / "env_runner.py", er_names, er_extra)
    runner_cls = env_runner.get("EnvRunner") or type("EnvRunner", (), {"__doc__": "annotation placeholder (the new side does not use the official EnvRunner)"})
    qwen = None
    pred_extra: dict[str, Any] = {"EnvRunner": runner_cls, "EpisodeState": utils["EpisodeState"],
                                  "SUBGOAL_TYPES": utils["SUBGOAL_TYPES"],
                                  "TASK_WITH_VIDEO_DEMO": utils["TASK_WITH_VIDEO_DEMO"]}
    if variant == VARIANT_QWENVL:
        qwen = extract_defs(d / "subgoal_prediction" / "qwenvl" / "api.py", ["Qwen3VLModel"],
                            {"imageio": imageio, **(qwen_extra or {})})
        pred_extra["Qwen3VLModel"] = qwen["Qwen3VLModel"]
    elif variant == VARIANT_MEMER:
        qwen = load_memer_model(d, swift_names=qwen_extra, compat=True)
        pred_extra["Qwen3VLModelMemER"] = qwen["Qwen3VLModelMemER"]
    predictor = extract_defs(d / "subgoal_predictor.py", PREDICTOR_NAMES[variant], pred_extra)
    if ws_module is None:
        from openpi_client import websocket_client_policy as ws_module  # noqa: N813 same name as official
    ev = extract_defs(d / "eval.py", EVAL_NAMES, {
        "_websocket_client_policy": ws_module, "pack_buffer": utils["pack_buffer"], "check_args": utils["check_args"],
        "TASK_NAME_LIST": utils["TASK_NAME_LIST"], "TASK_WITH_VIDEO_DEMO": utils["TASK_WITH_VIDEO_DEMO"],
        "SUBGOAL_TYPES": utils["SUBGOAL_TYPES"], "EpisodeState": utils["EpisodeState"],
        "RolloutRecorder": utils["RolloutRecorder"], "EnvRunner": runner_cls,
        "build_subgoal_predictor": predictor["build_subgoal_predictor"],
        "SubgoalPredictorBase": predictor["SubgoalPredictorBase"],
    })
    sha = {"utils.py": utils["__source_sha256__"], "env_runner.py": env_runner["__source_sha256__"],
           "subgoal_predictor.py": predictor["__source_sha256__"], "eval.py": ev["__source_sha256__"]}
    if qwen is not None:
        sha[SUBGOAL_MODEL_SOURCES[variant][0]] = qwen["__source_sha256__"]
    return {"variant": variant, "dir": str(d), "utils": utils, "env_runner": env_runner, "predictor": predictor,
            "qwen": qwen, "eval": ev, "sha256": sha, "EnvRunner": env_runner.get("EnvRunner"),
            "pack_state": env_runner["pack_state"], "Args": ev["Args"], "EpisodeEvaluator": ev["EpisodeEvaluator"],
            "memer_compat_sha256": MEMER_COMPAT_SHA256 if variant == VARIANT_MEMER else None}


def make_args(defs: dict, *, variant: str, host: str, port: int, max_steps: int, model_seed: Any,
              adapter_path: str | None = None, memer_adapter_path: str | None = None,
              save_dir: str = "runs/evaluation") -> Any:
    """Build the official ``Args`` per variant: ``subgoal_type="grounded_subgoal"``, exactly one of ``use_oracle`` /
    ``use_qwenvl`` / ``use_memer`` true (asserted after construction; the official ``build_subgoal_predictor``
    silently takes the highest priority when several are on, which is not allowed here), ``model_seed`` required and
    written explicitly into ``Args.model_seed`` (not the official default 42), then passed through the official
    ``check_args``.

    Adapter pairing: QwenVL must give exactly ``adapter_path``, MemER must give exactly ``memer_adapter_path``,
    Oracle must give neither; a wrong pairing raises ``ValueError``."""
    if variant not in VARIANTS:
        raise ValueError(f"variant={variant!r} is not one of {VARIANTS}")
    seed = check_policy_seed(model_seed)
    want_q, want_m = variant == VARIANT_QWENVL, variant == VARIANT_MEMER
    if bool(adapter_path) != want_q:
        raise ValueError(f"{variant}: qwenvl_groundSG_adapter_path must be used with, and only with, ground-sg-qwenvl (got {adapter_path!r})")
    if bool(memer_adapter_path) != want_m:
        raise ValueError(f"{variant}: memer_adapter_path must be used with, and only with, ground-sg-memer (got {memer_adapter_path!r})")
    kw: dict[str, Any] = dict(host=host, port=int(port), max_steps=int(max_steps), save_dir=save_dir,
                              subgoal_type="grounded_subgoal", use_oracle=variant == VARIANT_ORACLE,
                              use_qwenvl=want_q, use_memer=want_m, model_seed=seed)
    if want_q:
        kw["qwenvl_groundSG_adapter_path"] = str(adapter_path)
    if want_m:
        kw["memer_adapter_path"] = str(memer_adapter_path)
    args = defs["Args"](**kw)
    assert_one_predictor(args)
    defs["utils"]["check_args"](args)
    return args


def assert_one_predictor(args: Any) -> None:
    """Exactly one of ``use_oracle`` / ``use_qwenvl`` / ``use_memer`` is true, and ``use_gemini`` is false."""
    flags = (bool(args.use_oracle), bool(args.use_qwenvl), bool(getattr(args, "use_memer", False)))
    if sum(flags) != 1 or getattr(args, "use_gemini", False):
        raise AssertionError(f"use_oracle={args.use_oracle} use_qwenvl={args.use_qwenvl} "
                             f"use_memer={getattr(args, 'use_memer', None)} use_gemini={getattr(args, 'use_gemini', None)}"
                             ": exactly one of oracle / qwenvl / memer is required")


def build_predictor(defs: dict, args: Any, save_dir: str | Path) -> Any:
    """The official ``build_subgoal_predictor`` (asserting mutual exclusion once more before construction). QwenVL /
    MemER first set the offline runtime constraints and call ``seed_everything`` with ``Args.model_seed``; the real
    swift is imported here if swift names were not injected; engine parameters such as
    ``attn_impl='flash_attention_2'`` are always taken verbatim from the official source, unchanged."""
    assert_one_predictor(args)
    if args.use_qwenvl or getattr(args, "use_memer", False):
        apply_qwen_runtime_env()
        if "PtEngine" not in defs["qwen"]:
            defs["qwen"].update(import_swift_names())
        seed_everything(check_policy_seed(args.model_seed))
    return defs["predictor"]["build_subgoal_predictor"](args, Path(save_dir))


def canonical_bytes(obj: Any) -> bytes:
    """Normalized bytes of a request (same function on both sides): dicts sorted by key; arrays record dtype, shape
    and C-contiguous bytes; strings as UTF-8."""
    out = bytearray()

    def put(x: Any) -> None:
        if isinstance(x, dict):
            out.extend(b"{")
            for k in sorted(x, key=str):
                put(str(k))
                out.extend(b":")
                put(x[k])
                out.extend(b",")
            out.extend(b"}")
        elif isinstance(x, (list, tuple)):
            out.extend(b"[")
            for v in x:
                put(v)
                out.extend(b",")
            out.extend(b"]")
        elif isinstance(x, (bytes, bytearray)):
            out.extend(b"b%d:" % len(x) + bytes(x))
        elif isinstance(x, str):
            b = x.encode("utf-8")
            out.extend(b"s%d:" % len(b) + b)
        elif x is None or isinstance(x, (bool, int, float)):
            out.extend(("p:" + repr(x)).encode())
        else:
            a = np.ascontiguousarray(np.asarray(x))
            out.extend(f"a:{a.dtype.str}|{a.shape}|".encode() + a.tobytes())

    put(obj)
    return bytes(out)


def ws_shim(factory) -> Any:
    """Stand-in for ``_websocket_client_policy`` in ``eval.py``: ``MMEVLAWebsocketClientPolicy(host, port)`` ->
    ``factory``."""
    return types.SimpleNamespace(MMEVLAWebsocketClientPolicy=factory)
