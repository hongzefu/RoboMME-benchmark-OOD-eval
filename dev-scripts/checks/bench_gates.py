"""benchmark 仓 S1 仓级闸门：对 RoboMME-benchmark-OOD 的一个提交逐项输出 BENCH_* 判定行（1008 拆分方案第二部分 二′）。

用法（在评估仓根）：
  uv run --no-sync python dev-scripts/checks/bench_gates.py --bench /data/hongzefu/RoboMME-benchmark-OOD [--smoke --gpu 0]

除 --smoke 外全部只读（git 命令与文件读取）；--smoke 用 benchmark 仓自己的 .venv 起 dummy 跑 MoveCube × {hard-verify, ood} 各 1 局（2 次 reset）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OFFICIAL = "016ac1c4ef3df2b88488abc19db08f3de83647b5"
SRC = "fd0017d6a06b7a225ebd026e188b2c8dba10b951"
OLD_REPO = "/data/hongzefu/robomme_benchmark_MotionJEPANewTask"
ADDED_ROOTS = ("src/robomme_hard/", "tests/robomme_hard/")
# .claude/settings.json：规则正本 9ec0a8c 起各仓库项目级 Claude Code 设置（与规则三件同属规则文件；用户 2026-10-08 裁决「锁在 a5efb99」时并入白名单）。
ADDED_FILES = {"scripts/evaluation_ood.py", "scripts/README_ood.md", "AGENTS.md", "CLAUDE.md", "greatlakes.md", ".claude/settings.json"}
MODIFIED_OK = {"pyproject.toml", "readme.md", ".gitignore", "uv.lock"}
H1_CHANGED = {"README.md", "UPSTREAM.json", "__init__.py", "env_record_wrapper/hard_builder.py", "env_record_wrapper/hard_specs.py"}
GEN_SYMBOLS = ("from_v4_specs", "v4_episodes", "HARD_TRAIN_TASKS", "XHARD0_IN_TEST_HARD", "SPECS_ROOT_ENV", "_override_cells", "_root_specs")
V11 = "51e05feb05c61b9a5a25cef3ba189ad625f2cd83"
V12_PATHS = {
    "src/robomme_ood/robomme_env/utils/subgoal_evaluate_func.py",
    "src/robomme_ood/robomme_env/MoveCube.py",
    "tests/robomme_ood/unit/hard/tasks/test_stopcube.py",
    "tests/robomme_ood/unit/hard/tasks/test_movecube.py",
    "tests/robomme_ood/unit/hard/contracts.delta.json",
    "tests/robomme_ood/contract/benchmark_contracts.json",
}


def gate_version(b: Path, profile: str, base: str, candidate: str) -> int:
    """只读确定对象或暂存区；新版不调用旧版动态检查。"""
    if base != V11:
        raise ValueError(f"新版基线必须固定为 {V11}")
    def tree(ref):
        rows = git(b, "ls-files", "-s") if ref == "index" else git(b, "ls-tree", "-r", ref)
        result = {}
        for row in rows.splitlines():
            meta, path = row.split("\t", 1)
            fields = meta.split()
            if ref == "index" and fields[2] != "0":
                raise ValueError(f"暂存区有冲突：{path}")
            result[path] = (fields[0], fields[1] if ref == "index" else fields[2])
        return result
    before, after, upstream = tree(base), tree(candidate), tree(OFFICIAL)
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    official = {p for p in upstream if p.startswith("src/robomme/")}
    official_bad = {p for p in official if after.get(p) != upstream[p]}
    official_bad |= {p for p in after if p.startswith("src/robomme/") and p not in official}
    print(f"BENCH_UPSTREAM={'PASS' if official and not official_bad else 'FAIL'} files={len(official)} diffs={len(official_bad)}")
    def blob(path):
        return subprocess.run(["git", "-C", str(b), "cat-file", "blob", after[path][1]],
                              check=True, capture_output=True).stdout
    sha_path = "tests/robomme_ood/contract/packaged_specs.sha256"
    specs = {}
    for row in blob(sha_path).decode().splitlines():
        if row.strip():
            sha, path = row.split()
            specs[path.lstrip("*")] = sha
    bad_specs = sum(hashlib.sha256(blob("src/robomme_ood/env_metadata/ood/" + p)).hexdigest() != h
                    for p, h in specs.items())
    specs_ok = len(specs) == 5 and not bad_specs
    print(f"BENCH_SPECS_SHA={'PASS' if specs_ok else 'FAIL'} files={len(specs)} mismatches={bad_specs}")
    production = {p for p in V12_PATHS if p.startswith("src/")}
    scope_ok = (not changed if profile == "v1.1" else
                production <= changed <= V12_PATHS and before.keys() == after.keys()
                and all(before[p][0] == after[p][0] for p in changed))
    frozen = {p for p in before if p.startswith("src/robomme_ood/env_metadata/")}
    frozen |= {sha_path, "src/robomme_ood/UPSTREAM.json"}
    frozen_ok = bool(frozen) and all(before[p] == after.get(p) for p in frozen)
    for p in sorted(changed):
        print(f"CHANGED {p}")
    name = "BENCH_V12_SCOPE" if profile == "v1.2" else "BENCH_V11_SCOPE"
    ok = scope_ok and frozen_ok and specs_ok and bool(official) and not official_bad
    print(f"{name}={'PASS' if ok else 'FAIL'} tracked={len(after)} changed={len(changed)} frozen={len(frozen)}")
    return 0 if ok else 1


def git(repo: Path, *a: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=check, capture_output=True, text=True).stdout


def gate_delta(b: Path) -> str:
    rows = [ln.split("\t") for ln in git(b, "diff", "--name-status", OFFICIAL, "HEAD").splitlines() if ln]
    added_roots, added_files, modified, deleted, unexpected = set(), set(), set(), 0, []
    for st, *paths in rows:
        p = paths[-1]
        if st.startswith("D"):
            deleted += 1
            unexpected.append(f"D {p}")
        elif st == "A" and p.startswith(ADDED_ROOTS):
            added_roots.add(next(r for r in ADDED_ROOTS if p.startswith(r)))
        elif st == "A" and p in ADDED_FILES:
            added_files.add(p)
        elif st == "M" and p in MODIFIED_OK:
            modified.add(p)
        else:
            unexpected.append(f"{st} {p}")
    if "uv.lock" in modified:
        unexpected.append("M uv.lock（需逐包核对，本轮未预期新增测试依赖）")
    ok = not unexpected and len(added_roots) == 2 and len(added_files) == len(ADDED_FILES) and deleted == 0
    for u in unexpected:
        print(f"  意外：{u}")
    return (f"BENCH_DELTA={'PASS' if ok else 'FAIL'} added_roots={len(added_roots)} added_files={len(added_files)} "
            f"modified={len(modified)} deleted={deleted} unexpected={len(unexpected)}")


def gate_upstream(b: Path) -> str:
    paths = ["src/robomme", "challenge_interface", "scripts/dataset_replay.py", "scripts/evaluation.py", "scripts/run_example.py",
             "Dockerfile", "doc", "assets", "LICENSE"]
    diff = git(b, "diff", "--name-only", OFFICIAL, "HEAD", "--", *paths).split()
    porcelain = git(b, "status", "--porcelain").splitlines()
    porcelain = [p for p in porcelain if "docs/subagent-stats" not in p]
    ok = not diff and not porcelain
    return f"BENCH_UPSTREAM={'PASS' if ok else 'FAIL'} files={len(diff)} clean={int(not porcelain)}"


def gate_unchanged(b: Path) -> str:
    new = {ln.split("\t")[-1]: ln.split()[2] for ln in git(b, "ls-tree", "-r", "HEAD", "src/robomme_hard").splitlines()}
    old = {ln.split("\t")[-1]: ln.split()[2] for ln in git(Path(OLD_REPO), "ls-tree", "-r", SRC, "src/robomme_hard").splitlines()}
    old = {k: v for k, v in old.items() if not k.startswith("src/robomme_hard/env_metadata/train/")}
    diffs = []
    for k in sorted(set(new) | set(old)):
        rel = k[len("src/robomme_hard/"):]
        if rel in H1_CHANGED:
            continue
        if new.get(k) != old.get(k):
            diffs.append(k)
    for d in diffs:
        print(f"  不同：{d}")
    n = len([k for k in new if k[len("src/robomme_hard/"):] not in H1_CHANGED])
    return f"BENCH_UNCHANGED={'PASS' if not diffs else 'FAIL'} files={n} diffs={len(diffs)}"


def gate_specs_sha(b: Path) -> str:
    want = {}
    for ln in (b / "tests/robomme_hard/contract/packaged_specs.sha256").read_text().splitlines():
        if ln.strip():
            h, p = ln.split()
            want[p.lstrip("*")] = h
    bad = 0
    for p, h in want.items():
        got = hashlib.sha256((b / "src/robomme_hard/env_metadata/ood" / p).read_bytes()).hexdigest()
        bad += got != h
    return f"BENCH_SPECS_SHA={'PASS' if (len(want) == 5 and not bad) else 'FAIL'} files={len(want)}"


def gate_manifest(b: Path) -> str:
    m = json.loads((b / "src/robomme_hard/UPSTREAM.json").read_text())
    want = m.pop("manifest_sha256")
    got = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    dangling = 0

    def walk(o):
        nonlocal dangling
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and "/" in k and k.endswith((".py", ".json", ".jsonl")) and not (b / k).exists() \
                        and not k.startswith("src/robomme/env_metadata/train"):
                    dangling += k.startswith(("scripts/", "src/robomme_hard/"))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(m)
    ok = got == want and dangling == 0 and m.get("vendor") in ({}, None)
    return f"BENCH_MANIFEST={'PASS' if ok else 'FAIL'} self_sig={int(got == want)} dangling={dangling}"


def gate_entry_diff(b: Path) -> str:
    off = git(b, "show", f"{OFFICIAL}:scripts/evaluation.py").splitlines()
    new = (b / "scripts/evaluation_ood.py").read_text().splitlines()
    import difflib
    removed = [l[1:] for l in difflib.unified_diff(off, new, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in difflib.unified_diff(off, new, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    want_removed = {"from robomme.env_record_wrapper import BenchmarkEnvBuilder", 'dataset="test",', "max_steps=1300,"}
    rm_ok = len(removed) == 3 and all(any(w in r for w in want_removed) for r in removed)
    imp = any(a.strip() == "from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder" for a in added)
    ds = any(a.strip() == "dataset=DATASET," for a in added)
    ms = any(a.strip().startswith("max_steps=DATASET_MAX_STEPS[DATASET],") for a in added)
    tbl = any(a.strip() == 'DATASET_MAX_STEPS = {"hard-verify": 1300, "ood": 1800}' for a in added)
    dflt = any(a.strip() == 'DATASET = "ood"' for a in added)
    other = [a for a in added if a.strip() and not a.strip().startswith("#") and not (
        "robomme_hard.env_record_wrapper" in a or "dataset=DATASET" in a or "DATASET_MAX_STEPS" in a or a.strip() == 'DATASET = "ood"')]
    changes = sum([imp, ds, ms, tbl and dflt])
    ok = rm_ok and changes == 4 and not other
    return f"BENCH_ENTRY_DIFF={'PASS' if ok else 'FAIL'} changes={changes} unexpected={len(other) + (0 if rm_ok else 1)}"


def gate_no_gen(b: Path) -> str:
    hits = []
    for root in ("src/robomme_hard", "scripts"):
        for p in (b / root).rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    name = node.name
                elif isinstance(node, ast.Name):
                    name = node.id
                elif isinstance(node, ast.Attribute):
                    name = node.attr
                elif isinstance(node, ast.arg):
                    name = node.arg if node.arg == "specs_root" and p.name == "hard_builder.py" else None
                if name in GEN_SYMBOLS or (name == "specs_root" and isinstance(node, ast.arg)):
                    hits.append(f"{p.relative_to(b)}:{name}")
    absent = sum(not (b / d).exists() for d in ("src/robomme_hard/env_metadata/train", "scripts/injection-dev", "scripts/parity"))
    readme = (b / "readme.md").read_text(encoding="utf-8")
    gen_cmd = bool(re.search(r"generate_h5|injection-dev|freeze_specs|seed_layout", readme))
    for h in hits:
        print(f"  命中：{h}")
    ok = not hits and absent == 3 and not gen_cmd
    return f"BENCH_NO_GEN={'PASS' if ok else 'FAIL'} ast_hits={len(hits)} dirs_absent={absent}"


def gate_scripts(b: Path) -> str:
    files = sorted(p.name for p in (b / "scripts").iterdir())
    want = sorted(["dataset_replay.py", "evaluation.py", "run_example.py", "evaluation_ood.py", "README_ood.md"])
    return f"BENCH_SCRIPTS={'PASS' if files == want else 'FAIL'} files={len(files)}"


def run_bench_py(b: Path, code: str, env_extra: dict | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env["UV_PROJECT_ENVIRONMENT"] = str(b / ".venv")
    env.update(env_extra or {})
    return subprocess.run(["uv", "run", "--no-sync", "python", "-c", code], cwd=b, env=env, capture_output=True, text=True,
                          timeout=timeout)


def gate_datasets(b: Path) -> str:
    code = r'''
import unittest.mock as um, gymnasium
with um.patch.object(gymnasium, "make", side_effect=RuntimeError("禁止 gym.make")):
    from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder as B
    rej = 0
    for d in ("train", "test", "val", "xhard1"):
        try: B(env_id="MoveCube", dataset=d)
        except ValueError: rej += 1
    tasks = B.get_task_list()
    hv = sum(B(env_id=t, dataset="hard-verify").get_episode_num() for t in tasks)
    ood = sum(B(env_id=t, dataset="ood").get_episode_num() for t in tasks)
    dflt = B(env_id="MoveCube").dataset
    print(f"BENCH_DATASETS={'PASS' if (rej==4 and hv==192 and ood==800 and dflt=='ood' and len(tasks)==16) else 'FAIL'} rejected={rej} hard_verify={hv} ood={ood} default={dflt}")
'''
    r = run_bench_py(b, code)
    line = [l for l in r.stdout.splitlines() if l.startswith("BENCH_DATASETS=")]
    return line[-1] if line else f"BENCH_DATASETS=FAIL reason=no_output stderr_tail={r.stderr[-300:]!r}"


def gate_pytest(b: Path, name: str, nodes: list[str]) -> str:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run(["uv", "run", "--no-sync", "python", "-m", "pytest", *nodes, "-m", "slow", "-q", "-p", "no:cacheprovider"],
                       cwd=b, env=env, capture_output=True, text=True, timeout=1200)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    m_pass = re.search(r"(\d+) passed", tail)
    m_skip = re.search(r"(\d+) skipped", tail)
    bad = re.search(r"failed|error", tail)
    passed, skipped = int(m_pass.group(1)) if m_pass else 0, int(m_skip.group(1)) if m_skip else 0
    ok = r.returncode == 0 and passed > 0 and skipped == 0 and not bad
    return f"{name}={'PASS' if ok else 'FAIL'} passed={passed} skipped={skipped}  # {tail}"


def gate_smoke(b: Path, gpu: str) -> str:
    # 照官方 scripts/evaluation.py 的 DummyModel：基准关节角 + N(0, 0.01) 噪声、夹爪不动；终止于 terminated/truncated 或 status=error。
    code = r'''
import robomme_hard, numpy as np
print("robomme_hard.__file__ =", robomme_hard.__file__)
from robomme_hard.env_record_wrapper import BenchmarkEnvBuilder as B
np.random.seed(7)
base = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0], dtype=np.float32)
n = resets = 0
for ds, cap in (("hard-verify", 1300), ("ood", 1800)):
    b = B(env_id="MoveCube", dataset=ds, action_space="joint_angle", max_steps=cap)
    env = b.make_env_for_episode(0)
    obs, info = env.reset(); resets += 1
    steps = 0; outcome = "unknown"
    while True:
        noise = np.random.normal(0, 0.01, base.shape); noise[-1:] = 0.0
        obs, r, term, trunc, info = env.step(base + noise); steps += 1
        if info is not None and info.get("status") == "error":
            outcome = "error"; break
        if term or trunc:
            outcome = info.get("status", "unknown"); break
    env.close(); n += 1
    print(f"SMOKE {ds} steps={steps} outcome={outcome}")
print(f"BENCH_SMOKE={'PASS' if n==2 and resets==2 else 'FAIL'} identities={n} resets={resets}")
'''
    r = run_bench_py(b, code, {"CUDA_VISIBLE_DEVICES": gpu}, timeout=3600)
    for l in r.stdout.splitlines():
        if l.startswith(("robomme_hard.__file__", "SMOKE ")):
            print("  " + l)
    line = [l for l in r.stdout.splitlines() if l.startswith("BENCH_SMOKE=")]
    return line[-1] if line else f"BENCH_SMOKE=FAIL reason=no_output stderr_tail={r.stderr[-500:]!r}"


def main() -> int:
    ap = argparse.ArgumentParser(description="benchmark 仓 S1 仓级闸门")
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--profile", choices=("v1.0", "v1.1", "v1.2"), required=True)
    ap.add_argument("--base", default=V11)
    ap.add_argument("--candidate", default="HEAD")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="另跑 BENCH_SMOKE（2 次 reset，占 1 张卡）")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-slow", action="store_true", help="不跑 BENCH_PACKAGE／BENCH_REGISTRY")
    a = ap.parse_args()
    b = a.bench.resolve()
    if a.profile != "v1.0":
        if not a.static_only or a.smoke:
            ap.error("新版必须指定 --static-only，禁止 smoke")
        return gate_version(b, a.profile, a.base, a.candidate)
    if a.static_only:
        ap.error("v1.0 使用原门禁；新版静态检查请显式选择 v1.1 或 v1.2")
    print(f"bench={b} head={git(b, 'rev-parse', 'HEAD').strip()}")
    lines = [gate_delta(b), gate_upstream(b), gate_unchanged(b), gate_specs_sha(b), gate_manifest(b), gate_datasets(b),
             gate_entry_diff(b), gate_no_gen(b), gate_scripts(b)]
    if not a.skip_slow:
        lines.append(gate_pytest(b, "BENCH_PACKAGE", ["tests/robomme_hard/static/test_package.py"]))
        lines.append(gate_pytest(b, "BENCH_REGISTRY", ["tests/robomme_hard/contract/test_registry.py"]))
    if a.smoke:
        lines.append(gate_smoke(b, a.gpu))
    for l in lines:
        print(l)
    fails = [l for l in lines if "=FAIL" in l.split()[0]]
    print(f"BENCH_GATES={'PASS' if not fails else 'FAIL'} gates={len(lines)} fail={len(fails)}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
