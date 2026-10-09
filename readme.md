# RoboMME-benchmark-OOD-eval

Evaluation harness for [RoboMME-benchmark-OOD](https://github.com/hongzefu/RoboMME-benchmark-OOD). It runs a
policy on the benchmark's episodes and writes, for every episode, the official-layout video, lossless raw camera
frames, and the raw actions with a per-step record, plus a per-dataset `log.json` summary.

The benchmark and the model code are git submodules under `third_party/`, each pinned to a commit:

| Submodule | Used for |
|---|---|
| `third_party/robomme_benchmark` | Environments, task builders and episode specs (`hard-verify`, `ood`) |
| `third_party/mme-vla` | FrameSamp+Modulation and GroundSG policy servers |
| `third_party/SimpleMemVLA` | SimpleMemVLA policy server |
| `third_party/PonderPounce` | PonderPounce policy server |
| `third_party/Astra-on-RoboMME` | 3-tier Astra |

## Install

Linux x86_64 with an NVIDIA GPU and Vulkan (for ManiSkill / SAPIEN rendering), [uv](https://docs.astral.sh/uv/),
and `ffmpeg`/`ffprobe` on `PATH`.

```bash
git clone https://github.com/hongzefu/RoboMME-benchmark-OOD-eval.git
cd RoboMME-benchmark-OOD-eval
git submodule update --init  # first level only; do NOT use --recursive / --recurse-submodules
uv sync --extra dev          # main environment: benchmark, simulator, evaluation package, pytest
```

The submodules are required: `uv.lock` installs the benchmark and the OpenPI client as editable path dependencies
from them. Initialize only the first level: `third_party/mme-vla` and `third_party/Astra-on-RoboMME` carry their own
copy of the benchmark as a nested submodule, which must stay empty (the FrameSamp+Modulation / GroundSG server preflight refuses to start otherwise), so
`git clone --recurse-submodules` is not suitable.

Model servers run in their own interpreters, so the main environment never imports model code:

| Model | Server interpreter (override with) | How to build it |
|---|---|---|
| `perceptual-framesamp-modul`, `groundsg` | `third_party/mme-vla/.venv/bin/python` (`MME_VLA_PY`) | Follow the setup in `third_party/mme-vla` |
| `smvla` | `envs/smvla-env/.venv/bin/python` (`SMVLA_PY`) | See the header of `envs/smvla-env/pyproject.toml` |
| `pp` | `third_party/PonderPounce/.venv/bin/python` (`PP_PY`) | Follow the setup in `third_party/PonderPounce` |

`envs/client-env` is the client-side environment for GroundSG-QwenVL, PonderPounce and Astra; each `envs/*`
project is standalone and is synced with `UV_PROJECT_ENVIRONMENT=<dir>/.venv uv sync --frozen` from its directory
(details in each `pyproject.toml` header).

## Run

```bash
uv run scripts/evaluate.py --model <name> --dataset hard-verify[,ood] --seed <N> --out <dir> \
    [--tasks A,B] [--episodes a:b] [--gpus 0[,1]] [--ckpt <dir>] [model options]
```

| Option | Meaning |
|---|---|
| `--model` | `dummy`, `perceptual-framesamp-modul`, `groundsg`, `smvla`, `pp`, `astra` |
| `--dataset` | `hard-verify`, `ood`, or both comma-separated (one loaded policy runs them in order) |
| `--seed` | Model seed (`policy_seed`) |
| `--tasks` | Comma-separated task names; default is all 16 tasks |
| `--episodes a:b` | Half-open range of episode indices (`0:2` = episodes 0 and 1); default is all |
| `--gpus` | GPU ids; the policy server uses the first one (Astra's monitor uses the second) |
| `--ckpt` | Checkpoint directory; **required** for `perceptual-framesamp-modul`, `groundsg`, `smvla`, `pp` |
| `--groundsg-variant` | `ground-sg-oracle`, `ground-sg-qwenvl` or `ground-sg-memer` |
| `--cfg key=value` | Any other model option (repeatable); unknown `--name value` pairs are passed through too |

Episodes that already have a `result.json` are skipped, so an interrupted run can be resumed with the same command.

Examples:

```bash
# No weights needed: a random policy, one task, one episode
uv run scripts/evaluate.py --model dummy --dataset hard-verify --tasks VideoUnmask --episodes 0:1 --seed 0 --out runs/dummy

uv run scripts/evaluate.py --model smvla --dataset ood --seed 0 --ckpt ~/ckpts/smvla --out runs/smvla
uv run scripts/evaluate.py --model perceptual-framesamp-modul --dataset ood --seed 0 --ckpt ~/ckpts/framesamp/79999 --out runs/fs
uv run scripts/evaluate.py --model groundsg --groundsg-variant ground-sg-oracle --dataset ood --seed 0 --ckpt ~/ckpts/groundsg/79999 --out runs/gsg
uv run scripts/evaluate.py --model pp --dataset ood --seed 0 --ckpt ~/ckpts/pp --out runs/pp
```

### Checkpoints

There are no default checkpoint paths: the harness never guesses or downloads weights. Forgetting `--ckpt` for a
model that needs it is an argument error:

```
$ uv run scripts/evaluate.py --model smvla --dataset ood --seed 0 --out runs/smvla
scripts/evaluate.py: error: --ckpt is required for model smvla
```

What `--ckpt` must point to (checked before the server starts):

| Model | `--ckpt` points to | Required contents |
|---|---|---|
| `perceptual-framesamp-modul` | The step directory of the OpenPI checkpoint (e.g. `.../79999`) | `params/` and `assets/` inside it; `../history_config.txt` naming `perceptual-framesamp-modul.yaml` |
| `groundsg` | The step directory of the OpenPI checkpoint | `params/` and `assets/`; `../history_config.txt` naming `symbolic-grounded-subgoal.yaml` |
| `smvla` | The SimpleMemVLA checkpoint directory | Loadable by the SimpleMemVLA server |
| `pp` | The PonderPounce checkpoint directory | `norm_stats.json` inside it |

The path and a per-file sha256 of the checkpoint are recorded in each dataset's `log.json`.

## Outputs

```
<out>/rollouts/<model>/<dataset>/seed<seed>/
  results.jsonl                 one line per finished episode
  log.json                      dataset summary (success rate, identities, checkpoint fingerprint)
  progress.json
  videos/<Task>_ep<N>_<success|fail|timeout>_<goal>_<tier>.mp4      official-layout video
  raw/<Task>_ep<N>_<tier>/
    front.mkv, wrist.mkv        lossless AV1 4:4:4 camera streams
    arrays.npz, frames-*.jsonl  raw actions and per-step records
    meta.json, events.jsonl, trace.jsonl, result.json
```

Exit codes: `0` done; `2` argument error; `3` run blocked (episode identity mismatch, server mismatch or died,
Astra stop rule); `5` reset budget exhausted.

## Tests

```bash
uv run --no-sync python -m pytest -m 'not slow' -q    # CPU only, no simulator reset, no GPU, no network
uv run --no-sync python -m pytest -m slow -q          # spawns bash / ffmpeg / websockets / wheel builds
```

## Branches

`main` is the public, English-only release. Development happens on the `dev` branch, which also carries internal
tooling and run records; `main` is updated from `dev` by a one-way sync and should not receive direct commits.

## License

Apache License 2.0, see [LICENSE](LICENSE).
