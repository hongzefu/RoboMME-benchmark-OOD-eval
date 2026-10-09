#!/usr/bin/env bash
# dev 侧各模型 checkpoint 的缺省路径（只在 dev 分支；1009 公开 main 与 dev 拆分计划第二部分「一」归 D 的两行）。
# 评估包 src/robomme_ood_eval 已不再内置缺省 ckpt（原 servers/__init__.py::DEFAULT_CKPTS），不带 --ckpt 即报错；
# dev 侧调用点从本文件取路径后显式传 --ckpt。下面三条路径原样搬自原 DEFAULT_CKPTS 表；pp 原表即为 None，必须显式给。
#
# 变量名沿用既有覆盖习惯（与 dev-scripts/orig/orig_seat_lib.sh 及 docs/validation/legacy-names.md 同名）：
#   FRAMESAMP_MODUL_CKPT  perceptual-framesamp-modul
#   GROUNDSG_CKPT         groundsg（全部 GroundSG 变体共用）
#   SMVLA_CKPT            smvla
# 只在变量未设或为空时给缺省值，调用方事先 export 的值优先。注意前两条在 sled-vail 本机盘，GL 计算节点上不存在，
# 在 GL 上跑这两个模型须事先 export 成 NFS 路径或直接给 --ckpt。
#
# 两种用法：
#   source dev-scripts/gl/ckpt_paths.sh          # 导出上面三个变量
#   bash dev-scripts/gl/ckpt_paths.sh <模型名>    # 打印该模型的 ckpt（无缺省的模型打印空行），退出码 0
#   （run_eval_gl.sh 用第二种：它不 source 任何文件。）

: "${FRAMESAMP_MODUL_CKPT:=/data/hongzefu/robomme_policy_learning_MotionJEPA/v1-store/models/official-mme-vla/perceptual-framesamp-modul/79999}"
: "${GROUNDSG_CKPT:=/data/hongzefu/robomme_policy_learning-vqa-test/runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999}"
: "${SMVLA_CKPT:=/nfs/turbo/coe-chaijy-unreplicated/hongzefu/SimpleMemVLA/checkpoints/simplememvla_robomme}"
export FRAMESAMP_MODUL_CKPT GROUNDSG_CKPT SMVLA_CKPT

# 模型名 → ckpt；没有缺省的模型（pp、astra、dummy 等）返回空串
ckpt_default_of() {
  case "$1" in
    perceptual-framesamp-modul) echo "$FRAMESAMP_MODUL_CKPT";;
    groundsg) echo "$GROUNDSG_CKPT";;
    smvla) echo "$SMVLA_CKPT";;
    *) echo "";;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ $# -eq 1 ]] || { echo "用法：bash ckpt_paths.sh <模型名>" >&2; exit 2; }
  ckpt_default_of "$1"
fi
