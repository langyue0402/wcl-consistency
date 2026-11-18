#!/usr/bin/env bash
# 新机器一条命令起来：装依赖 -> 验 CUDA -> 起看门狗（自动拉数据、建缓存、诊断、冒烟、23 run campaign）。
#
#   cd /root/wcl && bash scripts/bootstrap.sh
#
# 环境变量：
#   TIERS="core"           只跑主表 12 run（78 h），默认 core seeds abl alpha
#   EPOCHS=50              每 run 上限轮数
#   POWER_W=320            若设了就给 GPU 限功耗（租来的 4090 建议 320，降掉卡概率）
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124   默认清华源装 torch
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PY="${PY:-python3}"
POWER_W="${POWER_W:-0}"
mkdir -p logs

stamp() { date '+%F %T'; }
say() { echo "[$(stamp)] [boot] $*" | tee -a logs/bootstrap.log; }

say "1/5 系统依赖"
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      unzip curl jq libsndfile1 ffmpeg build-essential 2>/dev/null || say "[warn] apt 装包有失败，继续"
else
  say "[warn] 不是 apt 系统，跳过系统包"
fi

say "2/5 Python 依赖（清华源）"
"$PY" -m pip install -q -i "$PIP_INDEX" --upgrade pip 2>/dev/null || true
"$PY" -m pip install -q -i "$PIP_INDEX" -r requirements.txt 2>&1 | tail -3
# XLS-R 前端要 4.46.x；5.x 的 wav2vec2 接口变过，会崩
"$PY" -m pip install -q -i "$PIP_INDEX" "transformers==4.46.3" 2>&1 | tail -2

say "3/5 验 CUDA 与前向"
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用：先看驱动/容器是否给了 GPU"
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).dtype)
PY
[ $? -ne 0 ] && { say "CUDA 自检失败，停止"; exit 1; }

say "4/5 GPU 状态"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader
if [ "$POWER_W" != "0" ]; then
  nvidia-smi -pl "$POWER_W" >/dev/null 2>&1 && say "功耗限制设为 ${POWER_W} W" || say "[warn] 限功耗失败（可能无权限）"
fi
ulimit -n 65536 2>/dev/null || say "[warn] ulimit 没抬到 65536"

say "5/5 起看门狗（数据/缓存/诊断/冒烟/campaign 全自动，掉卡自动重启续跑）"
TIERS="${TIERS:-core seeds abl alpha}" EPOCHS="${EPOCHS:-50}" \
  setsid bash scripts/run_guard.sh < /dev/null >> logs/guard.log 2>&1 &
sleep 3
say "已启动，跟随进度： tail -f logs/guard.log   （或 tail -f logs/one_shot.log）"
