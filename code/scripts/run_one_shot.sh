#!/usr/bin/env bash
# 一把跑完整个 WCL 实验（预检 -> 建缓存 -> 诊断 -> 烟测 -> campaign -> 汇总）。
#
# 设计目标只有一条：**机器不在空转**。所以每一段都能断点续跑，任何一段失败都
# 记日志并继续下一段，绝不因为一个 run 崩了就让整台卡闲着。
#
#   bash scripts/run_one_shot.sh                       # 全流程（默认 core seeds abl alpha）
#   bash scripts/run_one_shot.sh --skip-cache          # 缓存已建好
#   DRY=1 bash scripts/run_one_shot.sh                 # 只打印计划
#   TIERS="core" bash scripts/run_one_shot.sh          # 只跑主表
#
# 断线保活（重要）：
#   setsid bash scripts/run_one_shot.sh < /dev/null > logs/one_shot.log 2>&1 &
#   之后 tail -f logs/one_shot.log 看进度；脚本每完成一个 run 就写 logs/heartbeat.txt
#
# 约定：
#   * 只有 python3，没有 python
#   * eval/preprocess 的 worker 一多就 "Too many open files"，进来先抬 ulimit
#   * 缓存默认 float16（156 GB -> 78 GB），fail 时才回 float32
set -uo pipefail
# 国内直连 huggingface 常常拉不动，默认走镜像（可用环境变量覆盖）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python3}"
BASE="${BASE:-configs/three_arm_phys.yaml}"
CAMP="${CAMP:-camp_v4}"
TIERS="${TIERS:-core seeds abl alpha}"
EPOCHS="${EPOCHS:-50}"                 # 上限；早停通常 35-45 轮停
DEV_PHASES="${DEV_PHASES:-clean,vad}"  # 早停只看这两相，省 60% 评估时间
WORKERS="${WORKERS:-$(( $(nproc) / 2 ))}"
DTYPE="${DTYPE:-float16}"
SKIP_CACHE="${SKIP_CACHE:-0}"
BUDGET_HOURS="${BUDGET_HOURS:-155}"   # 23 run x 6.5 h = 150 h，留 5 h 余量
DRY="${DRY:-0}"
mkdir -p logs
HEART="logs/heartbeat.txt"
ulimit -n 65536 2>/dev/null || echo "[warn] ulimit -n 没抬上去，评估可能报 Too many open files"

stamp() { date '+%F %T'; }
beat()  { echo "$(stamp) | $*" >> "$HEART"; echo "[$(stamp)] $*"; }
run()   { if [ "$DRY" = "1" ]; then echo "[dry] $*"; else "$@"; fi; }

echo "=============================================================="
beat "one-shot 开始  base=$BASE camp=$CAMP tiers=[$TIERS] epochs=$EPOCHS dev_phases=$DEV_PHASES"
echo "=============================================================="

# ---------------------------------------------------------------- 0) campaign 清单
if [ ! -f "campaign/$CAMP/manifest.tsv" ]; then
  beat "生成 campaign 清单（epochs=$EPOCHS）"
  run "$PY" scripts/make_campaign.py --base "$BASE" --campaign "$CAMP" \
      --seeds 0 1 2 --tiers $TIERS --epochs "$EPOCHS"
fi

# ---------------------------------------------------------------- 0.5) 数据
DATA_ROOT=$("$PY" - "$BASE" <<'EOF'
import sys; sys.path.insert(0, ".")
from wcl.utils import load_config
print(load_config(sys.argv[1])["data"]["root"])
EOF
)
if [ ! -d "$DATA_ROOT/ASVspoof2019_LA_train/flac" ] || [ ! -d "$DATA_ROOT/ASVspoof2019_LA_eval/flac" ]; then
  beat "0/6 原始数据不全（$DATA_ROOT），从 HF 镜像拉取（多线程，7.5 GB 约 3 min）"
  run "$PY" scripts/fetch_data.py --config "$BASE" --workers 16 > logs/fetch_data.log 2>&1
  if [ $? -ne 0 ]; then
    beat "[warn] fetch_data 非零退出，看 logs/fetch_data.log；预检会在数据项上报 blocker"
  fi
else
  beat "0/6 原始数据已就位（$DATA_ROOT），跳过拉取"
fi

# ---------------------------------------------------------------- 1) 预检
beat "1/6 预检"
"$PY" scripts/preflight.py --config "$BASE" --tiers "$TIERS" \
    --manifest "campaign/$CAMP/manifest.tsv" --budget-hours "$BUDGET_HOURS" \
    > logs/preflight.log 2>&1
PRE=$?
tail -n 25 logs/preflight.log
if [ $PRE -ne 0 ]; then
  beat "预检有 blocker，停在这里（继续跑只会白烧机时）。详见 logs/preflight.log"
  exit 1
fi

# ---------------------------------------------------------------- 2) 缓存
if [ "$SKIP_CACHE" != "1" ]; then
  beat "2/6 建缓存（workers=$WORKERS dtype=$DTYPE，断点续建）"
  run "$PY" scripts/preprocess.py --config "$BASE" --workers "$WORKERS" --dtype "$DTYPE" \
      > logs/preprocess.log 2>&1
  [ $? -ne 0 ] && beat "[warn] preprocess 非零退出，看 logs/preprocess.log（已存在文件默认跳过）"
else
  beat "2/6 跳过建缓存"
fi

# ---------------------------------------------------------------- 3) 诊断
beat "3/6 诊断（6 项）"
"$PY" scripts/diagnose.py --config "$BASE" > logs/diagnose.log 2>&1
grep -E "^\[OK|^\[FAIL|^\[WARN|blocking|no blocking" logs/diagnose.log | tail -n 8
if grep -q "blocking issue" logs/diagnose.log && grep -q "found --" logs/diagnose.log; then
  beat "[warn] 诊断有不通过项：只有在你确定它是消融配置时才继续"
fi

# ---------------------------------------------------------------- 4) 烟测
# SKIP_SMOKE=1 时跳过：烟测已通过（w_prod 单调降）后重启流水线不必再烧 35 分钟
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
beat "4/6 烟测：wcl 臂 5 轮，只看 w_prod 降不降、显存、步时"
run "$PY" scripts/train.py --config "$BASE" --tag smoke --loss wcl \
    --epochs 5 --eval-every 2 --abort-on-flat --flat-patience 3 \
    --out-dir "outputs/${CAMP}_smoke" > logs/smoke.log 2>&1
if [ "$DRY" != "1" ]; then
  grep -E "^\[train\]|^\[lam\]|^\[guard\]|^\[dev\]" logs/smoke.log | tail -n 8
  SMOKE_RC=$?
  if grep -q "ABORT" logs/smoke.log; then
    beat "[stop] 烟测判定一致性项没被优化（guard ABORT）——先修目标函数，别排 campaign"
    exit 2
  fi
fi
else
  beat "4/6 烟测：SKIP_SMOKE=1，跳过（已完成 outputs/${CAMP}_smoke）"
fi

# ---------------------------------------------------------------- 5) campaign
beat "5/6 campaign：$TIERS（幂等，已完成的 run 自动跳过）"
TIERS="$TIERS" MANIFEST="campaign/$CAMP/manifest.tsv" CAMPAIGN="$CAMP" \
    run bash scripts/run_campaign.sh
CAMP_RC=$?
beat "campaign 结束 rc=$CAMP_RC（单个 run 失败不阻塞后续，详见 logs/*.log）"

# ---------------------------------------------------------------- 6) 汇总
beat "6/6 汇总表格"
run "$PY" scripts/collect_campaign.py --campaign "$CAMP" > logs/collect.log 2>&1
tail -n 30 logs/collect.log || true

beat "全部结束。产物：outputs/$CAMP/（每 run 一份 eval_*.json）、logs/、campaign/$CAMP/"
echo "下一步：把 collect 表贴进论文 §6.3，并按 05_版本历史 的约定留一版。"
