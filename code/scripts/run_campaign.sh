#!/usr/bin/env bash
# 跑整个 campaign（core / seeds / abl / alpha / afe）。幂等、可断点续跑。
#
#   bash scripts/run_campaign.sh                     # 默认 tier=core seeds abl alpha
#   TIERS="core" bash scripts/run_campaign.sh        # 只跑三臂主表
#   DRY=1 bash scripts/run_campaign.sh               # 只打印将要做什么
#   HOLD_OUT=cache/camp_v4_holdout bash scripts/run_campaign.sh   # 额外评 unseen 失真
#
# 每一轮都会：
#   1) 检查该 run 的缓存是否存在（缺失就打印 preprocess 命令并跳过，不浪费 GPU）
#   2) 若 eval_eval.json 已存在 -> 认为完成，跳过
#   3) 若 last.pt 存在 -> --resume 续训；否则从头训
#   4) 训完自动在 eval 上出五阶段表；有 holdout 缓存则再出一份
#
# 换机照旧要注意：
#   * 远端通常只有 python3，没有 python
#   * 句柄上限：eval/preprocess 的 num_workers 一高就 "Too many open files"
#   * ssh 断开要保住进程：setsid bash scripts/run_campaign.sh < /dev/null > campaign.log 2>&1 &
set -uo pipefail

CAMPAIGN="${CAMPAIGN:-camp_v4}"
MANIFEST="${MANIFEST:-campaign/$CAMPAIGN/manifest.tsv}"
TIERS="${TIERS:-core seeds abl alpha}"
PY="${PY:-python3}"
HOLD_OUT="${HOLD_OUT:-}"
ULIMIT="${ULIMIT:-65536}"

[ -f "$MANIFEST" ] || { echo "找不到 $MANIFEST，先跑 scripts/make_campaign.py"; exit 1; }
ulimit -n "$ULIMIT" 2>/dev/null || echo "[warn] ulimit 未提升到 $ULIMIT"
mkdir -p logs

echo "=============================================================="
echo "campaign=$CAMPAIGN  tiers=[$TIERS]  $(date '+%F %T')"
echo "=============================================================="

while IFS=$'\t' read -r name tier arm cfg cache; do
  name="${name%$'\r'}"; tier="${tier%$'\r'}"; arm="${arm%$'\r'}"
  cfg="${cfg%$'\r'}";  cache="${cache%$'\r'}"
  [ -z "$name" ] && continue
  [ "$name" = "name" ] && continue
  keep=0
  for t in $TIERS; do [ "$t" = "$tier" ] && keep=1; done
  [ $keep -eq 1 ] || continue

  out=$(awk '/^out_dir:/{print $2}' "$cfg" | tr -d "\r")
  run_dir="$out/$name"
  log="logs/$name.log"

  if [ ! -f "$cache/train/meta.json" ] || [ ! -f "$cache/eval/meta.json" ]; then
    echo "[skip] $name：缓存 $cache 不完整（缺 train 或 eval 的 meta.json）。先跑一次："
    echo "       $PY scripts/preprocess.py --config $cfg --splits train dev eval --workers 16"
    echo "       （同一缓存下的所有 run 共用这一份，只需跑一次）"
    continue
  fi
  if [ -f "$run_dir/eval_eval.json" ]; then
    echo "[done] $name 已有 eval_eval.json，跳过"
    continue
  fi
  if [ "${DRY:-0}" = "1" ]; then
    echo "[dry ] $name (tier=$tier arm=$arm) -> $log"
    continue
  fi

  echo "---- $(date '+%T') $name (tier=$tier arm=$arm) -> $log"
  if [ -f "$run_dir/last.pt" ]; then
    "$PY" scripts/train.py --config "$cfg" --tag "$name" --loss "$arm" \
        --resume "$run_dir/last.pt" >>"$log" 2>&1 || {
      echo "[fail] $name 训练失败，见 $log"; continue; }
  else
    "$PY" scripts/train.py --config "$cfg" --tag "$name" --loss "$arm" \
        >>"$log" 2>&1 || {
      echo "[fail] $name 训练失败，见 $log"; continue; }
  fi

  ckpt="$run_dir/best.pt"
  [ -f "$ckpt" ] || ckpt="$run_dir/final.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[fail] $name 没有 checkpoint，见 $log"; continue
  fi
  echo "       eval on eval split"
  "$PY" scripts/evaluate.py --config "$cfg" --ckpt "$ckpt" --split eval \
      --save-scores >>"$log" 2>&1 || echo "[warn] $name eval 失败"

  if [ -n "$HOLD_OUT" ] && [ -f "$HOLD_OUT/eval/meta.json" ]; then
    echo "       eval on held-out distortion draw ($HOLD_OUT)"
    "$PY" scripts/evaluate.py --config "$cfg" --ckpt "$ckpt" --split eval \
        --cache-dir "$HOLD_OUT" --out-suffix _holdout >>"$log" 2>&1 \
        || echo "[warn] holdout eval 失败"
  fi
done < "$MANIFEST"

echo "=============================================================="
echo "$(date '+%T') 汇总："
"$PY" scripts/collect_campaign.py --root "${OUT_ROOT:-outputs/$CAMPAIGN}" || true
