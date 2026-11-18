#!/usr/bin/env bash
# 三臂完整实验（ce / tfcl / wcl），同一配置、同一缓存、同一 schedule。
#
#   bash scripts/run_three_arm.sh                 # 全流程：预处理 -> 三臂训练 -> eval 五阶段 -> 汇总
#   SKIP_PREP=1 bash scripts/run_three_arm.sh     # 缓存已在，跳过预处理
#   ARMS="wcl" bash scripts/run_three_arm.sh      # 只跑一臂
#
# 注意（原机器上踩过的坑，换机照旧）：
#   * 远端只有 python3，没有 python；
#   * eval/preprocess 阶段 num_workers=4 会 "Too many open files"，
#     必须先把 ulimit -n 提到 65536；
#   * 后台跑要 setsid + < /dev/null，否则 ssh 断开被杀。
set -uo pipefail

CFG="${CFG:-configs/three_arm_phys.yaml}"
ARMS="${ARMS:-ce tfcl wcl}"
PY="${PY:-python3}"
SPLITS="${SPLITS:-train dev eval}"

echo "[0/4] 预检（梯度 / AGC / eps / randomize）"
"$PY" scripts/diagnose.py --config "$CFG" || {
  echo "预检未通过，先修问题再花 GPU 时间"; exit 1; }

echo "[1/4] 预处理（AFE 已改为真截断 VAD，缓存必须重建）"
if [ "${SKIP_PREP:-0}" = "1" ]; then
  echo "      跳过（SKIP_PREP=1）"
else
  "$PY" scripts/preprocess.py --config "$CFG" --splits $SPLITS || exit 1
fi

echo "[2/4] 三臂训练"
for arm in $ARMS; do
  echo "      -> arm=$arm"
  "$PY" scripts/train.py --config "$CFG" --tag "$arm" --loss "$arm" || exit 1
done

echo "[3/4] eval 集五阶段评估"
ulimit -n 65536 2>/dev/null || echo "      [warn] ulimit 未提升，可能触发 Too many open files"
# out_dir 从配置里读，避免换了 CFG 之后路径对不上
OUT=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open('$CFG',encoding='utf-8'))['out_dir'])")
for arm in $ARMS; do
  ckpt="$OUT/$arm/best.pt"
  [ -f "$ckpt" ] || { echo "      [skip] $ckpt 不存在"; continue; }
  echo "      -> eval $arm"
  "$PY" scripts/evaluate.py --config "$CFG" --ckpt "$ckpt" --split eval --save-scores || exit 1
done

echo "[4/4] 汇总三臂对比表"
"$PY" scripts/collect_arms.py --root "$OUT" | tee "$OUT/three_arm_table.md"
echo "完成：$OUT/three_arm_table.md"
