#!/bin/bash
cd /root/wcl || exit 1
export HF_ENDPOINT=https://hf-mirror.com
ulimit -n 65535
# 等 cleanonly s2 补算退出，避免抢 GPU
while pgrep -f '_safe_eval[.]py' >/dev/null; do sleep 30; done
echo "[chain] eval free at $(date)" >> logs/_chain_a1s1.log
PYTHONUNBUFFERED=1 python3 scripts/train.py --config campaign/camp_v4/wcl_a1.0__s1.yaml --tag wcl_a1.0__s1 --loss wcl > logs/wcl_a1.0__s1.log 2>&1
echo "[chain] train rc=$? at $(date)" >> logs/_chain_a1s1.log
PYTHONUNBUFFERED=1 python3 _safe_eval.py --config campaign/camp_v4/wcl_a1.0__s1.yaml --ckpt outputs/camp_v4/wcl_a1.0__s1/best.pt --split eval > logs/wcl_a1.0__s1_eval.log 2>&1
echo "[chain] eval rc=$? at $(date)" >> logs/_chain_a1s1.log

