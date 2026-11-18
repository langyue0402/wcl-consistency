#!/bin/bash
cd /root/wcl || exit 1
export HF_ENDPOINT=https://hf-mirror.com
ulimit -n 65535
# 等 a1.0__s1 训练+eval 全部结束，再补算 cleanonly s2 的 eval
while pgrep -f '_chain_a1s1[.]sh' >/dev/null; do sleep 60; done
echo "[chain2] a1s1 done at $(date)" >> logs/_chain_s2.log
if [ -f outputs/camp_v4/ce_cleanonly__s2/eval_eval.json ]; then echo "[chain2] s2 already has eval, skip" >> logs/_chain_s2.log; exit 0; fi
PYTHONUNBUFFERED=1 python3 _safe_eval.py --config campaign/camp_v4/ce_cleanonly__s2.yaml --ckpt outputs/camp_v4/ce_cleanonly__s2/best.pt --split eval > logs/_safe_eval_s2.log 2>&1
echo "[chain2] s2 eval rc=$? at $(date)" >> logs/_chain_s2.log

