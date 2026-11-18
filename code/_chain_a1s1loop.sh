#!/bin/bash
cd /root/wcl || exit 1
export HF_ENDPOINT=https://hf-mirror.com
ulimit -n 65535
attempt=1
while [ $attempt -le 4 ]; do
  echo "[loop] attempt $attempt start $(date)" >> logs/_chain_a1s1loop.log
  PYTHONUNBUFFERED=1 python3 scripts/train.py --config campaign/camp_v4/wcl_a1.0__s1.yaml --tag wcl_a1.0__s1 --loss wcl --resume outputs/camp_v4/wcl_a1.0__s1/last.pt >> logs/wcl_a1.0__s1_resume.log 2>&1
  rc=$?
  echo "[loop] attempt $attempt train rc=$rc at $(date)" >> logs/_chain_a1s1loop.log
  if [ $rc -eq 0 ]; then break; fi
  ep=$(python3 - <<'PY2'
import torch
c=torch.load('/root/wcl/outputs/camp_v4/wcl_a1.0__s1/last.pt',map_location='cpu',weights_only=False)
print(int(c.get('epoch',0)))
PY2
)
  echo "[loop] last.pt epoch=$ep" >> logs/_chain_a1s1loop.log
  if [ "$ep" -ge 50 ]; then echo "[loop] reached ep50, stop retry" >> logs/_chain_a1s1loop.log; break; fi
  attempt=$((attempt+1))
  sleep 10
done
PYTHONUNBUFFERED=1 python3 _safe_eval.py --config campaign/camp_v4/wcl_a1.0__s1.yaml --ckpt outputs/camp_v4/wcl_a1.0__s1/best.pt --split eval > logs/wcl_a1.0__s1_eval.log 2>&1
echo "[loop] final eval rc=$? at $(date)" >> logs/_chain_a1s1loop.log

