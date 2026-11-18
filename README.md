# Wasserstein Consistency Learning (WCL)

Reference implementation and full experimental record for the manuscript
*Wasserstein Consistency Learning: A Single Sinkhorn Transport Term Replaces
the Cosine and CKA Pair in ASVspoof 2019 LA Front-End Robustness*.

The repository contains the training/evaluation code, every run configuration,
the per-run evaluation scores, and the aggregate tables behind the paper.
Trained checkpoints (52 GB) are not included; see `docs/checkpoints_manifest.md`.

## What this code does

Consistency regularisation between a clean and a distorted view is the standard
recipe for front-end-robust anti-spoofing countermeasures, and it is usually
implemented as a hand-weighted sum of a softmax cross-attention cosine term and a
frequency-band CKA term. This code replaces both with a **single Sinkhorn
entropic-optimal-transport term** on the product ground space
`Ω̄ = [0,T'] × [0,F']`, with the cost

```
c(z, z') = ‖v − v'‖₂ + λ_t |t − t'| / (T'−1) + λ_f |f − f'| / (F'−1)
```

and reports three arms trained under one schedule:

| arm | objective |
|---|---|
| `ce` | dual-view cross-entropy, no output-side consistency |
| `tfcl` | cosine (softmax cross-attention) + frequency-band CKA, the existing design |
| `wcl` | the single transport term described above |

## Layout

```
code/               training, evaluation, campaign and diagnostic scripts
code/scripts/       train.py (training), evaluate.py (eval), diagnose.py, preflight.py,
                    run_one_shot.sh (whole campaign), run_campaign.sh, run_three_arm.sh
code/wcl/           model, AFE (AEC/NS/AGC/VAD cascade), losses (incl. the Sinkhorn term),
                    features, dataset
results/            master_table.csv (26 runs × 21 fields), run_metrics.csv,
                    dev_eval_gap.csv, stats_agg.json, DATA_LOG.md
results/per_run/    per-run eval_eval.json, history.json, run_config.json, scores_eval.npz
docs/               checkpoint manifest, git log, tags, worktree diff against the last tag
```

## Environment

Tested on Python 3.10.12, PyTorch 2.5.1+cu124, a single RTX 4090 (24 GB), 32 cores.
`transformers` must be pinned to **4.46.3**; newer versions break the XLS-R loading path.
Set `HF_ENDPOINT=https://hf-mirror.com` when downloading the backbone from mainland China.

```
pip install -r requirements.txt
```

## Reproducing a single run

ASVspoof 2019 LA must be obtained from its official release page. Training reads
pre-computed front-end features from a cache built once:

```bash
export HF_ENDPOINT=https://hf-mirror.com
ulimit -n 65535

python3 scripts/preflight.py --config configs/bench.yaml
python3 scripts/preprocess.py --config configs/bench.yaml --split train
python3 scripts/preprocess.py --config configs/bench.yaml --split dev
python3 scripts/preprocess.py --config configs/bench.yaml --split eval

python3 scripts/train.py --config configs/bench.yaml --loss wcl --seed 0 \
        --tag wcl__s0 --epochs 50 --eval-every 3
python3 scripts/evaluate.py --config configs/bench.yaml --split eval \
        --ckpt outputs/camp_v4/wcl__s0/best.pt --phases clean aec ns agc vad \
        --save-scores
```

`--loss` selects the arm (`ce`, `tfcl`, `wcl`); there is no `--arm` flag.
The whole 26-run campaign runs end to end with

```bash
setsid bash scripts/run_one_shot.sh < /dev/null > logs/one_shot.log 2>&1 &
```

## Notes that cost us time

* `preprocess.py` crashes on multiple splits in one invocation; run one split per call.
* The evaluation stage needs the same `ulimit -n` headroom as preprocessing; without it
  the run dies with *Too many open files*.
* Under `nohup`, background campaigns are killed when the session ends; use `setsid`.
* A checkpoint from a run that died mid-training must never be evaluated and reported:
  the last two seeds of `wcl_a1.0` needed three attempts (two segfaults at a dev-eval
  round boundary) before a full 50-epoch run was obtained.

## Results in one line

Across 26 runs (five seeds per consistency arm under one schedule), the three
objectives are **not distinguishable** on this corpus, backbone and distortion
family: all paired intervals cover zero, and the pairing that the campaign does
resolve is a per-stage one (AEC: WCL better by 0.593 EER points, five seeds out of
five, `p = 0.053`). Turning the geometric term off raises eval EER from 3.94 to
13.41. The transport residual saturates at 0.385–0.401 while the calibrated weight
`λ_eff` spans a factor of 10.7. See `results/master_table.csv`.

## Citation

See `CITATION.cff`.

## License

MIT for the code in `code/`; the documents under `results/` and `docs/` are released
under CC BY 4.0.
