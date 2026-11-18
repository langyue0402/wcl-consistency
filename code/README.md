# WCL — Wasserstein Consistency Learning for Robust Speech Deepfake Detection

用 Python 实现的「最优传输统一时频一致性」（WCL）完整训练/评估/推理工程，
对应论文 *Wasserstein Consistency Learning: An Optimal-Transport View of
Time–Frequency Invariance for Robust Speech Deepfake Detection* 的方法与实验设计。

> 说明：本工程是论文方法（WCL 目标 + 双视图训练 + Sinkhorn 时/频 Wasserstein
> 一致性约束）的自包含实现。工程提供**两套编码器**：
>
> | `model.arch` | 编码器 | 输入 | 用途 |
> |---|---|---|---|
> | `xlsr_aasist` | **XLS-R (300M) + AASIST 后端** | raw 波形 4s | **与 TFCL 同架构，做同场对比**（推荐） |
> | `cnn` | 轻量 2D-CNN | log-mel 谱图 | 快速冒烟、低资源环境 |
>
> 若要复现/对比 TFCL 论文的 "Ours" 行（0.55→9.78 EER），必须用 `xlsr_aasist`；
> 轻量 CNN 的数字与其不可比。

---

## 0. 架构对齐说明（xlsr_aasist）

TFCL (Xue et al., ACM MM '26) 的架构为：

- **前端**：XLS-R（`facebook/wav2vec2-xls-r-300m`），输出帧级表示 `F ∈ R^{T×1024}`；
- **后端**：AASIST（图注意力分类器）；
- **训练**：Adam，lr 1e-4，wd 1e-4，100 epoch，early stopping（patience 10），λ=0.3。

本工程 `wcl/model_xlsr_aasist.py` 复现该骨干，使 **WCL 与 TFCL 的差异只剩一致性损失本身**：

| 环节 | TFCL | 本工程 (WCL) |
|---|---|---|
| 前端 | XLS-R 300M | XLS-R 300M（同权重） |
| 后端 | AASIST（HS-GAL 图注意力） | AASIST 风格 HS-GAL（self-attention + FFN + 残差） |
| 帧级特征 | 1024 维 | 1024 维 |
| 频域分组 | 1024 → 16 段 | 1024 → 16 段（同切分） |
| 一致性损失 | 余弦 + CKA | **熵正则 Wasserstein (Sinkhorn)** |
| λ | 0.3 | 0.3 |

**与官方代码的差异（明示）**：AASIST 官方使用 `BatchNorm1d`，这里用 `LayerNorm`
（变长帧序列 + 小 batch 更稳定）；未实现官方 AASIST 的 SincConv 前端，因为前端
已被 XLS-R 替代。除此之外结构等价。

**限制**：XLS-R 的自监督表示没有显式频率轴，"频域"指 1024 维通道按 16 段分组
（与 TFCL 的 CKA 分组一致），不是物理 mel 频带。

## 1. 安装依赖

```bash
# 建议 Python 3.10+，先装 PyTorch（服务器为 CUDA 版）：
#   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# xlsr_aasist 架构还需要：
pip install "transformers>=4.30"
```

## 2. 拿数据：HF 镜像一条命令（推荐）

官方站点（datashare.ed.ac.uk）在国内速度通常只有几百 KB/s，一个整包 20+ GB 要挂几小时。
镜像仓库 `Bisher/ASVspoof_2019_LA` 存的是**同样的 flac 原始字节**（放在 parquet 的
`audio.bytes` 里，魔数 `fLaC`），`scripts/fetch_data.py` 会把它们原样写回官方目录结构，
不解码、不重编码：

```bash
python scripts/fetch_data.py --config configs/three_arm_phys.yaml --workers 16
```

实测（国内家庭宽带 + 8 线程，2026-11-09）：

| 项 | 数值 |
|---|---|
| 3 个 parquet 分片 | 7.55 GB（train 1.48 / dev 1.47 / eval 4.38） |
| 下载耗时 | **≈ 32 s / 1.5 GB ≈ 50 MB/s**（8 线程；16 线程更快） |
| 解出 flac | 130,461 个文件（25,380 + 24,844 + 71,237），dev 实测 24,844 个 ≈ 9 s |
| 全量数据准备 | **≈ 3–5 min**，对比之前在服务器上拉镜像的数小时 |

产物与官方布局完全一致，`scripts/preprocess.py` 直接可用：

```
data/asvspoof2019_la/
├── ASVspoof2019_LA_train/flac/LA_T_*.flac
├── ASVspoof2019_LA_dev/flac/LA_D_*.flac
├── ASVspoof2019_LA_eval/flac/LA_E_*.flac
├── ASVspoof2019_LA_cm_protocols/LA.cm.{train.trn,dev.trl,eval.trl}.txt
└── _fetch_manifest.json          # 分片 sha256、条数、来源
```

脚本保证的三件事：**逐分片校验 LFS sha256**（长度对但内容错的续传占位文件会被识别并重下）、
**条数核对**（与官方 25,380 / 24,844 / 71,237 比对，不符会 warn）、**flac 魔数检查**。
`--limit 200` 可先小跑一遍；`--clean` 解完删掉 7.5 GB parquet；`--extract-only` 只重解不重下。

> 注意：镜像的行数不保证等于官方公布数。若 eval 少于 71,237，**不要**把它写成"full eval set"，
> 论文里报 eval 数字时先核对 `_fetch_manifest.json` 与 `meta.json` 的条数。

官方渠道作为备选：在 [datashare](https://datashare.ed.ac.uk/handle/10283/3336) 免费申请后
下载解压，按上面同一结构放置即可，脚本不需要改。

## 3. 三步跑通：预处理 → 训练 → 评估/推理

### 3.1 预处理（提取各 AFE 阶段特征缓存）

```bash
# 【推荐】TFCL 同架构：raw 波形缓存（4s @16k，XLS-R 输入）
python scripts/preprocess.py --config configs/tfcl_arch.yaml --splits train dev eval

# 轻量 CNN：log-mel 缓存
python scripts/preprocess.py --config configs/train_full.yaml --splits train dev eval

# 快速冒烟（每 split 只取前 200 条，几分钟）
python scripts/preprocess.py --config configs/smoke.yaml --splits train dev --limit 200
```

缓存写入 `cache/{smoke|full|tfcl_arch}/{split}/{clean,aec,ns,agc,vad}/{utt}.npy` 与
`meta.json`。训练/评估都从缓存读取，不再重复提特征。

### 3.2 训练

```bash
python scripts/train.py --config configs/smoke.yaml            # 轻量 CNN 快速验证
python scripts/train.py --config configs/tfcl_arch_smoke.yaml --tag gpu_smoke  # GPU 冒烟（3 epoch，测显存/速度）
python scripts/train.py --config configs/tfcl_arch.yaml --tag run1              # TFCL 同架构全量训练
python scripts/train.py --config configs/train_full.yaml --tag run1_cnn         # CNN 全量
```

- 每 epoch 在 dev 集做**五阶段评估**（clean→AEC→NS→AGC→VAD 前缀），
  以五阶段平均 EER 为早停指标；最优模型存到 `outputs/{tag}/best.pt`，末轮存 `final.pt`。
- 目标函数即论文式 (5)：`L = CE(clean) + CE(distorted) + λ(W_t + W_f)`，
  其中 `W_t`/`W_f` 为 Sinkhorn 熵正则 1-Wasserstein 距离（k=20 次迭代，
  ε 从 0.1 线性退火到 0.01；λ 是唯一需调的超参——论文命题 4 的缩放恒等式）。
- `tfcl_arch.yaml` 采用 batch 8 + 梯度累积 4（等效 batch 32）、bf16 混合精度、
  冻结 XLS-R 前端，以适配单张 24 GB GPU；显存更小就减小 batch、增大 grad_accum。

### 3.3 评估（按 AFE 阶段出 EER/AUC 表）

```bash
python scripts/evaluate.py --config configs/smoke.yaml \
    --ckpt outputs/smoke/default/best.pt --split dev
```

输出形如：

```
phase   EER(%)      AUC   stage
clean     x.xx   0.9xxx
aec       x.xx   0.9xxx     AEC
ns        x.xx   0.9xxx     AEC+NS
agc       x.xx   0.9xxx     AEC+NS+AGC
vad       x.xx   0.9xxx     AEC+NS+AGC+VAD
```

同时打印 dev/vad 上的最优判定阈值（推理用）。

### 3.4 推理（单文件或目录打分）

```bash
# 干净条件下打分
python scripts/infer.py --config configs/smoke.yaml \
    --ckpt outputs/smoke/default/best.pt --input some_audio.flac

# 模拟经过完整 AFE 管线后再打分（复现鲁棒性设置）
python scripts/infer.py --config configs/smoke.yaml \
    --ckpt outputs/smoke/default/best.pt --input /path/to/dir --phase vad --threshold 0.5
```

输出每文件的 bonafide 概率（score>阈值判为真实）与判定结果。

---

## 4. 配置说明（configs/*.yaml）

| 字段 | 含义 |
|---|---|
| `feature.type` | `raw`（波形，XLS-R 输入）或 `logmel`（谱图，CNN 输入） |
| `feature.duration` | raw 模式下的固定时长（4.0s = 64000 samples） |
| `feature.max_frames` | log-mel 固定帧长（640 = 6.4s @10ms；smoke 用 320） |
| `model.arch` | `xlsr_aasist`（TFCL 同架构）或 `cnn`（轻量） |
| `model.frontend.freeze` | 是否冻结 XLS-R（默认 true；微调设 false 并调小 lr） |
| `train.view_schedule` | `random_stage`：每 batch 随机选一个 AFE 阶段作失真视图（推荐）；`full_cascade`：固定全管线（对齐论文协议） |
| `train.grad_accum` | 梯度累积步数（显存不足时增大） |
| `wcl.lambda_mode` | `abs`：直接用 `lambda`（与 TFCL 同口径）；`ratio`：`λ_eff = α·CE_ref/W_ref`（无量纲，跨骨干可迁移） |
| `wcl.alpha` | 仅 `ratio` 模式：一致性项初始占分类损失的比例 |
| `wcl.lambda / eps_init / eps_final / sinkhorn_iters` | 绝对权重、熵正则退火、Sinkhorn 迭代次数 |
| `afe.randomize` | 训练时对 AEC/AGC 等参数加随机扰动（把失真从单点扩展为失真族） |
| `eval.phases` | 阶段前缀评估序列 |

## 5. 与论文的对应关系

| 论文内容 | 本实现 |
|---|---|
| 双视图一致性训练（5.1） | `WCLTrainDataset` + `wcl_loss` |
| Sinkhorn 近似 W_{1,ε}（5.2） | `wcl/losses.py` log-domain Sinkhorn，k=20 |
| 时/频方向经验测度 μ^t、μ^f（5.1） | 特征图沿频率/时间池化后的帧级测度（`wcl_loss`） |
| ε 退火（5.4） | `annealed_eps`，0.1→0.01 线性 |
| 命题 4（λ/ε 缩放恒等式） | λ 为唯一超参，ε 随 epoch 退火；**Remark 1** 的 λ 量纲分析由 `resolve_lambda` 落地 |
| AFE 四阶段（AEC/NS/AGC/VAD） | `wcl/afe.py` 纯 Python 模拟（物理类型对齐，非 WebRTC） |
| 表 1 阶段前缀评估 | `evaluate.py` 五阶段 EER/AUC |

### 5.1 关于一致性权重 λ（论文 Remark 1）

λ 不是无量纲常数：它乘的是"质量×特征距离"的传输代价，平衡的量是 nats 的交叉熵，
所以 λ 的量纲是 nats/(特征单位·质量)，同一数值换一个特征尺度就代表不同的约束强度。
因此 `tfcl_arch.yaml` 默认用 `lambda_mode: ratio`：

```
λ_eff = α · CE_ref / W_ref      # CE_ref、W_ref 为初始若干 batch 的平均值
```

α 的含义是"训练开始时一致性项占分类损失的百分比"，跨骨干可迁移。启动日志会打印
标定结果，例如 `lambda_mode=ratio alpha=0.3 -> lambda_eff=0.6503`。做与 TFCL 的
严格对照实验时，可切到 `lambda_mode: abs` 并固定 `lambda: 0.3`（同口径），两种都跑
一遍即可验证 α 口径的稳定性。

## 6. 无数据冒烟（拿到 ASVspoof 之前先验环境）

工程内置一个**合成替身数据集**生成器，目录/协议结构与 ASVspoof 2019 LA 完全一致，
可用于在真实数据到位前跑通全链路：

```bash
# 1) 生成合成数据（默认 data/synth_asvspoof，train/dev 各 40 条）
python scripts/make_synthetic_data.py --train 120 --dev 60

# 2) 预处理（用 --root 指向合成数据）
python scripts/preprocess.py --config configs/smoke.yaml --root data/synth_asvspoof --splits train dev

# 3) 训练
python scripts/train.py --config configs/smoke.yaml --root data/synth_asvspoof --tag smoke

# 4) 评估 + 推理
python scripts/evaluate.py --config configs/smoke.yaml --root data/synth_asvspoof \
    --ckpt outputs/smoke/smoke/best.pt --split dev
python scripts/infer.py --config configs/smoke.yaml --root data/synth_asvspoof \
    --ckpt outputs/smoke/smoke/best.pt --input data/synth_asvspoof/ASVspoof2019_LA_train/flac --phase clean
```

> 合成数据的 EER/AUC 数值没有科学意义（spoof 特征被人为放大），只用于验证
> 「数据→特征→训练→评估→推理」整条链路可用。正式结论必须用 ASVspoof 2019 LA。

## 7. 快速自检

```bash
python tests/test_wcl.py     # 无需数据：Sinkhorn 数值、损失、AFE、特征、模型结构
```

## 8. 已知限制

- 简化 AFE（非 WebRTC）⇒ 失真类型对齐但幅度不可比，最终数字与 TFCL 论文仍有管线差异；
- XLS-R 的"频域"是 1024 维通道的 16 段分组，不是物理 mel 频带（与 TFCL 的 CKA 分组一致）；
- AASIST 后端用 LayerNorm 替代官方 BatchNorm1d（结构等价，非逐行复刻）；
- log-mel 固定帧长截断/补零，超长音频会丢尾部（可改 `max_frames`）；raw 模式固定 4s；
- 尚未实现"三档训练目标"（纯 CE / TFCL 式余弦+CKA / WCL）的开关，做对比实验前需补
  `--loss ce|tfcl|wcl`。

## 三臂对照实验（v2·几何对齐）

第一次 GPU 跑（`configs/tfcl_arch.yaml`，VAD 原位静零 + 纯特征代价矩阵）只测了
「按当前实现跑的 WCL」。v2 把两处理论前提补齐后再做 ce / tfcl / wcl 三臂对比：

| 文件 | 作用 |
|---|---|
| `configs/three_arm_phys.yaml` | 正式配置：`afe.vad.mode=truncate` + `wcl.cost_mode=physical` |
| `configs/three_arm_smoke.yaml` | CPU 可跑的微型配置（随机初始化小 XLS-R），验证三臂链路 |
| `scripts/check_afe_geometry.py` | 开跑前自检：VAD 真删了多少、支撑位移多少（<2% 会告警） |
| `scripts/run_three_arm.sh` | 预处理 → 三臂训练 → eval 五阶段 → 汇总表 一条龙 |
| `scripts/collect_arms.py` | 汇总 `outputs/*/` 的 `eval_{split}.json` 成对比表 + VAD 边际倍数 |

```bash
python3 scripts/check_afe_geometry.py --config configs/three_arm_phys.yaml --split dev --n 40
SKIP_PREP=1 bash scripts/run_three_arm.sh          # 缓存已在时
```

`wcl.losses.ground_cost` 的 `cost_mode="euclidean"` 与 `afe.vad.mode="zero"` 保留旧行为，
`configs/tfcl_arch.yaml` 已显式钉住 `mode: zero`，第一次跑因此仍可逐位复现。

单测：`python3 -m pytest tests -q`（24 项，含截断 VAD、物理代价矩阵、cost 分量）。

---

## v3：把「一致性项其实没在训练」修掉

第一次 GPU 跑（论文 §6 的单臂结果）有一个致命前提问题：`feat` 取自**冻结的 XLS-R**，
而 `lam*(W_t+W_f)` 由此**对所有可训练参数的梯度恒为 0**，训练等价于双视图 CE。
论文 §6.2 观察到的「two consistency loadings saturate early」（w_t 稳定在 0.326–0.329）
正是这个 bug 的指纹，不是「一致性项饱和」。

v3 的改动（配置全部已更新，**旧缓存必须重建**，`cache_dir` 已改名到 `*_v3`）：

| 改动 | 位置 | 说明 |
|---|---|---|
| `model.measure_from: backend` | `wcl/model_xlsr_aasist.py` | 测度网格建在 **AASIST 帧输出**上（可训练、且是分类器真正池化的张量）。参数仍为 0.69 M，`hidden=128 / n_subbands=8` → 8 带 × 16 通道 |
| `model.measure_from: frontend` | 同上 | 保留旧行为**仅用于复现 run 1**；`train.py` 检测到「frontend + freeze」会打 WARN |
| `wcl.cost_normalize: mean` | `wcl/losses.py` | 代价除以自身均值，`mean(C)/eps` 落到 [2,10]；否则 140→1400 会让 Sinkhorn 退化成硬指派 |
| `wcl.eps_init: 0.5 / eps_final: 0.1` | configs | 归一化后的 ε 才有「温度」含义 |
| `afe.randomize`（真实现）+ `afe.jitter` | `wcl/afe.py` | 每段参数在 ±40% 内抖动；原来只在 docstring 里出现过 |
| 逐 utterance 种子 | `preprocess.py` / `infer.py` | 原来每条音频都用 `seed=0`，抖动退化成常数 |
| AGC 增益曲线 | `wcl/afe.py` | `randomize` 时自动给 `fluctuation∈[0.15,0.5]`；常数增益会被逐句归一化完全抵消（这正是表 1 里 AEC+NS 与 AEC+NS+AGC 两列完全相同的原因） |
| TFCL 臂 | `wcl/model.py` `scripts/train.py` | 修 `out_dim`（CNN 骨干原来会直接崩）；batch < 8 时 CKA 在 batch 维退化会告警 |
| 一致性损失强制 fp32 | `scripts/train.py` | bf16 下 cost 的 0.4% 误差会被 1/ε 放大 |

### 开跑前先自检

```bash
python scripts/diagnose.py --config configs/three_arm_phys.yaml   # 4 项检查，FAIL 就不要开训
python -m pytest tests -q                                          # 29 项（含梯度/AGC 回归测试）
```

`diagnose.py` 检查：①一致性项是否真的产生梯度；②`agc` 与 `ns` 两个阶段是否还是同一个输入；
③`mean(C)/eps` 是否把 Sinkhorn 压成硬指派；④`afe.randomize` 是否真的被消费。

### 留出（unseen）失真缓存

`randomize` 打开后，`--afe-seed` 决定「这一份」失真族样本。训练/评测用的是一份固定的抽样，
**要支撑「不变性」的主张必须另建一份**：

```bash
python scripts/preprocess.py --config configs/three_arm_phys.yaml --splits eval \
    --afe-seed 7 --cache-dir cache/phys_v3_holdout
```

### 缓存体积（评估前务必确认磁盘）

raw 4 s @16 kHz = 250 KiB/文件，5 阶段 × 全量 LA = **约 156 GB / 607,305 个文件**
（train 32.5 + dev 31.8 + eval 91.2 GB）。磁盘不够时的三个选项：
① 只缓存 `clean`，失真在 `Dataset.__getitem__` 里实时算；
② 存 fp16（体积减半）；
③ eval 只为最终出表跑一次，训练期不预缓存。

### 复现并核对首跑（v1）

`configs/legacy_run1.yaml` 把首跑的每一项口径都钉回原状：测度在冻结 XLS-R 前端（`measure_from: frontend`）、
`n_subbands: 16`、不归一化代价（`cost_normalize: none`）、`eps 0.1 -> 0.01`、五阶段等概率、`vad.mode: zero`、
关闭抖动。用它跑诊断会得到首跑的四个缺陷指纹，且数值与事后审计一致：

| 检查 | 实测（legacy_run1） | 含义 |
|---|---|---|
| 梯度 | `max\|dGrad\| = 0.0` | 一致性项是常数，等价于纯双视图 CE |
| AGC | `rel.diff = 6.7e-07` | `agc` 与 `ns` 是同一份输入，表 1 两列同数 |
| 代价 | `mean(C)=13.8`，`cost/eps = 138 -> 1381` | Sinkhorn 塌成硬指派，ε 退火无温度含义 |

```bash
python scripts/diagnose.py --config configs/legacy_run1.yaml   # 预期 4 个 FAIL
```

`three_arm_phys.yaml` 已改为修正后的口径，诊断应输出 `no blocking issue found`。

---

# v4：目标函数、几何权重、跑批

v3 修掉了「一致性项没有梯度」。v4 修的是修完之后**仍然对不齐**的三件事，每一件都有实测数字。

## 4.1 目标函数：两个边缘传输 ≠ 乘积空间传输

论文 §3.1 把测度定义在 Ω̄ = [0,T]×[0,F]×R^C 上，代价是三段之和；§4.2 的定理 2 界的是
这个空间上的 W₁。但 §5.1 的目标函数写的是 `W_t(时间边缘) + W_f(频轴边缘)`，而**两个一维
边缘传输之和控制不了联合测度**：一个对角团和一个反对角团有完全相同的时间剖面和频轴剖面。

实测（`tests/test_wcl.py::test_product_cost_is_blind_spot_free`）：

| 量 | 对角团 → 反对角团 |
|---|---|
| `W_t + W_f` 相对自身下界的超额 | **0.0000**（恰好停在温度下限，什么也没看见） |
| 乘积空间 `W_1` 超额 | 5.8e-03（非零） |

所以 v3 的目标即便有了梯度，也仍然**不是定理 2 界住的那个量**。v4 改成在 T'×F' 个格点上
做一次传输：

```
C[(t,f),(t',f')] = ||v_{t,f} − v'_{t',f'}|| + λt·|t−t'|/(T−1) + λf·|f−f'|/(F−1)
```

`wcl.cost_mode: product`（默认）。旧的 `marginal` 保留为消融项——它是最接近被替换方法的
形式，而且正好用来证明「边缘形式不满足定理前提」。

## 4.2 几何权重：λt=1 时 |i−j| 仍然是隐形的

论文默认 `λt = 1`，含义是「跨越整段语音花 1 个特征距离」。T=200 时每帧 0.005，而 VAD 的
80–200 ms 位移是 4–10 帧：几何项给代价贡献 0.02–0.05，特征项是 O(1)。**OT 的匹配仍然由
外观决定**，`|i−j|` 只是给损失加了个常数——和首跑是同一个失效模式，小一个数量级。

`wcl.geo_mode: calibrate` 在初始 batch 上按「AFE 实际施加的位移」定价：

```
λt = βt · mean(特征距离) · (T−1) / (位移的格数)
```

位移格数由 `afe.vad.extra_trim_s` 与抖动直接算出（`wcl.utils.afe_trim_cells`），
100 ms ≈ 100/20 = 5 帧。实测（`diagnose.py` 第 5 项）：

```
λt = 1（论文默认）    → 100 ms 位移 = 0.025，占一个典型特征距离的 0.5%
λt = 136（标定后）    → 3.43，占 71%
```

λt 从此不是手填常数，而是 βt 这个无量纲比例——和 Remark 1 对 λ 的处理同一套逻辑。

## 4.3 命题 4 的恒等式方向错了（这一条要改论文）

草稿写的是 `λ·W_{1,ε} = λ·W_{1,ε/λ}`。实测（`python - <<'PY'` 见 `wcl_review_v4.md`）：

| λ | ε | `λ·W_{1,ε}` | `λ·W_{1,ε/λ}` |
|---|---|---|---|
| 2 | 0.1 | 1.302129 | 1.150280 |
| 4 | 0.05 | 2.300561 | 2.019903 |

不相等。正确恒等式是 **`λ·W_{1,ε}(c) = W_{1,λε}(λc) = λε·W_{1,1}(c/ε)`**（三项均精确吻合）。
后果：固定代价时 `(λ, ε)` 只以 `(c/ε, λε)` 进入——ε 是温度，**λε 才是强度**，不是 λ/ε。
所以「固定 λ 退火 ε」会把一致性强度再削弱 `ε_init/ε_final = 5` 倍。

`wcl.lambda_couple: const_product` 抵消这一点（纯温度退火），是消融项之一。

## 4.4 其它改动

| 改动 | 位置 | 为什么 |
|---|---|---|
| 三臂的 λ 都用 α 标定 | `train.py::calibrate_lambda` | tfcl 臂原来用参考文献的 0.3，而 WCL 臂用标定出的 ~0.95；基线被系统性削弱 |
| `wcl.tfcl_cka_dim: cell` | `losses.py` | CKA 在 batch 维估计时 B=8 只有 7 个自由度（两个独立随机矩阵 CKA≈0.89）；改在 T'×F' 格点上估计，基线才有力气 |
| `grid_stride` | `losses.py` | 1600 格点的 Sinkhorn 前+反向，CPU 上 9.3 s / 2 = 1.86 s / 4 = 0.48 s；默认 2 |
| `preprocess.py --workers` | 新 | 全量 LA 串行 5 阶段约 14 h，进程池后 1–2 h（都是计费的 CPU 时间） |
| `--seed/--cache-dir/--out-dir/--eval-every/--dev-limit/--no-eval` | `train.py` | 多种子、holdout、成本控制原来都要改 YAML |
| 每 epoch 写 `history.json` + `last.pt` | `train.py` | 原来只在训练结束写一次，11 h 的 run 崩一次就全丢 |
| `evaluate.py --cache-dir/--limit/--out-suffix` | 新 | 才能评 unseen 失真缓存，且不覆盖同家族的那份 |
| `train.dev_phases` | 新 | 只用来选模型时可只评 clean+vad，最终表仍走五阶段 |

## 4.5 在租来的机器上怎么跑

```bash
# 0) 环境
nvidia-smi && python3 -V && pip install -r requirements.txt
ulimit -n 65536                       # 否则 eval/preprocess 会 Too many open files

# 1) 数据（ASVspoof 2019 LA，约 23 GB）+ 预检（分钟级，先跑，不要跳）
export DATA_ROOT=/root/autodl-tmp/asvspoof2019_la
python3 scripts/diagnose.py --config configs/three_arm_phys.yaml
python3 -m pytest tests -q            # 33 项

# 2) 特征缓存（CPU，16 进程，约 1–2 h，156 GB）
python3 scripts/preprocess.py --config configs/three_arm_phys.yaml \
        --splits train dev eval --workers 16

# 3) 生成 campaign（22 个 run 的清单）
python3 scripts/make_campaign.py --base configs/three_arm_phys.yaml --seeds 0 1 2

# 4) 先只跑一个 run，确认 w_t / w_prod 真的在降、显存和步时符合预期
DRY=1 bash scripts/run_campaign.sh
TIERS="core" bash scripts/run_campaign.sh   # 断线用：setsid ... < /dev/null &

# 5) 汇总
python3 scripts/collect_campaign.py --root outputs/camp_v4
```

**盘**：五阶段全量缓存 156 GB（train 32.5 + dev 31.8 + eval 91.2）。租机器时按 ≥200 GB 数据盘申请；
不够就 `--dtype float16`（78 GB）或先把 eval 之外的两份建好、eval 最后单独建。

**时**（单卡 4090，按首跑实测 331 s/epoch 优化 + 11 min 五阶段 dev 推）：

| 配置 | s/epoch | 40 epoch/run | 22 个 run |
|---|---|---|---|
| 原始（eval_every=1，stride 1） | 991 | 11.0 h | 242 h |
| v4 默认（eval_every=3，stride 2） | ~600 | 6.7 h | ~147 h |
| `dev_phases: [clean, vad]` + eval_every 3 | ~430 | 4.8 h | ~106 h |

后两行是同一批实验的两种成本口径；论文里要写清楚用的是哪一个。

## 4.6 campaign 的层级

`scripts/make_campaign.py` 生成 `campaign/camp_v4/manifest.tsv`，按重要性分层：

- `core`：ce / tfcl / wcl × 3 seed = 9 个 run，是表 1 的全部内容
- `abl`：wcl 关掉一个部件 —— `marginal`（§5.1 原形式）、`nogeo`（β=0）、`lam1`（v3 口径）、
  `frontend`（首跑缺陷，作为对照）、`tonly` / `fonly`、`constprod`、`stride1` / `stride4`、
  `tfcl_ckabatch`（参考文献的 CKA 口径）
- `alpha`：α ∈ {0.1, 1.0}（0.3 已在 core）
- `afe`：`vadzero` / `nojitter`，**各需一份自己的缓存**，runner 发现缓存缺失会跳过并打印命令

跑批是幂等的：有 `eval_eval.json` 就跳过，有 `last.pt` 就 `--resume` 续训，所以 ssh 掉了直接重跑同一条命令。
