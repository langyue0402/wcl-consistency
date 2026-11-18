# camp_v4 权重本地备份清单

备份时间：2026-11-17 11:30–14:11（机时到期前）
落盘目录：/home/user/wcl_backup_20261117/ckpt/<run>/{best.pt,last.pt}

| run | best.pt | last.pt |
|---|---|---|
| ce__s0 | ✔ | ✔ |
| ce__s1 | ✔ | ✔ |
| ce__s2 | ✔ | ✔ |
| ce__s3 | ✔ | ✔ |
| ce__s4 | ✔ | ✔ |
| ce_cleanonly__s0 | ✔ | ✔ |
| ce_cleanonly__s1 | ✔ | ✔ |
| ce_cleanonly__s2 | ✔ | ✔ |
| tfcl__s0 | ✔ | ✔ |
| tfcl__s1 | ✔ | ✔ |
| tfcl__s2 | ✔ | ✔ |
| tfcl__s3 | ✔ | ✔ |
| tfcl__s4 | ✔ | — |
| wcl__s0 | ✔ | ✔ |
| wcl__s1 | ✔ | ✔ |
| wcl__s2 | ✔ | ✔ |
| wcl__s3 | ✔ | — |
| wcl__s4 | ✔ | — |
| wcl_a0.1__s0 | ✔ | — |
| wcl_a1.0__s0 | ✔ | — |
| wcl_a1.0__s1 | ✔ | — |
| wcl_constprod__s0 | ✔ | — |
| wcl_fonly__s0 | ✔ | — |
| wcl_marginal__s0 | ✔ | — |
| wcl_nogeo__s0 | ✔ | — |
| wcl_tonly__s0 | ✔ | — |

合计 41 个文件、52.1 GB。26 个 best.pt 全部到手；last.pt 到手 15/26，缺 11 个（tfcl__s4、wcl__s3、wcl__s4、wcl_a0.1__s0、wcl_a1.0__s0/s1、wcl_constprod/fonly/marginal/nogeo/tonly__s0）。

未备份且无需备份：`cache/phys_v4` 146 GB（可由 preprocess 重建）、`data` 15 GB（可重新下载）、各 run 的 `final.pt`（与 last.pt 同源序列化）。
代码侧：`code/wcl_code.tgz`（8 处未提交修复 + 脚本 + 7 个 seed yaml）、`code/wcl_worktree.diff`、gitlog、tags。
用途：best.pt 对应论文全部 eval 数字，可直接复现；last.pt 用于将来分离 dev 选轮落差中的 early-stopping 项。
