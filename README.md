# genaug-validation

验证课题:**生成式图像增强能否可度量地提升少样本目标检测?**
测试床:CDFSOD(NTIRE)+ 2 个业务场景(待定,预计每类几十张真图)。

## 实验设计

| 层 | 角色 | 检测器 |
|---|---|---|
| 强基线(CDFSOD) | 方法论验证的对照系 | FT-FSOD(MMGDINO Swin-B) |
| 部署基线(业务数据) | 业务落地的实际载体 | ECDet(业务数据到位后启用) |
| 生成侧 | 累积消融:`baseline → +trick1 → +trick1+trick2 → …`,每个 trick 的边际贡献可归因 | 只做生成模型路线(不做 copy-paste),具体 trick 边调研边定 |

协议:官方固定 support split,每格 3 seeds 报 mean±std;增强结果只与同一环境、同一 seed 协议下自跑的无增强 baseline 对比(发表数字仅用于复现校验,不作对照组,避免环境差异混入增强效果)。

## 进展

### ✅ FT-FSOD baseline 复现(CDFSOD Phase 1)· 2026-07-04

MMGDINO-B,官方 1/5/10-shot split,官方训练预算,seeds 42/43/44:

| Domain | Shot | ours (mean±std) | paper | Δ |
|---|---|---|---|---|
| clipart1k | 1 | 56.57 ± 0.29 | 55.6 | +0.97 |
| clipart1k | 5 | 60.10 ± 0.30 | 59.4 | +0.70 |
| clipart1k | 10 | 61.07 ± 0.55 | 59.6 | +1.47 |
| FISH | 1 | 42.37 ± 0.59 | 42.7 | −0.33 |
| FISH | 5 | 44.63 ± 1.26 | 45.5 | −0.87 |
| FISH | 10 | 45.77 ± 0.42 | 46.3 | −0.53 |

6 格均在发表值 ±1.5 mAP 内,复现可信。**此表即后续 clipart1k/FISH 上所有生成增强实验的固定对照组**,不随实验重跑。

Phase 1 先只建这两个域的 baseline:官方 config 各域训练预算差异大(如 DIOR 一个域 3-shot×3-seed 约 136 GPU·h),六域一次建齐性价比低;其余四域(ArTaxOr/DIOR/NEU-DET/UODD)在具体实验需要时按域补建,协议不变。

复现细节(环境、脚本、逐 run 结果与 manifest)见 `baselines/ftfsod_cdfsod/`。

### ⏳ 进行中:生成侧方案收敛

文献综述:`report/related-work.md`(生成增强 × 少样本检测/分类,约 40 篇)。候选路线:前景保留+背景生成(Domain-RAG 系)、少样本 LoRA 域定制生成器(DataDream/FLORA 系)、layout 条件生成(ODGEN 系)。定稿后先在 clipart1k 或 FISH 上与上表对照出第一批增强数字。

### 下一步

1. 第一个生成增强实验 vs Phase-1 baseline
2. 业务数据到位后:ECDet 部署基线 + 业务域上的增强验证
3. 其余 CDFSOD 域按需补 baseline

## 仓库结构与运行

```
baselines/ftfsod_cdfsod/   baseline 跑批脚本、逐 run 结果、复现文档
generation/                合成数据管线(生成模型路线)
experiments/               每个实验一个目录:配置、脚本、结果、结论
report/                    证据报告(related-work、baseline 表)
docs/specs/ docs/plans/    设计文档与实施计划
```

```bash
python3 -m pytest                                                # 单元测试
baselines/ftfsod_cdfsod/run_one.sh clipart1k 1 42 <gpu> <port>   # 单格 baseline
python3 baselines/ftfsod_cdfsod/aggregate_results.py baselines/ftfsod_cdfsod/results
```

环境与数据约定见 `AGENTS.md`。
