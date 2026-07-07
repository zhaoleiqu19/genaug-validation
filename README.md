# genaug-validation

验证课题:**生成式图像增强能否可度量地提升少样本目标检测?**
测试床:CDFSOD(NTIRE)+ 2 个业务场景(待定,预计每类几十张真图)。

## 实验设计

| 层 | 角色 | 检测器 |
|---|---|---|
| CDFSOD  | 方法验证的对照 | FT-FSOD(MMGDINO Swin-B) |
| 业务数据 | 落地的实际载体 | ECDet(业务数据到位后启用) |
| 生成侧 | 累积消融:`baseline → +different trick/method → …`,每个 trick 的边际贡献可归因 | 只做生成模型路线(不做 copy-paste),具体 method 边调研边定 |

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

###  进行中:生成侧方案收敛

文献综述:`report/related-work.md`。检测上四条候选路线,按对底层生成模型的要求分类:

| 路线 | 代表工作 | 对底层模型的要求 |
|---|---|---|
| 真图编辑图img2img(rung-0 在跑) | DA-Fusion | 需要平滑的 strength 曲线,不能是步数蒸馏模型 |
| 前景保留 + 背景生成 | Domain-RAG | 只需 T2I 出图质量,不涉及 strength 平滑度 |
| 少样本 LoRA 域定制 | DataDream / FLORA / LoFT | 需要成熟的 LoRA 微调工具链 |
| layout 条件生成 | ODGEN / GeoDiffusion / AeroGen | 绑定发布代码自带的底座,不是自由换的 |

前三条可共用同一生成器;layout 条件生成底座由具体复现代码决定,不纳入本轮筛选。

生成器筛选(可视化验证已完成):FLUX.2-klein-4B(步数蒸馏)strength 曲线断崖式——0.3–0.8 近乎 no-op,0.85 瞬间跳变到完全不同构图、框失效,不满足平滑控制的前提;FLUX.1-dev(guidance-蒸馏、非步数蒸馏)同批测试曲线平滑,**确定为 rung-0 生成器**,权重与环境已就绪。具体强度取值待剩余 support 图验证后定稿。

### 下一步

1. 强度定稿:剩余 FISH support 图 + clipart1k 补测,写入 rung-0 spec
2. 第一个生成增强实验(rung-0 img2img,FLUX.1-dev)vs Phase-1 baseline
3. 业务数据到位后:ECDet 部署基线 + 业务域上的增强验证
4. 其余 CDFSOD 域按需补 baseline

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
