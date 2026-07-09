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


| Domain    | Shot | ours (mean±std) | paper | Δ     |
| --------- | ---- | --------------- | ----- | ----- |
| clipart1k | 1    | 56.57 ± 0.29    | 55.6  | +0.97 |
| clipart1k | 5    | 60.10 ± 0.30    | 59.4  | +0.70 |
| clipart1k | 10   | 61.07 ± 0.55    | 59.6  | +1.47 |
| FISH      | 1    | 42.37 ± 0.59    | 42.7  | −0.33 |
| FISH      | 5    | 44.63 ± 1.26    | 45.5  | −0.87 |
| FISH      | 10   | 45.77 ± 0.42    | 46.3  | −0.53 |


6 格均在发表值 ±1.5 mAP 内,复现可信。**此表即后续 clipart1k/FISH 上所有生成增强实验的固定对照组**,不随实验重跑。

Phase 1 先只建这两个域的 baseline:官方 config 各域训练预算差异大(如 DIOR 一个域 3-shot×3-seed 约 136 GPU·h),六域一次建齐性价比低;其余四域(ArTaxOr/DIOR/NEU-DET/UODD)在具体实验需要时按域补建,协议不变。

复现细节(环境、脚本、逐 run 结果与 manifest)见 `baselines/ftfsod_cdfsod/`。

### ✅ 生成路线可视化预检(训练前,零 GPU-训练成本)· 2026-07-07

文献综述:`report/related-work.md`。检测任务的核心约束和分类不一样:错误的
生成会污染**标签**(多出没标注的同类物体、目标被抹除/错位),不只是像素质量,
所以每条候选路线在投入检测器训练之前,先在真实 K-shot support 图上过一遍
廉价的可视化标签完整性预检。生成器统一用 FLUX.1-dev(`flux2` 环境;
FLUX.2-klein-4B 步数蒸馏、strength 曲线断崖式跳变,预检已淘汰)。

| 路线                      | 域                    | 判定                                                      |
| ----------------------- | -------------------- | ------------------------------------------------------- |
| 全局 img2img(原 rung-0 设计) | FISH(5 张 support 全测) | **证伪**——无论 strength 都无法保证 5 张图同时标签完整                    |
| 锁定 GT 框、重绘背景            | clipart1k(6 张)       | **失败**(0/6)——base FLUX.1-dev 非 inpainting 专训模型,倾向整图重编场景 |
| 锁定背景、框内局部重绘(strength<1) | clipart1k(6 张)       | **可行**(6/6 @ strength 0.4,5/6 到 0.8 仍干净)                |
| 锁定背景、框内局部重绘(strength<1) | FISH(5 张 support 全测) | **部分可行**——2/5 有窄安全窗口,3/5(小目标/模糊目标)全 strength 溶解         |


结果与机制全文:`report/genaug-rung0-precheck.md`。  
**"锁背景、框内局部重绘"确定为 clipart1k 的rung-1 方法**。  
FISH 域在小、模糊目标的情况下，strength0.4只对框内重绘也会目标溶解，后面考虑尝试降低strength试一下。  

### ✅ clipart1k rung-1 Stage 1(锁背景、框内局部重绘,1-shot)· 2026-07-09

strength=0.4,N=4(每 annotation 4 个合成变体),1-shot split(20 real
images/20 annotations)→ 80 张合成图,与真图合并为 100 张训练集(每类 1
real+4 synthetic)。训练复用冻结的 FT-FSOD 1-shot 官方 config(仅换训练数据
路径,其余超参一字未动),3 seeds(42/43/44)。

**主读数——固定真实数据量下,增强是否有效?**(1real+4synth vs 1-shot baseline):

| | mean ± std | n |
|---|---|---|
| 1-shot baseline(纯真图) | 56.57 ± 0.29 | 3 |
| 1real+4synth(box repaint) | 56.17 ± 0.67 | 3 |

Δ = -0.40,信号阈值(2× baseline std)= 0.58。**无信号,判定门槛不通过→不
投入 Stage 2(5-shot)。**

**次读数——合成图能否替代真实数据采集?**(1real+4synth,5 张/类,与
5-shot baseline 张数严格对齐,spec 中预先声明预期跑输,非待验证假设):

| | mean ± std | n |
|---|---|---|
| 5-shot baseline(纯真图) | 60.10 ± 0.30 | 3 |
| 1real+4synth(box repaint) | 56.17 ± 0.67 | 3 |

Δ = -3.93,与 spec 预判一致——单源图像在固定姿态/背景下重绘出的 4 个变体,
多样性不及 4 张真正不同的真实照片。

生成 pipeline 本身端到端验证无误(80/80 张分批生成全部成功、标注合并无缺
失、输出质量抽检正常),null 结果判定为该方法在当前 strength 下的真实天花
板,不是 pipeline 问题。按预先登记的决策标准,Stage 2、以及用于排除"训练步
数混淆"的 5×epoch 对照实验均跳过(Δ≈0,没有需要排除的正向结果)。下一个
trick 需要直接提升合成数据的多样性(如更换为组合式生成路线),而不是继续
调这条路线的参数。结论全文与 pipeline 细节:
`experiments/clipart1k_box_repaint/README.md`。

### 下一步

1. clipart1k:围绕"提升合成多样性"设计下一个 trick(组合式生成等候选方向,
   未定案)——不再沿 box repaint 这条路线调参
2. FISH 生成增强暂不投入——域固有的小目标/低对比问题,后面再做实验尝试
3. 业务数据到位后:ECDet 部署基线 + 业务域上的增强验证
4. 其余 CDFSOD 域按需补 baseline



## 仓库结构与运行

```
baselines/ftfsod_cdfsod/   baseline 跑批脚本、逐 run 结果、复现文档
generation/                合成数据管线(生成模型路线)
experiments/               每个实验一个目录:配置、脚本、结果、结论
report/                    证据报告(related-work、baseline 表、生成路线预检)
docs/specs/ docs/plans/    设计文档与实施计划
```

```bash
python3 -m pytest                                                # 单元测试
baselines/ftfsod_cdfsod/run_one.sh clipart1k 1 42 <gpu> <port>   # 单格 baseline
python3 baselines/ftfsod_cdfsod/aggregate_results.py baselines/ftfsod_cdfsod/results
```

环境与数据约定见 `AGENTS.md`。