# genaug-validation

业务导向的验证课题:**生成式图像增强能否可度量地提升少样本目标检测?**

测试床:CDFSOD 基准(NTIRE 挑战赛数据,6 个跨域目标域)+ 2 个业务场景(待定,预计每类仅几十张真实训练图)。

## 实验结构(与 mentor 对齐后的方案)

| 层 | 角色 | 检测器 |
|---|---|---|
| 强基线(CDFSOD) | 方法论验证的对照系 | FT-FSOD(MM Grounding DINO Swin-B) |
| 部署基线(业务数据) | 业务落地的实际载体 | EdgeCrafter / ECDet(业务数据到位后启用) |
| 生成侧 | 累积消融:`baseline → +trick1 → +trick1+trick2 → …` | 只做生成模型路线(不做 copy-paste),具体 trick 边调研边定 |

核心对比原则:增强后的结果只和**我们自己在同一环境、同一 seed 协议下跑出的无增强 baseline** 比,论文发表数字仅作外部合理性校验,不作对照组(避免环境差异混入增强效果)。

## 仓库结构

```
baselines/     K-shot 纯真实数据基线(一切对比的参照数字)
  ftfsod_cdfsod/   FT-FSOD × CDFSOD:跑批脚本、结果、结论
generation/    合成数据管线(生成模型路线)
experiments/   每个实验一个目录:配置、启动脚本、结果、结论短注
report/        面向 mentor 的证据报告(related-work 综述、baseline 表)
docs/specs/    设计文档(动手前评审);docs/plans/  实施计划
notes/         工作笔记
```

## 运行

```bash
python3 -m pytest                                           # 单元测试
baselines/ftfsod_cdfsod/run_one.sh FISH 1 42 <gpu> 9999     # 单组 baseline(域 shot seed gpu port)
python3 baselines/ftfsod_cdfsod/aggregate_results.py baselines/ftfsod_cdfsod/results
```

环境与数据约定见 `AGENTS.md`(共享数据只读规则、conda 环境、机器网络坑)。

---

## Changelog

### 2026-07-03

**方案确立**(与 xuanlong 对齐)
- 方向调整:CDFSOD 强基线用 FT-FSOD(替代最初的 CD-ViTO/HF RT-DETR 方案);ECDet 留给业务数据;生成策略做累积消融、具体 trick 边做边定。设计文档:`docs/specs/2026-07-03-cdfsod-ftfsod-baseline-design.md`。
- 文献综述完成:`report/related-work.md`(生成增强 × 少样本检测,6 个板块 40+ 篇,含与我们最接近的 Domain-RAG/FLORA/DataDream 等)。

**FT-FSOD 复现管线打通**
- 环境 `ftfsod`(torch 2.6.0+cu124 / mmcv 2.1.0 源码编译 / FT-FSOD 自带 mmdet fork)。过程中定位并修复的问题,均已记档:
  - 本机代理导致连接 CLOSE-WAIT 卡死(下载 40KB/s → 去代理后 10+MB/s);
  - mmcv 需 devtoolset-11(gcc 11)编译,系统 gcc 4.8.5 无 C++17;
  - 缺 `fairscale`(所有 config 的 activation checkpointing 依赖);
  - torch 2.6 `weights_only=True` 新默认拒载 mmengine checkpoint(已补丁);
  - 上游 `dist_test.sh` 不向 `test.py` 转发参数(绕过,直接调 `test.py`)。
- 冒烟验证:**FISH 1-shot mAP=0.429**(官方固定划分,seed 42,MMGDINO-B,16 iter 官方预算,RTX 4090D 单卡;独立复评精确复现全部 6 项 AP)。

**Phase-1 baseline 范围决策**
- 官方 config 各域 `max_epochs` 差异大(16~100)且每 epoch 全量验证,全量 6 域 × 3 seed 需 4-5 天 GPU 时。决定 Phase 1 只跑 **clipart1k + FISH**(2 域 × 1/5/10-shot × 3 seeds = 18 组,官方设置不动),其余 4 域按后续实验需要再补——baseline 是按域一次性成本,不是每次实验的重复成本。
- Phase-1 跑批已启动(2 卡并行,预计 ~7h),完成后 baseline 表落 `report/`。

## Roadmap

- [ ] Phase-1 baseline 表:clipart1k + FISH × {1,5,10}-shot × 3 seeds(mean±std)→ `report/`
- [ ] 生成侧调研收敛 → 第一个生成策略的 spec(生成模型路线,累积消融阶梯)
- [ ] 第一个生成增强实验 vs Phase-1 baseline
- [ ] 业务数据到位后:ECDet 部署基线
- 暂缓:ArTaxOr / DIOR / NEU-DET / UODD 域(按需追加)、Swin-L、支持集重采样
