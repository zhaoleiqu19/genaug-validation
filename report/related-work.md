# Related Work：生成式数据增强 × 少样本检测/分类

> 整理目的：为「生成图像增强能否可度量地提升少样本目标检测」这一验证课题梳理相关工作。
> 选取原则：近两年（2024–2026）为主，更早的仅保留奠基性工作；只收与「生成数据 → 提升下游检测/分类」直接相关的论文。
> 整理日期：2026-07。⭐ = 与我们的设定（几十张真图、跨域、检测）最接近，优先精读。

## 领域地图

```
                        ┌─ 1. 生成增强直接用于 FSOD / CDFSOD（最同题的工作）
  生成 × 检测 ──────────┤
                        └─ 2. 检测数据生成引擎（layout-to-image / copy-paste，通用检测）

  生成 × 分类 ────────── 3. 生成增强用于分类（few-shot/fine-grained，方法论最成熟）

                        ┌─ 4. 特定域小数据案例（工业缺陷等，业务场景类比）
  落地与边界 ───────────┤
                        └─ 5. 何时有效 / 系统分析（决定我们实验该怎么设计）

  （附）CDFSOD testbed 相关，文末简列备查
```

---

## 1. 生成增强直接用于 FSOD / CDFSOD ⭐ 与我们目标最接近

- ⭐ **Domain-RAG: Retrieval-Guided Compositional Image Generation for Cross-Domain Few-Shot Object Detection** (NeurIPS 2025)
  <https://arxiv.org/abs/2506.05872> ｜ code: <https://github.com/LiYu0524/Domain-RAG>
  Training-free 组合式生成：检索域相似背景 → 用 Flux 生成域对齐背景 → 前景-背景合成（前景保真，框标注天然保留）。在 CDFSOD 上 SOTA。**与我们想验证的路线几乎同题，是最重要的参照/竞品。**

- ⭐ **MPAD: Multi-Perspective Data Augmentation for Few-shot Object Detection** (2025)
  <https://arxiv.org/abs/2502.18195>
  针对 FSOD 的生成增强框架：in-context 物体合成（ICOS）+ 生成过程中混合 prompt embedding（HPAS）产生 hard sample，并关注前景-背景关系。

- ⭐ **FLORA: Efficient Synthetic Data Generation for Object Detection in Low-Data Regimes via finetuning Flux LoRA** (2025)
  <https://arxiv.org/abs/2508.21712>
  在低数据检测场景下微调 Flux LoRA 来生成训练数据，强调低成本。**Flux + LoRA + 低数据检测，正是我们打算走的生成侧技术栈。**

- ⭐ **Few-Shot Synthetic Data Generation with Diffusion Models for Downstream Vision Tasks** (2026)
  <https://arxiv.org/abs/2605.11898>
  仅用 20–50 张真图微调 LoRA adapter，再用预训练扩散模型生成下游任务训练数据的轻量 pipeline。**数据量级与我们业务场景（几十张真图）完全一致。**

- **Boosting Few-Shot Detection with Large Language Models and Layout-to-Image Synthesis** (2024)
  <https://arxiv.org/abs/2410.06841>
  用 LLM 提出合理 layout，再 layout-to-image 合成新样本来增强 FSOD。

- **Diverse Instance Generation via Diffusion Models for Enhanced Few-Shot Object Detection in Remote Sensing Images** (2025)
  <https://arxiv.org/abs/2511.18031>
  遥感 FSOD 上用扩散模型生成多样实例做增强，域特定 FSOD+生成的案例。

- **Diverse Generation while Maintaining Semantic Coordination: A Diffusion-Based Data Augmentation Method for Object Detection** (2024)
  <https://arxiv.org/abs/2408.02891>
  面向检测的扩散增强，重点在生成多样性与语义/框一致性的平衡。

## 2. 检测数据生成引擎（layout-to-image / copy-paste，通用检测）

- ⭐ **ODGEN: Domain-specific Object Detection Data Generation with Diffusion Models** (NeurIPS 2024, Apple)
  <https://arxiv.org/abs/2405.15199>
  面向**特定域**检测数据集：先在目标域的物体 crop + 整图上微调扩散模型，再用视觉 prompt + 逐物体文本描述做 bbox 条件生成；YOLOv5/v7 上最高 +25.3 mAP@.50:.95。**「先域适配再生成」的代表，和我们的域偏移设定直接相关。**

- **InstaGen: Enhancing Object Detection by Training on Synthetic Dataset** (CVPR 2024)
  <https://arxiv.org/abs/2402.05937>
  给扩散模型接一个 grounding head 自动产生实例级标注，用纯合成数据扩充检测训练（偏开放词表场景）。

- **Gen2Det: Generate to Detect** (2023)
  <https://arxiv.org/abs/2312.04566>
  直接生成 scene-centric 图（而非单物体再粘贴），配套 image/instance 级过滤和训练 recipe；LVIS long-tail +2.13 box AP、**COCO 低数据设定 +2.27 box AP**。它的「过滤 + 训练策略比生成本身更重要」的结论值得借鉴。

- **DiffusionEngine: Diffusion Model is Scalable Data Engine for Object Detection** (2023)
  <https://arxiv.org/abs/2309.03893>
  预训练扩散模型 + Detection-Adapter，一步产出图像+框的可扩展数据引擎。

- **GeoDiffusion: Text-Prompted Geometric Control for Object Detection Data Generation** (ICLR 2024)
  <https://arxiv.org/abs/2306.04607>
  把 bbox、相机视角等几何条件编成文本 prompt 控制 T2I 生成检测数据（自动驾驶场景）。

- **AeroGen: Enhancing Remote Sensing Object Detection with Diffusion-Driven Data Generation** (CVPR 2025)
  <https://arxiv.org/abs/2411.15497>
  遥感检测的 layout 可控生成，首个同时支持水平框+旋转框条件；域特定生成引擎的近期代表。

- **X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion** (ICML 2023)
  <https://arxiv.org/abs/2212.03863>
  用 SD 生成 + CLIP 过滤物体实例做大规模 copy-paste，LVIS 长尾类 +6.8 box AP。**生成式 copy-paste 路线的奠基工作**（较旧但保留）。

- **Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection** (ICCV 2025)
  <https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_Cycle-Consistent_Learning_for_Joint_Layout-to-Image_Generation_and_Object_Detection_ICCV_2025_paper.pdf>
  生成器与检测器循环一致联合训练，代表「生成-检测闭环」的新方向。

## 3. 生成增强用于分类（few-shot / fine-grained，方法论最成熟的一块）

- ⭐ **DataDream: Few-shot Guided Dataset Generation** (ECCV 2024)
  <https://arxiv.org/abs/2407.10910>
  用少量真图微调 SD 的 LoRA，再用适配后的模型生成分类训练集。**「few-shot LoRA 定制生成器」这条路线在分类上的标准做法，我们要做的是它的检测版。**

- **DA-Fusion: Effective Data Augmentation With Diffusion Models** (ICLR 2024)
  <https://arxiv.org/abs/2302.07944>
  用 textual inversion 学新概念 token，再对真图做语义变体（img2img）增强；在扩散模型词表外的域（杂草识别）也有效。**「编辑真图」而非「从头生成」的代表。**

- **Diff-Mix: Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model** (CVPR 2024)
  <https://arxiv.org/abs/2403.19600>
  用扩散模型做类间图像插值（生成式 mixup），在 fine-grained/域特定数据集上稳定涨点。

- **Diff-II: Inversion Circle Interpolation for Data-scarce Classification** (CVPR 2025)
  <https://arxiv.org/abs/2408.16266>
  指出现有扩散增强难以兼顾 faithfulness 与 diversity；用同类样本 inversion 的圆插值 + 两阶段去噪解决，few-shot/long-tail/OOD 分类全面提升。**faithfulness–diversity 权衡的分析框架对我们设计实验有用。**

- **SaSPA: Advancing Fine-Grained Classification by Structure and Subject Preserving Augmentation** (NeurIPS 2024)
  <https://arxiv.org/abs/2406.14551>
  细粒度分类的生成增强：不直接以真图为源，用边缘/主体条件保结构生成，兼顾多样性与类别保真。

- **LoFT: LoRA-Fused Training Dataset Generation with Few-shot Guidance** (2025)
  <https://arxiv.org/abs/2505.11703>
  对每张真图各训一个 LoRA 再融合来生成数据，提升合成数据的保真+多样性。

- **Synthetic Data from Diffusion Models Improves ImageNet Classification** (TMLR 2023, Google)
  <https://arxiv.org/abs/2304.08466>
  在 ImageNet 上微调 Imagen 后生成数据增强训练集，首次在大规模分类上证明合成数据能带来显著提升（奠基性，保留）。

## 4. 特定域小数据案例（业务场景类比）

- **AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model** (AAAI 2024)
  <https://arxiv.org/abs/2312.05767>
  少样本工业缺陷生成：把缺陷解耦为外观 embedding + 空间位置，生成图像-mask 对用于下游缺陷检测/定位。

- **DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation** (CVPR 2025)
  <https://arxiv.org/abs/2408.13509>
  双分支互相关扩散：一支生成整图、一支生成缺陷区域，少样本下多样性/真实度/mask 精度均优，下游检测涨点。**若业务场景是缺陷类，这条线直接可用。**

- **Class-specific diffusion models improve military object detection in a low-data domain** (2026)
  <https://arxiv.org/abs/2604.18076>
  低数据、非常规域（军事目标）上训类别专用扩散模型生成检测数据。**「几十到几百张图的特定业务域」这一设定的直接类比。**

- **SynSur: An end-to-end generative pipeline for synthetic industrial surface defect generation and detection** (2026)
  <https://arxiv.org/abs/2604.26633>
  工业表面缺陷从生成到检测的端到端 pipeline。

## 5. 何时有效 / 系统分析（决定我们的实验设计）

- ⭐ **Scaling Laws of Synthetic Images for Model Training … for Now** (CVPR 2024, Google)
  <https://arxiv.org/abs/2312.04567>
  系统研究合成图像训练的 scaling 规律。关键结论：**合成数据在「真图供给有限」和「测试分布与训练分布差异大」时收益最大** —— 恰好是我们的少样本+跨域设定，可直接引用来论证课题合理性。

- ⭐ **Diffusion-Based Data Augmentation for Image Recognition: A Systematic Analysis and Evaluation** (2026)
  <https://arxiv.org/abs/2603.08364>
  对扩散增强方法的系统对比评测（哪些设计真正有效、公平比较），最新的一篇「盘点」，实验协议可参考。

- **Do We Need All the Synthetic Data? Targeted Image Augmentation via Diffusion Models** (2025)
  <https://arxiv.org/abs/2505.21574>
  质疑无差别扩量：只针对模型薄弱处定向生成更有效。对我们控制生成预算有直接指导意义。

- **Diversify, Don't Fine-Tune: Scaling Up Visual Recognition Training with Synthetic Images** (2023)
  <https://arxiv.org/abs/2312.02253>
  与 ODGEN/DataDream 相反的结论：微调生成器会导致过拟合、合成数据规模上不去，靠提升生成多样性更好。**「要不要在少量真图上微调生成器」是两派分歧点，正是我们可以用实验回答的问题。**

- **Advances in Diffusion Models for Image Data Augmentation: A Review** (2024)
  <https://arxiv.org/abs/2407.04103>
  扩散增强的综述：方法分类、评价指标与未来方向，可作术语和分类学的参考。

---

## 小结：对我们课题的启示

1. **路线图基本清晰，检测上有四条生成路线**：① 前景保留 + 背景生成/合成（Domain-RAG）；② layout 条件整图生成（ODGEN/GeoDiffusion/AeroGen）；③ 少样本 LoRA 定制生成器（DataDream/FLORA/LoFT，分类上成熟、检测上刚起步）；④ 真图编辑/插值（DA-Fusion/Diff-II，标注保留最自然）。
2. **共识**：收益集中在少样本、长尾、域偏移场景（Scaling Laws、Gen2Det）；生成后的**过滤与训练策略**往往比生成器本身更影响最终 mAP（Gen2Det、X-Paste）。
3. **未决问题（我们的验证空间）**：少样本下微调生成器是否值得（Diversify-Don't-Fine-Tune vs ODGEN/DataDream 两派矛盾）；faithfulness–diversity 如何权衡（Diff-II）；无差别生成 vs 定向生成（Targeted Augmentation）。
4. **与 SOTA 的关系**：CDFSOD 上生成增强已有 Domain-RAG（NeurIPS 2025）占住 training-free 组合路线；我们的差异化在于**业务导向的成本-收益度量**和 LoRA 域定制路线在检测上的验证。

## 附：CDFSOD testbed 相关（简列备查）

- [CD-ViTO](https://arxiv.org/abs/2402.03094) (ECCV 2024) — benchmark + baseline 检测器（[code](https://github.com/lovelyqian/CDFSOD-benchmark)）
- [NTIRE 2025 CDFSOD 挑战赛报告](https://arxiv.org/abs/2504.10685) ／ [NTIRE 2026 第二届报告](https://arxiv.org/abs/2604.11998) — 各队方法汇总，可对齐当前最强做法
- [Multi-Modal Prototypes for CDFSOD](https://arxiv.org/abs/2602.18811) (2026) — 非生成路线的最新进展，作对照
