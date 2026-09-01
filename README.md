# UniMoS-VC — Unified Multi-Omics Synergy Predictor

UniMoS-VC 是用于**药物组合协同效应预测**的可解释深度学习模型。在细胞条件化双线性核（pathway aggregation kernel）上，用虚拟细胞上下文 `z_cell`（scFoundation embedding + 功能活性 + 表达残差）同时条件化核轨道与残差轨道。

本仓库只发布**模型核心代码、训练/评估入口和正式超参**，不含原始数据、检查点或论文图表。

---

## 模型核心

```
输入
 ├── VirtualCellModule
 │     z_fm    = FoundationCellEncoder(scFoundation lookup)
 │     c       = c_fn + α_mut · c_mut_fn
 │     h_cell  = CellEncoder(HVG expression)
 │         └──→ z_cell = FuseMLP([z_fm ; c ; h_cell])
 │
 ├── 核轨道（可解释）
 │     pA, pB ∈ R^67   药物功能节点扰动向量
 │         └──→ PathwayAggregationKernel(pA, pB, z_cell)
 │                  W(z_cell) = W_base + ΔW_lowrank(z_cell)
 │                  score = pAᵀ W_sym pB + γ·(pA ⊙ pB)
 │
 └── 残差轨道（数据驱动）
       h_struct = StructureEncoder(fp/GIN)
       ri_repr  = RIEncoder(p, z_cell)
           └──→ SynergyHead → resid_logit

输出
  logit = core_logit + α(z_cell) · resid_logit
  ŷ_class  = σ(logit)        主任务：Loewe > 10 二分类
  ŷ_ri_A/B = RIHead(ri_repr) 辅助任务：单药 RI 回归
```

核心实现：

| 文件 | 作用 |
|------|------|
| `unimos/model/unimos.py` | LightningModule：组装前向、多任务损失、指标 |
| `unimos/model/virtual_cell.py` | `FoundationCellEncoder` + `FuseMLP` + `VirtualCellModule` |
| `unimos/model/pathway_kernel.py` | 细胞条件化双线性核 `W(z_cell)`、γ、功能节点贡献 |
| `unimos/model/encoders.py` | Structure / Cell / RI / Synergy heads |
| `unimos/model/gin_encoder.py` | 可选 GIN 分子图编码器 |
| `unimos/training/loss.py` | BCE/Focal + Huber RI + 核正则 |
| `unimos/training/metrics.py` | AUROC / AUPRC / 阈值选择 |

详细设计见 [`UniMoS-VC_架构设计.md`](UniMoS-VC_架构设计.md)。

---

## 环境

```bash
conda create -n unimos python=3.10
conda activate unimos
pip install -r requirements.txt
```

GIN 结构编码器还需要 `torch-geometric`（按本机 PyTorch / CUDA 版本安装）。

---

## 数据（需自行放置）

将预处理特征放到 `data/`（可用 `--proc-dir` / `--func-dir` 覆盖路径）：

| 文件 | 内容 |
|------|------|
| `data/Drugcombv15/drugcombv15_unimos_pair_study_both_pathway.csv` | 主表 |
| `data/processed/cell_fn_vectors.npy` | 细胞功能节点活性 |
| `data/processed/cell_mut_fn_vectors.npy` | 细胞突变功能节点 |
| `data/processed/cell_raw_expr.npy` | 细胞原始表达 |
| `data/processed/drug_morgan_fps.npy` | Morgan 指纹 |
| `data/processed/cell_feature_index.json` | 细胞名 → 行索引 |
| `data/processed/drug_morgan_index.json` | InChIKey → 行索引 |
| `data/processed/cell_fm_emb.npy` | scFoundation 细胞 embedding（VC） |
| `data/Drugcombv15/function_nodes/` | 每药一个功能节点 `.npy` |

VC embedding 可用：

```bash
python -m unimos.data.precompute_fm_emb \
    --depmap-csv data/Depmap/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv \
    --scfoundation-dir data/external/scfoundation/official \
    --ckpt data/external/scfoundation/hf_mirror/models.ckpt \
    --cell-feature-index data/processed/cell_feature_index.json \
    --output data/processed/cell_fm_emb.npy
```

---

## 训练

```bash
# 快速验证
python train.py --split ldo --seed 0 --fast-dev-run

# 正式超参（四场景）
python train.py --split lco --seed 34 --config configs_final/lco.yaml
python train.py --split ldo --seed 34 --config configs_final/ldo.yaml
python train.py --split lpo --seed 34 --config configs_final/lpo.yaml
python train.py --split random --seed 34 --config configs_final/random.yaml
```

评估拆分：`ldo`（leave-drug-out）/ `lco`（leave-cell-out）/ `lpo`（leave-pair-out）/ `random`。

```bash
python evaluate.py run --checkpoint checkpoints/lco/seed_34/best.ckpt --split lco --seed 34
python summarize.py --outputs-dir checkpoints/ --results-dir results/
```

超参搜索：

```bash
python tune.py --split ldo --seed 0 --n-trials 30 --output configs/best_hparams.yaml
```

新药对推理：

```bash
python scripts/predict_novel_pair.py --help
```

---

## 仓库结构

```
unimos/
├── model/                 模型核心
│   ├── unimos.py
│   ├── virtual_cell.py
│   ├── pathway_kernel.py
│   ├── encoders.py
│   └── gin_encoder.py
├── training/              损失与指标
├── data/                  Dataset / split / FM embedding 预计算
└── eval/                  基线、校准、可解释性导出
train.py                   单次训练
tune.py                    Optuna HPO
evaluate.py                评估 + 核权重导出
configs/default_hparams.yaml
configs_final/             四场景正式超参及结构消融
```
