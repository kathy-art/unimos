# UniMoS-VC 架构设计（v3）
## 融合虚拟细胞与细胞扰动信息的机制可解释药物协同预测

> 本文档在 `unimos_v2` 双轨架构基础上重构，目标是把"细胞扰动信息"和"虚拟细胞信息"以**有文献支撑、且不重蹈 Route A–F 覆辙**的方式接入，同时保留（并硬化）机制可解释性。
> 命名沿用现有约定：`p` = 药物功能节点扰动向量，`c` = 细胞功能节点活性，`z_cell` = 新增的虚拟细胞上下文向量。

---

## 0. 设计前提：什么该做、什么别再做

你已经用六条路线系统性证伪了"把扰动响应签名当逐样本特征"这条路（D、F 正式判死，C 失败，A/B 稀释）。这不是调参问题，是这类用法的结构性天花板（SynVerse 2025：无模型打赢 one-hot 基线；Virtual Cell Challenge 2025：难稳超"预测平均值"）。因此本架构的第一原则是**换入口**，而不是再堆一层扰动特征。

| 信息 | ❌ 已证伪的用法（别重建） | ✅ 本架构采用的用法 |
|------|--------------------------|---------------------|
| 药物扰动响应（LINCS/Tahoe 打药后表达变化） | 逐样本签名 → 拼进特征向量（Route D/F） | 不用作逐样本特征；仅用**遗传/双扰动**数据离线估耦合先验（入口三） |
| 虚拟细胞（基础模型） | 用它预测"打药后细胞变成什么样"再回填 | 用作**细胞上下文编码器**：基线表达 → embedding → 条件化核（入口一） |
| 细胞异质性 | 静态/bulk PPI 一刀切 | **细胞系特异**通路耦合结构 W_prior(cell)（入口二，SynCell/MultiSyn） |

三个入口的共同点：都进入模型的**条件化/结构**通道，而不是加法特征通道——这是它们能绕开"逐样本无独立增量"的原因。

---

## 1. 总体架构

保留你的双轨骨架（可解释核 + 柔性残差 + 门控融合），把改造集中在**细胞侧的三个杠杆**上。

```
                                   ┌─────────────────────────────────────────────┐
 药物侧                            │  虚拟细胞模块 VirtualCellModule (新)          │
 ─────                            │                                               │
 pA, pB ∈ R^67   (靶点→通路→社区)  │  z_fm   = FoundationCellEncoder(expr_base)    │ ← scGPT/scFoundation/State
 hA, hB          (FP / GIN /       │           (基础模型 embedding, 冻结或 LoRA)   │
                  药效团片段图)      │  c      = c_fn + α_mut · c_mut_fn  (67维活性) │ ← 保留
                                   │  s_struct = StructSummary(W_prior(cell))      │ ← 细胞系特异结构摘要
                                   │  ───────────────────────────────────────     │
                                   │  z_cell = FuseMLP([z_fm ; c ; s_struct])      │  (统一细胞上下文)
                                   └──────────────────────┬────────────────────────┘
                                                          │  z_cell 同时条件化两条轨道
   ┌──────────────────────────────────────────────────────┴───────────────────────────┐
   │ 核轨道（可解释，机制驱动）                                                          │
   │                                                                                     │
   │   W(cell) = W_base            ← 由真实遗传互作离线估计的表观耦合先验 (入口三)         │
   │          + FiLM_γ(z_cell) ⊙ W_prior(cell)   ← 细胞系特异通路耦合 (入口二)            │
   │          + ΔW_lowrank(z_cell)               ← LoRA式条件化 (保留, 秩 r)              │
   │   W_sym  = (W + Wᵀ)/2                                                                │
   │   contrib_A = pA ⊙ (W_sym · pB) ,  contrib_B = pB ⊙ (W_sym · pA)                     │
   │   same_fn   = −(pA ⊙ pB ⊙ γ).sum()          γ<0 垂直协同 / γ>0 冗余                 │
   │        → core_head (单层线性, 保持可解释) → core_logit                               │
   └─────────────────────────────────────────────────────────────────────────────────┘
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │ 残差轨道（柔性，数据驱动）                                                          │
   │   h_struct = StructureEncoder(hA) + StructureEncoder(hB)                            │
   │   ri_repr  = RIEncoder(p, z_cell)                                                    │
   │        → resid MLP → resid_logit                                                     │
   └─────────────────────────────────────────────────────────────────────────────────┘

 融合:  logit_class = core_logit + α(z_cell) · resid_logit        α = sigmoid(gate(z_cell))
        ŷ = σ(logit_class)
```

与 v2 的三点结构性差异：
1. **细胞侧从"67 维活性 + 原始表达 MLP"升级为 VirtualCellModule**：把基础模型 embedding（虚拟细胞）、功能活性、细胞系特异结构摘要融合成统一上下文 `z_cell`，它同时条件化核与残差。
2. **核矩阵从 `W_base + ΔW(c)` 升级为 `W_base + FiLM⊙W_prior(cell) + ΔW(z_cell)`**：新增一个**细胞系特异的结构化通路耦合先验**（SynCell 路线的核心），并把 `W_base` 锚定到**测量到的遗传互作**上。
3. **门控 α 从标量升级为 α(z_cell)**：让残差轨道的贡献也随细胞上下文变化（某些细胞系机制清晰、核轨道够用；某些需要残差补足）。

---

## 2. 新增/改造模块（逐一）

### 2.1 虚拟细胞上下文编码器 `FoundationCellEncoder`（入口一）

**动机**：冷启动到 unseen cell line 是全领域天花板；67 维手工功能活性覆盖不了新细胞系的分布。基础模型在数千万细胞上预训练，把细胞系放到一个预训练流形上，是目前唯一有证据能改善 cell 冷启动的做法（DeepCDR+scGPT/scFoundation 优于原版；scDrugMap 基准）。**你已经对药物侧做了这件事（MolFormer 预训练 embedding），这是细胞侧的对称补齐。**

**输入**：细胞系基线转录组。两种来源，按覆盖择优：
- 单细胞：细胞系的 scRNA-seq（若有 Tahoe/其他图谱覆盖）→ 直接喂基础模型 → 池化成细胞系 embedding。
- 伪批量：CCLE/DepMap 的 bulk 表达（覆盖 1000+ 细胞系，比扰动库宽 5–10 倍）→ 作为"单细胞样"输入喂基础模型。

**模型选择**（`configs` 开关 `fm_backbone`）：
- `scfoundation`（默认，合并数据最强）/ `scgpt`（零样本最好）/ `state_se`（Arc State 的 State-Embedding，最新但独立验证少）/ `uce`（微调后最好）。
- 模式 `fm_mode ∈ {frozen, lora, finetune}`，默认 `frozen`（零样本 embedding，最省算力也最不易过拟合）；`lora` 作为二阶段。

**输出**：`z_fm ∈ R^d_fm`（如 512），经投影 MLP 到 128 维。

```python
# unimos/model/virtual_cell.py  (新文件)
class FoundationCellEncoder(nn.Module):
    def __init__(self, backbone="scfoundation", mode="frozen", out_dim=128):
        self.backbone = load_pretrained(backbone)      # 冻结/LoRA
        self.proj = MLP(self.backbone.emb_dim, out_dim)
    def forward(self, expr_base):                       # (B, n_genes)
        with torch.set_grad_enabled(self.mode != "frozen"):
            emb = self.backbone.embed(expr_base)        # (B, d_fm)
        return self.proj(emb)                           # (B, 128)
```

> **落地建议**：先把 embedding 离线预算好存成 `.npy`（细胞系 → 向量，类似你现有的 `cell_fn_vectors.npy`），训练时查表即可，不必每步跑基础模型。真正需要梯度回传时才切 `lora`。

### 2.2 细胞系特异通路耦合先验 `W_prior(cell)`（入口二）

**动机**：这是 2025 年冷启动泛化最大的单点杠杆（SynCell）。同一药物对在不同细胞系协同/不协同，本质是通路耦合被重连了——静态先验（你 v2 的 `W_prior` 是固定跨社区邻接）无法表达这种重连。

**构造**（离线，每个细胞系一张 67×67 矩阵）：
1. 取全局 PPI（STRING）作骨架。
2. 用该细胞系的表达/活性做**激活基因掩码**（SynCell 做法）：只保留在该细胞系里表达/活跃的蛋白及其边，得到细胞系特异子网络 `G_cell`。
3. 用你**已有的**投影管线（靶点→通路→Leiden 社区，67 维）把蛋白级 `G_cell` 投影/聚合到功能节点级 → `W_prior(cell) ∈ R^{67×67}`。
   - 边权可用子网络里跨社区的连接密度 / 通路活性协变。

**接入核**（FiLM 式条件化，借 SynCell 的 FiLM 思路）：
```
W(cell) = W_base + FiLM_γ(z_cell) ⊙ W_prior(cell) + FiLM_β(z_cell) + ΔW_lowrank(z_cell)
```
- `FiLM_γ, FiLM_β = Linear(z_cell)`，逐节点缩放/平移，让**同一结构先验在不同细胞上下文下被自适应调制**。
- Frobenius 门控仍保留（`‖ΔW‖_F ≤ τ`），现在也对 FiLM 项加范数约束，防止先验被冲垮。

> **为什么这条不会撞 Route C 的墙**：Route C 的失败是"把几千维压到几十维本身制造统计伪影"。这里的 67 节点投影**你已经在用**（不是新压缩），而且 `W_prior(cell)` 是作为**结构先验注入**、其贡献要在端到端 AUROC 和消融下被检验（见 §5），不靠内部 p 值自证。

### 2.3 遗传互作先验 `W_base`（入口三：扰动数据的正确用法）

**动机**：你证伪的是逐样本用扰动签名。但真实**遗传互作（GI）**数据可以离线估一个通路×通路的表观耦合，作为 `W_base` 的初始化/正则先验——**只需足够估一个 67×67 矩阵，不需要逐样本覆盖**，这就同时绕开了三边交集稀疏（Route F 的 4127 行/18 细胞系困境）和"逐样本无独立增量"两个坑。它还给你现有的 γ（垂直协同/冗余）一个测量根基，而不是纯参数。

**构造**（离线，一次性）：
1. 数据：Norman et al. CRISPR 双基因扰动（你 Route C 用过）、Replogle Perturb-seq、或从 Tahoe 提取的组合。
2. 对基因对 (i,j) 估**遗传互作分数** GI(i,j)（双扰动效应 − 两个单扰动效应的加性预期；这正是"epistasis"）。
3. 聚合到功能节点级：`W_base^prior[a,b] = agg_{i∈a, j∈b} GI(i,j)`。
4. 用作 `W_base` 的先验：
   - 初始化 `W_base ← W_base^prior`，或
   - 加正则 `lambda_gi · ‖W_base − W_base^prior‖²`（软锚定，允许数据微调）。

> **关键纪律**：GI 先验只进 `W_base`（全局、离线、结构性），**绝不**进逐样本前向。这一步要单独做一次消融——比较 `W_base` 随机初始化 vs GI 先验初始化的端到端表现，若无增益就退回随机初始化，别硬留。

### 2.4 药物侧：保留机制核 + 可选药效团残差

- **核轨道药物表征不变**：`pA/pB`（靶点→通路→社区）本身就比整体指纹更 LDO-鲁棒（有已知靶点的新药也能投影到功能节点），且是可解释性的来源。保留。
- **残差轨道结构编码器新增药效团选项**（MultiSyn 路线，`struct_encoder="pharmacophore"`）：把药物表示为原子节点 + 携带药效团的片段节点的异构图，用异构图 transformer 编码。对 LDO 更稳，且注意力能高亮关键子结构。作为 `fp / gnn` 之外的第三选项。
- **靶点缺失药物的兜底**：对无靶点标注的新药，核轨道 `p` 会退化为零向量。此时让门控 α(z_cell) 自动上调残差轨道权重（数据驱动兜底），并在输出里标记"该预测缺机制支撑"。

---

## 3. 机制可解释性的硬化（应对不可辨识性）

这是本架构相对"直接拼装已有模型"的增量，也是回应 2026"BINN 可解释性的幻觉"和你 Route C 伪影教训的地方。**光输出一个 W_sym 不叫可解释——它可能是不可辨识解里的任意一个。** 三道硬化：

### 3.1 多 seed 稳定性选择（你已有 `summarize.py`，直接扩展）
- 跨 N 个随机种子训练，只报告在多数种子里**稳定同号、量级一致**的 `W_sym[i,j]` 和 `γ[k]`。
- 输出改为带跨种子置信区间的区间估计，而非点估计。不稳定的交互权重**不写进机制结论**。

### 3.2 充分性 / 必要性消融（借 DrugCell 的 RLIPP / TranSynergy 的 SHAP 精神）
- 对每条被模型高亮的协同机制（如"节点 a×b 跨社区协同"），做**必要性检验**：屏蔽 `W_sym[a,b]`，看该样本 logit 变化。变化显著才算真依赖。
- **充分性检验**：只保留 top-k 交互，看能否复现大部分预测。
- 这把"可解释输出"从"看图说话"升级为"可证伪的机制主张"。

### 3.3 结构约束缩小解空间（VNN 综述结论：稀疏约束提升可辨识性）
- `W_base` 稀疏正则（你已有 `lambda_W` 核范数）+ 对角/非对角分离约束。
- 若某类耦合有生物先验符号（如冗余通路应正、垂直通路应负），加软符号约束。
- GI 先验锚定（§2.3）本身也在缩小解空间——学到的 `ΔW` 才好解释为"相对测量基线的情境化重连"。

**可解释性输出（升级版）**：

| 输出 | v2 | v3 升级 |
|------|----|---------| 
| `W_sym[i,j]` | 点估计 | 跨种子区间 + 必要性消融通过标记 |
| `γ[k]` | 点估计 | 跨种子区间；与 GI 先验方向一致性核对 |
| `contrib_A/B` | 逐节点贡献 | 同上 + "是否有机制支撑"标志 |
| `W_prior(cell)` vs `W(cell)` | — | **新**：可视化"该细胞系相对全局基线的通路重连"，这是虚拟细胞机制解释的落点 |

---

## 4. 与现有代码的映射（refactor 清单）

| 现有文件 / 类 | 改动 |
|---------------|------|
| `unimos/model/encoders.py :: CellEncoder` | 保留，但降级为 `z_cell` 的一个输入分支；主力让位给 VirtualCellModule |
| **`unimos/model/virtual_cell.py`（新）** | `FoundationCellEncoder` + `FuseMLP` → 产出 `z_cell` |
| `unimos/model/pathway_kernel.py :: PathwayAggregationKernel` | `W(c)` → `W(cell)`：新增 `W_prior(cell)` 输入端口、FiLM 调制、GI 先验锚定；Frobenius 门控扩展到 FiLM 项 |
| `unimos/model/encoders.py :: SynergyHead` | 门控 `gate_logit` 标量 → `gate(z_cell)`；`RIEncoder` 输入 `c` → `z_cell` |
| `unimos/model/encoders.py :: StructureEncoder` | 新增 `struct_encoder="pharmacophore"` 分支（异构图 transformer） |
| `unimos/model/gin_encoder.py` | 保留；药效团分支可与之并列 |
| `unimos/training/loss.py :: UniMoSLoss` | 新增 `lambda_gi`（GI 先验锚定）、`lambda_film`（FiLM 范数）；`lambda_gate` 现作用于 α(z_cell) |
| **`unimos/data/build_cell_prior.py`（新）** | 离线：DepMap 表达 + STRING → 激活掩码 → 67 节点投影 → `W_prior(cell)` 存 `.npy` |
| **`unimos/data/build_gi_prior.py`（新）** | 离线：Norman/Perturb-seq → GI 分数 → 节点级聚合 → `W_base^prior` 存 `.npy` |
| **`unimos/data/precompute_fm_emb.py`（新）** | 离线：细胞系表达 → 基础模型 → `cell_fm_emb.npy`（细胞系→向量查表） |
| `data/processed/` | 新增 `cell_fm_emb.npy` `W_prior_percell/*.npy` `W_base_gi_prior.npy` |
| `tune.py` 搜索空间 | 新增 `fm_backbone`、`fm_mode`、`lambda_gi`、`lambda_film`、FiLM 隐宽；`w_prior_scale` 语义改为 per-cell |
| `evaluate.py` | 新增导出：per-cell 通路重连图、必要性消融、跨种子稳定性 |
| `summarize.py` | 扩展为稳定性选择引擎（§3.1）+ 校准指标聚合（§5.2） |

---

## 5. 评估协议（预注册，防评估陷阱）

你在 GATE-PD1 上吃过评估协议偏差的亏（"通过"→改协议→"不通过"），领域内也正在打这场官司（Ahlmann-Eltze 2025"没用" vs 2025-10"指标选错了"；"The Metric Picks the Winner" 2026）。所以**评估协议要和架构一起预注册**，否则任何"提升"都不可信。

### 5.1 必须打赢的基线（任何组件不打赢这些就不接入）
- **one-hot 编码**（SynVerse 的杀手基线）。
- **drug_mean / cell_mean / global_mean**（你自己反复被它们打赢的、Virtual Cell Challenge 的"平均值"基线）。
- **RF / XGBoost**（你 Route B 里 0.78–0.85、碾压机制模型 0.73 的那组）。
- **v2 现状模型**（自身消融基线）。

### 5.2 校准 / 排序指标（不只 AUROC/MSE）
- 主指标：**AUPRC**（正类 6.6% 极不平衡，AUROC 会虚高）+ **top-k precision**（协同发现的真实使用场景是排序取前 k 做湿实验）。
- 排序类：Spearman / 加权排序指标（借 2025-10 反驳论文的"校准指标"框架）。
- 冷启动分层报告：`ldo / lco / lpo / random` 分别报，重点看 `ldo`（unseen drug）和 `lco`（unseen cell）——这才是新架构声称要改善的地方。

### 5.3 每个组件一道 GATE（沿用你的预注册判据文化）
| GATE | 组件 | 通过判据（示例） |
|------|------|------------------|
| G-VC | 虚拟细胞 embedding | `lco` 上 AUPRC 提升 CI 不跨 0，且打赢 cell_mean + one-hot |
| G-PRIOR | 细胞系特异 W_prior | `lco` 提升 CI 不跨 0；打乱细胞系↔W_prior 配对后增益消失（否则是平凡效应） |
| G-GI | GI 先验 W_base | 端到端 vs 随机初始化增益 CI 不跨 0；否则退回随机初始化 |
| G-INTERP | 可解释性 | 高亮机制通过必要性消融比例 > 阈值；跨种子稳定率 > 阈值 |

> **打乱对照是硬性关卡**（你 Route D 已经这么做过）：任何"提升"都要能扛住"把身份/配对随机打乱后提升应消失"这一击，才能排除"参数变多"的平凡效应。

---

## 6. 分阶段落地路线（每阶段一道可证伪的门）

按"先证明有信号，再谈覆盖够不够广"的顺序（你审计里那条逻辑纪律），不要一次全上。

- **Phase 0 —— 评估地基**：先把 §5.1 基线 + §5.2 校准指标做成 harness，在 v2 上跑通。**没有这个地基，后面所有"提升"都不可信。**
- **Phase 1 —— 虚拟细胞上下文**（入口一，最低风险、证据最足）：加 `FoundationCellEncoder`，冻结 embedding 查表。过 G-VC 才继续。
- **Phase 2 —— 细胞系特异 W_prior**（入口二，泛化杠杆最大）：离线建 per-cell 通路耦合，FiLM 接入。过 G-PRIOR（含打乱对照）。
- **Phase 3 —— GI 先验 W_base**（入口三，扰动数据的正确用法）：离线估 GI，软锚定。过 G-GI，否则退回。
- **Phase 4 —— 可解释性硬化**：稳定性选择 + 必要性消融 + per-cell 重连可视化。过 G-INTERP。
- **Phase 5 —— 药效团残差 / LoRA 微调**：LDO 若仍不够，再上药效团分支和基础模型 LoRA。

每阶段独立预注册判据、独立写判决文件——避免 Route C 那种"失败但没收尾文档、证据散在日志"的状态。

---

## 7. 数据就绪度：现有资产 vs 三入口需求

**核心判断：这三个入口没有一个是"逐样本扰动特征"，因此判死 Route D/F/B 的三重覆盖稀疏性（"药A × 药B × 细胞"实测覆盖仅 Tahoe 1.75% / LINCS 3.45%）在本架构里根本不进入依赖链。** 老路线问"这条 (A,B,cell) 样本有没有实测扰动签名"（几乎永远没有）；UniMoS-VC 只问"这个细胞系长什么样"（168 个全有）和"通路耦合结构是什么"（一份 Norman 就够估）。逐入口的真实依赖与覆盖：

| 入口 | 真正的数据依赖 | 粒度 | 当前覆盖 | 现有数据够否 |
|------|----------------|------|----------|--------------|
| 一 · 虚拟细胞 embedding | 每**细胞系**基线表达 | 逐细胞系（与药对无关） | **168/168 = 100%** | ✅ DepMap 全约 19,215 基因在盘 |
| 二 · 细胞特异结构 W_prior(cell) | 每**细胞系**通路活性 | 逐细胞系 | **168/168 = 100%** | ✅ ssGSEA 168×2923 + M 已有 |
| 三 · GI 先验 W_base | **一份**双基因扰动 | 一次性、与样本无关 | Norman 在盘 | ✅ 够估一个 67×67 |
| 核 + 残差主监督 | DrugComb 协同标签 | 逐样本 | 251,541 行 | ✅ 不变 |

### 7.1 已有（绿灯，直接用）
- 主监督 + 核轨道 + 残差轨道：DrugComb 251,541 样本、`pA/pB`(3816×67)、Morgan(3816×2048)、细胞 67 维活性——原封不动。
- **入口一原料**：`OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` 全约 19,215 蛋白编码基因。残差轨道现在只喂 150 HVG，但基础模型要全表达，全表达在盘。附带收益：FM embedding 逐细胞、无标签，不像 150-HVG 选择那样有 fold 泄漏，可**顺手绕过 §6.4 记录的 LCO 特征选择泄漏坑**。
- **入口二/三的投影器**：`pathway_to_function_matrix M∈R^(2923×67)` 是固定线性算子，`任意基因集信号 × M → 67 维`。STRING 蛋白、Norman 基因都能沿同一条 `基因→2923通路→67节点` 投进现有坐标系——已建好，不重来。
- **入口三数据**：Norman 2019（33,694基因×111,668细胞，单+双 CRISPR）在盘，正是 GEARS 用的那份。

### 7.2 需加工（黄灯，是处理而非采集，全部从盘上派生）
详见 §9。摘要：三段离线派生脚本 + 一个基因级 M + 基因 ID 对齐；STRING 可选（第一版可不下）。

### 7.3 现有数据解决不了的（红灯，须补采——详见 §8）
**168 个细胞系是 LCO（unseen cell）评估的硬天花板，且它由 DrugComb×DepMap 交集决定，不是扰动数据的问题。** 入口一/二改善的是"怎么用"这 168 个细胞的上下文，但有协同标签+特征的细胞系仍只有 168 个。LCO 拆分时测试集只剩很少几个细胞系——与 Route A"EPV 太小、测不出置信区间"同源。**即便 UniMoS-VC 真提升 cell 冷启动，168 个细胞也很难有把握测出来。** 所以要补数据，第一优先是**更多细胞系的协同标签**，而不是任何扰动数据。

### 7.4 不在关键路径上（可延后）
你为扰动数据建的最贵基础设施——**Tahoe 315GB 单细胞管线、LINCS Level-5——基本不是 UniMoS-VC 的 blocker**。入口一用 DepMap（更干净、覆盖更宽）出细胞 embedding；入口三用 Norman 的遗传 GI。Tahoe/LINCS 降级为 ComboSciPlex 类机制存在性验证 + 入口三可选补充，不必等它们接入正式 Dataset 才动工。

---

## 8. 数据补充计划（按任务约束 + 优先级）

> 本模型联用任务只有两个标签：**Loewe > 10 二分类**（主）+ **RI 回归**（辅，单药相对抑制）。数据补充必须服从这两个标签的可映射性，否则合并即引入标签噪声。

### 8.1 合并的硬约束：Loewe + RI 只能经同一套 SynergyFinder 重算得到

- **RI** 是 DrugComb 用 SynergyFinder 从**单药剂量-响应曲线**算出的相对抑制（log10 剂量-响应曲线下归一化面积 ≈ 百分比抑制）。新源要贡献 RI，必须有单药剂量-响应曲线、过同一 SynergyFinder；只报 IC50/AUC 的源给不了同单位 RI。
- **Loewe** 是模型特定的协同度量，只能从**原始剂量-响应矩阵**用 SynergyFinder（匹配设置）算出。ComboScore/Bliss/ZIP/HSA **都不是** Loewe，不能替代——一个药对可 Loewe 协同但 ComboScore 中性。

**合并判据表**（能否并入你的 Loewe>10 + RI 主池）：

| 源 | 能否产出 Loewe+RI | 已在 DrugComb v1.5? | 结论 |
|----|:-----------------:|:-------------------:|------|
| NCI-ALMANAC | ✅ 但**只能靠 DrugComb 重算**（原生是 ComboScore=改进版 Bliss） | ✅ 是（103 药 / 303,737 组合 / 60 细胞系） | **已在你 1,432,351 行里**，勿重复下原始版（下了得到 ComboScore，还要自己重算） |
| O'Neil | ✅（常规处理即 Loewe） | ✅ 是（38 药 / 92,208 组合 / 39 细胞系） | 你的外部测试集——见 §8.4 |
| AZ-DREAM | ✅（Loewe） | 部分收录，但 **AZ 组合数据专有，需单独协议** | O'Neil 已当测试集，可弃 |
| GDSC/Jaaks 2022、NSCLC 2023 | ✅ 若能拿到原始剂量-响应 | **需核实**（v1.5 已扩到 30+ 源） | 只有"v1.5 未收 + 有原始曲线"的才是真·新细胞 |
| 只报 ComboScore/ZIP/AUC 的源 | ❌ 标签层不可合并 | — | 仅作辅助信号，不进主池（见 §8.6） |

### 8.2 第一优先——止损你已经有的细胞（最大杠杆、几乎零采集）

**关键事实：你只剩 168 个细胞系，不是缺 ALMANAC/O'Neil（它们已在 DrugComb v1.5、已被重算成 Loewe+RI），而是你下游的通路过滤 + DepMap 特征映射（231→188→168）把它们的细胞筛掉了。** 被筛掉的数据里已经含 ALMANAC 的 NCI-60、O'Neil 的 39 系。

行动：
1. 审计 366,701 → 251,541 具体在 **DepMap 特征那一步**死了多少行、涉及多少细胞系；
2. 修 Cellosaurus→DepMap ModelID 映射失败 + 对缺特征细胞补拉 CCLE；
3. 大概率白捡几十个细胞系，**一行协同标签都不用新采**。

这比采任何新筛都高产，且不引入任何跨筛标签问题（同源同单位）。

### 8.3 第二优先——真·新细胞（仅补 v1.5 未收录且有原始剂量-响应的源）

先拉 DrugComb v1.5 完整来源清单（已从最初 ALMANAC/ONEIL/FORCINA/CLOUD 扩到 30+ 源）比对；只有**既没被 v1.5 收录、又能拿到原始剂量-响应矩阵**的源（GDSC^2/Jaaks 2022、NSCLC 2023 等，需逐一核实）才值得采。采回后**必须走与 DrugComb 相同的 SynergyFinder 设置**重算成 Loewe + RI，再过你现有的通路 + DepMap 过滤。

### 8.4 O'Neil 外部测试集的正确切法（当前很可能在漏）

**O'Neil 已是 DrugComb 源之一，所以它现在就在你的 366,701 / 251,541 训练池里——若不显式切出，你的"外部测试"是漏的。** 两件事：
1. 按 `study_name`（ONEIL）把 O'Neil 行**从训练/验证中剔除**，单独作 held-out。
2. **注意实体泄漏**：O'Neil 的 39 系、38 药与 ALMANAC 高度重叠，按 study 切出后同样的 (药, 细胞) 实体模型在 ALMANAC 里仍见过。真测"外部泛化"要报**实体级留出**（O'Neil 独有的药/细胞子集），或同时报 study 级与实体级两套数，别只报 study 级。

### 8.5 强化入口的数据（仅在对应入口已显苗头后再投）

- **入口三 GI 先验**：Norman 仅 ~230 个双基因组合，够估粗糙 67×67。想更密 → 补 **Replogle 2022 全基因组 Perturb-seq**（约 250 万细胞）或 **Horlbeck 2018 遗传互作图谱**（直接给成对 GI）。规模/访问上线前复核。
- **入口二 细胞特异结构**：想从 DepMap bulk 升到蛋白级 cell-specific PPI（真 SynCell 做法）→ 补细胞系单细胞；手上 Tahoe 覆盖 ~50 系，先给这 50 个上蛋白级，其余 bulk 兜底。
- **入口一 虚拟细胞**：DepMap bulk 已够，除非要让基础模型吃单细胞输入，否则不用补。

### 8.6 不能合并的源怎么用

拿不到原始剂量-响应、或只有非 Loewe 协同分数（ComboScore/ZIP-only）的源，**不能进 Loewe>10 主标签池**。但可作辅助：单独一个协同度量头（多任务，类似 DeepTraSynergy 的辅助损失）、或排序级预训练（对比学习预热）。务必与主 Loewe 标签**分头、分损失**，别混成一个标签。

---



## 9. 数据处理管线（离线派生 + 接入）

### 9.1 三段离线派生脚本（对应 §4 refactor 清单）

| 脚本 | 输入（均在盘或标准下载） | 输出 | 逻辑 | 纯现有数据? |
|------|--------------------------|------|------|:-----------:|
| `data/precompute_fm_emb.py` | DepMap 全表达(19,215) → 基础模型 | `cell_fm_emb.npy`（细胞系→向量查表） | scFoundation/scGPT `embed`，冻结 | ✅ |
| `data/build_cell_prior.py` | 现有 2923 通路图 + ssGSEA(168×2923) + M（STRING 可选） | `W_prior_percell/*.npy`（每细胞 67×67） | 用细胞通路活性调制通路邻接 → ×M 投 67 维 | ✅（STRING 为可选增强） |
| `data/build_gi_prior.py` | Norman 2019（+可选 Replogle/Horlbeck） | `W_base_gi_prior.npy`（67×67） | GEARS/直接 epistasis 算 GI → 沿 M 聚到 67 节点 | ✅ |

### 9.2 基因 ID 对齐（入口一的唯一实际工作量）
DepMap 是**大写 gene symbol**（19,215 蛋白编码），基础模型要特定 panel/Ensembl ID + 自家归一化（如 scFoundation 的 read-depth-aware 输入）。写一个 symbol→模型词表的映射 + 归一化适配层。注意别用 Tahoe 的 62,710-基因 token 体系混淆——168 个细胞系用 DepMap 更干净。

### 9.3 基因级 M（入口二/三共用）
现有 M 是**通路级**（2923×67）。把 STRING/Norman 的基因映到 2923 通路，用你构建 ssGSEA 时同一套 KEGG/Reactome 基因集定义即可，补一个 `gene → 2923 pathway` 稀疏映射，再 `× M` 得 `gene → 67`。一次性构建，两入口复用。

### 9.4 标签调和 + study 协变量接入 `UniMoSDataset`
- 新增列：`synergy_source`（study/筛来源）、`synergy_metric`（原始度量类型）。
- `study_name`/`synergy_source` 作为可选 embedding 输入或仅用于 leave-study-out 拆分（`splits.py` 增 `lso`）。
- 标准化/截断/聚合阈值一律**只用训练 fold** 拟合（延续 §7.2 纪律），新入口的 FM embedding、W_prior、GI 先验同样遵守。

### 9.5 每次重建的审计（对接你 §17.2 清单，新增项）
在现有审计基础上补：各新细胞系的 Cellosaurus→DepMap 覆盖；跨筛标签度量分布与调和前后对照；三个新 `.npy` 的形状/dtype/NaN/全零行；每个 split 中新增细胞系数与正类数；`W_prior/W_base` 的稀疏度与跨种子稳定性（供 §3.1 用）。

---

## 10. 一句话总结这个架构的立场

它不是"再试一次扰动数据"，而是把"细胞扰动 / 虚拟细胞"这两类信息从**已证伪的加法特征通道**，改道进入**有证据支撑的三条通道**：虚拟细胞当上下文编码器、细胞系特异结构当条件化先验、遗传互作当离线结构锚定；同时用可辨识性硬化把"机制可解释"从口号变成可证伪的主张。每一步都配你惯用的预注册 GATE 和打乱对照——如果某个入口也没信号，你会**干净地**知道，而不是又停在"喜忧参半"。

---

## 参考文献

**细胞系特异结构 / 情境化协同**
- SynCell: Contextualized Drug Synergy Prediction. arXiv 2511.17695 (2025). — 单细胞衍生细胞系特异 PPI + FiLM + 异构图 GNN；unseen drug/cell 上 SOTA。
- MultiSyn (Jin et al.). BMC Biology 23:200 (2025). — 细胞系相关 PPI 属性 GAT + 药效团片段异构图 transformer；冷启动评测。
- MFSynDCP. BMC Bioinformatics (2024). — 多源特征交互。

**机制可解释 / visible network**
- DrugCell / DCell (Kuenzi & Ideker et al.). Cancer Cell (2020). — GO 层级 visible network；拓扑异构酶 II+MAPK/PI3K 协同经实验验证。
- TranSynergy (Liu & Xie). PLOS Comput Biol (2021). — 靶点→PPI 随机游走传播 + SHAP，显式解构通路 crosstalk。
- MOViDA (Ferraro et al.). Bioinformatics (2023). — 多组学 visible network 扩展到协同。
- P-NET (Elmarakeby et al.). Nature (2021). — 生物先验分层网络。

**虚拟细胞 / 基础模型作细胞编码器**
- Integrating single-cell foundation models with GNN for CDR. arXiv 2504.14361 (2025). — DeepCDR + scGPT/scFoundation embedding 优于原版。
- scDrugMap. Nature Communications (2025). — 8 个基础模型基准；scFoundation 合并最强 / UCE 微调最好 / scGPT 零样本最好。
- State (Adduri et al.). bioRxiv 2025.06.26.661135. — State-Embedding + State-Transition。
- AI-driven virtual cell models. npj Digital Medicine (2025). — 综述。
- Nature Genetics Review (Wu, 2026). — 指出现代黑箱虚拟细胞丢失可解释性。

**方法论陷阱（必读，用于设计评估协议）**
- SynVerse. bioRxiv 2025.04.30.651516. — 无模型打赢 one-hot 基线。
- The illusion of interpretability in biologically informed neural networks. bioRxiv 2026.05.07.723544. — BINN 权重因不可辨识性未必反映真机制。
- Visible neural networks for multi-omics: a critical review. PMC12310660 (2025). — 稀疏约束提升可辨识性。
- Ahlmann-Eltze et al. Nature Methods (2025) vs. rebuttal bioRxiv 2025.10.20.683304 — 扰动模型"没用" vs"指标选错了"之争。
- The Metric Picks the Winner. arXiv 2606.12639 (2026). — 评估指标翻转模型排名。
- Virtual Cell Challenge. Cell (2025) / Arc wrap-up. — 1200+ 队难稳超"平均值"基线。

**迁移 / 零样本（Phase 5+ 备选）**
- Predicting drug responses of unseen cell types through transfer learning with foundation models. Nature Computational Science (2025).
- MAP: knowledge-driven single-cell responses for unprofiled drugs. bioRxiv 2026.02.25.708091.
- BATCHIE. Nature Communications (2024/2025). — 主动学习/贝叶斯实验设计（若能介入湿实验）。

**数据源（§8 补充计划用）**
- **DrugComb** (Zheng et al., NAR 2021 update; Zagidullin et al., NAR 2019) + **SynergyFinder** (Ianevski et al.). — 合并层：用 SynergyFinder 统一重算 Loewe/Bliss/HSA/ZIP + RI；v1.5 收 8397 药 / 739,964 组合 / 2320 细胞系 / 30+ 源。**新源要并入必须过同一 SynergyFinder。**
- NCI-ALMANAC (Holbeck et al.). Cancer Res 77:3564 (2017). — 103 药 × 60 NCI-60，>290,000 样本；**原生 ComboScore（改进版 Bliss），非 Loewe；已在 DrugComb v1.5（经重算得 Loewe+RI）。**
- O'Neil / Merck (O'Neil et al.). Mol Cancer Ther (2016). — 38 药 × 39 细胞，~92,208 组合；Loewe；**已在 DrugComb v1.5**；本项目**外部测试集**（须按 study 切出、注意实体泄漏）。
- AstraZeneca-Sanger DREAM (Menden et al.). Nat Commun (2019). — 910 组合 × 85 细胞；**AZ 组合数据专有，需单独协议**；O'Neil 已当测试集可弃。
- Jaaks et al. Nature 603:166 (2022) / NSCLC combination landscape. Nat Commun 14:3830 (2023). — 潜在真·新细胞，须核实是否已在 v1.5 且有原始剂量-响应。
- Replogle et al. Cell (2022). — 全基因组 Perturb-seq（入口三加密可选）。
- Horlbeck et al. Cell (2018). — 成对遗传互作图谱（入口三，直接 GI）。
- DepMap / CCLE（细胞系基线组学，1000+ 细胞系）；STRING（可选，入口二蛋白级 PPI）。
- 数据集规模/访问以各库当前发布为准，上线前复核。
