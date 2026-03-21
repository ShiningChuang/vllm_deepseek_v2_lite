# DeepSeek-V2-Lite MoE Expert GPU 资源分配实验报告

**模型**: deepseek-ai/DeepSeek-V2-Lite
**硬件**: 2 × NVIDIA Tesla V100S PCIe 32GB
**数据集**: ShareGPT (N=200, seed=42, 结果可复现)
**MoE 拓扑**: 26 个 MoE 层，每层 64 个 routed experts，top-6 routing

---

## 核心结论

> **静态 GPU 内存平等分配，动态 GPU Compute 按热门度竞争分配。**

| 资源维度 | 分配方式 | 竞争性 |
|---------|---------|-------|
| GPU 显存（权重） | 每 expert 固定 **8.25 MB**，全部相等 | 无竞争 |
| CUDA Compute Blocks | 正比于 token 路由量（热门度） | 有竞争 |
| GPU SM 时间 | 热门 expert 占用更多 SM cycle | 有竞争 |

---

## 实验一：静态 GPU 内存分配

**结论：完全均等，无竞争。**

每个 MoE 层，所有 64 个 expert 各自持有：
- `w13_weight`: shape `(1, 2×intermediate, hidden)` — 门控+up proj 融合
- `w2_weight`:  shape `(1, hidden, intermediate)` — down proj

单个 expert 静态权重大小 = **8.25 MB**（FP16）。
26 层 × 64 experts × 8.25 MB ≈ **13.7 GB**，与模型加载日志 `14.71 GiB` 吻合（含 attention 权重）。

不论该 expert 在推理时是否被选中，它的权重始终常驻 GPU 显存。

📊 见图：`alloc_static_memory.png`

---

## 实验二：动态 CUDA Compute Block 分配

**结论：热门 expert 拿到显著更多的 CUDA blocks，存在明显竞争。**

FusedMoE 使用 Triton kernel，其 grid 大小为：

```
grid = Σ_e  ceil(tokens_routed_to_expert_e / BLOCK_SIZE_M)
```

其中 `BLOCK_SIZE_M = 16`（V100 小批量场景的默认值）。

### 热门度统计（跨所有层平均）

| | Expert ID | 选中概率 | 均匀基线 | 偏差 |
|---|---|---|---|---|
| 最热 | E50 | 0.1123 | 0.0938 | +1.20× |
| 第2热 | E16 | 0.1073 | 0.0938 | +1.14× |
| 最冷 | E3  | 0.0762 | 0.0938 | −0.81× |
| 第2冷 | E31 | 0.0796 | 0.0938 | −0.85× |

### CUDA Block 分配统计（跨层平均）

- 最热 expert 平均获得 **28.8 blocks/forward**
- 最冷 expert 平均获得 **19.8 blocks/forward**
- **热冷 block 比 = 1.5×**

热门 expert 每次 forward 占用的 GPU SM 时间约为冷门 expert 的 **1.5 倍**。

📊 见图：`alloc_cuda_blocks.png`

---

## 实验三：实测 GPU 层耗时与负载不均衡

**结论：层间耗时差异明显，负载不均衡度（max/mean）是重要影响因素。**

使用 CUDA Event 对每个 MoE 层的 forward 分别计时（200 × 平均 24279 tokens/layer）。

### 负载不均衡度（per-layer max/mean hotness ratio）

| 统计 | 值 |
|---|---|
| 最均衡层 | L25，ratio = 1.72 |
| 最不均衡层 | **L0**，ratio = **3.16** |
| 全层平均 ratio | 2.29 |

**L0（模型第 1 个 MoE 层）负载最不均衡**，部分 expert 热门度是平均值的 3× 以上。这符合 MoE 早期层学到更粗粒度路由的规律。

📊 见图：`alloc_layer_timing.png`

---

## 综合对比图解读

`alloc_summary.png` 包含 4 个子图：

- **A（左上）静态内存**：64 根等高柱，无任何差异
- **B（右上）动态 Compute block 占比**：可见明显不均匀，热门 expert 颜色更深
- **C（左下）5 冷 vs 5 热对比**：内存 share 完全相同（蓝色等高），compute block share（橙色）差距显著
- **D（右下）逐层耗时箱线图**：层间中位数和分布宽度均有差异，靠前的层（尤其 L0）方差更大

---

## 工程意义

1. **内存无法通过热门度释放**：冷门 expert 的显存无法回收，这是 MoE 显存开销大的根本原因。
2. **Compute 浪费在冷 expert padding 上**：Triton kernel 为每个 expert 至少分配 1 个 block，冷 expert 的 block 大量填充空 token，浪费 SM。
3. **ExpertDNS/vllm-epd 的 attention+FFN 分离意义**：通过将 attention worker (AW) 和 expert worker (EW) 解耦，可以让 EW 专注处理 expert 计算，并根据热门度动态调度，从根本上解决上述 compute 竞争与浪费问题。

---

---

## 实验四：同层不同 Expert 的实际执行时间差异（新实验）

**实验方法**：将 `DeepseekV2MoE.forward` 中的 FusedMoE 拆解，对每个 expert 单独用 CUDA Event 计时（F.linear 串行执行），观察同层内各 expert 执行时间的分布及其与热门度的关系。

**数据**：100 prompts × 26 MoE 层 × 64 experts，时间范围 0.1118 ~ 0.4991 ms。

### 核心发现：执行时间与热门度**几乎无相关**

| 指标 | 值 |
|---|---|
| R²（hotness, 执行时间） | **0.0006**（无相关） |
| 热门前5 expert 平均时间 | 0.1155 ms |
| 最冷后5 expert 平均时间 | 0.1164 ms |
| 全局最大/最小时间比 | **4.46×** |
| 层内最大/最小时间比均值 | **1.33×**（层间差异主导） |

**热门 expert 和冷门 expert 的单次执行时间几乎相同（0.1155 vs 0.1164 ms，差 <1%）。**

### 为什么热门度不影响单 expert 执行时间？—— 显存带宽瓶颈

**Micro-benchmark 实测**（受控实验，单 expert 合成权重，固定 V100S）：

| token 数 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 执行时间(ms) | 0.093 | 0.104 | 0.104 | 0.103 | 0.106 | 0.106 | 0.106 | 0.111 | 0.104 | 0.117 | **0.140** |

**1→256 个 token，执行时间几乎完全不变（相差 <15%）！** 直到 512 token 才开始明显上升。

原因是 **Roofline 分析**：

```
每个 expert 的权重大小：8.25 MB（FP16）
每个 token 的计算量：  8.65 MFLOPs
算术强度：            1.00 FLOP/byte per token

V100 平衡点（ridge point）：
  peak FP16 (112 TFLOPS) / 显存带宽 (900 GB/s) = 124 FLOP/byte

由于每 token 算术强度仅为 1.0 FLOP/byte << 124 FLOP/byte：
→ 需要 ≥ 124 个 token 才能从带宽瓶颈区进入计算瓶颈区
```

**ShareGPT 实验中每个 expert 每步实际接收的 token 数：约 2–5 个**，远低于 124 的平衡点。执行时间几乎全部花在**加载 8.25 MB 权重**上，与 token 多少无关。

若 token 数达到 128，理论上应快 128×，实测只快 1.19×，说明仍在带宽瓶颈区。

📊 见图：`microbench_time_vs_tokens.png`

而 FusedMoE 融合 kernel（vllm 正常推理路径）的行为完全不同：

```
FusedMoE Triton kernel 的时间分解（所有 expert 在单一 kernel 中）：
  ├── 总 CUDA blocks = Σ_e ceil(tokens_e / BLOCK_SIZE_M)
  ├── 热门 expert 占更多 blocks → 占更多 SM → 在 grid 中"挤占"冷 expert
  └── 热门 expert 直接贡献更多 kernel 执行时间
```

### 两种执行模式的对比

| | 串行单 expert（F.linear） | FusedMoE 融合 Kernel |
|---|---|---|
| 单 expert 耗时与热门度的关系 | **无关（R²≈0）** | 正比（blocks ∝ tokens） |
| 热/冷 expert 时间差 | <1% | ~1.5× |
| 层内执行时间方差来源 | GPU 调度随机性 | token 数量决定 |
| 总体耗时瓶颈 | kernel launch 固定开销 | 最热 expert 的 block 数 |

### 工程意义

1. **vllm 当前架构**（FusedMoE）：热门 expert 在融合 kernel 内占用更多 CUDA blocks，直接导致整个 MoE 层延迟由最热 expert 决定（长尾效应）。

2. **ExpertDNS/vllm-epd 的 EW 分离价值**：当 Expert Worker 独立执行时，每个 expert 的执行时间分布更像"串行"模式——固定开销主导，可以并行化多个 expert，消除 FusedMoE 的长尾效应。

3. **对 MoE 负载均衡的启示**：在 FusedMoE 路径下，负载均衡（减少 max hotness）直接降低延迟；但在 EW 分离路径下，每个 expert 执行时间接近常数，负载均衡的收益主要体现在吞吐量而非单请求延迟。

📊 见图：
- `per_expert_time_vs_hotness.png` — 热门度 vs 执行时间散点图（验证无相关）
- `per_expert_time_heatmap.png` — 26层×64 expert 执行时间热力图
- `per_expert_time_by_layer.png` — 6个代表层：时间分布 + 热门度叠加
- `per_expert_hot_vs_cold.png` — 热/冷 expert 时间分布对比（4格）

---

## 文件清单

| 文件 | 说明 |
|---|---|
| `alloc_static_memory.png` | 所有 expert 静态内存均等分配图 |
| `alloc_cuda_blocks.png` | hotness → CUDA block 分配散点图 + 冷热对比 |
| `alloc_layer_timing.png` | 逐层 GPU 实测耗时 + 负载不均衡相关性 |
| `alloc_summary.png` | 4 格综合对比总览图 |
| `alloc_hotness.npy` | 热门度矩阵 (26, 64) |
| `alloc_counts.npy` | token 路由计数矩阵 (26, 64) |
| `per_expert_time_vs_hotness.png` | 热门度 vs 执行时间散点（R²≈0） |
| `per_expert_time_heatmap.png` | 执行时间热力图 (26, 64) |
| `per_expert_time_by_layer.png` | 6个代表层时间+热门度叠加 |
| `per_expert_hot_vs_cold.png` | 热/冷 expert 时间对比（4格）|
| `microbench_time_vs_tokens.png` | 单 expert 时间 vs token 数 + Roofline 模型 |
| `per_expert_avg_time.npy` | 每 expert 平均执行时间 (26, 64) |
| `per_expert_hotness.npy` | 热门度矩阵 (26, 64)（本次实验） |
| `per_expert_counts.npy` | token 路由计数 (26, 64)（本次实验） |

**实验脚本**:
- `moe_analysis/analyze_gpu_allocation.py` — 实验一~三
- `moe_analysis/analyze_per_expert_timing.py` — 实验四

**运行环境**: Docker `vllm/vllm-openai:v0.10.2`，editable install `/workspace/vllm`
