# Fix sys.path: script dir is /workspace which shadows editable vllm install
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
GPU resource allocation experiment for DeepSeek-V2-Lite MoE.

Measures and plots three dimensions of resource allocation:
  1. Static GPU memory  — equal for all experts (same weight tensors)
  2. Dynamic CUDA blocks — proportional to tokens routed (hotness)
  3. Actual GPU wall-time per MoE layer — measured with CUDA events,
     correlated with per-layer load imbalance

All outputs written to the script's directory (/ /tmp fallback).
"""

import os, math, random, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
SEED             = 42
N_SAMPLES        = 200
MAX_NEW_TOKENS   = 64
HF_MODEL         = "deepseek-ai/DeepSeek-V2-Lite"
RESULT_DIR       = "/tmp"
OUTPUT_DIR       = os.path.dirname(os.path.abspath(__file__))

# DeepSeek-V2-Lite topology
N_LAYERS_TOTAL   = 27
FIRST_DENSE      = 1
N_MOE_LAYERS     = N_LAYERS_TOTAL - FIRST_DENSE   # 26
N_ROUTED_EXPERTS = 64
TOPK             = 6
BLOCK_SIZE_M     = 16    # FusedMoE default for small-M (tokens <= n_experts)

random.seed(SEED)
np.random.seed(SEED)

# ── globals ───────────────────────────────────────────────────────────────────
expert_counts   = defaultdict(lambda: np.zeros(N_ROUTED_EXPERTS, dtype=np.int64))
layer_times_ms  = defaultdict(list)   # layer_idx -> list of forward time (ms)


# ── patch helpers ─────────────────────────────────────────────────────────────
def patch_model(llm):
    """
    Attach two instance-level patches to each DeepseekV2MoE:
      - capture topk routing to accumulate expert_counts
      - wrap forward with CUDA events to measure wall-time per layer
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
    import types

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    moe_idx = 0
    for _, module in model.named_modules():
        if not isinstance(module, DeepseekV2MoE):
            continue

        layer_idx = moe_idx
        orig_fwd  = module.forward

        def make_patch(orig, li):
            def patched(self, hidden_states):
                hs = hidden_states.view(-1, hidden_states.shape[-1])

                # ── routing capture ──
                with torch.no_grad():
                    rlogits, _ = self.gate(hs)
                    _, topk_ids = grouped_topk(
                        hidden_states=hs,
                        gating_output=rlogits,
                        topk=TOPK,
                        renormalize=True,
                        num_expert_group=getattr(self.experts, "num_expert_group", 8),
                        topk_group=getattr(self.experts, "topk_group", 3),
                    )
                    ids_np = topk_ids.cpu().numpy().astype(np.int64).ravel()
                    np.add.at(expert_counts[li], ids_np, 1)

                # ── CUDA event timing ──
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                out = orig(hidden_states)
                t1.record()
                torch.cuda.synchronize()
                layer_times_ms[li].append(t0.elapsed_time(t1))
                return out
            return patched

        module.forward = types.MethodType(make_patch(orig_fwd, layer_idx), module)
        moe_idx += 1

    print(f"[hook] Patched {moe_idx} MoE layers.")
    return moe_idx


# ── data helpers ──────────────────────────────────────────────────────────────
def load_prompts(n):
    cache = os.path.join(RESULT_DIR, "sharegpt_cache.json")
    if os.path.exists(cache):
        with open(cache) as f:
            data = json.load(f)
    else:
        from datasets import load_dataset
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                          split="train")
        data = list(ds)
        with open(cache, "w") as f:
            json.dump(data, f)

    rng = random.Random(SEED)
    rng.shuffle(data)
    prompts = []
    for item in data:
        for turn in (item.get("conversations") or item.get("conversation") or []):
            role = turn.get("from", turn.get("role", ""))
            val  = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and val.strip():
                prompts.append(val.strip()[:512])
                break
        if len(prompts) >= n:
            break
    print(f"[data] {len(prompts)} prompts loaded.")
    return prompts


# ── static memory calculation ─────────────────────────────────────────────────
def calc_static_memory_mb(llm):
    """
    For each MoE layer, return per-expert static weight size in MB.
    All experts have identical shapes → equal allocation.
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    results = []
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for _, module in model.named_modules():
        if not isinstance(module, DeepseekV2MoE):
            continue
        # FusedMoE stores experts as two fused weight tensors: w13 and w2
        # w13 shape: (n_experts, 2*intermediate, hidden)
        # w2  shape: (n_experts, hidden, intermediate)
        w13 = module.experts.w13_weight   # (E, 2*I, H)
        w2  = module.experts.w2_weight    # (E, H, I)
        bytes_per_expert = (
            (w13.numel() // w13.shape[0]) * w13.element_size() +
            (w2.numel()  // w2.shape[0])  * w2.element_size()
        )
        results.append(bytes_per_expert / 1024**2)
        break  # all layers same shape → one is enough
    return results[0] if results else None


# ── CUDA block allocation ─────────────────────────────────────────────────────
def compute_cuda_blocks(hotness_matrix, block_size=BLOCK_SIZE_M):
    """
    hotness_matrix: (n_moe_layers, n_experts) selection probability
    Returns blocks_matrix (same shape): estimated CUDA blocks per expert per layer.
    Total tokens per layer = sum of counts / topk.
    blocks_i = ceil(tokens_i / block_size)
    """
    counts = np.load(os.path.join(OUTPUT_DIR, "expert_counts.npy"))
    tokens_per_layer = counts.sum(axis=1, keepdims=True) / TOPK
    # per-expert token count
    expert_tokens = counts / TOPK                         # (n_moe, n_experts) fractional
    blocks = np.ceil(expert_tokens / block_size).astype(int)
    blocks = np.maximum(blocks, 0)
    return blocks, expert_tokens


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_static_memory(mem_mb_per_expert, out_dir):
    x = np.arange(N_ROUTED_EXPERTS)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.bar(x, [mem_mb_per_expert] * N_ROUTED_EXPERTS,
           color="steelblue", width=1.0)
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Static GPU Memory (MB)", fontsize=12)
    ax.set_title("Static GPU Memory per Expert — Equal for All\n"
                 f"({mem_mb_per_expert:.1f} MB × {N_ROUTED_EXPERTS} experts = "
                 f"{mem_mb_per_expert * N_ROUTED_EXPERTS:.0f} MB per MoE layer)",
                 fontsize=11)
    ax.set_ylim(0, mem_mb_per_expert * 1.3)
    ax.axhline(mem_mb_per_expert, color="red", linestyle="--",
               linewidth=1.2, label="All equal")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "alloc_static_memory.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_cuda_blocks(blocks, hotness, out_dir):
    """Scatter: expert hotness vs. CUDA blocks (averaged across layers)."""
    avg_hotness = hotness.mean(axis=0)   # (n_experts,)
    avg_blocks  = blocks.mean(axis=0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter hotness vs blocks
    ax = axes[0]
    ax.scatter(avg_hotness, avg_blocks, alpha=0.6, s=40, color="darkorange")
    # regression line
    m, b = np.polyfit(avg_hotness, avg_blocks, 1)
    xs = np.linspace(avg_hotness.min(), avg_hotness.max(), 100)
    ax.plot(xs, m * xs + b, color="red", linewidth=1.5,
            label=f"Linear fit (R²={np.corrcoef(avg_hotness, avg_blocks)[0,1]**2:.3f})")
    uniform_h = TOPK / N_ROUTED_EXPERTS
    ax.axvline(uniform_h, color="gray", linestyle="--", linewidth=1,
               label=f"Uniform hotness ({uniform_h:.4f})")
    ax.set_xlabel("Avg Expert Selection Probability (hotness)", fontsize=11)
    ax.set_ylabel("Avg CUDA Blocks Allocated", fontsize=11)
    ax.set_title("Hotness → CUDA Block Allocation\n(proportional competition for GPU SMs)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # Right: bar chart top-10 hot vs bottom-10 cold experts
    sorted_idx = np.argsort(avg_hotness)
    cold10 = sorted_idx[:10]
    hot10  = sorted_idx[-10:]
    labels = ([f"E{i}\n(cold)" for i in cold10] +
              [f"E{i}\n(hot)" for i in hot10])
    vals   = list(avg_blocks[cold10]) + list(avg_blocks[hot10])
    colors = ["#4878d0"] * 10 + ["#ee854a"] * 10
    ax = axes[1]
    ax.bar(range(20), vals, color=colors)
    ax.set_xticks(range(20))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Avg CUDA Blocks", fontsize=11)
    ax.set_title("Top-10 Hottest vs Bottom-10 Coldest Experts\n(avg CUDA blocks per forward pass)",
                 fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4878d0", label="Cold experts"),
                       Patch(color="#ee854a", label="Hot experts")],
              fontsize=9)

    fig.suptitle("Dynamic GPU Compute Allocation: Hot Experts Get More CUDA Blocks",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "alloc_cuda_blocks.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_layer_timing(layer_times, hotness, out_dir):
    """Per-layer MoE wall-time vs. load imbalance."""
    n_moe = len(layer_times)
    avg_time = np.array([np.mean(layer_times[i]) for i in range(n_moe)])
    std_time = np.array([np.std(layer_times[i])  for i in range(n_moe)])

    # load imbalance: max/mean hotness per layer
    imbalance = hotness.max(axis=1) / hotness.mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-layer avg timing
    ax = axes[0]
    x = np.arange(n_moe)
    ax.bar(x, avg_time, color="steelblue", label="Avg forward time")
    ax.errorbar(x, avg_time, yerr=std_time, fmt="none", ecolor="gray",
                elinewidth=0.8, capsize=2)
    ax.set_xlabel("MoE Layer Index", fontsize=11)
    ax.set_ylabel("GPU Wall-Time (ms)", fontsize=11)
    ax.set_title("Actual GPU Time per MoE Layer\n(CUDA event measurement)", fontsize=10)
    ax.legend()

    # Right: scatter imbalance vs time
    ax = axes[1]
    ax.scatter(imbalance, avg_time, s=50, alpha=0.8, color="darkorange")
    for i, (xi, yi) in enumerate(zip(imbalance, avg_time)):
        if xi > imbalance.mean() + imbalance.std() or yi > avg_time.mean() + avg_time.std():
            ax.annotate(f"L{i}", (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    m, b = np.polyfit(imbalance, avg_time, 1)
    xs = np.linspace(imbalance.min(), imbalance.max(), 100)
    r2 = np.corrcoef(imbalance, avg_time)[0, 1] ** 2
    ax.plot(xs, m * xs + b, "r-", linewidth=1.5, label=f"Linear fit (R²={r2:.3f})")
    ax.set_xlabel("Load Imbalance (max / mean hotness)", fontsize=11)
    ax.set_ylabel("Avg GPU Wall-Time (ms)", fontsize=11)
    ax.set_title("Load Imbalance → GPU Time\n(more imbalanced = longer?)", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle("GPU Wall-Time per MoE Layer: Measured vs. Load Imbalance",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "alloc_layer_timing.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_summary_comparison(mem_mb, blocks, hotness, layer_times, out_dir):
    """
    4-panel summary:
      [A] static memory (all equal)
      [B] avg CUDA blocks per expert (varies with hotness)
      [C] hot vs cold expert: memory vs compute share
      [D] layer timing distribution
    """
    n_moe = len(layer_times)
    avg_hotness  = hotness.mean(axis=0)
    avg_blocks   = blocks.mean(axis=0).astype(float)
    total_blocks = avg_blocks.sum()
    block_share  = avg_blocks / total_blocks * 100   # % of compute

    mem_share = np.full(N_ROUTED_EXPERTS, 100.0 / N_ROUTED_EXPERTS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # A: static memory share (all equal)
    ax = axes[0, 0]
    ax.bar(range(N_ROUTED_EXPERTS), mem_share, color="steelblue", width=1.0)
    ax.axhline(100.0 / N_ROUTED_EXPERTS, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Expert ID"); ax.set_ylabel("Memory Share (%)")
    ax.set_title("A. Static GPU Memory Share — Equal (1/64 each)")
    ax.set_ylim(0, mem_share[0] * 2)

    # B: compute block share (varies)
    ax = axes[0, 1]
    colors_b = plt.cm.RdYlGn_r(
        (block_share - block_share.min()) / (block_share.max() - block_share.min() + 1e-9)
    )
    ax.bar(range(N_ROUTED_EXPERTS), block_share, color=colors_b, width=1.0)
    ax.axhline(100.0 / N_ROUTED_EXPERTS, color="gray", linestyle="--",
               linewidth=1, label="Equal baseline")
    ax.set_xlabel("Expert ID"); ax.set_ylabel("Compute Block Share (%)")
    ax.set_title("B. Dynamic GPU Compute Share — Proportional to Hotness")
    ax.legend(fontsize=8)

    # C: grouped bars — top/bottom 10 memory vs compute share
    sorted_idx  = np.argsort(avg_hotness)
    highlight   = np.concatenate([sorted_idx[:5], sorted_idx[-5:]])
    labels      = ([f"E{i}\n(cold)" for i in sorted_idx[:5]] +
                   [f"E{i}\n(hot)"  for i in sorted_idx[-5:]])
    x = np.arange(10)
    w = 0.35
    ax = axes[1, 0]
    ax.bar(x - w/2, mem_share[highlight], w, label="Memory share %", color="#4878d0")
    ax.bar(x + w/2, block_share[highlight], w, label="Compute block share %", color="#ee854a")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Share of total (%)")
    ax.set_title("C. Memory vs Compute Share: 5 Coldest vs 5 Hottest Experts")
    ax.legend(fontsize=9)
    ax.axhline(100.0 / N_ROUTED_EXPERTS, color="gray", linestyle="--", linewidth=1)

    # D: layer timing box
    ax = axes[1, 1]
    data_for_box = [layer_times[i] for i in range(n_moe)]
    bp = ax.boxplot(data_for_box, patch_artist=True,
                    medianprops=dict(color="red", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3))
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec6e8")
    ax.set_xlabel("MoE Layer Index"); ax.set_ylabel("GPU Wall-Time (ms)")
    ax.set_title("D. GPU Time Distribution per MoE Layer (all 200 prompts)")
    ax.set_xticks(range(1, n_moe + 1, 4))
    ax.set_xticklabels(range(0, n_moe, 4), fontsize=8)

    fig.suptitle(
        "GPU Resource Allocation in DeepSeek-V2-Lite MoE\n"
        "Memory = static & equal  |  Compute = dynamic & hotness-proportional",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "alloc_summary.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    from vllm import LLM, SamplingParams

    print("[model] Loading vllm LLM …")
    llm = LLM(
        model=HF_MODEL,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=2048,
        enforce_eager=True,
    )

    n_moe = patch_model(llm)

    # static memory
    mem_mb = calc_static_memory_mb(llm)
    print(f"[info] Static memory per expert per layer: {mem_mb:.2f} MB")

    # inference
    prompts = load_prompts(N_SAMPLES)
    params  = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
    print(f"[infer] Running {len(prompts)} prompts …")
    llm.generate(prompts, params)
    print("[infer] Done.")

    # aggregate hotness
    counts = np.zeros((n_moe, N_ROUTED_EXPERTS), dtype=np.float64)
    for i in range(n_moe):
        counts[i] = expert_counts[i].astype(np.float64)
    tokens_per_layer = counts.sum(axis=1, keepdims=True) / TOPK
    hotness = counts / np.maximum(tokens_per_layer, 1)

    np.save(os.path.join(OUTPUT_DIR, "alloc_hotness.npy"),  hotness)
    np.save(os.path.join(OUTPUT_DIR, "alloc_counts.npy"),   counts)

    # CUDA block allocation
    expert_tokens = counts / TOPK
    blocks = np.ceil(expert_tokens / BLOCK_SIZE_M).astype(int)

    # plots
    plot_static_memory(mem_mb, OUTPUT_DIR)
    plot_cuda_blocks(blocks, hotness, OUTPUT_DIR)
    plot_layer_timing(layer_times_ms, hotness, OUTPUT_DIR)
    plot_summary_comparison(mem_mb, blocks, hotness, layer_times_ms, OUTPUT_DIR)

    print("[done] All figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
