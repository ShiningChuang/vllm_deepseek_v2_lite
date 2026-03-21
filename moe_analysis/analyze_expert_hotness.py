# Fix sys.path: when this script runs from /workspace, Python adds /workspace
# to sys.path[0], which shadows the editable vllm install with the repo root.
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Expert hotness analysis for DeepSeek-V2-Lite on ShareGPT dataset.

- Fixed random seed for reproducibility.
- Patches DeepseekV2MoE.forward to capture per-layer expert routing decisions.
- Plots per-layer expert hotness heatmap + average hotness bar chart.

Run inside the vllm-dev docker:
    python3 analyze_expert_hotness.py
"""

import os
import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
N_SAMPLES        = 200      # number of ShareGPT conversations to process
MAX_TOKENS_INPUT = 512      # truncate prompts to this many chars (approx)
MAX_NEW_TOKENS   = 64
RESULT_DIR       = "/tmp"   # mapped to /data/lxzhong_home/result on host
OUTPUT_DIR       = os.path.dirname(os.path.abspath(__file__))
HF_MODEL         = "deepseek-ai/DeepSeek-V2-Lite"

# DeepSeek-V2-Lite MoE topology
N_LAYERS_TOTAL   = 27
FIRST_DENSE      = 1        # layers 0..first_dense-1 are dense
N_MOE_LAYERS     = N_LAYERS_TOTAL - FIRST_DENSE   # 26
N_ROUTED_EXPERTS = 64
TOPK             = 6

# ── global storage for captured routing ─────────────────────────────────────
# expert_counts[layer_idx] = np.array of shape (N_ROUTED_EXPERTS,)
expert_counts   = defaultdict(lambda: np.zeros(N_ROUTED_EXPERTS, dtype=np.int64))
_layer_counter  = [0]   # mutable int used inside the patch

# ── patch DeepseekV2MoE.forward ─────────────────────────────────────────────
def _make_patched_forward(original_forward, layer_idx):
    """
    Wrap forward() to intercept router_logits before they enter grouped_topk,
    recompute topk_ids, and accumulate expert_counts.
    """
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk

    def patched_forward(self, hidden_states):
        # ---- replicate routing logic from DeepseekV2MoE.forward ----
        num_tokens, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            import torch.distributed as dist
            # just observe the local shard — still informative
            pass

        router_logits, _ = self.gate(hs)

        # recompute topk without touching the model's own computation path
        with torch.no_grad():
            _, topk_ids = grouped_topk(
                hidden_states=hs,
                gating_output=router_logits,
                topk=TOPK,
                renormalize=True,           # matches model config norm_topk_prob
                num_expert_group=self.experts.num_expert_group
                    if hasattr(self.experts, "num_expert_group") else 8,
                topk_group=self.experts.topk_group
                    if hasattr(self.experts, "topk_group") else 3,
            )
            # topk_ids: (num_tokens, topk)  int32
            ids_np = topk_ids.cpu().numpy().astype(np.int64).ravel()
            np.add.at(expert_counts[layer_idx], ids_np, 1)

        return original_forward(self, hidden_states)

    return patched_forward


def install_hooks():
    """Find all DeepseekV2MoE instances and wrap their forward methods."""
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    original = DeepseekV2MoE.forward
    # We patch at the class level but need per-layer indices.
    # Use a closure over a layer counter that increments each forward pass.
    # Because all MoE layers share the class, we use instance-level patching
    # after the LLM is built.
    return original


def patch_model_instances(llm):
    """
    After the LLM is built, walk the underlying model and attach
    instance-level patched forwards so each layer has its own index.
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    import types

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # Walk modules
    moe_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV2MoE):
            layer_idx = moe_idx   # layer index among MoE layers (0-based)
            orig = module.forward
            module.forward = types.MethodType(
                _make_patched_forward(
                    lambda self, hs, _orig=orig: _orig(hs),
                    layer_idx
                ),
                module
            )
            moe_idx += 1
    print(f"[hook] Patched {moe_idx} MoE layers.")
    return moe_idx


# ── load ShareGPT dataset ────────────────────────────────────────────────────
def load_sharegpt_prompts(n: int, seed: int = SEED) -> list[str]:
    """
    Download (or load cached) ShareGPT and return n human-turn prompts.
    Falls back to a small synthetic set if network is unavailable.
    """
    cache_path = os.path.join(RESULT_DIR, "sharegpt_cache.json")

    if os.path.exists(cache_path):
        print(f"[data] Loading ShareGPT cache from {cache_path}")
        with open(cache_path) as f:
            data = json.load(f)
    else:
        try:
            from datasets import load_dataset
            print("[data] Downloading ShareGPT from HuggingFace datasets …")
            ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                              data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                              split="train")
            data = [item for item in ds]
            with open(cache_path, "w") as f:
                json.dump(data, f)
            print(f"[data] Cached {len(data)} records to {cache_path}")
        except Exception as e:
            print(f"[data] ShareGPT download failed ({e}), using synthetic prompts.")
            return _synthetic_prompts(n, seed)

    # Extract first human turn from each conversation
    rng = random.Random(seed)
    prompts = []
    shuffled = list(data)
    rng.shuffle(shuffled)
    for item in shuffled:
        convs = item.get("conversations") or item.get("conversation") or []
        for turn in convs:
            role = turn.get("from", turn.get("role", ""))
            val  = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and val.strip():
                prompts.append(val.strip()[:MAX_TOKENS_INPUT])
                break
        if len(prompts) >= n:
            break
    print(f"[data] Selected {len(prompts)} prompts.")
    return prompts


def _synthetic_prompts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    templates = [
        "Explain the concept of {} in simple terms.",
        "Write a short story about {}.",
        "What are the pros and cons of {}?",
        "How does {} work?",
        "Summarize the history of {}.",
    ]
    topics = ["machine learning", "quantum computing", "climate change",
              "ancient Rome", "the stock market", "photosynthesis",
              "blockchain", "neural networks", "evolution", "democracy"]
    prompts = []
    for _ in range(n):
        t = rng.choice(templates)
        o = rng.choice(topics)
        prompts.append(t.format(o))
    return prompts


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    import torch
    torch.manual_seed(SEED)

    print("[model] Loading vllm LLM …")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=HF_MODEL,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=2048,
        enforce_eager=True,
    )

    n_moe = patch_model_instances(llm)
    actual_n_moe = n_moe  # update in case model has different count

    prompts = load_sharegpt_prompts(N_SAMPLES)

    params = SamplingParams(
        temperature=0.0,    # greedy → fully deterministic given seed
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )

    print(f"[infer] Running {len(prompts)} prompts …")
    outputs = llm.generate(prompts, params)
    print(f"[infer] Done. Generated {len(outputs)} outputs.")

    # ── aggregate ─────────────────────────────────────────────────────────────
    # Normalise to selection probability per expert per layer
    counts_matrix = np.zeros((actual_n_moe, N_ROUTED_EXPERTS), dtype=np.float64)
    for i in range(actual_n_moe):
        counts_matrix[i] = expert_counts[i].astype(np.float64)

    # total tokens routed per layer = sum across experts / topk
    tokens_per_layer = counts_matrix.sum(axis=1, keepdims=True) / TOPK
    tokens_per_layer = np.maximum(tokens_per_layer, 1)   # avoid div/0

    # hotness = fraction of tokens that selected this expert
    hotness = counts_matrix / tokens_per_layer   # shape (n_moe, n_experts)

    # save raw data
    np.save(os.path.join(OUTPUT_DIR, "expert_counts.npy"), counts_matrix)
    np.save(os.path.join(OUTPUT_DIR, "expert_hotness.npy"), hotness)
    print(f"[save] Raw arrays saved to {OUTPUT_DIR}/")

    # ── plots ─────────────────────────────────────────────────────────────────
    _plot_heatmap(hotness, actual_n_moe, OUTPUT_DIR)
    _plot_avg_hotness(hotness, actual_n_moe, OUTPUT_DIR)
    _plot_per_layer_bars(hotness, actual_n_moe, OUTPUT_DIR)
    _plot_load_imbalance(hotness, actual_n_moe, OUTPUT_DIR)

    print("[done] All figures saved.")


def _plot_heatmap(hotness, n_moe, out_dir):
    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(hotness, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Selection probability")
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("MoE Layer Index (0 = layer 1 of model)", fontsize=12)
    ax.set_title("Expert Hotness Heatmap — DeepSeek-V2-Lite on ShareGPT", fontsize=13)
    ax.set_xticks(np.arange(0, N_ROUTED_EXPERTS, 8))
    ax.set_yticks(np.arange(n_moe))
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap_expert_hotness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def _plot_avg_hotness(hotness, n_moe, out_dir):
    avg = hotness.mean(axis=0)           # average across layers (n_experts,)
    std = hotness.std(axis=0)

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(N_ROUTED_EXPERTS)
    ax.bar(x, avg, color="steelblue", label="Mean across layers")
    ax.errorbar(x, avg, yerr=std, fmt="none", ecolor="gray",
                elinewidth=0.6, capsize=0, alpha=0.6, label="±1 std")
    uniform = TOPK / N_ROUTED_EXPERTS
    ax.axhline(uniform, color="red", linestyle="--", linewidth=1.2,
               label=f"Uniform baseline ({uniform:.4f})")
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("Avg selection probability", fontsize=12)
    ax.set_title("Average Expert Hotness across All MoE Layers", fontsize=13)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "avg_expert_hotness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def _plot_per_layer_bars(hotness, n_moe, out_dir):
    """Top-5 and bottom-5 experts per layer shown as grouped bars."""
    # pick a subset of layers to avoid 26-panel overload
    layers_to_show = list(range(0, n_moe, max(1, n_moe // 9)))[:9]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()

    for plot_i, li in enumerate(layers_to_show):
        ax = axes[plot_i]
        h = hotness[li]
        x = np.arange(N_ROUTED_EXPERTS)
        ax.bar(x, h, color="steelblue", width=1.0)
        uniform = TOPK / N_ROUTED_EXPERTS
        ax.axhline(uniform, color="red", linestyle="--", linewidth=0.9)
        ax.set_title(f"MoE layer {li} (model layer {li + FIRST_DENSE})",
                     fontsize=9)
        ax.set_xlabel("Expert ID", fontsize=7)
        ax.set_ylabel("P(selected)", fontsize=7)
        ax.tick_params(labelsize=6)

    for i in range(len(layers_to_show), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-Layer Expert Hotness Distribution", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "per_layer_expert_hotness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def _plot_load_imbalance(hotness, n_moe, out_dir):
    """Per-layer load imbalance: max/mean hotness ratio."""
    uniform = TOPK / N_ROUTED_EXPERTS
    max_h  = hotness.max(axis=1)
    mean_h = hotness.mean(axis=1)
    ratio  = max_h / np.maximum(mean_h, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.plot(np.arange(n_moe), max_h, marker="o", markersize=3,
            label="Max expert hotness")
    ax.axhline(uniform, color="red", linestyle="--", linewidth=1,
               label="Uniform baseline")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Max selection probability")
    ax.set_title("Hottest Expert per Layer")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(np.arange(n_moe), ratio, marker="s", markersize=3, color="darkorange",
            label="max / mean")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1,
               label="Perfect balance (=1)")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("max / mean ratio")
    ax.set_title("Load Imbalance per Layer")
    ax.legend(fontsize=9)

    fig.suptitle("Expert Load Analysis — DeepSeek-V2-Lite", fontsize=13)
    fig.tight_layout()
    path = os.path.join(out_dir, "load_imbalance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
