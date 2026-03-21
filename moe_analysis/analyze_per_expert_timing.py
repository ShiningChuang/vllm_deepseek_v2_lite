# Fix sys.path: /workspace shadows the editable vllm install
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Per-expert execution time measurement for DeepSeek-V2-Lite.

Strategy:
  - Load model with TP=1 so all 64 experts reside on a single GPU.
  - Patch DeepseekV2MoE.forward to:
      1. Compute routing (topk_ids) as usual.
      2. For each expert e, extract its assigned tokens and run the FFN
         manually (gate/up/down proj) wrapped in CUDA events.
  - Collect per-expert timing over N_SAMPLES prompts.
  - Correlate with hotness (token count) and plot.
"""

import os, random, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# ── config ───────────────────────────────────────────────────────────────────
SEED             = 42
N_SAMPLES        = 100      # fewer prompts — per-expert loop is slower
MAX_NEW_TOKENS   = 32
HF_MODEL         = "deepseek-ai/DeepSeek-V2-Lite"
RESULT_DIR       = "/tmp"
OUTPUT_DIR       = os.path.dirname(os.path.abspath(__file__))
HOTNESS_PATH     = os.path.join(OUTPUT_DIR, "alloc_hotness.npy")

N_MOE_LAYERS     = 26
N_EXPERTS        = 64   # global expert count
N_LOCAL_EXPERTS  = 32   # experts per rank with TP=2
TOPK             = 6
FIRST_DENSE      = 1

random.seed(SEED)
np.random.seed(SEED)

# ── storage ──────────────────────────────────────────────────────────────────
# per_expert_times[layer][expert] = list of (time_ms, n_tokens) tuples
per_expert_times  = defaultdict(lambda: defaultdict(list))
# token counts per expert per layer (for hotness recompute)
expert_tok_counts = defaultdict(lambda: np.zeros(N_EXPERTS, dtype=np.int64))


# ── patch ────────────────────────────────────────────────────────────────────
def patch_model(llm):
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
    from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
    import types

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Detect weight shapes from first MoE layer
    for _, m in model.named_modules():
        if isinstance(m, DeepseekV2MoE):
            w13 = m.experts.w13_weight  # (E, 2*I, H) or fraction if TP
            w2  = m.experts.w2_weight   # (E, ?, ?)
            print(f"[shape] w13={tuple(w13.shape)}  w2={tuple(w2.shape)}")
            break

    moe_idx = 0
    for _, module in model.named_modules():
        if not isinstance(module, DeepseekV2MoE):
            continue
        layer_idx = moe_idx
        orig_fwd  = module.forward

        def make_patch(orig, li):
            def patched(self, hidden_states):
                hs = hidden_states.view(-1, hidden_states.shape[-1])
                n_tokens = hs.shape[0]

                # ── routing ──────────────────────────────────────────────
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
                    # topk_ids: (n_tokens, TOPK) — global expert IDs 0..63
                    ids_np = topk_ids.cpu().numpy().astype(np.int64)
                    for eid in ids_np.ravel():
                        if 0 <= eid < N_EXPERTS:
                            expert_tok_counts[li][eid] += 1

                    # ── per-expert timed FFN ──────────────────────────────
                    w13w = self.experts.w13_weight  # (E_local, 2*I, H)
                    w2w  = self.experts.w2_weight   # (E_local, I, H) or (E_local, H, I)
                    E_local = w13w.shape[0]

                    # Determine local expert offset (for TP > 1 each rank owns a shard)
                    # With TP=1 offset=0; for safety detect via ep_rank
                    ep_rank = getattr(self, "ep_rank", 0)
                    ep_size = getattr(self, "ep_size", 1)
                    experts_per_rank = N_EXPERTS // ep_size
                    local_start = ep_rank * experts_per_rank

                    for local_e in range(E_local):
                        global_e = local_start + local_e
                        if global_e >= N_EXPERTS:
                            continue
                        # tokens assigned to this expert (global ID)
                        mask = (topk_ids == global_e).any(dim=1)
                        n_e  = int(mask.sum().item())

                        # We time even if n_e==0 to capture the empty-case overhead
                        x_e = hs[mask]   # (n_e, H)

                        t0 = torch.cuda.Event(enable_timing=True)
                        t1 = torch.cuda.Event(enable_timing=True)
                        t0.record()

                        if n_e > 0:
                            # gate+up projection
                            # w13w[local_e]: (2*I, H) → F.linear(x, w) = x @ w.T
                            gate_up = F.linear(x_e, w13w[local_e])   # (n_e, 2*I)
                            half_I  = gate_up.shape[1] // 2
                            gate    = gate_up[:, :half_I]
                            up      = gate_up[:, half_I:]
                            inter   = F.silu(gate) * up              # (n_e, I)
                            # down projection
                            # w2w[local_e] shape: try (H, I) → F.linear gives (n_e, H)
                            # or (I, H) → need .T handled by F.linear
                            _ = F.linear(inter, w2w[local_e])        # (n_e, H)

                        t1.record()
                        torch.cuda.synchronize()
                        elapsed = t0.elapsed_time(t1)  # ms
                        per_expert_times[li][global_e].append((elapsed, n_e))

                # Run the real forward for correctness
                return orig(hidden_states)
            return patched

        module.forward = types.MethodType(make_patch(orig_fwd, layer_idx), module)
        moe_idx += 1

    print(f"[hook] Patched {moe_idx} MoE layers.")
    return moe_idx


# ── data ─────────────────────────────────────────────────────────────────────
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


# ── aggregate ─────────────────────────────────────────────────────────────────
def aggregate():
    """
    Returns:
      avg_time  (n_moe, n_experts): mean exec time in ms (excluding zero-token calls)
      avg_time_all (n_moe, n_experts): mean exec time including zero-token calls
      hotness   (n_moe, n_experts): selection probability
    """
    avg_time     = np.zeros((N_MOE_LAYERS, N_EXPERTS))
    avg_time_all = np.zeros((N_MOE_LAYERS, N_EXPERTS))
    sample_n     = np.zeros((N_MOE_LAYERS, N_EXPERTS), dtype=int)

    for li in range(N_MOE_LAYERS):
        for ei in range(N_EXPERTS):
            records = per_expert_times[li][ei]
            if not records:
                continue
            times_all = np.array([r[0] for r in records])
            ntoks     = np.array([r[1] for r in records])
            avg_time_all[li, ei] = times_all.mean()
            active = times_all[ntoks > 0]
            avg_time[li, ei]     = active.mean() if len(active) > 0 else 0.0
            sample_n[li, ei]     = len(records)

    # hotness from token counts
    counts = expert_tok_counts
    counts_mat = np.zeros((N_MOE_LAYERS, N_EXPERTS))
    for li in range(N_MOE_LAYERS):
        counts_mat[li] = counts[li].astype(float)
    tokens_per_layer = counts_mat.sum(axis=1, keepdims=True) / TOPK
    hotness = counts_mat / np.maximum(tokens_per_layer, 1)

    return avg_time, avg_time_all, hotness, counts_mat


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_time_vs_hotness(avg_time, hotness, out_dir):
    """Scatter: per-expert avg execution time vs. hotness, all layers."""
    # Flatten across layers, exclude zero-time entries
    h_flat = hotness.ravel()
    t_flat = avg_time.ravel()
    mask   = t_flat > 0
    h_use, t_use = h_flat[mask], t_flat[mask]

    uniform = TOPK / N_EXPERTS
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(h_use, t_use, c=t_use, cmap="plasma",
                    s=18, alpha=0.55, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Exec time (ms)")
    # regression
    m, b = np.polyfit(h_use, t_use, 1)
    xs = np.linspace(h_use.min(), h_use.max(), 200)
    r2 = np.corrcoef(h_use, t_use)[0, 1] ** 2
    ax.plot(xs, m * xs + b, "r-", linewidth=2,
            label=f"Linear fit  R²={r2:.3f}")
    ax.axvline(uniform, color="gray", linestyle="--", linewidth=1,
               label=f"Uniform hotness ({uniform:.4f})")
    ax.set_xlabel("Expert Selection Probability (hotness)", fontsize=12)
    ax.set_ylabel("Avg Expert Execution Time (ms)", fontsize=12)
    ax.set_title(
        "Per-Expert Execution Time vs. Hotness\n"
        "(each point = one expert in one layer, 100 prompts × 26 layers × 64 experts)",
        fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "per_expert_time_vs_hotness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_layer_heatmap(avg_time, out_dir):
    """Heatmap: MoE layer × expert → exec time."""
    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(avg_time, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Avg Exec Time (ms)")
    ax.set_xlabel("Expert ID", fontsize=12)
    ax.set_ylabel("MoE Layer Index", fontsize=12)
    ax.set_title(
        "Per-Expert Execution Time Heatmap — DeepSeek-V2-Lite\n"
        "(brighter = longer execution, darker = fewer tokens → faster)",
        fontsize=12)
    ax.set_xticks(np.arange(0, N_EXPERTS, 8))
    ax.set_yticks(np.arange(N_MOE_LAYERS))
    fig.tight_layout()
    path = os.path.join(out_dir, "per_expert_time_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_time_distribution_per_layer(avg_time, hotness, out_dir, n_layers=6):
    """For selected layers: bar chart of per-expert time colored by hotness."""
    layers = np.linspace(0, N_MOE_LAYERS - 1, n_layers, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    for pi, li in enumerate(layers):
        ax = axes[pi]
        t  = avg_time[li]   # (64,)
        h  = hotness[li]    # (64,)
        sort_idx = np.argsort(h)[::-1]   # sort by hotness descending
        colors = plt.cm.RdYlGn_r(
            (t[sort_idx] - t[sort_idx].min()) /
            (t[sort_idx].max() - t[sort_idx].min() + 1e-9)
        )
        ax.bar(range(N_EXPERTS), t[sort_idx], color=colors, width=1.0)
        # overlay hotness as line (secondary axis)
        ax2 = ax.twinx()
        ax2.plot(range(N_EXPERTS), h[sort_idx], color="blue",
                 linewidth=1.2, alpha=0.6, label="Hotness")
        ax2.set_ylabel("Hotness", fontsize=7, color="blue")
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=6)
        ax2.set_ylim(0, h.max() * 1.5)

        uniform = TOPK / N_EXPERTS
        ax2.axhline(uniform, color="blue", linestyle="--",
                    linewidth=0.8, alpha=0.5)

        ax.set_title(f"MoE Layer {li}  (model layer {li + FIRST_DENSE})",
                     fontsize=9)
        ax.set_xlabel("Expert (sorted by hotness ↓)", fontsize=7)
        ax.set_ylabel("Exec Time (ms)", fontsize=7)
        ax.tick_params(labelsize=6)

    fig.suptitle(
        "Per-Expert Execution Time (bar) vs. Hotness (blue line)\n"
        "Experts sorted by hotness (left=hottest). Red bars = longer execution.",
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "per_expert_time_by_layer.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


def plot_cold_vs_hot_time(avg_time, hotness, out_dir):
    """
    4-panel:
      [A] Time std within each layer (how much variance across experts)
      [B] Time ratio hot/cold per layer
      [C] Avg time: top-8 hot vs bottom-8 cold experts (per layer trend)
      [D] Overall distribution: hot vs cold experts time CDF
    """
    avg_h_per_expert = hotness.mean(axis=0)  # (64,)
    sorted_by_h      = np.argsort(avg_h_per_expert)
    cold8 = sorted_by_h[:8]
    hot8  = sorted_by_h[-8:]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # A: time std per layer
    ax = axes[0, 0]
    t_std  = avg_time.std(axis=1)
    t_mean = avg_time.mean(axis=1)
    ax.bar(range(N_MOE_LAYERS), t_std, color="steelblue")
    ax.set_xlabel("MoE Layer Index"); ax.set_ylabel("Std of Exec Time (ms)")
    ax.set_title("A. Exec Time Variance Across Experts per Layer\n(higher = more unequal)")

    # B: max/min time ratio per layer (hot/cold)
    ax = axes[0, 1]
    t_max   = avg_time.max(axis=1)
    t_min_p = np.array([avg_time[li, avg_time[li] > 0].min()
                        if (avg_time[li] > 0).any() else 1.0
                        for li in range(N_MOE_LAYERS)])
    ratio = t_max / np.maximum(t_min_p, 1e-6)
    ax.plot(range(N_MOE_LAYERS), ratio, marker="o", markersize=4, color="darkorange")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1, label="Perfect equality")
    ax.set_xlabel("MoE Layer Index"); ax.set_ylabel("Max / Min Exec Time Ratio")
    ax.set_title("B. Hottest / Coldest Expert Time Ratio per Layer")
    ax.legend()

    # C: avg time for hot8 vs cold8 experts across layers
    ax = axes[1, 0]
    hot_avg  = avg_time[:, hot8].mean(axis=1)
    cold_avg = avg_time[:, cold8].mean(axis=1)
    x = np.arange(N_MOE_LAYERS)
    ax.plot(x, hot_avg,  marker="s", markersize=3, color="#ee854a", label="Top-8 hot experts")
    ax.plot(x, cold_avg, marker="o", markersize=3, color="#4878d0", label="Bottom-8 cold experts")
    ax.fill_between(x, cold_avg, hot_avg, alpha=0.15, color="gray")
    ax.set_xlabel("MoE Layer Index"); ax.set_ylabel("Avg Exec Time (ms)")
    ax.set_title("C. Hot vs Cold Expert Avg Exec Time per Layer")
    ax.legend()

    # D: CDF of execution time, hot vs cold
    ax = axes[1, 1]
    hot_times  = avg_time[:, hot8].ravel()
    cold_times = avg_time[:, cold8].ravel()
    hot_times  = hot_times[hot_times > 0]
    cold_times = cold_times[cold_times > 0]
    for times, label, color in [
        (hot_times,  "Hot experts (top-8)",  "#ee854a"),
        (cold_times, "Cold experts (bot-8)", "#4878d0"),
    ]:
        sorted_t = np.sort(times)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax.plot(sorted_t, cdf, color=color, linewidth=2, label=label)
    ax.set_xlabel("Exec Time (ms)"); ax.set_ylabel("CDF")
    ax.set_title("D. Execution Time CDF: Hot vs Cold Experts")
    ax.legend()

    fig.suptitle(
        "Execution Time Inequality: Hot Experts Take Longer Than Cold Experts",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "per_expert_hot_vs_cold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    from vllm import LLM, SamplingParams

    print("[model] Loading vllm LLM (TP=2) …")
    llm = LLM(
        model=HF_MODEL,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=2,   # rank0 holds experts 0-31, rank1 holds 32-63
        max_model_len=2048,
        enforce_eager=True,
    )

    patch_model(llm)

    prompts = load_prompts(N_SAMPLES)
    params  = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=SEED)
    print(f"[infer] Running {len(prompts)} prompts …")
    llm.generate(prompts, params)
    print("[infer] Done.")

    avg_time, avg_time_all, hotness, counts = aggregate()

    np.save(os.path.join(OUTPUT_DIR, "per_expert_avg_time.npy"),  avg_time)
    np.save(os.path.join(OUTPUT_DIR, "per_expert_hotness.npy"),   hotness)
    np.save(os.path.join(OUTPUT_DIR, "per_expert_counts.npy"),    counts)

    # Print summary stats
    flat_t = avg_time[avg_time > 0]
    flat_h = hotness.ravel()[avg_time.ravel() > 0]
    r2 = np.corrcoef(flat_h, flat_t)[0, 1] ** 2
    print(f"[stat] Time range: {flat_t.min():.4f} – {flat_t.max():.4f} ms")
    print(f"[stat] R²(hotness, time) = {r2:.4f}")

    plot_time_vs_hotness(avg_time, hotness, OUTPUT_DIR)
    plot_layer_heatmap(avg_time, OUTPUT_DIR)
    plot_time_distribution_per_layer(avg_time, hotness, OUTPUT_DIR)
    plot_cold_vs_hot_time(avg_time, hotness, OUTPUT_DIR)

    print("[done] All figures saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
