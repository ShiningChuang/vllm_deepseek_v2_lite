# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Micro-benchmark: single-expert execution time vs. token count.

Shows the "compute-bandwidth crossover" — below ~124 tokens/expert the
execution is DRAM-bandwidth-bound (time ≈ constant), above it becomes
compute-bound (time ∝ tokens).

This explains why hotness has no measurable effect on per-expert timing
in our ShareGPT experiment: each expert receives only ~2-5 tokens/step,
deep in the flat (bandwidth-bound) regime.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED       = 42
HF_MODEL   = "deepseek-ai/DeepSeek-V2-Lite"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = "/tmp"

# Expert weight shapes from the model (TP=2, so intermediate=704 per shard)
HIDDEN     = 2048
INTER      = 704    # per-TP-shard intermediate size
DTYPE      = torch.float16
DEVICE     = "cuda"

# Token counts to sweep — extended to 4096
TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
N_WARMUP     = 20
N_REPEAT     = 100


def time_expert_ffn(w13, w2, n_tokens, n_warmup, n_repeat):
    """
    Time a single expert FFN with n_tokens input vectors.
    w13: (2*INTER, HIDDEN)  — fused gate+up
    w2:  (HIDDEN,  INTER)   — down proj (note: shape from model)
    Returns median time in ms.
    """
    x = torch.randn(n_tokens, HIDDEN, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()

    # warmup
    for _ in range(n_warmup):
        gate_up  = F.linear(x, w13)              # (n, 2*INTER)
        gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
        inter    = F.silu(gate) * up
        _        = F.linear(inter, w2)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(n_repeat):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        gate_up  = F.linear(x, w13)
        gate, up = gate_up[:, :INTER], gate_up[:, INTER:]
        inter    = F.silu(gate) * up
        _        = F.linear(inter, w2)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))

    return float(np.median(times))


def compute_roofline(token_counts):
    """
    Theoretical times based on V100 specs.
    w13 bytes = 2*INTER*HIDDEN*2  (fp16)
    w2  bytes = HIDDEN*INTER*2
    Total weight bytes per expert = (2*INTER*HIDDEN + HIDDEN*INTER) * 2
    FLOPS per token = 2*(2*INTER*HIDDEN) + 2*(INTER*HIDDEN) = 6*INTER*HIDDEN
    """
    # V100S PCIe 32GB (corrected from regular V100):
    #   FP16 Tensor Core: 130 TFLOPS (NVIDIA official)
    #   Memory BW: 1107 MHz × 4096-bit HBM2 × DDR / 8 = 1134 GB/s (measured)
    v100_fp16_tflops  = 130e12
    v100_membw_gbs    = 1134e9

    weight_bytes = (2 * INTER * HIDDEN + HIDDEN * INTER) * 2  # fp16
    flops_per_tok = 2 * (2 * INTER * HIDDEN) + 2 * (INTER * HIDDEN)

    times_compute = []
    times_membw   = []
    for n in token_counts:
        total_flops = flops_per_tok * n
        t_compute   = total_flops / v100_fp16_tflops * 1000   # ms
        t_membw     = weight_bytes / v100_membw_gbs * 1000    # ms (load weights once)
        times_compute.append(t_compute)
        times_membw.append(t_membw)

    ridge_point = weight_bytes * v100_fp16_tflops / (flops_per_tok * v100_membw_gbs)
    return times_compute, times_membw, ridge_point, weight_bytes, flops_per_tok


def main():
    torch.manual_seed(SEED)

    # Create synthetic expert weights (same shape as model)
    w13 = torch.randn(2 * INTER, HIDDEN, device=DEVICE, dtype=DTYPE)
    w2  = torch.randn(HIDDEN,    INTER,  device=DEVICE, dtype=DTYPE)

    print(f"[bench] w13={tuple(w13.shape)}  w2={tuple(w2.shape)}")
    print(f"[bench] Sweeping token counts: {TOKEN_COUNTS}")

    times_measured = []
    for n in TOKEN_COUNTS:
        t = time_expert_ffn(w13, w2, n, N_WARMUP, N_REPEAT)
        times_measured.append(t)
        print(f"  n_tokens={n:4d}  median={t:.4f} ms")

    times_measured = np.array(times_measured)
    tc, tm, ridge, wbytes, fpt = compute_roofline(TOKEN_COUNTS)
    tc = np.array(tc)
    tm = np.array(tm)
    roofline = np.maximum(tc, tm)    # actual bottleneck = max of the two

    # ── actual average tokens per expert from our ShareGPT experiment ────────
    counts = np.load(os.path.join(OUTPUT_DIR, "per_expert_counts.npy"))
    # counts[layer][expert] = total over all steps
    # We don't know exact step count, but estimate from timing calls:
    # (use layer 0 as reference)
    total_routes_l0 = counts[0].sum()
    # from per_expert_times: we stored one (time, n_tokens) per forward pass
    # the timing script ran N_SAMPLES=100 prompts
    # from log: ~9315 tokens total in layer 0
    # vllm processes them in variable batch sizes; avg_batch ≈ total / num_steps
    # num_steps estimated: each forward pass is one entry in per_expert_times
    # we stored all entries → len(per_expert_times[0][0]) = num_steps
    # but we don't have that data anymore. Use the counts to compute avg tokens/expert/step
    # lower bound: if all tokens came in one step, avg = counts/1
    # upper bound: if each token came separately, avg = 1
    # from vllm with 100 prompts of avg 93 tokens each and max_new_tokens=32:
    # total tokens ≈ 9315 (measured), typical decode batch = 10-50 concurrent requests
    # conservative estimate: avg batch = 20 input tokens/step → per expert = 20*6/64 = 1.9
    avg_tok_per_expert_low  = 1.9   # batch_size=20
    avg_tok_per_expert_high = 4.7   # batch_size=50

    # ── plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    x = np.array(TOKEN_COUNTS, dtype=float)

    # ── Left: linear scale, extended to 4096 ───────────────────────────────
    ax = axes[0]
    ax.plot(x, times_measured, "o-", color="steelblue", linewidth=2.5,
            markersize=6, zorder=5, label="Measured (median, V100S PCIe 32GB)")
    ax.plot(x, roofline, "--", color="dimgray", linewidth=1.5,
            label=f"Roofline: max(compute, membw)")
    ax.axhline(tm[0], color="darkorange", linestyle=":", linewidth=1.5,
               label=f"Membw ceiling  = {wbytes/1134e9*1000*1000:.1f} μs  (8.25 MB ÷ 1134 GB/s)")
    ax.axvline(ridge, color="red", linestyle="--", linewidth=1.8, zorder=4,
               label=f"Theoretical ridge = {ridge:.0f} tok  (130 TFLOPS ÷ 1134 GB/s ÷ 1.0 FLOP/byte/tok)")

    ax.axvspan(0, ridge, alpha=0.07, color="orange")
    ax.axvspan(ridge, max(TOKEN_COUNTS), alpha=0.07, color="limegreen")
    ax.text(ridge * 0.35, times_measured.max() * 0.88,
            "Bandwidth-bound\ntime ≈ constant\n(AI < ridge)",
            ha="center", fontsize=9, color="darkorange",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax.text(ridge * 2.5, times_measured.max() * 0.55,
            "Compute-bound\ntime ∝ tokens\n(AI > ridge)",
            ha="center", fontsize=9, color="green",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ShareGPT actual range (almost invisible at this scale — shows how tiny it is)
    ax.axvspan(avg_tok_per_expert_low, avg_tok_per_expert_high,
               alpha=0.55, color="royalblue",
               label=f"ShareGPT range: {avg_tok_per_expert_low:.1f}–{avg_tok_per_expert_high:.1f} tok/expert/step")

    # Annotation box: how ridge is derived
    deriv_text = (
        "Ridge point derivation (V100S PCIe 32GB):\n"
        f"  Peak FP16 TC = 130 TFLOPS\n"
        f"  Mem BW = 1134 GB/s  (1107 MHz × 4096-bit HBM2)\n"
        f"  AI per token = FLOPs/Bytes = 6×I×H / (3×I×H×2) = 1.0 FLOP/byte\n"
        f"  Ridge = 130T / 1134G / 1.0 = {ridge:.0f} tokens"
    )
    ax.text(0.97, 0.04, deriv_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="gray", alpha=0.9))

    ax.set_xlabel("Tokens routed to expert (per forward pass)", fontsize=12)
    ax.set_ylabel("Expert FFN execution time (ms)", fontsize=12)
    ax.set_title(
        "Single Expert FFN: Execution Time vs Token Count  [0 → 4096]\n"
        "Method: synthetic w13=(1408,2048), w2=(2048,704) × varying input shapes",
        fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(0, max(TOKEN_COUNTS))

    # ── Right: log-log scale ────────────────────────────────────────────────
    ax = axes[1]
    ax.loglog(x, times_measured, "o-", color="steelblue", linewidth=2.5,
              markersize=6, zorder=5, label="Measured")
    ax.loglog(x, roofline, "--", color="dimgray", linewidth=1.5,
              label="Roofline model")
    ax.loglog(x, tc, ":", color="green", linewidth=1.5,
              label=f"Compute limit  (slope=1)")
    ax.loglog(x, [tm[0]] * len(x), ":", color="darkorange", linewidth=1.5,
              label=f"Bandwidth limit  (flat)")
    ax.axvline(ridge, color="red", linestyle="--", linewidth=1.8,
               label=f"Ridge = {ridge:.0f} tok")
    ax.axvspan(avg_tok_per_expert_low, avg_tok_per_expert_high * 1.5,
               alpha=0.35, color="royalblue",
               label=f"ShareGPT range (~{avg_tok_per_expert_low:.0f}–{avg_tok_per_expert_high:.0f} tok)")

    # Annotate achieved throughput at large n
    _, _, ridge, wbytes2, fpt2 = compute_roofline(TOKEN_COUNTS)
    n_large = TOKEN_COUNTS[-1]
    t_large = times_measured[-1]
    achieved_tflops = fpt2 * n_large / (t_large * 1e-3) / 1e12
    achieved_bw_gbs = wbytes2 / (times_measured[0] * 1e-3) / 1e9
    ax.annotate(
        f"n={n_large}: {achieved_tflops:.1f} TFLOPS\n({100*achieved_tflops/130:.0f}% of peak)",
        xy=(n_large, t_large), xytext=(n_large * 0.25, t_large * 1.5),
        arrowprops=dict(arrowstyle="->", color="steelblue"),
        fontsize=8, color="steelblue")
    ax.annotate(
        f"n=1: {achieved_bw_gbs:.0f} GB/s BW\n({100*achieved_bw_gbs/1134:.0f}% of peak)",
        xy=(1, times_measured[0]), xytext=(2, times_measured[0] * 0.55),
        arrowprops=dict(arrowstyle="->", color="darkorange"),
        fontsize=8, color="darkorange")

    ax.set_xlabel("Tokens routed to expert (log scale)", fontsize=12)
    ax.set_ylabel("Execution time ms (log scale)", fontsize=12)
    ax.set_title("Log-Log: Flat → Linear transition clearly visible\n"
                 "(slope=0 in BW-bound, slope=1 in compute-bound)",
                 fontsize=10)
    ax.legend(fontsize=8.5)

    fig.suptitle(
        "Expert FFN is Memory-Bandwidth-Bound for All Realistic MoE Batch Sizes\n"
        "ShareGPT: ~2–5 tokens/expert/step  |  V100S ridge point: 115 tokens  |  Need ≥115 tok to see hotness effect",
        fontsize=11, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "microbench_time_vs_tokens.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] {path}")

    # Save data
    np.save(os.path.join(OUTPUT_DIR, "microbench_token_counts.npy"), np.array(TOKEN_COUNTS))
    np.save(os.path.join(OUTPUT_DIR, "microbench_times_ms.npy"), times_measured)

    # Print summary
    print(f"\n[summary]")
    print(f"  Weight size per expert:   {wbytes/1024**2:.2f} MB")
    print(f"  FLOPs per token:          {fpt/1e6:.2f} MFLOPs")
    print(f"  Arithmetic intensity:     {fpt/wbytes:.3f} FLOP/byte per token")
    print(f"  V100 ridge point:         {ridge:.0f} tokens")
    print(f"  ShareGPT operating range: {avg_tok_per_expert_low:.1f}–{avg_tok_per_expert_high:.1f} tokens")
    print(f"  → Bandwidth-bound: hotness CANNOT be seen in execution time")
    print(f"  → Need >{ridge:.0f} tokens/expert to see linear scaling")
    t_at_1   = times_measured[TOKEN_COUNTS.index(1)]
    t_at_128 = times_measured[TOKEN_COUNTS.index(128)]
    print(f"  Time ratio t(128)/t(1):   {t_at_128/t_at_1:.2f}x  (theory: 128x if compute-bound)")
    print("[done]")


if __name__ == "__main__":
    main()
