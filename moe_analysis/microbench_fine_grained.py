# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Fine-grained token-count sweep to reveal the staircase (tile-boundary) pattern
in expert FFN execution time.

cuBLAS GEMM tiles the M dimension (= number of tokens) in chunks.
Each time n_tokens crosses a tile boundary, a new row of tiles is launched
and execution time jumps to the next step.

Sweep strategy:
  - Step=1 from 1 to 256  → see every individual step
  - Step=4 from 256 to 512 → confirm pattern continues
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Expert weight shape (V100S, TP=2 shard)
HIDDEN = 2048
INTER  = 704
DTYPE  = torch.float16
DEVICE = "cuda"

N_WARMUP = 30
N_REPEAT = 200   # more reps for fine-grained accuracy


def time_ffn(w13, w2, n_tokens):
    x = torch.randn(n_tokens, HIDDEN, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    for _ in range(N_WARMUP):
        o = F.linear(x, w13)
        g, u = o[:, :INTER], o[:, INTER:]
        _ = F.linear(F.silu(g) * u, w2)
    torch.cuda.synchronize()

    times = []
    for _ in range(N_REPEAT):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        o = F.linear(x, w13)
        g, u = o[:, :INTER], o[:, INTER:]
        _ = F.linear(F.silu(g) * u, w2)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.median(times)), float(np.std(times))


def detect_steps(token_counts, times):
    """
    Find token counts where time jumps significantly.
    A 'step' is defined as a jump > threshold × median_diff.
    Returns list of (token_boundary, time_before, time_after).
    """
    diffs = np.diff(times)
    threshold = np.percentile(np.abs(diffs), 90) * 2
    steps = []
    for i, d in enumerate(diffs):
        if d > threshold:
            steps.append((token_counts[i + 1], times[i], times[i + 1]))
    return steps


def main():
    torch.manual_seed(42)

    w13 = torch.randn(2 * INTER, HIDDEN, device=DEVICE, dtype=DTYPE)
    w2  = torch.randn(HIDDEN,    INTER,  device=DEVICE, dtype=DTYPE)

    # Fine sweep: step=1 for 1..256, step=4 for 256..512
    counts_fine   = list(range(1, 257, 1))
    counts_coarse = list(range(260, 513, 4))
    all_counts    = counts_fine + counts_coarse

    print(f"[bench] Sweeping {len(all_counts)} token counts (1→512, fine-grained)…")
    times_ms = []
    stds_ms  = []
    for i, n in enumerate(all_counts):
        t, s = time_ffn(w13, w2, n)
        times_ms.append(t)
        stds_ms.append(s)
        if n <= 64 or n % 32 == 0:
            print(f"  n={n:4d}  {t:.4f} ± {s:.4f} ms")

    times_ms = np.array(times_ms)
    stds_ms  = np.array(stds_ms)
    x        = np.array(all_counts)

    np.save(os.path.join(OUTPUT_DIR, "fine_token_counts.npy"), x)
    np.save(os.path.join(OUTPUT_DIR, "fine_times_ms.npy"),    times_ms)

    # Detect step boundaries
    steps = detect_steps(all_counts, times_ms)
    print(f"\n[steps] Detected {len(steps)} tile boundaries:")
    step_gaps = []
    for i, (boundary, t_before, t_after) in enumerate(steps):
        print(f"  boundary at n={boundary}: {t_before:.4f} → {t_after:.4f} ms  (+{(t_after-t_before)*1000:.1f} μs)")
        if i > 0:
            step_gaps.append(boundary - steps[i-1][0])
    if step_gaps:
        print(f"  Inferred tile size: {int(np.median(step_gaps))} tokens (median gap between steps)")

    # ── plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # A: Full range 1→512 with step markers
    ax = axes[0, 0]
    ax.plot(x, times_ms * 1000, color="steelblue", linewidth=0.9, alpha=0.8,
            label="Measured (median)")
    ax.fill_between(x,
                    (times_ms - stds_ms) * 1000,
                    (times_ms + stds_ms) * 1000,
                    alpha=0.2, color="steelblue", label="±1 std")
    for boundary, t_before, t_after in steps:
        ax.axvline(boundary, color="red", linewidth=0.8, alpha=0.6)
    if steps:
        ax.axvline(steps[0][0], color="red", linewidth=0.8, alpha=0.6,
                   label=f"Tile boundary (×{len(steps)} detected)")
    ax.set_xlabel("Tokens routed to expert", fontsize=11)
    ax.set_ylabel("Execution time (μs)", fontsize=11)
    ax.set_title("A. Full Range (1→512): Staircase Pattern", fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_minor_locator(MultipleLocator(16))
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.4)

    # B: Zoom in to 1→128 to see steps clearly
    ax = axes[0, 1]
    mask = x <= 128
    ax.plot(x[mask], times_ms[mask] * 1000, "o-", color="steelblue",
            linewidth=1.2, markersize=2.5, label="Measured")
    for boundary, t_before, t_after in steps:
        if boundary <= 128:
            ax.axvline(boundary, color="red", linewidth=1.2, alpha=0.8,
                       linestyle="--")
            ax.annotate(f"n={boundary}\n+{(t_after-t_before)*1000:.0f}μs",
                        xy=(boundary, t_after * 1000),
                        xytext=(boundary + 2, t_after * 1000 + 2),
                        fontsize=7.5, color="red")
    ax.set_xlabel("Tokens routed to expert", fontsize=11)
    ax.set_ylabel("Execution time (μs)", fontsize=11)
    ax.set_title("B. Zoom: 1→128, Step Boundaries Annotated", fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(16))
    ax.xaxis.set_minor_locator(MultipleLocator(4))
    ax.grid(which="minor", alpha=0.15)
    ax.grid(which="major", alpha=0.4)

    # C: Zoom into a single stair transition (around the first major step)
    if steps:
        # Find the clearest step: largest jump
        biggest = max(steps, key=lambda s: s[2] - s[1])
        center  = biggest[0]
        window  = 24
        lo, hi  = max(1, center - window), min(512, center + window)
        mask2   = (x >= lo) & (x <= hi)
        ax = axes[1, 0]
        ax.plot(x[mask2], times_ms[mask2] * 1000, "o-", color="steelblue",
                linewidth=1.5, markersize=5, label="Measured")
        ax.axvline(center, color="red", linewidth=1.8, linestyle="--",
                   label=f"Boundary n={center}")
        ax.axhline(biggest[1] * 1000, color="orange", linewidth=1.2,
                   linestyle=":", label=f"Before: {biggest[1]*1000:.1f} μs")
        ax.axhline(biggest[2] * 1000, color="green", linewidth=1.2,
                   linestyle=":", label=f"After:  {biggest[2]*1000:.1f} μs")
        ax.annotate("", xy=(center + 1, biggest[2] * 1000),
                    xytext=(center + 1, biggest[1] * 1000),
                    arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
        ax.text(center + 2, (biggest[1] + biggest[2]) / 2 * 1000,
                f"+{(biggest[2]-biggest[1])*1000:.1f} μs",
                color="purple", fontsize=10, fontweight="bold")
        ax.set_xlabel("Tokens routed to expert", fontsize=11)
        ax.set_ylabel("Execution time (μs)", fontsize=11)
        ax.set_title(f"C. Tile Boundary Zoom (n={lo}→{hi}): Largest Jump", fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.grid(alpha=0.3)

    # D: Histogram of time values — should cluster at discrete levels
    ax = axes[1, 1]
    ax.hist(times_ms * 1000, bins=80, color="steelblue", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    # Mark detected step levels
    if steps:
        levels = [steps[0][1] * 1000]
        for _, _, t_after in steps:
            levels.append(t_after * 1000)
        for i, lv in enumerate(levels):
            ax.axvline(lv, color="red", linewidth=1.5, linestyle="--",
                       label=f"Level {i+1}: {lv:.1f} μs" if i < 5 else None)
    ax.set_xlabel("Execution time (μs)", fontsize=11)
    ax.set_ylabel("Count (out of all n values)", fontsize=11)
    ax.set_title("D. Time Value Distribution\n"
                 "(discrete peaks confirm staircase, not continuous)", fontsize=11)
    ax.legend(fontsize=8)

    if step_gaps:
        tile_size = int(np.median(step_gaps))
        subtitle = (f"Inferred cuBLAS tile size in M-dim: {tile_size} tokens  "
                    f"| {len(steps)} steps detected in n=1→512")
    else:
        subtitle = "No clear steps detected — may need higher precision timing"

    fig.suptitle(
        f"Expert FFN Execution Time: Staircase Pattern from cuBLAS GEMM Tiling\n{subtitle}",
        fontsize=12, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "microbench_fine_grained.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")
    print("[done]")


if __name__ == "__main__":
    main()
