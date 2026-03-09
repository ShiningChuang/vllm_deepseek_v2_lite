#!/usr/bin/env python3
"""Plot SAR baseline vs shape_aware comparison from 4-GPU experiment (conc=8)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
baseline = {
    "avg":    0.3071,
    "p50":    0.2014,
    "p90":    0.7394,
    "blocks_total": None,   # not shown directly
}
shape_aware = {
    "avg":    0.3623,
    "p50":    0.2550,
    "p90":    0.7929,
    "blocks_total": None,
}

# blocks eliminated: +145,672 out of baseline total
# baseline avg_blocks/record=69.609, applicable=14,772
# baseline total blocks = 69.609 * 14,772 ≈ 1,028,258
baseline_total_blocks  = round(69.609 * 14772)
eliminated             = 145672
shape_aware_total_blocks = baseline_total_blocks - eliminated

# ── Layout ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle(
    "Baseline vs Shape-Aware Routing (4 * GPU), conc=8",
    fontsize=13, y=1.02
)

labels  = ["Baseline", "Shape-Aware"]
colors  = ["#4c72b0", "#dd8452"]
metrics = [
    ("avg goodput",  [baseline["avg"],  shape_aware["avg"]],  "Goodput (real/padded tokens)"),
    ("p50 goodput",  [baseline["p50"],  shape_aware["p50"]],  "Goodput (real/padded tokens)"),
    ("p90 goodput",  [baseline["p90"],  shape_aware["p90"]],  "Goodput (real/padded tokens)"),
    ("GEMM blocks (total)",
     [baseline_total_blocks, shape_aware_total_blocks],
     "Total padded GEMM blocks"),
]

for ax, (title, vals, ylabel) in zip(axes, metrics):
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    # value labels
    for bar, v in zip(bars, vals):
        if v > 1000:
            label = f"{v/1e6:.3f}M"
        else:
            label = f"{v:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # delta annotation
    v0, v1 = vals
    if v0 > 0:
        delta_pct = (v1 - v0) / v0 * 100
        sign = "+" if delta_pct >= 0 else ""
        color = "#2ca02c" if delta_pct >= 0 else "#d62728"
        # blocks eliminated → lower is better → flip color
        if "block" in title.lower():
            color = "#2ca02c" if delta_pct <= 0 else "#d62728"
        ax.text(
            0.5, 0.96,
            f"{sign}{delta_pct:.1f}%",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13, fontweight="bold", color=color
        )

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
out = "/data/lxzhong_home/result/sar_4gpu_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
