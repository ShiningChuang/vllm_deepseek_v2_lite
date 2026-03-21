"""
Replot SM% experiment — natural SM% from nvprof sm_efficiency measurement.

Natural SM% = sm_efficiency at 100% MPS (hardware counter, no assumptions).
Interpretation: MPS restricted to X% starts to meaningfully limit the kernel
when X < natural SM%.

Measured values (nvprof --metrics sm_efficiency, V100S PCIe 32GB):
  n=16  : 62.8%   n=128: 71.2%   n=512: 82.0%   n=1024: 79.6%
"""
import math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/result/alloc"
DATA_PATH  = OUTPUT_DIR + "/sm_pct_results.npy"
SM_PCTS      = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
TOKEN_COUNTS = [16, 128, 512, 1024]
HIDDEN, INTER = 2048, 704
RIDGE = 115

results = np.load(DATA_PATH, allow_pickle=True).item()

# ── Measured natural SM% (nvprof sm_efficiency at 100% MPS) ──────────────────
# Meaning: at full GPU, GEMM occupies this fraction of SM time.
# MPS restriction below this value starts to degrade performance.
NAT_SM_EFF = {16: 62.8, 128: 71.2, 512: 82.0, 1024: 79.6}
nat_pct = [NAT_SM_EFF[n] for n in TOKEN_COUNTS]

colors  = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
markers = ["o", "s", "^", "D"]

fig, axes = plt.subplots(2, 2, figsize=(17, 12))


def annotate_natural(ax, xs, ys_dict):
    """
    Natural-SM% annotations using measured sm_efficiency values.
    Each curve gets a ★ at x = nat_sm_eff (distinct for every n → no overlap).
    A per-curve vertical dashed line drops from the ★ to the x-axis.
    """
    for n, color, nat in zip(TOKEN_COUNTS, colors, nat_pct):
        ys = ys_dict[n]
        yn = np.interp(nat, xs, ys)
        # vertical guide line
        ax.axvline(nat, color=color, linewidth=0.9, linestyle=":",
                   alpha=0.5, zorder=2)
        # ★ marker
        ax.plot(nat, yn, "*", color=color, markersize=15, zorder=7,
                markeredgecolor="white", markeredgewidth=0.9)
        # label just above ★
        ax.text(nat, yn, f" {nat:.0f}%",
                fontsize=7.5, color=color, fontweight="bold",
                va="bottom", ha="left")


# ── A: Absolute time ────────────────────────────────────────────────────────
ax = axes[0, 0]
ys_abs = {}
for n, color, mk, nat in zip(TOKEN_COUNTS, colors, markers, nat_pct):
    xs = list(SM_PCTS)
    ys = [results[n][p] * 1000 for p in xs]
    ys_abs[n] = ys
    regime = "BW-bound" if n < RIDGE else f"{n/RIDGE:.1f}×ridge"
    ax.plot(xs, ys, marker=mk, color=color, linewidth=2.2, markersize=7,
            label=f"n={n}  [{regime}]  ★nat={nat:.0f}% sm_eff")
ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
ax.set_ylabel("Execution time (μs)", fontsize=11)
ax.set_title("A. Absolute Execution Time vs SM Allocation", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
annotate_natural(ax, list(SM_PCTS), ys_abs)

# ── B: Speedup vs 5% ────────────────────────────────────────────────────────
ax = axes[0, 1]
ys_spd = {}
for n, color, mk in zip(TOKEN_COUNTS, colors, markers):
    xs = list(SM_PCTS)
    ys_raw = [results[n][p] for p in xs]
    base = ys_raw[0]
    spd = [base / y for y in ys_raw]
    ys_spd[n] = spd
    ax.plot(xs, spd, marker=mk, color=color, linewidth=2.2, markersize=7, label=f"n={n}")
xs_ideal = np.array(SM_PCTS, dtype=float)
ax.plot(xs_ideal, xs_ideal / SM_PCTS[0], "--", color="gray",
        linewidth=1.5, alpha=0.7, label="Ideal linear")
ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
ax.set_ylabel("Speedup (vs 5% baseline)", fontsize=11)
ax.set_title("B. Speedup vs SM%\nBW-bound: flat  |  Compute-bound: ≈ linear", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
annotate_natural(ax, list(SM_PCTS), ys_spd)

# ── C: Slowdown vs 100% ─────────────────────────────────────────────────────
ax = axes[1, 0]
ys_slow = {}
for n, color, mk in zip(TOKEN_COUNTS, colors, markers):
    xs = list(SM_PCTS)
    ys_raw = [results[n][p] for p in xs]
    t100 = results[n][100]
    slow = [y / t100 for y in ys_raw]
    ys_slow[n] = slow
    ax.plot(xs, slow, marker=mk, color=color, linewidth=2.2, markersize=7, label=f"n={n}")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="100% SM (no limit)")
xs_ideal2 = np.linspace(SM_PCTS[0], 100, 200)
ax.plot(xs_ideal2, 100 / xs_ideal2, ":", color="gray", linewidth=1.2,
        alpha=0.6, label="Ideal 1/SM% curve")
ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
ax.set_ylabel("Slowdown factor  (× 100%-SM time)", fontsize=11)
ax.set_title("C. Slowdown Relative to Full GPU", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0, 105)
annotate_natural(ax, list(SM_PCTS), ys_slow)

# ── D: Summary table ────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.axis("off")

col_labels = ["Token\ncount", "Regime", "Natural SM%\n(nvprof sm_eff)",
              "Meaning", "Effect of\nMPS < natural%"]
rows_data  = []
meanings = {
    16:   "63% of SM-time\nused at full GPU",
    128:  "71% of SM-time\nused at full GPU",
    512:  "82% of SM-time\nused at full GPU",
    1024: "80% of SM-time\nused at full GPU",
}
for n, nat in zip(TOKEN_COUNTS, nat_pct):
    regime = "BW-bound" if n < RIDGE else "Compute-bound"
    effect = "time ≈ flat\n(memory bottleneck)" if n < RIDGE else "time ∝ 1/MPS%"
    rows_data.append([f"n = {n}", regime, f"{nat:.0f}%",
                      meanings[n], effect])

table = ax.table(cellText=rows_data, colLabels=col_labels,
                 cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.4)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#BDBDBD")
    if r == 0:
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor("#FAFAFA")
    if r > 0 and c == 3:   # regime column
        cell.set_facecolor("#BBDEFB" if rows_data[r-1][3] == "BW-bound" else "#C8E6C9")
    if r > 0 and c == 2:   # natural SM% column
        cell.set_text_props(fontweight="bold",
                            color="slateblue" if nat_pct[r-1] < 100 else "#E91E63")

ax.set_title("D. Natural SM% Per Token Count\n"
             "(nvprof sm_efficiency at 100% MPS — hardware counter)", fontsize=10)

# Legend: natural SM% colors explanation
legend_elements = [
    Line2D([0], [0], marker="*", color=c, markersize=13,
           markeredgecolor="white", linestyle="None",
           label=f"★ n={n}  sm_eff={nat:.0f}%  (measured, nvprof)")
    for c, n, nat in zip(colors, TOKEN_COUNTS, nat_pct)
]
ax.legend(handles=legend_elements, loc="lower center", fontsize=9,
          title="★ = natural SM utilization at 100% MPS", title_fontsize=9,
          bbox_to_anchor=(0.5, -0.02), frameon=True)

fig.suptitle(
    "Expert FFN: Execution Time vs SM Allocation  (CUDA MPS experiment)\n"
    f"V100S PCIe 32GB  ·  HIDDEN={HIDDEN}, INTER={INTER}  ·  Ridge ≈ {RIDGE} tokens\n"
    "★ = natural sm_efficiency (nvprof hardware counter at 100% MPS, no tile-size assumptions)",
    fontsize=11, fontweight="bold"
)
fig.tight_layout()

path = OUTPUT_DIR + "/microbench_sm_pct.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[plot] {path}")
