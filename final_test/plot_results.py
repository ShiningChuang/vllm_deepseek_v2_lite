#!/usr/bin/env python3
"""Generate plots from MoE phase analysis logs."""
import json
import os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULT_DIR = "/data/lxzhong_home/result"
OUT_DIR = RESULT_DIR
RUNS = [
    {"run_id": "13", "concurrency": 1, "label": "conc=1"},
    {"run_id": "14", "concurrency": 4, "label": "conc=4"},
    {"run_id": "15", "concurrency": 8, "label": "conc=8"},
]


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("where") == "after_align_before_gemm1":
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def goodput(rec):
    real = sum(rec["tokens_per_expert_assign"])
    padded = rec["num_tokens_post_padded"]
    return real / padded if padded > 0 else float("nan")


# Load data
all_data = []
for run in RUNS:
    path = os.path.join(RESULT_DIR, f"moe_shapes_run{run['run_id']}_rank0.jsonl")
    records = load_jsonl(path)
    conc = run["concurrency"]
    decode = [r for r in records if r["tokens_in_chunk"] <= conc]
    other  = [r for r in records if r["tokens_in_chunk"] >  conc]
    total  = len(records)
    all_data.append({
        "label": run["label"],
        "concurrency": conc,
        "total": total,
        "decode_pct": 100 * len(decode) / total,
        "other_pct":  100 * len(other)  / total,
        "decode_goodput": np.nanmean([goodput(r) for r in decode]),
        "other_goodput":  np.nanmean([goodput(r) for r in other]),
        "decode_avg_tic": np.mean([r["tokens_in_chunk"] for r in decode]) if decode else 0,
        "other_avg_tic":  np.mean([r["tokens_in_chunk"] for r in other])  if other  else 0,
        "records": records,
        "conc": conc,
    })

labels = [d["label"] for d in all_data]
x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("MoE Phase Analysis — ShareGPT Workload (DeepSeek-V2-Lite)", fontsize=13, y=1.02)

# ── Plot 1: Phase proportion ──────────────────────────────────────────────────
ax = axes[0]
decode_pcts = [d["decode_pct"] for d in all_data]
other_pcts  = [d["other_pct"]  for d in all_data]
bars1 = ax.bar(x, decode_pcts, 0.5, label="Decode", color="#4c72b0")
bars2 = ax.bar(x, other_pcts,  0.5, bottom=decode_pcts, label="Prefill/Mixed", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("% of MoE forward passes")
ax.set_ylim(0, 105)
ax.set_title("Phase Proportion")
ax.legend(loc="upper right")
for bar, pct in zip(bars1, decode_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, pct/2,
            f"{pct:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
for bar, d_pct, o_pct in zip(bars2, decode_pcts, other_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, d_pct + o_pct/2,
            f"{o_pct:.1f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")

# ── Plot 2: Goodput comparison ────────────────────────────────────────────────
ax = axes[1]
w = 0.35
decode_gp = [d["decode_goodput"] for d in all_data]
other_gp  = [d["other_goodput"]  for d in all_data]
b1 = ax.bar(x - w/2, decode_gp, w, label="Decode",        color="#4c72b0")
b2 = ax.bar(x + w/2, other_gp,  w, label="Prefill/Mixed", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Goodput  (real tokens / padded tokens)")
ax.set_ylim(0, 0.45)
ax.set_title("Padding Goodput\n(higher = less waste)")
ax.legend()
for b, v in zip(b1, decode_gp):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for b, v in zip(b2, other_gp):
    ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# ── Plot 3: tokens_in_chunk histogram (log-scale) ────────────────────────────
ax = axes[2]
bucket_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
colors = ["#4c72b0", "#dd8452", "#55a868"]
for di, d in enumerate(all_data):
    conc = d["conc"]
    records = d["records"]
    total = d["total"]
    counts = []
    for i, hi in enumerate(bucket_edges):
        lo = bucket_edges[i-1]+1 if i > 0 else 1
        c = sum(1 for r in records if lo <= r["tokens_in_chunk"] <= hi)
        counts.append(100 * c / total)
    xi = np.arange(len(bucket_edges))
    offset = (di - 1) * 0.25
    ax.bar(xi + offset, counts, 0.25, label=d["label"], color=colors[di], alpha=0.85)

bucket_labels = ["1","2","4","8","16","32","64","128","256","512","1K","2K","4K"]
ax.set_xticks(np.arange(len(bucket_edges)))
ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.4g}%"))
ax.set_ylabel("% of MoE records (log scale)")
ax.set_xlabel("tokens_in_chunk bucket")
ax.set_title("tokens_in_chunk Distribution")
ax.legend()
ax.axvline(x=2.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(2.6, ax.get_ylim()[1]*0.6, "decode | prefill\nthreshold varies", fontsize=7, color="gray")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "phase_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
