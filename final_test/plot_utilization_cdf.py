#!/usr/bin/env python3
"""
Plot per-expert last-block utilization CDF for Prefill/Mixed and Decode phases.

Last-block utilization for an expert with t tokens (BLOCK_SIZE_M = B):
  - if t % B == 0: utilization = 1.0   (last block is fully packed)
  - else:          utilization = (t % B) / B

Usage:
  python3 plot_utilization_cdf.py \
      --result-dir /data/lxzhong_home/result \
      --run-ids "23,24,25" \
      --labels "conc=1,conc=4,conc=8" \
      --concurrencies "1,4,8" \
      --output-prefix "dailymail" \
      --title-suffix "DailyMail Workload"
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLOCK_SIZE_M = 64


def last_block_util(t: int) -> float:
    if t <= 0:
        return float("nan")
    rem = t % BLOCK_SIZE_M
    return 1.0 if rem == 0 else rem / BLOCK_SIZE_M


def load_phase_utils(path: str, concurrency: int):
    """Return (prefill_utils, decode_utils) lists of per-expert utilizations."""
    prefill_utils = []
    decode_utils  = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("where") != "after_align_before_gemm1":
                continue
            tic = rec["tokens_in_chunk"]
            tpe = rec["tokens_per_expert_assign"]
            utils = [last_block_util(t) for t in tpe if t > 0]
            if not utils:
                continue
            if tic <= concurrency:
                decode_utils.extend(utils)
            else:
                prefill_utils.extend(utils)
    return prefill_utils, decode_utils


def plot_cdf(ax, data_by_label: dict, title: str, xlabel: str = "Last-block utilization"):
    """Plot CDF lines for each label + an 'all runs' line."""
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    all_vals = []

    for (label, vals), color in zip(data_by_label.items(), colors):
        if not vals:
            continue
        s = sorted(vals)
        cdf = np.arange(1, len(s) + 1) / len(s) * 100
        ax.plot(s, cdf, label=f"{label} (N={len(s):,})",
                color=color, linewidth=1.8)
        all_vals.extend(vals)

    # "all runs" composite line
    if all_vals:
        s = sorted(all_vals)
        cdf = np.arange(1, len(s) + 1) / len(s) * 100
        ax.plot(s, cdf, label=f"all runs (N={len(all_vals):,})",
                color="black", linewidth=2.0, linestyle="--")

        # Annotate % below 0.3
        pct_below = sum(1 for v in all_vals if v < 0.3) / len(all_vals) * 100
        ax.axvline(0.3, color="gray", linewidth=0.8, linestyle=":")
        ax.text(0.31, 10, f"{pct_below:.1f}% < 0.3",
                fontsize=9, color="gray", va="bottom")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="/data/lxzhong_home/result")
    parser.add_argument("--run-ids",       required=True,
                        help="e.g. '23,24,25'")
    parser.add_argument("--labels",        required=True,
                        help="e.g. 'conc=1,conc=4,conc=8'")
    parser.add_argument("--concurrencies", required=True,
                        help="e.g. '1,4,8'")
    parser.add_argument("--output-prefix", default="workload",
                        help="Prefix for output filenames.")
    parser.add_argument("--title-suffix",  default="",
                        help="Extra text appended to plot titles.")
    args = parser.parse_args()

    run_ids   = [x.strip() for x in args.run_ids.split(",")]
    labels    = [x.strip() for x in args.labels.split(",")]
    concs     = [int(x.strip()) for x in args.concurrencies.split(",")]

    prefill_by_label: dict = {}
    decode_by_label:  dict = {}

    for run_id, label, conc in zip(run_ids, labels, concs):
        path = os.path.join(args.result_dir, f"moe_shapes_run{run_id}_rank0.jsonl")
        if not os.path.exists(path):
            print(f"[warn] not found: {path}")
            prefill_by_label[label] = []
            decode_by_label[label]  = []
            continue
        p_utils, d_utils = load_phase_utils(path, conc)
        prefill_by_label[label] = p_utils
        decode_by_label[label]  = d_utils
        print(f"  {label}: prefill/mixed={len(p_utils):,}  decode={len(d_utils):,}")

    suffix = f"  —  {args.title_suffix}" if args.title_suffix else ""

    # ── Prefill/Mixed CDF ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_cdf(
        ax,
        prefill_by_label,
        title=f"Expert Last-Block Utilization CDF\nPrefill/Mixed Phase{suffix}",
    )
    plt.tight_layout()
    out_prefill = os.path.join(
        args.result_dir, f"expert_utilization_cdf_{args.output_prefix}.png"
    )
    plt.savefig(out_prefill, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_prefill}")

    # ── Decode CDF ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_cdf(
        ax,
        decode_by_label,
        title=f"Expert Last-Block Utilization CDF\nDecode Phase{suffix}",
    )
    plt.tight_layout()
    out_decode = os.path.join(
        args.result_dir, f"expert_utilization_cdf_decode_{args.output_prefix}.png"
    )
    plt.savefig(out_decode, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_decode}")


if __name__ == "__main__":
    main()
