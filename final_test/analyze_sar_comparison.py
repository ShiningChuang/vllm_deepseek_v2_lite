#!/usr/bin/env python3
"""
Analyze SAR=0 vs SAR=1 goodput for Prefill/Mixed phase only.

Filters out decode records (tokens_in_chunk <= concurrency).
Measures padding goodput = sum(tokens_per_expert_assign) / num_tokens_post_padded.
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLOCK_SIZE_M = 64


def load_prefill_records(path: str, concurrency: int):
    """Load Prefill/Mixed records (tokens_in_chunk > concurrency)."""
    goodputs = []
    block_savings = []
    n_total = 0
    n_decode_skipped = 0

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
            n_total += 1

            tic = rec["tokens_in_chunk"]
            if tic <= concurrency:
                n_decode_skipped += 1
                continue

            tpe = rec["tokens_per_expert_assign"]
            real = sum(tpe)
            padded = rec["num_tokens_post_padded"]
            if padded > 0:
                goodputs.append(real / padded)

            # Block consolidation: current blocks vs minimum possible
            cur_blocks = sum(
                math.ceil(t / BLOCK_SIZE_M) for t in tpe if t > 0
            )
            total_tokens = sum(tpe)
            min_blocks = math.ceil(total_tokens / BLOCK_SIZE_M) if total_tokens else 0
            block_savings.append(cur_blocks - min_blocks)

    return goodputs, block_savings, n_total, n_decode_skipped


def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return s[idx]


def print_stats(label, goodputs, block_savings):
    if not goodputs:
        print(f"  [{label}] No records.")
        return
    avg = sum(goodputs) / len(goodputs)
    s = sorted(goodputs)
    print(
        f"  [{label}] N={len(goodputs):,}  "
        f"avg={avg:.4f}  "
        f"p10={percentile(s,10):.4f}  "
        f"p50={percentile(s,50):.4f}  "
        f"p90={percentile(s,90):.4f}  "
        f"min={s[0]:.4f}  max={s[-1]:.4f}"
    )
    if block_savings:
        reducible = sum(1 for s in block_savings if s > 0)
        total_sav = sum(block_savings)
        total_cur = sum(
            math.ceil(g * BLOCK_SIZE_M / 1) for g in goodputs  # rough estimate
        )
        print(
            f"  [{label}] block_savings: "
            f"reducible={reducible:,}/{len(block_savings):,} "
            f"({reducible/len(block_savings)*100:.1f}%)  "
            f"total_saved={total_sav:,} blocks"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="SAR=0 JSONL file")
    parser.add_argument("--sar", required=True, help="SAR=1 JSONL file")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Decode threshold (tokens_in_chunk <= this → skip)")
    parser.add_argument("--output", default="sar_comparison_conc4.png")
    args = parser.parse_args()

    print(f"\nFilter: Prefill/Mixed only (tokens_in_chunk > {args.concurrency})")
    print(f"BLOCK_SIZE_M = {BLOCK_SIZE_M}")

    g0, bs0, nt0, nd0 = load_prefill_records(args.baseline, args.concurrency)
    g1, bs1, nt1, nd1 = load_prefill_records(args.sar, args.concurrency)

    print(f"\n{'='*65}")
    print(f"File: {args.baseline}  (SAR=0 baseline)")
    print(f"  Total logged: {nt0:,}  |  Decode skipped: {nd0:,}  |  Prefill/Mixed: {len(g0):,}")
    print_stats("SAR=0", g0, bs0)

    print(f"\n{'='*65}")
    print(f"File: {args.sar}  (SAR=1 shape-aware)")
    print(f"  Total logged: {nt1:,}  |  Decode skipped: {nd1:,}  |  Prefill/Mixed: {len(g1):,}")
    print_stats("SAR=1", g1, bs1)

    print(f"\n{'='*65}")
    print("Comparison: SAR=1 vs SAR=0")
    if g0 and g1:
        avg0 = sum(g0) / len(g0)
        avg1 = sum(g1) / len(g1)
        delta = avg1 - avg0
        rel = delta / avg0 * 100 if avg0 else float("nan")
        print(f"  avg goodput:  SAR=0={avg0:.4f}  SAR=1={avg1:.4f}")
        print(f"  delta: {delta:+.4f}  ({rel:+.2f}%)")
        if delta > 1e-6:
            print("  => SAR IMPROVES padding goodput ✓")
        elif delta < -1e-6:
            print("  => SAR HURTS padding goodput ✗")
        else:
            print("  => No measurable difference")

        # Record count sanity check
        diff_pct = abs(len(g0) - len(g1)) / max(len(g0), len(g1)) * 100
        print(f"\n  Record count: SAR=0={len(g0):,}  SAR=1={len(g1):,}  "
              f"diff={diff_pct:.1f}% (< 5% is healthy with fixed seed)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"SAR=0 vs SAR=1 — Prefill/Mixed phase (conc={args.concurrency})\n"
        f"[seed=42, {args.concurrency} concurrent requests, Prefill/Mixed only]",
        fontsize=12
    )

    # ── Panel 1: average goodput bar chart ───────────────────────────────────
    ax = axes[0]
    labels = ["SAR=0\n(baseline)", "SAR=1\n(shape-aware)"]
    avgs = [sum(g) / len(g) if g else 0 for g in [g0, g1]]
    colors = ["#4c72b0", "#dd8452"]
    bars = ax.bar(labels, avgs, color=colors, width=0.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 0.003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    if avgs[0] > 0:
        rel = (avgs[1] - avgs[0]) / avgs[0] * 100
        ax.set_title(
            f"Avg Padding Goodput\n"
            f"(SAR=1 vs SAR=0: {rel:+.2f}%)",
            fontsize=11
        )
    else:
        ax.set_title("Avg Padding Goodput", fontsize=11)
    ax.set_ylabel("Goodput  (real tokens / padded tokens)")
    ax.set_ylim(0, max(avgs) * 1.25 if avgs else 1)

    # ── Panel 2: CDF of goodput ───────────────────────────────────────────────
    ax = axes[1]
    for goodputs, label, color in zip([g0, g1], ["SAR=0", "SAR=1"], colors):
        if not goodputs:
            continue
        s = sorted(goodputs)
        cdf = np.arange(1, len(s) + 1) / len(s) * 100
        ax.plot(s, cdf, label=f"{label} (N={len(s):,})", color=color, linewidth=1.8)
    ax.set_xlabel("Goodput (real / padded tokens)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Goodput CDF — Prefill/Mixed Phase")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {args.output}")


if __name__ == "__main__":
    main()
