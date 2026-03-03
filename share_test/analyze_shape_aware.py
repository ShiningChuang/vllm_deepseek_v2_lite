#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Analyze shape-aware routing effectiveness.

Reads moe_shapes_run{N}_rank{R}.jsonl files and reports padding goodput
for cases where shape-aware routing is actually applicable, filtering out:

  1. Records where ALL non-zero token counts are exact multiples of
     BLOCK_SIZE_M  →  no padding savings are possible.
     Example: [8192, 0, 0, 8192, 0, 0, ...]

  2. Records where the total token count is too small (< --min-tokens).
     Example: [0, 0, 0, ..., 8, 0, 1, 2, ...]

Usage:
    # Single file (baseline)
    python analyze_shape_aware.py /tmp/moe_shapes_run1_rank0.jsonl

    # Compare baseline vs shape-aware
    python analyze_shape_aware.py \\
        /tmp/moe_shapes_run1_rank0.jsonl \\
        /tmp/moe_shapes_run2_rank0.jsonl \\
        --labels baseline shape_aware
"""

import argparse
import json
import sys
from pathlib import Path


BLOCK_SIZE_M = 64
MIN_TOKENS = 64  # records with fewer total tokens are excluded


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _all_multiples(tokens_per_expert: list[int], block_size: int) -> bool:
    """Return True if every non-zero count is already a multiple of block_size."""
    non_zero = [t for t in tokens_per_expert if t > 0]
    if not non_zero:
        return True  # empty → trivially aligned, shape-aware does nothing
    return all(t % block_size == 0 for t in non_zero)


def _too_few_tokens(tokens_per_expert: list[int], min_tokens: int) -> bool:
    return sum(tokens_per_expert) < min_tokens


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_records(jsonl_path: str,
                 block_size: int = BLOCK_SIZE_M,
                 min_tokens: int = MIN_TOKENS
                 ) -> tuple[list[dict], dict]:
    """
    Load and filter records from a JSONL file.

    Returns:
        (applicable_records, stats_dict)
    """
    stats = dict(total=0, skipped_mult=0, skipped_few=0, applicable=0)
    applicable = []

    with open(jsonl_path) as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            if rec.get("where") != "after_align_before_gemm1":
                continue
            stats["total"] += 1

            tpe: list[int] = rec["tokens_per_expert_assign"]

            if _too_few_tokens(tpe, min_tokens):
                stats["skipped_few"] += 1
                continue

            if _all_multiples(tpe, block_size):
                stats["skipped_mult"] += 1
                continue

            applicable.append(rec)

    stats["applicable"] = len(applicable)
    return applicable, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_goodputs(records: list[dict]) -> list[float]:
    """padding goodput = real_tokens / post_padded_tokens for each record."""
    result = []
    for rec in records:
        real = sum(rec["tokens_per_expert_assign"])
        padded = rec["num_tokens_post_padded"]
        if padded > 0:
            result.append(real / padded)
    return result


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the p-th percentile (0‥100) from a pre-sorted list."""
    if not sorted_values:
        return float("nan")
    idx = max(0, min(len(sorted_values) - 1, int(len(sorted_values) * pct / 100)))
    return sorted_values[idx]


def print_stats(goodputs: list[float], label: str = "") -> None:
    if not goodputs:
        print(f"  [{label}] No applicable records.")
        return

    s = sorted(goodputs)
    n = len(s)
    avg = sum(s) / n
    prefix = f"[{label}] " if label else ""
    print(
        f"  {prefix}N={n:,}  "
        f"avg={avg:.4f}  "
        f"p10={_percentile(s,10):.4f}  "
        f"p50={_percentile(s,50):.4f}  "
        f"p90={_percentile(s,90):.4f}  "
        f"min={s[0]:.4f}  max={s[-1]:.4f}"
    )


def compute_block_consolidation(records: list[dict],
                                block_size: int = BLOCK_SIZE_M) -> dict:
    """
    Analyse block consolidation potential (or realised savings).

    For each record:
        current_blocks = Σ ceil(t / block_size)  for every expert with t > 0
        min_blocks     = ceil(Σ t / block_size)   (lower bound: all tokens in one batch)
        savings        = current_blocks - min_blocks  (≥ 0)

    Example: expert_A=31, expert_B=32  →  current=2, min=1, savings=1
             shape-aware routes all 63 to one expert → current=1, savings=0

    Returns a dict with aggregate statistics.
    """
    if not records:
        return {}

    current_list: list[int] = []
    savings_list: list[int] = []

    for rec in records:
        tpe: list[int] = rec["tokens_per_expert_assign"]
        total_tokens = sum(tpe)
        cur = sum((t + block_size - 1) // block_size for t in tpe if t > 0)
        mn  = (total_tokens + block_size - 1) // block_size if total_tokens else 0
        current_list.append(cur)
        savings_list.append(cur - mn)

    n = len(records)
    total_cur  = sum(current_list)
    total_sav  = sum(savings_list)
    reducible  = [s for s in savings_list if s > 0]

    # distribution: how many records save exactly k blocks
    max_sav = max(savings_list) if savings_list else 0
    dist: dict[int, int] = {}
    for s in savings_list:
        dist[s] = dist.get(s, 0) + 1

    return dict(
        n=n,
        reducible_count=len(reducible),
        reducible_pct=len(reducible) / n * 100 if n else 0.0,
        avg_current_blocks=total_cur / n,
        total_blocks=total_cur,
        total_savings=total_sav,
        overall_savings_pct=total_sav / total_cur * 100 if total_cur else 0.0,
        avg_savings_per_reducible=sum(reducible) / len(reducible) if reducible else 0.0,
        savings_dist=dist,
        max_savings=max_sav,
    )


def print_block_consolidation_stats(bs: dict, label: str = "") -> None:
    """Print block consolidation analysis."""
    if not bs:
        print(f"  [{label}] No data.")
        return
    prefix = f"[{label}] " if label else ""
    n = bs["n"]
    print(
        f"  {prefix}Reducible records  : "
        f"{bs['reducible_count']:,} / {n:,}  ({bs['reducible_pct']:.1f}%)\n"
        f"  {prefix}Avg blocks/record  : {bs['avg_current_blocks']:.3f}\n"
        f"  {prefix}Block savings      : {bs['total_savings']:,} blocks saved in total  "
        f"({bs['overall_savings_pct']:.2f}% of all blocks)\n"
        f"  {prefix}Avg saved/reducible: {bs['avg_savings_per_reducible']:.3f} blocks"
    )
    # savings distribution (skip 0)
    dist = bs["savings_dist"]
    if bs["max_savings"] > 0:
        print(f"  {prefix}Savings distribution (blocks saved → record count):")
        for k in range(1, bs["max_savings"] + 1):
            cnt = dist.get(k, 0)
            if cnt == 0:
                continue
            bar = "#" * min(40, int(cnt / n * 400))  # scale to max 40 chars at 10%
            print(f"    save={k:3d}  {bar}  {cnt:,}  ({cnt/n*100:.1f}%)")


def print_histogram(goodputs: list[float],
                    label: str = "",
                    n_bins: int = 10) -> None:
    if not goodputs:
        return
    lo, hi = min(goodputs), max(goodputs)
    if lo == hi:
        print(f"  [{label}] All values = {lo:.4f}")
        return
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for v in goodputs:
        idx = min(n_bins - 1, int((v - lo) / width))
        bins[idx] += 1
    bar_scale = 40 / max(bins)
    print(f"\n  Goodput histogram [{label}]:")
    for i, count in enumerate(bins):
        lo_b = lo + i * width
        hi_b = lo_b + width
        bar = "#" * int(count * bar_scale)
        print(f"    [{lo_b:.3f}, {hi_b:.3f})  {bar}  {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze shape-aware routing effectiveness from MoE shape logs"
    )
    parser.add_argument(
        "files", nargs="+",
        help="One or two JSONL log files to analyse"
    )
    parser.add_argument(
        "--block-size", type=int, default=BLOCK_SIZE_M,
        help=f"GEMM tile size M (default: {BLOCK_SIZE_M})"
    )
    parser.add_argument(
        "--min-tokens", type=int, default=MIN_TOKENS,
        help=f"Min total tokens per record to include (default: {MIN_TOKENS})"
    )
    parser.add_argument(
        "--labels", nargs="+",
        help="Human-readable labels for each file"
    )
    parser.add_argument(
        "--hist", action="store_true",
        help="Print a goodput histogram for each file"
    )
    args = parser.parse_args()

    if len(args.files) > 2:
        print("ERROR: at most 2 files supported for comparison", file=sys.stderr)
        sys.exit(1)

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) < len(args.files):
        labels += [Path(f).stem for f in args.files[len(labels):]]

    all_goodputs: list[list[float]] = []
    all_block_stats: list[dict] = []

    for path, label in zip(args.files, labels):
        records, stats = load_records(path, args.block_size, args.min_tokens)
        goodputs = compute_goodputs(records)
        block_stats = compute_block_consolidation(records, args.block_size)
        all_goodputs.append(goodputs)
        all_block_stats.append(block_stats)

        print(f"\n{'='*65}")
        print(f"File : {path}")
        print(f"Label: {label}")
        print(f"  Total records logged       : {stats['total']:,}")
        print(f"  Skipped – all multiples    : {stats['skipped_mult']:,}  "
              f"(block_size={args.block_size})")
        print(f"  Skipped – too few tokens   : {stats['skipped_few']:,}  "
              f"(< {args.min_tokens} total tokens)")
        print(f"  Applicable records         : {stats['applicable']:,}")
        print()
        print(f"  -- Padding goodput --")
        print_stats(goodputs, label)
        if args.hist:
            print_histogram(goodputs, label)
        print(f"\n  -- Block consolidation (31+32→1-block optimisation) --")
        print_block_consolidation_stats(block_stats, label)

    # --- comparison (two files) ---
    if len(all_goodputs) == 2:
        g0, g1 = all_goodputs
        bs0, bs1 = all_block_stats

        print(f"\n{'='*65}")
        print(f"Comparison: [{labels[1]}] vs [{labels[0]}]")

        # goodput comparison
        if g0 and g1:
            avg0 = sum(g0) / len(g0)
            avg1 = sum(g1) / len(g1)
            delta = avg1 - avg0
            rel = delta / avg0 * 100 if avg0 else float("nan")
            print(f"\n  [Goodput]")
            print(f"  avg goodput delta : {delta:+.4f}  ({rel:+.2f}%)")
            if delta > 1e-6:
                print("  => shape-aware routing IMPROVES padding goodput ✓")
            elif delta < -1e-6:
                print("  => shape-aware routing HURTS padding goodput ✗")
            else:
                print("  => no measurable difference")

        # block consolidation comparison
        if bs0 and bs1:
            cur0, cur1 = bs0["avg_current_blocks"], bs1["avg_current_blocks"]
            delta_blk = cur1 - cur0
            rel_blk = delta_blk / cur0 * 100 if cur0 else float("nan")

            # potential savings in baseline vs realised savings in shape-aware
            pot0 = bs0["overall_savings_pct"]   # unused potential in baseline
            sav0 = bs0["total_savings"]          # total potential blocks saveable
            sav1 = bs1["total_savings"]          # remaining potential in shape-aware

            # realised = blocks that were actually eliminated
            realised = bs0["total_blocks"] - bs1["total_blocks"]
            realised_pct = realised / bs0["total_blocks"] * 100 if bs0["total_blocks"] else float("nan")
            # what fraction of the consolidation potential was captured
            captured_pct = realised / sav0 * 100 if sav0 else float("nan")

            print(f"\n  [Block consolidation]")
            print(f"  avg blocks/record  : {cur0:.3f} → {cur1:.3f}  "
                  f"(delta {delta_blk:+.3f}, {rel_blk:+.2f}%)")
            print(f"  Blocks eliminated  : {realised:,}  ({realised_pct:+.2f}% of baseline total)")
            print(f"  Baseline potential : {sav0:,} blocks saveable  ({pot0:.2f}% of its total)")
            print(f"  Potential captured : {captured_pct:.1f}%  "
                  f"(shape-aware realised {captured_pct:.1f}% of baseline's consolidation headroom)")
            if bs1["reducible_pct"] < bs0["reducible_pct"] - 0.5:
                print("  => shape-aware routing reduces consolidation headroom ✓")
            else:
                print("  => little/no reduction in consolidation headroom")


if __name__ == "__main__":
    main()
