#!/usr/bin/env python3
"""
Analyze MoE shape log files and classify records into two phases:

  Decode        : tokens_in_chunk <= concurrency
                  Each active decode request contributes exactly 1 token,
                  so the batch size equals the number of concurrent requests.

  Prefill/Mixed : tokens_in_chunk > concurrency
                  Either a pure prefill (prompt tokens only, no decode running)
                  or a mixed batch (prompt tokens + decode tokens together via
                  continuous batching).

Note: startup dummy records (tokens_in_chunk == max_model_len, appearing
before any real request) are already stripped by run_all.sh before logging.

Usage:
  python3 analyze_phases.py \
      --result-dir /data/lxzhong_home/result \
      --run-ids 13,14,15 \
      --labels "conc=1,conc=4,conc=8" \
      --concurrencies "1,4,8"
"""
import argparse
import json
import os
import glob
from collections import defaultdict
from typing import List, Dict, Tuple


# ---------- padding waste ----------
def padding_waste(rec: dict) -> Tuple[int, int]:
    """Return (sum of tokens_per_expert_assign, num_tokens_post_padded)."""
    real = sum(rec["tokens_per_expert_assign"])
    padded = rec["num_tokens_post_padded"]
    return real, padded


# ---------- read one JSONL ----------
def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
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


# ---------- analyse one run ----------
def analyse_run(records: List[dict], concurrency: int, label: str):
    decode_records = [r for r in records if r["tokens_in_chunk"] <= concurrency]
    other_records  = [r for r in records if r["tokens_in_chunk"] >  concurrency]

    total = len(records)
    if total == 0:
        print(f"\n[{label}] No records found.")
        return

    def stats(recs):
        if not recs:
            return 0, 0.0, float("nan"), float("nan")
        n = len(recs)
        pct = 100.0 * n / total
        tics = [r["tokens_in_chunk"] for r in recs]
        avg_tic = sum(tics) / len(tics)
        wastes = []
        for r in recs:
            real, padded = padding_waste(r)
            if padded > 0:
                wastes.append(real / padded)
        avg_waste = sum(wastes) / len(wastes) if wastes else float("nan")
        return n, pct, avg_tic, avg_waste

    d_n, d_pct, d_avg_tic, d_waste = stats(decode_records)
    o_n, o_pct, o_avg_tic, o_waste = stats(other_records)

    print(f"\n{'='*60}")
    print(f"Run: {label}  (concurrency={concurrency}, total MoE records: {total})")
    print(f"{'='*60}")
    print()
    print(f"  {'Phase':<20} {'Count':>8}  {'%':>6}  {'avg tic':>8}  {'goodput(real/pad)':>18}")
    print(f"  {'-'*20} {'-'*8}  {'-'*6}  {'-'*8}  {'-'*18}")
    print(f"  {'Decode (tic<=conc)':<20} {d_n:>8}  {d_pct:>5.1f}%  {d_avg_tic:>8.2f}  {d_waste:>18.4f}")
    print(f"  {'Prefill/Mixed (tic>conc)':<20} {o_n:>8}  {o_pct:>5.1f}%  {o_avg_tic:>8.2f}  {o_waste:>18.4f}")
    print()

    # tokens_in_chunk histogram
    print("  tokens_in_chunk histogram:")
    buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 99999]
    bucket_counts = defaultdict(int)
    for rec in records:
        tic = rec["tokens_in_chunk"]
        for i, b in enumerate(buckets):
            if tic <= b:
                bucket_counts[i] += 1
                break
    prev = 0
    for i, b in enumerate(buckets):
        c = bucket_counts[i]
        if c == 0:
            prev = b
            continue
        lo = prev + 1 if prev else 1
        hi = b
        tag = " [decode]" if hi <= concurrency else " [prefill/mixed]"
        bar = "#" * min(40, int(40 * c / total))
        print(f"    [{lo:>5}-{hi:>5}]{tag}: {c:>6} ({100*c/total:4.1f}%)  {bar}")
        prev = b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="/data/lxzhong_home/result")
    parser.add_argument("--run-ids", default="",
                        help="Comma-separated run IDs (e.g. '13,14,15').")
    parser.add_argument("--labels", default="",
                        help="Comma-separated labels, same order as --run-ids.")
    parser.add_argument("--concurrencies", default="",
                        help="Comma-separated concurrency per run (e.g. '1,4,8'). "
                             "Must match --run-ids order.")
    parser.add_argument("--ranks", default="0",
                        help="GPU ranks to include (comma-sep). Default: 0.")
    args = parser.parse_args()

    result_dir = args.result_dir
    ranks = [int(r) for r in args.ranks.split(",")]

    # Determine run IDs
    if args.run_ids:
        run_ids = [x.strip() for x in args.run_ids.split(",")]
    else:
        all_files = glob.glob(os.path.join(result_dir, "moe_shapes_run*_rank0.jsonl"))
        run_ids = sorted(
            set(
                os.path.basename(f).split("_rank")[0].replace("moe_shapes_run", "")
                for f in all_files
            )
        )
        if not run_ids:
            print(f"No moe_shapes_run* files found in {result_dir}")
            return

    labels_list = [l.strip() for l in args.labels.split(",")] if args.labels else []
    conc_list = [int(c.strip()) for c in args.concurrencies.split(",")] \
        if args.concurrencies else [1] * len(run_ids)

    print(f"Analysing {len(run_ids)} run(s): {run_ids}")
    print(f"Ranks included: {ranks}")

    for i, run_id in enumerate(run_ids):
        label = labels_list[i] if i < len(labels_list) else f"run={run_id}"
        concurrency = conc_list[i] if i < len(conc_list) else 1

        all_records = []
        for rank in ranks:
            path = os.path.join(result_dir, f"moe_shapes_run{run_id}_rank{rank}.jsonl")
            if not os.path.exists(path):
                print(f"  [warn] file not found: {path}")
                continue
            recs = load_jsonl(path)
            print(f"  Loaded {len(recs)} records from {os.path.basename(path)}")
            all_records.extend(recs)

        analyse_run(all_records, concurrency, label)


if __name__ == "__main__":
    main()
