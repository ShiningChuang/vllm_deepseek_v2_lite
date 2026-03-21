# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Measure actual cuBLAS block count for expert FFN via nvprof --print-gpu-trace.
No hardware counter permission needed — just kernel timeline tracing.

Natural SM% = total GEMM blocks / 80 SMs × 100%
(shows how many waves cuBLAS needs, and how much the MPS limit matters)
"""

import os, sys, subprocess, re, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR    = "/workspace/result/alloc"
HIDDEN, INTER = 2048, 704
DTYPE         = torch.float16
DEVICE        = "cuda"
N_SMs         = 80   # V100S has 80 SMs

TOKEN_COUNTS  = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# ── Worker mode ───────────────────────────────────────────────────────────────
if "--worker" in sys.argv:
    n = int(sys.argv[sys.argv.index("--worker") + 1])
    torch.manual_seed(42)
    w13 = torch.randn(2 * INTER, HIDDEN, device=DEVICE, dtype=DTYPE)
    w2  = torch.randn(HIDDEN,    INTER,  device=DEVICE, dtype=DTYPE)
    x   = torch.randn(n,         HIDDEN, device=DEVICE, dtype=DTYPE)
    torch.cuda.synchronize()
    # Run once — nvprof traces every kernel launch
    o = F.linear(x, w13)
    g, u = o[:, :INTER], o[:, INTER:]
    _ = F.linear(F.silu(g) * u, w2)
    torch.cuda.synchronize()
    sys.exit(0)


# ── Parse nvprof --print-gpu-trace output ─────────────────────────────────────
# Relevant columns in trace:
#   Start  Duration  Grid Size  Block Size  Regs  SSMem  DSMem  Device  Context  Stream  Name
TRACE_HEADER_PATTERN = re.compile(
    r"^\s*Start\s+Duration\s+Grid Size\s+Block Size"
)
TRACE_ROW_PATTERN = re.compile(
    r"^\s*[\d.]+\w+\s+"          # Start
    r"[\d.]+\w+\s+"              # Duration
    r"\((\d+)\s+(\d+)\s+(\d+)\)"  # Grid Size  (gx gy gz)
    r"\s+\(\d+\s+\d+\s+\d+\)"   # Block Size
    r".+?(\S+)\s*$"              # Name (last token)
)

GEMM_KEYWORDS = ["gemm", "cutlass", "sgemm", "hgemm", "cublaslt",
                  "splitkreduce"]  # include splitK reduction


def parse_trace(text):
    """
    Return list of (full_line, grid_x, grid_y, grid_z) for every kernel row.
    We keep the full line so that is_gemm_kernel() can search it for keywords
    regardless of how nvprof truncates/formats the name column.
    """
    results = []
    in_table = False
    for line in text.splitlines():
        if TRACE_HEADER_PATTERN.match(line):
            in_table = True
            continue
        if not in_table:
            continue
        # Extract grid tuple (first parenthesised triple followed by another)
        grid_m = re.search(r'\((\d+)\s+(\d+)\s+(\d+)\)\s+\(', line)
        if not grid_m:
            continue
        gx, gy, gz = int(grid_m.group(1)), int(grid_m.group(2)), int(grid_m.group(3))
        # Strip trailing [invocation_id]
        clean = re.sub(r'\s*\[\d+\]\s*$', '', line.rstrip())
        results.append((clean, gx, gy, gz))
    return results


def is_gemm_kernel(full_line):
    ll = full_line.lower()
    return any(kw in ll for kw in GEMM_KEYWORDS)


def run_trace(n):
    log = f"/tmp/nvprof_trace_n{n}.txt"
    cmd = [
        "nvprof", "--print-gpu-trace",
        "--log-file", log,
        sys.executable, __file__, "--worker", str(n),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        return open(log).read()
    except FileNotFoundError:
        return ""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    measured = {}  # {n: {"w13_blocks": int, "w2_blocks": int, "all_kernels": [...]}}

    for n in TOKEN_COUNTS:
        print(f"\n[trace] n={n:5d}")
        text    = run_trace(n)
        kernels = parse_trace(text)

        if not kernels:
            print("  FAILED — no trace data")
            measured[n] = None
            continue

        gemm_kernels = [(name, gx*gy*gz) for name, gx, gy, gz in kernels
                        if is_gemm_kernel(name)]
        all_kernels  = [(name, gx*gy*gz) for name, gx, gy, gz in kernels]

        for full_line, blocks in all_kernels:
            # Extract a readable name: last non-bracket token in the line
            name_guess = re.sub(r'.*\s', '', re.sub(r'\s*\[\d+\]\s*$', '', full_line).rstrip())
            short = full_line[-70:]
            tag   = " ← GEMM" if is_gemm_kernel(full_line) else ""
            print(f"  {blocks:5d} blocks  ...{short}{tag}")

        measured[n] = {
            "gemm_kernels": gemm_kernels,
            "all_kernels":  all_kernels,
        }

    # ── Compute natural SM% ───────────────────────────────────────────────────
    # For a sequential FFN, kernels run one after another.
    # Natural SM% for each kernel = its block_count / N_SMs × 100%
    # We report the PEAK (bottleneck kernel) and the GEMM sum per forward pass.

    def sm_pct(blocks):
        return blocks / N_SMs * 100.0

    def wave_efficiency(blocks):
        """Fraction of SM-slots actually used = blocks / (waves × N_SMs)"""
        if blocks == 0:
            return 0.0
        waves = math.ceil(blocks / N_SMs)
        return blocks / (waves * N_SMs) * 100.0

    rows = []
    for n in TOKEN_COUNTS:
        d = measured.get(n)
        if d is None:
            rows.append((n, float("nan"), float("nan"), float("nan"),
                         float("nan"), float("nan"), "FAILED"))
            continue
        gemm  = d["gemm_kernels"]
        if not gemm:
            rows.append((n, 0, 0, 0, 0, 0, "no GEMM found"))
            continue
        # Peak single-kernel block count (bottleneck)
        max_blocks  = max(b for _, b in gemm)
        total_blocks = sum(b for _, b in gemm)
        peak_sm_pct  = sm_pct(max_blocks)
        total_sm_pct = sm_pct(total_blocks)
        wave_eff     = wave_efficiency(max_blocks)
        # Theoretical (old assumption TILE_M=128)
        theo = min(100.0, math.ceil(n/128) * 27 / N_SMs * 100.0)
        rows.append((n, max_blocks, total_blocks, peak_sm_pct,
                     total_sm_pct, theo, ""))

    # Print table
    print("\n" + "="*90)
    print(f"{'n':>6}  {'peak blk':>9}  {'total blk':>10}  "
          f"{'peak SM%':>9}  {'total SM%':>10}  {'theo SM%':>9}  note")
    print("-"*90)
    for n, pk, tot, pk_pct, tot_pct, theo, note in rows:
        print(f"{n:>6}  {pk:>9.0f}  {tot:>10.0f}  "
              f"{pk_pct:>8.1f}%  {tot_pct:>9.1f}%  {theo:>8.1f}%  {note}")

    np.save(OUTPUT_DIR + "/sm_natural_results.npy", {"rows": rows})

    # ── Plots ─────────────────────────────────────────────────────────────────
    valid = [(n, pk, tot, pk_pct, tot_pct, theo)
             for n, pk, tot, pk_pct, tot_pct, theo, note in rows
             if not math.isnan(pk_pct)]
    ns       = [r[0] for r in valid]
    pk_pcts  = [r[3] for r in valid]
    tot_pcts = [r[4] for r in valid]
    theos    = [r[5] for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, logx in zip(axes, [False, True]):
        plot = ax.semilogx if logx else ax.plot
        plot(ns, pk_pcts,  "o-", color="steelblue",  lw=2.2, ms=8,
             label="Measured: peak kernel block count / 80 SMs")
        plot(ns, tot_pcts, "s--", color="tomato",    lw=2.0, ms=7,
             label="Measured: total GEMM blocks / 80 SMs (sequential)")
        plot(ns, theos,    "^:", color="darkorange",  lw=1.8, ms=7,
             label="Theoretical (TILE_M=128 assumption — was wrong!)")
        ax.axhline(100, color="gray", lw=1.2, ls="--", alpha=0.6,
                   label="100% = 1 full wave of all 80 SMs")
        ax.set_xlabel("Token count" + (" (log)" if logx else ""), fontsize=11)
        ax.set_ylabel("SM% = blocks / 80 × 100%", fontsize=11)
        ax.set_title("Log scale" if logx else "Linear scale", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Annotate peak SM% values
        for n, pct in zip(ns, pk_pcts):
            ax.annotate(f"{pct:.0f}%", xy=(n, pct),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color="steelblue")

    fig.suptitle(
        "Expert FFN: Actual cuBLAS Block Count → Natural SM%\n"
        "V100S PCIe 32GB  ·  Measured via nvprof --print-gpu-trace  ·  "
        "No hardware counter permission needed\n"
        "cuBLAS chose TILE_M=32 + split-K, NOT TILE_M=128 as assumed",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    path = OUTPUT_DIR + "/microbench_sm_natural.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")
    print("[done]")


if __name__ == "__main__":
    main()
