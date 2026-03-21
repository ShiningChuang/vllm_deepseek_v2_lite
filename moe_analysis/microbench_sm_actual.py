# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Directly measure SM utilization (sm_efficiency) for expert FFN
via nvprof --metrics sm_efficiency.

sm_efficiency = "% of time ≥1 warp is active on a multiprocessor,
                 averaged across all 80 SMs"  (NVIDIA definition)

This is the authoritative hardware counter — no assumptions about
tile size, blocks-per-SM, etc.
"""

import os, sys, csv, io, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR    = "/workspace/result/alloc"
HIDDEN, INTER = 2048, 704
N_WARMUP      = 5
N_REPEAT      = 20   # nvprof averages across all invocations
TOKEN_COUNTS  = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GEMM_KW       = ["gemm", "cutlass"]

# ── Worker: run exactly N_REPEAT forward passes, one token count ─────────────
if os.environ.get("_SM_WORKER"):
    n = int(os.environ["N_TOKENS"])
    torch.manual_seed(42)
    w13 = torch.randn(2*INTER, HIDDEN, device="cuda", dtype=torch.float16)
    w2  = torch.randn(HIDDEN,  INTER,  device="cuda", dtype=torch.float16)
    x   = torch.randn(n,       HIDDEN, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(N_WARMUP):
        o = F.linear(x, w13); g,u = o[:,:INTER],o[:,INTER:]
        _ = F.linear(F.silu(g)*u, w2)
    torch.cuda.synchronize()
    for _ in range(N_REPEAT):
        o = F.linear(x, w13); g,u = o[:,:INTER],o[:,INTER:]
        _ = F.linear(F.silu(g)*u, w2)
    torch.cuda.synchronize()
    sys.exit(0)


# ── Parse nvprof CSV ──────────────────────────────────────────────────────────
def run_nvprof(n):
    log = f"/tmp/nvprof_sm_n{n}.csv"
    env = {**os.environ, "_SM_WORKER": "1", "N_TOKENS": str(n)}
    subprocess.run(
        ["nvprof", "--csv", "--metrics", "sm_efficiency",
         "--log-file", log, sys.executable, __file__],
        env=env, capture_output=True, text=True, timeout=180)
    try:
        return open(log).read()
    except FileNotFoundError:
        return ""


def parse_csv(text):
    """Return list of (kernel_name, avg_sm_eff%) for GEMM kernels."""
    rows = [l for l in text.splitlines()
            if not l.startswith("==") and l.strip()]
    if not rows:
        return []
    parsed = list(csv.reader(rows))
    hdr = next((r for r in parsed if "Metric Name" in r), None)
    if not hdr:
        return []
    ki = hdr.index("Kernel")
    ai = hdr.index("Avg")
    results = {}
    for row in parsed[parsed.index(hdr)+1:]:
        if len(row) <= max(ki, ai):
            continue
        name    = row[ki].strip('"')
        val_str = row[ai].strip().strip('"').rstrip('%')
        try:
            v = float(val_str)
            results.setdefault(name, []).append(v)
        except ValueError:
            pass
    return [(k, float(np.mean(v))) for k, v in results.items()]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gemm_eff = {}   # {n: avg sm_efficiency of GEMM kernels}
    all_eff  = {}   # {n: [(name_short, eff), ...]}

    for n in TOKEN_COUNTS:
        print(f"[n={n:5d}]  profiling ...", flush=True)
        entries = parse_csv(run_nvprof(n))
        if not entries:
            print("  FAILED")
            gemm_eff[n] = float("nan")
            continue

        gemm = [(name, eff) for name, eff in entries
                if any(kw in name.lower() for kw in GEMM_KW)]
        other = [(name, eff) for name, eff in entries
                 if not any(kw in name.lower() for kw in GEMM_KW)]

        all_eff[n] = entries
        for name, eff in gemm:
            short = name.split("<")[-1][:55] if "<" in name else name[-55:]
            print(f"  GEMM   {eff:5.1f}%  {short}")
        for name, eff in other:
            short = name[-45:]
            print(f"  other  {eff:5.1f}%  {short}")

        gemm_eff[n] = np.mean([e for _, e in gemm]) if gemm else float("nan")
        print(f"  → GEMM avg sm_efficiency = {gemm_eff[n]:.1f}%")

    np.save(OUTPUT_DIR + "/sm_actual_results.npy",
            {"gemm_eff": gemm_eff, "all_eff": all_eff})

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print(f"{'n':>6}  {'GEMM sm_efficiency (avg)':>24}")
    print("-"*45)
    for n in TOKEN_COUNTS:
        v = gemm_eff.get(n, float("nan"))
        print(f"{n:>6}  {v:>23.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    xs  = [n for n in TOKEN_COUNTS if not np.isnan(gemm_eff.get(n, float("nan")))]
    ys  = [gemm_eff[n] for n in xs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, logx in zip(axes, [False, True]):
        pf = ax.semilogx if logx else ax.plot
        pf(xs, ys, "o-", color="steelblue", lw=2.2, ms=8,
           label="Measured: GEMM sm_efficiency (nvprof hardware counter)")
        ax.axhline(100, color="gray", ls="--", lw=1.2, alpha=0.5,
                   label="100% = all 80 SMs always busy")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}%", xy=(x, y),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8.5, color="steelblue")
        ax.set_xlabel("Tokens routed to expert" + (" (log)" if logx else ""),
                      fontsize=11)
        ax.set_ylabel("sm_efficiency (%)\n"
                      "= % of time ≥1 warp active per SM, avg over 80 SMs",
                      fontsize=10)
        ax.set_title("Log scale" if logx else "Linear scale", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 110)

    fig.suptitle(
        "Expert FFN: Actual SM Utilization — nvprof sm_efficiency\n"
        "V100S PCIe 32GB  ·  Hardware counter (no assumptions)\n"
        "sm_efficiency = Multiprocessor Activity (NVIDIA definition)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = OUTPUT_DIR + "/microbench_sm_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")
    print("[done]")


if __name__ == "__main__":
    main()
