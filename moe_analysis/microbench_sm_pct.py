# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Expert FFN execution time vs GPU SM allocation (CUDA MPS experiment).

Uses CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to limit SM availability per client.

Key hypothesis:
  - BW-bound  (n < ridge ≈ 115): time ~constant regardless of SM%
    because cuBLAS only needs ~14–34% SMs for small GEMMs anyway
  - Compute-bound (n > 115): time ∝ 1/SM% — fewer SMs → more serial waves

How it works:
  1. Start nvidia-cuda-mps-control daemon
  2. For each SM% in [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]:
     spawn a subprocess with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=X
     (MPS enforces this at hardware level)
  3. Plot time vs SM% for several token counts
"""

import os
import sys
import json
import math
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/users/lxzhong/vllm_deepseek_v2_lite/moe_analysis/result/alloc"
MPS_PIPE   = "/tmp/nvidia-mps"
MPS_LOG    = "/tmp/nvidia-mps-log"

HIDDEN = 2048
INTER  = 704
DTYPE  = torch.float16
DEVICE = "cuda"

N_WARMUP = 30
N_REPEAT = 150

SM_PCTS      = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
TOKEN_COUNTS = [16, 128, 512, 1024]   # BW-bound, near-ridge, 4.5×, 9×

# ── Worker subprocess mode ────────────────────────────────────────────────────
if os.environ.get("_SM_WORKER"):
    n_tokens = int(os.environ["N_TOKENS"])

    torch.manual_seed(42)
    w13 = torch.randn(2 * INTER, HIDDEN, device=DEVICE, dtype=DTYPE)
    w2  = torch.randn(HIDDEN,    INTER,  device=DEVICE, dtype=DTYPE)
    x   = torch.randn(n_tokens,  HIDDEN, device=DEVICE, dtype=DTYPE)

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

    print(json.dumps({
        "n_tokens":  n_tokens,
        "median_ms": float(np.median(times)),
        "std_ms":    float(np.std(times)),
        "p5_ms":     float(np.percentile(times, 5)),
        "p95_ms":    float(np.percentile(times, 95)),
    }))
    sys.exit(0)


# ── Main orchestrator ─────────────────────────────────────────────────────────
def start_mps():
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG,  exist_ok=True)
    env = {**os.environ,
           "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE,
           "CUDA_MPS_LOG_DIRECTORY":  MPS_LOG}
    r = subprocess.run(["nvidia-cuda-mps-control", "-d"],
                       env=env, capture_output=True, text=True, timeout=10)
    ok = r.returncode == 0 or "already" in (r.stderr + r.stdout).lower()
    if ok:
        time.sleep(1.5)
        print(f"[mps] daemon started  (pipe={MPS_PIPE})")
    else:
        print(f"[mps] WARN: {r.stderr.strip() or r.stdout.strip()}")
    return ok


def run_worker(n_tokens, sm_pct):
    env = {**os.environ,
           "_SM_WORKER":  "1",
           "N_TOKENS":    str(n_tokens),
           "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(sm_pct),
           "CUDA_MPS_PIPE_DIRECTORY": MPS_PIPE,
           "CUDA_MPS_LOG_DIRECTORY":  MPS_LOG}
    try:
        r = subprocess.run([sys.executable, __file__],
                           env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        print(f"\n    [timeout] n={n_tokens}, sm={sm_pct}%")
        return None
    for line in reversed(r.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except Exception:
            continue
    if r.returncode != 0:
        print(f"\n    [error] n={n_tokens}, sm={sm_pct}%: {r.stderr[-200:]}")
    return None


def natural_sm_pct(n, n_sms=80, tile_m=128, tile_n=128):
    """Approximate % of SMs a cuBLAS GEMM will occupy for this token count.
    w13: (n × 2048) × (2048 × 1408)  →  ceil(n/tile_m) × ceil(1408/tile_n) blocks
    w2:  (n × 704)  × (704  × 2048)  →  ceil(n/tile_m) × ceil(2048/tile_n) blocks
    """
    bm    = math.ceil(n / tile_m)
    bn_w13 = math.ceil(1408 / tile_n)
    bn_w2  = math.ceil(2048 / tile_n)
    total  = bm * (bn_w13 + bn_w2)
    return min(100.0, total / n_sms * 100.0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mps_ok = start_mps()
    if not mps_ok:
        print("[warn] Running without MPS — SM% limits won't be enforced; "
              "results will show baseline only")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results = {n: {} for n in TOKEN_COUNTS}   # results[n][pct] = median_ms

    for n in TOKEN_COUNTS:
        print(f"\n[sweep] n={n} tokens  (natural SM% ≈ {natural_sm_pct(n):.0f}%)")
        for pct in SM_PCTS:
            print(f"  SM={pct:3d}%  ", end="", flush=True)
            d = run_worker(n, pct)
            if d:
                results[n][pct] = d["median_ms"]
                print(f"{d['median_ms']*1000:.1f} ± {d['std_ms']*1000:.1f} μs")
            else:
                results[n][pct] = float("nan")
                print("FAILED")

    np.save(os.path.join(OUTPUT_DIR, "sm_pct_results.npy"), results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    ridge  = 115
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    labels = [f"n={n} ({'BW-bound' if n < ridge else f'~{n/ridge:.1f}×ridge'})"
              for n in TOKEN_COUNTS]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # A: Absolute time vs SM%
    ax = axes[0, 0]
    for n, color, label in zip(TOKEN_COUNTS, colors, labels):
        xs = [p for p in SM_PCTS if not math.isnan(results[n].get(p, float("nan")))]
        ys = [results[n][p] * 1000 for p in xs]   # μs
        ax.plot(xs, ys, "o-", color=color, linewidth=2.2, markersize=7, label=label)
        # Mark natural SM%
        xn = natural_sm_pct(n)
        if xs and ys:
            yn = np.interp(xn, xs, ys)
            ax.axvline(xn, color=color, linewidth=0.8, linestyle=":",  alpha=0.5)
            ax.annotate(f"{xn:.0f}%\n(natural)", xy=(xn, yn),
                        xytext=(xn + 3, yn * 1.08), fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
    ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
    ax.set_ylabel("Execution time (μs)", fontsize=11)
    ax.set_title("A. Absolute Execution Time vs SM Allocation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)

    # B: Speedup relative to 5%
    ax = axes[0, 1]
    for n, color, label in zip(TOKEN_COUNTS, colors, labels):
        xs = [p for p in SM_PCTS if not math.isnan(results[n].get(p, float("nan")))]
        ys = [results[n][p] for p in xs]
        if ys and ys[0] > 0:
            speedup = [ys[0] / y for y in ys]
            ax.plot(xs, speedup, "o-", color=color, linewidth=2.2, markersize=7, label=label)
    # Ideal linear reference
    xs_ideal = np.array(SM_PCTS, dtype=float)
    ax.plot(xs_ideal, xs_ideal / SM_PCTS[0], "--", color="gray",
            linewidth=1.5, alpha=0.7, label="Ideal (linear with SM%)")
    ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
    ax.set_ylabel("Speedup (vs 5% baseline)", fontsize=11)
    ax.set_title("B. Speedup vs SM%\nBW-bound: flat   |   Compute-bound: approaches linear", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)

    # C: Slowdown relative to 100%
    ax = axes[1, 0]
    for n, color, label in zip(TOKEN_COUNTS, colors, labels):
        xs = [p for p in SM_PCTS if not math.isnan(results[n].get(p, float("nan")))]
        ys = [results[n][p] for p in xs]
        t100 = results[n].get(100, float("nan"))
        if not math.isnan(t100) and t100 > 0:
            norm = [y / t100 for y in ys]
            ax.plot(xs, norm, "o-", color=color, linewidth=2.2, markersize=7, label=label)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="100% SM baseline")
    # Ideal 1/x curve
    xs_ideal = np.linspace(SM_PCTS[0], 100, 100)
    ax.plot(xs_ideal, 100 / xs_ideal, "--", color="gray", linewidth=1.0,
            alpha=0.5, label="Ideal 1/SM% (compute)")
    ax.set_xlabel("SM allocation (MPS %)", fontsize=11)
    ax.set_ylabel("Slowdown factor  (×100%-SM time)", fontsize=11)
    ax.set_title("C. Slowdown Relative to Full GPU (100%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 105)

    # D: Heatmap of execution time (μs)
    ax = axes[1, 1]
    matrix = np.array([[results[n].get(p, float("nan")) * 1000
                        for p in SM_PCTS] for n in TOKEN_COUNTS])
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(SM_PCTS)))
    ax.set_xticklabels([f"{p}%" for p in SM_PCTS], fontsize=9)
    ax.set_yticks(range(len(TOKEN_COUNTS)))
    ax.set_yticklabels([f"n={n}" for n in TOKEN_COUNTS], fontsize=10)
    ax.set_xlabel("SM allocation (%)", fontsize=11)
    ax.set_title("D. Execution Time Heatmap (μs)\n(row = token count, col = SM%)", fontsize=11)
    plt.colorbar(im, ax=ax, label="μs")
    for i in range(len(TOKEN_COUNTS)):
        for j in range(len(SM_PCTS)):
            v = matrix[i, j]
            if not math.isnan(v):
                vmax = np.nanmax(matrix)
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if v > vmax * 0.6 else "black")

    fig.suptitle(
        "Expert FFN: Execution Time vs SM Allocation  (CUDA MPS experiment)\n"
        f"V100S PCIe 32GB  ·  HIDDEN={HIDDEN}, INTER={INTER}  ·  Ridge ≈ {ridge} tokens\n"
        "BW-bound (n<115): time unaffected by SM limit  ·  Compute-bound (n>115): time ∝ 1/SM%",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "microbench_sm_pct.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[plot] {path}")

    # ── Text summary ──────────────────────────────────────────────────────────
    print("\n[summary] Execution time (μs)")
    header = f"{'SM%':>5} " + " ".join(f"n={n:>4}" for n in TOKEN_COUNTS)
    print(header)
    print("-" * len(header))
    for pct in SM_PCTS:
        row = f"{pct:4d}% "
        for n in TOKEN_COUNTS:
            v = results[n].get(pct, float("nan"))
            row += f"  {v*1000:6.1f}μs" if not math.isnan(v) else "    FAIL  "
        print(row)
    print("\n[done]")


if __name__ == "__main__":
    main()
