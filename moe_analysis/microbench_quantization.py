# Fix sys.path
import sys as _sys
_sys.path = [p for p in _sys.path if p not in ("/workspace", "")]

"""
Two GPU quantization effects on expert FFN execution:

1. TILE QUANTIZATION (每级 block 利用率)
   - cuBLAS 将 M 维（token 数）按 TILE_M 取整
   - 刚跨过 tile 边界时，新 block 只有少量 token → 利用率骤降
   - 对延迟影响小，但浪费 SM 算力

2. WAVE QUANTIZATION (整 GPU 负载满载后的等待)
   - GPU 有限 SM，每次能并发执行的 tile 数 = N_SM × occupancy
   - 当总 tile 数超过一个 wave 容量时，多出的 tile 要排队到下一 wave
   - 对延迟影响大（可能跳升 50-100%）

实验设计：
  - 细粒度扫 n=1..1000，step=1，分别单独测两个 GEMM（w13 和 w2）
  - 计算 achieved FLOPs/s 和 achieved BW/s
  - 找 tile 边界（利用率凹陷）和 wave 边界（时间突变）
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

HIDDEN     = 2048
INTER      = 704
DTYPE      = torch.float16
DEVICE     = "cuda"
N_SM       = 80          # V100S PCIe SMs
FP16_TOPS  = 130e12      # V100S FP16 Tensor Core peak
MEM_BW_GBS = 1134e9      # V100S HBM2 bandwidth

N_WARMUP = 20
N_REPEAT = 150


def time_gemm_single(A, W, n_repeat, n_warmup):
    """Time a single F.linear call."""
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        _ = F.linear(A, W)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        _ = F.linear(A, W)
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.median(times)), float(np.percentile(times, 10)), float(np.percentile(times, 90))


def sweep(w, n_range, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Sweep token count, return (times_median, times_p10, times_p90)."""
    M_out, K = w.shape  # w is (out_features, in_features)
    medians, p10s, p90s = [], [], []
    for n in n_range:
        A = torch.randn(n, K, device=DEVICE, dtype=DTYPE)
        med, lo, hi = time_gemm_single(A, w, n_repeat, n_warmup)
        medians.append(med)
        p10s.append(lo)
        p90s.append(hi)
    return np.array(medians), np.array(p10s), np.array(p90s)


def detect_wave_boundaries(n_range, times_ms, min_jump_pct=8.0):
    """
    Detect wave quantization boundaries: places where time jumps by > min_jump_pct%.
    Returns list of (n_boundary, jump_pct, time_before, time_after).
    """
    x = np.array(n_range)
    t = np.array(times_ms)
    # smooth slightly to reduce noise
    from scipy.ndimage import uniform_filter1d
    t_smooth = uniform_filter1d(t, size=3)
    pct_change = np.diff(t_smooth) / t_smooth[:-1] * 100
    boundaries = []
    for i, pc in enumerate(pct_change):
        if pc > min_jump_pct:
            boundaries.append((x[i+1], pc, t[i], t[i+1]))
    return boundaries


def main():
    torch.manual_seed(42)

    # W13: gate+up fused projection (n, HIDDEN) → (n, 2*INTER)
    w13 = torch.randn(2 * INTER, HIDDEN, device=DEVICE, dtype=DTYPE)   # (1408, 2048)
    # W2: down projection (n, INTER) → (n, HIDDEN)
    w2  = torch.randn(HIDDEN, INTER,     device=DEVICE, dtype=DTYPE)   # (2048, 704)

    # ── PART 1: fine sweep n=1..400 step=1 (tile + early wave effects) ──────
    n_fine = list(range(1, 401))
    print(f"[sweep] w13 ({2*INTER}×{HIDDEN}): n=1..400 step=1 …")
    t13_med, t13_lo, t13_hi = sweep(w13, n_fine)
    print(f"[sweep] w2  ({HIDDEN}×{INTER}): n=1..400 step=1 …")
    t2_med,  t2_lo,  t2_hi  = sweep(w2,  n_fine)

    # ── PART 2: extended sweep n=1..2000 step=4 (wave quantization) ────────
    n_ext = list(range(1, 2001, 4))
    print(f"[sweep] w13: n=1..2000 step=4 …")
    t13x_med, _, _ = sweep(w13, n_ext, n_repeat=100)
    print(f"[sweep] w2:  n=1..2000 step=4 …")
    t2x_med,  _, _ = sweep(w2,  n_ext, n_repeat=100)

    x_fine = np.array(n_fine)
    x_ext  = np.array(n_ext)

    # FLOPs and bytes for each GEMM
    def flops(n, M_out, K): return 2 * n * M_out * K
    def util_tflops(n, t_ms, M_out, K): return flops(n, M_out, K) / (t_ms * 1e-3) / 1e12

    # Achieved FLOPs/s arrays
    u13_fine = np.array([util_tflops(n, t, 2*INTER, HIDDEN) for n, t in zip(n_fine, t13_med)])
    u2_fine  = np.array([util_tflops(n, t, HIDDEN,  INTER)  for n, t in zip(n_fine, t2_med)])
    u13_ext  = np.array([util_tflops(n, t, 2*INTER, HIDDEN) for n, t in zip(n_ext, t13x_med)])
    u2_ext   = np.array([util_tflops(n, t, HIDDEN,  INTER)  for n, t in zip(n_ext, t2x_med)])

    # Detect wave boundaries (extended range)
    try:
        wb13 = detect_wave_boundaries(n_ext, t13x_med, min_jump_pct=6.0)
        wb2  = detect_wave_boundaries(n_ext, t2x_med,  min_jump_pct=6.0)
    except ImportError:
        # scipy not available: simple detection
        wb13 = []
        wb2  = []
        for i in range(1, len(n_ext)):
            if t13x_med[i] / t13x_med[i-1] - 1 > 0.06:
                wb13.append((n_ext[i], (t13x_med[i]/t13x_med[i-1]-1)*100, t13x_med[i-1], t13x_med[i]))
            if t2x_med[i] / t2x_med[i-1] - 1 > 0.06:
                wb2.append((n_ext[i], (t2x_med[i]/t2x_med[i-1]-1)*100, t2x_med[i-1], t2x_med[i]))

    print(f"\n[wave] w13 boundaries: {[(b[0], f'{b[1]:.1f}%') for b in wb13[:10]]}")
    print(f"[wave] w2  boundaries: {[(b[0], f'{b[1]:.1f}%') for b in wb2[:10]]}")

    # Infer tile size from wave boundaries
    def infer_tile(boundaries, N_SM=80, N_out=None):
        # total_tiles = ceil(n/TILE_M) * ceil(N_out/TILE_N) = wave_size
        # wave boundary: ceil(n/TILE_M) * ceil(N_out/TILE_N) ≈ k * 80
        if len(boundaries) < 2: return None
        ns = [b[0] for b in boundaries[:5]]
        gaps = np.diff(ns)
        return int(np.median(gaps)) if len(gaps) > 0 else None

    # ── PLOTS ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)

    # ── Row 0: Tile quantization — achieved FLOPs/s, n=1..400 ────────────────

    for col, (label, u_fine, t_med, t_lo, t_hi, M_out, K) in enumerate([
        ("w13  (n×2048)→(n×1408)", u13_fine, t13_med, t13_lo, t13_hi, 2*INTER, HIDDEN),
        ("w2   (n×704)→(n×2048)",  u2_fine,  t2_med,  t2_lo,  t2_hi,  HIDDEN,  INTER),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(x_fine, u_fine, color="steelblue", linewidth=0.8, alpha=0.9)
        ax.axhline(FP16_TOPS / 1e12, color="red", linestyle="--", linewidth=1.2,
                   label=f"Peak {FP16_TOPS/1e12:.0f} TFLOPS")
        ax.set_xlabel("Tokens (M dimension)", fontsize=10)
        ax.set_ylabel("Achieved FP16 TFLOPS", fontsize=10)
        ax.set_title(f"Tile Quantization — GPU Utilization\n{label}  n=1→400", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(1, 400)
        ymax = max(u_fine.max(), FP16_TOPS/1e12) * 1.1
        ax.set_ylim(0, ymax)

        # Annotate: tile boundaries show as local dips in utilization
        # Pick the most prominent dips
        from scipy.ndimage import uniform_filter1d
        u_smooth = uniform_filter1d(u_fine, 3)
        diffs = np.diff(u_smooth)
        # Find local minima in utilization (just after a tile boundary)
        dip_candidates = []
        for i in range(1, len(u_smooth)-1):
            if u_smooth[i] < u_smooth[i-1] * 0.97 and u_smooth[i] < u_smooth[i+1] * 0.97:
                dip_candidates.append((x_fine[i], u_smooth[i]))
        # Show first 4 dips
        for di, (nx, uy) in enumerate(dip_candidates[:4]):
            ax.annotate(f"n={nx}", xy=(nx, uy), xytext=(nx+5, uy+0.3),
                        fontsize=7, color="darkorange",
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))

    # ── Row 1: Time vs n, n=1..400, highlight tile boundaries + utilization ──

    ax = fig.add_subplot(gs[1, :])
    ax.plot(x_fine, t13_med * 1000, color="steelblue", linewidth=0.8,
            label="w13  (n×2048→n×1408)  μs", alpha=0.9)
    ax.fill_between(x_fine, t13_lo*1000, t13_hi*1000, alpha=0.15, color="steelblue")
    ax.plot(x_fine, t2_med * 1000, color="darkorange", linewidth=0.8,
            label="w2   (n×704→n×2048)   μs", alpha=0.9)
    ax.fill_between(x_fine, t2_lo*1000, t2_hi*1000, alpha=0.15, color="darkorange")

    # Mark each tile boundary as a vertical shaded line
    # Detect tile jumps: where time increases > noise threshold
    def find_tile_jumps(times, threshold_us=1.5):
        jumps = []
        for i in range(1, len(times)):
            delta = (times[i] - times[i-1]) * 1000  # μs
            if delta > threshold_us:
                jumps.append(i)
        return jumps

    jumps13 = find_tile_jumps(t13_med)
    jumps2  = find_tile_jumps(t2_med)
    for j in jumps13[:12]:
        ax.axvline(x_fine[j], color="steelblue", linewidth=0.6, alpha=0.4, linestyle=":")
    for j in jumps2[:12]:
        ax.axvline(x_fine[j], color="darkorange", linewidth=0.6, alpha=0.4, linestyle=":")

    # Annotate tile size
    if len(jumps13) >= 2:
        gaps = np.diff([x_fine[j] for j in jumps13[:8]])
        tile13 = int(np.median(gaps)) if len(gaps) else "?"
        ax.text(0.02, 0.92, f"w13 tile_M ≈ {tile13} tokens", transform=ax.transAxes,
                color="steelblue", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    if len(jumps2) >= 2:
        gaps = np.diff([x_fine[j] for j in jumps2[:8]])
        tile2 = int(np.median(gaps)) if len(gaps) else "?"
        ax.text(0.02, 0.82, f"w2  tile_M ≈ {tile2} tokens", transform=ax.transAxes,
                color="darkorange", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    ax.set_xlabel("Tokens (M dimension)  n=1→400", fontsize=11)
    ax.set_ylabel("Execution time (μs)", fontsize=11)
    ax.set_title("Tile Quantization: Time Steps  (dotted lines = tile boundaries detected from time jumps)", fontsize=11)
    ax.legend(fontsize=9)

    # ── Row 2: Wave quantization — n=1..2000 ─────────────────────────────────

    ax = fig.add_subplot(gs[2, :])
    ax.plot(x_ext, t13x_med * 1000, color="steelblue", linewidth=1.0,
            label="w13  n=1→2000", alpha=0.9)
    ax.plot(x_ext, t2x_med * 1000, color="darkorange", linewidth=1.0,
            label="w2   n=1→2000", alpha=0.9)

    # Mark wave boundaries
    all_wb = []
    for wb, color, lbl in [(wb13, "steelblue", "w13"), (wb2, "darkorange", "w2")]:
        for i, (n_b, pct, t_before, t_after) in enumerate(wb[:8]):
            ax.axvline(n_b, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.annotate(f"n={n_b}\n+{pct:.0f}%",
                        xy=(n_b, t_after * 1000),
                        xytext=(n_b + 20, t_after * 1000 + 8),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            all_wb.append((n_b, pct, lbl))

    # Expected wave boundaries (theoretical)
    # For w13: (n × 1408), TILE estimated from row 1
    # N_out tiles per row: ceil(1408/TILE_N) for TILE_N=64 → 22; for 128 → 11
    # wave1 full: ceil(n/TILE_M) * ceil(N/TILE_N) = N_SM
    # → n_wave = ceil(N_SM / ceil(N/TILE_N)) * TILE_M
    ax.set_xlabel("Tokens (M dimension)  n=1→2000", fontsize=11)
    ax.set_ylabel("Execution time (μs)", fontsize=11)
    ax.set_title("Wave Quantization: Time jumps when total tiles exceed one wave\n"
                 "(dashed lines = detected wave boundaries; annotated % = jump magnitude)",
                 fontsize=11)
    ax.legend(fontsize=9)

    # Shade waves
    if wb13:
        wave_starts = [0] + [b[0] for b in wb13[:6]]
        for i in range(0, len(wave_starts)-1, 2):
            ax.axvspan(wave_starts[i], wave_starts[i+1], alpha=0.04, color="steelblue")
        for i in range(1, len(wave_starts)-1, 2):
            ax.axvspan(wave_starts[i], wave_starts[i+1], alpha=0.04, color="gray")

    fig.suptitle(
        "Tile Quantization vs Wave Quantization in cuBLAS FP16 GEMM (V100S PCIe)\n"
        "Tile quant → small periodic utilization dips  |  Wave quant → large sudden time jumps",
        fontsize=12, fontweight="bold")

    path = os.path.join(OUTPUT_DIR, "microbench_quantization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {path}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print("\n[summary] Tile quantization (n=1..400):")
    print(f"  w13: min util={u13_fine.min():.2f} TFLOPS, max={u13_fine.max():.2f} TFLOPS, "
          f"ratio={u13_fine.max()/u13_fine.min():.1f}×")
    print(f"  w2:  min util={u2_fine.min():.2f} TFLOPS,  max={u2_fine.max():.2f} TFLOPS,  "
          f"ratio={u2_fine.max()/u2_fine.min():.1f}×")
    print(f"\n[summary] Wave quantization boundaries detected:")
    print(f"  w13: {[(b[0], f'+{b[1]:.0f}%') for b in wb13[:8]]}")
    print(f"  w2:  {[(b[0], f'+{b[1]:.0f}%') for b in wb2[:8]]}")

    print("[done]")


if __name__ == "__main__":
    main()
