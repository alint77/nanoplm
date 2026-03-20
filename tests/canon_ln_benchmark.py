"""Fused LN+Conv Canon benchmark with roofline analysis.

Compares three paths:
  1. Unfused: nn.LayerNorm + Triton Canon conv (separate ops)
  2. Fused:   varlen_ln_canon_conv (single fused op)
  3. Ref:     nn.LayerNorm + roll-based Python Canon (torch.compile baseline)

Reports per-kernel timings, achieved bandwidth, and roofline efficiency.

Usage::

    python tests/canon_ln_benchmark.py
    python tests/canon_ln_benchmark.py --T 65536 --C 768 --radius 3 --n-seqs 256
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# ── Setup ────────────────────────────────────────────────────────────────

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))


# ── Timing utility ───────────────────────────────────────────────────────


def _time_cuda(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


@dataclass(frozen=True)
class BenchResult:
    name: str
    avg_ms: float
    med_ms: float
    min_ms: float
    achieved_gbps: float
    min_theoretical_us: float
    efficiency_pct: float


def _summarize(name: str, times_ms: list[float], total_bytes: int, mem_bw_gbps: float) -> BenchResult:
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    mn = min(times_ms)
    achieved = (total_bytes / 1e9) / (avg / 1e3) if avg > 0 else 0
    theoretical_us = (total_bytes / 1e9) / mem_bw_gbps * 1e6
    eff = (theoretical_us / (avg * 1e3) * 100) if avg > 0 else 0
    return BenchResult(
        name=name, avg_ms=avg, med_ms=med, min_ms=mn,
        achieved_gbps=achieved, min_theoretical_us=theoretical_us,
        efficiency_pct=eff,
    )


# ── IO cost model ────────────────────────────────────────────────────────
# All counts in bytes.  We assume contiguous (T, C) bf16 tensors.

def _bytes_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _bytes_ln_fwd(T: int, C: int, eb: int) -> int:
    """Read x (T,C), write out (T,C), write mean+rstd (T,) fp32."""
    return T * C * eb * 2 + T * 4 * 2


def _bytes_conv_fwd(T: int, C: int, eb: int) -> int:
    """Read LN_out (T,C), write conv_out (T,C)."""
    return T * C * eb * 2


def _bytes_fused_fwd(T: int, C: int, eb: int) -> int:
    """Read x (T,C), write out (T,C), write mean+rstd (T,) fp32.
    No intermediate LN output materialized."""
    return T * C * eb * 2 + T * 4 * 2


def _bytes_ln_bwd(T: int, C: int, eb: int) -> int:
    """Read grad_out (T,C), x (T,C), mean+rstd (T,); write grad_x (T,C), partial dgamma."""
    return T * C * eb * 3 + T * 4 * 2 + T * C * eb


def _bytes_conv_bwd_dx(T: int, C: int, eb: int) -> int:
    """Read grad_out (T,C), write grad_x (T,C)."""
    return T * C * eb * 2


def _bytes_conv_bwd_dw(T: int, C: int, K: int, eb: int) -> int:
    """Read grad_out (T,C), x (T,C)."""
    return T * C * eb * 2


def _bytes_fused_bwd(T: int, C: int, K: int, eb: int) -> int:
    """Total fused backward IO:
    - conv_bwd_dx: read grad_out (T,C), write grad_conv_x (T,C)
    - ln_bwd: read grad_ln_out (T,C), x (T,C), mean+rstd; write grad_x (T,C)
    - conv_bwd_dw_db: read grad_out (T,C), x (T,C), mean+rstd; recompute LN inline
    """
    conv_dx = T * C * eb * 2
    ln_bwd = T * C * eb * 3 + T * 4 * 2 + T * C * eb
    conv_dw = T * C * eb * 2 + T * 4 * 2  # recompute LN from x + stats
    return conv_dx + ln_bwd + conv_dw


# ── Test data ─────────────────────────────────────────────────────────────


def _make_data(T, C, K, n_seqs, device="cuda", dtype=torch.bfloat16):
    x = torch.randn(T, C, device=device, dtype=dtype)
    weight = torch.randn(C, K, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)
    ln_weight = torch.ones(C, device=device, dtype=dtype)

    seq_len = max(1, T // n_seqs)
    lengths = [seq_len] * (T // seq_len)
    remainder = T - sum(lengths)
    if remainder > 0:
        lengths[-1] += remainder
    offsets = [0]
    for sl in lengths:
        offsets.append(offsets[-1] + sl)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    positions = torch.arange(T, device=device, dtype=cu_seqlens.dtype)
    seq_id = torch.searchsorted(cu_seqlens[1:], positions, right=True)

    return x, weight, bias, ln_weight, seq_id


# ── GPU detection ─────────────────────────────────────────────────────────


def _detect_mem_bw():
    props = torch.cuda.get_device_properties("cuda")
    name = props.name.lower()
    known = {
        "5090": 1792.0, "5080": 960.0, "4090": 1008.0, "4080": 716.8,
        "a100": 2039.0, "h100": 3350.0, "h200": 4800.0,
        "a6000": 768.0, "l40": 864.0,
    }
    for key, bw in known.items():
        if key in name:
            return bw
    return 1000.0


def _detect_peak_tflops():
    props = torch.cuda.get_device_properties("cuda")
    name = props.name.lower()
    # bf16 tensor core TFLOPS
    known = {
        "5090": 209.5, "5080": 112.0, "4090": 165.2, "4080": 97.5,
        "a100": 312.0, "h100": 989.0, "h200": 989.0,
    }
    for key, tf in known.items():
        if key in name:
            return tf
    return 100.0


# ── Benchmark runners ────────────────────────────────────────────────────


def _bench_unfused(T, C, K, radius, n_seqs, iters, warmup, mem_bw, dtype):
    from nanoplm.pretraining.models.modern_bert.canon_ops import varlen_canon_conv

    eb = _bytes_elem(dtype)
    x, weight, bias, ln_weight, seq_id = _make_data(T, C, K, n_seqs, dtype=dtype)
    ln = nn.LayerNorm(C, device="cuda", dtype=dtype)
    ln.weight.data.copy_(ln_weight)
    ln.bias.data.zero_()

    # Forward
    def fwd():
        with torch.no_grad():
            h = ln(x)
            out = h + varlen_canon_conv(h, seq_id, weight, bias, radius)
        return out

    fwd_times = _time_cuda(fwd, iters, warmup)
    fwd_bytes = _bytes_ln_fwd(T, C, eb) + _bytes_conv_fwd(T, C, eb)
    fwd_r = _summarize("unfused_fwd", fwd_times, fwd_bytes, mem_bw)

    # Forward + backward
    def fwd_bwd():
        x_ = x.detach().clone().requires_grad_(True)
        h = ln(x_)
        out = h + varlen_canon_conv(h, seq_id, weight, bias, radius)
        out.float().sum().backward()

    fwd_bwd_times = _time_cuda(fwd_bwd, iters, warmup)
    bwd_bytes = _bytes_ln_bwd(T, C, eb) + _bytes_conv_bwd_dx(T, C, eb) + _bytes_conv_bwd_dw(T, C, K, eb)
    total_bytes = fwd_bytes + bwd_bytes
    fwd_bwd_r = _summarize("unfused_fwd+bwd", fwd_bwd_times, total_bytes, mem_bw)

    return fwd_r, fwd_bwd_r


def _bench_fused(T, C, K, radius, n_seqs, iters, warmup, mem_bw, dtype):
    from nanoplm.pretraining.models.modern_bert.canon_ops import varlen_ln_canon_conv

    eb = _bytes_elem(dtype)
    x, weight, bias, ln_weight, seq_id = _make_data(T, C, K, n_seqs, dtype=dtype)
    ln_eps = 1e-5

    # Forward
    def fwd():
        with torch.no_grad():
            varlen_ln_canon_conv(x, seq_id, ln_weight, ln_eps, weight, bias, radius)

    fwd_times = _time_cuda(fwd, iters, warmup)
    fwd_bytes = _bytes_fused_fwd(T, C, eb)
    fwd_r = _summarize("fused_fwd", fwd_times, fwd_bytes, mem_bw)

    # Forward + backward
    def fwd_bwd():
        x_ = x.detach().clone().requires_grad_(True)
        out = varlen_ln_canon_conv(x_, seq_id, ln_weight, ln_eps, weight, bias, radius)
        out.float().sum().backward()

    fwd_bwd_times = _time_cuda(fwd_bwd, iters, warmup)
    bwd_bytes = _bytes_fused_bwd(T, C, K, eb)
    total_bytes = fwd_bytes + bwd_bytes
    fwd_bwd_r = _summarize("fused_fwd+bwd", fwd_bwd_times, total_bytes, mem_bw)

    return fwd_r, fwd_bwd_r


# ── Main ──────────────────────────────────────────────────────────────────


def _print_table(results: list[BenchResult]):
    hdr = f"  {'Kernel':30s}  {'avg ms':>8s}  {'med ms':>8s}  {'min ms':>8s}  {'GB/s':>8s}  {'eff%':>6s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(
            f"  {r.name:30s}  {r.avg_ms:8.3f}  {r.med_ms:8.3f}  {r.min_ms:8.3f}  "
            f"{r.achieved_gbps:8.1f}  {r.efficiency_pct:5.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Fused LN+Conv Canon benchmark")
    parser.add_argument("--T", type=int, default=65536, help="Total tokens")
    parser.add_argument("--C", type=int, default=768, help="Hidden size")
    parser.add_argument("--radius", type=int, default=3, help="Conv radius")
    parser.add_argument("--n-seqs", type=int, default=256, help="Number of sequences")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--mem-bw", type=float, default=None, help="Override peak mem BW (GB/s)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)

    mem_bw = args.mem_bw or _detect_mem_bw()
    peak_tflops = _detect_peak_tflops()
    K = 2 * args.radius + 1
    T, C = args.T, args.C
    eb = 2  # bf16

    print(f"Fused LN+Conv Canon benchmark")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Peak mem BW: {mem_bw:.0f} GB/s")
    print(f"  Peak compute: {peak_tflops:.1f} TFLOPS (bf16)")
    print(f"  Ridge point: {peak_tflops * 1e3 / mem_bw:.1f} FLOP/byte")
    print(f"  T={T}, C={C}, K={K} (radius={args.radius}), n_seqs={args.n_seqs}")
    print()

    # ── Roofline analysis ──
    fwd_flops = T * C * (2 * K + 5)  # conv: 2*K FLOP/elem, LN: ~5 FLOP/elem (sub, mul, add, mul, add)
    bwd_flops = fwd_flops * 3  # rough: dx + dw + LN bwd
    total_flops = fwd_flops + bwd_flops

    fused_fwd_bytes = _bytes_fused_fwd(T, C, eb)
    fused_bwd_bytes = _bytes_fused_bwd(T, C, K, eb)
    fused_total_bytes = fused_fwd_bytes + fused_bwd_bytes

    unfused_fwd_bytes = _bytes_ln_fwd(T, C, eb) + _bytes_conv_fwd(T, C, eb)
    unfused_bwd_bytes = (_bytes_ln_bwd(T, C, eb) + _bytes_conv_bwd_dx(T, C, eb)
                         + _bytes_conv_bwd_dw(T, C, K, eb))
    unfused_total_bytes = unfused_fwd_bytes + unfused_bwd_bytes

    print("Roofline analysis:")
    print(f"  {'':30s}  {'Unfused':>12s}  {'Fused':>12s}  {'Savings':>8s}")
    print(f"  {'FWD IO (MB)':30s}  {unfused_fwd_bytes/1e6:12.1f}  {fused_fwd_bytes/1e6:12.1f}  {(1 - fused_fwd_bytes/unfused_fwd_bytes)*100:7.0f}%")
    print(f"  {'BWD IO (MB)':30s}  {unfused_bwd_bytes/1e6:12.1f}  {fused_bwd_bytes/1e6:12.1f}  {(1 - fused_bwd_bytes/unfused_bwd_bytes)*100:7.0f}%")
    print(f"  {'Total IO (MB)':30s}  {unfused_total_bytes/1e6:12.1f}  {fused_total_bytes/1e6:12.1f}  {(1 - fused_total_bytes/unfused_total_bytes)*100:7.0f}%")
    print(f"  {'Arith intensity (FLOP/byte)':30s}  {total_flops/unfused_total_bytes:12.2f}  {total_flops/fused_total_bytes:12.2f}")
    print(f"  {'Regime':30s}  {'mem-bound':>12s}  {'mem-bound':>12s}")
    print(f"  {'Theoretical min FWD (us)':30s}  {unfused_fwd_bytes/1e9/mem_bw*1e6:12.0f}  {fused_fwd_bytes/1e9/mem_bw*1e6:12.0f}")
    print(f"  {'Theoretical min BWD (us)':30s}  {unfused_bwd_bytes/1e9/mem_bw*1e6:12.0f}  {fused_bwd_bytes/1e9/mem_bw*1e6:12.0f}")
    print(f"  {'Theoretical min total (us)':30s}  {unfused_total_bytes/1e9/mem_bw*1e6:12.0f}  {fused_total_bytes/1e9/mem_bw*1e6:12.0f}")
    print()

    # ── Run benchmarks ──
    print("Running benchmarks...")
    print()

    uf_fwd, uf_fwd_bwd = _bench_unfused(
        T, C, K, args.radius, args.n_seqs,
        args.iters, args.warmup, mem_bw, torch.bfloat16,
    )
    f_fwd, f_fwd_bwd = _bench_fused(
        T, C, K, args.radius, args.n_seqs,
        args.iters, args.warmup, mem_bw, torch.bfloat16,
    )

    print()
    print("Results:")
    _print_table([uf_fwd, f_fwd, uf_fwd_bwd, f_fwd_bwd])

    # ── Derived backward ──
    uf_bwd_ms = uf_fwd_bwd.avg_ms - uf_fwd.avg_ms
    f_bwd_ms = f_fwd_bwd.avg_ms - f_fwd.avg_ms
    uf_bwd_gbps = (unfused_bwd_bytes / 1e9) / (uf_bwd_ms / 1e3) if uf_bwd_ms > 0 else 0
    f_bwd_gbps = (fused_bwd_bytes / 1e9) / (f_bwd_ms / 1e3) if f_bwd_ms > 0 else 0
    uf_bwd_theo = unfused_bwd_bytes / 1e9 / mem_bw * 1e6
    f_bwd_theo = fused_bwd_bytes / 1e9 / mem_bw * 1e6
    uf_bwd_eff = (uf_bwd_theo / (uf_bwd_ms * 1e3) * 100) if uf_bwd_ms > 0 else 0
    f_bwd_eff = (f_bwd_theo / (f_bwd_ms * 1e3) * 100) if f_bwd_ms > 0 else 0

    print()
    print("Derived backward (fwd+bwd - fwd):")
    print(f"  {'unfused_bwd':30s}  {uf_bwd_ms:8.3f} ms  {uf_bwd_gbps:8.1f} GB/s  {uf_bwd_eff:5.1f}%")
    print(f"  {'fused_bwd':30s}  {f_bwd_ms:8.3f} ms  {f_bwd_gbps:8.1f} GB/s  {f_bwd_eff:5.1f}%")

    # ── Speedup summary ──
    print()
    print("Speedup summary:")
    if uf_fwd.avg_ms > 0 and f_fwd.avg_ms > 0:
        print(f"  FWD:     {uf_fwd.avg_ms/f_fwd.avg_ms:.2f}x  ({uf_fwd.avg_ms - f_fwd.avg_ms:+.3f} ms)")
    if uf_bwd_ms > 0 and f_bwd_ms > 0:
        print(f"  BWD:     {uf_bwd_ms/f_bwd_ms:.2f}x  ({uf_bwd_ms - f_bwd_ms:+.3f} ms)")
    if uf_fwd_bwd.avg_ms > 0 and f_fwd_bwd.avg_ms > 0:
        print(f"  FWD+BWD: {uf_fwd_bwd.avg_ms/f_fwd_bwd.avg_ms:.2f}x  ({uf_fwd_bwd.avg_ms - f_fwd_bwd.avg_ms:+.3f} ms)")

    # ── Per-step projection ──
    n_calls = 24  # typical 24 Canon layers in ModernBERT
    uf_step = uf_fwd_bwd.avg_ms * n_calls
    f_step = f_fwd_bwd.avg_ms * n_calls
    print()
    print(f"Per-step projection ({n_calls} Canon calls):")
    print(f"  Unfused overhead:  {uf_step:.1f} ms")
    print(f"  Fused overhead:    {f_step:.1f} ms")
    print(f"  Recovered:         {uf_step - f_step:.1f} ms/step")


if __name__ == "__main__":
    main()
