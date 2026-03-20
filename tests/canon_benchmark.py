"""Canon varlen depthwise conv microbenchmark.

Compares the original ``_varlen_canon_inner`` (roll-based, autograd backward)
against the Triton ``varlen_canon_conv`` op.  Both are benchmarked under
torch.compile (the only path that matters).

Reports per-call times, achieved bandwidth, and roofline efficiency for
forward+backward.

Usage::

    python tests/canon_benchmark.py
    python tests/canon_benchmark.py --T 65536 --C 768 --radius 3 --n-seqs 256
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def _add_repo_to_path() -> None:
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
    max_ms: float
    total_bytes: int
    achieved_gbps: float
    min_theoretical_ms: float
    efficiency_pct: float


def _summarize(name, times_ms, total_bytes, mem_bw_gbps):
    avg = statistics.mean(times_ms)
    med = statistics.median(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    achieved = (total_bytes / 1e9) / (avg / 1e3) if avg > 0 else 0
    theoretical_ms = (total_bytes / 1e9) / mem_bw_gbps * 1e3
    eff = (theoretical_ms / avg * 100) if avg > 0 else 0
    return BenchResult(
        name=name, avg_ms=avg, med_ms=med, min_ms=mn, max_ms=mx,
        total_bytes=total_bytes, achieved_gbps=achieved,
        min_theoretical_ms=theoretical_ms, efficiency_pct=eff,
    )


def _print_result(r):
    print(
        f"  {r.name:40s}  "
        f"avg={r.avg_ms:8.3f}ms  med={r.med_ms:8.3f}ms  "
        f"min={r.min_ms:8.3f}ms  max={r.max_ms:8.3f}ms  "
        f"BW={r.achieved_gbps:7.1f} GB/s  "
        f"eff={r.efficiency_pct:5.1f}%"
    )


def _bytes_fwd(T, C, elem_bytes):
    return T * C * elem_bytes * 2  # read x + write out


def _bytes_bwd(T, C, elem_bytes):
    return T * C * elem_bytes * 4  # grad_x (read go + write gx) + grad_w (read go + read x)


# ── Benchmark ─────────────────────────────────────────────────────────────


def _make_test_data(T, C, K, n_seqs, device="cuda", dtype=torch.bfloat16):
    x = torch.randn(T, C, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(C, K, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(C, device=device, dtype=dtype, requires_grad=True)

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

    return x, weight, bias, seq_id


def _bench_compiled(name, conv_fn, T, C, radius, n_seqs, iters, warmup, mem_bw_gbps):
    K = 2 * radius + 1
    elem_bytes = 2  # bf16
    x, weight, bias, seq_id = _make_test_data(T, C, K, n_seqs)

    def make_fwd_bwd(fn):
        def inner(x_, seq_id_, w_, b_):
            out = fn(x_, seq_id_, w_, b_, radius)
            return out.float().sum()
        return inner

    compiled_fn = torch.compile(make_fwd_bwd(conv_fn), fullgraph=False)

    print(f"  Compiling {name}...", end=" ", flush=True)
    for _ in range(warmup + 5):
        x_ = x.detach().clone().requires_grad_(True)
        w_ = weight.detach().clone().requires_grad_(True)
        b_ = bias.detach().clone().requires_grad_(True)
        loss = compiled_fn(x_, seq_id, w_, b_)
        loss.backward()
    torch.cuda.synchronize()
    print("done", flush=True)

    # Forward only
    def fwd_only():
        with torch.no_grad():
            compiled_fn(x.detach(), seq_id, weight.detach(), bias.detach())

    fwd_times = _time_cuda(fwd_only, iters, warmup)
    fwd_result = _summarize(
        f"{name}_fwd", fwd_times,
        _bytes_fwd(T, C, elem_bytes), mem_bw_gbps,
    )

    # Forward + backward
    def fwd_bwd():
        x_ = x.detach().clone().requires_grad_(True)
        w_ = weight.detach().clone().requires_grad_(True)
        b_ = bias.detach().clone().requires_grad_(True)
        loss = compiled_fn(x_, seq_id, w_, b_)
        loss.backward()

    fwd_bwd_times = _time_cuda(fwd_bwd, iters, warmup)
    total_bytes = _bytes_fwd(T, C, elem_bytes) + _bytes_bwd(T, C, elem_bytes)
    fwd_bwd_result = _summarize(f"{name}_fwd+bwd", fwd_bwd_times, total_bytes, mem_bw_gbps)

    bwd_avg = fwd_bwd_result.avg_ms - fwd_result.avg_ms
    bwd_bytes = _bytes_bwd(T, C, elem_bytes)
    bwd_achieved = (bwd_bytes / 1e9) / (bwd_avg / 1e3) if bwd_avg > 0 else 0
    bwd_theoretical = (bwd_bytes / 1e9) / mem_bw_gbps * 1e3
    bwd_eff = (bwd_theoretical / bwd_avg * 100) if bwd_avg > 0 else 0

    return fwd_result, fwd_bwd_result, bwd_avg, bwd_achieved, bwd_eff


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


def main():
    _add_repo_to_path()

    parser = argparse.ArgumentParser(description="Canon varlen conv benchmark")
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
    K = 2 * args.radius + 1

    print(f"Canon varlen conv benchmark (compiled)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Peak mem BW: {mem_bw:.0f} GB/s")
    print(f"  T={args.T}, C={args.C}, K={K} (radius={args.radius}), n_seqs={args.n_seqs}")
    print(f"  iters={args.iters}, warmup={args.warmup}")
    print()

    elem_bytes = 2
    fwd_bytes = _bytes_fwd(args.T, args.C, elem_bytes)
    bwd_bytes = _bytes_bwd(args.T, args.C, elem_bytes)
    total_bytes = fwd_bytes + bwd_bytes
    fwd_flops = args.T * args.C * 2 * K
    total_flops = fwd_flops + fwd_flops * 2
    print(f"  Minimum IO -- fwd: {fwd_bytes/1e6:.1f} MB, bwd: {bwd_bytes/1e6:.1f} MB, total: {total_bytes/1e6:.1f} MB")
    print(f"  Arithmetic intensity: {total_flops / total_bytes:.2f} FLOP/byte (memory-bound)")
    print(f"  Theoretical min -- fwd: {fwd_bytes/1e9/mem_bw*1e6:.0f} us, bwd: {bwd_bytes/1e9/mem_bw*1e6:.0f} us, total: {total_bytes/1e9/mem_bw*1e6:.0f} us")
    print()

    from nanoplm.pretraining.models.modern_bert.modeling import _varlen_canon_inner
    from nanoplm.pretraining.models.modern_bert.canon_ops import varlen_canon_conv

    ref_fwd, ref_fwd_bwd, ref_bwd_ms, ref_bwd_gbps, ref_bwd_eff = _bench_compiled(
        "ref", _varlen_canon_inner, args.T, args.C, args.radius, args.n_seqs,
        args.iters, args.warmup, mem_bw,
    )
    new_fwd, new_fwd_bwd, new_bwd_ms, new_bwd_gbps, new_bwd_eff = _bench_compiled(
        "triton", varlen_canon_conv, args.T, args.C, args.radius, args.n_seqs,
        args.iters, args.warmup, mem_bw,
    )

    print()
    print("Results (compiled, torch.compile):")
    _print_result(ref_fwd)
    _print_result(new_fwd)
    print(f"  {'ref_bwd (derived)':40s}  avg={ref_bwd_ms:8.3f}ms  BW={ref_bwd_gbps:7.1f} GB/s  eff={ref_bwd_eff:5.1f}%")
    print(f"  {'triton_bwd (derived)':40s}  avg={new_bwd_ms:8.3f}ms  BW={new_bwd_gbps:7.1f} GB/s  eff={new_bwd_eff:5.1f}%")
    _print_result(ref_fwd_bwd)
    _print_result(new_fwd_bwd)

    print()
    if ref_fwd_bwd.avg_ms > 0 and new_fwd_bwd.avg_ms > 0:
        speedup = ref_fwd_bwd.avg_ms / new_fwd_bwd.avg_ms
        saved = ref_fwd_bwd.avg_ms - new_fwd_bwd.avg_ms
        print(f"  fwd+bwd speedup: {speedup:.2f}x  ({saved:+.3f} ms/call)")

        if ref_bwd_ms > 0 and new_bwd_ms > 0:
            speedup_bwd = ref_bwd_ms / new_bwd_ms
            saved_bwd = ref_bwd_ms - new_bwd_ms
            print(f"  bwd-only speedup: {speedup_bwd:.2f}x  ({saved_bwd:+.3f} ms/call)")

        n_calls = 24
        ref_step = ref_fwd_bwd.avg_ms * n_calls
        new_step = new_fwd_bwd.avg_ms * n_calls
        base_step = 573.46
        print()
        print(f"  Per-step projection ({n_calls} Canon calls):")
        print(f"    ref Canon overhead: {ref_step:.1f} ms  ({ref_step/base_step*100:.1f}% of base)")
        print(f"    triton Canon overhead: {new_step:.1f} ms  ({new_step/base_step*100:.1f}% of base)")
        print(f"    step time: ref={base_step + ref_step:.1f}ms, triton={base_step + new_step:.1f}ms")
        print(f"    recovered: {ref_step - new_step:.1f} ms/step")


if __name__ == "__main__":
    main()
