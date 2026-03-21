"""Targeted H100 sweep for Canon Triton kernels.

Runs a single Canon kernel per invocation and benchmarks a hand-picked launch
set. Results are appended to ``sweep.log`` after every trial so a long run can
be tailed live.

This is intentionally not a full grid search. The candidate sets are biased
toward H100-friendly shapes:
  - 2D streaming kernels prefer BLOCK_C in {64, 128} and modest staging.
  - reduction kernels prefer larger BLOCK_T with smaller BLOCK_C to limit
    accumulator pressure / atomic contention.
  - 1D row kernels sweep a targeted occupancy surface rather than a blind grid.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Callable

import torch


def _add_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


@dataclass(frozen=True)
class LaunchConfig:
    name: str
    block_t: int | None = None
    block_c: int | None = None
    num_warps: int = 4
    num_stages: int | None = None
    program_multiplier: int | None = None

    def summary(self) -> str:
        parts = [self.name]
        if self.block_t is not None:
            parts.append(f"BLOCK_T={self.block_t}")
        if self.block_c is not None:
            parts.append(f"BLOCK_C={self.block_c}")
        parts.append(f"warps={self.num_warps}")
        if self.num_stages is not None:
            parts.append(f"stages={self.num_stages}")
        if self.program_multiplier is not None:
            parts.append(f"prog_mult={self.program_multiplier}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Trial:
    cfg: LaunchConfig
    avg_ms: float
    med_ms: float
    min_ms: float
    max_ms: float


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(message.rstrip() + "\n")


def _time_cuda(fn: Callable[[], None], *, warmup: int, iters: int) -> Trial:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))

    return Trial(
        cfg=LaunchConfig(name="timing_placeholder"),
        avg_ms=statistics.mean(samples),
        med_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def _make_seq_id(T: int, n_seqs: int, *, device: torch.device) -> torch.Tensor:
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
    return torch.searchsorted(cu_seqlens[1:], positions, right=True)


def _candidate_streaming_2d() -> list[LaunchConfig]:
    return [
        LaunchConfig("baseline", block_t=64, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t32_c64_w4_s2", block_t=32, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t32_c64_w8_s2", block_t=32, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t32_c128_w4_s2", block_t=32, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t32_c128_w8_s2", block_t=32, block_c=128, num_warps=8, num_stages=2),
        LaunchConfig("t64_c64_w4_s2", block_t=64, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t64_c64_w8_s2", block_t=64, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t64_c64_w4_s3", block_t=64, block_c=64, num_warps=4, num_stages=3),
        LaunchConfig("t64_c128_w4_s2", block_t=64, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t64_c128_w8_s2", block_t=64, block_c=128, num_warps=8, num_stages=2),
        LaunchConfig("t64_c128_w4_s3", block_t=64, block_c=128, num_warps=4, num_stages=3),
        LaunchConfig("t64_c128_w8_s3", block_t=64, block_c=128, num_warps=8, num_stages=3),
        LaunchConfig("t128_c64_w4_s2", block_t=128, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t128_c64_w8_s2", block_t=128, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t128_c64_w4_s3", block_t=128, block_c=64, num_warps=4, num_stages=3),
        LaunchConfig("t128_c128_w4_s2", block_t=128, block_c=128, num_warps=4, num_stages=2),
    ]


def _candidate_streaming_2d_32_generic() -> list[LaunchConfig]:
    # Generic H100-biased 32-point set for the non-LN 2D streaming kernels.
    return [
        LaunchConfig("baseline", block_t=64, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t16_c32_w1_s1", block_t=16, block_c=32, num_warps=1, num_stages=1),
        LaunchConfig("t16_c32_w1_s2", block_t=16, block_c=32, num_warps=1, num_stages=2),
        LaunchConfig("t16_c64_w2_s1", block_t=16, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t16_c64_w2_s2", block_t=16, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t16_c64_w4_s1", block_t=16, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t16_c128_w2_s1", block_t=16, block_c=128, num_warps=2, num_stages=1),
        LaunchConfig("t16_c128_w4_s1", block_t=16, block_c=128, num_warps=4, num_stages=1),
        LaunchConfig("t32_c32_w1_s1", block_t=32, block_c=32, num_warps=1, num_stages=1),
        LaunchConfig("t32_c32_w1_s2", block_t=32, block_c=32, num_warps=1, num_stages=2),
        LaunchConfig("t32_c64_w1_s1", block_t=32, block_c=64, num_warps=1, num_stages=1),
        LaunchConfig("t32_c64_w1_s2", block_t=32, block_c=64, num_warps=1, num_stages=2),
        LaunchConfig("t32_c64_w2_s1", block_t=32, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t32_c64_w2_s2", block_t=32, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t32_c64_w4_s1", block_t=32, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t32_c64_w4_s2", block_t=32, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t32_c64_w8_s2", block_t=32, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t32_c128_w2_s1", block_t=32, block_c=128, num_warps=2, num_stages=1),
        LaunchConfig("t32_c128_w2_s2", block_t=32, block_c=128, num_warps=2, num_stages=2),
        LaunchConfig("t32_c128_w4_s1", block_t=32, block_c=128, num_warps=4, num_stages=1),
        LaunchConfig("t32_c128_w4_s2", block_t=32, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t32_c128_w8_s2", block_t=32, block_c=128, num_warps=8, num_stages=2),
        LaunchConfig("t64_c32_w1_s1", block_t=64, block_c=32, num_warps=1, num_stages=1),
        LaunchConfig("t64_c64_w2_s1", block_t=64, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t64_c64_w2_s2", block_t=64, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t64_c64_w4_s1", block_t=64, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t64_c64_w4_s2", block_t=64, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t64_c128_w2_s1", block_t=64, block_c=128, num_warps=2, num_stages=1),
        LaunchConfig("t64_c128_w4_s1", block_t=64, block_c=128, num_warps=4, num_stages=1),
        LaunchConfig("t64_c128_w4_s2", block_t=64, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t128_c64_w2_s1", block_t=128, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t128_c64_w4_s1", block_t=128, block_c=64, num_warps=4, num_stages=1),
    ]


def _candidate_streaming_2d_160_generic() -> list[LaunchConfig]:
    seeds = _candidate_streaming_2d_32_generic()

    def score(bt: int, bc: int, nw: int, ns: int) -> tuple[float, int, int, int, int]:
        s = 0.0
        s += {16: 0.0, 32: 0.1, 64: 0.35, 128: 0.9, 256: 1.6}.get(bt, 3.0)
        s += {32: 0.2, 64: 0.0, 128: 0.15, 256: 1.2}.get(bc, 3.0)
        s += {1: 0.25, 2: 0.0, 4: 0.1, 8: 0.8}.get(nw, 3.0)
        s += {1: 0.0, 2: 0.03, 3: 0.1, 4: 0.18}.get(ns, 1.0)
        if bt in (16, 32, 64) and bc in (64, 128) and nw in (2, 4):
            s -= 0.15
        return (s, bt, bc, nw, ns)

    pool: list[LaunchConfig] = []
    for bt in (8, 16, 32, 64, 128, 256):
        for bc in (16, 32, 64, 128, 256):
            for nw in (1, 2, 4, 8):
                for ns in (1, 2, 3, 4):
                    if bt * bc > 8192:
                        continue
                    if nw == 8 and bt >= 128 and bc >= 128:
                        continue
                    pool.append(
                        LaunchConfig(
                            f"t{bt}_c{bc}_w{nw}_s{ns}",
                            block_t=bt,
                            block_c=bc,
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )

    seed_keys = {(c.block_t, c.block_c, c.num_warps, c.num_stages) for c in seeds}
    ranked = sorted(
        (
            c for c in pool
            if (c.block_t, c.block_c, c.num_warps, c.num_stages) not in seed_keys
        ),
        key=lambda c: score(c.block_t, c.block_c, c.num_warps, c.num_stages),
    )
    return seeds + ranked[:128]


def _candidate_streaming_2d_32_h100() -> list[LaunchConfig]:
    # H100-biased expansion for the fused LN+Conv forward kernel.
    #
    # Why these 32:
    # - center the search on the two winning families from the first sweep:
    #   (32,128,4,*) and (32,64,2,*)
    # - add BLOCK_T=16 variants to test whether cutting the live row footprint
    #   helps H100 occupancy even more for this register-heavy kernel.
    # - keep BLOCK_C in {64, 128}; the previous sweep showed larger-T / low-warp
    #   variants can fall into pathological codegen regimes, so we only keep a
    #   few anchor points there.
    # - include the trace-like and fallback configs explicitly so we always
    #   measure them against the improved set.
    return [
        LaunchConfig("trace_like", block_t=32, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("fallback", block_t=64, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t16_c64_w2_s1", block_t=16, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t16_c64_w2_s2", block_t=16, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t16_c64_w4_s1", block_t=16, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t16_c64_w4_s2", block_t=16, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t16_c64_w4_s3", block_t=16, block_c=64, num_warps=4, num_stages=3),
        LaunchConfig("t16_c128_w2_s1", block_t=16, block_c=128, num_warps=2, num_stages=1),
        LaunchConfig("t16_c128_w2_s2", block_t=16, block_c=128, num_warps=2, num_stages=2),
        LaunchConfig("t16_c128_w2_s3", block_t=16, block_c=128, num_warps=2, num_stages=3),
        LaunchConfig("t16_c128_w4_s1", block_t=16, block_c=128, num_warps=4, num_stages=1),
        LaunchConfig("t16_c128_w4_s2", block_t=16, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t16_c128_w4_s3", block_t=16, block_c=128, num_warps=4, num_stages=3),
        LaunchConfig("t8_c64_w2_s1", block_t=8, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t32_c64_w1_s1", block_t=32, block_c=64, num_warps=1, num_stages=1),
        LaunchConfig("t32_c64_w1_s2", block_t=32, block_c=64, num_warps=1, num_stages=2),
        LaunchConfig("t32_c64_w2_s1", block_t=32, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t32_c64_w2_s2", block_t=32, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t32_c64_w2_s3", block_t=32, block_c=64, num_warps=2, num_stages=3),
        LaunchConfig("t32_c64_w4_s1", block_t=32, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t32_c64_w4_s2", block_t=32, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t32_c128_w2_s1", block_t=32, block_c=128, num_warps=2, num_stages=1),
        LaunchConfig("t32_c128_w2_s2", block_t=32, block_c=128, num_warps=2, num_stages=2),
        LaunchConfig("t32_c128_w2_s3", block_t=32, block_c=128, num_warps=2, num_stages=3),
        LaunchConfig("t32_c128_w4_s1", block_t=32, block_c=128, num_warps=4, num_stages=1),
        LaunchConfig("t32_c128_w4_s2", block_t=32, block_c=128, num_warps=4, num_stages=2),
        LaunchConfig("t32_c128_w4_s3", block_t=32, block_c=128, num_warps=4, num_stages=3),
        LaunchConfig("t32_c128_w4_s4", block_t=32, block_c=128, num_warps=4, num_stages=4),
        LaunchConfig("t32_c128_w8_s1", block_t=32, block_c=128, num_warps=8, num_stages=1),
        LaunchConfig("t64_c64_w4_s1", block_t=64, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t64_c64_w4_s2", block_t=64, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t64_c128_w4_s1", block_t=64, block_c=128, num_warps=4, num_stages=1),
    ]


def _candidate_streaming_2d_160_h100() -> list[LaunchConfig]:
    # Expanded H100-biased search for canon_ln_fwd.
    #
    # This adds 128 more candidates beyond the tuned 32-set, for 160 total.
    # We score candidates by a heuristic that rewards:
    # - small/moderate BLOCK_T because this kernel is register-heavy in the T dimension
    # - BLOCK_C near 64/128 because those were the only competitive C-tiles so far
    # - low/moderate warp counts (1/2/4) because 8 warps was consistently bad
    # - low stages because this kernel has no shared-memory pipeline to amortize
    #
    # We still keep a handful of anchor points in the worse regimes so the sweep
    # can falsify the heuristic instead of assuming it is always right.
    seeds = _candidate_streaming_2d_32_h100()

    def score(bt: int, bc: int, nw: int, ns: int) -> tuple[float, int, int, int, int]:
        s = 0.0
        if bt == 16:
            s += 0.0
        elif bt == 32:
            s += 0.3
        elif bt == 24:
            s += 0.6
        elif bt == 48:
            s += 1.1
        elif bt == 64:
            s += 1.6
        else:
            s += 2.5

        if bc == 64:
            s += 0.0
        elif bc == 128:
            s += 0.2
        elif bc == 96:
            s += 0.5
        elif bc == 48:
            s += 0.8
        elif bc == 32:
            s += 1.0
        else:
            s += 1.8

        if nw == 2:
            s += 0.0
        elif nw == 4:
            s += 0.2
        elif nw == 1:
            s += 0.35
        elif nw == 8:
            s += 1.5
        else:
            s += 3.0

        if ns == 1:
            s += 0.0
        elif ns == 2:
            s += 0.05
        elif ns == 3:
            s += 0.15
        else:
            s += 0.3

        # Mild preference for the specific winner families we already observed.
        if bt in (16, 32) and bc == 64 and nw == 2:
            s -= 0.35
        if bt in (16, 32) and bc == 128 and nw == 4:
            s -= 0.2
        return (s, bt, bc, nw, ns)

    pool: list[LaunchConfig] = []
    for bt in (8, 16, 32, 64, 128):
        for bc in (32, 64, 128, 256):
            for nw in (1, 2, 4, 8):
                for ns in (1, 2, 3, 4):
                    # Avoid obviously pathological giant thread-block shapes.
                    if nw == 8 and bt >= 64 and bc >= 128:
                        continue
                    if bt >= 96 and nw <= 2:
                        continue
                    name = f"t{bt}_c{bc}_w{nw}_s{ns}"
                    pool.append(
                        LaunchConfig(
                            name,
                            block_t=bt,
                            block_c=bc,
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )

    seed_keys = {(c.block_t, c.block_c, c.num_warps, c.num_stages) for c in seeds}
    ranked = sorted(
        (
            c for c in pool
            if (c.block_t, c.block_c, c.num_warps, c.num_stages) not in seed_keys
        ),
        key=lambda c: score(c.block_t, c.block_c, c.num_warps, c.num_stages),
    )
    return seeds + ranked[:128]


def _candidate_reduction_2d(*, partial: bool) -> list[LaunchConfig]:
    baseline_name = "baseline"
    baseline = LaunchConfig(
        baseline_name,
        block_t=256,
        block_c=64,
        num_warps=4,
        num_stages=2,
    )
    configs = [
        baseline,
        LaunchConfig("t64_c32_w4_s2", block_t=64, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t64_c32_w8_s2", block_t=64, block_c=32, num_warps=8, num_stages=2),
        LaunchConfig("t64_c64_w4_s2", block_t=64, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t64_c64_w8_s2", block_t=64, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t128_c32_w4_s2", block_t=128, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t128_c32_w8_s2", block_t=128, block_c=32, num_warps=8, num_stages=2),
        LaunchConfig("t128_c64_w4_s2", block_t=128, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t128_c64_w8_s2", block_t=128, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t128_c64_w4_s3", block_t=128, block_c=64, num_warps=4, num_stages=3),
        LaunchConfig("t128_c64_w8_s3", block_t=128, block_c=64, num_warps=8, num_stages=3),
        LaunchConfig("t256_c32_w4_s2", block_t=256, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t256_c32_w8_s2", block_t=256, block_c=32, num_warps=8, num_stages=2),
        LaunchConfig("t256_c64_w4_s2", block_t=256, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t256_c64_w8_s2", block_t=256, block_c=64, num_warps=8, num_stages=2),
        LaunchConfig("t256_c64_w4_s3", block_t=256, block_c=64, num_warps=4, num_stages=3),
    ]
    if partial:
        configs[-1] = LaunchConfig("t256_c64_w8_s3", block_t=256, block_c=64, num_warps=8, num_stages=3)
    return configs


def _candidate_reduction_2d_32(*, partial: bool) -> list[LaunchConfig]:
    baseline = LaunchConfig("baseline", block_t=256, block_c=64, num_warps=4, num_stages=2)
    configs = [
        baseline,
        LaunchConfig("t64_c16_w2_s1", block_t=64, block_c=16, num_warps=2, num_stages=1),
        LaunchConfig("t64_c16_w2_s2", block_t=64, block_c=16, num_warps=2, num_stages=2),
        LaunchConfig("t64_c32_w2_s1", block_t=64, block_c=32, num_warps=2, num_stages=1),
        LaunchConfig("t64_c32_w2_s2", block_t=64, block_c=32, num_warps=2, num_stages=2),
        LaunchConfig("t64_c32_w4_s1", block_t=64, block_c=32, num_warps=4, num_stages=1),
        LaunchConfig("t64_c32_w4_s2", block_t=64, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t64_c64_w2_s1", block_t=64, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t64_c64_w4_s1", block_t=64, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t64_c64_w4_s2", block_t=64, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t128_c16_w2_s1", block_t=128, block_c=16, num_warps=2, num_stages=1),
        LaunchConfig("t128_c16_w2_s2", block_t=128, block_c=16, num_warps=2, num_stages=2),
        LaunchConfig("t128_c32_w2_s1", block_t=128, block_c=32, num_warps=2, num_stages=1),
        LaunchConfig("t128_c32_w2_s2", block_t=128, block_c=32, num_warps=2, num_stages=2),
        LaunchConfig("t128_c32_w4_s1", block_t=128, block_c=32, num_warps=4, num_stages=1),
        LaunchConfig("t128_c32_w4_s2", block_t=128, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t128_c64_w2_s1", block_t=128, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t128_c64_w2_s2", block_t=128, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t128_c64_w4_s1", block_t=128, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t128_c64_w4_s2", block_t=128, block_c=64, num_warps=4, num_stages=2),
        LaunchConfig("t128_c64_w4_s3", block_t=128, block_c=64, num_warps=4, num_stages=3),
        LaunchConfig("t256_c16_w2_s1", block_t=256, block_c=16, num_warps=2, num_stages=1),
        LaunchConfig("t256_c16_w2_s2", block_t=256, block_c=16, num_warps=2, num_stages=2),
        LaunchConfig("t256_c32_w2_s1", block_t=256, block_c=32, num_warps=2, num_stages=1),
        LaunchConfig("t256_c32_w2_s2", block_t=256, block_c=32, num_warps=2, num_stages=2),
        LaunchConfig("t256_c32_w4_s1", block_t=256, block_c=32, num_warps=4, num_stages=1),
        LaunchConfig("t256_c32_w4_s2", block_t=256, block_c=32, num_warps=4, num_stages=2),
        LaunchConfig("t256_c32_w4_s3", block_t=256, block_c=32, num_warps=4, num_stages=3),
        LaunchConfig("t256_c64_w2_s1", block_t=256, block_c=64, num_warps=2, num_stages=1),
        LaunchConfig("t256_c64_w2_s2", block_t=256, block_c=64, num_warps=2, num_stages=2),
        LaunchConfig("t256_c64_w4_s1", block_t=256, block_c=64, num_warps=4, num_stages=1),
        LaunchConfig("t256_c64_w4_s2", block_t=256, block_c=64, num_warps=4, num_stages=2),
    ]
    if partial:
        configs[-1] = LaunchConfig("t256_c64_w8_s2", block_t=256, block_c=64, num_warps=8, num_stages=2)
    return configs


def _candidate_reduction_2d_160(*, partial: bool) -> list[LaunchConfig]:
    seeds = _candidate_reduction_2d_32(partial=partial)

    def score(bt: int, bc: int, nw: int, ns: int) -> tuple[float, int, int, int, int]:
        s = 0.0
        s += {128: 0.0, 256: 0.05, 64: 0.2, 32: 0.7, 512: 1.2}.get(bt, 3.0)
        s += {32: 0.0, 16: 0.08, 64: 0.12, 128: 0.7}.get(bc, 3.0)
        s += {2: 0.0, 4: 0.05, 1: 0.45, 8: 0.6}.get(nw, 3.0)
        s += {1: 0.05, 2: 0.0, 3: 0.06, 4: 0.15}.get(ns, 1.0)
        if bt in (128, 256) and bc in (16, 32, 64) and nw in (2, 4):
            s -= 0.15
        if partial and bt == 256 and bc == 32:
            s -= 0.05
        return (s, bt, bc, nw, ns)

    pool: list[LaunchConfig] = []
    for bt in (32, 64, 128, 256, 512):
        for bc in (16, 32, 64, 128):
            for nw in (1, 2, 4, 8):
                for ns in (1, 2, 3, 4):
                    if bt * bc > 16384:
                        continue
                    if nw == 8 and bt <= 64 and bc <= 16:
                        continue
                    pool.append(
                        LaunchConfig(
                            f"t{bt}_c{bc}_w{nw}_s{ns}",
                            block_t=bt,
                            block_c=bc,
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )

    seed_keys = {(c.block_t, c.block_c, c.num_warps, c.num_stages) for c in seeds}
    ranked = sorted(
        (
            c for c in pool
            if (c.block_t, c.block_c, c.num_warps, c.num_stages) not in seed_keys
        ),
        key=lambda c: score(c.block_t, c.block_c, c.num_warps, c.num_stages),
    )
    return seeds + ranked[:128]


def _candidate_ln_stats() -> list[LaunchConfig]:
    configs: list[LaunchConfig] = []
    for num_warps in (1, 2, 4, 8):
        for num_stages in (1, 2, 3, 4):
            name = f"w{num_warps}_s{num_stages}"
            if num_warps == 4 and num_stages == 1:
                name = "baseline"
            configs.append(LaunchConfig(name, num_warps=num_warps, num_stages=num_stages))
    return configs


def _candidate_ln_stats_32() -> list[LaunchConfig]:
    configs: list[LaunchConfig] = []
    for block_n in (1024, 2048):
        for num_warps in (1, 2, 4, 8):
            for num_stages in (1, 2, 3, 4):
                name = f"bn{block_n}_w{num_warps}_s{num_stages}"
                if block_n == 1024 and num_warps == 4 and num_stages == 1:
                    name = "baseline"
                configs.append(
                    LaunchConfig(
                        name,
                        num_warps=num_warps,
                        num_stages=num_stages,
                        block_c=block_n,
                    )
                )
    return configs


def _candidate_ln_rowwise() -> list[LaunchConfig]:
    configs: list[LaunchConfig] = []
    for prog_mult in (2, 4, 8, 12):
        for num_warps in (1, 2, 4, 8):
            name = f"pm{prog_mult}_w{num_warps}"
            if prog_mult == 8 and num_warps == 4:
                name = "baseline"
            configs.append(
                LaunchConfig(
                    name,
                    num_warps=num_warps,
                    program_multiplier=prog_mult,
                )
            )
    return configs


def _candidate_ln_rowwise_32() -> list[LaunchConfig]:
    prog_mults = (1, 2, 3, 4, 6, 8, 12, 16)
    configs: list[LaunchConfig] = []
    for prog_mult in prog_mults:
        for num_warps in (1, 2, 4, 8):
            name = f"pm{prog_mult}_w{num_warps}"
            if prog_mult == 8 and num_warps == 4:
                name = "baseline"
            configs.append(
                LaunchConfig(
                    name,
                    num_warps=num_warps,
                    program_multiplier=prog_mult,
                )
            )
    return configs


def _candidate_ln_rowwise_160() -> list[LaunchConfig]:
    # Deeper H100 pass for 1D rowwise kernels: vary the residency surface by
    # sweeping many program-multiplier choices while keeping warp counts small.
    prog_mults = (
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 14, 16, 18, 20, 24, 28, 32, 36, 40,
        44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
        88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
    )
    configs: list[LaunchConfig] = []
    for prog_mult in prog_mults:
        for num_warps in (1, 2, 4, 8):
            name = f"pm{prog_mult}_w{num_warps}"
            if prog_mult == 8 and num_warps == 4:
                name = "baseline"
            configs.append(
                LaunchConfig(
                    name,
                    num_warps=num_warps,
                    program_multiplier=prog_mult,
                )
            )
    return configs


def _launcher_bundle(args: argparse.Namespace):
    import triton
    from nanoplm.pretraining.models.modern_bert import canon_triton_kernels as k

    device = torch.device("cuda")
    T = int(args.T)
    C = int(args.C)
    radius = int(args.radius)
    K = 2 * radius + 1
    fp32_accum = True
    seq_id = _make_seq_id(T, args.n_seqs, device=device)
    block_n = triton.next_power_of_2(C)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    acc_dtype = torch.float32

    x = torch.randn((T, C), device=device, dtype=torch.bfloat16).contiguous()
    grad_out = torch.randn((T, C), device=device, dtype=torch.bfloat16).contiguous()
    weight = torch.randn((C, K), device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((C,), device=device, dtype=torch.bfloat16).contiguous()
    ln_weight = torch.ones((C,), device=device, dtype=torch.bfloat16).contiguous()
    mean = torch.empty((T,), device=device, dtype=acc_dtype)
    rstd = torch.empty((T,), device=device, dtype=acc_dtype)
    out = torch.empty_like(x)
    grad_x = torch.empty_like(x)
    grad_w = torch.zeros((C, K), device=device, dtype=acc_dtype)
    grad_b = torch.zeros((C,), device=device, dtype=acc_dtype)

    k._ln_stats_kernel[(T,)](
        x,
        mean,
        rstd,
        T,
        C,
        x.stride(0),
        x.stride(1),
        1e-5,
        BLOCK_N=block_n,
        FP32_ACCUM=fp32_accum,
        num_warps=4,
    )
    torch.cuda.synchronize()

    def streaming_2d_launcher(
        kernel: Callable,
        cfg: LaunchConfig,
        *kernel_args,
    ) -> Callable[[], None]:
        grid = (triton.cdiv(T, cfg.block_t), triton.cdiv(C, cfg.block_c))

        def run() -> None:
            kernel[grid](
                *kernel_args,
                T,
                C,
                x.stride(0),
                x.stride(1),
                RADIUS=radius,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        return run

    def reduction_2d_launcher(
        kernel: Callable,
        cfg: LaunchConfig,
        zero_fn: Callable[[], None],
        *kernel_args,
    ) -> Callable[[], None]:
        grid = (triton.cdiv(T, cfg.block_t), triton.cdiv(C, cfg.block_c))

        def run() -> None:
            zero_fn()
            kernel[grid](
                *kernel_args,
                T,
                C,
                x.stride(0),
                x.stride(1),
                RADIUS=radius,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        return run

    def ln_stats_launcher(cfg: LaunchConfig) -> Callable[[], None]:
        this_block_n = cfg.block_c if cfg.block_c is not None else block_n

        def run() -> None:
            k._ln_stats_kernel[(T,)](
                x,
                mean,
                rstd,
                T,
                C,
                x.stride(0),
                x.stride(1),
                1e-5,
                BLOCK_N=this_block_n,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        return run

    def ln_rowwise_launcher(kernel: Callable, cfg: LaunchConfig, *kernel_args) -> Callable[[], None]:
        num_programs = num_sms * cfg.program_multiplier
        rows_per_program = math.ceil(T / num_programs)
        partial_dgamma = torch.zeros((num_programs, C), device=device, dtype=acc_dtype)

        def run() -> None:
            partial_dgamma.zero_()
            kernel[(num_programs,)](
                *kernel_args,
                grad_x,
                partial_dgamma,
                T,
                C,
                x.stride(0),
                x.stride(1),
                rows_per_program,
                BLOCK_N=block_n,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
            )

        return run

    def fused_ln_rowwise_launcher(cfg: LaunchConfig) -> Callable[[], None]:
        num_programs = num_sms * cfg.program_multiplier
        rows_per_program = math.ceil(T / num_programs)
        partial_dgamma = torch.zeros((num_programs, C), device=device, dtype=acc_dtype)

        def run() -> None:
            partial_dgamma.zero_()
            k._fused_conv_bwd_dx_ln_bwd_kernel[(num_programs,)](
                grad_out,
                x,
                seq_id,
                mean,
                rstd,
                ln_weight,
                weight,
                grad_x,
                partial_dgamma,
                T,
                C,
                x.stride(0),
                x.stride(1),
                rows_per_program,
                RADIUS=radius,
                BLOCK_N=block_n,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
            )

        return run

    def ln_dw_partial_launcher(cfg: LaunchConfig) -> Callable[[], None]:
        max_t_blocks = triton.cdiv(T, cfg.block_t)
        partial_w = torch.zeros((max_t_blocks, C * K), device=device, dtype=acc_dtype)
        partial_b = torch.zeros((max_t_blocks, C), device=device, dtype=acc_dtype)
        grid = (triton.cdiv(T, cfg.block_t), triton.cdiv(C, cfg.block_c))

        def run() -> None:
            partial_w.zero_()
            partial_b.zero_()
            k._canon_ln_bwd_dw_db_partial_kernel[grid](
                grad_out,
                x,
                seq_id,
                mean,
                rstd,
                ln_weight,
                partial_w,
                partial_b,
                T,
                C,
                x.stride(0),
                x.stride(1),
                RADIUS=radius,
                BLOCK_T=cfg.block_t,
                BLOCK_C=cfg.block_c,
                FP32_ACCUM=fp32_accum,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
            )

        return run

    if args.kernel == "canon_ln_fwd" and args.candidate_set == 160:
        ln_fwd_candidates = _candidate_streaming_2d_160_h100()
    elif args.kernel == "canon_ln_fwd" and args.candidate_set == 32:
        ln_fwd_candidates = _candidate_streaming_2d_32_h100()
    else:
        ln_fwd_candidates = _candidate_streaming_2d()

    if args.candidate_set == 160:
        streaming_candidates = _candidate_streaming_2d_160_generic()
        reduction_candidates = _candidate_reduction_2d_160(partial=False)
        reduction_partial_candidates = _candidate_reduction_2d_160(partial=True)
        ln_stats_candidates = _candidate_ln_stats_32()
        ln_rowwise_candidates = _candidate_ln_rowwise_160()
    elif args.candidate_set == 32:
        streaming_candidates = _candidate_streaming_2d_32_generic()
        reduction_candidates = _candidate_reduction_2d_32(partial=False)
        reduction_partial_candidates = _candidate_reduction_2d_32(partial=True)
        ln_stats_candidates = _candidate_ln_stats_32()
        ln_rowwise_candidates = _candidate_ln_rowwise_32()
    else:
        streaming_candidates = _candidate_streaming_2d()
        reduction_candidates = _candidate_reduction_2d(partial=False)
        reduction_partial_candidates = _candidate_reduction_2d(partial=True)
        ln_stats_candidates = _candidate_ln_stats()
        ln_rowwise_candidates = _candidate_ln_rowwise()

    kernels: dict[str, tuple[list[LaunchConfig], Callable[[LaunchConfig], Callable[[], None]]]] = {
        "canon_fwd": (
            streaming_candidates,
            lambda cfg: streaming_2d_launcher(k._canon_fwd_kernel, cfg, x, seq_id, weight, bias, out),
        ),
        "canon_bwd_dx": (
            streaming_candidates,
            lambda cfg: streaming_2d_launcher(k._canon_bwd_dx_kernel, cfg, grad_out, seq_id, weight, grad_x),
        ),
        "canon_bwd_dw_db": (
            reduction_candidates,
            lambda cfg: reduction_2d_launcher(
                k._canon_bwd_dw_db_kernel,
                cfg,
                lambda: (grad_w.zero_(), grad_b.zero_()),
                grad_out,
                x,
                seq_id,
                grad_w,
                grad_b,
            ),
        ),
        "canon_ln_stats": (
            ln_stats_candidates,
            ln_stats_launcher,
        ),
        "canon_ln_fwd": (
            ln_fwd_candidates,
            lambda cfg: streaming_2d_launcher(
                k._canon_ln_fwd_kernel,
                cfg,
                x,
                seq_id,
                mean,
                rstd,
                ln_weight,
                weight,
                bias,
                out,
            ),
        ),
        "canon_ln_bwd": (
            ln_rowwise_candidates,
            lambda cfg: ln_rowwise_launcher(
                k._ln_bwd_kernel,
                cfg,
                grad_out,
                x,
                mean,
                rstd,
                ln_weight,
            ),
        ),
        "canon_fused_conv_ln_bwd": (
            ln_rowwise_candidates,
            fused_ln_rowwise_launcher,
        ),
        "canon_ln_bwd_dw_db": (
            reduction_partial_candidates,
            ln_dw_partial_launcher,
        ),
    }
    return kernels, num_sms


def main() -> None:
    _add_repo_to_path()

    parser = argparse.ArgumentParser(description="Targeted H100 sweep for a single Canon Triton kernel")
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=[
            "canon_fwd",
            "canon_bwd_dx",
            "canon_bwd_dw_db",
            "canon_ln_stats",
            "canon_ln_fwd",
            "canon_ln_bwd",
            "canon_fused_conv_ln_bwd",
            "canon_ln_bwd_dw_db",
        ],
        help="Single kernel to benchmark.",
    )
    parser.add_argument("--T", type=int, default=65536)
    parser.add_argument("--C", type=int, default=768)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--n-seqs", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--candidate-set", type=int, choices=(16, 32, 160), default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default="sweep.log")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log_path = Path(args.log_file)
    device_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability()
    kernels, num_sms = _launcher_bundle(args)
    candidates, launcher_for = kernels[args.kernel]

    header = (
        f"{_timestamp()} START kernel={args.kernel} device={device_name} cc={cc_major}.{cc_minor} "
        f"sms={num_sms} T={args.T} C={args.C} radius={args.radius} n_seqs={args.n_seqs} "
        f"warmup={args.warmup} iters={args.iters} candidates={len(candidates)}"
    )
    print(header, flush=True)
    _append_log(log_path, header)

    trials: list[Trial] = []
    failed = 0
    for idx, cfg in enumerate(candidates, start=1):
        line_prefix = f"{_timestamp()} kernel={args.kernel} trial={idx}/{len(candidates)} cfg=\"{cfg.summary()}\""
        try:
            fn = launcher_for(cfg)
            timed = _time_cuda(fn, warmup=args.warmup, iters=args.iters)
            trial = Trial(
                cfg=cfg,
                avg_ms=timed.avg_ms,
                med_ms=timed.med_ms,
                min_ms=timed.min_ms,
                max_ms=timed.max_ms,
            )
            trials.append(trial)
            msg = (
                f"{line_prefix} status=ok avg_ms={trial.avg_ms:.4f} "
                f"med_ms={trial.med_ms:.4f} min_ms={trial.min_ms:.4f} max_ms={trial.max_ms:.4f}"
            )
        except Exception as exc:
            failed += 1
            msg = f"{line_prefix} status=failed error={type(exc).__name__}: {exc}"
        print(msg, flush=True)
        _append_log(log_path, msg)

    if not trials:
        raise RuntimeError(f"No valid configs succeeded for {args.kernel}")

    trials.sort(key=lambda t: t.avg_ms)
    best = trials[0]
    baseline = next((t for t in trials if t.cfg.name == "baseline"), None)
    summary = (
        f"{_timestamp()} DONE kernel={args.kernel} best=\"{best.cfg.summary()}\" best_avg_ms={best.avg_ms:.4f} "
        f"failed={failed}"
    )
    if baseline is not None:
        speedup = baseline.avg_ms / best.avg_ms if best.avg_ms > 0 else float("inf")
        summary += f" baseline_avg_ms={baseline.avg_ms:.4f} baseline_speedup={speedup:.3f}x"
    print(summary, flush=True)
    _append_log(log_path, summary)

    print("\nTop configs:")
    for trial in trials[: max(1, args.topk)]:
        print(
            f"  avg={trial.avg_ms:8.4f} ms  med={trial.med_ms:8.4f} ms  "
            f"min={trial.min_ms:8.4f} ms  {trial.cfg.summary()}"
        )

    payload = {
        "kernel": args.kernel,
        "device_name": device_name,
        "cc": [cc_major, cc_minor],
        "num_sms": num_sms,
        "shape": {
            "T": args.T,
            "C": args.C,
            "radius": args.radius,
            "n_seqs": args.n_seqs,
        },
        "timing": {"warmup": args.warmup, "iters": args.iters},
        "failed": failed,
        "best": {
            "cfg": asdict(best.cfg),
            "avg_ms": best.avg_ms,
            "med_ms": best.med_ms,
            "min_ms": best.min_ms,
            "max_ms": best.max_ms,
        },
        "baseline": None if baseline is None else {
            "cfg": asdict(baseline.cfg),
            "avg_ms": baseline.avg_ms,
            "med_ms": baseline.med_ms,
            "min_ms": baseline.min_ms,
            "max_ms": baseline.max_ms,
        },
        "top": [
            {
                "cfg": asdict(trial.cfg),
                "avg_ms": trial.avg_ms,
                "med_ms": trial.med_ms,
                "min_ms": trial.min_ms,
                "max_ms": trial.max_ms,
            }
            for trial in trials[: max(1, args.topk)]
        ],
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON results: {out_path}")


if __name__ == "__main__":
    main()
