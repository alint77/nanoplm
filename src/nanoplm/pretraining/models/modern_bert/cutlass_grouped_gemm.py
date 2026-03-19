from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


_EXTENSION = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _build_directory() -> Path:
    build_dir = _repo_root() / ".cache" / "torch_extensions" / "nanoplm_cutlass_grouped_gemm"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    repo_root = _repo_root()
    cutlass_root = repo_root / "third_party" / "cutlass"
    if not cutlass_root.exists():
        raise RuntimeError(
            "Vendored CUTLASS was not found at "
            f"{cutlass_root}. Expected third_party/cutlass to exist."
        )

    source_root = Path(__file__).resolve().parent / "csrc"
    sources = [
        str(source_root / "moe_cutlass_grouped_gemm.cpp"),
        str(source_root / "moe_cutlass_grouped_gemm.cu"),
    ]
    include_paths = [
        str(cutlass_root / "include"),
    ]

    _EXTENSION = load(
        name="nanoplm_cutlass_grouped_gemm",
        sources=sources,
        extra_include_paths=include_paths,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        with_cuda=True,
        build_directory=str(_build_directory()),
        verbose=os.environ.get("NANOPLM_VERBOSE_CUTLASS_BUILD") == "1",
    )
    return _EXTENSION


def grouped_gemm(a, b, batch_sizes, trans_a: bool = False, trans_b: bool = False):
    return _load_extension().grouped_gemm(a, b, batch_sizes, trans_a, trans_b)
