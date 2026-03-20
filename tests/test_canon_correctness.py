"""Correctness tests for Triton Canon varlen depthwise conv vs reference.

Tests ``varlen_canon_conv`` (Triton kernels registered via torch.library)
against ``_varlen_canon_inner`` (pure-PyTorch roll-based reference) to verify:

  1. Forward outputs match within bf16/fp32 tolerance.
  2. grad_x, grad_weight, grad_bias match the autograd-derived gradients.
  3. Finite-difference gradient check passes (fp64).
  4. Full ModernBertCanonLayer integration works end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


def _import_ops():
    from nanoplm.pretraining.models.modern_bert import canon_ops  # noqa: F401

    return canon_ops


def _import_reference():
    from nanoplm.pretraining.models.modern_bert.modeling import (
        _varlen_canon_inner,
    )

    return _varlen_canon_inner


# ── Helpers ──────────────────────────────────────────────────────────────


def _rand(shape, *, dtype=torch.bfloat16, device=_DEVICE, requires_grad=False):
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


def _make_cu_seqlens(seq_lengths, device=_DEVICE):
    offsets = [0]
    for sl in seq_lengths:
        offsets.append(offsets[-1] + sl)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def _make_seq_id(cu_seqlens):
    T = int(cu_seqlens[-1].item())
    positions = torch.arange(T, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    return torch.searchsorted(cu_seqlens[1:], positions, right=True)


def _assert_close(name, actual, expected, atol, rtol):
    if actual.dtype != expected.dtype:
        actual = actual.to(expected.dtype)
    diff = (actual.float() - expected.float()).abs()
    ref_abs = expected.float().abs().clamp(min=1e-8)
    max_abs = diff.max().item()
    max_rel = (diff / ref_abs).max().item()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    assert ok, (
        f"{name}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} "
        f"(atol={atol}, rtol={rtol}), shapes={tuple(actual.shape)}/{tuple(expected.shape)}"
    )


FWD_ATOL = 1e-3
FWD_RTOL = 1e-3
BWD_ATOL = 1e-2
BWD_RTOL = 1e-2
BWD_ATOL_FP32 = 1e-5
BWD_RTOL_FP32 = 1e-5


# ── Forward tests ────────────────────────────────────────────────────────


class TestForward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ops = _import_ops()
        self.ref_fn = _import_reference()

    @pytest.mark.parametrize("radius", [1, 2, 3])
    @pytest.mark.parametrize(
        "seq_lengths",
        [
            # n_seqs >= 2 only: the model uses F.conv1d for single sequences,
            # and torch.roll wrap-around differs from Triton bounds checking
            # at positions 0 / T-1 when all tokens share the same seq_id.
            [32, 32],
            [10, 20, 34],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9],
        ],
    )
    @pytest.mark.parametrize("C", [64, 768])
    def test_forward_matches_reference(self, C, seq_lengths, radius):
        K = 2 * radius + 1
        cu = _make_cu_seqlens(seq_lengths)
        T = int(cu[-1].item())
        seq_id = _make_seq_id(cu)

        x = _rand((T, C))
        weight = _rand((C, K))
        bias = _rand((C,))

        ref = self.ref_fn(x, seq_id, weight, bias, radius)
        new = self.ops.varlen_canon_conv(x, seq_id, weight, bias, radius)

        _assert_close("fwd", new, ref, FWD_ATOL, FWD_RTOL)

    def test_forward_fp32(self):
        seq_lengths = [20, 30, 14]
        C, radius = 128, 2
        K = 2 * radius + 1
        cu = _make_cu_seqlens(seq_lengths)
        T = int(cu[-1].item())
        seq_id = _make_seq_id(cu)

        x = _rand((T, C), dtype=torch.float32)
        weight = _rand((C, K), dtype=torch.float32)
        bias = _rand((C,), dtype=torch.float32)

        ref = self.ref_fn(x, seq_id, weight, bias, radius)
        new = self.ops.varlen_canon_conv(x, seq_id, weight, bias, radius)

        _assert_close("fwd_fp32", new, ref, 1e-5, 1e-5)

    def test_forward_short_sequences(self):
        seq_lengths = [2, 2, 2, 2]
        C, radius = 32, 1
        K = 2 * radius + 1
        cu = _make_cu_seqlens(seq_lengths)
        T = int(cu[-1].item())
        seq_id = _make_seq_id(cu)

        x = _rand((T, C))
        weight = _rand((C, K))
        bias = _rand((C,))

        ref = self.ref_fn(x, seq_id, weight, bias, radius)
        new = self.ops.varlen_canon_conv(x, seq_id, weight, bias, radius)
        _assert_close("short_seqs", new, ref, FWD_ATOL, FWD_RTOL)


# ── Backward tests ───────────────────────────────────────────────────────


class TestBackward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ops = _import_ops()
        self.ref_fn = _import_reference()

    def _run_backward_comparison(self, seq_lengths, C, radius, dtype=torch.bfloat16):
        K = 2 * radius + 1
        cu = _make_cu_seqlens(seq_lengths)
        T = int(cu[-1].item())
        seq_id = _make_seq_id(cu)

        x_ref = _rand((T, C), dtype=dtype, requires_grad=True)
        w_ref = _rand((C, K), dtype=dtype, requires_grad=True)
        b_ref = _rand((C,), dtype=dtype, requires_grad=True)

        ref_out = self.ref_fn(x_ref, seq_id, w_ref, b_ref, radius)
        ref_out.float().sum().backward()

        x_new = x_ref.detach().clone().requires_grad_(True)
        w_new = w_ref.detach().clone().requires_grad_(True)
        b_new = b_ref.detach().clone().requires_grad_(True)

        new_out = self.ops.varlen_canon_conv(x_new, seq_id, w_new, b_new, radius)
        new_out.float().sum().backward()

        is_fp32 = dtype == torch.float32
        atol = BWD_ATOL_FP32 if is_fp32 else BWD_ATOL
        rtol = BWD_RTOL_FP32 if is_fp32 else BWD_RTOL

        _assert_close("grad_x", x_new.grad, x_ref.grad, atol, rtol)
        _assert_close("grad_w", w_new.grad, w_ref.grad, atol, rtol)
        _assert_close("grad_b", b_new.grad, b_ref.grad, atol, rtol)

    @pytest.mark.parametrize("radius", [1, 2, 3])
    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [32, 32],
            [10, 20, 34],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9],
        ],
    )
    def test_backward_bf16(self, seq_lengths, radius):
        self._run_backward_comparison(seq_lengths, C=64, radius=radius)

    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_backward_fp32(self, radius):
        self._run_backward_comparison(
            [20, 30, 14], C=128, radius=radius, dtype=torch.float32
        )

    def test_backward_large_C(self):
        self._run_backward_comparison([128, 128], C=768, radius=3)

    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_backward_two_tokens_per_seq(self, radius):
        self._run_backward_comparison([2, 2, 2, 2], C=32, radius=radius)


# ── Gradient check (finite differences) ──────────────────────────────────


class TestGradcheck:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ops = _import_ops()

    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_gradcheck(self, radius):
        K = 2 * radius + 1
        T, C = 12, 8
        cu = _make_cu_seqlens([4, 4, 4])
        seq_id = _make_seq_id(cu)

        x = torch.randn(T, C, device=_DEVICE, dtype=torch.float64, requires_grad=True)
        w = torch.randn(C, K, device=_DEVICE, dtype=torch.float64, requires_grad=True)
        b = torch.randn(C, device=_DEVICE, dtype=torch.float64, requires_grad=True)

        def fn(x_, w_, b_):
            return self.ops.varlen_canon_conv(x_, seq_id, w_, b_, radius)

        torch.autograd.gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_multi_sequence(self):
        radius = 2
        K = 2 * radius + 1
        T, C = 16, 6
        cu = _make_cu_seqlens([5, 6, 5])
        seq_id = _make_seq_id(cu)

        x = torch.randn(T, C, device=_DEVICE, dtype=torch.float64, requires_grad=True)
        w = torch.randn(C, K, device=_DEVICE, dtype=torch.float64, requires_grad=True)
        b = torch.randn(C, device=_DEVICE, dtype=torch.float64, requires_grad=True)

        def fn(x_, w_, b_):
            return self.ops.varlen_canon_conv(x_, seq_id, w_, b_, radius)

        torch.autograd.gradcheck(fn, (x, w, b), eps=1e-6, atol=1e-4, rtol=1e-3)


# ── Integration with ModernBertCanonLayer ─────────────────────────────────


class TestCanonLayerIntegration:
    @pytest.fixture(autouse=True)
    def setup(self):
        _import_ops()
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertCanonLayer,
        )

        self.CanonLayer = ModernBertCanonLayer

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_layer_forward_backward(self, kernel_size):
        C = 64
        layer = self.CanonLayer(C, kernel_size=kernel_size).to(_DEVICE).to(torch.bfloat16)
        layer.train()

        cu = _make_cu_seqlens([32, 32])
        T = int(cu[-1].item())

        x = _rand((T, C), requires_grad=True)
        out = layer(x, cu_seqlens=cu)

        assert out.shape == x.shape
        assert out.dtype == x.dtype

        out.float().sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert layer.conv.weight.grad is not None
        assert layer.conv.bias.grad is not None

    def test_layer_padded_path_unchanged(self):
        C, kernel_size = 64, 5
        layer = self.CanonLayer(C, kernel_size=kernel_size).to(_DEVICE).to(torch.bfloat16)
        layer.train()

        B, S = 4, 32
        x = _rand((B, S, C), requires_grad=True)
        out = layer(x)

        assert out.shape == (B, S, C)
        out.float().sum().backward()
        assert x.grad is not None
