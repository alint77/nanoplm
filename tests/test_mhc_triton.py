"""Test correctness and benchmark fused Triton mHC kernels vs PyTorch reference."""

import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/nanoplm/src")

from nanoplm.pretraining.models.modern_bert.modeling import (
    _mhc_sinkhorn_unrolled,
    MHCLayer,
    _NormedMLP,
    ModernBertSwiGLUMLP,
    ModernBertConfig,
)
from nanoplm.pretraining.models.modern_bert.mhc_triton import (
    mhc_coeffs_fused,
    mhc_pre_aggregate,
    mhc_post_res_merge,
    FusedMHCLayer,
)


def test_sinkhorn_correctness():
    """Test that the fused Triton kernel matches the PyTorch reference."""
    torch.manual_seed(42)
    T, n, C = 128, 4, 768
    m = n * n + 2 * n  # 24
    K = n * C

    # Simulate inputs: compute mix and invr the same way FusedMHCLayer._coeffs does
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    phi = torch.randn(K, m, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(m, device="cuda", dtype=torch.float32)
    alpha_pre = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    alpha_post = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    alpha_res = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Compute mix and invr (matching FusedMHCLayer._coeffs)
    invr = torch.rsqrt(x.pow(2).float().mean(dim=-1) + 1e-6)  # (T,)
    mix = (x @ phi).float()  # (T, m) — NOT premultiplied by invr

    # Reference path: the kernel applies invr * alpha * mix + bias
    scaled_mix = mix * invr.unsqueeze(-1)
    pre_logits = scaled_mix[:, :n] * alpha_pre + b[:n]
    post_logits = scaled_mix[:, n:2*n] * alpha_post + b[n:2*n]
    res_logits = scaled_mix[:, 2*n:] * alpha_res + b[2*n:]

    ref_pre = torch.sigmoid(pre_logits)
    ref_post = torch.sigmoid(post_logits) * 2.0
    ref_res = _mhc_sinkhorn_unrolled(res_logits.float(), tmax=20, eps=1e-6)

    # Fused path: pass raw mix and invr
    fused_pre, fused_post, fused_res = mhc_coeffs_fused(
        mix, invr, alpha_pre, alpha_post, alpha_res, b,
        n=4, tmax=20, eps=1e-6, post_mult=2.0,
    )

    # Compare
    print("=== Sinkhorn Correctness ===")
    print(f"h_pre  max diff: {(ref_pre - fused_pre).abs().max().item():.2e}")
    print(f"h_post max diff: {(ref_post - fused_post).abs().max().item():.2e}")
    print(f"h_res  max diff: {(ref_res - fused_res).abs().max().item():.2e}")

    assert (ref_pre - fused_pre).abs().max() < 1e-4, "h_pre mismatch!"
    assert (ref_post - fused_post).abs().max() < 1e-4, "h_post mismatch!"
    assert (ref_res - fused_res).abs().max() < 1e-3, "h_res mismatch!"
    print("PASSED\n")


def test_pre_aggregate_correctness():
    """Test pre-aggregate kernel."""
    torch.manual_seed(42)
    T, n, C = 256, 4, 768

    x = torch.randn(T, n, C, device="cuda", dtype=torch.bfloat16)
    h_pre = torch.randn(T, n, device="cuda", dtype=torch.float32)

    # Reference
    ref = (x.float() * h_pre.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)

    # Fused
    fused = mhc_pre_aggregate(x, h_pre)

    diff = (ref.float() - fused.float()).abs().max().item()
    print(f"=== Pre-Aggregate Correctness ===")
    print(f"max diff: {diff:.2e}")
    assert diff < 1e-2, f"Pre-aggregate mismatch: {diff}"
    print("PASSED\n")


def test_post_res_merge_correctness():
    """Test post+res merge kernel."""
    torch.manual_seed(42)
    T, n, C = 256, 4, 768

    x = torch.randn(T, n, C, device="cuda", dtype=torch.bfloat16)
    f_out = torch.randn(T, C, device="cuda", dtype=torch.bfloat16)
    h_res = torch.randn(T, n, n, device="cuda", dtype=torch.float32).abs()  # make positive
    h_res = h_res / h_res.sum(dim=-1, keepdim=True)  # row-normalize
    h_post = torch.randn(T, n, device="cuda", dtype=torch.float32).abs()

    # Reference: h_res @ x + h_post * f_out
    ref = torch.zeros_like(x).float()
    for i in range(n):
        acc = torch.zeros(T, C, device="cuda", dtype=torch.float32)
        for j in range(n):
            acc += h_res[:, i, j].unsqueeze(-1) * x[:, j, :].float()
        ref[:, i, :] = acc + h_post[:, i].unsqueeze(-1) * f_out.float()
    ref = ref.to(torch.bfloat16)

    # Fused
    fused = mhc_post_res_merge(x, f_out, h_res, h_post)

    diff = (ref.float() - fused.float()).abs().max().item()
    print(f"=== Post+Res Merge Correctness ===")
    print(f"max diff: {diff:.2e}")
    assert diff < 1e-1, f"Post+res merge mismatch: {diff}"
    print("PASSED\n")


def test_full_layer_correctness():
    """Test FusedMHCLayer vs MHCLayer end-to-end."""
    torch.manual_seed(42)
    n, C = 4, 768
    T = 128

    config = ModernBertConfig(hidden_size=C)

    # Simple identity sublayer for testing
    class IdentitySublayer(nn.Module):
        def forward(self, x, **kwargs):
            return x * 0.1

    sublayer = IdentitySublayer().cuda()

    # Reference
    ref_layer = MHCLayer(sublayer, n=n, c=C).cuda()

    # Fused
    fused_layer = FusedMHCLayer(sublayer, n=n, c=C).cuda()

    # Copy parameters
    with torch.no_grad():
        fused_layer.phi.copy_(ref_layer.phi)
        fused_layer.b.copy_(ref_layer.b)
        fused_layer.alpha_pre.copy_(ref_layer.alpha_pre)
        fused_layer.alpha_post.copy_(ref_layer.alpha_post)
        fused_layer.alpha_res.copy_(ref_layer.alpha_res)

    x = torch.randn(T, n, C, device="cuda", dtype=torch.bfloat16)

    ref_out = ref_layer(x)
    fused_out = fused_layer(x)

    diff = (ref_out.float() - fused_out.float()).abs().max().item()
    print(f"=== Full Layer Correctness ===")
    print(f"max diff: {diff:.2e}")
    assert diff < 0.5, f"Full layer mismatch: {diff}"
    print("PASSED\n")


def benchmark_comparison():
    """Benchmark fused vs reference MHCLayer."""
    torch.manual_seed(42)
    n, C = 4, 768
    T = 2048  # typical token count

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(C)
            self.linear1 = nn.Linear(C, 2 * C)
            self.linear2 = nn.Linear(C, C)
        def forward(self, x, **kwargs):
            x = self.norm(x)
            a, b = self.linear1(x).chunk(2, dim=-1)
            return self.linear2(torch.nn.functional.silu(a) * b)

    sublayer = DummyMLP().cuda().to(torch.bfloat16)

    ref_layer = MHCLayer(sublayer, n=n, c=C).cuda().to(torch.bfloat16)
    fused_layer = FusedMHCLayer(sublayer, n=n, c=C).cuda().to(torch.bfloat16)

    with torch.no_grad():
        fused_layer.phi.copy_(ref_layer.phi)
        fused_layer.b.copy_(ref_layer.b)
        fused_layer.alpha_pre.copy_(ref_layer.alpha_pre)
        fused_layer.alpha_post.copy_(ref_layer.alpha_post)
        fused_layer.alpha_res.copy_(ref_layer.alpha_res)

    x = torch.randn(T, n, C, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(5):
        _ = ref_layer(x)
        _ = fused_layer(x)
    torch.cuda.synchronize()

    # Benchmark reference
    N_ITERS = 50
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = ref_layer(x)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - t0) / N_ITERS * 1000

    # Benchmark fused
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = fused_layer(x)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / N_ITERS * 1000

    print(f"=== Benchmark (T={T}, C={C}, n={n}) ===")
    print(f"Reference MHCLayer: {ref_time:.3f} ms/iter")
    print(f"Fused MHCLayer:     {fused_time:.3f} ms/iter")
    print(f"Speedup:            {ref_time / fused_time:.2f}x")

    # Also benchmark without mHC (plain sublayer + residual) for baseline
    torch.cuda.synchronize()
    x_single = torch.randn(T, C, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        _ = sublayer(x_single)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = sublayer(x_single)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - t0) / N_ITERS * 1000
    print(f"Baseline (no mHC):  {baseline_time:.3f} ms/iter")
    print(f"Fused overhead vs baseline: {((fused_time / baseline_time) - 1) * 100:.1f}%")


if __name__ == "__main__":
    print("Running mHC Triton kernel tests...\n")
    test_sinkhorn_correctness()
    test_pre_aggregate_correctness()
    test_post_res_merge_correctness()
    test_full_layer_correctness()
    benchmark_comparison()
    print("\nAll tests passed!")
