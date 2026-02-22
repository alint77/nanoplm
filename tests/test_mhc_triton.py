import pytest
import torch
from torch.autograd import gradcheck

from nanoplm.pretraining.models.modern_bert.triton_mhc import (
    mhc_post_distribute,
    mhc_pre_aggregate,
)


def _pre_ref(x: torch.Tensor, h_pre: torch.Tensor) -> torch.Tensor:
    return (x * h_pre.unsqueeze(-1)).sum(dim=-2)


def _post_ref(
    x: torch.Tensor,
    f_out: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("...ij,...jc->...ic", h_res, x) + h_post.unsqueeze(-1) * f_out.unsqueeze(-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
def test_mhc_pre_post_triton_forward_matches_reference() -> None:
    torch.manual_seed(0)

    t, n, c = 32, 4, 96
    dtype = torch.bfloat16
    device = "cuda"

    x = torch.randn(t, n, c, device=device, dtype=dtype)
    h_pre = torch.sigmoid(torch.randn(t, n, device=device, dtype=torch.float32)).to(dtype)
    f_out = torch.randn(t, c, device=device, dtype=dtype)
    h_post = (2.0 * torch.sigmoid(torch.randn(t, n, device=device, dtype=torch.float32))).to(dtype)
    h_res = torch.randn(t, n, n, device=device, dtype=torch.float32).softmax(dim=-1).to(dtype)

    pre_ref = _pre_ref(x, h_pre)
    pre_tri = mhc_pre_aggregate(x, h_pre, use_triton=True)
    assert torch.allclose(pre_ref.float(), pre_tri.float(), atol=1e-2, rtol=1e-2)

    post_ref = _post_ref(x, f_out, h_post, h_res)
    post_tri = mhc_post_distribute(x, f_out, h_post, h_res, use_triton=True)
    assert torch.allclose(post_ref.float(), post_tri.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels")
def test_mhc_pre_post_triton_backward_matches_reference() -> None:
    torch.manual_seed(1)

    t, n, c = 8, 4, 64
    dtype = torch.float32
    device = "cuda"

    # Pre backward
    x_tri = torch.randn(t, n, c, device=device, dtype=dtype, requires_grad=True)
    h_pre_tri = torch.sigmoid(torch.randn(t, n, device=device, dtype=dtype)).requires_grad_()

    x_ref = x_tri.detach().clone().requires_grad_()
    h_pre_ref = h_pre_tri.detach().clone().requires_grad_()

    upstream_pre = torch.randn(t, c, device=device, dtype=dtype)

    pre_tri = mhc_pre_aggregate(x_tri, h_pre_tri, use_triton=True)
    pre_ref = _pre_ref(x_ref, h_pre_ref)

    grads_tri = torch.autograd.grad((pre_tri * upstream_pre).sum(), (x_tri, h_pre_tri))
    grads_ref = torch.autograd.grad((pre_ref * upstream_pre).sum(), (x_ref, h_pre_ref))

    assert torch.allclose(grads_tri[0], grads_ref[0], atol=1e-4, rtol=1e-4)
    assert torch.allclose(grads_tri[1], grads_ref[1], atol=1e-4, rtol=1e-4)

    # Post backward
    x_tri = torch.randn(t, n, c, device=device, dtype=dtype, requires_grad=True)
    f_tri = torch.randn(t, c, device=device, dtype=dtype, requires_grad=True)
    h_post_tri = torch.sigmoid(torch.randn(t, n, device=device, dtype=dtype)).requires_grad_()
    h_res_tri = torch.randn(t, n, n, device=device, dtype=dtype).softmax(dim=-1).requires_grad_()

    x_ref = x_tri.detach().clone().requires_grad_()
    f_ref = f_tri.detach().clone().requires_grad_()
    h_post_ref = h_post_tri.detach().clone().requires_grad_()
    h_res_ref = h_res_tri.detach().clone().requires_grad_()

    upstream_post = torch.randn(t, n, c, device=device, dtype=dtype)

    post_tri = mhc_post_distribute(x_tri, f_tri, h_post_tri, h_res_tri, use_triton=True)
    post_ref = _post_ref(x_ref, f_ref, h_post_ref, h_res_ref)

    grads_tri = torch.autograd.grad(
        (post_tri * upstream_post).sum(),
        (x_tri, f_tri, h_post_tri, h_res_tri),
    )
    grads_ref = torch.autograd.grad(
        (post_ref * upstream_post).sum(),
        (x_ref, f_ref, h_post_ref, h_res_ref),
    )

    # grad_x accumulates in a different reduction order in Triton vs torch.einsum.
    # Keep tolerance realistic for float32 GPU reductions.
    for grad_tri, grad_ref in zip(grads_tri, grads_ref):
        assert torch.allclose(grad_tri, grad_ref, atol=2e-3, rtol=5e-3)


def test_mhc_pre_post_gradcheck_cpu_float64() -> None:
    torch.manual_seed(2)

    t, n, c = 4, 4, 16

    x = torch.randn(t, n, c, dtype=torch.float64, requires_grad=True)
    h_pre = torch.randn(t, n, dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        lambda x_, h_: mhc_pre_aggregate(x_, h_, use_triton=False),
        (x, h_pre),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )

    x = torch.randn(t, n, c, dtype=torch.float64, requires_grad=True)
    f_out = torch.randn(t, c, dtype=torch.float64, requires_grad=True)
    h_post = torch.randn(t, n, dtype=torch.float64, requires_grad=True)
    h_res = torch.randn(t, n, n, dtype=torch.float64, requires_grad=True)

    assert gradcheck(
        lambda x_, f_, hp_, hr_: mhc_post_distribute(x_, f_, hp_, hr_, use_triton=False),
        (x, f_out, h_post, h_res),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
