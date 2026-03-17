import pytest
import torch


def _tiny_config(**overrides):
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

    cfg = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        use_block_attnres=True,
        block_attnres_num_blocks=2,
        use_canon_layers=False,
        use_repo=False,
    )
    cfg.update(overrides)
    return ModernBertConfig(**cfg)


def _tiny_hf_config(**overrides):
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig

    cfg = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=32,
        use_block_attnres=True,
        block_attnres_num_blocks=2,
        use_canon_layers=False,
        use_repo=False,
    )
    cfg.update(overrides)
    return ProtModernBertMLMConfig(**cfg)


def test_block_attnres_op_zero_query_returns_mean():
    from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

    op = BlockAttnResOp(hidden_size=4, eps=1e-5)
    src0 = torch.ones(2, 4)
    src1 = torch.full((2, 4), 3.0)
    out = op([src0, src1])
    assert torch.allclose(out, torch.full_like(out, 2.0))


class TestBlockAttnResModel:
    def test_construction(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _tiny_config()
        model = ModernBertForMaskedLM(cfg)
        assert model.model.layers[0].attn_res is not None
        assert model.model.layers[0].mlp_res is None
        assert model.model.layers[1].attn_res is not None
        assert model.model.layers[1].mlp_res is not None
        assert model.model.block_attnres_final is not None
        assert model.model.block_attnres_block_sizes == (1, 2)

    def test_forward_backward(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _tiny_config()
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out["logits"].shape == (bsz, seq, cfg.vocab_size)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_forward_backward_with_attn_checkpointing(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

        cfg = _tiny_config(
            activation_checkpointing=True,
            activation_checkpointing_mode="attn",
        )
        model = ModernBertForMaskedLM(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)
        attention_mask = torch.ones((bsz, seq), dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        assert out["loss"] is not None
        out["loss"].backward()

    def test_validation_block_count_upper_bound(self):
        from nanoplm.pretraining.models.modern_bert.modeling import ModernBertConfig

        with pytest.raises(ValueError, match="block_attnres_num_blocks must be <="):
            ModernBertConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                use_block_attnres=True,
                block_attnres_num_blocks=2,
                use_canon_layers=False,
            )

    @pytest.mark.parametrize(
        ("field_name", "value", "match"),
        [
            ("use_mhc_lite", True, "use_block_attnres=true is not compatible with use_mhc_lite=true"),
            ("use_resid_lambdas", True, "use_block_attnres=true is not compatible with use_resid_lambdas=true"),
            ("use_x0_lambdas", True, "use_block_attnres=true is not compatible with use_x0_lambdas=true"),
        ],
    )
    def test_validation_incompatible_flags(self, field_name, value, match):
        kwargs = {field_name: value}
        with pytest.raises(ValueError, match=match):
            _tiny_config(**kwargs)

    def test_validation_rejects_layer_checkpoint_mode(self):
        with pytest.raises(ValueError, match="activation_checkpointing_mode='layer'"):
            _tiny_config(
                activation_checkpointing=True,
                activation_checkpointing_mode="layer",
            )


def test_hf_wrapper_rejects_block_attnres():
    from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLM

    cfg = _tiny_hf_config()
    with pytest.raises(ValueError, match="only in the pure-torch path"):
        ProtModernBertMLM(cfg)


def test_te_wrapper_rejects_block_attnres():
    pytest.importorskip("transformer_engine")

    from nanoplm.pretraining.models.modern_bert.pure_model import TEProtModernBertMLM

    cfg = _tiny_hf_config()
    with pytest.raises(ValueError, match="only in the pure-torch path"):
        TEProtModernBertMLM(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Triton kernel tests
# ═══════════════════════════════════════════════════════════════════════════

def _has_triton_cuda():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
class TestBlockAttnResTriton:
    """Numerical agreement between fused Triton and PyTorch reference."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    @staticmethod
    def _ref_forward(sources, query, norm_weight, eps=1e-5):
        """Pure PyTorch reference (no custom ops)."""
        import torch.nn.functional as F

        logits = torch.stack([
            torch.einsum(
                '...d, d -> ...',
                F.rms_norm(s, (s.shape[-1],), weight=norm_weight, eps=eps),
                query,
            )
            for s in sources
        ], dim=0)
        alpha = logits.softmax(dim=0)
        result = alpha[0].unsqueeze(-1) * sources[0]
        for i in range(1, len(sources)):
            result = result + alpha[i].unsqueeze(-1) * sources[i]
        return result

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_forward_agreement(self, N):
        torch.manual_seed(42)
        T, D = 128, 64
        sources = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(N)]
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
        norm_weight = torch.ones(D, device="cuda", dtype=torch.bfloat16)

        ref = self._ref_forward(sources, query, norm_weight)

        stacked = torch.stack(sources, dim=0)
        result, alpha, inv_rms = torch.ops.nanoplm_bar.fused_block_attnres(
            stacked, query, norm_weight, 1e-5,
        )

        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_backward_grad_stacked(self, N):
        """Check grad flows back through sources correctly."""
        torch.manual_seed(42)
        T, D = 128, 64
        sources = [
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(N)
        ]
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
        norm_weight = torch.ones(D, device="cuda", dtype=torch.bfloat16)

        # Reference backward
        query_ref = query.clone().requires_grad_(True)
        nw_ref = norm_weight.clone().requires_grad_(True)
        sources_ref = [s.detach().clone().requires_grad_(True) for s in sources]
        ref = self._ref_forward(sources_ref, query_ref, nw_ref)
        ref.sum().backward()

        # Triton backward
        query_tri = query.clone().requires_grad_(True)
        nw_tri = norm_weight.clone().requires_grad_(True)
        sources_tri = [s.detach().clone().requires_grad_(True) for s in sources]
        stacked = torch.stack(sources_tri, dim=0)  # autograd tracks through stack
        result, _alpha, _inv_rms = torch.ops.nanoplm_bar.fused_block_attnres(
            stacked, query_tri, nw_tri, 1e-5,
        )
        result.sum().backward()

        for j in range(N):
            torch.testing.assert_close(
                sources_tri[j].grad, sources_ref[j].grad,
                atol=5e-2, rtol=5e-2,
                msg=f"source grad mismatch at j={j}",
            )

    @pytest.mark.parametrize("N", [2, 4])
    def test_backward_param_grads(self, N):
        """Check grad_query and grad_norm_weight."""
        torch.manual_seed(42)
        T, D = 128, 64
        sources = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(N)]

        # Reference
        query_ref = (torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_(True)
        nw_ref = torch.ones(D, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        ref = self._ref_forward(sources, query_ref, nw_ref)
        ref.sum().backward()

        # Triton
        query_tri = query_ref.detach().clone().requires_grad_(True)
        nw_tri = nw_ref.detach().clone().requires_grad_(True)
        stacked = torch.stack(sources, dim=0)
        result, _, _ = torch.ops.nanoplm_bar.fused_block_attnres(
            stacked, query_tri, nw_tri, 1e-5,
        )
        result.sum().backward()

        # bf16 precision for values ~50-700 is ~0.25-4.0, so use loose atol
        torch.testing.assert_close(
            query_tri.grad, query_ref.grad, atol=1.0, rtol=5e-2,
            msg="query grad mismatch",
        )
        torch.testing.assert_close(
            nw_tri.grad, nw_ref.grad, atol=1e-1, rtol=5e-2,
            msg="norm_weight grad mismatch",
        )

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_state_path_grad_agreement(self, N):
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            fused_block_attnres_from_state,
        )

        torch.manual_seed(42)
        T, D = 128, 64
        completed_ref = [
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(N - 1)
        ]
        partial_ref = torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        query_ref = (torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01).requires_grad_(True)
        nw_ref = torch.ones(D, device="cuda", dtype=torch.bfloat16).requires_grad_(True)

        ref = self._ref_forward([*completed_ref, partial_ref], query_ref, nw_ref)
        ref.sum().backward()

        completed_tri = [
            s.detach().clone().requires_grad_(True)
            for s in completed_ref
        ]
        partial_tri = partial_ref.detach().clone().requires_grad_(True)
        query_tri = query_ref.detach().clone().requires_grad_(True)
        nw_tri = nw_ref.detach().clone().requires_grad_(True)
        result = fused_block_attnres_from_state(
            partial_tri,
            query_tri,
            nw_tri,
            1e-5,
            tuple(completed_tri),
        )
        result.sum().backward()

        for i in range(N - 1):
            torch.testing.assert_close(
                completed_tri[i].grad,
                completed_ref[i].grad,
                atol=5e-2,
                rtol=5e-2,
                msg=f"completed source grad mismatch at i={i}",
            )
        torch.testing.assert_close(
            partial_tri.grad,
            partial_ref.grad,
            atol=5e-2,
            rtol=5e-2,
            msg="partial grad mismatch",
        )
        torch.testing.assert_close(
            query_tri.grad,
            query_ref.grad,
            atol=1.0,
            rtol=5e-2,
            msg="state-path query grad mismatch",
        )
        torch.testing.assert_close(
            nw_tri.grad,
            nw_ref.grad,
            atol=1e-1,
            rtol=5e-2,
            msg="state-path norm_weight grad mismatch",
        )


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
def test_fsdp_sharded_attnres_params_work_with_standard_clip_and_fused_adamw(tmp_path):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

    init_method = f"file://{tmp_path / 'pg'}"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=init_method, rank=0, world_size=1)
    try:
        cfg = _tiny_config()
        model = ModernBertForMaskedLM(cfg).cuda().train()
        mesh = init_device_mesh("cuda", (1,))
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        )
        assert isinstance(model.model.layers[0].attn_res.query, DTensor)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")
        attention_mask = torch.ones((2, 8), device="cuda", dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=False,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert isinstance(grad_norm, DTensor)
    finally:
        dist.destroy_process_group()
