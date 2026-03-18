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
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertForMaskedLM,
        )

        cfg = _tiny_config()
        model = ModernBertForMaskedLM(cfg)
        assert model.model.layers[0].attn_res is not None
        assert model.model.layers[0].mlp_res is None
        assert model.model.layers[1].attn_res is not None
        assert model.model.layers[1].mlp_res is not None
        assert model.model.block_attnres_final is not None
        assert model.model.block_attnres_block_sizes == (1, 2)

    def test_forward_backward(self):
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertForMaskedLM,
        )

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
        from nanoplm.pretraining.models.modern_bert.modeling import (
            ModernBertForMaskedLM,
        )

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
            (
                "use_mhc_lite",
                True,
                "use_block_attnres=true is not compatible with use_mhc_lite=true",
            ),
            (
                "use_resid_lambdas",
                True,
                "use_block_attnres=true is not compatible with use_resid_lambdas=true",
            ),
            (
                "use_x0_lambdas",
                True,
                "use_block_attnres=true is not compatible with use_x0_lambdas=true",
            ),
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

        logits = torch.stack(
            [
                torch.einsum(
                    "...d, d -> ...",
                    F.rms_norm(s, (s.shape[-1],), weight=norm_weight, eps=eps),
                    query,
                )
                for s in sources
            ],
            dim=0,
        )
        alpha = logits.softmax(dim=0)
        result = alpha[0].unsqueeze(-1) * sources[0]
        for i in range(1, len(sources)):
            result = result + alpha[i].unsqueeze(-1) * sources[i]
        return result

    @pytest.mark.parametrize("N", [2, 4, 8])
    def test_forward_agreement(self, N):
        torch.manual_seed(42)
        T, D = 128, 64
        sources = [
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(N)
        ]
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
        norm_weight = torch.ones(D, device="cuda", dtype=torch.bfloat16)

        ref = self._ref_forward(sources, query, norm_weight)

        stacked = torch.stack(sources, dim=0)
        result, alpha, inv_rms = torch.ops.nanoplm_bar.fused_block_attnres(
            stacked,
            query,
            norm_weight,
            1e-5,
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
            stacked,
            query_tri,
            nw_tri,
            1e-5,
        )
        result.sum().backward()

        for j in range(N):
            torch.testing.assert_close(
                sources_tri[j].grad,
                sources_ref[j].grad,
                atol=5e-2,
                rtol=5e-2,
                msg=f"source grad mismatch at j={j}",
            )

    @pytest.mark.parametrize("N", [2, 4])
    def test_backward_param_grads(self, N):
        """Check grad_query and grad_norm_weight."""
        torch.manual_seed(42)
        T, D = 128, 64
        sources = [
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(N)
        ]

        # Reference
        query_ref = (
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
        ).requires_grad_(True)
        nw_ref = torch.ones(D, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        ref = self._ref_forward(sources, query_ref, nw_ref)
        ref.sum().backward()

        # Triton
        query_tri = query_ref.detach().clone().requires_grad_(True)
        nw_tri = nw_ref.detach().clone().requires_grad_(True)
        stacked = torch.stack(sources, dim=0)
        result, _, _ = torch.ops.nanoplm_bar.fused_block_attnres(
            stacked,
            query_tri,
            nw_tri,
            1e-5,
        )
        result.sum().backward()

        # bf16 precision for values ~50-700 is ~0.25-4.0, so use loose atol
        torch.testing.assert_close(
            query_tri.grad,
            query_ref.grad,
            atol=1.0,
            rtol=5e-2,
            msg="query grad mismatch",
        )
        torch.testing.assert_close(
            nw_tri.grad,
            nw_ref.grad,
            atol=1e-1,
            rtol=5e-2,
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
        partial_ref = torch.randn(
            T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        query_ref = (
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
        ).requires_grad_(True)
        nw_ref = torch.ones(D, device="cuda", dtype=torch.bfloat16).requires_grad_(True)

        ref = self._ref_forward([*completed_ref, partial_ref], query_ref, nw_ref)
        ref.sum().backward()

        completed_tri = [s.detach().clone().requires_grad_(True) for s in completed_ref]
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
class TestBatchedDreduction:
    """Phase 1: batched completed-block D-reduction forward equivalence."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    @staticmethod
    def _ref_logits_inv_rms(completed_refs, qw_list, eps=1e-5):
        """Pure PyTorch reference for batched D-reduction outputs."""

        NC = len(completed_refs)
        Q = len(qw_list)
        T = completed_refs[0].shape[0]
        logits = torch.empty(
            Q, NC, T, device=completed_refs[0].device, dtype=torch.float32
        )
        inv_rms = torch.empty(
            NC, T, device=completed_refs[0].device, dtype=torch.float32
        )
        for j in range(NC):
            src = completed_refs[j].float()
            rms = (src.pow(2).mean(dim=-1) + eps).rsqrt()  # (T,)
            inv_rms[j] = rms
            normed = src * rms.unsqueeze(-1)
            for q in range(Q):
                qw = qw_list[q].float()
                logits[q, j] = (normed * qw.unsqueeze(0)).sum(dim=-1)
        return logits, inv_rms

    @pytest.mark.parametrize("NC", [1, 3, 8])
    @pytest.mark.parametrize("Q", [1, 2, 4])
    def test_batched_dreduction_logits_inv_rms(self, NC, Q):
        """Batched kernel logits/inv_rms match per-sublayer reference."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
        )

        torch.manual_seed(42)
        T, D = 128, 64
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01 for _ in range(Q)
        ]

        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)
        ref_logits, ref_inv_rms = self._ref_logits_inv_rms(completed, qw_list)

        torch.testing.assert_close(
            inv_rms,
            ref_inv_rms,
            atol=1e-3,
            rtol=1e-3,
            msg="inv_rms mismatch",
        )
        torch.testing.assert_close(
            logits,
            ref_logits,
            atol=1e-2,
            rtol=1e-2,
            msg="logits mismatch",
        )

    @pytest.mark.parametrize("NC", [1, 3])
    @pytest.mark.parametrize("Q", [1, 2])
    def test_precomputed_forward_matches_standard(self, NC, Q):
        """forward_state_precomputed gives same output as forward_state."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

        ops = [BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)]
        # Give each op a non-trivial query
        for op in ops:
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        # Gather qw and launch batched D-reduction
        qw_list = [op.compute_qw(partial) for op in ops]
        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)

        for q, op in enumerate(ops):
            standard_out = op.forward_state(NC, partial, completed)
            precomputed_out = op.forward_state_precomputed(
                NC,
                partial,
                completed,
                logits[q],
                inv_rms,
            )
            torch.testing.assert_close(
                precomputed_out,
                standard_out,
                atol=1e-2,
                rtol=1e-2,
                msg=f"precomputed vs standard mismatch for sublayer q={q}",
            )

    @pytest.mark.parametrize("NC", [1, 3])
    def test_precomputed_backward_matches_standard(self, NC):
        """Backward through precomputed path produces same grads as standard."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        Q = 2

        # Standard path
        completed_std = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(NC)
        )
        partial_std = torch.randn(
            T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ops_std = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_std):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        loss_std = sum(
            op.forward_state(NC, partial_std, completed_std).sum() for op in ops_std
        )
        loss_std.backward()

        # Precomputed path
        completed_pre = tuple(
            c.detach().clone().requires_grad_(True) for c in completed_std
        )
        partial_pre = partial_std.detach().clone().requires_grad_(True)
        ops_pre = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_pre):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        qw_list = [op.compute_qw(partial_pre) for op in ops_pre]
        logits, inv_rms = batched_completed_dreduction(completed_pre, qw_list, 1e-5)

        loss_pre = sum(
            op.forward_state_precomputed(
                NC, partial_pre, completed_pre, logits[q], inv_rms
            ).sum()
            for q, op in enumerate(ops_pre)
        )
        loss_pre.backward()

        torch.testing.assert_close(
            partial_pre.grad,
            partial_std.grad,
            atol=5e-2,
            rtol=5e-2,
            msg="partial grad mismatch",
        )
        for j in range(NC):
            torch.testing.assert_close(
                completed_pre[j].grad,
                completed_std[j].grad,
                atol=5e-2,
                rtol=5e-2,
                msg=f"completed[{j}] grad mismatch",
            )


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
class TestOnlineMerge:
    """Phase 2: online partial merge — completed weighted sum + merge."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    @staticmethod
    def _ref_completed_wsum(completed_refs, logits_q, eps=1e-5):
        """Pure PyTorch reference for completed weighted sum with online softmax.

        Returns (m, l, acc) running state matching the Triton kernel's output.
        logits_q: (NC, T) f32 — precomputed logits for one query.
        """
        NC = len(completed_refs)
        T, D = completed_refs[0].shape
        m = torch.full(
            (T,), -1e30, device=completed_refs[0].device, dtype=torch.float32
        )
        l = torch.zeros((T,), device=completed_refs[0].device, dtype=torch.float32)
        acc = torch.zeros((T, D), device=completed_refs[0].device, dtype=torch.float32)

        for j in range(NC):
            logit_j = logits_q[j]  # (T,)
            new_m = torch.maximum(m, logit_j)
            correction = torch.exp(m - new_m)
            exp_j = torch.exp(logit_j - new_m)
            acc = (
                acc * correction.unsqueeze(-1)
                + exp_j.unsqueeze(-1) * completed_refs[j].float()
            )
            l = l * correction + exp_j
            m = new_m

        return m, l, acc

    @pytest.mark.parametrize("NC", [1, 3, 8])
    def test_completed_wsum_matches_reference(self, NC):
        """completed_wsum kernel output matches pure-PyTorch online softmax."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            completed_wsum,
        )

        torch.manual_seed(42)
        T, D = 128, 64
        Q = 2
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01 for _ in range(Q)
        ]

        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)

        # Test for each query
        for q in range(Q):
            running_m, running_l, running_acc = completed_wsum(
                completed, logits[q], inv_rms, 1e-5
            )
            ref_m, ref_l, ref_acc = self._ref_completed_wsum(completed, logits[q])

            torch.testing.assert_close(
                running_m,
                ref_m,
                atol=1e-3,
                rtol=1e-3,
                msg=f"running_m mismatch for q={q}",
            )
            torch.testing.assert_close(
                running_l,
                ref_l,
                atol=1e-3,
                rtol=1e-3,
                msg=f"running_l mismatch for q={q}",
            )
            torch.testing.assert_close(
                running_acc,
                ref_acc,
                atol=1e-2,
                rtol=1e-2,
                msg=f"running_acc mismatch for q={q}",
            )

    @pytest.mark.parametrize("NC", [1, 3])
    @pytest.mark.parametrize("Q", [1, 2])
    def test_online_merge_forward_matches_standard(self, NC, Q):
        """Full pipeline (dreduction → wsum → merge) gives same result as forward_state."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            completed_wsum,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

        ops = [BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)]
        for op in ops:
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        # Gather qw and launch batched D-reduction
        qw_list = [op.compute_qw(partial) for op in ops]
        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)

        for q, op in enumerate(ops):
            # Standard path
            standard_out = op.forward_state(NC, partial, completed)

            # Phase 2 online merge path
            running_m, running_l, running_acc = completed_wsum(
                completed, logits[q], inv_rms, 1e-5
            )
            merge_out = op.forward_state_online_merge(
                NC,
                partial,
                completed,
                running_m,
                running_l,
                running_acc,
                logits[q],
                inv_rms,
            )

            torch.testing.assert_close(
                merge_out,
                standard_out,
                atol=1e-2,
                rtol=1e-2,
                msg=f"online merge vs standard mismatch for sublayer q={q}",
            )

    @pytest.mark.parametrize("NC", [1, 3])
    def test_online_merge_backward_matches_standard(self, NC):
        """Backward through online merge path produces same grads as standard."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            completed_wsum,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        Q = 2

        # Standard path
        completed_std = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(NC)
        )
        partial_std = torch.randn(
            T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ops_std = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_std):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        loss_std = sum(
            op.forward_state(NC, partial_std, completed_std).sum() for op in ops_std
        )
        loss_std.backward()

        # Online merge path
        completed_mrg = tuple(
            c.detach().clone().requires_grad_(True) for c in completed_std
        )
        partial_mrg = partial_std.detach().clone().requires_grad_(True)
        ops_mrg = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_mrg):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        qw_list = [op.compute_qw(partial_mrg) for op in ops_mrg]
        logits, inv_rms = batched_completed_dreduction(completed_mrg, qw_list, 1e-5)

        loss_mrg = torch.tensor(0.0, device="cuda")
        for q, op in enumerate(ops_mrg):
            running_m, running_l, running_acc = completed_wsum(
                completed_mrg, logits[q], inv_rms, 1e-5
            )
            out = op.forward_state_online_merge(
                NC,
                partial_mrg,
                completed_mrg,
                running_m,
                running_l,
                running_acc,
                logits[q],
                inv_rms,
            )
            loss_mrg = loss_mrg + out.sum()
        loss_mrg.backward()

        torch.testing.assert_close(
            partial_mrg.grad,
            partial_std.grad,
            atol=5e-2,
            rtol=5e-2,
            msg="partial grad mismatch",
        )
        for j in range(NC):
            torch.testing.assert_close(
                completed_mrg[j].grad,
                completed_std[j].grad,
                atol=5e-2,
                rtol=5e-2,
                msg=f"completed[{j}] grad mismatch",
            )


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
class TestBatchedCompletedWsum:
    """Phase 3: batched completed weighted sum with query-pair batching."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    @staticmethod
    def _ref_completed_wsum(completed_refs, logits_q, eps=1e-5):
        """Pure PyTorch reference for completed weighted sum (same as TestOnlineMerge)."""
        NC = len(completed_refs)
        T, D = completed_refs[0].shape
        m = torch.full(
            (T,), -1e30, device=completed_refs[0].device, dtype=torch.float32
        )
        l = torch.zeros((T,), device=completed_refs[0].device, dtype=torch.float32)
        acc = torch.zeros((T, D), device=completed_refs[0].device, dtype=torch.float32)

        for j in range(NC):
            logit_j = logits_q[j]  # (T,)
            new_m = torch.maximum(m, logit_j)
            correction = torch.exp(m - new_m)
            exp_j = torch.exp(logit_j - new_m)
            acc = (
                acc * correction.unsqueeze(-1)
                + exp_j.unsqueeze(-1) * completed_refs[j].float()
            )
            l = l * correction + exp_j
            m = new_m

        return m, l, acc

    @pytest.mark.parametrize("Q", [1, 2, 3, 4])
    @pytest.mark.parametrize("NC", [1, 3, 8])
    def test_batched_completed_wsum_matches_per_query(self, Q, NC):
        """batched_completed_wsum output matches per-query PyTorch reference."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            batched_completed_wsum,
        )

        torch.manual_seed(42)
        T, D = 128, 64
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01 for _ in range(Q)
        ]

        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)
        running_m, running_l, running_acc = batched_completed_wsum(
            completed, logits, inv_rms, 1e-5
        )

        assert running_m.shape == (Q, T)
        assert running_l.shape == (Q, T)
        assert running_acc.shape == (Q, T, D)

        for q in range(Q):
            ref_m, ref_l, ref_acc = self._ref_completed_wsum(completed, logits[q])

            torch.testing.assert_close(
                running_m[q],
                ref_m,
                atol=1e-3,
                rtol=1e-3,
                msg=f"running_m mismatch for q={q}",
            )
            torch.testing.assert_close(
                running_l[q],
                ref_l,
                atol=1e-3,
                rtol=1e-3,
                msg=f"running_l mismatch for q={q}",
            )
            torch.testing.assert_close(
                running_acc[q],
                ref_acc,
                atol=1e-2,
                rtol=1e-2,
                msg=f"running_acc mismatch for q={q}",
            )

    @pytest.mark.parametrize("NC", [1, 3])
    @pytest.mark.parametrize("Q", [1, 2, 3, 4])
    def test_batched_wsum_forward_matches_standard(self, NC, Q):
        """Full pipeline (dreduction -> batched_wsum -> merge) matches forward_state."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            batched_completed_wsum,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

        ops = [BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)]
        for op in ops:
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        qw_list = [op.compute_qw(partial) for op in ops]
        logits, inv_rms = batched_completed_dreduction(completed, qw_list, 1e-5)

        # Phase 3 batched path
        running_m, running_l, running_acc = batched_completed_wsum(
            completed, logits, inv_rms, 1e-5
        )

        for q, op in enumerate(ops):
            # Standard path
            standard_out = op.forward_state(NC, partial, completed)

            # Phase 3 online merge path
            merge_out = op.forward_state_online_merge(
                NC,
                partial,
                completed,
                running_m[q],
                running_l[q],
                running_acc[q],
                logits[q],
                inv_rms,
            )

            torch.testing.assert_close(
                merge_out,
                standard_out,
                atol=1e-2,
                rtol=1e-2,
                msg=f"batched wsum merge vs standard mismatch for q={q}",
            )

    @pytest.mark.parametrize("NC", [1, 3])
    def test_batched_wsum_backward_matches_standard(self, NC):
        """Backward through batched wsum + merge matches standard backward."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            batched_completed_wsum,
        )
        from nanoplm.pretraining.models.modern_bert.modeling import BlockAttnResOp

        torch.manual_seed(42)
        T, D = 128, 64
        Q = 2

        # Standard path
        completed_std = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(NC)
        )
        partial_std = torch.randn(
            T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        ops_std = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_std):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        loss_std = sum(
            op.forward_state(NC, partial_std, completed_std).sum() for op in ops_std
        )
        loss_std.backward()

        # Batched wsum merge path
        completed_mrg = tuple(
            c.detach().clone().requires_grad_(True) for c in completed_std
        )
        partial_mrg = partial_std.detach().clone().requires_grad_(True)
        ops_mrg = [
            BlockAttnResOp(D, eps=1e-5).cuda().to(torch.bfloat16) for _ in range(Q)
        ]
        for i, op in enumerate(ops_mrg):
            torch.manual_seed(100 + i)
            op.query.data = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            op.norm_weight.data = (
                torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5
            )

        qw_list = [op.compute_qw(partial_mrg) for op in ops_mrg]
        logits, inv_rms = batched_completed_dreduction(completed_mrg, qw_list, 1e-5)
        running_m, running_l, running_acc = batched_completed_wsum(
            completed_mrg, logits, inv_rms, 1e-5
        )

        loss_mrg = torch.tensor(0.0, device="cuda")
        for q, op in enumerate(ops_mrg):
            out = op.forward_state_online_merge(
                NC,
                partial_mrg,
                completed_mrg,
                running_m[q],
                running_l[q],
                running_acc[q],
                logits[q],
                inv_rms,
            )
            loss_mrg = loss_mrg + out.sum()
        loss_mrg.backward()

        torch.testing.assert_close(
            partial_mrg.grad,
            partial_std.grad,
            atol=5e-2,
            rtol=5e-2,
            msg="partial grad mismatch (batched wsum)",
        )
        for j in range(NC):
            torch.testing.assert_close(
                completed_mrg[j].grad,
                completed_std[j].grad,
                atol=5e-2,
                rtol=5e-2,
                msg=f"completed[{j}] grad mismatch (batched wsum)",
            )


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
class TestBatchedStateBwd:
    """Phase 4: batched backward — Q_BATCH sublayers sharing completed refs."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    @staticmethod
    def _make_inputs(T, D, NC, Q_BATCH, seed=42):
        """Create random inputs for backward testing.

        Returns completed_refs (shared), and per-sublayer lists of
        (grad_result, partial, query, norm_weight, alpha, inv_rms).
        """
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            _fused_block_attnres_state_cuda,
        )

        torch.manual_seed(seed)
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        partials = []
        queries = []
        norm_weights = []
        grad_results = []
        alphas = []
        inv_rmss = []

        for q in range(Q_BATCH):
            torch.manual_seed(seed + 100 + q)
            partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
            query = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01
            nw = torch.randn(D, device="cuda", dtype=torch.bfloat16).abs() + 0.5

            # Run forward to get alpha and inv_rms
            result, alpha, inv_rms = _fused_block_attnres_state_cuda(
                completed, partial, query, nw, 1e-5
            )

            grad_result = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

            partials.append(partial)
            queries.append(query)
            norm_weights.append(nw)
            grad_results.append(grad_result)
            alphas.append(alpha)
            inv_rmss.append(inv_rms)

        return (
            completed,
            partials,
            queries,
            norm_weights,
            grad_results,
            alphas,
            inv_rmss,
        )

    @pytest.mark.parametrize("NC", [1, 3, 8])
    @pytest.mark.parametrize("Q_BATCH", [1, 2])
    def test_batched_bwd_matches_per_sublayer(self, NC, Q_BATCH):
        """Batched backward matches individual per-sublayer backward calls.

        Verifies:
        - grad_completed (summed) == sum of per-sublayer grad_completed
        - per-sublayer grad_partial matches individually
        - per-sublayer R matches individually
        """
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            _fused_block_attnres_state_bwd_cuda,
            batched_state_bwd,
        )

        T, D = 128, 64
        (
            completed,
            partials,
            queries,
            norm_weights,
            grad_results,
            alphas,
            inv_rmss,
        ) = self._make_inputs(T, D, NC, Q_BATCH)

        # Per-sublayer backward (reference)
        per_sub_grad_completed = []
        per_sub_grad_partial = []
        per_sub_R = []
        for q in range(Q_BATCH):
            gc, gp, R = _fused_block_attnres_state_bwd_cuda(
                grad_results[q],
                completed,
                partials[q],
                queries[q],
                norm_weights[q],
                alphas[q],
                inv_rmss[q],
            )
            per_sub_grad_completed.append(gc)
            per_sub_grad_partial.append(gp)
            per_sub_R.append(R)

        # Batched backward
        batch_gc, batch_gp_list, batch_R_list = batched_state_bwd(
            grad_results,
            completed,
            partials,
            queries,
            norm_weights,
            alphas,
            inv_rmss,
        )

        # Check summed grad_completed
        ref_gc_sum = sum(gc.float() for gc in per_sub_grad_completed)
        torch.testing.assert_close(
            batch_gc.float(),
            ref_gc_sum,
            atol=5e-2,
            rtol=5e-2,
            msg="batched grad_completed (summed) mismatch",
        )

        # Check per-sublayer grad_partial
        for q in range(Q_BATCH):
            torch.testing.assert_close(
                batch_gp_list[q],
                per_sub_grad_partial[q],
                atol=1e-2,
                rtol=1e-2,
                msg=f"grad_partial[{q}] mismatch",
            )

        # Check per-sublayer R
        for q in range(Q_BATCH):
            torch.testing.assert_close(
                batch_R_list[q],
                per_sub_R[q],
                atol=5e-2,
                rtol=5e-2,
                msg=f"R[{q}] mismatch",
            )

    @pytest.mark.parametrize("NC", [1, 3, 8])
    def test_batched_bwd_grad_completed_sum(self, NC):
        """Summed grad_completed from batched kernel equals sum of individual calls."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            _fused_block_attnres_state_bwd_cuda,
            batched_state_bwd,
        )

        T, D = 128, 64
        Q_BATCH = 2
        (
            completed,
            partials,
            queries,
            norm_weights,
            grad_results,
            alphas,
            inv_rmss,
        ) = self._make_inputs(T, D, NC, Q_BATCH)

        # Individual backward calls
        gc_0, _, _ = _fused_block_attnres_state_bwd_cuda(
            grad_results[0],
            completed,
            partials[0],
            queries[0],
            norm_weights[0],
            alphas[0],
            inv_rmss[0],
        )
        gc_1, _, _ = _fused_block_attnres_state_bwd_cuda(
            grad_results[1],
            completed,
            partials[1],
            queries[1],
            norm_weights[1],
            alphas[1],
            inv_rmss[1],
        )
        ref_sum = gc_0.float() + gc_1.float()

        # Batched
        batch_gc, _, _ = batched_state_bwd(
            grad_results,
            completed,
            partials,
            queries,
            norm_weights,
            alphas,
            inv_rmss,
        )

        for j in range(NC):
            torch.testing.assert_close(
                batch_gc[j].float(),
                ref_sum[j],
                atol=5e-2,
                rtol=5e-2,
                msg=f"grad_completed[{j}] sum mismatch",
            )

    @pytest.mark.parametrize("D", [64, 256, 768])
    def test_batched_bwd_various_dims(self, D):
        """Batched backward works across various D dimensions."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            _fused_block_attnres_state_bwd_cuda,
            batched_state_bwd,
        )

        T, NC, Q_BATCH = 64, 2, 2
        (
            completed,
            partials,
            queries,
            norm_weights,
            grad_results,
            alphas,
            inv_rmss,
        ) = self._make_inputs(T, D, NC, Q_BATCH, seed=7)

        # Reference per-sublayer
        per_sub_gc = []
        per_sub_gp = []
        per_sub_R = []
        for q in range(Q_BATCH):
            gc, gp, R = _fused_block_attnres_state_bwd_cuda(
                grad_results[q],
                completed,
                partials[q],
                queries[q],
                norm_weights[q],
                alphas[q],
                inv_rmss[q],
            )
            per_sub_gc.append(gc)
            per_sub_gp.append(gp)
            per_sub_R.append(R)

        # Batched
        batch_gc, batch_gp_list, batch_R_list = batched_state_bwd(
            grad_results,
            completed,
            partials,
            queries,
            norm_weights,
            alphas,
            inv_rmss,
        )

        ref_gc_sum = sum(gc.float() for gc in per_sub_gc)
        torch.testing.assert_close(
            batch_gc.float(),
            ref_gc_sum,
            atol=5e-2,
            rtol=5e-2,
            msg=f"grad_completed sum mismatch (D={D})",
        )
        for q in range(Q_BATCH):
            torch.testing.assert_close(
                batch_gp_list[q],
                per_sub_gp[q],
                atol=1e-2,
                rtol=1e-2,
                msg=f"grad_partial[{q}] mismatch (D={D})",
            )
            torch.testing.assert_close(
                batch_R_list[q],
                per_sub_R[q],
                atol=5e-2,
                rtol=5e-2,
                msg=f"R[{q}] mismatch (D={D})",
            )


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
class TestCompileCompat:
    """Phase 5: verify custom ops trace under torch.compile(fullgraph=True)."""

    @pytest.fixture(autouse=True)
    def _load_ops(self):
        import nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops  # noqa: F401

    def test_state_forward_no_graph_break(self):
        """fused_block_attnres_from_state traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            fused_block_attnres_from_state,
        )

        T, D = 64, 128
        completed = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16)]
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        def fn(p, q, nw, *refs):
            return fused_block_attnres_from_state(p, q, nw, 1e-5, tuple(refs))

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        result = compiled_fn(partial, query, norm_weight, *completed)
        assert result.shape == (T, D)
        assert cnt.frame_count == 1  # single graph, no breaks

    def test_state_precomputed_forward_no_graph_break(self):
        """fused_block_attnres_from_state_precomputed traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            fused_block_attnres_from_state_precomputed,
        )

        T, D, NC = 64, 128, 1
        completed = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16)]
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(NC, T, device="cuda", dtype=torch.float32)
        inv_rms = torch.randn(NC, T, device="cuda", dtype=torch.float32).abs() + 0.1

        def fn(p, q, nw, lg, ir, *refs):
            return fused_block_attnres_from_state_precomputed(
                p, q, nw, lg, ir, 1e-5, tuple(refs)
            )

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        result = compiled_fn(partial, query, norm_weight, logits, inv_rms, *completed)
        assert result.shape == (T, D)
        assert cnt.frame_count == 1

    def test_online_merge_forward_no_graph_break(self):
        """fused_block_attnres_online_merge traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            fused_block_attnres_online_merge,
        )

        T, D, NC = 64, 128, 1
        completed = [torch.randn(T, D, device="cuda", dtype=torch.bfloat16)]
        partial = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
        query = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        running_m = torch.randn(T, device="cuda", dtype=torch.float32)
        running_l = torch.randn(T, device="cuda", dtype=torch.float32).abs() + 0.1
        running_acc = torch.randn(T, D, device="cuda", dtype=torch.float32)
        logits = torch.randn(NC, T, device="cuda", dtype=torch.float32)
        inv_rms = torch.randn(NC, T, device="cuda", dtype=torch.float32).abs() + 0.1

        def fn(p, q, nw, rm, rl, ra, lg, ir, *refs):
            return fused_block_attnres_online_merge(
                p, q, nw, rm, rl, ra, lg, ir, 1e-5, tuple(refs)
            )

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        result = compiled_fn(
            partial,
            query,
            norm_weight,
            running_m,
            running_l,
            running_acc,
            logits,
            inv_rms,
            *completed,
        )
        assert result.shape == (T, D)
        assert cnt.frame_count == 1

    def test_online_merge_backward_compiles(self):
        """Compiled backward must keep the custom Triton state-bwd op opaque."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            fused_block_attnres_online_merge,
        )

        T, D, NC = 64, 128, 1
        completed = [
            torch.randn(
                T,
                D,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
        ]
        partial = torch.randn(
            T,
            D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query = torch.randn(
            D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        norm_weight = torch.randn(
            D,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        running_m = torch.randn(T, device="cuda", dtype=torch.float32)
        running_l = torch.randn(T, device="cuda", dtype=torch.float32).abs() + 0.1
        running_acc = torch.randn(T, D, device="cuda", dtype=torch.float32)
        logits = torch.randn(NC, T, device="cuda", dtype=torch.float32)
        inv_rms = torch.randn(NC, T, device="cuda", dtype=torch.float32).abs() + 0.1

        def fn(p, q, nw, c0, rm, rl, ra, lg, ir):
            return fused_block_attnres_online_merge(
                p, q, nw, rm, rl, ra, lg, ir, 1e-5, (c0,)
            ).float().sum()

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        loss = compiled_fn(
            partial,
            query,
            norm_weight,
            completed[0],
            running_m,
            running_l,
            running_acc,
            logits,
            inv_rms,
        )
        loss.backward()
        assert partial.grad is not None
        assert query.grad is not None
        assert norm_weight.grad is not None
        assert completed[0].grad is not None
        assert cnt.frame_count >= 1

    def test_batched_dreduction_no_graph_break(self):
        """batched_completed_dreduction traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
        )

        T, D, NC, Q = 64, 128, 2, 2
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) for _ in range(Q)
        ]

        def fn(c0, c1, qw0, qw1):
            return batched_completed_dreduction((c0, c1), [qw0, qw1], 1e-5)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        logits, inv_rms = compiled_fn(*completed, *qw_list)
        assert logits.shape == (Q, NC, T)
        assert inv_rms.shape == (NC, T)
        assert cnt.frame_count == 1

    def test_batched_dreduction_async_no_graph_break(self):
        """batched_completed_dreduction_async traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction_async,
        )

        T, D, NC, Q = 64, 128, 2, 2
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) for _ in range(Q)
        ]

        def fn(c0, c1, qw0, qw1):
            return batched_completed_dreduction_async((c0, c1), [qw0, qw1], 1e-5)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        logits, inv_rms = compiled_fn(*completed, *qw_list)
        torch.cuda.synchronize()
        assert logits.shape == (Q, NC, T)
        assert inv_rms.shape == (NC, T)
        assert cnt.frame_count == 1

    def test_batched_precompute_no_graph_break(self):
        """batched_completed_precompute traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_precompute,
        )

        T, D, NC, Q = 64, 128, 2, 2
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) for _ in range(Q)
        ]

        def fn(c0, c1, qw0, qw1):
            return batched_completed_precompute((c0, c1), [qw0, qw1], 1e-5)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        logits, inv_rms, running_m, running_l, running_acc = compiled_fn(
            *completed, *qw_list
        )
        assert logits.shape == (Q, NC, T)
        assert inv_rms.shape == (NC, T)
        assert running_m.shape == (Q, T)
        assert running_l.shape == (Q, T)
        assert running_acc.shape == (Q, T, D)
        assert cnt.frame_count == 1

    def test_batched_wsum_no_graph_break(self):
        """batched_completed_wsum traces without graph breaks."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_wsum,
        )

        T, D, NC, Q = 64, 128, 2, 2
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        logits = torch.randn(Q, NC, T, device="cuda", dtype=torch.float32)
        inv_rms = torch.randn(NC, T, device="cuda", dtype=torch.float32).abs() + 0.1

        def fn(c0, c1, lg, ir):
            return batched_completed_wsum((c0, c1), lg, ir, 1e-5)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        running_m, running_l, running_acc = compiled_fn(*completed, logits, inv_rms)
        assert running_m.shape == (Q, T)
        assert running_l.shape == (Q, T)
        assert running_acc.shape == (Q, T, D)
        assert cnt.frame_count == 1

    def test_batched_precompute_matches_separate_ops(self):
        """Async batched precompute must match explicit dreduction + wsum."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            batched_completed_precompute,
            batched_completed_wsum,
        )

        T, D, NC, Q = 64, 128, 2, 3
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01 for _ in range(Q)
        ]

        logits_ref, inv_rms_ref = batched_completed_dreduction(completed, qw_list, 1e-5)
        running_m_ref, running_l_ref, running_acc_ref = batched_completed_wsum(
            completed, logits_ref, inv_rms_ref, 1e-5
        )
        logits, inv_rms, running_m, running_l, running_acc = (
            batched_completed_precompute(completed, qw_list, 1e-5)
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(logits, logits_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(inv_rms, inv_rms_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_m, running_m_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_l, running_l_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_acc, running_acc_ref, atol=1e-2, rtol=1e-2)

    def test_batched_dreduction_async_matches_sync_then_wsum(self):
        """Async phase 1 must feed the same phase 3 running state as sync phase 1."""
        from nanoplm.pretraining.models.modern_bert.block_attnres_triton_ops import (
            batched_completed_dreduction,
            batched_completed_dreduction_async,
            batched_completed_wsum,
        )

        T, D, NC, Q = 64, 128, 2, 3
        completed = tuple(
            torch.randn(T, D, device="cuda", dtype=torch.bfloat16) for _ in range(NC)
        )
        qw_list = [
            torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.01 for _ in range(Q)
        ]

        logits_ref, inv_rms_ref = batched_completed_dreduction(completed, qw_list, 1e-5)
        running_m_ref, running_l_ref, running_acc_ref = batched_completed_wsum(
            completed, logits_ref, inv_rms_ref, 1e-5
        )

        logits, inv_rms = batched_completed_dreduction_async(completed, qw_list, 1e-5)
        running_m, running_l, running_acc = batched_completed_wsum(
            completed, logits, inv_rms, 1e-5
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(logits, logits_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(inv_rms, inv_rms_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_m, running_m_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_l, running_l_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(running_acc, running_acc_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
def test_compiled_block_attnres_model_tensor_state_no_graph_break():
    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM

    cfg = _tiny_config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
    )
    model = ModernBertForMaskedLM(cfg).cuda().eval()
    input_ids = torch.randint(0, cfg.vocab_size, (64,), device="cuda")
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)

    cnt = torch._dynamo.testing.CompileCounter()
    compiled_model = torch.compile(model.model, backend=cnt, fullgraph=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = compiled_model(input_ids, None, cu_seqlens, 64)
    assert out.shape == (64, cfg.hidden_size)
    assert cnt.frame_count == 1


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
def test_block_attnres_fsdp_disables_forward_input_cast(tmp_path):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM
    from nanoplm.pretraining.pure_pipeline import _fully_shard_transformer_layer

    init_method = f"file://{tmp_path / 'pg'}"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=init_method, rank=0, world_size=1)
    try:
        cfg = _tiny_config(hidden_size=128, intermediate_size=256, num_attention_heads=4)
        model = ModernBertForMaskedLM(cfg).cuda().train()
        mesh = init_device_mesh("cuda", (1,))
        fsdp_kwargs = dict(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        )
        _fully_shard_transformer_layer(
            model.model.layers[0],
            mesh=mesh,
            fsdp_kwargs=fsdp_kwargs,
            sublayer=False,
        )
        fully_shard(
            model.model.layers[1],
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=fsdp_kwargs["mp_policy"],
        )

        attnres_policy = fully_shard.state(model.model.layers[0])._mp_policy
        standard_policy = fully_shard.state(model.model.layers[1])._mp_policy
        assert attnres_policy.cast_forward_inputs is False
        assert standard_policy.cast_forward_inputs is True
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
def test_fsdp_sublayer_mode_skips_parent_layer_wrapper(tmp_path):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM
    from nanoplm.pretraining.pure_pipeline import _fully_shard_transformer_layer

    init_method = f"file://{tmp_path / 'pg'}"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=init_method, rank=0, world_size=1)
    try:
        cfg = _tiny_config(hidden_size=128, intermediate_size=256, num_attention_heads=4)
        model = ModernBertForMaskedLM(cfg).cuda().train()
        mesh = init_device_mesh("cuda", (1,))
        fsdp_kwargs = dict(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        )

        layer = model.model.layers[1]
        _fully_shard_transformer_layer(
            layer,
            mesh=mesh,
            fsdp_kwargs=fsdp_kwargs,
            sublayer=True,
        )

        assert fully_shard.state(layer.attn) is not None
        assert layer.mlp is not None
        assert fully_shard.state(layer.mlp) is not None
        assert fully_shard.state(layer) is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not _has_triton_cuda(), reason="needs CUDA + triton")
def test_fsdp_replicated_attnres_params_work_with_mixed_clip_and_fused_adamw(tmp_path):
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    from nanoplm.pretraining.models.modern_bert.modeling import ModernBertForMaskedLM
    from nanoplm.pretraining.optim import extend_homogeneous_param_groups
    from nanoplm.pretraining.pure_pipeline import (
        _clip_grad_norm_mixed,
        _fully_shard_root_groups,
        _fully_shard_transformer_layer,
        _fsdp_kwargs_with_ignored_params,
    )

    init_method = f"file://{tmp_path / 'pg'}"
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=init_method, rank=0, world_size=1)
    try:
        cfg = _tiny_config()
        model = ModernBertForMaskedLM(cfg).cuda().train()
        mesh = init_device_mesh("cuda", (1,))
        fsdp_kwargs = dict(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        )
        for layer in model.model.layers:
            _fully_shard_transformer_layer(
                layer,
                mesh=mesh,
                fsdp_kwargs=fsdp_kwargs,
                sublayer=False,
            )
        _fully_shard_root_groups(model, mesh=mesh, fsdp_kwargs=fsdp_kwargs)
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            **_fsdp_kwargs_with_ignored_params(fsdp_kwargs, model),
        )
        assert not isinstance(model.model.layers[0].attn_res.query, DTensor)
        assert not isinstance(model.model.layers[0].attn_res.norm_weight, DTensor)
        assert not isinstance(model.model.block_attnres_final.query, DTensor)

        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)
        adamw_groups: list[dict] = []
        extend_homogeneous_param_groups(
            adamw_groups,
            decay,
            weight_decay=0.01,
        )
        extend_homogeneous_param_groups(
            adamw_groups,
            no_decay,
            weight_decay=0.0,
        )
        optimizer = torch.optim.AdamW(
            adamw_groups,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")
        attention_mask = torch.ones((2, 8), device="cuda", dtype=torch.long)
        labels = torch.randint(0, cfg.vocab_size, (2, 8), device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        out["loss"].backward()
        grad_norm = _clip_grad_norm_mixed(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=False,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert isinstance(grad_norm, torch.Tensor)
    finally:
        dist.destroy_process_group()
