#!/usr/bin/env python3
"""Downstream benchmark script for TE-trained nanoPLM checkpoints.

Loads a pure-TE checkpoint directly using the TEModernBertForMaskedLM model
and runs biotrainer's autoeval pipeline to measure downstream task
performance on PBC or FLIP benchmarks.

Usage
-----
    python te_downstream_benchmark.py \
        --checkpoint-dir output/pretraining_checkpoints/run-17021143/checkpoint-5000 \
        --framework pbc \
        --output-dir autoeval_output \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Generator, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# -- nanoplm imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nanoplm.pretraining.models.modern_bert.model import ProtModernBertMLMConfig
from nanoplm.pretraining.models.modern_bert.pure_model import TEProtModernBertMLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

# -- biotrainer imports ------------------------------------------------------
from biotrainer.autoeval import autoeval_pipeline


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_te_checkpoint(
    checkpoint_dir: str | Path,
    config: ProtModernBertMLMConfig,
    device: torch.device,
) -> TEProtModernBertMLM:
    """Instantiate TEProtModernBertMLM and load checkpoint weights directly."""
    checkpoint_dir = Path(checkpoint_dir)
    model_path = checkpoint_dir / "pytorch_model.bin"

    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"Loading TE checkpoint from {model_path} ...")
    te_sd = torch.load(model_path, map_location="cpu", weights_only=True)

    # Skip _extra_state keys — these hold FP8 calibration/scaling data from
    # training and are not needed for inference (and cause CUDA kernel errors
    # when FP8 autocast is not active).
    clean_sd = {k: v for k, v in te_sd.items() if "_extra_state" not in k}
    skipped = len(te_sd) - len(clean_sd)
    if skipped:
        print(f"  Skipped {skipped} FP8 _extra_state entries")

    model = TEProtModernBertMLM(config)
    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {n_params:.1f}M parameters on {device}")
    return model


# ---------------------------------------------------------------------------
# Embedding functions for biotrainer autoeval
# ---------------------------------------------------------------------------

def _tokenize_batch(
    tokenizer: ProtModernBertTokenizer,
    batch_seqs: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of amino-acid sequences for inference."""
    spaced = [" ".join(list(s)) for s in batch_seqs]
    encoded = tokenizer(
        spaced,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def _run_backbone(
    model: TEProtModernBertMLM,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the TE backbone to get hidden states (batch, seq_len, hidden).

    Uses the TEModernBertModel encoder directly (no prediction head / decoder).
    The TE model runs in flat/varlen mode internally, so we unpad → encode →
    re-pad to get a (batch, seq_len, hidden) output tensor.
    """
    from nanoplm.pretraining.models.modern_bert.modelling_te import _unpad_input

    batch, seq_len = input_ids.shape

    # Unpad to flat (total_tokens,) layout expected by TE encoder
    indices, cu_seqlens, max_seqlen_t = _unpad_input(attention_mask)
    flat_ids = input_ids.view(-1)[indices]
    max_seqlen = int(max_seqlen_t.item())

    # Run backbone in bf16 (no FP8 autocast — inference only)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        hidden_flat = model.model(
            flat_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    # Scatter back to (batch, seq_len, hidden)
    hidden_padded = torch.zeros(
        (batch * seq_len, hidden_flat.shape[-1]),
        device=hidden_flat.device,
        dtype=hidden_flat.dtype,
    )
    hidden_padded[indices] = hidden_flat
    hidden_padded = hidden_padded.view(batch, seq_len, -1)

    return hidden_padded


@torch.no_grad()
def embed_per_residue(
    model: TEProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (sequence, per_residue_embedding) for each input sequence.

    Per-residue embedding shape: (seq_len, hidden_size).
    Special tokens (EOS) are stripped from the output.
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-residue"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = _tokenize_batch(tokenizer, batch_seqs, device)

        hidden = _run_backbone(model, encoded["input_ids"], encoded["attention_mask"])

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Extract only amino acid token embeddings (skip EOS at end)
            emb = hidden[j, :seq_len, :].float().cpu()
            yield seq, emb


@torch.no_grad()
def embed_per_sequence(
    model: TEProtModernBertMLM,
    tokenizer: ProtModernBertTokenizer,
    sequences: Iterable[str],
    device: torch.device,
    batch_size: int = 16,
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Yield (sequence, per_sequence_embedding) for each input sequence.

    Per-sequence embedding shape: (hidden_size,) — mean pool over residue tokens.
    """
    seq_list = list(sequences)

    for i in tqdm(range(0, len(seq_list), batch_size), desc="Embedding per-sequence"):
        batch_seqs = seq_list[i : i + batch_size]
        encoded = _tokenize_batch(tokenizer, batch_seqs, device)

        hidden = _run_backbone(model, encoded["input_ids"], encoded["attention_mask"])

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Mean pool over amino acid tokens only
            emb = hidden[j, :seq_len, :].mean(dim=0).float().cpu()
            yield seq, emb


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_model_config(yaml_path: str | Path = "pretrain.yaml") -> ProtModernBertMLMConfig:
    """Build a ProtModernBertMLMConfig from pretrain.yaml model section."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    m = cfg["model"]
    return ProtModernBertMLMConfig(
        vocab_size=m.get("vocab_size", 32),
        hidden_size=m["hidden_size"],
        intermediate_size=m["intermediate_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        mlp_activation=m.get("mlp_activation", "swiglu"),
        mlp_dropout=m.get("mlp_dropout", 0.0),
        mlp_bias=m.get("mlp_bias", False),
        attention_bias=m.get("attention_bias", False),
        attention_dropout=m.get("attention_dropout", 0.0),
        classifier_activation=m.get("classifier_activation", "gelu"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run downstream benchmarks on a TE-trained nanoPLM checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the TE checkpoint directory (contains pytorch_model.bin).",
    )
    p.add_argument(
        "--framework",
        type=str,
        default="pbc",
        choices=["pbc", "flip"],
        help="Benchmark framework to evaluate on.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="autoeval_output",
        help="Directory to save autoeval results.",
    )
    p.add_argument(
        "--config-yaml",
        type=str,
        default="pretrain.yaml",
        help="Path to pretrain.yaml for model config.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding computation.",
    )
    p.add_argument(
        "--min-seq-length",
        type=int,
        default=0,
        help="Minimum sequence length filter for benchmark datasets.",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2000,
        help="Maximum sequence length filter for benchmark datasets.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detected if not specified).",
    )
    p.add_argument(
        "--embedder-name",
        type=str,
        default=None,
        help="Name for the embedder in the report. Defaults to checkpoint dir name.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config and model
    config = load_model_config(args.config_yaml)
    model = load_te_checkpoint(args.checkpoint_dir, config, device)
    tokenizer = ProtModernBertTokenizer()

    # Embedder name for the report
    embedder_name = args.embedder_name or Path(args.checkpoint_dir).resolve().stem
    print(f"Embedder name: {embedder_name}")

    # Run autoeval
    print(f"\nStarting autoeval pipeline: framework={args.framework}")
    print("=" * 60)

    current_progress = None
    for progress in autoeval_pipeline(
        embedder_name=embedder_name,
        framework=args.framework,
        output_dir=args.output_dir,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        custom_embedding_function_per_residue=lambda seqs: embed_per_residue(
            model, tokenizer, seqs, device, args.batch_size
        ),
        custom_embedding_function_per_sequence=lambda seqs: embed_per_sequence(
            model, tokenizer, seqs, device, args.batch_size
        ),
    ):
        current_progress = progress
        print(
            f"[{progress.completed_tasks}/{progress.total_tasks}] "
            f"{progress.current_framework_name}: {progress.current_task_name}"
        )

    # Final report
    if current_progress and current_progress.final_report:
        print("\n" + "=" * 60)
        print("AUTOEVAL COMPLETE — Final Report Summary")
        print("=" * 60)
        current_progress.final_report.summary()
    else:
        print("\n[WARN] No final report generated.")


if __name__ == "__main__":
    main()
