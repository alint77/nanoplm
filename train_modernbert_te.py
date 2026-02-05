import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import transformer_engine.pytorch as te

import wandb


import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from nanoplm.pretraining.dataset import LoadShardedFastaMLMDataset
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

# ====================
# Fixed settings (match HF ModernBERT + repo impl)
# ====================
USE_DDP = int(os.environ.get("WORLD_SIZE", "1")) > 1
H100_PEAK_TFLOPS = 989

# Data (fixed to repo pretraining defaults)
TRAIN_HDF5_DIR = "output/data/pretrain_shards/train_hdf5"
VAL_HDF5_DIR = "output/data/pretrain_shards/val_hdf5"
MAX_LENGTH = 512
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ====================
# Model settings (hardcoded)
# ====================
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2048
NUM_LAYERS = 16
NUM_HEADS = 16

# ====================
# Optimization settings
# ====================
BATCH_SIZE = 128
NUM_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.05
EVAL_STEPS_PERCENT = 0.025

SEED = 42

# ====================
# Distributed init
# ====================
from torch.distributed import init_process_group
import torch.distributed as dist

if USE_DDP:
    init_process_group(backend="nccl", device_id=int(os.environ.get("LOCAL_RANK", "0")))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True
    seed_offset = 0

torch.manual_seed(SEED + seed_offset)
random.seed(SEED + seed_offset)
np.random.seed(SEED + seed_offset)

# ====================
# Helper functions
# ====================

def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)


def build_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask
    mask = ~attention_mask.to(torch.bool)
    return mask.unsqueeze(1).unsqueeze(2)


def trunc_normal_init_(weight: torch.Tensor, std: float, cutoff_factor: float):
    cutoff = cutoff_factor * std
    torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-cutoff, b=cutoff)


def make_trunc_init(std: float, cutoff_factor: float):
    def _init(weight: torch.Tensor):
        trunc_normal_init_(weight, std=std, cutoff_factor=cutoff_factor)
    return _init


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LayerNormNoBias(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, None, self.eps)


def create_norm(hidden_size: int, eps: float, bias: bool) -> nn.Module:
    if bias:
        return te.LayerNorm(hidden_size, eps=eps)
    return LayerNormNoBias(hidden_size, eps=eps)




class ModernBertRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, max_seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


class ModernBertMLP(nn.Module):
    def __init__(self):
        super().__init__()
        std_in = 0.02
        std_out = 0.02 / math.sqrt(2.0 * NUM_LAYERS)
        cutoff = 2.0

        self.mlp = te.LayerNormMLP(
            hidden_size=HIDDEN_SIZE,
            ffn_hidden_size=INTERMEDIATE_SIZE,
            bias=False,
            normalization="LayerNorm",
            eps=1e-5,
            activation="geglu",
            init_method=make_trunc_init(std_in, cutoff),
            output_layer_init_method=make_trunc_init(std_out, cutoff),
        )

    def _unwrap(self, out):
        return out[0] if isinstance(out, tuple) else out

    def forward(self, hidden_states: torch.Tensor, is_first_microbatch: bool = False) -> torch.Tensor:
        out = self.mlp(hidden_states, is_first_microbatch=is_first_microbatch)
        return self._unwrap(out)


class ModernBertBlock(nn.Module):
    def __init__(self, layer_number: int, attn_type: str, is_first_layer: bool):
        super().__init__()
        self.hidden_size = HIDDEN_SIZE
        self.num_heads = NUM_HEADS
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.attn_type = attn_type
        self.layer_number = layer_number
        self.is_first_layer = is_first_layer

        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        std_in = 0.02
        std_out = 0.02 / math.sqrt(2.0 * NUM_LAYERS)
        cutoff = 2.0

        if is_first_layer:
            self.attn_norm = None
            self.qkv = te.Linear(
                in_features=HIDDEN_SIZE,
                out_features=3 * HIDDEN_SIZE,
                bias=False,
                init_method=make_trunc_init(std_in, cutoff),
            )
        else:
            self.attn_norm = None
            self.qkv = te.LayerNormLinear(
                in_features=HIDDEN_SIZE,
                out_features=3 * HIDDEN_SIZE,
                bias=False,
                normalization="LayerNorm",
                eps=1e-5,
                init_method=make_trunc_init(std_in, cutoff),
            )

        if attn_type == "local":
            half_window = 128 // 2
            window_size = (half_window, half_window)
        else:
            window_size = (-1, -1)

        self.attn = te.DotProductAttention(
            num_attention_heads=self.num_heads,
            kv_channels=self.head_dim,
            attention_dropout=0.0,
            attn_mask_type="padding",
            window_size=window_size,
            qkv_format="bshd",
            layer_number=layer_number,
        )
        self.attn_proj = te.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=False,
            init_method=make_trunc_init(std_out, cutoff),
        )
        self.attn_dropout = nn.Identity()
        self.mlp = ModernBertMLP()

    def _unwrap(self, out):
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        is_first_microbatch: bool = False,
    ) -> torch.Tensor:
        residual = x

        if self.attn_norm is None:
            x_normed = x
        else:
            x_normed = self.attn_norm(x)

        qkv = self.qkv(x_normed, is_first_microbatch=is_first_microbatch)
        qkv = self._unwrap(qkv)
        bsz, seqlen, _ = qkv.shape
        qkv = qkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_out = self.attn(
            q,
            k,
            v,
            attention_mask=attention_mask,
            attn_mask_type="padding" if attention_mask is not None else "no_mask",
        )
        attn_out = self._unwrap(attn_out)
        attn_out = attn_out.contiguous().view(bsz, seqlen, self.hidden_size)

        attn_out = self.attn_proj(attn_out, is_first_microbatch=is_first_microbatch)
        attn_out = self._unwrap(attn_out)
        x = residual + self.attn_dropout(attn_out)

        residual = x
        mlp_out = self.mlp(x, is_first_microbatch=is_first_microbatch)
        mlp_out = self._unwrap(mlp_out)
        x = residual + mlp_out
        return x


class ModernBertMLMHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        std_out = 0.02 / math.sqrt(2.0 * NUM_LAYERS)
        cutoff = 2.0
        self.dense = te.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=False,
            init_method=make_trunc_init(std_out, cutoff),
        )
        self.norm = create_norm(HIDDEN_SIZE, eps=1e-5, bias=False)
        self.decoder = te.Linear(
            HIDDEN_SIZE,
            vocab_size,
            bias=True,
            init_method=make_trunc_init(std_out, cutoff),
        )

    def _unwrap(self, out):
        return out[0] if isinstance(out, tuple) else out

    def forward(self, hidden_states: torch.Tensor, is_first_microbatch: bool = False) -> torch.Tensor:
        x = self.dense(hidden_states, is_first_microbatch=is_first_microbatch)
        x = self._unwrap(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.decoder(x, is_first_microbatch=is_first_microbatch)
        x = self._unwrap(x)
        return x


class ModernBertForMaskedLM(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tok_embeddings = nn.Embedding(vocab_size, HIDDEN_SIZE, padding_idx=pad_token_id)
        self.emb_norm = create_norm(HIDDEN_SIZE, eps=1e-5, bias=False)
        self.emb_drop = nn.Identity()

        self.rotary_emb_global = ModernBertRotaryEmbedding(
            dim=HIDDEN_SIZE // NUM_HEADS,
            base=160000.0,
        )
        self.rotary_emb_local = ModernBertRotaryEmbedding(
            dim=HIDDEN_SIZE // NUM_HEADS,
            base=10000.0,
        )

        self.layers = nn.ModuleList()
        for i in range(NUM_LAYERS):
            is_first = (i == 0)
            attn_type = "global" if (i % 3 == 0) else "local"
            self.layers.append(
                ModernBertBlock(
                    layer_number=i + 1,
                    attn_type=attn_type,
                    is_first_layer=is_first,
                )
            )

        self.final_norm = create_norm(HIDDEN_SIZE, eps=1e-5, bias=False)
        self.mlm_head = ModernBertMLMHead(vocab_size)

        self._init_weights()
        if True:
            self.mlm_head.decoder.weight = self.tok_embeddings.weight

    def _init_weights(self):
        trunc_normal_init_(self.tok_embeddings.weight, std=0.02, cutoff_factor=2.0)

        for module in self.modules():
            if isinstance(module, (nn.Linear, te.Linear)):
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, (nn.LayerNorm, te.LayerNorm, LayerNormNoBias)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_first_microbatch: bool = False,
    ):
        bsz, seqlen = input_ids.shape

        hidden_states = self.tok_embeddings(input_ids)
        hidden_states = self.emb_norm(hidden_states)
        hidden_states = self.emb_drop(hidden_states)

        attn_mask = build_attention_mask(attention_mask)

        cos_global, sin_global = self.rotary_emb_global(seqlen, hidden_states.device, hidden_states.dtype)
        cos_local, sin_local = self.rotary_emb_local(seqlen, hidden_states.device, hidden_states.dtype)

        for block in self.layers:
            cos = cos_global if block.attn_type == "global" else cos_local
            sin = sin_global if block.attn_type == "global" else sin_local
            hidden_states = block(
                hidden_states,
                attention_mask=attn_mask,
                cos=cos,
                sin=sin,
                is_first_microbatch=is_first_microbatch,
            )

        hidden_states = self.final_norm(hidden_states)

        if labels is None:
            logits = self.mlm_head(hidden_states, is_first_microbatch=is_first_microbatch)
            return logits, None

        flat_labels = labels.view(-1)
        mask = flat_labels != -100
        if mask.any():
            masked_hidden = hidden_states.view(-1, hidden_states.size(-1))[mask]
            masked_labels = flat_labels[mask]
            logits = self.mlm_head(masked_hidden, is_first_microbatch=is_first_microbatch)
            loss = F.cross_entropy(logits, masked_labels)
        else:
            logits = None
            loss = hidden_states.new_tensor(0.0, requires_grad=True)

        return logits, loss


tokenizer = ProtModernBertTokenizer()
model = ModernBertForMaskedLM(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id).to(device)

if USE_DDP:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp_local_rank],
    )
    raw_model = model.module
else:
    raw_model = model

# Parameter count + MFU setup
num_params = sum(p.numel() for p in raw_model.parameters())
print0(f"model params: {num_params / 1e6:.2f}M")
flops_per_token = 6 * num_params
total_peak_flops = H100_PEAK_TFLOPS * 1e12 * ddp_world_size

# data
train_ds = LoadShardedFastaMLMDataset(TRAIN_HDF5_DIR, load_all_in_memory=False)
val_ds = LoadShardedFastaMLMDataset(VAL_HDF5_DIR, load_all_in_memory=False)

train_collator = ProtDataCollatorForLM(
    tokenizer=tokenizer,
    mlm_probability=0.3,
    mask_token_probability=0.8,
    random_token_probability=0.1,
    keep_probability=0.1,
    pad_to_multiple_of=None,
)
val_collator = ProtDataCollatorForLM(
    tokenizer=tokenizer,
    mlm_probability=0.3,
    mask_token_probability=0.8,
    random_token_probability=0.1,
    keep_probability=0.1,
    pad_to_multiple_of=None,
)

train_sampler = None
val_sampler = None
if USE_DDP:
    train_sampler = DistributedSampler(train_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

num_workers = max(1, min(4 * ddp_world_size, (os.cpu_count() or 1) - 2))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=num_workers,
    prefetch_factor=2,
    pin_memory=PIN_MEMORY and device != "cpu",
    persistent_workers=PERSISTENT_WORKERS and num_workers > 0,
    collate_fn=train_collator,
    drop_last=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    shuffle=False,
    num_workers=num_workers,
    prefetch_factor=2,
    pin_memory=PIN_MEMORY and device != "cpu",
    persistent_workers=PERSISTENT_WORKERS and num_workers > 0,
    collate_fn=val_collator,
    drop_last=False,
)

optimizer = torch.optim.AdamW(
    raw_model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    fused=True,
)

grad_accum_steps = max(1, GRADIENT_ACCUMULATION_STEPS)
actual_total_batch_size = BATCH_SIZE * grad_accum_steps * ddp_world_size * MAX_LENGTH
steps_per_epoch = len(train_loader)
update_steps_per_epoch = max(1, math.ceil(steps_per_epoch / grad_accum_steps))
num_training_steps = update_steps_per_epoch * NUM_EPOCHS
warmup_steps = int(num_training_steps * WARMUP_RATIO)
eval_every = max(1, int(num_training_steps * EVAL_STEPS_PERCENT))

print0(f"grad_accum_steps: {grad_accum_steps}")
print0(f"total_batch_size (tokens): {actual_total_batch_size}")
print0(f"steps_per_epoch: {steps_per_epoch} | update_steps_per_epoch: {update_steps_per_epoch}")
print0(f"num_training_steps: {num_training_steps} | warmup_steps: {warmup_steps} | eval_every: {eval_every}")

def _lr_lambda(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if num_training_steps <= warmup_steps:
        return 1.0
    return max(
        0.0,
        float(num_training_steps - step) / float(max(1, num_training_steps - warmup_steps)),
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

if master_process and wandb is not None:
    wandb.init(project="nanoplm-modernbert-te")
    wandb.config.update(
        {
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "num_epochs": NUM_EPOCHS,
            "num_training_steps": num_training_steps,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
        },
        allow_val_change=True,
    )

@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    if val_sampler is not None:
        val_sampler.set_epoch(0)
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        autocast_device = "cuda" if device != "cpu" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=True, dtype=torch.bfloat16):
            _, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                is_first_microbatch=True,
            )
        losses.append(loss.detach().float())
    model.train()
    mean_loss = torch.stack(losses).mean()
    if USE_DDP:
        mean_loss = mean_loss.to(device)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss = mean_loss / ddp_world_size
    return mean_loss.item()

model.train()
optimizer.zero_grad(set_to_none=True)

tlast = time.time()
last_log_step = 0
data_time_accum = 0.0
global_step = 0
micro_step = 0

for epoch in range(NUM_EPOCHS):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    for step_in_epoch, batch in enumerate(train_loader):
        micro_step += 1
        is_accum_boundary = (micro_step % grad_accum_steps == 0) or (step_in_epoch == steps_per_epoch - 1)
        is_first_microbatch = ((micro_step - 1) % grad_accum_steps == 0)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        if USE_DDP:
            model.require_backward_grad_sync = is_accum_boundary

        autocast_device = "cuda" if device != "cpu" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=True, dtype=torch.bfloat16):
            _, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                is_first_microbatch=is_first_microbatch,
            )

        loss = loss / grad_accum_steps
        loss.backward()

        if is_accum_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            should_log = True
            should_eval = (global_step % eval_every == 0)

            if should_log and master_process:
                torch.cuda.synchronize() if device != "cpu" else None
                tnow = time.time()
                dt = tnow - tlast
                steps_since = global_step - last_log_step
                last_log_step = global_step
                tokens_processed = steps_since * actual_total_batch_size
                tok_per_sec = int(tokens_processed / max(dt, 1e-6))
                flops_achieved = flops_per_token * tok_per_sec
                mfu = (flops_achieved / total_peak_flops * 100) if device != "cpu" else 0.0
                loss_value = (loss.detach().float() * grad_accum_steps).item()
                grad_norm_value = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

                print0(
                    f"step {global_step}: loss {loss_value:.4f} | tok/s {tok_per_sec:,} | "
                    f"grad_norm {grad_norm_value:.2f} | mfu_h100 {mfu:.2f}%"
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": loss_value,
                            "train/grad_norm": grad_norm_value,
                            "train/tok_per_sec": tok_per_sec,
                            "mfu_h100": mfu,
                            "lr": scheduler.get_last_lr()[0],
                            "step": global_step,
                        },
                        step=global_step,
                    )
                tlast = tnow
                data_time_accum = 0.0

            if should_eval:
                if USE_DDP:
                    dist.barrier()
                val_loss = evaluate()
                if USE_DDP:
                    dist.barrier()
                if master_process:
                    print0(f"step {global_step}: val_loss {val_loss:.4f}")
                    if wandb is not None:
                        wandb.log({"val/loss": val_loss, "step": global_step}, step=global_step)

print0("Training complete")
