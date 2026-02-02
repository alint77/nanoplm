import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, NVFP4BlockScaling
except Exception as exc:
    raise SystemExit(
        "transformer_engine is required for this script. Install transformer_engine[pytorch]."
    ) from exc

try:
    import wandb
except Exception:
    wandb = None

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from nanoplm.pretraining.dataset import (
    LoadShardedFastaMLMDataset,
    LazyFastaMLMDataset,
)
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer

# ====================
# Training toggles
# ====================
USE_FP8 = True
USE_NVFP4 = False
USE_FP8_WEIGHTS = False
USE_AMP = True
USE_FP8_EVAL = False
USE_NVFP4_EVAL = False
USE_FSDP2 = True
USE_DDP = False
USE_COMPILE_MODEL = False
USE_AC_BLOCKS = False
USE_AC_ATTENTION = False
USE_AC_LM_HEAD = False
USE_FAST_FSDP = True

USE_WANDB = True
WANDB_PROJECT = "nanoplm-modernbert-te"
WANDB_RUN_NAME = None

# Peak BF16 TFLOPS per H100 SXM (used for MFU estimate)
H100_PEAK_TFLOPS = 989

# ====================
# Data settings
# ====================
USE_LAZY_DATASET = False
TRAIN_HDF5_DIR = "output/data/pretrain_shards/train_hdf5"
VAL_HDF5_DIR = "output/data/pretrain_shards/val_hdf5"
TRAIN_FASTA = "output/data/split/train.fasta"
VAL_FASTA = "output/data/split/val.fasta"
MAX_LENGTH = 512
LOAD_ALL_IN_MEMORY = False

NUM_WORKERS = 8
PREFETCH_FACTOR = 2
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# ====================
# Model settings
# ====================
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 2048
NUM_LAYERS = 16
NUM_HEADS = 16
ATTENTION_DROPOUT = 0.0
MLP_DROPOUT = 0.0
MLP_ACTIVATION = "swiglu"
CLASSIFIER_ACTIVATION = "gelu"
MLP_BIAS = False
ATTENTION_BIAS = False

# ====================
# Optimization settings
# ====================
TOTAL_BATCH_SIZE = 32768  # total tokens per update across all GPUs
BATCH_SIZE = 256
MAX_STEPS = 2000  # optimizer steps
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
MIN_LR = 1e-6
EVAL_EVERY = 200
EVAL_ITERS = 25
LOG_EVERY = 50

SEED = 1337

# ====================
# Distributed init
# ====================
from torch.distributed import init_process_group
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

# Auto-enable DDP if launched with torchrun and multiple ranks
if not USE_FSDP2:
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    if env_world > 1:
        USE_DDP = True

if USE_DDP or USE_FSDP2:
    init_process_group(backend="nccl", device_id=int(os.environ.get("LOCAL_RANK", "0")))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    data_parallel_group = dist.new_group(backend="nccl")
    if USE_FSDP2:
        device_mesh = init_device_mesh("cuda", (ddp_world_size,))
    else:
        device_mesh = None
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    master_process = True
    seed_offset = 0
    data_parallel_group = None
    device_mesh = None

torch.manual_seed(SEED + seed_offset)
random.seed(SEED + seed_offset)

# ====================
# FP8 recipes
# ====================
if USE_NVFP4:
    recipe = NVFP4BlockScaling()
else:
    recipe = DelayedScaling()

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
    return mask.unsqueeze(1).unsqueeze(1)


def get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "swiglu":
        def _swiglu(x):
            x, gate = x.chunk(2, dim=-1)
            return F.silu(gate) * x
        return _swiglu
    raise ValueError(f"Unsupported activation: {name}")


# TE init helpers

def te_init_method(weight):
    fan_out = weight.size(0)
    fan_in = weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)


def te_output_layer_init_method(weight):
    torch.nn.init.zeros_(weight)


@dataclass
class ModernBertTEConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int = 1024
    attention_dropout: float = 0.0
    mlp_dropout: float = 0.0
    mlp_activation: str = "swiglu"
    classifier_activation: str = "gelu"
    attention_bias: bool = False
    mlp_bias: bool = False


class ModernBertBlock(nn.Module):
    def __init__(self, config: ModernBertTEConfig, layer_number: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.qkv = te.LayerNormLinear(
            in_features=config.hidden_size,
            out_features=3 * config.hidden_size,
            bias=config.attention_bias,
            normalization="RMSNorm",
            init_method=te_init_method,
        )
        self.attn = te.DotProductAttention(
            num_attention_heads=self.num_heads,
            kv_channels=self.head_dim,
            attention_dropout=config.attention_dropout,
            attn_mask_type="padding",
            qkv_format="bshd",
            layer_number=layer_number,
        )
        self.attn_proj = te.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
            init_method=te_output_layer_init_method,
        )
        self.attn_dropout = nn.Dropout(config.mlp_dropout)

        self.mlp = te.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            bias=config.mlp_bias,
            normalization="RMSNorm",
            activation=config.mlp_activation,
            init_method=te_init_method,
            output_layer_init_method=te_output_layer_init_method,
        )
        self.mlp_dropout = nn.Dropout(config.mlp_dropout)

    def _unwrap(self, out):
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_first_microbatch: bool = False,
        checkpoint_core_attention: bool = False,
    ) -> torch.Tensor:
        residual = x
        qkv = self.qkv(x, is_first_microbatch=is_first_microbatch)
        qkv = self._unwrap(qkv)
        bsz, seqlen, _ = qkv.shape
        qkv = qkv.view(bsz, seqlen, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn_out = self.attn(
            q,
            k,
            v,
            attention_mask=attention_mask,
            attn_mask_type="padding" if attention_mask is not None else "no_mask",
            checkpoint_core_attention=checkpoint_core_attention,
        )
        attn_out = self._unwrap(attn_out)
        attn_out = attn_out.contiguous().view(bsz, seqlen, self.hidden_size)
        attn_out = self.attn_proj(attn_out, is_first_microbatch=is_first_microbatch)
        attn_out = self._unwrap(attn_out)
        x = residual + self.attn_dropout(attn_out)

        residual = x
        mlp_out = self.mlp(x, is_first_microbatch=is_first_microbatch)
        mlp_out = self._unwrap(mlp_out)
        x = residual + self.mlp_dropout(mlp_out)
        return x


class ModernBertMLMHead(nn.Module):
    def __init__(self, config: ModernBertTEConfig):
        super().__init__()
        self.dense = te.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            init_method=te_init_method,
        )
        self.act = get_activation(config.classifier_activation)
        self.norm = te.LayerNorm(config.hidden_size)
        self.decoder = te.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            init_method=te_output_layer_init_method,
        )
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def _unwrap(self, out):
        return out[0] if isinstance(out, tuple) else out

    def forward(self, hidden_states: torch.Tensor, is_first_microbatch: bool = False) -> torch.Tensor:
        x = self.dense(hidden_states, is_first_microbatch=is_first_microbatch)
        x = self._unwrap(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.decoder(x, is_first_microbatch=is_first_microbatch)
        x = self._unwrap(x)
        return x + self.bias


class ModernBertForMaskedLM(nn.Module):
    def __init__(self, config: ModernBertTEConfig, pad_token_id: int):
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=pad_token_id)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.emb_norm = te.LayerNorm(config.hidden_size)
        self.emb_drop = nn.Dropout(0.0)
        self.layers = nn.ModuleList([
            ModernBertBlock(config, layer_number=i + 1) for i in range(config.num_hidden_layers)
        ])
        self.final_norm = te.RMSNorm(config.hidden_size)
        self.mlm_head = ModernBertMLMHead(config)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=1.0)
        if self.tok_embeddings.weight.device.type == "cuda":
            self.tok_embeddings.to(dtype=torch.bfloat16)
            self.pos_embeddings.to(dtype=torch.bfloat16)

    def _loss_from_hidden(self, hidden_states: torch.Tensor, labels: torch.Tensor, is_first_microbatch: bool):
        logits = self.mlm_head(hidden_states, is_first_microbatch=is_first_microbatch)
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        is_first_microbatch: bool = False,
    ):
        bsz, seqlen = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seqlen, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        tok_emb = self.tok_embeddings(input_ids)
        pos_emb = self.pos_embeddings(position_ids)
        hidden_states = self.emb_norm(tok_emb + pos_emb)
        hidden_states = self.emb_drop(hidden_states)

        attn_mask = build_attention_mask(attention_mask)

        for block in self.layers:
            if USE_AC_BLOCKS:
                hidden_states = te.checkpoint(
                    block,
                    hidden_states,
                    attn_mask,
                    is_first_microbatch,
                    USE_AC_ATTENTION,
                    use_reentrant=True,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attn_mask,
                    is_first_microbatch=is_first_microbatch,
                    checkpoint_core_attention=USE_AC_ATTENTION,
                )

        hidden_states = self.final_norm(hidden_states)

        if labels is not None and USE_AC_LM_HEAD and self.training:
            loss = te.checkpoint(
                self._loss_from_hidden,
                hidden_states,
                labels,
                is_first_microbatch,
                use_reentrant=True,
            )
            logits = None
        else:
            logits = self.mlm_head(hidden_states, is_first_microbatch=is_first_microbatch)
            loss = None
            if labels is not None:
                vocab_size = logits.size(-1)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

        return logits, loss


# ====================
# Build model + data
# ====================

tokenizer = ProtModernBertTokenizer()

model_config = ModernBertTEConfig(
    hidden_size=HIDDEN_SIZE,
    intermediate_size=INTERMEDIATE_SIZE,
    num_hidden_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=MAX_LENGTH,
    attention_dropout=ATTENTION_DROPOUT,
    mlp_dropout=MLP_DROPOUT,
    mlp_activation=MLP_ACTIVATION,
    classifier_activation=CLASSIFIER_ACTIVATION,
    attention_bias=ATTENTION_BIAS,
    mlp_bias=MLP_BIAS,
)

with te.quantized_model_init(enabled=USE_FP8_WEIGHTS, recipe=recipe):
    model = ModernBertForMaskedLM(model_config, pad_token_id=tokenizer.pad_token_id).to(device)

if USE_FSDP2:
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    for block in model.layers:
        fully_shard(
            block,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=(not USE_FAST_FSDP),
        )
    fully_shard(model.mlm_head, mesh=device_mesh, mp_policy=mp_policy, reshard_after_forward=(not USE_FAST_FSDP))
    fully_shard(model, mesh=device_mesh, mp_policy=mp_policy, reshard_after_forward=(not USE_FAST_FSDP))
    raw_model = model
elif USE_DDP:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp_local_rank],
        process_group=data_parallel_group,
    )
    raw_model = model.module
else:
    raw_model = model

if USE_COMPILE_MODEL:
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# Parameter count and MFU setup
num_params = sum(p.numel() for p in raw_model.parameters())
print0(f"model params: {num_params / 1e6:.2f}M")
flops_per_token = 6 * num_params  # rough transformer estimate
total_peak_flops = H100_PEAK_TFLOPS * 1e12 * ddp_world_size

# data
if USE_LAZY_DATASET:
    train_ds = LazyFastaMLMDataset(TRAIN_FASTA, tokenizer, MAX_LENGTH)
    val_ds = LazyFastaMLMDataset(VAL_FASTA, tokenizer, MAX_LENGTH)
else:
    train_ds = LoadShardedFastaMLMDataset(TRAIN_HDF5_DIR, load_all_in_memory=LOAD_ALL_IN_MEMORY)
    val_ds = LoadShardedFastaMLMDataset(VAL_HDF5_DIR, load_all_in_memory=LOAD_ALL_IN_MEMORY)

train_collator = ProtDataCollatorForLM(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    mask_token_probability=0.8,
    random_token_probability=0.1,
    keep_probability=0.1,
    pad_to_multiple_of=8 if (USE_FP8 or USE_NVFP4) else None,
)
val_collator = ProtDataCollatorForLM(
    tokenizer=tokenizer,
    mlm_probability=0.15,
    mask_token_probability=0.8,
    random_token_probability=0.1,
    keep_probability=0.1,
    pad_to_multiple_of=None,
)

train_sampler = None
val_sampler = None
if USE_DDP or USE_FSDP2:
    train_sampler = DistributedSampler(train_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY and device != "cpu",
    persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    collate_fn=train_collator,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
    pin_memory=PIN_MEMORY and device != "cpu",
    persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    collate_fn=val_collator,
    drop_last=False,
)

# ====================
# Optimizer / scheduler
# ====================

def build_param_groups(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith("bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

try:
    optimizer = torch.optim.AdamW(
        build_param_groups(raw_model),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        fused=(device != "cpu"),
    )
except TypeError:
    optimizer = torch.optim.AdamW(
        build_param_groups(raw_model),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
    )

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    total_iters=WARMUP_STEPS,
)
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, MAX_STEPS - WARMUP_STEPS),
    eta_min=MIN_LR,
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    [warmup_scheduler, main_scheduler],
    milestones=[WARMUP_STEPS],
)

# ====================
# Training loop
# ====================

grad_accum_steps = max(1, math.ceil(TOTAL_BATCH_SIZE / (BATCH_SIZE * ddp_world_size * MAX_LENGTH)))
actual_total_batch_size = BATCH_SIZE * grad_accum_steps * ddp_world_size * MAX_LENGTH
max_micro_steps = MAX_STEPS * grad_accum_steps

print0(f"grad_accum_steps: {grad_accum_steps}")
print0(f"total_batch_size (tokens): {actual_total_batch_size}")

if USE_WANDB and master_process and wandb is not None:
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "total_batch_size": actual_total_batch_size,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_heads": NUM_HEADS,
            "mlp_activation": MLP_ACTIVATION,
            "use_fp8": USE_FP8,
            "use_nvfp4": USE_NVFP4,
            "use_amp": USE_AMP,
            "use_fsdp2": USE_FSDP2,
            "use_ddp": USE_DDP,
            "grad_accum_steps": grad_accum_steps,
        },
    )


def iter_dataloader(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()

def evaluate():
    model.eval()
    losses = []
    it = iter_dataloader(val_loader, val_sampler)
    for _ in range(EVAL_ITERS):
        batch = next(it)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        autocast_device = "cuda" if device != "cpu" else "cpu"
        with torch.amp.autocast(autocast_device, enabled=USE_AMP and device != "cpu", dtype=torch.bfloat16):
            with te.autocast(enabled=(USE_FP8_EVAL or USE_NVFP4_EVAL), recipe=recipe, amax_reduction_group=data_parallel_group):
                _, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    is_first_microbatch=True,
                )
        losses.append(loss.detach().float())
    model.train()
    mean_loss = torch.stack(losses).mean()
    if USE_DDP or USE_FSDP2:
        mean_loss = mean_loss.to(device)
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss = mean_loss / ddp_world_size
    return mean_loss.item()


train_iter = iter_dataloader(train_loader, train_sampler)

model.train()
optimizer.zero_grad(set_to_none=True)

tlast = time.time()
last_log_step = 0
data_time_accum = 0.0

for micro_step in range(max_micro_steps):
    step = micro_step // grad_accum_steps
    should_log = (step > 0 and step % LOG_EVERY == 0 and (micro_step + 1) % grad_accum_steps == 0)
    should_eval = (step > 0 and step % EVAL_EVERY == 0 and (micro_step + 1) % grad_accum_steps == 0)

    t_data_start = time.time()
    batch = next(train_iter)
    t_data_end = time.time()
    data_time_accum += (t_data_end - t_data_start)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    if USE_DDP:
        model.require_backward_grad_sync = ((micro_step + 1) % grad_accum_steps == 0)
    if USE_FSDP2:
        model.set_requires_gradient_sync((micro_step + 1) % grad_accum_steps == 0)

    autocast_device = "cuda" if device != "cpu" else "cpu"
    with torch.amp.autocast(autocast_device, enabled=USE_AMP and device != "cpu", dtype=torch.bfloat16):
        with te.autocast(enabled=(USE_FP8 or USE_NVFP4), recipe=recipe, amax_reduction_group=data_parallel_group):
            _, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                is_first_microbatch=(micro_step % grad_accum_steps == 0),
            )

    loss = loss / grad_accum_steps
    loss.backward()

    if (micro_step + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if should_log and master_process:
        torch.cuda.synchronize() if device != "cpu" else None
        tnow = time.time()
        dt = tnow - tlast
        steps_since = step - last_log_step
        last_log_step = step
        tokens_processed = steps_since * actual_total_batch_size
        tok_per_sec = int(tokens_processed / max(dt, 1e-6))
        global_tok_per_sec = tok_per_sec
        flops_achieved = flops_per_token * global_tok_per_sec
        mfu = (flops_achieved / total_peak_flops * 100) if device != "cpu" else 0.0
        loss_value = (loss.detach().float() * grad_accum_steps).item()
        grad_norm_value = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
        data_pct = (data_time_accum / max(dt, 1e-6)) * 100.0

        print0(
            f"step {step}: loss {loss_value:.4f} | tok/s {tok_per_sec:,} | grad_norm {grad_norm_value:.2f} | mfu_h100 {mfu:.2f}% | data {data_pct:.1f}%"
        )
        if USE_WANDB and wandb is not None:
            wandb.log(
                {
                    "train/loss": loss_value,
                    "train/grad_norm": grad_norm_value,
                    "train/tok_per_sec": tok_per_sec,
                    "train/data_pct": data_pct,
                    "lr": scheduler.get_last_lr()[0],
                    "step": step,
                },
                step=step,
            )
        tlast = tnow
        data_time_accum = 0.0

    if should_eval and master_process:
        pass

    if should_eval:
        if USE_DDP or USE_FSDP2:
            dist.barrier()
        val_loss = evaluate()
        if USE_DDP or USE_FSDP2:
            dist.barrier()
        if master_process:
            print0(f"step {step}: val_loss {val_loss:.4f}")
            if USE_WANDB and wandb is not None:
                wandb.log({"val/loss": val_loss, "step": step}, step=step)

print0("Training complete")
