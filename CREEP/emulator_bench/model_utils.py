"""Loads the original CREEP scripts and builds the model exactly as step_01_train_CREEP.py does."""

from __future__ import annotations

import importlib.util
import os
import queue
import random
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import transformers.models.t5.modeling_t5 as t5_modeling

from common import CREEP_ROOT, ORIGINAL_DIR

if str(CREEP_ROOT) not in sys.path:
    sys.path.insert(0, str(CREEP_ROOT))

from CREEP.models import AEFacilitatorModel, CREEPModel  # noqa: E402
from CREEP.utils.tokenization import SmilesTokenizer  # noqa: E402
from transformers import AutoModel, AutoTokenizer, BertModel, T5EncoderModel, T5Tokenizer  # noqa: E402

PROTEIN_HUB_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
TEXT_HUB_ID = "allenai/scibert_scivocab_uncased"
RXNFP_DIR = CREEP_ROOT / "data" / "pretrained_rxnfp"


def load_original(name: str):
    """Import an original script by path without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, ORIGINAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_reaction_tokenizer() -> SmilesTokenizer:
    tokenizer = SmilesTokenizer.from_pretrained(str(RXNFP_DIR / "vocab.txt"))
    if tokenizer.unk_token_id in tokenizer("CC(=O)O>>CC")["input_ids"]:
        raise RuntimeError(
            f"rxnfp tokenizer gives [UNK] for plain SMILES with transformers {transformers.__version__}; use 4.39.1"
        )
    return tokenizer


def build_model(args, weights_dir: Path):
    """Pretrained backbones, projection heads and facilitators, in the original creation order."""
    protein_cache = str(Path(weights_dir) / "pretrained_ProtT5")
    text_cache = str(Path(weights_dir) / "pretrained_SciBert")
    protein_tokenizer = T5Tokenizer.from_pretrained(
        PROTEIN_HUB_ID, do_lower_case=False, cache_dir=protein_cache
    )
    protein_model = T5EncoderModel.from_pretrained(
        PROTEIN_HUB_ID, cache_dir=protein_cache, torch_dtype=torch.float32
    )
    protein_dim = 1024

    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_HUB_ID, cache_dir=text_cache)
    text_model = AutoModel.from_pretrained(TEXT_HUB_ID, cache_dir=text_cache)
    text_dim = 768

    reaction_tokenizer = load_reaction_tokenizer()
    reaction_model = BertModel.from_pretrained(str(RXNFP_DIR))
    reaction_dim = 256

    protein2latent_model = nn.Linear(protein_dim, args.SSL_emb_dim)
    text2latent_model = nn.Linear(text_dim, args.SSL_emb_dim)
    reaction2latent_model = nn.Linear(reaction_dim, args.SSL_emb_dim)

    reaction2protein_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)
    protein2reaction_facilitator_model = AEFacilitatorModel(args.SSL_emb_dim)

    model = CREEPModel(
        protein_model,
        text_model,
        reaction_model,
        protein2latent_model,
        text2latent_model,
        reaction2latent_model,
        reaction2protein_facilitator_model,
        protein2reaction_facilitator_model,
        args.protein_backbone_model,
        args.text_backbone_model,
        args.reaction_backbone_model,
    )
    return model, (protein_tokenizer, text_tokenizer, reaction_tokenizer)


def make_optimizer(
    model: CREEPModel, args, resident_gb: float | None = None, in_backward: bool = False
) -> optim.Adam | BackwardAdam:
    groups = [
        {
            "params": model.protein_model.parameters(),
            "lr": args.protein_lr * args.protein_lr_scale,
        },
        {"params": model.text_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {
            "params": model.reaction_model.parameters(),
            "lr": args.reaction_lr * args.reaction_lr_scale,
        },
        {
            "params": model.protein2latent_model.parameters(),
            "lr": args.protein_lr * args.protein_lr_scale,
        },
        {"params": model.text2latent_model.parameters(), "lr": args.text_lr * args.text_lr_scale},
        {
            "params": model.reaction2latent_model.parameters(),
            "lr": args.reaction_lr * args.reaction_lr_scale,
        },
        {
            "params": model.reaction2protein_facilitator_model.parameters(),
            "lr": args.reaction_lr * args.reaction_lr_scale,
        },
        {
            "params": model.protein2reaction_facilitator_model.parameters(),
            "lr": args.protein_lr * args.protein_lr_scale,
        },
    ]
    if in_backward:
        return BackwardAdam(groups, weight_decay=args.decay)
    if resident_gb is None:
        return optim.Adam(groups, weight_decay=args.decay)
    return OffloadedAdam(groups, resident_gb=resident_gb, weight_decay=args.decay)


def checkpoint_blocks(model: CREEPModel, protein: bool = True) -> None:
    """Recompute each transformer block in backward instead of storing its activations (same values, less memory).

    With protein=False only SciBERT and rxnfp are checkpointed. They hold most of the activations that ProtT5 does not.
    preserve_rng_state=False is safe because the models run in eval mode without dropout.
    """
    from torch.utils.checkpoint import checkpoint

    def wrap(block):
        forward = block.forward
        block.forward = lambda *a, **k: checkpoint(
            forward, *a, use_reentrant=False, preserve_rng_state=False, **k
        )

    blocks = [*model.protein_model.encoder.block] if protein else []
    for block in [*blocks, *model.text_model.encoder.layer, *model.reaction_model.encoder.layer]:
        wrap(block)


class OffloadedAdam(optim.Adam):
    """Adam with most moments in pinned CPU memory. Each chunk of parameters is moved to the GPU, updated and moved back.

    The last chunks, up to resident_gb of moments, stay on the GPU. The update is the stock Adam update on the GPU,
    so the numbers equal those of optim.Adam.
    """

    def __init__(self, params, resident_gb: float = 0.0, chunk_numel: int = 1 << 26, **kwargs):
        super().__init__(params, **kwargs)
        chunks = []
        for group in self.param_groups:
            chunk, size = [], 0
            for p in group["params"]:
                chunk.append(p)
                size += p.numel()
                if size >= chunk_numel:
                    chunks.append((group, chunk))
                    chunk, size = [], 0
            if chunk:
                chunks.append((group, chunk))
        budget = resident_gb * 2**30
        self.chunks, self.resident = [], []
        for group, chunk in reversed(chunks):
            numel = sum(p.numel() for p in chunk)
            keep = numel * 8 <= budget
            budget -= numel * 8 * keep
            (self.resident if keep else self.chunks).insert(0, (group, chunk))
        for chunks, offload in ((self.chunks, True), (self.resident, False)):
            for _, chunk in chunks:
                for p in chunk:
                    self._init_state(p, offload)
        self.copy_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _init_state(self, p, offload: bool, exp_avg=None, exp_avg_sq=None, step=None) -> None:
        def make(x):
            x = torch.zeros(p.shape, dtype=p.dtype) if x is None else x.to("cpu", copy=True)
            return x.pin_memory() if offload else x.to(p.device)

        self.state[p] = {
            "step": torch.zeros(()) if step is None else step.cpu(),
            "exp_avg": make(exp_avg),
            "exp_avg_sq": make(exp_avg_sq),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Keep the offloaded moments on the CPU instead of moving them to the parameter device."""
        saved_groups = state_dict["param_groups"]
        for group, saved in zip(self.param_groups, saved_groups):
            group.update({k: v for k, v in saved.items() if k != "params"})
        index = {
            id(p): i
            for group, saved in zip(self.param_groups, saved_groups)
            for p, i in zip(group["params"], saved["params"])
        }
        for chunks, offload in ((self.chunks, True), (self.resident, False)):
            for _, chunk in chunks:
                for p in chunk:
                    self._init_state(p, offload, **state_dict["state"][index[id(p)]])

    def _update(self, group, chunk) -> None:
        groups = self.param_groups
        self.param_groups = [{**group, "params": chunk}]
        super().step()
        self.param_groups = groups

    @torch.no_grad()
    def step(self, closure=None):
        keys = ("exp_avg", "exp_avg_sq")

        def up(chunk):
            return [
                [self.state[p][k].to(p.device, non_blocking=True) for k in keys] for p in chunk
            ]

        main, copy = torch.cuda.current_stream(), self.copy_stream
        if self.chunks:
            with torch.cuda.stream(copy):
                staged = up(self.chunks[0][1])
        for i, (group, chunk) in enumerate(self.chunks):
            main.wait_stream(copy)
            moments = staged
            if i + 1 < len(self.chunks):
                with torch.cuda.stream(copy):
                    staged = up(self.chunks[i + 1][1])
            cpu = [self.state[p] for p in chunk]
            for p, (m, v), state in zip(chunk, moments, cpu):
                self.state[p] = {"step": state["step"], "exp_avg": m, "exp_avg_sq": v}
            self._update(group, chunk)
            copy.wait_stream(main)
            with torch.cuda.stream(copy):
                for p, (m, v), state in zip(chunk, moments, cpu):
                    state["exp_avg"].copy_(m, non_blocking=True)
                    state["exp_avg_sq"].copy_(v, non_blocking=True)
                    self.state[p] = state
            del moments
        for group, chunk in self.resident:
            self._update(group, chunk)
        main.wait_stream(copy)


class Prefetch:
    """Build the next batches in a thread. The batch order and the Python random draws stay the same."""

    def __init__(self, loader, depth: int = 2, transform=None):
        self.loader, self.depth, self.transform = loader, depth, transform

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        buffer: queue.Queue = queue.Queue(self.depth)
        batches = iter(self.loader)

        def fill():
            try:
                for batch in batches:
                    buffer.put(self.transform(batch) if self.transform else batch)
            except Exception as exc:  # re-raised in the consumer
                buffer.put(exc)
            else:
                buffer.put(None)

        threading.Thread(target=fill, daemon=True).start()
        while (batch := buffer.get()) is not None:
            if isinstance(batch, Exception):
                raise batch
            yield batch


class _NoScaler:
    """GradScaler stand-in for bf16, which needs no loss scaling."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def apply_amp(device: torch.device) -> str:
    """Switch the original fp16 autocast to bf16 on Ampere or newer. Returns the precision in use."""
    if device.type != "cuda":
        return "fp32"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return "fp16"
    torch.cuda.amp.autocast = lambda *a, **k: torch.autocast(
        "cuda", **{"dtype": torch.bfloat16, **k}
    )
    torch.cuda.amp.GradScaler = _NoScaler
    return "bf16"


def disable_autocast_cache() -> None:
    """Cast the weights again at each use instead of caching the casts. Same values, about 2.4 GB less GPU memory."""
    autocast = torch.cuda.amp.autocast
    torch.cuda.amp.autocast = lambda *a, **k: autocast(*a, **{**k, "cache_enabled": False})


def trim_padding(batch: dict) -> dict:
    """Cut the SciBERT and rxnfp inputs to the longest real sequence of the batch.

    The attention masks keep the padded positions out, so the CLS outputs stay the same (up to float rounding).
    ProtT5 keeps its full length because its output is a mean over all positions, padding included.
    """
    for name in ("text", "reaction"):
        mask = batch[f"{name}_sequence_attention_mask"]
        length = int(mask.sum(-1).max())
        for key in (f"{name}_sequence_input_ids", f"{name}_sequence_attention_mask"):
            batch[key] = batch[key][..., :length]
    return batch


def sdpa_t5_attention(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """T5 encoder self-attention with the memory-efficient SDPA kernel.

    The first layer builds the relative position bias plus mask once, as a contiguous tensor in the compute dtype
    (SDPA falls back to the math kernel otherwise). T5 does not scale the scores, so scale=1.0.
    Dropout is not applied: the models run in eval mode.
    """
    batch, length = hidden_states.shape[:2]

    def heads(x):
        return x.view(batch, length, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    q, k, v = (
        heads(self.q(hidden_states)),
        heads(self.k(hidden_states)),
        heads(self.v(hidden_states)),
    )
    if position_bias is None:
        bias = self.compute_bias(length, length, device=q.device)
        position_bias = (bias.to(q.dtype) + mask.to(q.dtype)).contiguous()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=position_bias, scale=1.0)
    out = self.o(out.transpose(1, 2).reshape(batch, length, self.inner_dim))
    return out, None, position_bias


def dense_act_dense_forward(self, hidden_states):
    """T5 feed-forward without the cast of the hidden states to the fp32 weight dtype before wo (autocast casts again)."""
    return self.wo(self.act(self.wi(hidden_states)))


def patch_t5() -> None:
    """Use the SDPA attention and the shorter feed-forward for every T5 model in this process."""
    t5_modeling.T5Attention.forward = sdpa_t5_attention
    t5_modeling.T5DenseActDense.forward = dense_act_dense_forward


class BackwardAdam:
    """One stock Adam per parameter, stepped in backward as soon as the gradient of that parameter is final.

    The gradient is freed at once, so the full gradient buffer never exists. The update equals that of optim.Adam.
    zero_grad and step do nothing because backward already did both.
    """

    def __init__(self, groups, **kwargs):
        self.optimizers = {}
        for group in groups:
            for p in group["params"]:
                self.optimizers[p] = optim.Adam([p], lr=group["lr"], **kwargs)
                p.register_post_accumulate_grad_hook(self._step)

    def _step(self, p) -> None:
        self.optimizers[p].step()
        p.grad = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure=None) -> None:
        pass

    def state_dict(self) -> dict:
        return {"parameters": [o.state_dict() for o in self.optimizers.values()]}

    def load_state_dict(self, state_dict: dict) -> None:
        for optimizer, saved in zip(self.optimizers.values(), state_dict["parameters"]):
            optimizer.load_state_dict(saved)
