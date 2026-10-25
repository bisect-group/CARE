"""Exactness tests for the memory options: checkpointed blocks and the offloaded Adam give the numbers of the originals."""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, T5Config, T5EncoderModel

from model_utils import OffloadedAdam, Prefetch, checkpoint_blocks


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.protein_model = T5EncoderModel(
            T5Config(vocab_size=128, d_model=64, d_ff=128, d_kv=16, num_layers=4, num_heads=4)
        ).eval()
        bert = BertConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=128,
        )
        self.text_model = BertModel(bert).eval()
        self.reaction_model = BertModel(bert).eval()


def grads(checkpointed: bool) -> list[torch.Tensor]:
    model = Tiny()
    if checkpointed:
        checkpoint_blocks(model)
    ids = torch.randint(1, 90, (4, 32), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[:, 20:] = 0
    loss = (
        model.protein_model(ids, mask).last_hidden_state.mean(1).sum()
        + model.text_model(ids, mask).pooler_output.sum()
        + model.reaction_model(ids, mask).last_hidden_state[:, 0].sum()
    )
    loss.backward()
    return [p.grad for p in model.parameters() if p.grad is not None]


def test_checkpoint_blocks_keep_gradients():
    assert all(torch.equal(a, b) for a, b in zip(grads(False), grads(True)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("resident_gb", [0.0, 0.001, 1.0])
def test_offloaded_adam_equals_adam(resident_gb):
    def build():
        torch.manual_seed(0)
        return nn.Sequential(
            nn.Linear(300, 500), nn.ReLU(), nn.Linear(500, 300), nn.Linear(300, 1)
        ).cuda()

    def groups(model):
        return [
            {"params": model[:2].parameters(), "lr": 1e-3},
            {"params": model[2:].parameters(), "lr": 1e-4},
        ]

    plain, offloaded = build(), build()
    optimizers = [
        torch.optim.Adam(groups(plain)),
        OffloadedAdam(groups(offloaded), resident_gb=resident_gb, chunk_numel=100_000),
    ]
    x = torch.randn(64, 300, device="cuda")
    for model, optimizer in zip((plain, offloaded), optimizers):
        scaler = torch.cuda.amp.GradScaler()
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(x).float().pow(2).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), offloaded.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("resident_gb", [0.0, 0.001])
def test_offloaded_adam_continues_equally_after_state_dict_reload(resident_gb):
    def steps(model, optimizer, n):
        x = torch.randn(64, 300, device="cuda", generator=torch.Generator("cuda").manual_seed(1))
        for _ in range(n):
            optimizer.zero_grad()
            model(x).pow(2).mean().backward()
            optimizer.step()

    def build():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(300, 500), nn.Linear(500, 1)).cuda()

    def adam(model):
        return OffloadedAdam(model.parameters(), resident_gb=resident_gb, chunk_numel=100_000)

    first, resumed = build(), build()
    first_adam = adam(first)
    steps(first, first_adam, 3)
    torch.cuda.synchronize()  # the moments are copied to the CPU asynchronously, as train.py waits before saving
    buffer = io.BytesIO()
    torch.save(first_adam.state_dict(), buffer)
    resumed.load_state_dict(first.state_dict())
    resumed_adam = adam(resumed)
    resumed_adam.load_state_dict(torch.load(io.BytesIO(buffer.getvalue()), weights_only=False))
    steps(first, first_adam, 3)
    steps(resumed, resumed_adam, 3)
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), resumed.parameters()))


def test_prefetch_keeps_order():
    assert list(Prefetch(range(10))) == list(range(10))


def test_prefetch_raises_loader_errors():
    def failing():
        yield 1
        raise ValueError("bad batch")

    batches = Prefetch(failing())
    with pytest.raises(ValueError, match="bad batch"):
        list(batches)
