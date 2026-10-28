"""Tests of the exact and bf16 speed-ups behind train.py --fast."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, T5Config, T5EncoderModel
from transformers.models.t5.modeling_t5 import T5Attention, T5DenseActDense

import common
import train
from model_utils import (
    BackwardAdam,
    dense_act_dense_forward,
    sdpa_t5_attention,
    trim_padding,
)
from testing import FIXTURES, make_tiny_dataset, tiny_backbones

SOURCE = FIXTURES / "dataset/with_protein_information/random_splits_grouped_rxn_key"
EC2GO = FIXTURES / "ec2go"


@pytest.fixture(autouse=True)
def restore_t5(monkeypatch):
    """train.main patches the T5 classes for the whole process; undo it after each test."""
    monkeypatch.setattr(T5Attention, "forward", T5Attention.forward)
    monkeypatch.setattr(T5DenseActDense, "forward", T5DenseActDense.forward)


def t5() -> T5EncoderModel:
    torch.manual_seed(0)
    config = T5Config(vocab_size=128, d_model=64, d_ff=128, d_kv=16, num_layers=4, num_heads=4)
    return T5EncoderModel(config).eval()


def t5_run(model: T5EncoderModel):
    ids = torch.randint(1, 100, (4, 32), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[:, 20:] = 0
    hidden = model(ids, mask).last_hidden_state
    hidden.pow(2).mean().backward()
    return hidden.detach(), [p.grad.clone() for p in model.parameters()]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def test_sdpa_attention_matches_eager(monkeypatch):
    out, grads = t5_run(t5())
    monkeypatch.setattr(T5Attention, "forward", sdpa_t5_attention)
    fast_out, fast_grads = t5_run(t5())
    assert rel(fast_out, out) < 1e-5
    assert all(rel(a, b) < 1e-4 for a, b in zip(fast_grads, grads))


def test_feed_forward_without_round_trip_is_identical(monkeypatch):
    out, grads = t5_run(t5())
    monkeypatch.setattr(T5DenseActDense, "forward", dense_act_dense_forward)
    fast_out, fast_grads = t5_run(t5())
    assert torch.equal(fast_out, out)
    assert all(torch.equal(a, b) for a, b in zip(fast_grads, grads))


def test_feed_forward_skips_cast_under_bf16_autocast(monkeypatch):
    model = t5()
    ids = torch.randint(1, 100, (2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(ids).last_hidden_state
        monkeypatch.setattr(T5DenseActDense, "forward", dense_act_dense_forward)
        assert torch.equal(model(ids).last_hidden_state, out)


def test_trim_padding_keeps_embeddings():
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    model = BertModel(config).eval()
    lengths = [5, 9, 3, 7]
    ids = torch.zeros(4, 24, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(1, 100, (n,))
        mask[i, :n] = 1
    batch = {
        f"{name}_sequence_{kind}": x[None]
        for name in ("text", "reaction")
        for kind, x in (("input_ids", ids), ("attention_mask", mask))
    }
    batch["protein_sequence_input_ids"] = ids[None]
    trimmed = trim_padding(dict(batch))
    assert trimmed["text_sequence_input_ids"].shape == (1, 4, 9)
    assert trimmed["protein_sequence_input_ids"].shape == (1, 4, 24)
    with torch.no_grad():
        full = model(ids, mask)
        short = model(
            trimmed["text_sequence_input_ids"][0], trimmed["text_sequence_attention_mask"][0]
        )
    assert rel(short.pooler_output, full.pooler_output) < 1e-5
    assert rel(short.last_hidden_state[:, 0], full.last_hidden_state[:, 0]) < 1e-5


def make_net() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(20, 30), nn.ReLU(), nn.Linear(30, 1))


def groups(net: nn.Module) -> list[dict]:
    return [
        {"params": net[0].parameters(), "lr": 1e-3},
        {"params": net[2].parameters(), "lr": 1e-2},
    ]


def steps(net: nn.Module, optimizer, n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        x = torch.randn(8, 20, generator=torch.Generator().manual_seed(i))
        optimizer.zero_grad()
        net(x).pow(2).mean().backward()
        optimizer.step()


def test_backward_adam_equals_adam():
    plain, fused = make_net(), make_net()
    steps(plain, torch.optim.Adam(groups(plain), weight_decay=0.01), 5)
    steps(fused, BackwardAdam(groups(fused), weight_decay=0.01), 5)
    assert all(torch.equal(a, b) for a, b in zip(plain.parameters(), fused.parameters()))
    assert all(p.grad is None for p in fused.parameters())


def test_backward_adam_resume_reproduces_next_steps():
    straight = make_net()
    steps(straight, BackwardAdam(groups(straight), weight_decay=0.01), 5)

    first = make_net()
    optimizer = BackwardAdam(groups(first), weight_decay=0.01)
    steps(first, optimizer, 3)
    saved = copy.deepcopy((first.state_dict(), optimizer.state_dict()))

    resumed = make_net()
    resumed.load_state_dict(saved[0])
    resumed_optimizer = BackwardAdam(groups(resumed), weight_decay=0.01)
    resumed_optimizer.load_state_dict(saved[1])
    steps(resumed, resumed_optimizer, 2, start=3)
    assert all(torch.equal(a, b) for a, b in zip(straight.parameters(), resumed.parameters()))


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    work = tmp_path_factory.mktemp("fast")
    with pytest.MonkeyPatch.context() as patch:
        # One cluster per sequence stands in for mmseqs, which is an external tool.
        patch.setattr(common, "mmseqs_clusters", lambda seqs, *a: {s: i for i, s in enumerate(seqs)})
        yield work, make_tiny_dataset(SOURCE, work / "data", sizes=(96, 24, 48))


def run_fast(tiny_data, out: str, epochs: int, device: torch.device) -> Path:
    work, data = tiny_data
    args = [
        "--split", "random_splits_grouped_rxn_key/all",
        "--seed", "666",
        "--dataset_root", str(data),
        "--out_root", str(work / out),
        "--ec2go", str(EC2GO),
        "--fast",
        "--epochs", str(epochs),
        "--limit_ecs", "3",
    ]  # fmt: skip
    with tiny_backbones(work):
        return train.main(args, device=device)


def test_fast_train_and_resume_on_cpu(tiny_data, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # GradScaler stays off on CPU
    cpu = torch.device("cpu")
    straight = run_fast(tiny_data, "straight", 2, cpu)
    run_fast(tiny_data, "resumed", 1, cpu)
    resumed = run_fast(tiny_data, "resumed", 2, cpu)
    assert len(pd.read_csv(resumed / "train_log.csv")) == 2
    a = torch.load(straight / "lastmodel.pth")
    b = torch.load(resumed / "lastmodel.pth")
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fast_train_and_resume_on_gpu(tiny_data):
    cuda = torch.device("cuda:0")
    run_fast(tiny_data, "gpu", 1, cuda)
    run = run_fast(tiny_data, "gpu", 2, cuda)
    assert len(pd.read_csv(run / "train_log.csv")) == 2
