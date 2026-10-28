"""Tests of the head training on cached vectors: sampler draws, loss, optimizer and resume."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler  # other tests replace torch.cuda.amp.GradScaler
from torch.utils.data import DataLoader

import train_frozen
from common import train_args
from model_utils import load_original, seed_everything
from train_frozen import (
    RowSampler,
    cl_loss,
    contrastive_loss,
    epoch_rows,
    make_head_optimizer,
    make_heads,
    train_step,
)
from CREEP.datasets import CREEPDatasetMineBatch
from CREEP.datasets import dataset_CREEP

ARGS = train_args()


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    path = tmp_path_factory.mktemp("tables")
    rng = random.Random(0)
    ecs = [f"1.1.{i // 5 + 1}.{i % 5 + 1}" for i in range(40)]
    letters = "ACDEFGHIKLMNPQRSTVWY"
    proteins = [
        (
            "".join(rng.choices(letters, k=rng.randint(20, 40))),
            ec,
            rng.randint(0, 9) + 10 * (j % 3),
        )
        for j, ec in enumerate(ecs * 3)
    ]
    pd.DataFrame(proteins, columns=["Sequence", "EC number", "clusterRes50"]).to_csv(
        path / "protein2EC.csv", index=False
    )
    reactions = [(f"C{'C' * rng.randint(0, 9)}>>O{i}", ec) for i, ec in enumerate(ecs * 2)]
    pd.DataFrame(reactions, columns=["Reaction", "EC number"]).to_csv(
        path / "reaction_train.csv", index=False
    )
    pd.DataFrame({"EC number": ecs, "Text": [f"text of {ec}" for ec in ecs]}).to_csv(
        path / "text2EC.csv", index=False
    )
    return path


def original_draws(tables, seed: int, epochs: int) -> list[tuple[str, str, str]]:
    """(protein, text, reaction) of every sample the original dataset encodes."""
    seen = []

    def record(sequence, tokenizer, max_len):
        seen.append(sequence)
        return torch.zeros(max_len, dtype=torch.long), torch.zeros(max_len, dtype=torch.long)

    original = dataset_CREEP.encode_sequence
    dataset_CREEP.encode_sequence = record
    try:
        dataset = CREEPDatasetMineBatch(
            f"{tables}/", str(tables / "reaction_train.csv"), None, None, None, 2, 2, 2, 15
        )
        seed_everything(seed)
        for _ in range(epochs):
            for _ in DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0):
                pass
    finally:
        dataset_CREEP.encode_sequence = original
    triples = np.array(seen, dtype=object).reshape(
        -1, 3
    )  # protein, text, reaction per encode order
    return [(p.replace(" ", ""), t, r) for p, t, r in triples]


def test_sampler_draws_equal_the_original_dataset(tables):
    sampler = RowSampler(tables, 15)
    seed_everything(7)
    rows = torch.cat([epoch_rows(sampler) for _ in range(2)]).reshape(-1, 3)
    got = [
        (sampler.proteins[p], sampler.texts[t], sampler.reactions[r]) for p, t, r in rows.tolist()
    ]
    assert got == original_draws(tables, 7, 2)
    assert len(got) == 2 * len(sampler) * 16


def random_vectors(seed: int, n: int = 16):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, 256, generator=g) for _ in range(3)]


def test_loss_equals_the_original_do_CL():
    original = load_original("step_01_train_CREEP")
    protein, text, reaction = random_vectors(0)
    pairs = (
        (protein, text),
        (text, protein),
        (text, reaction),
        (protein, reaction),
        (reaction, text),
        (reaction, protein),
    )
    want = [original.do_CL(x, y, ARGS) for x, y in pairs]
    for (x, y), (loss, acc) in zip(pairs, want):
        got_loss, got_acc = cl_loss(x, y, ARGS)
        assert torch.allclose(got_loss, loss, atol=1e-7)
        assert got_acc.item() == pytest.approx(acc)
    loss, acc = contrastive_loss(protein, text, reaction, ARGS)
    assert loss.item() == pytest.approx(sum(float(x) for x, _ in want) / 6, rel=1e-6)
    assert acc.item() == pytest.approx(sum(a for _, a in want) / 6)


def test_optimizer_groups_have_the_original_learning_rates():
    optimizer = make_head_optimizer(make_heads(ARGS), ARGS)
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([1e-5, 1e-6, 1e-5])
    assert all(g["weight_decay"] == 0 for g in optimizer.param_groups)


def test_train_step_matches_the_original_formula():
    original = load_original("step_01_train_CREEP")
    vectors = {
        "protein": torch.randn(30, 1024),
        "text": torch.randn(30, 768),
        "reaction": torch.randn(30, 256),
    }
    rows = torch.randint(0, 30, (16, 3))
    torch.manual_seed(0)
    heads = make_heads(ARGS)
    torch.manual_seed(0)
    reference = make_heads(ARGS)
    optimizer, ref_optimizer = (
        make_head_optimizer(heads, ARGS),
        make_head_optimizer(reference, ARGS),
    )
    scaler = GradScaler(enabled=False)
    for _ in range(3):
        loss, _ = train_step(heads, optimizer, scaler, vectors, rows, ARGS)
        p = reference["protein"](vectors["protein"][rows[:, 0]])
        t = reference["text"](vectors["text"][rows[:, 1]])
        r = reference["reaction"](vectors["reaction"][rows[:, 2]])
        pairs = ((p, t), (t, p), (t, r), (p, r), (r, t), (r, p))
        want = sum(original.do_CL(x, y, ARGS)[0] for x, y in pairs) / 6
        ref_optimizer.zero_grad()
        want.backward()
        ref_optimizer.step()
        assert loss.item() == pytest.approx(want.item(), abs=1e-6)
    for a, b in zip(heads.parameters(), reference.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


def test_discovered_rows_index_the_cache_order(tables):
    sampler = RowSampler(tables, 15)
    assert sampler.proteins == sorted(set(sampler.proteins))
    assert isinstance(make_heads(ARGS), nn.ModuleDict)
    assert train_frozen.LOG_EVERY > 0


def test_candidate_pool_defaults_to_all_proteins():
    import evaluate
    import queue_pipeline

    assert evaluate.parse_args(["--split", "s", "--seed", "1"]).reference == "all"
    assert queue_pipeline.parse_args([]).reference == "all"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_train_step_matches_the_original_fp16_autocast_step():
    """The original loop body (autocast, six do_CL, GradScaler, Adam) on the same cached vectors."""
    original = load_original("step_01_train_CREEP")
    device = torch.device("cuda:0")
    vectors = {
        "protein": torch.randn(30, 1024, device=device),
        "text": torch.randn(30, 768, device=device),
        "reaction": torch.randn(30, 256, device=device),
    }
    rows = torch.randint(0, 30, (16, 3), device=device)
    torch.manual_seed(0)
    heads = make_heads(ARGS).to(device)
    torch.manual_seed(0)
    reference = make_heads(ARGS).to(device)
    optimizer, ref_optimizer = (
        make_head_optimizer(heads, ARGS),
        make_head_optimizer(reference, ARGS),
    )
    scaler, ref_scaler = GradScaler(), GradScaler()
    for _ in range(5):
        loss, _ = train_step(heads, optimizer, scaler, vectors, rows, ARGS)
        with torch.autocast("cuda", dtype=torch.float16):
            p = reference["protein"](vectors["protein"][rows[:, 0]])
            t = reference["text"](vectors["text"][rows[:, 1]])
            r = reference["reaction"](vectors["reaction"][rows[:, 2]])
            pairs = ((p, t), (t, p), (t, r), (p, r), (r, t), (r, p))
            want = (
                ARGS.alpha_contrastive * sum(original.do_CL(x, y, ARGS)[0] for x, y in pairs) / 6
            )
        ref_optimizer.zero_grad()
        ref_scaler.scale(want).backward()
        ref_scaler.step(ref_optimizer)
        ref_scaler.update()
        assert loss.item() == pytest.approx(want.item(), rel=1e-4)
    for a, b in zip(heads.parameters(), reference.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
