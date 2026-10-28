"""The frozen-encoder path against the original CREEP code with the three backbones frozen.

equivalence_frozen.py holds both sides. Here they run on the synthetic fixture with tiny random stand-ins for ProtT5
and SciBERT (rxnfp is the real model in this repository), and the comparison asserts.
"""

from __future__ import annotations

import os
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

import cache_frozen
import common
import equivalence_frozen as eq
import train_frozen
from common import build_tables, discover_splits, train_args
from metrics import ec_level_accuracy
from model_utils import seed_everything
from testing import FIXTURES, make_tiny_dataset, tiny_backbones

SOURCE = FIXTURES / "dataset/with_protein_information/random_splits_grouped_rxn_key"
EPOCHS, LIMIT_ECS = 2, 24
LOSS_BOUND = 1e-6  # relative; about 8 fp32 ulp. Measured on this fixture: 1.2e-7.
K = 10  # the original -> production gap must stay within 10x the production-path noise (perturbed heads) gap
NOISE_U = 2**-11  # one rounding of the weights at the production precision (TF32 / fp16 unit roundoff)
NEGATIVE_CONTROL = os.environ.get("EQUIV_NEGATIVE_CONTROL") == "1"
MAX_LEN = 512  # protein_max_sequence_len and reaction_max_sequence_len of the original: longer inputs are truncated


def add_long_rows(data) -> None:
    """Make the first rows of every split longer than MAX_LEN, so the truncation and the padding trim are exercised."""
    for path in data.rglob("*.parquet"):
        frame = pd.read_parquet(path)
        frame.iloc[:3, frame.columns.get_loc("sequence")] = ["ACDEFGHIKLMNPQRSTVWY" * 30] * 3
        frame.iloc[:3, frame.columns.get_loc("rxn_smiles")] = ["C" * 300 + "O>>" + "C" * 300 + "N"] * 3
        frame.to_parquet(path, index=False)


def run_both(tmp_path_factory, mode: str, device: torch.device, heads: str, noise: bool = False, scale: float = 1.0):
    """Returns the reference, the candidate (initial heads times `scale`) and, with `noise`, the candidate rerun from perturbed initial heads."""
    work = tmp_path_factory.mktemp(mode)
    cli = Namespace(mode=mode, ec2go=FIXTURES / "ec2go", epochs=EPOCHS, limit_ecs=LIMIT_ECS)
    with pytest.MonkeyPatch.context() as patch, tiny_backbones(work):
        # One cluster per sequence stands in for mmseqs, which is an external tool.
        patch.setattr(common, "mmseqs_clusters", lambda seqs, *a: {s: i for i, s in enumerate(seqs)})
        patch.setattr(cache_frozen, "revision", lambda *a: "rev")
        patch.setattr(eq, "DEVICE", device)
        data = make_tiny_dataset(SOURCE, work / "data", sizes=(96, 24, 48))
        add_long_rows(data)
        out_root = work / "out"
        split = next(s for s in discover_splits(data) if s.key == eq.SPLIT)
        tables = build_tables(split, out_root, cli.ec2go)
        eq.build_cache(cli, split, data, out_root, work, tables, device)
        seed_everything(eq.SEED)
        init = {k: v.state_dict() for k, v in train_frozen.make_heads(train_args()).items()}
        reference = eq.run_reference(cli, tables, work, init)

        def candidate_run(name, perturb, factor=1.0):
            root = work / name
            root.mkdir()
            (root / "embeddings").symlink_to((out_root / "embeddings").resolve())
            if not perturb and factor == 1.0:
                return eq.run_candidate(cli, split, data, root, work, heads)
            real_make_heads = train_frozen.make_heads
            generator = torch.Generator().manual_seed(123)

            def perturbed_heads(args):
                module = real_make_heads(args)
                with torch.no_grad():
                    for p in module.parameters():
                        if perturb:
                            p += NOISE_U * p.abs() * torch.randn(p.shape, generator=generator)
                        p *= factor
                return module

            patch.setattr(train_frozen, "make_heads", perturbed_heads)
            try:
                return eq.run_candidate(cli, split, data, root, work, heads)
            finally:
                patch.setattr(train_frozen, "make_heads", real_make_heads)

        candidate = candidate_run("out_candidate", False, scale)
        noisy = candidate_run("out_candidate_noise", True) if noise else None
    return reference, candidate, noisy


@pytest.fixture(scope="module")
def cpu_fp32(tmp_path_factory):
    return run_both(tmp_path_factory, "fp32", torch.device("cpu"), "fp32")[:2]


def test_sampler_draws_equal_the_original(cpu_fp32):
    reference, candidate = cpu_fp32
    assert candidate["samples"] == reference["samples"]
    assert len(reference["samples"]) == EPOCHS * LIMIT_ECS * train_args().batch_size


def test_fp32_step_losses_match(cpu_fp32):
    reference, candidate = cpu_fp32
    assert len(candidate["losses"]) == len(reference["losses"]) == EPOCHS * LIMIT_ECS
    rel = np.abs(candidate["losses"] - reference["losses"]) / np.abs(reference["losses"])
    assert rel.max() <= LOSS_BOUND


def test_fp32_final_heads_match(cpu_fp32):
    reference, candidate = cpu_fp32
    for modality in eq.HEADS:
        for kind in ("weight", "bias"):
            a, b = candidate["final"][modality][kind], reference["final"][modality][kind]
            assert eq.diff(a, b)[2] <= 1e-5, (modality, kind)


def test_fp32_rankings_and_ec_level_accuracy_match(cpu_fp32):
    reference, candidate = cpu_fp32
    ref, cand = reference["predictions"], candidate["predictions"]
    assert [list(r[:10]) for r in ref["ranked_ecs"]] == [list(r[:10]) for r in cand["ranked_ecs"]]
    assert ec_level_accuracy(cand) == ec_level_accuracy(ref)


def mean_head_difference(a: dict, b: dict) -> float:
    return float(np.mean([(a[m][k] - b[m][k]).abs().mean().item() for m in eq.HEADS for k in ("weight", "bias")]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_production_fp16_within_noise_control(tmp_path_factory):
    """Production precision: original fp16 autocast loop against the cache (fp16 autocast, SDPA, trimmed) and fp16 heads.

    A noise control reruns the production path with the initial heads perturbed by 2**-11 (relative, one fp16 rounding).
    Floor = |perturbed production - production|, gap = |production - original|. The per-step loss gap and the gap of the
    final heads (mean absolute difference) must stay within K times the floor. Step 0 is judged like every other step.
    The EC level accuracy gap at k=1 and k=10, per level, must stay within max(K times its floor, one test reaction).
    The fixture has rows longer than the 512 token limit. With EQUIV_NEGATIVE_CONTROL=1 the production initial heads
    are multiplied by 1.05 and the test must fail.
    """
    reference, candidate, noise = run_both(
        tmp_path_factory, "production", torch.device("cuda:0"), "fp16", noise=True,
        scale=1.05 if NEGATIVE_CONTROL else 1.0,
    )
    ref = reference["losses"]
    gap, floor = np.abs(candidate["losses"] - ref), np.abs(noise["losses"] - candidate["losses"])
    tiny = 1e-7 * np.abs(ref).max()
    final_gap = mean_head_difference(candidate["final"], reference["final"])
    final_floor = mean_head_difference(noise["final"], candidate["final"])
    final_tiny = 1e-7 * float(np.mean([v.abs().mean().item() for m in reference["final"].values() for v in m.values()]))
    tiny0 = 1e-7 * abs(ref[0])
    gap0, floor0 = gap[0], floor[0]
    print(
        f"creep fp16: max gap={gap.max():.3e}, max floor={floor.max():.3e}, "
        f"ratio={gap.max() / max(floor.max(), tiny):.2f}, K={K}, step0 gap={gap0:.3e}, floor={floor0:.3e}, ratio={gap0 / max(floor0, tiny0):.2f}, "
        f"final gap={final_gap:.3e}, final floor={final_floor:.3e}, final ratio={final_gap / max(final_floor, final_tiny):.2f}"
    )
    assert gap0 <= K * max(floor0, tiny0)
    assert gap.max() <= K * max(floor.max(), tiny)
    assert final_gap <= K * max(final_floor, final_tiny)
    ref_acc, cand_acc, noise_acc = (ec_level_accuracy(x["predictions"]) for x in (reference, candidate, noise))
    one_row = 100 / len(reference["predictions"])
    for k in ("k=1", "k=10"):
        for level, value in ref_acc[k].items():
            acc_gap, acc_floor = abs(cand_acc[k][level] - value), abs(noise_acc[k][level] - cand_acc[k][level])
            print(f"creep fp16 {k} {level}: accuracy gap={acc_gap:.2f}, floor={acc_floor:.2f}")
            assert acc_gap <= max(K * acc_floor, one_row), (k, level)
