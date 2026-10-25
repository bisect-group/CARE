"""End-to-end smoke test on CPU: tiny random backbones, 64 train / 16 val / 16 test rows, one epoch and a resumed second epoch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import evaluate
import train
from common import DEFAULT_DATASET_ROOT
from testing import make_tiny_dataset, tiny_backbones

SOURCE = "with_protein_information/random_splits_grouped_rxn_key"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work_dir", type=Path, required=True, help="scratch folder for the tiny data and outputs"
    )
    p.add_argument("--ec2go", type=Path, required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--seed", type=int, default=666)
    cli = p.parse_args()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)

    data = make_tiny_dataset(cli.dataset_root / SOURCE, cli.work_dir / "data")
    out = cli.work_dir / "out"
    common = [
        "--split",
        "random_splits_grouped_rxn_key/all",
        "--seed",
        str(cli.seed),
        "--dataset_root",
        str(data),
        "--out_root",
        str(out),
    ]
    device = torch.device("cpu")
    with tiny_backbones(cli.work_dir):
        run = train.main(
            [*common, "--ec2go", str(cli.ec2go), "--epochs", "1", "--limit_ecs", "6"],
            device=device,
        )
        evaluate.main(common, device=device)
        train.main(
            [*common, "--ec2go", str(cli.ec2go), "--epochs", "2", "--limit_ecs", "6"],
            device=device,
        )
        assert len(pd.read_csv(run / "train_log.csv")) == 2, (
            "resume from checkpoint_last.pt failed"
        )
        evaluate.main([*common, "--reference", "train"], device=device)
    for name in (
        "bestmodel.pth",
        "lastmodel.pth",
        "checkpoint_last.pt",
        "train_log.csv",
        "predictions.parquet",
        "metrics.json",
        "run_summary.json",
        "train_summary.json",
    ):
        assert (run / name).exists(), name
    metrics = json.loads((run / "metrics.json").read_text())
    print("rows", metrics["queries"], "k=1", metrics["accuracy"]["k=1"])
    print("SMOKE OK", run)


if __name__ == "__main__":
    main()
