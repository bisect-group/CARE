"""Run A (original scripts) against Run B (wrapper) on tiny data, on CPU. Compares in the order of the spec."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import evaluate
import train
from common import DEFAULT_DATASET_ROOT, ORIGINAL_DIR, build_tables, discover_splits
from CREEP.datasets import CREEPDatasetMineBatch
from metrics import ec_level_accuracy
from model_utils import RXNFP_DIR
from testing import hash_state, hash_tensors, make_tiny_dataset, tiny_backbones

SOURCE = "with_protein_information/random_splits_grouped_rxn_key"


class Recorder:
    """Logs batch hashes, initial weights and the weights at the start of each epoch."""

    def __init__(self):
        self.batches, self.init_weights, self.epoch_weights, self.optimizer = [], None, [], None

    @contextlib.contextmanager
    def install(self):
        rec = self
        real_getitem = CREEPDatasetMineBatch.__getitem__
        real_adam, real_scaler = torch.optim.Adam, torch.cuda.amp.GradScaler

        def getitem(dataset, index):
            batch = real_getitem(dataset, index)
            rec.batches.append(hash_tensors(*[batch[k] for k in sorted(batch)]))
            return batch

        class Adam(real_adam):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                rec.optimizer = self
                rec.init_weights = rec.weights()

        class Scaler(real_scaler):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                rec.epoch_weights.append(rec.weights())

        with (
            mock.patch.object(CREEPDatasetMineBatch, "__getitem__", getitem),
            mock.patch.object(torch.optim, "Adam", Adam),
            mock.patch.object(torch.cuda.amp, "GradScaler", Scaler),
        ):
            yield

    def weights(self) -> str:
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        return hash_tensors(*params)


def run_original(
    root: Path, tables: Path, seed: int, epochs: int, reference_csv: Path, queries_csv: Path
):
    """step_01 training, step_02 extraction and downstream_retrieval.py, run as scripts with relative paths."""
    (root / "processed_data").mkdir(parents=True)
    (root / "splits" / "task2").mkdir(parents=True)
    (root / "CREEP" / "data").mkdir(parents=True)
    (root / "CREEP" / "data" / "pretrained_rxnfp").symlink_to(RXNFP_DIR)
    for name in ("protein2EC.csv", "text2EC.csv"):
        shutil.copy(tables / name, root / "processed_data" / name)
    shutil.copy(
        tables / "reaction_train.csv", root / "splits" / "task2" / "easy_reaction_train.csv"
    )
    shutil.copy(reference_csv, root / "processed_data" / "protein2EC_clustered50.csv")
    shutil.copy(queries_csv, root / "splits" / "task2" / "easy_reaction_test.csv")
    ecs = sorted(pd.read_csv(reference_csv)["EC number"].unique())
    (root / "processed_data" / "EC_list.txt").write_text("\n".join(ecs) + "\n")

    script_dir = root / "task2_baselines" / "CREEP"
    script_dir.mkdir(parents=True)
    out = script_dir / "output"
    stdout = sys.stdout
    cwd = Path.cwd()
    stack = contextlib.ExitStack()
    try:
        os.chdir(script_dir)
        sys.argv = [
            "step_01",
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--output_model_dir",
            str(out),
        ]
        namespace = runpy.run_path(
            str(ORIGINAL_DIR / "step_01_train_CREEP.py"), run_name="__main__"
        )
        sys.stdout = stdout
        final = hash_state(namespace["model"].state_dict())
        for modality, extra in (
            ("protein", ["--dataset=all_proteins", "--get_cluster_centers"]),
            ("reaction", ["--dataset=easy_reaction_test"]),
        ):
            sys.argv = ["step_02", f"--pretrained_folder={out}", f"--modality={modality}", *extra]
            runpy.run_path(str(ORIGINAL_DIR / "step_02_extract_CREEP.py"), run_name="__main__")
        os.chdir(root / "task2_baselines")
        sys.argv = [
            "downstream_retrieval",
            f"--pretrained_folder={out}",
            "--query_dataset=easy_reaction_test",
            "--reference_dataset=all_ECs",
            "--query_modality=reaction",
            "--reference_modality=protein",
        ]
        runpy.run_path(str(ORIGINAL_DIR.parent / "downstream_retrieval.py"), run_name="__main__")
    finally:
        stack.close()
        sys.stdout = stdout
        os.chdir(cwd)
    similarities = np.load(
        out
        / "retrieval_results"
        / "easy_reaction_test_reaction2protein_retrieval_similarities.npy"
    )
    return final, similarities, np.array(ecs), (out / "log.txt").read_text()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work_dir", type=Path, required=True)
    p.add_argument("--ec2go", type=Path, required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--epochs", type=int, default=3)
    cli = p.parse_args()
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (no GPU is free)"
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    data = make_tiny_dataset(cli.dataset_root / SOURCE, cli.work_dir / "data")
    out = cli.work_dir / "out_b"
    key = "random_splits_grouped_rxn_key/all"
    split = next(s for s in discover_splits(data) if s.key == key)
    tables = build_tables(split, out, cli.ec2go)
    common = [
        "--split",
        key,
        "--seed",
        str(cli.seed),
        "--dataset_root",
        str(data),
        "--out_root",
        str(out),
    ]
    device = torch.device("cpu")

    with tiny_backbones(cli.work_dir):
        rec_b = Recorder()
        with rec_b.install():
            run = train.main(
                [*common, "--ec2go", str(cli.ec2go), "--epochs", str(cli.epochs)], device=device
            )
        evaluate.main([*common, "--reference", "all"], device=device)
        rec_a = Recorder()
        root_a = cli.work_dir / "orig_root"
        shutil.rmtree(root_a, ignore_errors=True)
        with rec_a.install():
            final_a, sims, ecs, log_a = run_original(
                root_a,
                tables,
                cli.seed,
                cli.epochs,
                tables / "reference_all.csv",
                tables / "test_queries.csv",
            )

    report = {}
    report["1_batches_equal"] = rec_a.batches == rec_b.batches
    report["1_n_batches"] = [len(rec_a.batches), len(rec_b.batches)]
    report["2_initial_weights_equal"] = rec_a.init_weights == rec_b.init_weights
    report["3_epoch_start_weights_equal"] = rec_a.epoch_weights == rec_b.epoch_weights
    log_b = pd.read_csv(run / "train_log.csv")
    lines_a = [line for line in log_a.splitlines() if line.startswith("CL Loss")]
    totals_a = [float(line.split("Total Loss: ")[1].split("\t")[0]) for line in lines_a]
    report["4_epoch_losses_a"], report["4_epoch_losses_b"] = (
        totals_a,
        log_b.total_loss.round(5).tolist(),
    )
    report["4_losses_equal"] = totals_a == report["4_epoch_losses_b"]
    best_a = max(i + 1 for i, line in enumerate(log_a.split("Epoch ")[1:]) if "save model" in line)
    best_b = json.loads((run / "train_summary.json").read_text())["best_epoch"]
    report["4_best_epoch"] = [best_a, best_b]
    report["4_final_weights_equal"] = final_a == hash_state(torch.load(run / "lastmodel.pth"))

    ranked_a = []
    for row in sims:
        frame = pd.DataFrame({"EC": ecs, "reaction similarity": row})
        frame["rank"] = frame["reaction similarity"].rank(ascending=False)
        frame.sort_values("rank", ascending=True, inplace=True)
        ranked_a.append(frame["EC"].values[:50].tolist())
    predictions = pd.read_parquet(run / "predictions.parquet")
    ranked_b = [list(r) for r in predictions["ranked_ecs"]]
    report["5_ranked_lists_equal"] = ranked_a == ranked_b
    report["5_queries"] = len(ranked_a)
    truth = [list(t) for t in predictions["true_ecs"]]
    acc_a = ec_level_accuracy(pd.DataFrame({"true_ecs": truth, "ranked_ecs": ranked_a}))
    report["5_accuracy_a_k1"], report["5_accuracy_b_k1"] = (
        acc_a["k=1"],
        json.loads((run / "metrics.json").read_text())["accuracy"]["k=1"],
    )
    report["5_accuracy_equal"] = (
        acc_a == json.loads((run / "metrics.json").read_text())["accuracy"]
    )
    print(json.dumps(report, indent=2))
    verdict = all(v for k, v in report.items() if k.endswith("_equal"))
    print("VERDICT", "EQUIVALENT" if verdict else "MISMATCH")


if __name__ == "__main__":
    main()
