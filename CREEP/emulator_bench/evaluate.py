"""Rank EC numbers for the test reactions with the original CREEP extraction and retrieval steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    Split,
    discover_splits,
    hparams,
    train_args,
    write_json,
)
from metrics import K_VALUES, ec_level_accuracy
from cache_frozen import open_stores
from model_utils import apply_amp, build_model, load_original
from train_frozen import frozen_run_dir, make_heads
from CREEP.datasets import SingleModalityDataset  # noqa: E402
from CREEP.models import SingleModalityModel  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--weights_dir", type=Path, default=None)
    p.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on Ampere or newer instead of the original fp16",
    )
    p.add_argument(
        "--frozen",
        action="store_true",
        help="use the heads of train_frozen.py on the cached encoder vectors",
    )
    p.add_argument(
        "--reference",
        choices=["all", "train"],
        default="all",
        help="proteins that define the EC vectors: all split proteins (default, the pool of every retrieval baseline) or train only",
    )
    return p.parse_args(argv)


def extract(
    extractor,
    model,
    modality: str,
    csv_path: Path,
    tokenizer,
    max_len: int,
    device,
    batch_size: int,
):
    """Latent vectors from the original step_02 extract() loop."""
    extractor.args, extractor.model, extractor.device = (
        argparse.Namespace(verbose=False),
        model,
        device,
    )
    model.to(device).eval()
    dataset = SingleModalityDataset(
        file=str(csv_path), tokenizer=tokenizer, max_sequence_len=max_len, modality=modality
    )
    repr_array = extractor.extract(
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    )
    model.cpu()
    return repr_array


def ec_centers(repr_array: np.ndarray, reference: pd.DataFrame, ec_list: np.ndarray) -> np.ndarray:
    """Mean protein vector per EC, as in step_02 --get_cluster_centers."""
    ec2index = reference.reset_index().groupby("EC number")["index"].apply(list).to_dict()
    centers = np.zeros((len(ec_list), repr_array.shape[1]))
    for i, ec in enumerate(ec_list):
        if ec in ec2index:
            centers[i] = np.mean(repr_array[ec2index[ec]], axis=0)
    return centers


def rank_ecs(query_repr: np.ndarray, centers: np.ndarray, ec_list: np.ndarray, top: int):
    """Cosine similarity (zero norms set to 1), then rank like downstream_retrieval.py and rank_tabulate_similarities.py."""
    ranked, scores = [], []
    for i in range(len(query_repr)):
        query = query_repr[i].reshape(1, -1)
        normalization = np.linalg.norm(query) * np.linalg.norm(centers, axis=1)
        normalization[normalization == 0] = 1
        similarity = np.dot(query, centers.T) / normalization
        frame = pd.DataFrame({"EC": ec_list, "reaction similarity": similarity[0]})
        frame["reaction rank"] = frame["reaction similarity"].rank(ascending=False)
        frame["overall rank"] = frame["reaction rank"]
        frame.sort_values("overall rank", ascending=True, inplace=True)
        ranked.append(frame["EC"].values[:top].tolist())
        scores.append(frame["reaction similarity"].values[:top])
    return ranked, scores


def frozen_representations(run, out_root, weights_dir, sequences, reactions, device):
    """Cached encoder vectors through the trained heads."""
    heads = make_heads(train_args())
    heads.load_state_dict(torch.load(run / "bestmodel.pth", map_location="cpu"))
    heads.to(device).eval()
    stores = open_stores(out_root, weights_dir)

    def project(name: str, items: pd.Series) -> np.ndarray:
        codes, unique = pd.factorize(items)
        vectors = torch.from_numpy(stores[name].load(list(unique))).to(device)
        # like the original extract(): fp16 autocast on CUDA, fp16 output
        with (
            torch.no_grad(),
            torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
        ):
            return heads[name](vectors).cpu().numpy()[codes]

    return project("protein", sequences), project("reaction", reactions)


def finetuned_representations(run, weights_dir, tables, reference_csv, device):
    """Protein and test-reaction vectors of the fine-tuned model, from the original step_02 extract()."""
    args = train_args()
    model, (protein_tokenizer, _, reaction_tokenizer) = build_model(args, weights_dir)
    model.load_state_dict(torch.load(run / "bestmodel.pth", map_location="cpu"))
    extractor = load_original("step_02_extract_CREEP")
    batch_size = hparams("extract")["batch_size"]
    protein = SingleModalityModel(
        model.protein_model, model.protein2latent_model, args.protein_backbone_model, "protein"
    )
    protein_repr = extract(
        extractor,
        protein,
        "protein",
        reference_csv,
        protein_tokenizer,
        args.protein_max_sequence_len,
        device,
        batch_size,
    )
    reaction = SingleModalityModel(
        model.reaction_model, model.reaction2latent_model, args.reaction_backbone_model, "reaction"
    )
    reaction_repr = extract(
        extractor,
        reaction,
        "reaction",
        tables / "test_queries.csv",
        reaction_tokenizer,
        args.reaction_max_sequence_len,
        device,
        batch_size,
    )
    return protein_repr, reaction_repr


def main(argv=None, device: torch.device | None = None) -> Path:
    cli = parse_args(argv)
    weights_dir = cli.weights_dir or cli.out_root / "pretrained"
    split = next(s for s in discover_splits(cli.dataset_root) if s.key == cli.split)
    run = (frozen_run_dir if cli.frozen else Split.run_dir)(split, cli.out_root, cli.seed)
    tables = split.tables_dir(cli.out_root)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cli.amp:
        apply_amp(device)

    reference_csv = tables / f"reference_{cli.reference}.csv"
    reference = pd.read_csv(reference_csv)
    queries = pd.read_csv(tables / "test_queries.csv")
    ec_list = np.array(sorted(reference["EC number"].unique()))
    if cli.frozen:
        protein_repr, reaction_repr = frozen_representations(
            run, cli.out_root, weights_dir, reference["Sequence"], queries["Reaction"], device
        )
    else:
        protein_repr, reaction_repr = finetuned_representations(
            run, weights_dir, tables, reference_csv, device
        )

    ranked, scores = rank_ecs(
        reaction_repr, ec_centers(protein_repr, reference, ec_list), ec_list, max(K_VALUES)
    )
    predictions = pd.DataFrame(
        {
            "rxn_smiles": queries["Reaction"],
            "true_ecs": [ecs.split(";") for ecs in queries["EC number"]],
            "ranked_ecs": ranked,
            "ranked_scores": scores,
        }
    )
    tmp = run / ".predictions.parquet.tmp"
    predictions.to_parquet(tmp, index=False)
    tmp.replace(run / "predictions.parquet")

    summary = {
        "split": split.key,
        "seed": cli.seed,
        "checkpoint": str(run / "bestmodel.pth"),
        "queries": len(predictions),
        "reference": cli.reference,
        "candidate_proteins": len(reference),
        "candidate_ecs": len(ec_list),
        "accuracy": ec_level_accuracy(predictions),
    }
    write_json(run / "metrics.json", summary)
    train_summary = run / "train_summary.json"
    write_json(
        run / "run_summary.json",
        {
            **summary,
            "prepare": json.loads((tables / "summary.json").read_text()),
            "train": json.loads(train_summary.read_text()) if train_summary.exists() else None,
        },
    )
    print(f"[creep] {split.key} seed {cli.seed}: {summary['accuracy']['k=1']}", flush=True)
    return run


if __name__ == "__main__":
    main()
