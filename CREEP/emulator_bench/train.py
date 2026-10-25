"""Train CREEP on one EMULaToR split with the original step_01_train_CREEP.py training loop."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    TASK,
    build_tables,
    discover_splits,
    train_args,
    write_json,
)
from model_utils import (
    Prefetch,
    apply_amp,
    build_model,
    checkpoint_blocks,
    disable_autocast_cache,
    load_original,
    make_optimizer,
    patch_t5,
    seed_everything,
    trim_padding,
)
from CREEP.datasets import CREEPDatasetMineBatch  # noqa: E402

EPOCH_LINE = re.compile(
    r"CL Loss: (\S+)\tCL Acc: (\S+)\tGenerative Loss: (\S+)\tTotal Loss: (\S+)\tTime: (\S+)"
)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split", required=True, help="split key such as enzyme_sequence_splits/threshold_0.4"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--weights_dir",
        type=Path,
        default=None,
        help="Hugging Face cache for ProtT5 and SciBERT (default OUT/pretrained)",
    )
    p.add_argument(
        "--ec2go",
        type=Path,
        default=None,
        help="ec2go file with EC descriptions (default OUT/reference/ec2go)",
    )
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--amp",
        action="store_true",
        help="bf16 autocast on Ampere or newer instead of the original fp16",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="bf16 (needs Ampere or newer or the CPU), SDPA attention and a shorter feed-forward in ProtT5, "
        "SciBERT and rxnfp inputs cut to the longest sequence, Adam stepped in backward; "
        "use with --checkpoint_activations all; not with --offload_adam",
    )
    p.add_argument(
        "--checkpoint_activations",
        choices=["all", "bert"],
        default=None,
        help="recompute transformer blocks in backward (all, or SciBERT and rxnfp only); same numbers, less GPU memory",
    )
    p.add_argument(
        "--offload_adam",
        type=float,
        nargs="?",
        const=0.0,
        default=None,
        metavar="RESIDENT_GB",
        help="keep the Adam moments in pinned CPU memory, except RESIDENT_GB of them on the GPU "
        "(default 0); same numbers, less GPU memory",
    )
    p.add_argument(
        "--epochs", type=int, default=None, help="smoke tests only; the original default is 40"
    )
    p.add_argument("--limit_ecs", type=int, default=None, help="smoke tests only")
    return p.parse_args(argv)


def main(argv=None, device: torch.device | None = None) -> Path:
    cli = parse_args(argv)
    weights_dir = cli.weights_dir or cli.out_root / "pretrained"
    ec2go = cli.ec2go or cli.out_root / "reference" / "ec2go"
    split = next(s for s in discover_splits(cli.dataset_root) if s.key == cli.split)
    run = split.run_dir(cli.out_root, cli.seed)
    run.mkdir(parents=True, exist_ok=True)
    tables = build_tables(split, cli.out_root, ec2go, cli.mmseqs, cli.threads)

    args = train_args(seed=cli.seed)
    if cli.epochs is not None:
        args.epochs = cli.epochs
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cli.fast and cli.offload_adam is not None:
        raise SystemExit(
            "--fast steps Adam in backward on the GPU; do not combine with --offload_adam"
        )
    precision = (
        apply_amp(device) if cli.amp or cli.fast else ("fp16" if device.type == "cuda" else "fp32")
    )
    if cli.fast:
        if precision == "fp16":
            raise SystemExit("--fast needs bf16 (Ampere or newer)")
        patch_t5()
    if cli.offload_adam is not None or cli.fast:
        disable_autocast_cache()
    print(
        f"[creep] {TASK}/{split.key} seed {cli.seed} device {device} precision {precision}",
        flush=True,
    )

    seed_everything(args.seed)
    model, (protein_tokenizer, text_tokenizer, reaction_tokenizer) = build_model(args, weights_dir)
    model.to(device)
    optimizer = make_optimizer(model, args, resident_gb=cli.offload_adam, in_backward=cli.fast)
    if cli.checkpoint_activations:
        checkpoint_blocks(model, protein=cli.checkpoint_activations == "all")

    dataset = CREEPDatasetMineBatch(
        dataset_path=f"{tables}/",
        train_file=str(tables / "reaction_train.csv"),
        protein_tokenizer=protein_tokenizer,
        text_tokenizer=text_tokenizer,
        protein_max_sequence_len=args.protein_max_sequence_len,
        text_max_sequence_len=args.text_max_sequence_len,
        reaction_tokenizer=reaction_tokenizer,
        reaction_max_sequence_len=args.reaction_max_sequence_len,
        n_neg=args.batch_size - 1,
    )
    if cli.limit_ecs is not None:
        dataset.full_list = dataset.full_list[: cli.limit_ecs]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    original = load_original("step_01_train_CREEP")
    original.args, original.model, original.optimizer, original.device = (
        args,
        model,
        optimizer,
        device,
    )
    original.optimal_loss = 1e10
    state = {"epoch": 0, "best_epoch": 0, "history": []}

    def save_best(save_best=True):
        atomic_save(model.state_dict(), run / "bestmodel.pth")
        state["best_epoch"] = state["epoch"] + 1

    original.save_model = save_best

    last = run / "checkpoint_last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        original.optimal_loss = ckpt["optimal_loss"]
        state.update(epoch=ckpt["epoch"], best_epoch=ckpt["best_epoch"], history=ckpt["history"])
        print(f"[creep] resumed after epoch {state['epoch']}", flush=True)

    while state["epoch"] < args.epochs:
        print(f"Epoch {state['epoch'] + 1}", flush=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            original.train(Prefetch(dataloader, transform=trim_padding if cli.fast else None))
        print(buffer.getvalue(), end="", flush=True)
        cl, acc, gen, total, seconds = map(float, EPOCH_LINE.search(buffer.getvalue()).groups())
        state["epoch"] += 1
        state["history"].append(
            {
                "epoch": state["epoch"],
                "contrastive_loss": cl,
                "contrastive_acc": acc,
                "generative_loss": gen,
                "total_loss": total,
                "seconds": seconds,
            }
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the offloaded Adam state is copied asynchronously
        atomic_save(
            {
                "epoch": state["epoch"],
                "best_epoch": state["best_epoch"],
                "history": state["history"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimal_loss": original.optimal_loss,
                "rng": rng_state(),
            },
            last,
        )
        pd.DataFrame(state["history"]).to_csv(run / "train_log.csv", index=False)

    atomic_save(model.state_dict(), run / "lastmodel.pth")
    write_json(
        run / "train_summary.json",
        {
            "split": split.key,
            "seed": cli.seed,
            "epochs": args.epochs,
            "best_epoch": state["best_epoch"],
            "best_train_loss": original.optimal_loss,
            "precision": precision,
            "hparams": vars(args),
            "tables": str(tables),
            "data": json.loads((tables / "summary.json").read_text()),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return run


if __name__ == "__main__":
    main()
