"""Train the three CREEP projection heads on cached frozen-encoder vectors.

The sampler makes the draws of the original CREEPDatasetMineBatch and returns cache rows instead of tokens.
The loss, the Adam groups and the model selection are those of step_01_train_CREEP.py. The heads run under fp16 autocast with a GradScaler, as in the original loop.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from cache_frozen import fingerprints, open_stores
from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    Split,
    build_tables,
    discover_splits,
    train_args,
    write_json,
)
from model_utils import seed_everything
from train import atomic_save, rng_state, set_rng_state
from CREEP.datasets import CREEPDatasetMineBatch  # noqa: E402

FROZEN_TASK = "enzyme_retrieval_frozen"
LOG_EVERY = 1000


def frozen_run_dir(split: Split, out_root: Path, seed: int) -> Path:
    return Path(out_root) / FROZEN_TASK / split.group / split.name / f"seed_{seed}"


class RowSampler(CREEPDatasetMineBatch):
    """Per step: 16 rows of (protein, text, reaction) cache indices, drawn exactly like the original dataset."""

    def __init__(self, tables: Path, n_neg: int):
        super().__init__(
            f"{tables}/", str(tables / "reaction_train.csv"), None, None, None, 0, 0, 0, n_neg
        )
        self.proteins = sorted(
            {s.replace(" ", "") for v in self.clusterid2proteinseq.values() for s in v}
        )
        self.reactions = sorted({r for v in self.ec2rxns.values() for r in v})
        self.texts = sorted(set(self.ec2text.values()))
        protein, reaction, text = (
            {x: i for i, x in enumerate(items)}
            for items in (self.proteins, self.reactions, self.texts)
        )
        self.clusterid2proteinseq = {
            k: [protein[s.replace(" ", "")] for s in v]
            for k, v in self.clusterid2proteinseq.items()
        }
        self.ec2rxns = {ec: [reaction[r] for r in v] for ec, v in self.ec2rxns.items()}
        self.ec2text = {ec: text[t] for ec, t in self.ec2text.items()}

    def __getitem__(self, index):
        anchor = self.full_list[index]
        other_ecs = list(self.ec2rxns.keys())
        other_ecs.remove(anchor)
        ecs = [anchor, *random.sample(other_ecs, self.n_negs)]
        rows = [
            (self.sample_protein(ec), self.ec2text[ec], random.choice(self.ec2rxns[ec]))
            for ec in ecs
        ]
        return torch.tensor(rows)


def epoch_rows(dataset: RowSampler) -> torch.Tensor:
    """Rows of all steps of one epoch (steps, batch, 3). The draws do not depend on the model, so they can be made first."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return torch.stack([batch[0] for batch in loader])


def make_heads(args) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "protein": nn.Linear(1024, args.SSL_emb_dim),
            "text": nn.Linear(768, args.SSL_emb_dim),
            "reaction": nn.Linear(256, args.SSL_emb_dim),
        }
    )


def make_head_optimizer(heads: nn.ModuleDict, args) -> optim.Adam:
    lrs = {
        "protein": args.protein_lr * args.protein_lr_scale,
        "text": args.text_lr * args.text_lr_scale,
        "reaction": args.reaction_lr * args.reaction_lr_scale,
    }
    groups = [{"params": heads[name].parameters(), "lr": lr} for name, lr in lrs.items()]
    return optim.Adam(groups, weight_decay=args.decay)


def cl_loss(x: torch.Tensor, y: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    """EBM_NCE loss and accuracy of the original do_CL. The accuracy is a tensor, so nothing waits for the GPU."""
    neg_y = torch.cat([y.roll(-(i + 1), 0) for i in range(args.CL_neg_samples)], dim=0)
    neg_x = x.repeat((args.CL_neg_samples, 1))
    pred_pos = torch.sum(x * y, dim=1) / args.T
    pred_neg = torch.sum(neg_x * neg_y, dim=1) / args.T
    loss_pos = F.binary_cross_entropy_with_logits(pred_pos, torch.ones_like(pred_pos))
    loss_neg = F.binary_cross_entropy_with_logits(pred_neg, torch.zeros_like(pred_neg))
    loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)
    acc = ((pred_pos > 0).sum() + (pred_neg < 0).sum()).float() / (len(pred_pos) + len(pred_neg))
    return loss, acc


def contrastive_loss(protein, text, reaction, args) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over the six pairs of the original three-modality loss."""
    pairs = (
        (protein, text),
        (text, protein),
        (text, reaction),
        (protein, reaction),
        (reaction, text),
        (reaction, protein),
    )
    losses, accs = zip(*(cl_loss(x, y, args) for x, y in pairs))
    return sum(losses) / 6, sum(accs) / 6


def train_step(heads, optimizer, scaler, vectors: dict, rows: torch.Tensor, args):
    """One step of the original loop: heads, loss and Adam under fp16 autocast with a GradScaler (both off on the CPU)."""
    device = rows.device
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        protein = heads["protein"](vectors["protein"][rows[:, 0]])
        text = heads["text"](vectors["text"][rows[:, 1]])
        reaction = heads["reaction"](vectors["reaction"][rows[:, 2]])
        loss, acc = contrastive_loss(protein, text, reaction, args)
        loss = args.alpha_contrastive * loss
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach().float(), acc


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split", required=True, help="split key such as enzyme_sequence_splits/threshold_0.4"
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--weights_dir", type=Path, default=None, help="default OUT/pretrained")
    p.add_argument("--ec2go", type=Path, default=None, help="default OUT/reference/ec2go")
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--threads", type=int, default=8)
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
    run = frozen_run_dir(split, cli.out_root, cli.seed)
    run.mkdir(parents=True, exist_ok=True)
    tables = build_tables(split, cli.out_root, ec2go, cli.mmseqs, cli.threads)
    args = train_args(seed=cli.seed)
    if cli.epochs is not None:
        args.epochs = cli.epochs
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[creep] frozen {split.key} seed {cli.seed} device {device}", flush=True)

    dataset = RowSampler(tables, args.batch_size - 1)
    if cli.limit_ecs is not None:
        dataset.full_list = dataset.full_list[: cli.limit_ecs]
    stores = open_stores(cli.out_root, weights_dir)
    vectors = {
        name: torch.from_numpy(stores[name].load(items)).to(device)
        for name, items in (
            ("protein", dataset.proteins),
            ("reaction", dataset.reactions),
            ("text", dataset.texts),
        )
    }

    seed_everything(args.seed)
    heads = make_heads(args).to(device)
    optimizer = make_head_optimizer(heads, args)
    seed_everything(args.seed)  # the sampling stream does not depend on the head initialization
    state = {"epoch": 0, "best_epoch": 0, "optimal_loss": 1e10, "history": []}
    last = run / "checkpoint_last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        heads.load_state_dict(ckpt["heads"])
        optimizer.load_state_dict(ckpt["optimizer"])
        set_rng_state(ckpt["rng"])
        state.update({k: ckpt[k] for k in state})
        print(f"[creep] resumed after epoch {state['epoch']}", flush=True)

    while state["epoch"] < args.epochs:
        start = time.time()
        rows = epoch_rows(dataset).to(device)
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda"
        )  # new each epoch, as in the original
        loss_sum = acc_sum = torch.zeros((), device=device)
        for step, batch in enumerate(rows, 1):
            loss, acc = train_step(heads, optimizer, scaler, vectors, batch, args)
            loss_sum, acc_sum = loss_sum + loss, acc_sum + acc
            if step % LOG_EVERY == 0:
                print(
                    f"[creep] epoch {state['epoch'] + 1} step {step}/{len(rows)} loss {loss.item():.5f}",
                    flush=True,
                )
        mean_loss, mean_acc = (loss_sum / len(rows)).item(), (acc_sum / len(rows)).item()
        seconds = time.time() - start
        state["epoch"] += 1
        if mean_loss < state["optimal_loss"]:
            state["optimal_loss"], state["best_epoch"] = mean_loss, state["epoch"]
            atomic_save(heads.state_dict(), run / "bestmodel.pth")
        state["history"].append(
            {
                "epoch": state["epoch"],
                "contrastive_loss": mean_loss,
                "contrastive_acc": mean_acc,
                "seconds": seconds,
                "step_ms": seconds / len(rows) * 1000,
            }
        )
        print(
            f"Epoch {state['epoch']} CL Loss: {mean_loss:.5f} CL Acc: {mean_acc:.5f} Time: {seconds:.1f}s",
            flush=True,
        )
        atomic_save(
            {
                "heads": heads.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": rng_state(),
                **state,
            },
            last,
        )
        pd.DataFrame(state["history"]).to_csv(run / "train_log.csv", index=False)

    atomic_save(heads.state_dict(), run / "lastmodel.pth")
    write_json(
        run / "train_summary.json",
        {
            "split": split.key,
            "seed": cli.seed,
            "epochs": args.epochs,
            "best_epoch": state["best_epoch"],
            "best_train_loss": state["optimal_loss"],
            "precision": "heads under fp16 autocast with GradScaler on cached vectors",
            "hparams": vars(args),
            "tables": str(tables),
            "data": json.loads((tables / "summary.json").read_text()),
            "cache": fingerprints(weights_dir),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return run


if __name__ == "__main__":
    main()
