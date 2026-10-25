"""The frozen-encoder path against the original CREEP code with only the three backbones frozen.

Reference: original model forward, dataset, do_CL, GradScaler loop and Adam, with requires_grad off on the backbones
and the backbones left out of the optimizer. Candidate: cache_frozen.py, train_frozen.py and evaluate.py --frozen.
--mode fp32 runs both sides without autocast, without the SDPA patch and without trimming.
--mode production runs the original fp16 autocast against the cache settings (fp16 autocast, SDPA, trimmed rxnfp and SciBERT).
Runs on one GPU (cuda:0 of the visible devices). Prints the differences. It sets no tolerance.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from transformers.models.t5.modeling_t5 import T5Attention, T5DenseActDense
from torch.utils.data import DataLoader

import cache_frozen
import evaluate
import train_frozen
from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    build_tables,
    discover_splits,
    train_args,
)
from CREEP.datasets import CREEPDatasetMineBatch, dataset_CREEP
from CREEP.models import SingleModalityModel
from metrics import K_VALUES, ec_level_accuracy
from model_utils import _NoScaler, build_model, load_original, seed_everything
from testing import make_tiny_dataset

SOURCE = "with_protein_information/random_splits_grouped_rxn_key"
SPLIT = "random_splits_grouped_rxn_key/all"
SEED = 666
HEADS = {
    "protein": "protein2latent_model",
    "text": "text2latent_model",
    "reaction": "reaction2latent_model",
}


@contextlib.contextmanager
def no_autocast():
    """fp32 reference: the original loops call torch.cuda.amp.autocast and GradScaler."""
    autocast, scaler = torch.cuda.amp.autocast, torch.cuda.amp.GradScaler
    torch.cuda.amp.autocast = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.amp.GradScaler = _NoScaler
    try:
        yield
    finally:
        torch.cuda.amp.autocast, torch.cuda.amp.GradScaler = autocast, scaler


@torch.no_grad()
def encode_fp32(modality, tokenizer, model, texts, device):
    """Untrimmed fp32 forward at the full max_length: the vectors the original model computes."""
    max_len = getattr(train_args(), f"{modality}_max_sequence_len")
    parts = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        tokens = tokenizer(
            [" ".join(t) for t in batch] if modality == "protein" else batch,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        out = model(tokens.input_ids, tokens.attention_mask)
        vector = {
            "protein": lambda: out.last_hidden_state.mean(1),
            "text": lambda: out.pooler_output,
            "reaction": lambda: out.last_hidden_state[:, 0],
        }[modality]()
        parts.append(vector.float().cpu())
    return torch.cat(parts).numpy()


def fill_cache_fp32(out_root, weights_dir, tables, device):
    stores = cache_frozen.open_stores(out_root, weights_dir)
    entities = cache_frozen.split_entities(tables, "train")
    for modality, store in stores.items():
        todo = store.missing(entities[modality])
        tokenizer, model = cache_frozen.load_encoder(modality, weights_dir, device)
        for i in range(0, len(todo), cache_frozen.CHUNK):
            part = todo[i : i + cache_frozen.CHUNK]
            store.write(part, encode_fp32(modality, tokenizer, model, part, device))
        del model
        torch.cuda.empty_cache()


def build_cache(cli, split, data, out_root, weights_dir, tables, device):
    if cli.mode == "fp32":
        fill_cache_fp32(out_root, weights_dir, tables, device)
    else:
        cache_frozen.main(
            ["--dataset_root", str(data), "--out_root", str(out_root), "--ec2go", str(cli.ec2go),
             "--weights_dir", str(weights_dir), "--splits", split.key],
            device=device,
        )  # fmt: skip


def precision_step(precision: str):
    """Training step of the candidate heads. fp16 is train_frozen.train_step; fp32 and bf16 are variants for comparison.

    train_frozen.main creates the GradScaler at the start of each epoch, as the original does.
    """
    real_step = train_frozen.train_step

    def step(heads, optimizer, scaler, vectors, rows, args):
        if precision == "fp16":
            return real_step(heads, optimizer, scaler, vectors, rows, args)
        dtype = torch.bfloat16 if precision == "bf16" else None
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            protein = heads["protein"](vectors["protein"][rows[:, 0]])
            text = heads["text"](vectors["text"][rows[:, 1]])
            reaction = heads["reaction"](vectors["reaction"][rows[:, 2]])
            loss, acc = train_frozen.contrastive_loss(protein, text, reaction, args)
            loss = args.alpha_contrastive * loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.detach().float(), acc

    return step


def run_candidate(cli, split, data, out_root, weights_dir, precision):
    """train_frozen.main and evaluate.main --frozen on the cache of out_root, heads in the given precision."""
    device = torch.device("cuda:0")
    record = {"loss": [], "rows": [], "sampler": None}
    real_rows, real_sampler, real_step = (
        train_frozen.epoch_rows,
        train_frozen.RowSampler,
        train_frozen.train_step,
    )
    precise_step = precision_step(precision)

    def step(*a, **k):
        loss, acc = precise_step(*a, **k)
        record["loss"].append(loss.clone())
        return loss, acc

    def rows(dataset):
        out = real_rows(dataset)
        record["rows"].append(out.clone())
        return out

    def sampler(*a, **k):
        record["sampler"] = real_sampler(*a, **k)
        return record["sampler"]

    train_frozen.train_step, train_frozen.epoch_rows, train_frozen.RowSampler = step, rows, sampler
    common = [
        "--split", split.key, "--seed", str(SEED), "--dataset_root", str(data),
        "--out_root", str(out_root), "--weights_dir", str(weights_dir),
    ]  # fmt: skip
    torch.cuda.reset_peak_memory_stats()
    try:
        run = train_frozen.main(
            [*common, "--ec2go", str(cli.ec2go), "--epochs", str(cli.epochs), "--limit_ecs", str(cli.limit_ecs)],
            device=device,
        )  # fmt: skip
    finally:
        train_frozen.train_step, train_frozen.epoch_rows, train_frozen.RowSampler = (
            real_step,
            real_rows,
            real_sampler,
        )
    peak = torch.cuda.max_memory_allocated() / 2**30
    evaluate.main([*common, "--frozen", "--reference", "train"], device=device)
    sampler_ = record["sampler"]
    seen = [
        (sampler_.proteins[p], sampler_.texts[t], sampler_.reactions[r])
        for p, t, r in torch.cat(record["rows"]).reshape(-1, 3).tolist()
    ]
    last = torch.load(run / "lastmodel.pth", map_location="cpu")
    final = {m: {k: last[f"{m}.{k}"] for k in ("weight", "bias")} for m in HEADS}
    return {
        "losses": torch.stack(record["loss"]).cpu().numpy(),
        "samples": seen,
        "final": final,
        "predictions": pd.read_parquet(run / "predictions.parquet"),
        "step_ms": pd.read_csv(run / "train_log.csv").step_ms.mean(),
        "peak_gib": peak,
    }


def run_reference(cli, tables, weights_dir, init_heads):
    """The original loop with frozen backbones. Returns step losses, drawn samples, final and best heads, predictions."""
    device = torch.device("cuda:0")
    args = train_args(seed=SEED)
    original = load_original("step_01_train_CREEP")
    losses, seen = [], []
    real_do_cl, real_encode = original.do_CL, dataset_CREEP.encode_sequence

    def do_cl(x, y, a):
        loss, acc = real_do_cl(x, y, a)
        losses.append(loss.item())
        return loss, acc

    patches = contextlib.ExitStack()
    if cli.mode == "fp32":
        patches.enter_context(no_autocast())
    with patches:
        model, (ptok, ttok, rtok) = build_model(args, weights_dir)
        for backbone in (model.protein_model, model.text_model, model.reaction_model):
            backbone.requires_grad_(False)
        for modality, name in HEADS.items():
            getattr(model, name).load_state_dict(init_heads[modality])
        model.to(device)
        lr = {
            "protein": args.protein_lr * args.protein_lr_scale,
            "text": args.text_lr * args.text_lr_scale,
            "reaction": args.reaction_lr * args.reaction_lr_scale,
        }
        optimizer = optim.Adam(
            [
                {"params": model.protein2latent_model.parameters(), "lr": lr["protein"]},
                {"params": model.text2latent_model.parameters(), "lr": lr["text"]},
                {"params": model.reaction2latent_model.parameters(), "lr": lr["reaction"]},
                {
                    "params": model.reaction2protein_facilitator_model.parameters(),
                    "lr": lr["reaction"],
                },
                {
                    "params": model.protein2reaction_facilitator_model.parameters(),
                    "lr": lr["protein"],
                },
            ],
            weight_decay=args.decay,
        )
        dataset = CREEPDatasetMineBatch(
            dataset_path=f"{tables}/",
            train_file=str(tables / "reaction_train.csv"),
            protein_tokenizer=ptok,
            text_tokenizer=ttok,
            protein_max_sequence_len=args.protein_max_sequence_len,
            text_max_sequence_len=args.text_max_sequence_len,
            reaction_tokenizer=rtok,
            reaction_max_sequence_len=args.reaction_max_sequence_len,
            n_neg=args.batch_size - 1,
        )
        dataset.full_list = dataset.full_list[: cli.limit_ecs]

        def encode(sequence, tokenizer, max_len):
            seen.append(sequence)
            return real_encode(sequence, tokenizer, max_len)

        best = {}

        def save_model(save_best=True):
            best.update(
                copy.deepcopy({m: getattr(model, n).state_dict() for m, n in HEADS.items()})
            )

        original.args, original.model, original.optimizer, original.device = (
            args,
            model,
            optimizer,
            device,
        )
        original.optimal_loss, original.save_model, original.do_CL = 1e10, save_model, do_cl
        dataset_CREEP.encode_sequence = encode
        seed_everything(SEED)
        try:
            for _ in range(cli.epochs):
                original.train(DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0))
        finally:
            dataset_CREEP.encode_sequence = real_encode
        final = {m: copy.deepcopy(getattr(model, n).state_dict()) for m, n in HEADS.items()}

        extractor = load_original("step_02_extract_CREEP")
        reference_csv, queries_csv = tables / "reference_train.csv", tables / "test_queries.csv"
        for modality, name in HEADS.items():
            getattr(model, name).load_state_dict(best[modality])
        batch_size = 8  # per-sample outputs; the original 256 needs more GPU memory than is free
        protein = SingleModalityModel(
            model.protein_model, model.protein2latent_model, "ProtT5", "protein"
        )
        protein_repr = evaluate.extract(
            extractor,
            protein,
            "protein",
            reference_csv,
            ptok,
            args.protein_max_sequence_len,
            device,
            batch_size,
        )
        reaction = SingleModalityModel(
            model.reaction_model, model.reaction2latent_model, "rxnfp", "reaction"
        )
        reaction_repr = evaluate.extract(
            extractor,
            reaction,
            "reaction",
            queries_csv,
            rtok,
            args.reaction_max_sequence_len,
            device,
            batch_size,
        )
    reference = pd.read_csv(reference_csv)
    queries = pd.read_csv(queries_csv)
    ec_list = np.array(sorted(reference["EC number"].unique()))
    ranked, scores = evaluate.rank_ecs(
        reaction_repr,
        evaluate.ec_centers(protein_repr, reference, ec_list),
        ec_list,
        max(K_VALUES),
    )
    predictions = pd.DataFrame(
        {
            "true_ecs": [e.split(";") for e in queries["EC number"]],
            "ranked_ecs": ranked,
            "ranked_scores": scores,
        }
    )
    # the reference seen list holds the sequences as encoded: protein (space-joined), text, reaction per sample
    samples = [
        (p.replace(" ", ""), t, r) for p, t, r in np.array(seen, dtype=object).reshape(-1, 3)
    ]
    final = {m: {k: v.cpu() for k, v in heads.items()} for m, heads in final.items()}
    return {
        "losses": np.array(losses).reshape(-1, 6).mean(1),
        "samples": samples,
        "final": final,
        "predictions": predictions,
    }


def diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """max abs diff, max abs diff relative to max |b|, norm of the diff relative to the norm of b."""
    d = (a - b).abs()
    return d.max().item(), (d.max() / b.abs().max()).item(), ((a - b).norm() / b.norm()).item()


def report(label: str, ref: dict, cand: dict, init: dict, epochs: int) -> None:
    """Print how cand differs from ref."""
    print(f"== {label}")
    print(
        "sampler draws identical (EC, protein, reaction, text of every step):",
        ref["samples"] == cand["samples"],
    )
    abs_d = np.abs(cand["losses"] - ref["losses"])
    rel_d = abs_d / np.abs(ref["losses"])
    print(f"step losses: max abs {abs_d.max():.3e} (step {abs_d.argmax() + 1}), max rel {rel_d.max():.3e}, "
          f"mean abs {abs_d.mean():.3e}")  # fmt: skip
    print("epoch-mean losses reference:", np.round(ref["losses"].reshape(epochs, -1).mean(1), 5),
          "candidate:", np.round(cand["losses"].reshape(epochs, -1).mean(1), 5))  # fmt: skip
    for modality in HEADS:
        for kind in ("weight", "bias"):
            a, b = cand["final"][modality][kind], ref["final"][modality][kind]
            moved = (b - init[modality][kind]).norm().item()
            d = diff(a, b)
            print(f"head {modality}.{kind}: max abs {d[0]:.3e}, norm rel {d[2]:.3e}, "
                  f"diff norm / movement norm {(a - b).norm().item() / moved:.3e}")  # fmt: skip
    ref_p, cand_p = ref["predictions"], cand["predictions"]
    print("top-k lists identical:", {
        k: sum(list(a[:k]) == list(b[:k]) for a, b in zip(ref_p["ranked_ecs"], cand_p["ranked_ecs"]))
        for k in (1, 3, 10, 50)
    }, "of", len(ref_p))  # fmt: skip
    ref_acc, cand_acc = ec_level_accuracy(ref_p), ec_level_accuracy(cand_p)
    for k in ("k=1", "k=10", "k=50"):
        print(f"accuracy {k} (candidate - reference, % points):",
              {lvl: round(cand_acc[k][lvl] - ref_acc[k][lvl], 2) for lvl in ref_acc[k]})  # fmt: skip
    if "step_ms" in cand:
        print(
            f"head step time {cand['step_ms']:.2f} ms, peak GPU memory during head training {cand['peak_gib']:.2f} GiB"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["fp32", "production"], required=True)
    p.add_argument("--work_dir", type=Path, required=True)
    p.add_argument("--ec2go", type=Path, default=DEFAULT_OUT_ROOT / "reference" / "ec2go")
    p.add_argument("--weights_dir", type=Path, default=DEFAULT_OUT_ROOT / "pretrained")
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--rows", type=int, nargs=3, default=(800, 200, 200), help="train, val, test rows"
    )
    p.add_argument("--limit_ecs", type=int, default=150)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--heads",
        nargs="+",
        choices=["fp32", "fp16", "bf16"],
        default=["fp32"],
        help="candidate head precisions",
    )
    p.add_argument(
        "--noise_floor", action="store_true", help="also run the reference twice and compare"
    )
    p.add_argument("--max_gb", type=float, default=8.0, help="cap on this process's GPU memory")
    cli = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(
        cli.max_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory
    )
    work = cli.work_dir / cli.mode
    if work.exists():
        shutil.rmtree(work)
    data = make_tiny_dataset(cli.dataset_root / SOURCE, work / "data", sizes=tuple(cli.rows))
    out_root = work / "out"
    split = next(s for s in discover_splits(data) if s.key == SPLIT)
    tables = build_tables(split, out_root, cli.ec2go)

    attention, feed_forward = T5Attention.forward, T5DenseActDense.forward
    build_cache(cli, split, data, out_root, cli.weights_dir, tables, torch.device("cuda:0"))
    T5Attention.forward, T5DenseActDense.forward = (
        attention,
        feed_forward,
    )  # the reference is the unpatched original
    seed_everything(SEED)
    init = {k: v.state_dict() for k, v in train_frozen.make_heads(train_args()).items()}
    reference = run_reference(cli, tables, cli.weights_dir, init)
    steps = len(reference["losses"])
    for precision in cli.heads:
        candidate_root = work / f"out_{precision}"
        candidate_root.mkdir()
        (candidate_root / "embeddings").symlink_to((out_root / "embeddings").resolve())
        candidate = run_candidate(cli, split, data, candidate_root, cli.weights_dir, precision)
        report(f"mode {cli.mode}, heads {precision}: {steps} steps, {cli.epochs} epochs of {cli.limit_ecs} ECs",
               reference, candidate, init, cli.epochs)  # fmt: skip
    if cli.noise_floor:
        report(
            "noise floor: the reference run twice",
            reference,
            run_reference(cli, tables, cli.weights_dir, init),
            init,
            cli.epochs,
        )
    print("peak GPU memory GiB (whole process)", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main()
