"""Cache the frozen CREEP encoder outputs: one fp32 vector per unique sequence, reaction and EC text.

The cache is shared by all splits and seeds. Re-running only encodes what is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    build_tables,
    discover_splits,
    sha256_file,
    train_args,
    write_json,
)
from model_utils import (
    PROTEIN_HUB_ID,
    RXNFP_DIR,
    TEXT_HUB_ID,
    load_reaction_tokenizer,
    patch_t5,
)
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertModel,
    T5EncoderModel,
    T5Tokenizer,
)

CACHE_VERSION = 1
MODALITIES = ("protein", "reaction", "text")
BATCH = {"protein": 32, "reaction": 128, "text": 128}
CHUNK = 2048  # entities per cache file


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def cache_root(out_root: Path) -> Path:
    return Path(out_root) / "embeddings" / f"frozen_v{CACHE_VERSION}"


def revision(hub_id: str, cache_dir: Path) -> str | None:
    return AutoConfig.from_pretrained(hub_id, cache_dir=str(cache_dir))._commit_hash


def fingerprints(weights_dir: Path) -> dict[str, dict]:
    """Model and recipe of each modality. A cache with another fingerprint is stale."""
    args = train_args()
    protein_dir, text_dir = (
        Path(weights_dir) / "pretrained_ProtT5",
        Path(weights_dir) / "pretrained_SciBert",
    )
    base = {"version": CACHE_VERSION, "precision": "fp16 autocast, fp32 vectors"}
    return {
        "protein": {
            **base,
            "model": PROTEIN_HUB_ID,
            "revision": revision(PROTEIN_HUB_ID, protein_dir),
            "max_length": args.protein_max_sequence_len,
            "recipe": "space-joined residues, padded to max_length, mean over all positions, SDPA T5",
        },
        "reaction": {
            **base,
            "model": "rxnfp",
            "weights_sha256": sha256_file(RXNFP_DIR / "pytorch_model.bin"),
            "vocab_sha256": sha256_file(RXNFP_DIR / "vocab.txt"),
            "max_length": args.reaction_max_sequence_len,
            "recipe": "CLS vector, padding cut to the longest sequence of the batch",
        },
        "text": {
            **base,
            "model": TEXT_HUB_ID,
            "revision": revision(TEXT_HUB_ID, text_dir),
            "max_length": args.text_max_sequence_len,
            "recipe": "pooler_output of the EC description, padding cut to the longest sequence of the batch",
        },
    }


class Store:
    """Vectors of one modality in chunk files. Each entity is keyed by the SHA-1 of its text."""

    def __init__(self, root: Path, modality: str, fingerprint: dict):
        self.dir = Path(root) / modality
        stamp = self.dir / "fingerprint.json"
        if stamp.exists():
            if json.loads(stamp.read_text()) != fingerprint:
                raise RuntimeError(
                    f"{self.dir} was built with other settings; remove it to rebuild"
                )
        else:
            write_json(stamp, fingerprint)

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("chunk_*.npz"))

    def keys(self) -> set[str]:
        found: set[str] = set()
        for file in self.files():
            with np.load(file) as data:
                found.update(data["keys"].tolist())
        return found

    def missing(self, texts, shard: int = 0, shards: int = 1) -> list[str]:
        """Texts of this shard that are not cached, shortest first. A shard owns the keys with key mod shards == shard."""
        have = self.keys()
        todo = {
            t for t in texts if (k := text_key(t)) not in have and int(k[:8], 16) % shards == shard
        }
        return sorted(todo, key=lambda t: (len(t), t))

    def write(self, texts: list[str], vectors: np.ndarray) -> None:
        keys = np.array([text_key(t) for t in texts])
        name = f"chunk_{text_key(''.join(keys))[:16]}.npz"
        tmp = self.dir / f".{name}.tmp"
        with open(tmp, "wb") as handle:
            np.savez(handle, keys=keys, vectors=vectors.astype(np.float32))
        os.replace(tmp, self.dir / name)

    def load(self, texts: list[str]) -> np.ndarray:
        """Rows in the order of texts (unique texts)."""
        row = {text_key(t): i for i, t in enumerate(texts)}
        out, found = None, set()
        for file in self.files():
            with np.load(file) as data:
                hits = [(i, row[k]) for i, k in enumerate(data["keys"].tolist()) if k in row]
                if not hits:
                    continue
                src, dst = map(list, zip(*hits))
                vectors = data["vectors"]
                if out is None:
                    out = np.empty((len(texts), vectors.shape[1]), np.float32)
                out[dst] = vectors[src]
                found.update(dst)
        if len(found) < len(texts):
            raise KeyError(
                f"{len(texts) - len(found)} {self.dir.name} vectors are not cached; run cache_frozen.py"
            )
        return out


def open_stores(out_root: Path, weights_dir: Path) -> dict[str, Store]:
    root = cache_root(out_root)
    return {m: Store(root, m, f) for m, f in fingerprints(weights_dir).items()}


def split_entities(tables: Path, reference: str) -> dict[str, set[str]]:
    """Unique sequences, reactions and EC texts that training and evaluation of one split read."""

    def column(name: str, header: str) -> set[str]:
        return set(pd.read_csv(Path(tables) / name, usecols=[header])[header])

    return {
        "protein": column("protein2EC.csv", "Sequence")
        | column(f"reference_{reference}.csv", "Sequence"),
        "reaction": column("reaction_train.csv", "Reaction")
        | column("test_queries.csv", "Reaction"),
        "text": column("text2EC.csv", "Text"),
    }


def load_encoder(modality: str, weights_dir: Path, device: torch.device):
    """Tokenizer and eval-mode model, loaded as the original build_model does."""
    weights_dir = Path(weights_dir)
    if modality == "protein":
        cache = str(weights_dir / "pretrained_ProtT5")
        tokenizer = T5Tokenizer.from_pretrained(
            PROTEIN_HUB_ID, do_lower_case=False, cache_dir=cache
        )
        model = T5EncoderModel.from_pretrained(
            PROTEIN_HUB_ID, cache_dir=cache, torch_dtype=torch.float32
        )
        if device.type == "cuda":
            patch_t5()
    elif modality == "text":
        cache = str(weights_dir / "pretrained_SciBert")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_HUB_ID, cache_dir=cache)
        model = AutoModel.from_pretrained(TEXT_HUB_ID, cache_dir=cache)
    else:
        tokenizer, model = load_reaction_tokenizer(), BertModel.from_pretrained(str(RXNFP_DIR))
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def encode(modality: str, tokenizer, model, texts: list[str], device: torch.device) -> np.ndarray:
    """Original CREEP vectors (fp16 autocast on CUDA).

    ProtT5 keeps all max_length positions because its vector is a mean over padding too.
    SciBERT and rxnfp batches are sorted by length and cut to their longest sequence.
    """
    args = train_args()
    max_len = getattr(args, f"{modality}_max_sequence_len")
    tokens = tokenizer(
        [" ".join(t) for t in texts] if modality == "protein" else texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    lengths = tokens.attention_mask.sum(1)
    order = torch.arange(len(texts)) if modality == "protein" else lengths.argsort()
    batch, parts = BATCH[modality], []
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(texts), batch):
            ix = order[i : i + batch]
            width = max_len if modality == "protein" else int(lengths[ix].max())
            out = model(
                tokens.input_ids[ix, :width].to(device),
                tokens.attention_mask[ix, :width].to(device),
            )
            if modality == "protein":
                vector = out.last_hidden_state.mean(1)
            elif modality == "text":
                vector = out.pooler_output
            else:
                vector = out.last_hidden_state[:, 0]
            parts.append(vector.float().cpu())
    result = torch.empty_like(torch.cat(parts))
    result[order] = torch.cat(parts)
    return result.numpy()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--weights_dir", type=Path, default=None, help="default OUT/pretrained")
    p.add_argument("--ec2go", type=Path, default=None, help="default OUT/reference/ec2go")
    p.add_argument("--splits", nargs="*", default=None, help="default all found")
    p.add_argument(
        "--reference",
        choices=["all", "train"],
        default="train",
        help="reference proteins to cache (see evaluate.py)",
    )
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument(
        "--shard", type=int, default=0, help="this process encodes the keys of one shard"
    )
    p.add_argument(
        "--shards", type=int, default=1, help="processes that share the cache; one per GPU"
    )
    return p.parse_args(argv)


def main(argv=None, device: torch.device | None = None) -> None:
    cli = parse_args(argv)
    weights_dir = cli.weights_dir or cli.out_root / "pretrained"
    ec2go = cli.ec2go or cli.out_root / "reference" / "ec2go"
    entities: dict[str, set[str]] = {m: set() for m in MODALITIES}
    for split in discover_splits(cli.dataset_root, cli.splits):
        tables = build_tables(split, cli.out_root, ec2go, cli.mmseqs, cli.threads)
        for modality, items in split_entities(tables, cli.reference).items():
            entities[modality] |= items
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    stores = open_stores(cli.out_root, weights_dir)
    for modality in MODALITIES:
        store = stores[modality]
        todo = store.missing(entities[modality], cli.shard, cli.shards)
        print(
            f"[creep] cache {modality}: {len(entities[modality])} unique, {len(todo)} to encode"
            f" (shard {cli.shard}/{cli.shards})",
            flush=True,
        )
        if not todo:
            continue
        tokenizer, model = load_encoder(modality, weights_dir, device)
        start = time.time()
        for i in range(0, len(todo), CHUNK):
            part = todo[i : i + CHUNK]
            store.write(part, encode(modality, tokenizer, model, part, device))
            done = i + len(part)
            print(
                f"[creep] cache {modality} {done}/{len(todo)} {done / (time.time() - start):.1f}/s",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
