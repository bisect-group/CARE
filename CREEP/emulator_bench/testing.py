"""Test helpers: tiny random backbones and a tiny dataset. Not used by real runs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
import transformers
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

import model_utils

FIXTURES = Path(__file__).parent / "tests" / "fixtures"
PROTT5_SPIECE = FIXTURES / "spiece.model"  # small unigram model with the T5 special tokens and 100 extra ids
TINY_TEXT_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    *"abcdefghijklmnopqrstuvwxyz0123456789-.,;()[]+ ",
    *[f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz0123456789-.,;()[]+"],
]


def hash_tensors(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def hash_state(state: dict) -> str:
    return hash_tensors(*[state[k].float() for k in sorted(state)])


@contextlib.contextmanager
def tiny_backbones(vocab_dir: Path):
    """Replace ProtT5 and SciBERT by small random models with fixed weights. rxnfp stays real."""
    vocab = Path(vocab_dir) / "vocab.txt"
    vocab.write_text("\n".join(TINY_TEXT_VOCAB) + "\n")

    def seeded(model):
        with torch.random.fork_rng():
            torch.manual_seed(0)
            return model().eval()  # from_pretrained returns eval mode

    def protein_model(*a, **k):
        config = T5Config(
            vocab_size=128, d_model=1024, d_ff=64, d_kv=16, num_layers=1, num_heads=2
        )
        return seeded(lambda: T5EncoderModel(config))

    def text_model(*a, **k):
        config = BertConfig(
            vocab_size=len(TINY_TEXT_VOCAB),
            hidden_size=768,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
        return seeded(lambda: BertModel(config))

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                transformers.T5Tokenizer,
                "from_pretrained",
                lambda *a, **k: T5Tokenizer(str(PROTT5_SPIECE), do_lower_case=False),
            )
        )
        stack.enter_context(
            mock.patch.object(transformers.T5EncoderModel, "from_pretrained", protein_model)
        )
        stack.enter_context(
            mock.patch.object(
                transformers.AutoTokenizer,
                "from_pretrained",
                lambda *a, **k: BertTokenizer(str(vocab)),
            )
        )
        stack.enter_context(
            mock.patch.object(transformers.AutoModel, "from_pretrained", text_model)
        )
        stack.enter_context(
            mock.patch.object(model_utils, "T5Tokenizer", transformers.T5Tokenizer)
        )
        yield


def make_tiny_dataset(source_split: Path, root: Path, sizes=(64, 16, 16), seed: int = 0) -> Path:
    """Copy a few rows of one real split (with sequences) into root/<split>. Returns the dataset root."""
    target = Path(root) / "with_protein_information" / "random_splits_grouped_rxn_key"
    target.mkdir(parents=True, exist_ok=True)
    for name, n in zip(("train", "val", "test"), sizes):
        frame = pd.read_parquet(
            source_split / f"{name}.parquet", columns=["rxn_smiles", "sequence", "ec_number"]
        )
        frame = frame.dropna()
        frame = frame[frame.ec_number.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")]
        frame.sample(n, random_state=seed).to_parquet(target / f"{name}.parquet", index=False)
    return Path(root)


def tmpdir(prefix: str, base: Path) -> Path:
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def dump(path: Path, data: object) -> None:
    Path(path).write_text(json.dumps(data, indent=2) + "\n")
