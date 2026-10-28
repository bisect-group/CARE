"""Writes the synthetic fixture: a split with sequences, an ec2go table and a small ProtT5-style tokenizer."""

import random
import tempfile
from pathlib import Path

import pandas as pd
import sentencepiece as spm

HERE = Path(__file__).parent
rng = random.Random(0)
letters = "ACDEFGHIKLMNPQRSTVWY"
ecs = [f"1.{a}.1.{b}" for a in (1, 2, 3, 4) for b in range(1, 7)]
molecules = ["C" * n for n in range(1, 9)]


def rows(n_per_ec: int) -> pd.DataFrame:
    out = []
    for ec in ecs:
        for _ in range(n_per_ec):
            left, right = rng.sample(molecules, 2)
            out.append(
                {
                    "rxn_smiles": f"{left}O>>{right}={rng.choice(['O', 'N'])}.{rng.choice(molecules)}",
                    "sequence": "".join(rng.choices(letters, k=rng.randint(20, 40))),
                    "ec_number": ec,
                }
            )
    return pd.DataFrame(out).drop_duplicates()


split = HERE / "dataset/with_protein_information/random_splits_grouped_rxn_key"
for name, n in (("train", 4), ("val", 1), ("test", 2)):
    rows(n).to_parquet(split / f"{name}.parquet", index=False)

with open(HERE / "ec2go", "w") as handle:
    for ec in ecs:
        parts = ec.split(".")
        for level in range(1, 4):
            label = ".".join(parts[:level] + ["-"] * (4 - level))
            handle.write(f"EC:{label} > GO:class {label} activity ; GO:0000000\n")
        handle.write(f"EC:{ec} > GO:enzyme {ec} activity ; GO:0000000\n")

with tempfile.NamedTemporaryFile("w", suffix=".txt") as corpus:
    corpus.write("\n".join(" ".join(letters) for _ in range(5)) + "\n")
    corpus.flush()
    spm.SentencePieceTrainer.train(
        input=corpus.name, model_prefix=str(HERE / "spiece"), vocab_size=24, model_type="unigram",
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
    )  # fmt: skip
(HERE / "spiece.vocab").unlink()
