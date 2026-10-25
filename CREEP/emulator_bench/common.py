"""Paths, split discovery and cached training tables for the CREEP wrapper."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

WRAPPER_DIR = Path(__file__).resolve().parent
CREEP_ROOT = WRAPPER_DIR.parent
CARE_ROOT = CREEP_ROOT.parent
ORIGINAL_DIR = CARE_ROOT / "task2_baselines" / "CREEP"
EMULATOR_ROOT = CARE_ROOT.parents[1]

TASK = "enzyme_retrieval"
DEFAULT_DATASET_ROOT = (
    EMULATOR_ROOT / "data" / "processed" / "datasets" / "enzyme_retrieval_dataset"
)
DEFAULT_OUT_ROOT = EMULATOR_ROOT / "outputs" / "baselines" / "CREEP"
DEFAULT_SEEDS = (666, 777, 888)

SPLIT_FILES = ("train", "val", "test")
COLUMNS = ["rxn_smiles", "sequence", "ec_number"]
FULL_EC = r"\d+\.\d+\.\d+\.\d+"
TABLES_VERSION = 2
CLUSTER_VERSION = 1


def hparams(section: str) -> dict:
    """Original defaults of one section of default_hparams.json."""
    data = json.loads((WRAPPER_DIR / "default_hparams.json").read_text())
    return {k: v["value"] for k, v in data[section].items()}


def train_args(**overrides) -> Namespace:
    """Namespace with the argparse defaults of step_01_train_CREEP.py."""
    values = hparams("train")
    values.update(overrides)
    return Namespace(**values)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def write_json(path: Path, data: object) -> None:
    atomic_write_text(Path(path), json.dumps(data, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Split:
    group: str
    name: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.group}/{self.name}"

    def run_dir(self, out_root: Path, seed: int) -> Path:
        return Path(out_root) / TASK / self.group / self.name / f"seed_{seed}"

    def tables_dir(self, out_root: Path) -> Path:
        return Path(out_root) / "embeddings" / f"tables_v{TABLES_VERSION}" / self.group / self.name


def discover_splits(dataset_root: Path, selected: list[str] | None = None) -> list[Split]:
    """Find every train/val/test folder that has the columns CREEP needs."""
    root = Path(dataset_root)
    splits: dict[str, Split] = {}
    for train in sorted(root.rglob("train.parquet")):
        folder = train.parent
        if not all((folder / f"{s}.parquet").exists() for s in SPLIT_FILES):
            continue
        if not set(COLUMNS) <= set(pq.read_schema(train).names):
            continue
        parts = list(folder.relative_to(root).parts)
        if parts and parts[0] == "with_protein_information":
            parts = parts[1:]
        if folder.name.startswith("threshold_"):
            split = Split("/".join(parts[:-1]), folder.name, folder)
        else:
            split = Split("/".join(parts), "all", folder)
        if split.key in splits:
            raise ValueError(
                f"Two folders map to {split.key}: {splits[split.key].path} and {folder}"
            )
        splits[split.key] = split
    if selected:
        splits = {k: s for k, s in splits.items() if k in selected or s.group in selected}
    return list(splits.values())


def load_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    """Rows with a reaction, a protein sequence and a full four-level EC number."""
    frame = pd.read_parquet(path, columns=COLUMNS)
    counts = {"raw_rows": len(frame)}
    frame = frame.dropna()
    frame = frame[frame.rxn_smiles.str.strip().ne("") & frame.sequence.str.strip().ne("")]
    counts["dropped_missing_value_rows"] = counts["raw_rows"] - len(frame)
    frame = frame.assign(ec_number=frame.ec_number.str.strip())
    full = frame.ec_number.str.fullmatch(FULL_EC)
    counts["dropped_non_full_ec_rows"] = int((~full).sum())
    frame = frame[full].reset_index(drop=True)
    counts["kept_rows"] = len(frame)
    return frame, counts


def ec_descriptions(ec2go: Path, ecs: list[str]) -> pd.DataFrame:
    """EC number to text, built like rawdata_processing.ipynb (text2EC.csv)."""
    ec2desc = {}
    for line in Path(ec2go).read_text().splitlines():
        if not line.startswith("EC:"):
            continue
        line = line.strip().split(">")
        ec2desc[line[0][3:-1]] = line[1].split(";")[0][4:-1]

    def describe(ec: str) -> str:
        dashes = ec.count("-")
        levels = [".".join(ec.split(".")[:n]) + "." + ".".join(["-"] * (4 - n)) for n in (1, 2, 3)]
        parts = [
            ec2desc.get(level, "") if dashes < 3 - i else "" for i, level in enumerate(levels)
        ]
        parts.append(ec2desc.get(ec, ""))
        description = "; ".join(parts)
        for _ in range(3):
            description = description.replace(" ; ", " ")
        description = description.replace(" activity", "")
        if description[0] == ";":
            description = description[2:]
        if description[-2:] == "; ":
            description = description[:-2]
        return description

    return pd.DataFrame({"EC number": ecs, "Text": [describe(ec) for ec in ecs]})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mmseqs_clusters(
    sequences: list[str], out_root: Path, mmseqs: str, threads: int
) -> dict[str, int]:
    """Sequence to cluster id, from mmseqs easy-cluster at 50% identity (as in CARE clusterRes50)."""
    params = hparams("mmseqs50")
    options = [
        "--min-seq-id",
        str(params["min_seq_id"]),
        "-c",
        str(params["coverage"]),
        "--cov-mode",
        str(params["cov_mode"]),
    ]
    digest = hashlib.sha256("\n".join([*options, *sequences]).encode()).hexdigest()[:20]
    cache = Path(out_root) / "embeddings" / f"mmseqs50_v{CLUSTER_VERSION}" / digest
    result = cache / "cluster.tsv"
    if not result.exists():
        cache.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=cache, prefix=".work_"))
        (work / "seqs.fasta").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(sequences)))
        subprocess.run(
            [
                mmseqs,
                "easy-cluster",
                work / "seqs.fasta",
                work / "res",
                work / "tmp",
                *options,
                "--threads",
                str(threads),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        os.replace(work / "res_cluster.tsv", result)
        shutil.rmtree(work)
    pairs = pd.read_csv(result, sep="\t", header=None, names=["rep", "member"])
    rep_of = dict(zip(pairs.member, pairs.rep))
    return {s: rep_of[i] for i, s in enumerate(sequences)}


def build_tables(
    split: Split, out_root: Path, ec2go: Path, mmseqs: str = "mmseqs", threads: int = 8
) -> Path:
    """Write the CSV tables that the original CREEP classes read; reuse them when the inputs match."""
    target = split.tables_dir(out_root)
    fingerprint = {
        "version": TABLES_VERSION,
        "files": {
            s: [
                (split.path / f"{s}.parquet").stat().st_size,
                (split.path / f"{s}.parquet").stat().st_mtime_ns,
            ]
            for s in SPLIT_FILES
        },
        "ec2go_sha256": sha256_file(ec2go),
        "mmseqs50": [CLUSTER_VERSION, *hparams("mmseqs50").values()],
    }
    stamp = target / "fingerprint.json"
    if stamp.exists() and json.loads(stamp.read_text()) == fingerprint:
        return target

    frames, counts = {}, {}
    for name in SPLIT_FILES:
        frames[name], counts[name] = load_rows(split.path / f"{name}.parquet")
    pairs = pd.concat(frames.values())[["sequence", "ec_number"]].drop_duplicates()
    sequences = sorted(pairs.sequence.unique())
    cluster_of = mmseqs_clusters(sequences, out_root, mmseqs, threads)

    train = frames["train"]
    reactions = train[["rxn_smiles", "ec_number"]].drop_duplicates()
    proteins = train[["sequence", "ec_number"]].drop_duplicates()
    proteins = proteins.assign(clusterRes50=proteins.sequence.map(cluster_of))

    def representatives(pair_table: pd.DataFrame) -> pd.DataFrame:
        """One protein per 50% cluster and EC number, like the clustered50 reference of CARE."""
        table = pair_table.assign(cluster=pair_table.sequence.map(cluster_of))
        table = table.groupby(["cluster", "ec_number"]).sample(1, random_state=42)
        return table.sort_values(["ec_number", "cluster"])[["sequence", "ec_number"]]

    references = {"all": representatives(pairs), "train": representatives(proteins)}
    queries = frames["test"].groupby("rxn_smiles").ec_number.agg(lambda x: sorted(set(x)))

    tmp = Path(tempfile.mkdtemp(dir=_ensure(target.parent), prefix=f".{target.name}."))
    reactions.rename(columns={"rxn_smiles": "Reaction", "ec_number": "EC number"}).to_csv(
        tmp / "reaction_train.csv", index=False
    )
    proteins.rename(columns={"sequence": "Sequence", "ec_number": "EC number"}).to_csv(
        tmp / "protein2EC.csv", index=False
    )
    ec_descriptions(ec2go, sorted(reactions.ec_number.unique())).to_csv(
        tmp / "text2EC.csv", index=False
    )
    for name, reference in references.items():
        reference.rename(columns={"sequence": "Sequence", "ec_number": "EC number"}).to_csv(
            tmp / f"reference_{name}.csv", index=False
        )
    queries.apply(";".join).rename_axis("Reaction").reset_index().rename(
        columns={"ec_number": "EC number"}
    ).to_csv(tmp / "test_queries.csv", index=False)
    summary = {
        "rows": counts,
        "train_reaction_ec_pairs": len(reactions),
        "train_protein_ec_pairs": len(proteins),
        "train_ecs": int(reactions.ec_number.nunique()),
        "reference_proteins": {k: len(v) for k, v in references.items()},
        "reference_ecs": {k: int(v.ec_number.nunique()) for k, v in references.items()},
        "test_query_reactions": len(queries),
        "unique_sequences": len(sequences),
        "mmseqs50_clusters": len(set(cluster_of.values())),
    }
    write_json(tmp / "summary.json", summary)
    write_json(tmp / "fingerprint.json", fingerprint)
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)
    return target


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
