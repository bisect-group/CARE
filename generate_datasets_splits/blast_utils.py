import os
import pathlib
import tempfile
from typing import List, Union

import pandas as pd


def download_diamond() -> str:
    root = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "scripts/install_external_tools.sh").is_file())
    diamond = os.environ.get("DIAMOND_BIN", str(root / "external/diamond-2.1.9/diamond"))
    if not os.path.isfile(diamond):
        raise FileNotFoundError(f"Missing DIAMOND executable: {diamond}. Run: bash scripts/install_external_tools.sh")
    return diamond


def seqs_to_fasta(seqs: List[str], fname: str):
    with open(fname, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">{i}\n{seq}\n\n")


def make_diamond_db(seqs_or_fasta: Union[str, List[str]], diamond_fname: str) -> None:
    fasta_fname = seqs_or_fasta
    os.system(f"{download_diamond()} makedb --in {fasta_fname} --db {diamond_fname} --ignore-warnings")  # nosec

def diamond_alignment(seqs_or_fasta: Union[str, List[str]], ref_db: str) -> pd.DataFrame:
    temp_dir = tempfile.mkdtemp()
    fasta = seqs_or_fasta

    diamond = download_diamond()
    output = os.path.join(temp_dir, "diamond.tsv")
    outcols = "qseqid scovhsp pident sseqid"
    os.system(f"{diamond} blastp -k 1 --query {fasta} --db {ref_db} --out {output} --outfmt 6 {outcols} --quiet")  # nosec
    
    if not os.path.exists(output):
        df = pd.DataFrame(columns=["sequence", "aln_coverage", "max_id", "ref_entry_id"])
    else:
        df = pd.read_csv(output, sep="\t", header=None, index_col=0, names=["aln_coverage", "max_id", "ref_entry_id"])
    
    df["max_id"] *= df["aln_coverage"] / 100
    return df
