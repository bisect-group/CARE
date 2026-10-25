"""Rebuild the summary CSVs from the saved predictions of every finished run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import DEFAULT_OUT_ROOT, TASK, atomic_write_text
from metrics import LEVELS, ec_level_accuracy
from train_frozen import FROZEN_TASK


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--frozen", action="store_true", help="summarize the frozen-encoder runs")
    cli = p.parse_args()
    root = cli.out_root / (FROZEN_TASK if cli.frozen else TASK)
    rows = []
    for path in sorted(root.glob("**/seed_*/predictions.parquet")):
        group_name, seed = (
            path.parent.parent.relative_to(root),
            int(path.parent.name.split("_")[1]),
        )
        predictions = pd.read_parquet(path)
        metrics = ec_level_accuracy(predictions)
        saved = path.parent / "metrics.json"
        if saved.exists() and json.loads(saved.read_text())["accuracy"] != json.loads(
            json.dumps(metrics)
        ):
            print(f"[creep] metrics.json differs from the predictions: {path.parent}", flush=True)
        for k, levels in metrics.items():
            rows.append(
                {
                    "split_group": "/".join(group_name.parts[:-1]),
                    "split_name": group_name.name,
                    "seed": seed,
                    "k": int(k.removeprefix("k=")),
                    "queries": len(predictions),
                    **levels,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No predictions.parquet under {root}")
    by_seed = pd.DataFrame(rows)
    keys = ["split_group", "split_name", "k"]
    summary = by_seed.groupby(keys)[[f"level_{level}" for level in LEVELS]].agg(["mean", "std"])
    summary.columns = [f"{level}_{stat}" for level, stat in summary.columns]
    summary["seeds"] = by_seed.groupby(keys)["seed"].nunique()
    atomic_write_text(root / "summary_by_seed.csv", by_seed.to_csv(index=False))
    atomic_write_text(root / "summary.csv", summary.reset_index().to_csv(index=False))
    print(f"[creep] wrote {root / 'summary.csv'} ({len(by_seed)} rows)", flush=True)


if __name__ == "__main__":
    main()
