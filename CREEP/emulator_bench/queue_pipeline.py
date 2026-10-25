"""Launch train + evaluate for all splits and seeds, one GPU per run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from train_frozen import frozen_run_dir
from common import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUT_ROOT,
    DEFAULT_SEEDS,
    WRAPPER_DIR,
    Split,
    build_tables,
    discover_splits,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--weights_dir", type=Path, default=None, help="default OUT/pretrained")
    p.add_argument("--ec2go", type=Path, default=None, help="default OUT/reference/ec2go")
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="split groups or group/name keys; default all found",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--runs_per_gpu", type=int, default=1)
    p.add_argument(
        "--amp", action="store_true", help="bf16 autocast on Ampere or newer instead of fp16"
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="exact and bf16 speed-ups (see train.py --fast) with recomputed activations; not with --low_memory",
    )
    p.add_argument(
        "--low_memory",
        action="store_true",
        help="fit a 24 GB GPU with the same numbers: recompute activations, keep Adam moments on the CPU",
    )
    p.add_argument(
        "--adam_resident_gb",
        type=float,
        default=4.0,
        help="with --low_memory: Adam moments kept on the GPU",
    )
    p.add_argument(
        "--frozen",
        action="store_true",
        help="frozen ProtT5, SciBERT and rxnfp: cache their vectors once (sharded over --gpus), then train the heads; "
        "not with --amp, --fast or --low_memory",
    )
    p.add_argument("--reference", choices=["all", "train"], default="all", help="see evaluate.py")
    p.add_argument("--mmseqs", default="mmseqs")
    p.add_argument("--threads", type=int, default=8, help="mmseqs threads")
    p.add_argument("--dry_run", action="store_true", help="list the runs and stop")
    return p.parse_args(argv)


def run_dir(split: Split, cli: argparse.Namespace, seed: int) -> Path:
    return (frozen_run_dir if cli.frozen else Split.run_dir)(split, cli.out_root, seed)


def is_done(run_dir: Path, cli: argparse.Namespace) -> bool:
    """A run is done when its metrics were computed with the requested reference."""
    metrics = run_dir / "metrics.json"
    return metrics.exists() and json.loads(metrics.read_text()).get("reference") == cli.reference


def cache_encoders(cli: argparse.Namespace, ec2go: Path, splits: list[str]) -> None:
    """One cache_frozen.py process per GPU; each encodes its own shard of the missing vectors."""
    command = [
        sys.executable,
        str(WRAPPER_DIR / "cache_frozen.py"),
        "--dataset_root", str(cli.dataset_root),
        "--out_root", str(cli.out_root),
        "--ec2go", str(ec2go),
        "--reference", cli.reference,
        "--mmseqs", cli.mmseqs,
        "--shards", str(len(cli.gpus)),
        "--splits", *splits,
    ]  # fmt: skip
    if cli.weights_dir:
        command += ["--weights_dir", str(cli.weights_dir)]
    print(f"[creep] caching encoder vectors on GPUs {cli.gpus}", flush=True)
    cli.out_root.mkdir(parents=True, exist_ok=True)
    logs = [open(cli.out_root / f"cache_shard_{i}.log", "a") for i in range(len(cli.gpus))]
    processes = [
        subprocess.Popen(
            [*command, "--shard", str(i)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for i, (gpu, log) in enumerate(zip(cli.gpus, logs))
    ]
    if any(process.wait() for process in processes):
        raise SystemExit(f"[creep] cache failed, see {cli.out_root}/cache_shard_*.log")


def main(argv=None) -> int:
    cli = parse_args(argv)
    if cli.fast and cli.low_memory:
        raise SystemExit("--fast and --low_memory exclude each other")
    if cli.frozen and (cli.amp or cli.fast or cli.low_memory):
        raise SystemExit("--frozen excludes --amp, --fast and --low_memory")
    ec2go = cli.ec2go or cli.out_root / "reference" / "ec2go"
    splits = discover_splits(cli.dataset_root, cli.splits)
    jobs = [(s, seed) for s in splits for seed in cli.seeds]
    pending = [(s, seed) for s, seed in jobs if not is_done(run_dir(s, cli, seed), cli)]
    for s, seed in jobs:
        state = "todo" if (s, seed) in pending else "done"
        print(f"[creep] {state} {s.key} seed {seed} -> {run_dir(s, cli, seed)}", flush=True)
    if cli.dry_run or not pending:
        return 0

    for s in {s.key: s for s, _ in pending}.values():
        print(f"[creep] tables {s.key}", flush=True)
        build_tables(s, cli.out_root, ec2go, cli.mmseqs, cli.threads)

    if cli.frozen:
        cache_encoders(cli, ec2go, [s.key for s in {s.key: s for s, _ in pending}.values()])

    slots: Queue = Queue()
    for gpu in cli.gpus * cli.runs_per_gpu:
        slots.put(gpu)
    failed, lock = [], threading.Lock()

    def run(job):
        s, seed = job
        gpu = slots.get()
        try:
            folder = run_dir(s, cli, seed)
            folder.mkdir(parents=True, exist_ok=True)
            common = [
                "--split",
                s.key,
                "--seed",
                str(seed),
                "--dataset_root",
                str(cli.dataset_root),
                "--out_root",
                str(cli.out_root),
            ]
            if cli.weights_dir:
                common += ["--weights_dir", str(cli.weights_dir)]
            amp = ["--amp"] if cli.amp or cli.fast else []
            fast = ["--fast", "--checkpoint_activations", "all"] if cli.fast else []
            memory = (
                ["--checkpoint_activations", "all", "--offload_adam", str(cli.adam_resident_gb)]
                if cli.low_memory
                else []
            )
            if cli.frozen:
                steps = [
                    ["train_frozen.py", *common, "--ec2go", str(ec2go), "--mmseqs", cli.mmseqs],
                    ["evaluate.py", *common, "--reference", cli.reference, "--frozen"],
                ]
            else:
                steps = [
                    [
                        "train.py",
                        *common,
                        "--ec2go",
                        str(ec2go),
                        "--mmseqs",
                        cli.mmseqs,
                        *amp,
                        *fast,
                        *memory,
                    ],
                    ["evaluate.py", *common, "--reference", cli.reference, *amp],
                ]
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
            with open(folder / "log.txt", "a") as log:
                for step in steps:
                    code = subprocess.run(
                        [sys.executable, str(WRAPPER_DIR / step[0]), *step[1:]],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ).returncode
                    if code:
                        with lock:
                            failed.append((s.key, seed, step[0]))
                        print(
                            f"[creep] FAILED {step[0]} {s.key} seed {seed}, see {folder / 'log.txt'}",
                            flush=True,
                        )
                        return
            print(f"[creep] finished {s.key} seed {seed}", flush=True)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(cli.gpus) * cli.runs_per_gpu) as pool:
        list(pool.map(run, pending))
    print(f"[creep] {len(pending) - len(failed)} runs ok, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
