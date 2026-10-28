# CREEP EMULaToR wrapper

This wrapper trains CREEP (CARE, Contrastive Reaction-EnzymE Pretraining) with its original settings
on the EMULaToR enzyme retrieval splits. It runs seeds 666, 777 and 888 on every split folder.
It does no tuning. `REPORT.md` has the settings, checks and open issues. `commands.txt` has the commands.
All commands run from the submodule root `baselines/CARE`.

## How it works

- `common.py` finds the split folders and builds the CSV tables that the original CREEP classes read.
- `train.py` calls the original `train()` loop of `task2_baselines/CREEP/step_01_train_CREEP.py`.
- `evaluate.py` calls the original `extract()` of `step_02_extract_CREEP.py` and ranks ECs like
  `downstream_retrieval.py` and `rank_tabulate_similarities.py`.
- `queue_pipeline.py` runs all splits and seeds. `aggregate_results.py` rebuilds the summaries from the saved predictions.
- Original CREEP code is not changed.

## Data

- The wrapper reads every folder with `train.parquet`, `val.parquet` and `test.parquet` that has the columns
  `rxn_smiles`, `sequence` and `ec_number`. It never hardcodes a split name or a threshold.
  Today this means the folders under `with_protein_information/`.
- A flat folder gets the split name `all`. A `threshold_*` folder keeps its name.
- Only rows with a reaction, a sequence and a full four-level EC number are used. The run summary counts the dropped rows.
- Training uses the train file only. CREEP has no validation step, so the val file plays no part in training.
- Protein clusters (`clusterRes50`) come from `mmseqs easy-cluster` at 50% identity, as in CARE. They are cached.
- EC descriptions (the text modality) are built from the GO `ec2go` file, as in the CARE notebook.
  Download it once (see `commands.txt`).

## Output

Everything goes to `<EMULaToR>/outputs/baselines/CREEP` (`--out_root`). The dataset folder is read only.

- `enzyme_retrieval/<split_group>/<split_name>/seed_<s>/`:
  `bestmodel.pth`, `lastmodel.pth`, `checkpoint_last.pt`, `train_log.csv`, `train_summary.json`,
  `predictions.parquet`, `metrics.json`, `run_summary.json`, `log.txt`
- `embeddings/mmseqs50_v1`, `embeddings/tables_v2`: cached clusters and tables
- `enzyme_retrieval/summary.csv`, `summary_by_seed.csv`: written by `aggregate_results.py`
- `pretrained/`: ProtT5 and SciBERT download folder (`--weights_dir`)

`predictions.parquet` and `metrics.json` have the same columns and keys as the CLIPZyme wrapper:
`rxn_smiles`, `true_ecs`, `ranked_ecs` (top 50), `ranked_scores`, and `accuracy` (EC levels 4 to 1 for
k in 1, 3, 5, 10, 20, 30, 40, 50, in percent). Each test reaction is ranked against the candidate EC numbers.
A reaction with several ECs counts each EC with equal weight.

The EC vectors come from the mean ProtT5-CREEP vector of the reference proteins of each EC.
The candidate pool is all split proteins (train, val and test) by design, for every retrieval baseline. This is `--reference all`, the default.
`--reference train` keeps train proteins only.

## Launcher

`queue_pipeline.py --gpus 0 1 2 --runs_per_gpu N [--amp] [--low_memory] [--reference all|train]`. Each run gets one GPU through `CUDA_VISIBLE_DEVICES`.
Finished runs (with a `metrics.json` that has the requested `--reference`) are skipped. A run with `checkpoint_last.pt` resumes.
`--amp` uses bf16 on Ampere or newer. The default keeps the original fp16 autocast with GradScaler.
`--low_memory` fits a 24 GB GPU with the same numbers. It recomputes activations and keeps most Adam moments in CPU memory.
It is slower than the original on a big GPU. The original setup needs about 70 GiB of GPU memory.
Use the `creep` conda env (transformers 4.39.1, torch 2.2.0). transformers 5 breaks the rxnfp tokenizer.

## Frozen encoders

ProtT5, SciBERT and rxnfp stay frozen. Only the three heads train (protein 1024 to 256, text 768 to 256, reaction 256 to 256).
The two facilitators have `alpha_generative = 0`, so they do nothing and are skipped.

- `cache_frozen.py` encodes every unique sequence, reaction SMILES and EC text once and stores fp32 vectors in
  `embeddings/frozen_v1/{protein,reaction,text}`. All splits and seeds share them. A `fingerprint.json` (model, revision, recipe)
  per modality stops a stale cache from being read. A rerun encodes only the missing vectors. `--shard i --shards n` splits the work over GPUs.
  ProtT5 uses the exact CREEP recipe (space-joined residues, padded to 512, mean over all 512 positions, fp16 autocast, SDPA attention).
  SciBERT and rxnfp batches are sorted by length and cut to the longest sequence.
- `train_frozen.py` keeps the cached vectors on the GPU. The sampler subclasses the original `CREEPDatasetMineBatch` and makes the same draws
  (batch 16, one step per full EC per epoch, 40 epochs). Loss: the original `do_CL` over six pairs (accuracy kept as a tensor, no per-step sync).
  Adam has the original learning rates (text x0.1). The heads run under fp16 autocast with a GradScaler (new each epoch), as in the original loop; `evaluate.py --frozen` projects under fp16 autocast like the original `extract()`. Output: `enzyme_retrieval_frozen/<split>/seed_<s>/`
  (`bestmodel.pth` and `lastmodel.pth` hold the heads only; `checkpoint_last.pt` resumes).
- `evaluate.py --frozen` projects cached reference-protein and test-reaction vectors through the heads, then ranks as before.
- `queue_pipeline.py --frozen --gpus ... --runs_per_gpu N` runs the cache (one shard per GPU), then training and evaluation of every run.
- `aggregate_results.py --frozen` summarizes `enzyme_retrieval_frozen`.
- `equivalence_frozen.py` compares this path with the original code whose backbones are frozen.
  Tests: `test_cache_frozen.py`, `test_train_frozen.py`.

## Weights and size

ProtT5 (`Rostlab/prot_t5_xl_half_uniref50-enc`) and SciBERT (`allenai/scibert_scivocab_uncased`) load from the Hugging Face
hub into `--weights_dir`. rxnfp is in the submodule (`CREEP/data/pretrained_rxnfp`).
The original run takes about 36 h on one H100 with 80 GB (40 epochs). One A10 run with `--low_memory` takes about 250 to 290 h (`REPORT.md`, section 11).
Each run writes about 30 GB of checkpoints.
`--epochs` and `--limit_ecs` exist for smoke tests only.
`train.py` also takes `--checkpoint_activations all|bert` and `--offload_adam [RESIDENT_GB]`.

`--fast` (train.py, and queue_pipeline.py with `--checkpoint_activations all` added) uses the exact and bf16 speed-ups:
bf16 autocast, SDPA attention and a feed-forward without the fp32 cast in ProtT5, SciBERT and rxnfp inputs cut to the batch's longest sequence
(ProtT5 keeps 512 because its output is a mean over all positions), and one stock Adam per parameter stepped in backward.
It needs Ampere or newer, and excludes `--offload_adam` and `--low_memory`. `checkpoint_last.pt` keeps the per-parameter Adam states, so resume works.
The numbers differ from the fp16 path by bf16 rounding. Tests: `test_fast_path.py`.

## Verify equivalence

```
emulator_bench/verify_equivalence.sh          # CPU, about 2 minutes, no data, weights or downloads needed
emulator_bench/verify_equivalence.sh --gpu    # also the fp16, bf16 and offload tests, needs a CUDA device
emulator_bench/verify_equivalence.sh --negative-control  # GPU tests must FAIL, needs a CUDA device
```

Run it in the `creep` conda env (set `PYTHON` to its interpreter). The script fails if any test is skipped in CPU mode.
It uses the synthetic split in `emulator_bench/tests/fixtures` (made by `make_fixture.py`). ProtT5 and SciBERT are tiny random
models. A small sentencepiece model replaces the ProtT5 tokenizer. rxnfp is the real model of this repository.
mmseqs is replaced by one cluster per sequence.

- `test_equivalence_frozen.py` runs the original loop with frozen backbones and the frozen-encoder path (cache, head training,
  ranking) on the fixture in fp32 on CPU. The sampler draws must be identical. Step losses must agree within 1e-6 relative.
  The final heads must agree within 1e-5 relative norm. The top-10 EC lists and the EC level accuracy must be identical.
- `test_speedups.py`, `test_fast_path.py`, `test_cache_frozen.py` and `test_train_frozen.py` compare the exact speed-ups
  (checkpointed blocks, SDPA attention, feed-forward without the cast, padding trim, backward Adam, cached vectors) with the original code.
- The `gpu` test in `test_equivalence_frozen.py` compares the production path (frozen encoders, heads under fp16 autocast with GradScaler, as in all recorded runs) with the original fp16 loop. A noise control reruns the production path with initial heads perturbed by 2**-11 (relative, one fp16 rounding). The per-step loss gap and the final head gap (mean absolute difference) between production and the original loop must stay within 10 times the gap the perturbation causes. Step 0 is judged like every other step: its gap must be at most 10 times its floor.
  The EC level accuracy gap at k=1 and k=10, per level, must stay within the larger of 10 times its floor (production against the perturbed run) and one test reaction. The fixture has rows longer than the 512 token limit, so truncation and the padding trim are covered. The test prints the measured ratio (run with `-s`).
  `--negative-control` multiplies the initial heads of the production run by 1.05. Every GPU test must then fail; the script exits 0 only if so.

Not exact: fp16 runs (rounding differs from fp32), and bf16 (`--amp`, not used by any recorded run) and the offloaded Adam on GPU (tested with `--gpu`).
`equivalence_frozen.py` prints the same comparison for real weights and real data on a GPU. It sets no tolerance.
