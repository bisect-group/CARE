# CREEP wrapper report

## 1. Method summary

CREEP (Contrastive Reaction-EnzymE Pretraining, CARE) aligns three modalities in a 256-dimensional space.
ProtT5-XL (mean over tokens), SciBERT (pooled output) and rxnfp (CLS token) each feed one linear head.
Training uses mined batches. One step takes one anchor EC number and 15 other EC numbers.
For each EC it draws a protein (random 50% cluster, then a random member), a reaction and the EC text.
The loss is the EBM-NCE contrastive loss over all six modality pairs (temperature 0.1).
All three backbones are fine-tuned with Adam, 40 epochs, batch size 16, fp16 autocast.
For retrieval, the reaction vector is compared by cosine similarity with the mean protein vector of each EC.
The ranked EC list is the output. The wrapper ranks reactions only (reaction to protein), as in the camera-ready figure.
The CARE paper PDF is not in the submodule. The settings come from `README.md` of CARE and the released code.

## 2. Original default settings

Source files are in `CARE/task2_baselines/`. The same table with the source of each value is `default_hparams.json`.
A test (`default_hparams.json` against the argparse defaults of `step_01_train_CREEP.py`) found no difference.

| Setting | Value | Source |
|---|---|---|
| Epochs | 40 | `CREEP/step_01_train_CREEP.py:192`, `README.md` |
| Batch size | 16 (1 anchor EC + 15 negative ECs) | `step_01:193` |
| Optimizer | Adam, weight decay 0, no scheduler | `step_01:213,292` |
| Learning rates | protein 1e-5, reaction 1e-5, text 1e-5 x 0.1 | `step_01:204-209` |
| Embedding size | 256 | `step_01:195` |
| Max length | 512 for protein, text, reaction | `step_01:201-203` |
| Loss | EBM_NCE, T 0.1, 1 negative, no normalize, 3 modalities | `step_01:210-221` |
| Generative loss weight | 0 | `step_01:218` |
| Backbones | ProtT5-XL half encoder, SciBERT, rxnfp | `step_01:247-258` |
| Precision | fp16 autocast + GradScaler | `step_01:116,133` |
| Best model | lowest mean training loss of an epoch | `step_01:179-183` |
| Extraction | batch 256, fp16 autocast | `step_02_extract_CREEP.py:23,35,54` |
| Retrieval | cosine to the mean protein vector of each EC, zero norm set to 1 | `step_02:226-250`, `downstream_retrieval.py` |
| Cluster ids | mmseqs easy-cluster, 50% identity, coverage 0.8, cov-mode 0 | `generate_datasets_splits/rawdata_processing.ipynb` |
| Seeds | 666, 777, 888 (original default 42) | Leak-CURBER protocol |

Paper and code agree where they were compared (README: 40 epochs, batch 16). The code wins elsewhere.

## 3. What the wrapper does

1. `common.discover_splits` finds every `train/val/test` folder with `rxn_smiles`, `sequence` and `ec_number`.
2. `common.build_tables` writes the CSV files the original classes read: reaction-EC pairs, protein-EC pairs with
   `clusterRes50`, EC text, the reference proteins and the test reactions. They are cached in `OUT/embeddings/tables_v2`.
3. `train.py` builds the model like `step_01` (same creation order, same seeding), then calls the original
   `CREEPDatasetMineBatch` and the original `train()` once per epoch. It saves `bestmodel.pth` (through the original
   `save_model` hook), `lastmodel.pth` and the resumable `checkpoint_last.pt` (model, Adam, all RNG states).
4. `evaluate.py` loads `bestmodel.pth`, calls the original `extract()` for reference proteins and test reactions,
   builds the mean vector per EC, and ranks with the code of `downstream_retrieval.py` and `rank_tabulate_similarities.py`.
5. `queue_pipeline.py` runs train and evaluate for all splits and seeds (`--gpus`, `--runs_per_gpu`, `--amp`).
   `aggregate_results.py` rebuilds `summary.csv` and `summary_by_seed.csv` from `predictions.parquet`.

`--amp` swaps the original fp16 autocast and GradScaler for bf16 autocast on Ampere or newer.
Without `--amp` the original precision is used (fp32 on CPU).

## 4. Files

Added, all in `CARE/CREEP/emulator_bench/`: `README.md`, `REPORT.md`, `commands.txt`, `default_hparams.json`,
`common.py`, `model_utils.py`, `metrics.py`, `train.py`, `evaluate.py`, `queue_pipeline.py`, `aggregate_results.py`,
`testing.py`, `smoke_test.py`, `equivalence_check.py`, `test_speedups.py`.
Original code is not changed. Nothing was removed.
The uncommitted edits to `generate_datasets_splits/blast_utils.py` and `task1_baselines/BLAST/blast_utils.py` are not mine and were left alone.

## 5. Bugs fixed

No bug blocked a run, so no original file was edited.

## 6. Deviations and why behaviour is unchanged

| Deviation | Why |
|---|---|
| Data come from EMULaToR parquet files, not CARE CSV files | Required by the task. The original classes read the converted CSV files unchanged. |
| Rows without a sequence or without a full four-level EC are dropped and counted (`run_summary.json`, `prepare.rows`) | CREEP needs a protein for each EC. Same rule as the CLIPZyme wrapper. |
| `clusterRes50` is computed on the unique sequences of train, val and test with the CARE mmseqs command | The EMULaToR files have no cluster ids. Train proteins only draw from clusters of train sequences. |
| EC text is built from the GO `ec2go` file with the notebook code | The EMULaToR files have no EC text. The file version is newer than the CARE copy. |
| Training uses the train file only; the val file is not used for selection | CREEP has no validation step. It picks the best model by training loss. |
| The reference has one protein per (50% cluster, EC) pair | The original clustered50 file is not available. See open issues. |
| One query per unique test reaction, all its ECs are true | Same as the CLIPZyme and Horizyn wrappers. |
| One `bestmodel.pth` with all weights instead of 8 component files | Spec layout. `step_02` loads the same tensors. |
| Only the top 50 ranks are saved | Metrics use k up to 50 only. |
| `--reference` defaults to `train`: the EC vectors use train proteins only. The original uses the proteins of all splits (`--reference all`) | User decision. `all` ranks ECs with val and test proteins, a test leak. CLIPZyme and Horizyn use train proteins only. `--reference all` still runs the original protocol. |

Pure speed-ups and memory options are in section 11. `num_workers` stays 0, because workers would change the order of the Python random draws.

## 7. Commands

See `commands.txt` (run from `baselines/CARE`). The main command is

```
export PY=/home/adhil/.conda/envs/creep/bin/python   # transformers 4.39.1, torch 2.2.0; see section 11
cd /home/adhil/github/EMULaToR/baselines/CARE
$PY CREEP/emulator_bench/queue_pipeline.py --gpus 0 1 2 --runs_per_gpu 1
$PY CREEP/emulator_bench/aggregate_results.py
```

## 8. Smoke test

`smoke_test.py` runs on CPU with tiny random ProtT5 and SciBERT models (real rxnfp), 64 train, 16 val and 16 test rows
from `with_protein_information/random_splits_grouped_rxn_key`, seed 666. It trains one epoch (6 ECs), evaluates,
resumes from `checkpoint_last.pt` for a second epoch, and evaluates with `--reference train`.
Result: all output files exist, the resumed log has 2 epochs, 16 test queries ranked. Passed (about 2 min).
`build_tables` was also run on the real `uniprot_time_splits` (209,569 unique sequences, 38,458 clusters, 52,224 reference proteins,
2,891 test reactions, 5,204 train ECs). The row drops are in that run summary: train 43,348 without sequence and 81,686 without full EC.
Also checked: `queue_pipeline.py --help` and `--dry_run` list 3 folders today (`random_splits_grouped_rxn_key`, `uniprot_time_splits`,
`enzyme_sequence_splits/threshold_0.4`); the `default_hparams.json` test against the argparse defaults; `ruff check`; `py_compile`.
Real weights and real batches ran on a GPU in section 11. Not tested: `evaluate.py` with real weights and `aggregate_results.py` on real runs.

## 9. Equivalence

Run A is the original `step_01_train_CREEP.py`, `step_02_extract_CREEP.py` and `downstream_retrieval.py`, run unchanged as scripts.
Run B is the wrapper. Both use the same tiny data (64/16/16 rows, seed 666, 3 epochs, 165 steps), CPU, deterministic algorithms, 8 threads.
The tiny ProtT5 and SciBERT models are random with fixed weights, because the real weights are missing. rxnfp and the tokenizers are real.
The check compares, in order: (1) 165 of 165 batch hashes equal; (2) initial weights after seeding equal;
(3) weights at the start of each epoch equal (optimizer, loss, batch order); (4) epoch train loss 3.51171, 1.94428, 1.74572 in both,
best epoch 3 in both, final weights equal; (5) the top-50 EC lists of all 16 test reactions equal, and the accuracy equal
(k=1: level 4 0.0, level 3 0.0, level 2 0.0, level 1 6.25).

**Verdict: EQUIVALENT** on this setup. The check does not cover the real weights, fp16 on GPU, or a full-size run.
The check was rerun in the `creep` env (transformers 4.39.1, torch 2.2.0, CPU) after the speed-up changes, with the same result: 165 of 165 batch hashes,
weights, losses, best epoch and top-50 lists equal (the wrapper trains through the `Prefetch` thread). Run A needs no patch there.
The check uses the wrapper with `--reference all`, to compare with the original `all_ECs` reference.
No run used mldb for this: its transformers 5.19 breaks the rxnfp tokenizer (section 11).

## 10. Open issues

Resolved by the user: `--reference` defaults to `train` (see section 6). Use `--reference all` for the original protocol.

1. Weights: ProtT5 and SciBERT are now on disk in `outputs/baselines/CREEP/pretrained` (2.4 GB and 0.44 GB). The local `Rostlab/prot_t5_xl_uniref50` (11 GB, fp32, encoder and decoder) is a different file and is not used.
2. Environment: use the `creep` conda env (python 3.10, torch 2.2.0+cu121, transformers 4.39.1). Never use mldb (section 11).
3. Cost: see section 11. One run takes about 250 to 290 h on a 24 GB A10 with the exact memory options. DECISION NEEDED: whether to run on A10 at all.
4. Model selection follows the original (lowest training loss). The val file plays no part, so `bestmodel.pth` is not "best on validation".
5. EC text uses the current GO `ec2go` file (2026-07 release). The CARE copy is older, so some descriptions can differ. Download it once (see `commands.txt`).
6. The one-protein-per-(cluster, EC) reference and the cluster computation over train, val and test sequences are my choices. The original `protein2EC_clustered50.csv` is not available.
7. Rows without sequence (3.5% in random splits) and rows with partial EC are dropped in train and test. Horizyn keeps partial EC rows in training.
8. Folders without a `sequence` column are skipped. The top-level `reaction_drfp_tanimoto_splits` has none, so DRFP runs only after the rebuild adds it under `with_protein_information/`.
   `enzyme_structure_splits` and new threshold folders are found automatically.
9. `--amp` (bf16) ran on an A10 for timing only (section 11). It is not faster than fp16 there. The evaluation (`evaluate.py`) was not run with real weights.
10. The CLIPZyme and Horizyn wrappers were being rewritten while I worked. `predictions.parquet` and `metrics.json` follow the current CLIPZyme layout. Check the Horizyn layout before one aggregator reads all three.
11. The CARE paper PDF is not in the submodule.

## 11. Speed-ups and memory

All numbers: A10 (22 GiB usable), real ProtT5, SciBERT and rxnfp weights, real batches of the `uniprot_time_splits` split (batch 16, seed 666),
original fp16 autocast with GradScaler, `creep` env. Step time is the mean over steps that update the weights.
Profiling scripts were in the scratchpad, not in the repo. The code is in `model_utils.py` and `train.py`.

**Environment.** transformers 5.19 (mldb) breaks the wrapper in three ways. The rxnfp tokenizer gives `[UNK]` for every atom (`C`, `O`, ...).
ProtT5 loads as fp16 instead of fp32. The random heads get a different initialisation, because `from_pretrained` uses the random stream in 4.39 and not in 5.19.
With transformers 4.39.1 and 5.19 the three backbones give the same outputs for the same token ids (max difference 3e-6 in fp32 on CPU).
`build_model` now loads ProtT5 with `torch_dtype=torch.float32` (the 4.39 default) and stops with an error if the rxnfp tokenizer gives `[UNK]`.

**Memory of the original setup.** ProtT5 has 1.208 B parameters, SciBERT 0.110 B, rxnfp 0.007 B. All three train in fp32 with Adam.
Weights 4.9 GiB, gradients 4.9 GiB and Adam moments 9.9 GiB are 19.7 GiB before any activation. All three models run in eval mode (no dropout).
Activations without checkpointing (measured with 1 to 2 ProtT5 layers, extrapolated): about 1.8 GiB per ProtT5 layer (43 GiB for 24 layers) and
about 7.4 GiB for SciBERT and rxnfp together. The original step needs about 70 GiB, so it fits an 80 GB GPU and nothing smaller.

| Option (same numbers) | Peak reserved, full model | Step time on A10 | Note |
|---|---|---|---|
| Original (no option) | about 70 GiB (extrapolated) | not run | needs 80 GB |
| `--checkpoint_activations all` only | does not fit (out of memory at 22 GiB, about 24 GiB needed) | not run | recompute each block in backward |
| `all` + `--offload_adam 0` (all moments on the CPU) | 16.4 GiB | 6.1 s | 2.6 s of the step is the Adam transfer (PCIe about 8 GB/s) |
| `all` + `--offload_adam 4.5` | 20.8 GiB | 4.9 s | 4.5 GiB of moments stay on the GPU |
| `all` + `--offload_adam 6.0` (also drops the autocast weight cache) | 21.0 GiB | 4.55 s | fits an exclusive A10 |

`--offload_adam` also turns off the autocast cache of fp16 weight copies. The values are the same, and it saves 1.7 GiB.
`--low_memory` in the launcher sets `--checkpoint_activations all --offload_adam 5` (5 GiB resident, a safer margin than 6) and `expandable_segments`.
One run needs the whole A10, so `--runs_per_gpu` stays 1.

**Exactness evidence.**
- Checkpointed blocks: gradients are bit-equal on CPU (bf16) and on the GPU (fp16) for T5 and BERT stacks (`test_speedups.py`).
- Offloaded Adam: the weights are bit-equal to `torch.optim.Adam` after 5 steps with GradScaler, for 0, 0.001 and 1 GB resident, also after a state-dict reload (`test_speedups.py`).
- End to end on the GPU, tiny backbones in eval mode, 3 epochs, deterministic mode: the final weights hash and the train log of the baseline, a baseline repeat,
  `--checkpoint_activations all`, `bert`, `--offload_adam 0`, and both together are all equal.
  The tiny models must be in eval mode for this. In train mode (dropout) the checkpointed run draws different dropout masks. The real backbones load in eval mode and the original never calls `train()`.
- Real weights, real batches, 24 layers: the mean loss over 14 steps is 2.473 in every variant (offload 0, 4.5, 6.0, with and without prefetch, no cache).
  Per-step losses of the first 8 steps match to 3 digits with and without checkpointing (2 ProtT5 layers).

**Data loading.** One step builds 48 tokenised sequences (about 0.04 s in the main process, 1% of the step). `Prefetch` builds the next two batches in a thread.
The batch order and the Python random draws stay the same (batch hashes equal in the equivalence check). It gives 0.04 s per step. A tokenisation cache would give no more, so there is none.
`num_workers` stays 0. rxnfp featurisation is the SmilesTokenizer regex, in the same 0.04 s.

**Rejected or not exact (not applied).**
- DECISION NEEDED: trim the SciBERT and rxnfp padding to the longest sequence of the batch (text 26 of 512 tokens on average, reaction 203). Step 4.55 to 4.24 s (7%). It is exact in math (the attention mask hides the padding, and SciBERT and rxnfp use the first token only), but the kernels get other shapes, so the fp16 results change in the last bits. The ProtT5 padding cannot be trimmed: the mean over all 512 positions includes the padded ones.
- DECISION NEEDED: `--amp` (bf16 on A10): 4.98 s against 4.94 s. No gain. It changes the numbers.
- DECISION NEEDED: 8-bit Adam, bf16 weights or Adam moments, or a smaller batch with gradient accumulation. They change the numbers and were not run. The loss uses the other samples of the batch as negatives (`cycle_index`), not BatchNorm, but a smaller batch changes the negatives. Only an exact split would keep them: encode in chunks, compute the loss on all 16, then backpropagate. That is checkpointing with more code.
- Adam on the CPU: not bit-equal to the GPU kernels, and the CPU host is shared.
- Fused or foreach choices of Adam: the stock Adam already uses foreach on the GPU.

**Cost per run (40 epochs, one step per EC, batch 16).** Steps per epoch: 5,204 (`uniprot_time_splits`), 5,767 (`random_splits_grouped_rxn_key`); the sequence split is about the same size.

| GPU | s/step | h/epoch | h per run | Basis |
|---|---|---|---|---|
| A10 24 GB, exact low-memory options | 4.6 | 6.6 to 7.4 | 263 to 295 | measured (4.55 s plus about 0.1 s for skipped-step average) |
| A100 80 GB, no options (original) | about 1 | about 1.4 | about 55 | estimate: A10 compute 3.4 s with recompute, 2.6 s without, divided by about 3 |
| Blackwell 32 GB, checkpointing, Adam on the GPU, no autocast cache | about 1.6 | about 2.4 | about 95 | estimate: 23 GiB needed; compute about 2.3 times the A10 |
| H100 80 GB (README) | about 0.6 | | 36 | README of CARE |

The A100 and Blackwell rows are estimates from the A10 numbers, not measurements. Nine runs on A10 need about 2,400 to 2,700 GPU hours.
Each run also needs about 50 min for the reference embedding of 52,000 proteins (estimate from the forward pass time) and writes about 30 GB.
Peak GPU memory of the best A10 setup is 21.0 GiB; host pinned memory is about 5 GiB.
