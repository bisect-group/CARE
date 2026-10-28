#!/usr/bin/env bash
# Checks that the CREEP wrapper (emulator_bench) reproduces the original CREEP code path.
# Expected environment: conda env "creep" (transformers 4.39.1, torch 2.2.0). This script does not create it.
# Usage: emulator_bench/verify_equivalence.sh [--gpu | --negative-control]
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" HF_HUB_OFFLINE=1
echo "Expected environment: conda env creep (python: $PYTHON)"

PYTEST=("$PYTHON" -m pytest -q -rs -p no:cacheprovider --color=no)
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if [[ "${1:-}" == "--negative-control" ]]; then
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || {
    echo "FAIL: the negative control needs a CUDA device." >&2
    exit 1
  }
  echo "Negative control: the production run starts from heads times 1.05. Every GPU test must FAIL."
  if EQUIV_NEGATIVE_CONTROL=1 "${PYTEST[@]}" -s -m gpu . | tee "$log"; then
    echo "FAIL: a GPU test passed with the negative control (the bound is vacuous)." >&2
    exit 1
  fi
  grep -q "ratio=" "$log" || { echo "FAIL: the GPU test crashed before it measured a ratio." >&2; exit 1; }
  grep "ratio=" "$log"
  echo "Negative control failed as expected."
  exit 0
fi

echo "CPU mode"
CUDA_VISIBLE_DEVICES="" "${PYTEST[@]}" -m "not gpu" . | tee "$log"
if grep -Eq "[0-9]+ skipped" "$log"; then
  echo "FAIL: a test was skipped in CPU mode" >&2
  exit 1
fi

if [[ "${1:-}" == "--gpu" ]]; then
  echo "GPU mode (production precision)"
  "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)" || {
    echo "No CUDA device is visible: the GPU tests are skipped." >&2
    exit 0
  }
  "${PYTEST[@]}" -s -m gpu . | tee "$log"
  if grep -Eq "[0-9]+ skipped" "$log"; then
    echo "FAIL: a GPU test was skipped" >&2
    exit 1
  fi
fi
echo "Equivalence checks passed."
