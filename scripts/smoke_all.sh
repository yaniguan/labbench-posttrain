#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec .venv/bin/python scripts/smoke_all.py "$@"
