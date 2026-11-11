#!/usr/bin/env sh
# Full BioProBench train splits (47k PQA / 28k ORD / 93k ERR / 107k GEN) live on HuggingFace.
# The repo ships a 500-row train sample per task plus the full test splits.
# Running this script replaces those 500-row train files.
set -e
cd "$(dirname "$0")/../data/bioprobench"
for t in PQA ORD ERR GEN; do
  curl -L -o "${t}_train.json" "https://huggingface.co/datasets/BioProBench/BioProBench/resolve/main/${t}_train.json"
done
