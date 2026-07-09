#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --job-name=nanoGPT-sample
#SBATCH --output=sample-%j.out
#SBATCH --error=sample-%j.err
#SBATCH --mem=8G

set -euo pipefail

SIF=./nanogpt.sif
REPO=~/nanoGPT
OUT_DIR=out-shakespeare-char

cd "$REPO"

singularity exec --nv --home "$HOME" --env PYTHONUNBUFFERED=1 "$SIF" \
    python sample.py --out_dir="$OUT_DIR" --num_samples=10 --max_new_tokens=500
