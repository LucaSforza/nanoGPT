#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --job-name=nanoGPT
#SBATCH --output=nanoGPT-%j.out
#SBATCH --error=nanoGPT-%j.err
#SBATCH --mem=16G

set -euo pipefail

# --- Config ---
SIF=./nanogpt.sif
IMAGE=docker://softdream1/nanogpt:latest
REPO=~/nanoGPT
WANDB=1   # set to 0 to disable wandb logging

# --- Build Singularity image (cached) ---
if [ ! -f "$SIF" ]; then
    singularity build "$SIF" "$IMAGE"
fi

cd "$REPO"

WANDB_FLAG=""
if [ "$WANDB" = 1 ]; then
    WANDB_FLAG="--wandb_log=True"
fi

# Prepare data
singularity exec --nv "$SIF" python data/shakespeare_char/prepare.py

# Train (--compile=False because container lacks gcc for Triton)
singularity exec --nv --home "$HOME" "$SIF" \
    python train.py config/train_shakespeare_char.py --compile=False $WANDB_FLAG
