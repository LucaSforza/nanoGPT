#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --job-name=nanoGPT
#SBATCH --output=nanoGPT-%j.out
#SBATCH --error=nanoGPT-%j.err
#SBATCH --mem=16G

set -euo pipefail

SIF=~/nanoGPT.sif
IMAGE=docker://lucasforza/nanogpt:latest
REPO=~/nanoGPT

# Build Singularity image from Docker Hub (only if not exists)
if [ ! -f "$SIF" ]; then
    singularity build "$SIF" "$IMAGE"
fi

cd "$REPO"

# Prepare data
singularity exec --nv "$SIF" python data/shakespeare_char/prepare.py

# Train
singularity exec --nv "$SIF" python train.py config/train_shakespeare_char.py
