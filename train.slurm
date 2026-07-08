#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=30:00
#SBATCH --job-name=nanoGPT
#SBATCH --output=nanoGPT-%j.out
#SBATCH --error=nanoGPT-%j.err
#SBATCH --mem=16G

set -euo pipefail

cd /home/sforza_2050030/nanoGPT

# Install uv if not available
if ! command -v uv &>/dev/null; then
    pip install --quiet uv
fi

# Install all dependencies from pyproject.toml
uv sync

# Prepare Shakespeare character-level dataset
uv run python data/shakespeare_char/prepare.py

# Train
uv run python train.py config/train_shakespeare_char.py
