#!/bin/bash
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --time=2:00
#SBATCH --signal=USR1@60
#SBATCH --job-name=nanoGPT
#SBATCH --output=nanoGPT-%j.out
#SBATCH --error=nanoGPT-%j.err
#SBATCH --mem=16G

set -euo pipefail

# --- Auto-requeue on time limit ---
_requeue() {
    echo "REQUEUE: Time limit approaching, saving checkpoint and requeueing..."
    scontrol requeue "$SLURM_JOB_ID"
}
trap _requeue USR1

# --- Config ---
SIF=./nanogpt.sif
IMAGE=docker://softdream1/nanogpt:latest
REPO=~/nanoGPT
OUT_DIR=out-shakespeare-char
WANDB=1

# --- Build Singularity image (cached) ---
if [ ! -f "$SIF" ]; then
    singularity build "$SIF" "$IMAGE"
fi

cd "$REPO"

# Auto-resume if checkpoint exists
INIT="--init_from=scratch"
if [ -f "$OUT_DIR/ckpt.pt" ]; then
    echo "Checkpoint found, resuming from $OUT_DIR/ckpt.pt"
    INIT="--init_from=resume"
fi

WANDB_FLAG=""
if [ "$WANDB" = 1 ]; then
    WANDB_FLAG="--wandb_log=True"
fi

# Prepare data (idempotent)
singularity exec --nv "$SIF" python data/shakespeare_char/prepare.py

# Train with auto-save and fast eval
singularity exec --nv --home "$HOME" --env PYTHONUNBUFFERED=1 "$SIF" \
    python train.py config/train_shakespeare_char.py \
        --compile=False --eval_iters=20 --always_save_checkpoint=True \
        $INIT $WANDB_FLAG

echo "Training completed successfully"
