# nanoGPT — just command runner
# Use: just <recipe> [args...]
# Requires: uv, just

uv := "uv run python"

# ── Setup ────────────────────────────────────────────────────────────────────

# Install all dependencies
install:
    uv sync

# ── Data preparation ────────────────────────────────────────────────────────

# Prepare Shakespeare character-level dataset
prepare-shakespeare-char:
    {{uv}} data/shakespeare_char/prepare.py

# Prepare Shakespeare BPE-tokenized dataset
prepare-shakespeare:
    {{uv}} data/shakespeare/prepare.py

# Prepare OpenWebText dataset
prepare-openwebtext:
    {{uv}} data/openwebtext/prepare.py

# ── Training ────────────────────────────────────────────────────────────────

# Train Shakespeare char-level (default)
train-shakespeare-char config="config/train_shakespeare_char.py":
    {{uv}} train.py {{config}}

# Train Shakespeare char-level on CPU
train-shakespeare-char-cpu:
    {{uv}} train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Train Shakespeare char-level on Apple Silicon (MPS)
train-shakespeare-char-mps:
    {{uv}} train.py config/train_shakespeare_char.py --device=mps --compile=False

# Train with an arbitrary config file
train config="config/train_shakespeare_char.py":
    {{uv}} train.py {{config}}

# Train GPT-2 (124M) with DDP (8 GPUs)
train-gpt2:
    torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Benchmark model performance
bench:
    {{uv}} bench.py

# ── Evaluation ──────────────────────────────────────────────────────────────

# Evaluate pretrained GPT-2 checkpoints (no training)
eval-gpt2:
    {{uv}} train.py config/eval_gpt2.py

eval-gpt2-medium:
    {{uv}} train.py config/eval_gpt2_medium.py

eval-gpt2-large:
    {{uv}} train.py config/eval_gpt2_large.py

eval-gpt2-xl:
    {{uv}} train.py config/eval_gpt2_xl.py

# ── Sampling / Inference ────────────────────────────────────────────────────

# Sample from a trained model (set out_dir)
sample out_dir="out":
    {{uv}} sample.py --out_dir={{out_dir}}

# Sample from pretrained GPT-2
sample-pretrained init_from="gpt2-xl":
    {{uv}} sample.py --init_from={{init_from}} --num_samples=5 --max_new_tokens=100

# ── Quickstart (all-in-one) ──────────────────────────────────────────────────

# Full pipeline: prepare → train → sample (Shakespeare char, CPU-friendly)
demo:
    {{uv}} data/shakespeare_char/prepare.py
    {{uv}} train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
    {{uv}} sample.py --out_dir=out-shakespeare-char --device=cpu

# ── Help ─────────────────────────────────────────────────────────────────────

# List all available recipes
default:
    @just --list
