# nanoGPT — just command runner
# Use: just <recipe> [args...] [-- extra-flags]
# Requires: uv, just

uv := "uv run python"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
    uv sync

# ── Data preparation ────────────────────────────────────────────────────────

prepare-shakespeare-char:
    {{uv}} data/shakespeare_char/prepare.py

prepare-shakespeare:
    {{uv}} data/shakespeare/prepare.py

prepare-openwebtext:
    {{uv}} data/openwebtext/prepare.py

# ── Training ────────────────────────────────────────────────────────────────

train-shakespeare-char config="config/train_shakespeare_char.py" *flags:
    {{uv}} train.py {{config}} {{flags}}

train-shakespeare-char-cpu *flags:
    {{uv}} train.py config/train_shakespeare_char.py --device=cpu --compile=False {{flags}}

# Train on Framework 13 (AMD CPU, no GPU) with CPU-friendly settings
train-framework-13 *flags:
    {{uv}} train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=10 {{flags}}

train-shakespeare-char-mps *flags:
    {{uv}} train.py config/train_shakespeare_char.py --device=mps --compile=False {{flags}}

train config="config/train_shakespeare_char.py" *flags:
    {{uv}} train.py {{config}} {{flags}}

train-gpt2:
    torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

bench *flags:
    {{uv}} bench.py {{flags}}

# ── Evaluation ──────────────────────────────────────────────────────────────

eval-gpt2:
    {{uv}} train.py config/eval_gpt2.py

eval-gpt2-medium:
    {{uv}} train.py config/eval_gpt2_medium.py

eval-gpt2-large:
    {{uv}} train.py config/eval_gpt2_large.py

eval-gpt2-xl:
    {{uv}} train.py config/eval_gpt2_xl.py

# ── Sampling / Inference ────────────────────────────────────────────────────

sample out_dir="out" *flags:
    {{uv}} sample.py --out_dir={{out_dir}} {{flags}}

sample-pretrained init_from="gpt2-xl" *flags:
    {{uv}} sample.py --init_from={{init_from}} --num_samples=5 --max_new_tokens=100 {{flags}}

# ── Quickstart (all-in-one) ──────────────────────────────────────────────────

demo:
    {{uv}} data/shakespeare_char/prepare.py
    {{uv}} train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
    {{uv}} sample.py --out_dir=out-shakespeare-char --device=cpu

# ── Help ─────────────────────────────────────────────────────────────────────

# ── Docker ───────────────────────────────────────────────────────────────────

tag := "lucasforza/nanogpt:latest"

# Build Docker image
build:
    docker build -t {{tag}} .

# Push Docker image to Docker Hub
push:
    docker push {{tag}}

# Build + Push in one step
deploy: build push

default:
    @just --list
