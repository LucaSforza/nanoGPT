# nanoGPT

Deprecated in favor of [nanochat](https://github.com/karpathy/nanochat). Existing code still works but no active development.

## Key files

- `train.py` — training loop with all config as module-level globals (~340 lines)
- `model.py` — `GPT` + `GPTConfig` dataclass, supports Flash Attention / manual attn (~330 lines)
- `sample.py` — inference from checkpoint or pretrained GPT-2
- `configurator.py` — exec-based config override: `python train.py config/foo.py --batch_size=32`
- `bench.py` — stripped-down train loop for benchmarking only

## Config system

Config is NOT YAML/JSON. Each `config/*.py` is a plain Python file whose module-level vars override defaults in `train.py` or `sample.py` via `exec()` + `globals()`.

Order of precedence: defaults in script < config file < `--key=value` CLI args.

CLI overrides check type match (`assert type(attempt) == type(globals()[key])`). Unknown keys raise `ValueError`.

## Common commands

```sh
# Prepare data (must run before training)
python data/shakespeare_char/prepare.py

# Train (single GPU)
python train.py config/train_shakespeare_char.py

# CPU fallback
python train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Apple Silicon
python train.py config/train_shakespeare_char.py --device=mps

# DDP multi-GPU
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Multi-node (add NCCL_IB_DISABLE=1 if no Infiniband)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=IP --master_port=1234 train.py

# Sample from trained model
python sample.py --out_dir=out-shakespeare-char

# Sample from pretrained GPT-2
python sample.py --init_from=gpt2-xl --start="..." --num_samples=5 --max_new_tokens=100

# Evaluate pretrained checkpoints (no training)
python train.py config/eval_gpt2.py
```

## Data format

Expects `data/{dataset}/` with `train.bin` + `val.bin` (uint16 token IDs) and optional `meta.pkl` (with `vocab_size`). Prepare scripts are under `data/{dataset}/prepare.py`.

## Dependencies

`pip install torch numpy transformers datasets tiktoken wandb tqdm`

`torch.compile` (PyTorch 2.0) is on by default — disable with `--compile=False` if unavailable.

## Type behavior

`dtype` auto-selects `bfloat16` if available, else `float16`. Float16 auto-enables `GradScaler`. Use `--dtype=float32` to disable mixed precision.

## Init modes

- `init_from='scratch'` — fresh model, `vocab_size` defaults to 50304 (GPT-2 vocab rounded up)
- `init_from='resume'` — loads `{out_dir}/ckpt.pt` (strips `_orig_mod.` prefix from compiled state_dict keys)
- `init_from='gpt2*'` — loads OpenAI weights via `model.py` `from_pretrained()`

## Checkpoints

Saved as `{out_dir}/ckpt.pt` with keys: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `config`. Saving only on val loss improvement unless `always_save_checkpoint=True`.

## No tests

No test framework, no CI. No lint/format/typecheck config. Verify via `train.py` or `bench.py`.