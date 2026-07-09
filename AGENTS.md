# nanoGPT

Fork: `github.com/LucaSforza/nanoGPT.git`. Deprecated upstream in favor of nanochat, but actively used via fork.

## Workflow (cluster)

```sh
# SSH to cluster, then:
ssh cluster2
cd ~/nanoGPT && git pull

# Full pipeline (Docker build → push → SIF → Slurm training)
just deploy-sbatch

# Or step by step:
just build          # docker build -t softdream1/nanogpt:latest
just push           # docker push softdream1/nanogpt:latest
just sif            # singularity build nanogpt.sif docker://softdream1/nanogpt:latest
just sbatch-singularity  # sbatch train.singularity.slurm

# Sample from trained model (GPU node, via Slurm)
just sample-singularity

# Check results
cat sample-*.out
```

## Key files in this repo

- `train.py` — training loop, config as module-level globals
- `model.py` — `GPT` + `GPTConfig` dataclass
- `sample.py` — inference from checkpoint or pretrained GPT-2
- `justfile` — all commands (Docker, Singularity, Slurm, training, sampling)
- `Dockerfile` — PyTorch 2.6 + CUDA 12.4 container
- `train.singularity.slurm` — Slurm batch with auto-requeue
- `sample.slurm` — Slurm batch for inference
- `config/*.py` — exec-based config overrides (NOT YAML/JSON)

## Just recipes (cluster)

| Recipe | What it does |
|--------|-------------|
| `just build` | `docker build -t softdream1/nanogpt:latest` |
| `just push` | `docker push softdream1/nanogpt:latest` |
| `just sif` | Build `nanogpt.sif` from Docker Hub |
| `just deploy` | build + push |
| `just deploy-sbatch` | build → push → sif → sbatch (full pipeline) |
| `just sbatch-singularity` | `sbatch train.singularity.slurm` |
| `just sample-singularity` | `sbatch sample.slurm` |
| `just train-framework-13` | CPU training on local laptop |
| `just demo` | Full shakespeare-char quickstart (CPU) |

## Cluster environment

- **Host**: `cluster2` (SSH)
- **Partition**: `students` (default, ~30 nodes with Quadro RTX 6000 24GB)
- **Time limit**: 30 min (partition default)
- **Software**: `singularity-ce 3.9.8`, `docker`, `uv` in `~/.local/bin`, `just` in `~/.local/bin`
- **Repo**: `/home/sforza_2050030/nanoGPT`
- **Docker Hub**: `softdream1/nanogpt:latest`

## Training details

- **Shakespeare char-level**: 10.65M params, 5000 iters, ~2 min on RTX 6000
- **Loss reference**: 1.4697 (config default), 1.45-1.50 typical
- **`--compile=False`** required in container (no gcc for Triton kernels)
- **`--always_save_checkpoint=True`** saves at every eval for resume
- **`--eval_iters=20`** (reduced from 200) for faster evaluation

## Checkpoint system (custom)

- `best.pt` — saved only when val loss improves (use for sampling)
- `ckpt.pt` — saved every eval when `--always_save_checkpoint=True` (use for resume)
- When loss improves, both files are saved with the same weights

`sample.py` prefers `best.pt` over `ckpt.pt` automatically.

## Auto-requeue mechanism

`train.singularity.slurm`:
- `#SBATCH --signal=USR1@120` → Slurm sends USR1 2 min before timeout
- Trap handler calls `scontrol requeue $SLURM_JOB_ID`
- Next instance detects `ckpt.pt` and resumes via `--init_from=resume`
- Loop continues until training completes (5000 iters)

## Wandb

- Credentials in `~/.netrc` on cluster
- Singularity passes them via `--home "$HOME"`
- Enable with `WANDB=1` in Slurm script (default: on)

## Quirks

- Python buffers stdout in Slurm → use `--env PYTHONUNBUFFERED=1` in Singularity
- `torch.cuda.amp.GradScaler` warning is harmless (auto-disables without CUDA)
- `singularity build docker://` → auth error if image not pushed first
- `singularity build docker-daemon://` → fails if Docker API version mismatch (use `docker save | singularity build docker-archive://-` instead)

## Dependencies

`just install` → `uv sync` installs from `pyproject.toml` (torch, numpy, tiktoken, wandb, transformers, datasets)

## Config system

1. Defaults in script
2. Config file overrides (via `exec()`)
3. CLI `--key=value` overrides (type-checked: `assert type(attempt) == type(globals()[key])`)
