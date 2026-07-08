FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# Install nanoGPT dependencies
RUN pip install --no-cache-dir \
    numpy \
    tiktoken \
    tqdm \
    wandb \
    transformers \
    datasets

COPY . .

CMD ["python", "train.py", "config/train_shakespeare_char.py"]
