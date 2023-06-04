import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import io
import gc
import time
import fairscale.nn.model_parallel.initialize as fs_init
import zstandard as zstd

from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from model import Transformer
from model import ModelArgs
from transformers import AutoTokenizer
from generation import LLaMA
from lion_pytorch import Lion


def setup_model_parallel():
    local_rank = 0
    world_size = 1

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


# rank, size = setup_model_parallel()

"""# Hyperparameters"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model_path = '.'  # where the model.py files live
# sys.path.append(os.path.abspath(model_path))

data_path = '/cmlscratch/vsahil1/cse493s-hw2'
test_path = os.path.join(data_path, 'test.jsonl.zst')
val_path = os.path.join(data_path, 'val.jsonl.zst')
train_path = os.path.join(data_path, '00.jsonl.zst')
print(train_path)

batch_size = 12

# train loop config
eval_interval = 10
log_interval = 1

# StepLR parameters
step_size = 30
gamma = 0.1

# val loss config
eval_iters = 20

# optimizer config (AdamW)
lr = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

"""# Load and Tokenize Data"""


def read_line(line):
    line = line.strip()
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return data['text']


def read_file(file_path, gbs=1.0, entries=10000):
    with open(file_path, 'rb') as file:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(file)
        stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

        lines = []
        for line in stream:
            line = read_line(line)
            if line is not None:
                lines.append(line)
            if len(lines) == entries:
                break
    return lines


# https://huggingface.co/transformers/v3.0.2/preprocessing.html#base-use
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# encoding = tokenizer("Hello I am a human")
# print(tokenizer.vocab_size)
# print(encoding)

eval_dataset = tokenizer(read_file(val_path, entries=500), padding=True, truncation=True, return_tensors="pt")['input_ids']
train_dataset = tokenizer(read_file(train_path, entries=500), padding=True, truncation=True, return_tensors="pt")['input_ids']
print(type(eval_dataset))
eval_dataset.to(device)
train_dataset.to(device)


def get_batch(data: np.ndarray):
    ix = torch.randint(len(data) - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i]).numpy()[:-1]) for i in ix])  # first n - 1
    y = torch.stack([torch.from_numpy((data[i]).numpy()[1:]) for i in ix])  # last n - 1
    return x, y


print(get_batch(train_dataset))

"""Training loop"""


def train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_data: np.ndarray,
        val_data: np.ndarray,
        grad_clip: float=1.0,
        checkpoint=None,
        iters: int = 2,
        out_dir: str='version1'
) -> None:
    train_losses = []
    val_losses = []
    out_dir = os.path.join(out_dir)
    start_iter = 0
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    if checkpoint:
        train_losses = checkpoint['train_loss']
        val_losses = checkpoint['val_loss']
        start_iter = checkpoint['start_iter']

        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    import ipdb; ipdb.set_trace()
    for i in range(start_iter, start_iter + iters):
        # evaluate the loss on train/val sets and write checkpoints
        if i > 0 and i % eval_interval == 0:
            val_loss = validate(model, val_data)
            dt = time.time() - t0
            if i % log_interval == 0:
                print(f"Validating iter {i}: loss {val_loss:.4f}, time: {dt * 1000:.2f}ms")
            val_losses.append(val_loss)

        t0 = time.time()

        input_ids, targets = get_batch(train_data)  # type: ignore[union-attr,arg-type]
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model.forward(input_ids, 0, train=True)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)

        loss = torch.nn.functional.cross_entropy(logits, targets.reshape(-1), ignore_index=tokenizer._pad_token_type_id)
        loss.backward()
        loss = loss.detach().cpu().numpy()
        train_losses.append(loss)

        # Save checkpoint after getting training loss of this iteration
        if i > 0 and i % eval_interval == 0:
            print(f"Saving checkpoint to {out_dir}")
            torch.save({
                'train_loss': train_losses,
                'val_loss': val_losses,
                'start_iter': i + 1,
                'model': model.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(out_dir, f"{i:04}-ckpt.pt"))

            # Gradient clipping
        if grad_clip != 0.0:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate

        # Logging
        dt = time.time() - t0
        if i % log_interval == 0:
            print(f"Training iter {i}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms")
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def validate(model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    print("Validating ...")
    model.eval()  # change to eval mode
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        input_ids, targets = get_batch(val_data)  # type: ignore[union-attr,arg-type]
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model.forward(input_ids, 0)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        loss = torch.nn.functional.cross_entropy(logits, targets.reshape(-1), ignore_index=tokenizer._pad_token_type_id)
        loss = loss.detach().cpu().numpy()
        losses[k] = loss.item()
        gc.collect()
        torch.cuda.empty_cache()
    out = losses.mean()
    model.train()  # change back to train mode
    return out


# set up checkpoint to continue from
checkpoint = None
# file = f"/content/drive/Shareddrives/493S_hw2/ckpts/" + sorted(os.listdir(f"/content/drive/Shareddrives/493S_hw2/ckpts"))[-1]
# checkpoint = torch.load(file)

# determine model version and output directory of checkpoints
out_dir = '/cmlscratch/vsahil1/cse493s-hw2/ckpts'
# !mkdir {model_path}/{out_dir}
# !ls {model_path}


def main():
    torch.cuda.empty_cache()
    model = Transformer(ModelArgs())
    model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    # Ablation study
    optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    train(model, optimizer, train_dataset, eval_dataset, checkpoint=None, iters=501, out_dir=out_dir)


main()

"""# Results"""

def plot_losses():
    # get latest checkpoint
    file = os.path.join(model_path, out_dir, sorted(os.listdir(os.path.join(model_path, out_dir)))[-1])
    checkpoint = torch.load(file)
    print(file)
    train_losses = checkpoint['train_loss']
    val_losses = checkpoint['val_loss']

    train_losses = [loss.item() for loss in train_losses]
    print(len(train_losses))
    val_losses = [loss.item() for loss in val_losses]

    plt.figure(figsize=(10, 5))
    plt.title("Loss during training")
    plt.plot(train_losses, label="Training loss")
    plt.plot((np.arange(len(val_losses)) + 1) * 10, val_losses, label="Validation loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_path, "plots"))
