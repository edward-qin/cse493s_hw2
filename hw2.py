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
import wandb


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
# print(train_path)

"""# Load and Tokenize Data"""


def read_line(line):
    line = line.strip()
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return data['text']


def read_file(file_path, gbs=1.0, start_entry=0, len_of_entries=10000):
    with open(file_path, 'rb') as file:
        decompressor = zstd.ZstdDecompressor()
        stream_reader = decompressor.stream_reader(file)
        stream = io.TextIOWrapper(stream_reader, encoding='utf-8')

        lines = []
        count = 0
        for line in stream:
            if count < start_entry:
                count += 1
                continue
            line = read_line(line)
            if line is not None:
                lines.append(line)
            if len(lines) == len_of_entries:
                break
            if len(lines) % 10000 == 0:
                print(len(lines), 'lines read')
    return lines


# https://huggingface.co/transformers/v3.0.2/preprocessing.html#base-use
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# encoding = tokenizer("Hello I am a human")
# print(tokenizer.vocab_size)
# print(encoding)
# eval_dataset.to(device)       ## do not send to device here.
# train_dataset.to(device)


def get_batch(data: np.ndarray, batch_size, random_batch=False, batch_start_id=None):     ## This is picking random datapoints from the dataset -- why not get it sequentially. 
    if random_batch:
        ix = torch.randint(len(data) - 1, (batch_size,))
    else:
        ## here we will get data in sequences
        ix = torch.arange(batch_start_id, min(batch_start_id + batch_size, len(data) - 1))

    # x = torch.stack([torch.from_numpy((data[i]).numpy()[:-1]) for i in ix])  # first n - 1
    # y = torch.stack([torch.from_numpy((data[i]).numpy()[1:]) for i in ix])  # last n - 1
    x = torch.stack([data[i][:-1] for i in ix])  # first n - 1 -- no numpy
    y = torch.stack([data[i][1:] for i in ix])  # last n - 1 -- no numpy
    return x, y


# print(get_batch(train_dataset, config))

"""Training loop"""


def train(
        model: torch.nn.Module,
        config: dict,
        optimizer: torch.optim.Optimizer,
        train_dataset_size: int,
        train_data: np.ndarray,
        val_data: np.ndarray,
        grad_clip: float=1.0,
        checkpoint=None,
        out_dir: str='version1',
) -> None:

    train_losses = []
    val_losses = []
    out_dir = os.path.join(out_dir)
    start_iter = 0
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    wandb_run_name = str(config['optimizer']) + "_" + str(config['batch_size']) + "_" + str(config['lr']) + "_" + str(config['weight_decay']) + "_" + str(config['beta1']) + "_" + str(config['beta2']) + "_" + str(config['epochs']) + "_" + str(train_dataset_size) + "_" + str(config['step_size']) + "_" + str(config['gamma']) + "_" + str(config['model_dim'])
    # print(wandb_run_name)
    if config['wandb']:
        wandb.init(project="cse493s-hw2", name=wandb_run_name, config=config)

    if checkpoint:
        train_losses = checkpoint['train_loss']
        val_losses = checkpoint['val_loss']
        start_iter = checkpoint['start_iter']

        model.load_state_dict(checkpoint['model'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    batch_start_id_cur = 0
    num_iters_per_epoch = len(train_data) // config['batch_size']
    # config['epochs'] = config['epochs'] * num_iters_per_epoch       ## the following loop is running on iterations and not epochs
    # config['epochs'] = 400      ## let stop here. it is converging. 
    # import ipdb; ipdb.set_trace()

    for i in range(start_iter, start_iter + config['epochs']):
        # evaluate the loss on train/val sets and write checkpoints
        if i > 0 and i % config['eval_interval'] == 0:
            val_loss = validate(model, val_data)
            dt = time.time() - t0
            val_losses.append(val_loss)
            if i % config['log_interval'] == 0:
                print(f"Validating iter {i}: loss {val_loss:.4f}, time: {dt * 1000:.2f}ms")
                if config['wandb']: wandb.log({"val_loss": val_loss, "iter": i})

        t0 = time.time()

        input_ids, targets = get_batch(train_data, config['batch_size'], random_batch=False, batch_start_id=batch_start_id_cur)
        batch_start_id_cur += len(input_ids)        ## usually the len(input_ids) will be equal to the batch size, but in the last batch it will be less than the batch size.
        if batch_start_id_cur >= len(train_data) - 1:
            batch_start_id_cur = 0
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model.forward(input_ids, 0, train=True)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)

        loss = torch.nn.functional.cross_entropy(logits, targets.reshape(-1), ignore_index=tokenizer._pad_token_type_id)
        loss.backward()
        loss = loss.detach()    #.cpu().numpy()
        train_losses.append(loss.item())

        # Save checkpoint after getting training loss of this iteration
        # if i > 0 and i % config['eval_interval'] == 0:
        #     print(f"Saving checkpoint to {out_dir}")
        #     torch.save({
        #         'train_loss': train_losses,
        #         'val_loss': val_losses,
        #         'start_iter': i + 1,
        #         'model': model.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'config': config,
        #     }, os.path.join(out_dir, f"pile_ckpt_{wandb_run_name}.pt"))

            # Gradient clipping
        if grad_clip != 0.0:
            clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()  # Update learning rate

        # Logging
        dt = time.time() - t0
        if i % config['log_interval'] == 0:
            print(f"Training iter {i}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms")
            if config['wandb']: wandb.log({"train_loss": loss.item(), "iter": i})
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def validate(model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    print("Validating ...")
    model.eval()  # change to eval mode
    # eval_iters = val_data.shape[0] // config['batch_size']
    # losses = torch.zeros(eval_iters)
    batch_start_id_cur = 0
    val_loss = 0
    batch_size = 256
    while True:
        input_ids, targets = get_batch(val_data, batch_size, random_batch=False, batch_start_id=batch_start_id_cur)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        logits = model.forward(input_ids, 0, train=True)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        loss = torch.nn.functional.cross_entropy(logits, targets.reshape(-1), ignore_index=tokenizer._pad_token_type_id)
        # loss = loss.cpu().numpy()
        # losses[k] = loss.item()
        val_loss += loss.item()
        batch_start_id_cur += len(input_ids)
        if batch_start_id_cur >= len(val_data) - 1:
            break

    # out = losses.mean()
    # out = val_loss / len(val_data)
    model.train()  # change back to train mode
    gc.collect()
    torch.cuda.empty_cache()
    return val_loss


# set up checkpoint to continue from
# checkpoint = None
# file = f"/content/drive/Shareddrives/493S_hw2/ckpts/" + sorted(os.listdir(f"/content/drive/Shareddrives/493S_hw2/ckpts"))[-1]
# checkpoint = torch.load(file)

# determine model version and output directory of checkpoints
out_dir = '/nfshomes/vsahil1/cse493s_hw2/ckpts'


def main(config):
    torch.cuda.empty_cache()
    model_config = ModelArgs()
    model_config.dim = config['model_dim']
    model = Transformer(model_config)
    # print(model.tok_embeddings.weight.shape)
    model.to(device)
    train_dataset_size = 300000     ## we need to read more data only after 1500 iterations. 
    eval_dataset = tokenizer(read_file(val_path, start_entry=0, len_of_entries=10000), padding=True, truncation=True, return_tensors="pt")['input_ids']
    train_dataset = tokenizer(read_file(train_path, start_entry=0, len_of_entries=train_dataset_size), padding=True, truncation=True, return_tensors="pt")['input_ids']
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(config['beta1'], config['beta2']))
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(config['beta1'], config['beta2']))
    elif config['optimizer'] == 'lion':     # Ablation study
        optimizer = Lion(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], betas=(config['beta1'], config['beta2']))
    train(model, config, optimizer, train_dataset_size, train_dataset, eval_dataset, checkpoint=None, out_dir=out_dir)

# lion_64_0.0005_0.1_0.9_0.95_2500_160000

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'lion'])
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--wandb', action='store_true', default=False)
    args = parser.parse_args()
    config = vars(args)
    main(config)


# """# Results"""

# def plot_losses():
#     # get latest checkpoint
#     file = os.path.join(model_path, out_dir, sorted(os.listdir(os.path.join(model_path, out_dir)))[-1])
#     checkpoint = torch.load(file)
#     print(file)
#     train_losses = checkpoint['train_loss']
#     val_losses = checkpoint['val_loss']

#     train_losses = [loss.item() for loss in train_losses]
#     print(len(train_losses))
#     val_losses = [loss.item() for loss in val_losses]

#     plt.figure(figsize=(10, 5))
#     plt.title("Loss during training")
#     plt.plot(train_losses, label="Training loss")
#     plt.plot((np.arange(len(val_losses)) + 1) * 10, val_losses, label="Validation loss")
#     plt.xlabel("iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.savefig(os.path.join(model_path, "plots"))
