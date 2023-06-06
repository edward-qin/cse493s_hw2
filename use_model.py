"""# Testing"""
import torch
import sys
import os
import numpy as np
from generation import LLaMA
# import model2   # inference only version
from model import ModelArgs
from model import Transformer
from transformers import AutoTokenizer

# import fairscale.nn.model_parallel.initialize as fs_init


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

torch.cuda.empty_cache()
out_dir = '/nfshomes/vsahil1/cse493s_hw2/ckpts'
file = os.path.join(out_dir, sorted(os.listdir(os.path.join(out_dir)))[-1])
print(file)
checkpoint = torch.load(file)
model = Transformer(ModelArgs())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.to(device)
model.load_state_dict(checkpoint, strict=False)

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
generator = LLaMA(model, tokenizer)

prompts = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that ",
    "Building a website can be done in 10 simple steps:\n",
    # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
    """Tweet: "I hate it when my phone battery dies."
          Sentiment: Negative
          ###
          Tweet: "My day has been 👍"
          Sentiment: Positive
          ###
          Tweet: "This is the link to the article"
          Sentiment: Neutral
          ###
          Tweet: "This new music video was incredibile"
          Sentiment:""",
    """Translate English to French:

          sea otter => loutre de mer

          peppermint => menthe poivrée

          plush girafe => girafe peluche

          cheese =>""",
]

# import ipdb; ipdb.set_trace()
tokens = [(tokenizer(x)['input_ids']) for x in prompts]     # match list of list of ints format
# print(tokens)

# results = generator.generate(tokens, max_gen_len=256)

# for result in results:
#     print(result)
#     print("\n==================================\n")

for id, token in enumerate(tokens):
    print(prompts[id])
    token = torch.tensor(token[1:-1]).unsqueeze(0).to(device)       ## removing the CLS and SEP token
    print("\n==================================\n")
    result = generator.generate_one_prompt(token, max_gen_len=256)
    print(result)
