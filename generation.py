# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

# from tokenizer import Tokenizer
# from model2 import Transformer
from model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def generate_one_prompt(
        self,
        # prompts: List[str],
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0,
        top_p: float = 0.95,
    ) -> List[str]:

        assert prompt_tokens.shape[0] == 1      ## only generating one prompt at a time

        # import ipdb; ipdb.set_trace()
        for _ in range(max_gen_len):
            ## generate one token at a time
            logits = self.model.forward(prompt_tokens, 0, train=False)
            # print("logits", logits.shape)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            prompt_tokens = torch.cat((prompt_tokens, next_token.unsqueeze(0)), dim=1)
            # print("prompt_tokens", prompt_tokens.shape)
            if next_token == self.tokenizer.sep_token_id:
                break

        return self.tokenizer.decode(prompt_tokens.squeeze(0).tolist())
        
        # decoded = []
        # for i, t in enumerate(prompt_tokens.tolist()):
        #     # cut to max gen len
        #     # t = t[: len(prompt_tokens[i]) + max_gen_len]
        #     # cut to eos tok if any
        #     try:
        #         t = t[: t.index(self.tokenizer._eos_token)]
        #     except ValueError:
        #         pass
        #     decoded = self.tokenizer.decode(t)
        # return decoded

        # return prompt_tokens
    


    def generate(
        self,
        # prompts: List[str],
        prompt_tokens,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:

        # import ipdb; ipdb.set_trace()
        bsz = len(prompt_tokens)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer._pad_token_type_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer._pad_token_type_id
        start_pos = min_prompt_size
        prev_pos = 0

        # print("token device", tokens.device)
        for cur_pos in range(start_pos, total_len):
            # logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos, train=False)     ## starting the second loop, this is only conditioning on the last token and not the entire sequence -- that is a problem. 
            logits = self.model.forward(tokens[:, :cur_pos], prev_pos, train=False)
            # print("logits", logits.shape)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer._eos_token)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    # print(probs_sort.shape)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
