#!/usr/bin/env python
"""
    jlai.interp.lens.modal.local
"""

import asyncio
import numpy as np
from tqdm import tqdm

import modal

from .common import app
from .remote import LensInference

class LensClient:
    def __init__(self, model_str='Qwen/Qwen3-0.6B', padding_side='right', devel=False):
        if devel:
            _model_cls = LensInference
        else:
            _model_cls = modal.Cls.from_name(app.name, 'LensInference')

        self.model_str    = model_str
        self.padding_side = padding_side
        self.model        = _model_cls(model_str=self.model_str, padding_side=self.padding_side)

    def hook_names(self):
        return self.model.hook_names.remote()

    def forward(self, messages : list):
        return self.model.forward.remote.aio(messages=messages)

    async def aforward(self, messages, **kwargs):
        return await self.model.forward.remote.aio(messages=messages, **kwargs)

    async def abatched_forward(self, messages, tokens_per_batch=1024, **kwargs):
        n_tokens = self.model.messages2tokens.remote(messages)
        assert max(n_tokens) <= tokens_per_batch, "Max token count per batch is less than the max token count per message"
        
        # Sort by token count for efficient batching
        asort           = np.argsort(n_tokens)
        sorted_messages = [messages[i] for i in asort]
        sorted_n_tokens = [n_tokens[i] for i in asort]
        
        # --
        # Create batches
        
        print('batched_forward: creating batches')
        
        all_bidxs   = []
        curr_batch  = []
        curr_n_toks = 0
        for i, tokens in enumerate(sorted_n_tokens):
            if curr_n_toks + tokens > tokens_per_batch and curr_batch:
                all_bidxs.append(curr_batch)
                print(f'\n_tokens={curr_n_toks} | tbatch={curr_batch}')
                curr_batch  = []
                curr_n_toks = 0
            
            curr_batch.append(i)
            curr_n_toks += tokens
        
        if curr_batch:
            all_bidxs.append(curr_batch)
            print(f'\tn_tokens={curr_n_toks} | batch={curr_batch}')
        
        # --
        # Process all all_bidxs in parallel across different machines
        # [TODO] control the number of machines?  With a semaphore here maybe?
        
        async def _process_batch(bidxs):
            print('_process_batch: submit')
            batch_res = await self.aforward([sorted_messages[i] for i in bidxs], **kwargs)
            return [(asort[idx], res) for idx, res in zip(bidxs, batch_res)]

        tasks = [_process_batch(bidxs) for bidxs in all_bidxs]
        pbar  = tqdm(total=len(messages))
        for task in asyncio.as_completed(tasks):
            for gidx, res in (await task):
                pbar.update(1)
                yield gidx, res