import sys, base64, multiprocessing, itertools, collections
from typing import Optional, Union, Literal, List

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import librosa


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
        super().__init__()
        assert n_state % n_head == 0, f"n_state {n_state} must be divisible by n_head {n_head}"

        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

        self.kv_caching = kv_caching
        self.max_self_attn_cache_len = max_self_attn_cache_len
    
    def forward(self, x:torch.Tensor, xa:Optional[torch.Tensor]=None, mask:Optional[torch.Tensor]=None, len: Optional[int]=None):
        if self.kv_caching == 'cross':
            if xa is not None:
                k, v = self.key(xa), self.value(xa)
                if not hasattr(self, 'cache_k'):
                    self.cache_k, self.cache_v = k, v
                else:
                    self.cache_k = k
                    self.cache_v = v 
            else:
                k, v = self.cache_k, self.cache_v
        else:
            k, v = self.key(x), self.value(x)
            if self.kv_caching == 'self':
                if not hasattr(self, 'cache_k'):
                    self.cache_k = torch.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2], device=x.device)
                    self.cache_v = torch.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2], device=x.device)
                k = torch.cat([self.cache_k[:, :len], k], dim=1)
                v = torch.cat([self.cache_v[:, :len], k], dim=1)
                padding = (0, 0, 0, self.max_self_attn_cache_len - len - x.shape[1])
                self.cache_k = F.pad(k, padding).contiguous()
                self.cache_v = F.pad(v, padding).contiguous()
            
        q = self.query(x)
        n_ctx = q.shape[1]
        assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
        head_dim = q.shape[-1] // self.n_head
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, head_dim).permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, :n_ctx, :n_ctx] if mask is not None else None)
        wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(wv)



