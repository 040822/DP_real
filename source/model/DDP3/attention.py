"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_kv_past = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x, cond):
        q = self.to_q(x)
        k, v = self.to_kv(cond).chunk(2, dim=-1)

        q = rearrange(q, "B T (h d) -> B h T d", h=self.heads)
        k = rearrange(k, "B T (h d) -> B h T d", h=self.heads)
        v = rearrange(v, "B T (h d) -> B h T d", h=self.heads)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)
        x = rearrange(x, "B h T d -> B T (h d)")
        x = x.to(q.dtype)

        x = self.to_out(x)
        return x