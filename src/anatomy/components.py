"""Baseline units with GPT-NeoX (Pythia) semantics. Submodule names match HF GPTNeoXLayer
so its weights load directly. embed_lookup, ln and out_proj are plain nn.Embedding,
nn.LayerNorm and nn.Linear (see units.py)."""
import torch
import torch.nn.functional as F
from torch import nn

from anatomy.config import ModelSpec


class AttnProj(nn.Module):
    """Fused QKV Linear(d, 3d) on the block input + output Linear(d, d) on the attention context."""

    def __init__(self, m: ModelSpec):
        super().__init__()
        self.query_key_value = nn.Linear(m.d, 3 * m.d)
        self.dense = nn.Linear(m.d, m.d)

    def forward(self, x, ctx):
        return self.query_key_value(x), self.dense(ctx)


class Rope(nn.Module):
    """Rotary embedding on the first rotary_dims of each head of Q and K, eager ops as in HF."""

    def __init__(self, m: ModelSpec, seq: int):
        super().__init__()
        r = m.rotary_dims
        inv_freq = 1.0 / (m.rotary_base ** (torch.arange(0, r, 2, dtype=torch.float32) / r))
        freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), -1)
        # Buffers follow module.to(dtype), so forward does no casts.
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.r = r

    def _apply_one(self, x):
        xr, xp = x[..., : self.r], x[..., self.r :]
        x1, x2 = xr[..., : self.r // 2], xr[..., self.r // 2 :]
        xr = xr * self.cos + torch.cat((-x2, x1), -1) * self.sin
        return torch.cat((xr, xp), -1)

    def forward(self, q, k):  # (b, h, s, hd)
        return self._apply_one(q), self._apply_one(k)


class AttnCore(nn.Module):
    """Eager attention: QK^T, scale, causal mask, softmax, ·V, merge heads (GPT-NeoX eager path)."""

    def __init__(self, m: ModelSpec, seq: int):
        super().__init__()
        self.register_buffer("mask", torch.ones(seq, seq, dtype=torch.bool).tril(), persistent=False)
        self.scale = m.head_dim**-0.5

    def forward(self, q, k, v):  # (b, h, s, hd) -> (b, s, d)
        b, h, s, hd = q.shape
        w = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        w = torch.where(self.mask, w, torch.finfo(w.dtype).min)
        w = torch.softmax(w, dim=-1)
        return torch.matmul(w, v).transpose(1, 2).reshape(b, s, h * hd)


class MLP(nn.Module):
    def __init__(self, m: ModelSpec):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(m.d, m.ffn)
        self.dense_4h_to_h = nn.Linear(m.ffn, m.d)

    def forward(self, x):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class Loss(nn.Module):
    def forward(self, logits, targets):
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class Block(nn.Module):
    """Parallel-residual decoder block: x + attn(ln1(x)) + mlp(ln2(x)). The denominator unit."""

    def __init__(self, m: ModelSpec, seq: int):
        super().__init__()
        self.m = m
        self.input_layernorm = nn.LayerNorm(m.d, eps=m.eps)
        self.post_attention_layernorm = nn.LayerNorm(m.d, eps=m.eps)
        self.attention = AttnProj(m)
        self.rope = Rope(m, seq)
        self.core = AttnCore(m, seq)
        self.mlp = MLP(m)

    def forward(self, x):
        b, s, _ = x.shape
        m = self.m
        qkv = self.attention.query_key_value(self.input_layernorm(x))
        q, k, v = qkv.view(b, s, m.heads, 3 * m.head_dim).transpose(1, 2).chunk(3, dim=-1)
        q, k = self.rope(q, k)
        attn = self.attention.dense(self.core(q, k, v))
        return self.mlp(self.post_attention_layernorm(x)) + attn + x
