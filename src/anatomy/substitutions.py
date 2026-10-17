"""Substitution units. Each replaces one baseline unit and keeps its call signature."""
import torch
import torch.nn.functional as F
from torch import nn

from anatomy.components import Rope
from anatomy.config import ModelSpec

LINEAR_CHUNK = 64  # linear_core chunk size (decided)
FACTOR_E = 128  # factorized_embed rank (Lan et al., 2020)
GQA_DIV = 4  # kv heads = heads / 4


class SdpaCore(nn.Module):
    """Fused causal attention, memory-efficient backend called directly so it can't silently
    fall back to math (FA2 is unsupported on T4)."""

    def forward(self, q, k, v):  # (b, h, s, hd) -> (b, s, d)
        b, h, s, hd = q.shape
        if q.device.type == "cpu":  # tests only: CPU has no mem-efficient kernel
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, None, q.requires_grad, is_causal=True
            )[0]
        return o.transpose(1, 2).reshape(b, s, h * hd)


class LinearCore(nn.Module):
    """Causal linear attention (Katharopoulos et al., 2020), phi(x) = elu(x) + 1, chunked:
    masked quadratic inside each chunk plus a prefix state over earlier chunks."""

    def __init__(self, chunk: int = LINEAR_CHUNK, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mask", torch.ones(chunk, chunk, dtype=torch.bool).tril(), persistent=False)
        self.chunk, self.eps = chunk, eps

    def forward(self, q, k, v):  # (b, h, s, hd) -> (b, s, d)
        b, h, s, hd = q.shape
        n, c = s // self.chunk, self.chunk
        q, k = F.elu(q) + 1, F.elu(k) + 1
        q, k, v = (t.reshape(b, h, n, c, hd) for t in (q, k, v))
        kv = k.transpose(-1, -2) @ v  # (b, h, n, hd, hd) per-chunk state
        state = kv.cumsum(2) - kv  # sum over earlier chunks only
        z = k.sum(3, keepdim=True)  # (b, h, n, 1, hd) normalizer state
        zprev = z.cumsum(2) - z
        a = torch.where(self.mask, q @ k.transpose(-1, -2), 0.0)  # intra-chunk, causal incl. diagonal
        num = q @ state + a @ v
        den = (q * zprev).sum(-1, keepdim=True) + a.sum(-1, keepdim=True)
        return (num / (den + self.eps)).reshape(b, h, s, hd).transpose(1, 2).reshape(b, s, h * hd)


class GqaProj(nn.Module):
    """attn_proj with kv heads = heads / 4: fused Q + K + V projection, the K/V repeat to all
    heads (materialized, as in HF repeat_kv), and the output projection."""

    def __init__(self, m: ModelSpec):
        super().__init__()
        self.m, self.kvh = m, m.heads // GQA_DIV
        self.query_key_value = nn.Linear(m.d, m.d + 2 * self.kvh * m.head_dim)
        self.dense = nn.Linear(m.d, m.d)

    def forward(self, x, ctx):
        m, kvh = self.m, self.kvh
        b, s, _ = x.shape
        q, k, v = self.query_key_value(x).split([m.d, kvh * m.head_dim, kvh * m.head_dim], -1)

        def rep(t):
            t = t.view(b, s, kvh, 1, m.head_dim).expand(b, s, kvh, m.heads // kvh, m.head_dim)
            return t.reshape(b, s, m.d)

        return q, rep(k), rep(v), self.dense(ctx)


class FactorizedEmbedIn(nn.Module):
    """ALBERT-style input embedding: Embedding(vocab, E) + Linear(E, d), no bias."""

    def __init__(self, m: ModelSpec, e: int = FACTOR_E):
        super().__init__()
        self.embed = nn.Embedding(m.vocab, e)
        self.proj = nn.Linear(e, m.d, bias=False)

    def forward(self, ids):
        return self.proj(self.embed(ids))


class FactorizedEmbedOut(nn.Module):
    """Mirrored output side: Linear(d, E) + Linear(E, vocab), no bias (like Pythia embed_out)."""

    def __init__(self, m: ModelSpec, e: int = FACTOR_E):
        super().__init__()
        self.proj = nn.Linear(m.d, e, bias=False)
        self.unembed = nn.Linear(e, m.vocab, bias=False)

    def forward(self, x):
        return self.unembed(self.proj(x))


class RMSNorm(nn.Module):
    """Zhang & Sennrich (2019), eager ops in the run dtype."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RMSNormFused(nn.Module):
    """Same function as RMSNorm through F.rms_norm, which dispatches to PyTorch's fused CUDA kernel
    where the build has one (the Kaggle notebook logs which ops actually ran)."""

    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class RopeFused(nn.Module):
    """The same rotary function as Rope, compiled by torch.compile (Inductor) so the slicing,
    rotate_half, multiply-add and concatenation run as fused kernels instead of eager operators.
    Compilation happens on the first call (inside the harness warm-up)."""

    def __init__(self, m: ModelSpec, seq: int):
        super().__init__()
        self.rope = Rope(m, seq)
        # Compiled code is cached per function, not per module: without a reset, every new config's
        # shape/dtype/grad mode recompiles the same Rope.forward until the recompile limit fails the unit.
        torch._dynamo.reset()
        self._fn = torch.compile(self.rope, fullgraph=True, dynamic=False)

    def forward(self, q, k):  # (b, h, s, hd)
        return self._fn(q, k)
