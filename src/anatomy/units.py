"""Unit registry: build any component, substitution or the block with its inputs and
grad_outputs, and run one fwd+bwd step. Shared by the analytical counter and the harness."""
import torch
from torch import nn

from anatomy import components as C
from anatomy import substitutions as S
from anatomy.config import ModelSpec, RunSpec

COMPONENTS = ("embed_lookup", "ln", "attn_proj", "rope", "attn_core", "mlp", "out_proj", "loss")
# substitution unit -> (substitution name, baseline unit it replaces)
SUBSTITUTIONS = {
    "sdpa_core": ("sdpa_core", "attn_core"),
    "linear_core": ("linear_core", "attn_core"),
    "gqa_proj": ("gqa_proj", "attn_proj"),
    "factorized_embed_in": ("factorized_embed", "embed_lookup"),
    "factorized_embed_out": ("factorized_embed", "out_proj"),
    "rmsnorm": ("rmsnorm", "ln"),
    "rmsnorm_fused": ("rmsnorm_fused", "ln"),
    "rope_fused": ("rope_fused", "rope"),
}
# Substitutions that remove a unit outright: nothing runs, so they are not measured. Their substituted
# model energy is the baseline minus the removed unit's measured energy (substitute energy = 0).
REMOVALS = {"nope": "rope"}  # NoPE: no positional encoding (Haviv et al., 2022; Kazemnejad et al., 2023)
UNITS = (*COMPONENTS, *SUBSTITUTIONS, "block")


def kind(unit: str) -> str:
    return "block" if unit == "block" else "substitution" if unit in SUBSTITUTIONS else "component"


def count_per_model(unit: str, m: ModelSpec) -> int:
    unit = SUBSTITUTIONS.get(unit, (None, unit))[1]
    if unit == "ln":
        return 2 * m.layers + 1  # two per block + final norm, pooled
    if unit in ("embed_lookup", "out_proj", "loss"):
        return 1
    return m.layers  # attn_proj, rope, attn_core, mlp, block


def _module(unit: str, m: ModelSpec, seq: int) -> nn.Module:
    return {
        "embed_lookup": lambda: nn.Embedding(m.vocab, m.d),
        "ln": lambda: nn.LayerNorm(m.d, eps=m.eps),
        "attn_proj": lambda: C.AttnProj(m),
        "rope": lambda: C.Rope(m, seq),
        "attn_core": lambda: C.AttnCore(m, seq),
        "mlp": lambda: C.MLP(m),
        "out_proj": lambda: nn.Linear(m.d, m.vocab, bias=False),
        "loss": lambda: C.Loss(),
        "block": lambda: C.Block(m, seq),
        "sdpa_core": lambda: S.SdpaCore(),
        "linear_core": lambda: S.LinearCore(),
        "gqa_proj": lambda: S.GqaProj(m),
        "factorized_embed_in": lambda: S.FactorizedEmbedIn(m),
        "factorized_embed_out": lambda: S.FactorizedEmbedOut(m),
        "rmsnorm": lambda: S.RMSNorm(m.d, m.eps),
        "rmsnorm_fused": lambda: S.RMSNormFused(m.d, m.eps),
        "rope_fused": lambda: S.RopeFused(m, seq),
    }[unit]()


def make_inputs(unit: str, run: RunSpec, device) -> tuple[torch.Tensor, ...]:
    m, b, s = run.model, run.batch, run.seq_len

    def act(*shape):
        return torch.randn(*shape, device=device, dtype=run.dtype, requires_grad=True)

    def ids():
        return torch.randint(0, m.vocab, (b, s), device=device)

    heads = (b, m.heads, s, m.head_dim)
    if unit in ("embed_lookup", "factorized_embed_in"):
        return (ids(),)
    if unit in ("attn_proj", "gqa_proj"):
        return act(b, s, m.d), act(b, s, m.d)
    if unit in ("rope", "rope_fused"):
        return act(*heads), act(*heads)
    if unit in ("attn_core", "sdpa_core", "linear_core"):
        return act(*heads), act(*heads), act(*heads)
    if unit == "loss":
        return act(b, s, m.vocab), ids()
    return (act(b, s, m.d),)  # ln, rmsnorm(_fused), mlp, out_proj, factorized_embed_out, block


def build(unit: str, run: RunSpec, device="cpu"):
    """Return (module, inputs, grad_outputs); grad_outputs is None for the scalar loss."""
    with torch.device(device):
        mod = _module(unit, run.model, run.seq_len).to(run.dtype)
    inputs = make_inputs(unit, run, device)
    with torch.no_grad():
        out = mod(*inputs)
    outs = out if isinstance(out, tuple) else (out,)
    grads = None if unit == "loss" else tuple(torch.randn_like(o) for o in outs)
    return mod, inputs, grads


def step(mod: nn.Module, inputs, grads) -> None:
    """One fwd + bwd, then drop grads so every repetition does identical work."""
    out = mod(*inputs)
    torch.autograd.backward(out if isinstance(out, tuple) else (out,), grads)
    for t in (*mod.parameters(), *inputs):
        t.grad = None


def param_count(m: ModelSpec) -> tuple[int, int]:
    """(total, non-embedding) parameters of the full model assembled from our units."""
    def n(unit):
        with torch.device("meta"):
            return sum(p.numel() for p in _module(unit, m, 1).parameters())

    non_embed = m.layers * n("block") + n("ln")  # + final norm
    return non_embed + n("embed_lookup") + n("out_proj"), non_embed
