import math

import pytest
import torch

from anatomy.analytical import count, rows
from anatomy.config import RunSpec, load_grid, load_model
from anatomy.units import COMPONENTS, SUBSTITUTIONS, UNITS, build, param_count, step

PUBLISHED = {  # (total, non-embedding)
    "pythia-160m": (162_322_944, 85_056_000),
    "pythia-1.4b": (1_414_647_808, 1_208_602_624),
    "pythia-6.9b": (6_857_302_016, 6_444_163_072),
}
M160 = load_model("pythia-160m")


@pytest.mark.parametrize("name", PUBLISHED)
def test_param_counts_match_published(name):
    assert param_count(load_model(name)) == PUBLISHED[name]


def test_grid():
    runs = load_grid()
    assert len(runs) == 18 and all(r.batch * r.seq_len == 8192 for r in runs)


def test_linear_flops_and_bytes_by_hand():
    run = RunSpec(M160, 512, "fp32", 8192)
    n, d, v, p = 8192, M160.d, M160.vocab, 4
    c = count("out_proj", run)  # no bias: fwd mm, bwd dX mm, bwd dW mm
    assert c.mm_flops == 3 * 2 * n * d * v and c.ew_flops == 0
    fwd = (n * d + d * v + n * v) * p
    dx = (n * v + v * d + n * d) * p
    dw = (n * v + n * d + v * d) * p
    assert c.bytes == fwd + dx + dw


def test_layernorm_flops_and_bytes_by_hand():
    run = RunSpec(M160, 512, "fp32", 8192)
    n, d, p = 8192, M160.d, 4
    c = count("ln", run)
    assert c.mm_flops == 0 and c.ew_flops == (8 + 15) * n * d
    fwd = (n * d + 2 * d) * p + (n * d + 2 * n) * p  # x, gamma, beta -> y, mean, rstd
    bwd = (2 * n * d + 2 * n + 2 * d) * p + (n * d + 2 * d) * p  # dy, x, mean, rstd, gamma, beta -> dx, dgamma, dbeta
    assert c.bytes == fwd + bwd


@pytest.fixture(scope="module")
def table():
    return rows()


def test_c2_sums_to_one(table):
    for run in load_grid():
        key = (run.model.name, run.seq_len, run.precision)
        sel = [r for r in table if (r["model"], r["seq_len"], r["precision"]) == key]
        assert len(sel) == len(UNITS)
        assert math.isclose(sum(r["c2"] for r in sel if r["component"] in COMPONENTS), 1.0)


def test_memory_bound_flags(table):
    by = lambda c: [r["memory_bound"] for r in table if r["component"] == c]
    for c in ("embed_lookup", "ln", "rope", "loss", "rmsnorm"):
        assert all(by(c)), c
    for c in ("attn_proj", "mlp", "out_proj"):
        assert not any(by(c)), c


def test_row_order_invariant(table):
    key = lambda r: (r["model"], r["seq_len"], r["precision"], r["component"])
    assert sorted(table, key=key) == sorted(reversed(table), key=key)


def tiny(seq=128):
    return RunSpec(M160, seq, "fp32", 2 * seq)


@pytest.mark.parametrize("unit", UNITS)
def test_every_unit_steps_on_cpu(unit):
    mod, inputs, grads = build(unit, tiny(), "cpu")
    step(mod, inputs, grads)
    assert all(p.grad is None for p in mod.parameters())


def test_block_matches_hf_gptneox_layer():
    from transformers import GPTNeoXConfig
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXRotaryEmbedding

    from anatomy.components import Block

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(hidden_size=768, num_attention_heads=12, intermediate_size=3072, rotary_pct=0.25,
                        rotary_emb_base=10000, layer_norm_eps=1e-5, use_parallel_residual=True, hidden_act="gelu",
                        attn_implementation="eager")
    hf = GPTNeoXLayer(cfg, layer_idx=0).eval()
    s = 64
    ours = Block(M160, s).eval()
    ours.load_state_dict(hf.state_dict())
    x = torch.randn(2, s, 768)
    pos = torch.arange(s)[None]
    mask = torch.full((s, s), torch.finfo(x.dtype).min).triu(1)[None, None]
    with torch.no_grad():
        ref = hf(x, attention_mask=mask, position_ids=pos, position_embeddings=GPTNeoXRotaryEmbedding(cfg)(x, pos))
        torch.testing.assert_close(ours(x), ref[0] if isinstance(ref, tuple) else ref, rtol=1e-4, atol=1e-4)


def test_sdpa_core_matches_attn_core():
    run = tiny()
    from anatomy.components import AttnCore
    from anatomy.substitutions import SdpaCore

    q, k, v = (torch.randn(2, M160.heads, 128, M160.head_dim) for _ in range(3))
    torch.testing.assert_close(SdpaCore()(q, k, v), AttnCore(M160, run.seq_len)(q, k, v), rtol=1e-4, atol=1e-5)


def test_linear_core_matches_quadratic_reference():
    import torch.nn.functional as F

    from anatomy.substitutions import LinearCore

    b, h, s, hd = 2, 3, 256, 16
    q, k, v = (torch.randn(b, h, s, hd, dtype=torch.float64) for _ in range(3))
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    a = (fq @ fk.transpose(-1, -2)).tril()
    ref = ((a @ v) / (a.sum(-1, keepdim=True) + 1e-6)).transpose(1, 2).reshape(b, s, h * hd)
    torch.testing.assert_close(LinearCore()(q, k, v), ref)


def test_substitution_shapes_and_gqa_heads():
    run = tiny()
    for sub, (_, base) in SUBSTITUTIONS.items():
        ms, xs, _ = build(sub, run)
        mb, xb, _ = build(base, run)
        torch.manual_seed(0)
        out_s = ms(*xs)
        out_b = mb(*xb)
        flat = lambda o: o if isinstance(o, tuple) else (o,)
        # gqa_proj returns q, k, v separately; attn_proj returns fused qkv
        assert sum(t.numel() for t in flat(out_s)) == sum(t.numel() for t in flat(out_b)), sub
    for name, kvh in (("pythia-160m", 3), ("pythia-1.4b", 4), ("pythia-6.9b", 8)):
        from anatomy.substitutions import GqaProj

        with torch.device("meta"):
            assert GqaProj(load_model(name)).kvh == kvh
