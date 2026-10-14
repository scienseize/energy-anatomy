"""Analytical model: FLOPs and bytes per unit (fwd + bwd), C1, ridge, memory-bound, C2, t_hat.

Counting runs the real module on the meta device under a dispatch mode, so it sees exactly
the aten ops autograd issues (no hand-enumeration). Per op:
  bytes = reads + writes of every tensor it touches at the run dtype, no reuse between ops
          (DESIGN §7 upper bound; op-level, so the extra passes of a multi-kernel op are not counted).
  FLOPs = 2mkn for matmuls (bwd is exactly 2x fwd via the traced mm ops); elementwise ops use
          the per-element constants in EW_FLOPS (convention, applied to the largest operand).
"""
import csv
import math
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from anatomy.config import DATA, RunSpec, load_grid, load_yaml
from anatomy.units import COMPONENTS, SUBSTITUTIONS, UNITS, build, count_per_model, step

aten = torch.ops.aten

EW_FLOPS = {
    aten.add: 1, aten.sub: 1, aten.mul: 1, aten.div: 1, aten.neg: 1, aten.where: 1,
    aten.pow: 1, aten.rsqrt: 1, aten.sum: 1, aten.mean: 1, aten.cumsum: 1,
    aten.elu: 2, aten.elu_backward: 2,
    aten.gelu: 5, aten.gelu_backward: 10,  # 0.5x(1+erf(x/√2)); dy(Φ + xφ)
    aten._softmax: 5, aten._softmax_backward_data: 4,  # max, sub, exp, sum, div; P(dP - Σ P dP)
    aten._log_softmax: 5, aten._log_softmax_backward_data: 4,
    aten.native_layer_norm: 8,  # mean, var (3), normalize (2), affine (2)
    aten.native_layer_norm_backward: 15,  # dx (10) + dgamma, dbeta (5)
    aten.nll_loss_forward: 1, aten.nll_loss_backward: 1,  # per target
    aten.embedding_dense_backward: 1,  # one add per gathered element
    # data movement only
    aten.embedding: 0, aten.cat: 0, aten.clone: 0, aten.copy_: 0, aten.flip: 0,
    aten.slice_backward: 0, aten.ones_like: 0, aten.scalar_tensor: 0,
}
MM = {aten.mm, aten.addmm, aten.bmm}
ATTN = {aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_efficient_attention_backward}
FREE = {aten._unsafe_view, aten.new_empty_strided}  # aliasing / allocation only
WRITE_ONLY = {aten.ones_like, aten.scalar_tensor}


@dataclass
class Count:
    mm_flops: float = 0.0
    ew_flops: float = 0.0
    bytes: float = 0.0

    @property
    def flops(self) -> float:
        return self.mm_flops + self.ew_flops


def _nbytes(t: torch.Tensor) -> int:
    # An expanded tensor only reads its storage; a slice only reads its own elements.
    n = min(t.numel(), t.untyped_storage().nbytes() // t.element_size())
    return n * t.element_size()


def _named(func, args, kwargs) -> dict:
    named = dict(zip((a.name for a in func._schema.arguments), args))
    named.update(kwargs)
    return named


class Counter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.c = Count()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        op = func.overloadpacket
        if func.is_view or op in FREE:
            return out
        a = _named(func, args, kwargs)
        ins = [t for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)]
        outs = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]

        if op in MM:
            x, w = (a["self"], a["mat2"]) if op is not aten.addmm else (a["mat1"], a["mat2"])
            *batch, m, k = x.shape
            self.c.mm_flops += 2 * math.prod(batch) * m * k * w.shape[-1]
            if op is aten.addmm:
                self.c.mm_flops += m * w.shape[-1]  # fused bias epilogue
        elif op in ATTN:
            b, h, sq, hd = a["query"].shape
            f = 4 * b * h * sq * a["key"].shape[2] * hd * (0.5 if a.get("is_causal") else 1.0)
            self.c.mm_flops += f if op is aten._scaled_dot_product_efficient_attention else 2.5 * f
        else:
            if op not in EW_FLOPS:
                raise KeyError(f"no FLOP rule for {func}; add it to EW_FLOPS")
            if op in (aten.nll_loss_forward, aten.nll_loss_backward):
                n = a["target"].numel()
            elif op is aten.embedding_dense_backward:
                n = a["grad_output"].numel()
            else:
                n = max(t.numel() for t in ins + outs)
            self.c.ew_flops += EW_FLOPS[op] * n

        if op in WRITE_ONLY:
            ins = []
        elif op is aten.nll_loss_backward:
            ins = [a["grad_output"], a["target"]]  # `self` passes the shape only
        elif op is aten.copy_:
            ins = [a["src"]]
        self.c.bytes += sum(map(_nbytes, ins)) + sum(map(_nbytes, outs))
        return out


def count(unit: str, run: RunSpec) -> Count:
    mod, inputs, grads = build(unit, run, "meta")
    with Counter() as ctr:
        step(mod, inputs, grads)
    return ctr.c


def roofline(c: Count, precision: str, hw: dict) -> dict:
    """Effective peak (decided): P_eff = F / Σ F_k/P_k, ridge = P_eff/BW, t_hat = max(F/P_eff, B/BW).
    Matmuls use the tensor-core peak only in FP16; everything else the CUDA-core peak."""
    peak = hw["peak_flops"]
    p_mm = peak["fp16_tensor"] if precision == "fp16" else peak["fp32"]
    p_ew = peak[precision]
    t_compute = c.mm_flops / p_mm + c.ew_flops / p_ew
    p_eff = c.flops / t_compute
    ridge = p_eff / hw["bandwidth"]
    c1 = c.flops / c.bytes
    return {"c1": c1, "ridge": ridge, "memory_bound": c1 < ridge,
            "t_hat": max(t_compute, c.bytes / hw["bandwidth"])}


FIELDS = ["model", "seq_len", "precision", "component", "count_per_model", "flops", "bytes",
          "c1", "ridge", "memory_bound", "c2", "t_hat"]


def rows(runs=None, hw=None) -> list[dict]:
    runs = runs or load_grid()
    hw = hw or load_yaml("hardware/t4.yaml")
    out = []
    for run in runs:
        m = run.model
        per = {}
        for u in UNITS:
            c = count(u, run)
            n = count_per_model(u, m)
            per[u] = {"model": m.name, "seq_len": run.seq_len, "precision": run.precision,
                      "component": u, "count_per_model": n, "flops": c.flops, "bytes": c.bytes,
                      **roofline(c, run.precision, hw), "c2": None}
        total = lambda us: sum(per[u]["flops"] * per[u]["count_per_model"] for u in us)
        base = total(COMPONENTS)
        for u in COMPONENTS:
            per[u]["c2"] = per[u]["flops"] * per[u]["count_per_model"] / base
        for u, (sub, _) in SUBSTITUTIONS.items():
            group = [v for v, (s, _) in SUBSTITUTIONS.items() if s == sub]
            replaced = [SUBSTITUTIONS[v][1] for v in group]
            # C2 within the model that has this substitution swapped in
            per[u]["c2"] = per[u]["flops"] * per[u]["count_per_model"] / (base - total(replaced) + total(group))
        out += per.values()
    return out


def write(path=DATA / "derived" / "analytical.csv") -> list[dict]:
    rs = rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rs)
    return rs
