"""Load and validate configs/ into ModelSpec and RunSpec."""
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    d: int
    heads: int
    layers: int
    ffn: int
    vocab: int
    rotary_pct: float
    rotary_base: float
    eps: float

    @property
    def head_dim(self) -> int:
        return self.d // self.heads

    @property
    def rotary_dims(self) -> int:
        return int(self.head_dim * self.rotary_pct)


@dataclass(frozen=True)
class RunSpec:
    model: ModelSpec
    seq_len: int
    precision: str
    tokens: int

    @property
    def batch(self) -> int:
        return self.tokens // self.seq_len

    @property
    def dtype(self) -> torch.dtype:
        return DTYPES[self.precision]


def load_model(name: str) -> ModelSpec:
    c = json.loads((CONFIGS / "pythia" / f"{name}.json").read_text())
    # Our modules implement exactly this GPT-NeoX variant; refuse anything else.
    assert c["use_parallel_residual"] and not c["tie_word_embeddings"], name
    assert c["hidden_act"] == "gelu" and c["attention_bias"], name
    assert c["hidden_size"] % c["num_attention_heads"] == 0, name
    return ModelSpec(
        name=name,
        d=c["hidden_size"],
        heads=c["num_attention_heads"],
        layers=c["num_hidden_layers"],
        ffn=c["intermediate_size"],
        vocab=c["vocab_size"],
        rotary_pct=c["rotary_pct"],
        rotary_base=c["rotary_emb_base"],
        eps=c["layer_norm_eps"],
    )


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((CONFIGS / rel).read_text())


def load_grid() -> list[RunSpec]:
    g = load_yaml("grid.yaml")
    runs = []
    for name in g["models"]:
        m = load_model(name)
        for seq in g["seq_lens"]:
            assert g["tokens_per_pass"] % seq == 0, f"tokens_per_pass not divisible by {seq}"
            for prec in g["precisions"]:
                assert prec in DTYPES, prec
                runs.append(RunSpec(m, seq, prec, g["tokens_per_pass"]))
    return runs
