import csv
import json

import pytest

from anatomy.config import DATA
from anatomy.export import _value, build


def test_value_parsing():
    assert [_value(v) for v in ("", "True", "3", "0.25", "1e-07", "fp32")] == [None, True, 3, 0.25, 1e-07, "fp32"]


def test_export_copies_derived_values_without_computing():
    if not (DATA / "derived" / "shares_summary.csv").exists():
        pytest.skip("run anatomy analyze first")
    out = build()
    json.dumps(out, allow_nan=False)  # valid JSON, no NaN
    assert len(out["configs"]) == 18 and len(out["profile"]) == 18
    with open(DATA / "derived" / "shares_summary.csv") as f:
        row = next(r for r in csv.DictReader(f) if r["component"] == "attn_core")
    cid = f"{row['model']}/{row['seq_len']}/{row['precision']}"
    prof = next(p for p in out["profile"] if p["config"] == cid)["components"]["attn_core"]
    rank = next(p for p in out["ranking"]["per_config"] if p["config"] == cid)["components"]["attn_core"]
    assert prof["c3_med"] == float(row["c3_med"]) and rank["c4_med"] == float(row["c4_med"])
