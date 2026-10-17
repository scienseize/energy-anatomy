import random
from collections import defaultdict

import numpy as np
import pytest

from anatomy import analysis as A
from anatomy.config import DATA

T, F = True, False


@pytest.mark.parametrize("cases, both, verdict", [
    ([(T, 0.1, 0.01)] * 10 + [(T, -0.1, 0.01)] * 8, 10, "supported"),  # 10/18 > half
    ([(T, 0.1, 0.01)] * 9 + [(T, -0.1, 0.01)] * 9, 9, "not supported"),  # 9/18 is not a majority
    ([(T, 0.01, 0.01)] * 18, 0, "not supported"),  # median == IQR fails the strict rule
    ([(T, 0.01, 0.02)] * 18, 0, "not supported"),  # positive but inside the spread
    ([(F, 0.5, 0.01)] * 18, 0, "not tested"),  # never memory-bound
    ([(F, -0.5, 0.01)] * 16 + [(T, 0.1, 0.01)] * 2, 2, "supported"),  # only memory-bound configs count
])
def test_decision_rule(cases, both, verdict):
    r = A.decide(cases)
    assert (r["both"], r["verdict"]) == (both, verdict)


def test_iqr_is_linear_type7():
    assert A.iqr(range(1, 11)) == pytest.approx(4.5)  # Q1 = 3.25, Q3 = 7.75


def test_spearman():
    assert A.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert A.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert A.spearman([1, 1, 2], [1, 1, 2]) == pytest.approx(1)  # ties get average ranks


def _row(run_id, unit, j, ts="2026-01-01T00:00:00"):
    return {"run_id": run_id, "ts": ts, "model": "m", "seq_len": "512", "precision": "fp32", "unit": unit,
            "pass_idx": str(j)}


def test_select_session_takes_the_complete_run():
    partial = [_row("a", "u", 0)]
    full = [_row("b", u, j, "2026-01-02T00:00:00") for u in ("u", "v") for j in range(2)]
    kept = A.select_session(partial + full, ("u", "v"), 2)
    assert {r["run_id"] for r in kept} == {"b"}
    with pytest.raises(ValueError):
        A.select_session(partial, ("u", "v"), 2)


@pytest.fixture(scope="module")
def measured():
    raw = DATA / "raw" / "passes.csv"
    if not raw.exists():
        pytest.skip("no measured data")
    an = A._read(DATA / "derived" / "analytical.csv")
    passes = A.select_session(A._read(raw), (*A.COMPONENTS, "block"), 10)
    return passes, A._read(DATA / "raw" / "idle.csv"), an


def test_share_identities(measured):
    """Per config and pass: sum C3 + Unattributed = 1, sum C4 = -Unattributed, component-normalized C3 sums to 1."""
    pp = A.per_pass(*measured, 10)
    g = defaultdict(dict)
    for r in pp:
        g[(r["model"], r["seq_len"], r["precision"], r["pass_idx"], r["energy"])][r["component"]] = r
    assert len(g) == 18 * 10 * 2
    for rows in g.values():
        u = rows.pop("unattributed")["c3"]
        assert sum(r["c3"] for r in rows.values()) + u == pytest.approx(1)
        assert sum(r["c4"] for r in rows.values()) == pytest.approx(-u)
        assert sum(r["c3_components"] for r in rows.values()) == pytest.approx(1)


def test_summary_invariant_to_row_order(measured):
    passes, idle, an = measured
    base = A.summarize(A.per_pass(passes, idle, an, 10), an, passes)
    shuffled = passes[:]
    random.Random(0).shuffle(shuffled)
    again = A.summarize(A.per_pass(shuffled, idle, an, 10), an, shuffled)
    assert [r["rank_c4"] for r in base] == [r["rank_c4"] for r in again]
    assert np.allclose([r["c3_med"] for r in base], [r["c3_med"] for r in again])


def test_substitution_delta_by_hand():
    """One config, one pass: E_total = L*block + embed + ln + out_proj + loss; swapping ln (count 2L+1)
    for rmsnorm changes it by (2L+1) * (E_ln - E_rms). Substitutions not measured are skipped."""
    L = 2
    energy = {"block": 10.0, "embed_lookup": 1.0, "ln": 0.5, "attn_proj": 2.0, "rope": 0.2, "attn_core": 3.0,
              "mlp": 4.0, "out_proj": 2.0, "loss": 1.0, "rmsnorm": 0.75}
    count = {"ln": 2 * L + 1, "rmsnorm": 2 * L + 1, "block": L, "embed_lookup": 1, "out_proj": 1, "loss": 1,
             "attn_proj": L, "rope": L, "attn_core": L, "mlp": L}
    passes = [{"run_id": "r", "ts": "t", "model": "m", "seq_len": "512", "precision": "fp32", "unit": u,
               "pass_idx": "0", "status": "ok", "reps": "1", "time_s": "1", "energy_j": str(v)} for u, v in energy.items()]
    analytical = [{"model": "m", "seq_len": "512", "precision": "fp32", "component": u, "count_per_model": str(n)}
                  for u, n in count.items()]
    idle = [{"run_id": "r", "ts": "t", "avg_power_w": "0"}]
    row = next(r for r in A.substitution_deltas(passes, idle, analytical, 1) if r["substitution"] == "rmsnorm")
    base = L * 10.0 + 1.0 + 0.5 + 2.0 + 1.0
    assert row["substitution"] == "rmsnorm" and row["e_total_base_j"] == pytest.approx(base)
    assert row["e_total_sub_j"] == pytest.approx(base + (2 * L + 1) * 0.25)
    assert row["delta_pct"] == pytest.approx(-(2 * L + 1) * 0.25 / base * 100)  # negative = costs more


def test_quadrant_split():
    """Decision 30: waste split at C4 = 0 (strict), capability at the median rank, median rank in the lower half."""
    assert A.quadrant(0.01, 3, 4.0) == "high waste, high capability"
    assert A.quadrant(0.01, 4, 4.0) == "high waste, low capability"  # median rank -> lower-capability half
    assert A.quadrant(0.0, 1, 4.0) == "low waste, high capability"  # C4 = 0 is not waste
    assert A.quadrant(-0.2, 7, 4.0) == "low waste, low capability"


def test_rank_shift_sign():
    """shift = rank_from - rank_to: a component moving from rank 5 to rank 2 climbs by +3 (more waste)."""
    summary = [{"model": m, "seq_len": 512, "precision": "fp32", "component": c, "rank_c4": r}
               for m, ranks in (("pythia-160m", {"ln": 5, "mlp": 2}), ("pythia-6.9b", {"ln": 2, "mlp": 5}))
               for c, r in ranks.items()]
    shifts = {s["component"]: s["shift"] for s in A.rank_shifts(summary) if s["moderator"] == "scale"}
    assert shifts == {"ln": 3, "mlp": -3}


def test_removal_substitution_costs_nothing():
    """NoPE removes rope: substituted total = baseline - rope energy x count, with no substitute energy."""
    L = 2
    energy = {"block": 10.0, "embed_lookup": 1.0, "ln": 0.5, "attn_proj": 2.0, "rope": 0.2, "attn_core": 3.0,
              "mlp": 4.0, "out_proj": 2.0, "loss": 1.0}
    count = {"ln": 2 * L + 1, "block": L, "embed_lookup": 1, "out_proj": 1, "loss": 1, "attn_proj": L, "rope": L,
             "attn_core": L, "mlp": L}
    passes = [{"run_id": "r", "ts": "t", "model": "m", "seq_len": "512", "precision": "fp32", "unit": u,
               "pass_idx": "0", "status": "ok", "reps": "1", "time_s": "1", "energy_j": str(v)} for u, v in energy.items()]
    analytical = [{"model": "m", "seq_len": "512", "precision": "fp32", "component": u, "count_per_model": str(n)}
                  for u, n in count.items()]
    rows = {r["substitution"]: r for r in A.substitution_deltas(passes, [{"run_id": "r", "ts": "t", "avg_power_w": "0"}],
                                                                analytical, 1)}
    base = L * 10.0 + 1.0 + 0.5 + 2.0 + 1.0
    assert set(rows) == {"nope"}  # measured substitutions absent from this session are skipped
    assert rows["nope"]["e_total_sub_j"] == pytest.approx(base - L * 0.2)
