"""Analysis (DESIGN §8): per-pass shares C3, waste C4 = C3 - C2, the H1 decision rule, and the
analytical-vs-measured validation. Writes data/derived/shares*.csv and tables T4-T7.

Conventions (decided 2026-10-17):
  IQR = Q3 - Q1 with linear (Hyndman-Fan type 7) quantiles, numpy's default.
  T4 compares analytical C3 with measured C3 renormalized over the 8 components (no Unattributed).
  Dynamic energy subtracts the nearest preceding idle window's power x window time.
"""
import csv
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

from anatomy.config import DATA
from anatomy.report import write_table
from anatomy.units import COMPONENTS

RAW, DERIVED = DATA / "raw", DATA / "derived"
TESTED = ("embed_lookup", "ln", "rope", "attn_core")  # H1 components; ln is pooled with the final norm
USABLE = ("ok", "short_window")  # short windows are kept (flagged); oom rows carry no energy
MODEL_ORDER = ("pythia-160m", "pythia-1.4b", "pythia-6.9b")
COMP_ORDER = (*COMPONENTS, "unattributed")


def grid_order(r: dict) -> tuple:
    """Rows in grid order (model size, seq, fp32 before fp16, component order) for readable tables."""
    return (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99, int(r["seq_len"]),
            r["precision"] != "fp32", COMP_ORDER.index(r["component"]) if "component" in r else 0)


def iqr(x) -> float:
    q1, q3 = np.percentile(x, [25, 75])  # linear interpolation
    return float(q3 - q1)


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def select_session(rows: list[dict], units, passes: int) -> list[dict]:
    """Per config keep the one run id that covers every unit x pass (configs are never split
    across sessions); the latest such run id if a config was deliberately rerun."""
    by = defaultdict(lambda: defaultdict(set))  # config -> run_id -> {(unit, pass)}
    last_ts = {}
    for r in rows:
        cfg = (r["model"], int(r["seq_len"]), r["precision"])
        if r["unit"] in units:
            by[cfg][r["run_id"]].add((r["unit"], int(r["pass_idx"])))
        last_ts[r["run_id"]] = max(last_ts.get(r["run_id"], ""), r["ts"])
    need = {(u, j) for u in units for j in range(passes)}
    keep = {}
    for cfg, runs in by.items():
        full = [rid for rid, got in runs.items() if need <= got]
        if not full:
            raise ValueError(f"no session covers every unit and pass for {cfg}")
        keep[cfg] = max(full, key=last_ts.get)
    return [r for r in rows if keep.get((r["model"], int(r["seq_len"]), r["precision"])) == r["run_id"]]


def _idle_lookup(idle: list[dict]):
    """ts -> power of the nearest idle window at or before ts, per run id."""
    per = defaultdict(list)
    for r in idle:
        per[r["run_id"]].append((r["ts"], float(r["avg_power_w"])))
    for v in per.values():
        v.sort()

    def power(run_id: str, ts: str) -> float:
        v = per[run_id]
        i = bisect_right([t for t, _ in v], ts) - 1
        return v[max(i, 0)][1]

    return power


def per_pass(passes: list[dict], idle: list[dict], analytical: list[dict], n_passes: int) -> list[dict]:
    """One row per (config, component|unattributed, pass): C3 and C4 on total and dynamic energy.
    Pass j of every unit in a config forms one matched set (DESIGN §8)."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    idle_w = _idle_lookup(idle)
    e = defaultdict(dict)  # (cfg, unit) -> pass -> (total J/rep, dynamic J/rep)
    for r in passes:
        if r["status"] not in USABLE:
            continue
        reps, t, joules = int(r["reps"]), float(r["time_s"]), float(r["energy_j"])
        dyn = joules - idle_w(r["run_id"], r["ts"]) * t
        cfg = (r["model"], int(r["seq_len"]), r["precision"])
        e[(cfg, r["unit"])][int(r["pass_idx"])] = (joules / reps, dyn / reps)

    out = []
    for cfg in sorted({c for c, _ in e}):
        count = {c: int(an[(*cfg, c)]["count_per_model"]) for c in COMPONENTS}
        layers = int(an[(*cfg, "block")]["count_per_model"])
        for j in range(n_passes):
            units = (*COMPONENTS, "block")
            if any(j not in e[(cfg, u)] for u in units):
                continue  # incomplete matched set (oom): drop pass j for this config
            for k, kind in ((0, ""), (1, "_dyn")):
                ec = {c: e[(cfg, c)][j][k] * count[c] for c in COMPONENTS}
                # final norm = one ln instance; the other 2L ln instances live inside the blocks
                total = layers * e[(cfg, "block")][j][k] + sum(
                    e[(cfg, c)][j][k] for c in ("embed_lookup", "ln", "out_proj", "loss"))
                for c in COMPONENTS:
                    c3 = ec[c] / total
                    out.append({"model": cfg[0], "seq_len": cfg[1], "precision": cfg[2], "component": c,
                                "pass_idx": j, "energy": kind or "total", "c3": c3,
                                "c4": c3 - float(an[(*cfg, c)]["c2"]),
                                "c3_components": ec[c] / sum(ec.values())})
                out.append({"model": cfg[0], "seq_len": cfg[1], "precision": cfg[2], "component": "unattributed",
                            "pass_idx": j, "energy": kind or "total", "c3": 1 - sum(ec.values()) / total,
                            "c4": None, "c3_components": None})
    return out


def summarize(pp: list[dict], analytical: list[dict], passes: list[dict]) -> list[dict]:
    """Per (config, component): medians and IQRs over passes, plus C1, C2, memory-bound,
    energy per FLOP, measured power and rank by median C4 within the config."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    g = defaultdict(list)
    for r in pp:
        g[(r["model"], r["seq_len"], r["precision"], r["component"], r["energy"])].append(r)
    epr, power = defaultdict(list), defaultdict(list)
    for r in passes:
        if r["status"] in USABLE:
            k = (r["model"], int(r["seq_len"]), r["precision"], r["unit"])
            epr[k].append(float(r["energy_per_rep_j"]))
            power[k].append(float(r["avg_power_w"]))
    out = []
    for (m, s, p, c, kind), rs in sorted(g.items()):
        if kind != "total":
            continue
        dyn = g[(m, s, p, c, "_dyn")]
        c3 = [r["c3"] for r in rs]
        row = {"model": m, "seq_len": s, "precision": p, "component": c,
               "n_passes": len(rs), "c3_med": float(np.median(c3)), "c3_iqr": iqr(c3),
               "c3_dyn_med": float(np.median([r["c3"] for r in dyn])), "c3_dyn_iqr": iqr([r["c3"] for r in dyn])}
        if c == "unattributed":
            row.update(c1=None, c2=None, memory_bound=None, c4_med=None, c4_iqr=None, c4_dyn_med=None, c4_dyn_iqr=None,
                       c3_components_med=None, c4_components_med=None, c4_components_iqr=None, energy_per_flop_j=None, power_w_med=None, rank_c4=None)
        else:
            a = an[(m, s, p, c)]
            c4 = [r["c4"] for r in rs]
            row.update(c1=float(a["c1"]), c2=float(a["c2"]), memory_bound=a["memory_bound"] == "True",
                       c4_med=float(np.median(c4)), c4_iqr=iqr(c4),
                       c4_dyn_med=float(np.median([r["c4"] for r in dyn])), c4_dyn_iqr=iqr([r["c4"] for r in dyn]),
                       c3_components_med=float(np.median([r["c3_components"] for r in rs])),
                       c4_components_med=float(np.median([r["c3_components"] - float(a["c2"]) for r in rs])),
                       c4_components_iqr=iqr([r["c3_components"] - float(a["c2"]) for r in rs]),
                       energy_per_flop_j=float(np.median(epr[(m, s, p, c)])) / float(a["flops"]),
                       power_w_med=float(np.median(power[(m, s, p, c)])))
        out.append(row)
    for cfg in {(r["model"], r["seq_len"], r["precision"]) for r in out}:
        rows = sorted((r for r in out if (r["model"], r["seq_len"], r["precision"]) == cfg and r["c4_med"] is not None),
                      key=lambda r: -r["c4_med"])
        for i, r in enumerate(rows, 1):
            r["rank_c4"] = i
    return sorted(out, key=grid_order)


def decide(cases: list[tuple[bool, float, float]]) -> dict:
    """Decision rule for one component. cases: (memory_bound, c4_med, c4_iqr) per config.
    Supported if, among memory-bound configs, a strict majority has c4_med > 0 and c4_med > IQR."""
    mb = [(m, q) for bound, m, q in cases if bound]
    pos = sum(m > 0 for m, _ in mb)
    above = sum(m > q for m, q in mb)
    both = sum(m > 0 and m > q for m, q in mb)
    verdict = "not tested" if not mb else "supported" if both > len(mb) / 2 else "not supported"
    return {"configs": len(cases), "memory_bound": len(mb), "c4_pos": pos, "c4_above_iqr": above,
            "both": both, "verdict": verdict}


def spearman(a, b) -> float:
    """Spearman rho with average ranks for ties."""
    def ranks(x):
        x = np.asarray(x, dtype=float)
        order = x.argsort()
        r = np.empty(len(x))
        r[order] = np.arange(1, len(x) + 1)
        for v in np.unique(x):
            r[x == v] = r[x == v].mean()
        return r

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def validate(summary: list[dict], analytical: list[dict]) -> list[dict]:
    """T4: analytical C3 (t_hat x count, weighted by the measured median power of the component's
    roofline class in that config, normalized over the 8 components) vs measured C3 over components."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    out = []
    for cfg in dict.fromkeys((r["model"], r["seq_len"], r["precision"]) for r in summary):
        rows = {r["component"]: r for r in summary if (r["model"], r["seq_len"], r["precision"]) == cfg}
        cls_power = {}
        for bound in (True, False):
            ws = [rows[c]["power_w_med"] for c in COMPONENTS if rows[c]["memory_bound"] == bound]
            cls_power[bound] = float(np.mean(ws)) if ws else float(np.mean([rows[c]["power_w_med"] for c in COMPONENTS]))
        est = {c: float(an[(*cfg, c)]["t_hat"]) * int(an[(*cfg, c)]["count_per_model"])
               * cls_power[rows[c]["memory_bound"]] for c in COMPONENTS}
        tot = sum(est.values())
        a = [est[c] / tot for c in COMPONENTS]
        m = [rows[c]["c3_components_med"] for c in COMPONENTS]
        rho = spearman(a, m)
        for c, x, y in zip(COMPONENTS, a, m):
            out.append({"model": cfg[0], "seq_len": cfg[1], "precision": cfg[2], "component": c,
                        "memory_bound": rows[c]["memory_bound"], "c3_analytical": x, "c3_measured": y,
                        "abs_diff": abs(x - y), "spearman_rho": rho})
    return out


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(rows: list[dict], digits: int = 4) -> list[dict]:
    def f(v):
        return f"{v:.{digits}g}" if isinstance(v, float) else "" if v is None else v
    return [{k: f(v) for k, v in r.items()} for r in rows]


def run(passes_csv: Path = RAW / "passes.csv", idle_csv: Path = RAW / "idle.csv",
        analytical_csv: Path = DERIVED / "analytical.csv", n_passes: int = 10) -> dict:
    analytical = _read(analytical_csv)
    passes = select_session(_read(passes_csv), (*COMPONENTS, "block"), n_passes)
    pp = per_pass(passes, _read(idle_csv), analytical, n_passes)
    summary = summarize(pp, analytical, passes)
    _write(DERIVED / "shares.csv", [r for r in pp if r["energy"] == "total"])
    _write(DERIVED / "shares_dynamic.csv", [r for r in pp if r["energy"] == "_dyn"])
    _write(DERIVED / "shares_summary.csv", summary)

    t5 = [{k: r[k] for k in ("model", "seq_len", "precision", "component", "c1", "c2", "c3_med", "c3_iqr")}
          for r in summary]
    t7 = [{k: r[k] for k in ("model", "seq_len", "precision", "component", "memory_bound", "c4_med", "c4_iqr",
                             "rank_c4")} for r in summary if r["component"] != "unattributed"]
    t6 = [{"component": c, **decide([(r["memory_bound"], r["c4_med"], r["c4_iqr"])
                                     for r in summary if r["component"] == c])} for c in TESTED]
    t4 = validate(summary, analytical)
    # Appendix: the same quantities on dynamic energy (total - idle power x window time)
    ta = [{k: r[k] for k in ("model", "seq_len", "precision", "component", "c3_med", "c3_dyn_med", "c4_med",
                             "c4_dyn_med", "c4_dyn_iqr")} for r in summary if r["component"] != "unattributed"]
    ta_rule = [{"component": c, **decide([(r["memory_bound"], r["c4_dyn_med"], r["c4_dyn_iqr"])
                                          for r in summary if r["component"] == c])} for c in TESTED]
    # Appendix: the rule with C3 renormalized over the 8 components, i.e. any Unattributed energy
    # (negative in 8 configs) spread over the components in proportion to their energy
    tn_rule = [{"component": c, **decide([(r["memory_bound"], r["c4_components_med"], r["c4_components_iqr"])
                                          for r in summary if r["component"] == c])} for c in TESTED]
    for name, rows in (("T4", t4), ("T5", t5), ("T6", t6), ("T7", t7), ("TA1_dynamic", ta),
                       ("TA2_dynamic_rule", ta_rule), ("TA3_normalized_rule", tn_rule)):
        write_table(name, _fmt(rows))
    return {"summary": summary, "t4": t4, "t6": t6, "t6_dynamic": ta_rule, "t6_normalized": tn_rule}
