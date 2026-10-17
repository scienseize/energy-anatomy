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
from anatomy.units import COMPONENTS, REMOVALS, SUBSTITUTIONS

RAW, DERIVED = DATA / "raw", DATA / "derived"
TESTED = ("embed_lookup", "ln", "rope", "attn_core")  # H1 components; ln is pooled with the final norm
USABLE = ("ok", "short_window")  # short windows are kept (flagged); oom rows carry no energy
MODEL_ORDER = ("pythia-160m", "pythia-1.4b", "pythia-6.9b")
COMP_ORDER = (*COMPONENTS, "unattributed")
SUB_ORDER = (*dict.fromkeys(sub for sub, _ in SUBSTITUTIONS.values()), *REMOVALS)
MANUAL = DATA / "manual" / "substitutions.csv"


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


def _energy(passes: list[dict], idle: list[dict]) -> dict:
    """(config, unit) -> pass -> (total J per rep, dynamic J per rep), usable passes only."""
    idle_w = _idle_lookup(idle)
    e = defaultdict(dict)
    for r in passes:
        if r["status"] not in USABLE:
            continue
        reps, t, joules = int(r["reps"]), float(r["time_s"]), float(r["energy_j"])
        dyn = joules - idle_w(r["run_id"], r["ts"]) * t
        cfg = (r["model"], int(r["seq_len"]), r["precision"])
        e[(cfg, r["unit"])][int(r["pass_idx"])] = (joules / reps, dyn / reps)
    return e


def _model_total(e: dict, cfg, j: int, k: int, layers: int) -> float:
    """E_total = L x E_block + E_embed + E_final_norm + E_out_proj + E_loss. The final norm is one ln
    instance; the other 2L ln instances live inside the blocks."""
    return layers * e[(cfg, "block")][j][k] + sum(e[(cfg, c)][j][k] for c in ("embed_lookup", "ln", "out_proj", "loss"))


def per_pass(passes: list[dict], idle: list[dict], analytical: list[dict], n_passes: int) -> list[dict]:
    """One row per (config, component|unattributed, pass): C3 and C4 on total and dynamic energy.
    Pass j of every unit in a config forms one matched set (DESIGN §8)."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    e = _energy(passes, idle)
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
                total = _model_total(e, cfg, j, k, layers)
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


def substitution_deltas(passes: list[dict], idle: list[dict], analytical: list[dict], n_passes: int) -> list[dict]:
    """RQ5 per pass (DESIGN section 5): E_total with a substitution swapped in =
    E_total(baseline) - sum(replaced unit energy x count) + sum(substitute energy x count),
    on matched passes. delta_pct = (baseline - substituted) / baseline x 100, positive = saving.
    factorized_embed swaps its in and out units together."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    e = _energy(passes, idle)
    groups = defaultdict(list)
    for unit, (sub, replaced) in SUBSTITUTIONS.items():
        groups[sub].append((unit, replaced))
    for sub, replaced in REMOVALS.items():
        groups[sub].append((None, replaced))  # removal: nothing runs in its place
    out = []
    for cfg in dict.fromkeys(c for c, _ in e):
        layers = int(an[(*cfg, "block")]["count_per_model"])
        count = lambda u: int(an[(*cfg, u)]["count_per_model"])
        for j in range(n_passes):
            if any(j not in e[(cfg, u)] for u in (*COMPONENTS, "block")):
                continue
            base = _model_total(e, cfg, j, 0, layers)
            for sub, members in groups.items():
                if any(u is not None and j not in e.get((cfg, u), {}) for u, _ in members):
                    continue  # substitution not measured in this session (e.g. added later)
                removed = sum(e[(cfg, rep)][j][0] * count(rep) for _, rep in members)
                added = sum(e[(cfg, u)][j][0] * count(u) for u, _ in members if u is not None)
                total = base - removed + added
                out.append({"model": cfg[0], "seq_len": cfg[1], "precision": cfg[2], "substitution": sub,
                            "replaces": "+".join(rep for _, rep in members), "pass_idx": j,
                            "e_total_base_j": base, "e_total_sub_j": total,
                            "delta_pct": (base - total) / base * 100,
                            "replaced_c3": removed / base, "sub_c3": added / total})
    return out


def summarize_substitutions(deltas: list[dict], analytical: list[dict], manual: dict) -> list[dict]:
    """T10: per (config, substitution) medians over passes, IQR of the saving, and the published
    value from data/manual/substitutions.csv (left blank until filled by hand)."""
    an = {(r["model"], int(r["seq_len"]), r["precision"], r["component"]): r for r in analytical}
    g = defaultdict(list)
    for r in deltas:
        g[(r["model"], r["seq_len"], r["precision"], r["substitution"], r["replaces"])].append(r)
    out = []
    for (m, s, p, sub, rep), rs in g.items():
        units = [u for u, (x, _) in SUBSTITUTIONS.items() if x == sub]  # empty for a removal
        d = [r["delta_pct"] for r in rs]
        out.append({"model": m, "seq_len": s, "precision": p, "substitution": sub, "replaces": rep,
                    "n_passes": len(rs),
                    "e_total_base_j": float(np.median([r["e_total_base_j"] for r in rs])),
                    "e_total_sub_j": float(np.median([r["e_total_sub_j"] for r in rs])),
                    "delta_pct_med": float(np.median(d)), "delta_pct_iqr": iqr(d),
                    "replaced_c3_med": float(np.median([r["replaced_c3"] for r in rs])),
                    "sub_c3_med": float(np.median([r["sub_c3"] for r in rs])),
                    "sub_memory_bound": "+".join(an[(m, s, p, u)]["memory_bound"] for u in units),
                    "published_value": manual.get(sub, {}).get("published_value", "")})
    return sorted(out, key=lambda r: (grid_order(r), SUB_ORDER.index(r["substitution"])))


def priority_table(summary: list[dict], rule: list[dict], manual: dict) -> list[dict]:
    """T9: priority component -> mechanism -> substitution -> source. Waste columns are computed;
    mechanism and source come from data/manual/substitutions.csv."""
    verdict = {r["component"]: r["verdict"] for r in rule}
    out = []
    for sub in SUB_ORDER:  # listed even before measurement: T9 is the plan, T10 the outcome
        replaced = list(dict.fromkeys(rep for x, rep in SUBSTITUTIONS.values() if x == sub)) or [REMOVALS[sub]]
        def rng(c, key, f):
            v = [r[key] for r in summary if r["component"] == c]
            return f"{f(min(v))} to {f(max(v))}"

        mm = manual.get(sub, {})
        out.append({"priority_component": "; ".join(replaced),
                    "h1_verdict": "; ".join(verdict.get(c, "reported only") for c in replaced),
                    "c4_med_range": "; ".join(rng(c, "c4_med", lambda x: f"{x:+.4f}") for c in replaced),
                    "rank_c4_range": "; ".join(rng(c, "rank_c4", str) for c in replaced),
                    "mechanism": mm.get("mechanism", ""), "substitution": sub, "source": mm.get("source", "")})
    return out


CAPABILITY = DATA / "manual" / "capability.csv"
MODERATORS = {  # C5 (decision 28): moderator -> (field, from value, to value, the fields held fixed)
    "scale": ("model", "pythia-160m", "pythia-6.9b", ("seq_len", "precision")),
    "sequence length": ("seq_len", 512, 2048, ("model", "precision")),
    "precision": ("precision", "fp32", "fp16", ("model", "seq_len")),
}


def quadrant(c4: float, rank: int, median_rank: float) -> str:
    """F8 split (decision 30): waste at C4 = 0, capability at the median rank (1 = most capable).
    A component exactly at the median goes to the lower-capability half."""
    waste = "high waste" if c4 > 0 else "low waste"
    cap = "high capability" if rank < median_rank else "low capability"
    return f"{waste}, {cap}"


def capability_table(summary: list[dict], rule: list[dict], capability: list[dict]) -> list[dict]:
    """T8: capability rank and evidence (data/manual/capability.csv) next to each component's waste,
    pooled over configs as the median of the per-config median C4, and its F8 quadrant."""
    verdict = {r["component"]: r["verdict"] for r in rule}
    ranked = [int(r["capability_rank"]) for r in capability if r["capability_rank"]]
    median_rank = float(np.median(ranked)) if ranked else None
    out = []
    for r in capability:
        c4 = [x["c4_med"] for x in summary if x["component"] == r["component"]]
        pooled = float(np.median(c4)) if c4 else None
        rank = int(r["capability_rank"]) if r["capability_rank"] else None
        out.append({"component": r["component"], "capability_rank": rank,
                    "c4_pooled_med": pooled, "c4_min": min(c4) if c4 else None, "c4_max": max(c4) if c4 else None,
                    "h1_verdict": verdict.get(r["component"], "reported only"),
                    "quadrant": quadrant(pooled, rank, median_rank) if rank and pooled is not None else "",
                    "evidence": r["evidence"], "citation": r["citation"], "note": r["note"]})
    return sorted(out, key=lambda r: (r["capability_rank"] is None, r["capability_rank"] or 0))


def rank_shifts(summary: list[dict]) -> list[dict]:
    """F9 data: for each C5 moderator, the change in each component's C4 rank between two configs
    that differ only in that moderator. shift = rank_from - rank_to, so positive = the component
    climbs toward rank 1 (more waste) when the moderator changes."""
    rank = {(r["model"], r["seq_len"], r["precision"], r["component"]): r["rank_c4"]
            for r in summary if r["component"] in COMPONENTS}
    out = []
    for mod, (field, a, b, fixed) in MODERATORS.items():
        others = dict.fromkeys(tuple(r[f] for f in fixed) for r in summary)  # summary is in grid order
        for vals in others:
            ctx = dict(zip(fixed, vals))
            for c in COMPONENTS:
                key = lambda v: tuple({**ctx, field: v}[k] for k in ("model", "seq_len", "precision")) + (c,)
                if key(a) in rank and key(b) in rank:
                    out.append({"moderator": mod, "from": a, "to": b,
                                "held": " ".join(str(ctx[f]).removeprefix("pythia-") for f in fixed),
                                "component": c, "rank_from": rank[key(a)], "rank_to": rank[key(b)],
                                "shift": rank[key(a)] - rank[key(b)]})
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
    manual = {r["substitution"]: r for r in _read(MANUAL)} if MANUAL.exists() else {}
    deltas = substitution_deltas(passes, _read(idle_csv), analytical, n_passes)
    t10 = summarize_substitutions(deltas, analytical, manual)
    t9 = priority_table(summary, t6, manual)
    _write(DERIVED / "substitutions.csv", deltas)
    shifts = rank_shifts(summary)
    _write(DERIVED / "rank_shifts.csv", shifts)
    t8 = capability_table(summary, t6, _read(CAPABILITY)) if CAPABILITY.exists() else []
    if t8:
        _write(DERIVED / "capability.csv", t8)
        write_table("T8", _fmt(t8))
    write_table("T9", _fmt(t9))
    write_table("T10", _fmt(t10))
    return {"summary": summary, "t4": t4, "t6": t6, "t6_dynamic": ta_rule, "t6_normalized": tn_rule,
            "t9": t9, "t10": t10, "t8": t8, "shifts": shifts}
