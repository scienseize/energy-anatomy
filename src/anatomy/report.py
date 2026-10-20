"""Tables (CSV + Markdown) and figures (PNG + SVG) for the thesis."""
import csv
from pathlib import Path

from anatomy.config import DATA, load_grid, load_yaml
from anatomy.units import COMPONENTS, param_count

TABLES, FIGURES = DATA / "tables", DATA / "figures"
PASSES = DATA / "raw" / "passes.csv"
# Validated categorical order (dataviz reference palette), fixed per component.
COLORS = dict(zip(COMPONENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
MARKERS = dict(zip(COMPONENTS, "osD^v<>P"))  # secondary encoding, so identity isn't color-only
# Manuscript names for figure labels; figure numbers and titles live in the manuscript captions.
LABELS = dict(zip(COMPONENTS, ["Input embedding lookup", "Layer normalization", "Attention projections",
                               "Positional encoding (RoPE)", "Attention core", "Feedforward", "Output projection",
                               "Cross-entropy loss"]))


def write_table(name: str, rows: list[dict]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0])
    with open(TABLES / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (TABLES / f"{name}.md").write_text("\n".join(md) + "\n")


def t3(sources: Path = DATA / "manual" / "sources.csv") -> list[dict]:
    """T3: Phase 1 screening log, every source the study cites scored against Table 1 (curated by hand)."""
    with open(sources) as f:
        rows = [{k: r[k] for k in ("id", "citation", "year", "venue", "used_as", "content", "training_relevance",
                                   "recency", "verifiability", "model_scale", "decision")} for r in csv.DictReader(f)]
    write_table("T3", rows)
    return rows


def t2() -> list[dict]:
    """T2: the profiled grid. A config counts as profiled once any unit has an ok pass."""
    done = set()
    if PASSES.exists():
        with open(PASSES) as f:
            done = {(r["model"], int(r["seq_len"]), r["precision"]) for r in csv.DictReader(f) if r["status"] == "ok"}
    rows = []
    for run in load_grid():
        m = run.model
        total, non_embed = param_count(m)
        rows.append({
            "model": m.name, "params": total, "non_embedding_params": non_embed, "layers": m.layers,
            "d_model": m.d, "heads": m.heads, "vocab": m.vocab, "seq_len": run.seq_len, "batch": run.batch,
            "tokens": run.tokens, "precision": run.precision,
            "profiled": (m.name, run.seq_len, run.precision) in done,
        })
    write_table("T2", rows)
    return rows


def f6(analytical: Path = DATA / "derived" / "analytical.csv") -> None:
    """F6: roofline scatter of baseline components (analytical C1 vs attainable FLOP/s), one panel
    per precision. Measured points get overlaid once profiling lands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    hw = load_yaml("hardware/t4.yaml")
    bw, peak = hw["bandwidth"], hw["peak_flops"]
    with open(analytical) as f:
        rows = [r for r in csv.DictReader(f) if r["component"] in COMPONENTS]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.logspace(-2.3, 3.6, 200)
    for ax, prec in zip(axes, ("fp32", "fp16")):
        roofs = {"CUDA cores": peak[prec]} | ({"tensor cores": peak["fp16_tensor"]} if prec == "fp16" else {})
        for label, p in roofs.items():
            ax.plot(x, np.minimum(p, x * bw) / 1e12, color="#888780", lw=1.2, ls="-" if label == "tensor cores" or prec == "fp32" else "--", zorder=1)
            ax.annotate(f"{label} {p / 1e12:g} TFLOP/s\nridge {p / bw:.0f} FLOP/B", (p / bw * 0.8, p / 1e12 * 1.15),
                        ha="right", va="bottom", fontsize=7.5, color="#5f5e5a")
        for c in COMPONENTS:
            pts = [r for r in rows if r["precision"] == prec and r["component"] == c]
            c1 = np.array([float(r["c1"]) for r in pts])
            # attainable = F / t_hat
            perf = np.array([float(r["flops"]) / float(r["t_hat"]) for r in pts])
            ax.scatter(c1, perf / 1e12, s=28, color=COLORS[c], marker=MARKERS[c], edgecolor="white",
                       linewidth=0.8, label=LABELS[c], zorder=3)
        ax.set(xscale="log", yscale="log", title=prec.upper(), xlabel="C1 = FLOPs / bytes (FLOP/B)")
        ax.grid(True, which="major", color="#e5e4de", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(top=400)
    axes[0].set_ylabel("attainable TFLOP/s (analytical)")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F6.{ext}", dpi=200)
    plt.close(fig)


def _grid_panels(summary_csv: Path):
    """Rows of shares_summary.csv grouped into a 3 (model) x 6 (seq x precision) panel grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(summary_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["component"] in COMPONENTS]
    models = list(dict.fromkeys(r["model"] for r in rows))
    cols = list(dict.fromkeys((r["seq_len"], r["precision"]) for r in rows))
    fig, axes = plt.subplots(len(models), len(cols), figsize=(17, 8.5), sharex=True, sharey=True)
    panels = []
    for i, m in enumerate(models):
        for j, (s, p) in enumerate(cols):
            sel = {r["component"]: r for r in rows if (r["model"], r["seq_len"], r["precision"]) == (m, s, p)}
            ax = axes[i, j]
            ax.set_title(f"Pythia {m.removeprefix('pythia-').upper()} · {s} · {p.upper()}", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, axis="x", color="#e5e4de", lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
            panels.append((ax, sel))
    return fig, axes, panels, plt


def f5(summary_csv: Path = DATA / "derived" / "shares_summary.csv") -> None:
    """F5: analytical compute share C2 vs measured energy share C3 (median, IQR whisker) per component."""
    import numpy as np

    fig, axes, panels, plt = _grid_panels(summary_csv)
    y = np.arange(len(COMPONENTS))[::-1]
    for ax, sel in panels:
        c2 = [float(sel[c]["c2"]) for c in COMPONENTS]
        c3 = [float(sel[c]["c3_med"]) for c in COMPONENTS]
        err = [float(sel[c]["c3_iqr"]) / 2 for c in COMPONENTS]
        ax.barh(y + 0.2, c2, height=0.38, color="#b4b2a9", label="C2 compute share (analytical)")
        ax.barh(y - 0.2, c3, height=0.38, color="#2a78d6", xerr=err, error_kw={"lw": 0.8, "ecolor": "#0d366b"},
                label="C3 energy share (measured median, IQR)")
    axes[0, 0].set_yticks(y, [LABELS[c] for c in COMPONENTS])
    for ax in axes[-1]:
        ax.set_xlabel("share of model total", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F5.{ext}", dpi=200)
    plt.close(fig)


def f7(summary_csv: Path = DATA / "derived" / "shares_summary.csv") -> None:
    """F7: diverging waste score C4 = C3 - C2 (median, IQR whisker); * marks memory-bound components."""
    import numpy as np

    fig, axes, panels, plt = _grid_panels(summary_csv)
    y = np.arange(len(COMPONENTS))[::-1]
    for ax, sel in panels:
        c4 = np.array([float(sel[c]["c4_med"]) for c in COMPONENTS])
        err = [float(sel[c]["c4_iqr"]) / 2 for c in COMPONENTS]
        ax.barh(y, c4, height=0.7, color=np.where(c4 > 0, "#e34948", "#2a78d6"), xerr=err,
                error_kw={"lw": 0.8, "ecolor": "#2c2c2a"})
        ax.axvline(0, color="#5f5e5a", lw=0.8)
    labels = [LABELS[c] + (" *" if panels[0][1][c]["memory_bound"] == "True" else "") for c in COMPONENTS]
    axes[0, 0].set_yticks(y, labels)
    for ax in axes[-1]:
        ax.set_xlabel("C4 = C3 − C2", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F7.{ext}", dpi=200)
    plt.close(fig)


def f10(t10_csv: Path = TABLES / "T10.csv") -> None:
    """F10: model energy change from each substitution, % of baseline E_total (median, IQR whisker).
    Blue = saving, red = the substitution costs more energy than the unit it replaces."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(t10_csv) as f:
        rows = list(csv.DictReader(f))
    subs = list(dict.fromkeys(r["substitution"] for r in rows))
    cfgs = list(dict.fromkeys((r["model"], r["seq_len"], r["precision"]) for r in rows))
    labels = [f"{m.removeprefix('pythia-')}·{s}·{p.upper()}" for m, s, p in cfgs]
    fig, axes = plt.subplots(len(subs), 1, figsize=(12, 2.1 * len(subs)), sharex=True)
    x = np.arange(len(cfgs))
    for ax, sub in zip(axes, subs):
        by = {(r["model"], r["seq_len"], r["precision"]): r for r in rows if r["substitution"] == sub}
        d = np.array([float(by[c]["delta_pct_med"]) for c in cfgs])
        err = [float(by[c]["delta_pct_iqr"]) / 2 for c in cfgs]
        ax.bar(x, d, width=0.7, color=np.where(d > 0, "#2a78d6", "#e34948"), yerr=err,
               error_kw={"lw": 0.8, "ecolor": "#2c2c2a"})
        ax.axhline(0, color="#5f5e5a", lw=0.8)
        ax.set_title(f"{sub} (replaces {by[cfgs[0]]['replaces']})", fontsize=9, loc="left")
        ax.set_ylabel("ΔE_model (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", color="#e5e4de", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(x, labels, rotation=60, ha="right")
    fig.suptitle("F10. Change in model energy per substitution, % of baseline (blue: saving; red: costs more)", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F10.{ext}", dpi=200)
    plt.close(fig)


def f8(t8_csv: Path = TABLES / "T8.csv") -> None:
    """F8: waste x capability quadrant (decision 30). y = median C4 pooled over the 18 configs,
    whisker = range over configs; x = capability rank (1 = most capable); split at C4 = 0 and
    between the median rank and the rank above it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(t8_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["capability_rank"]]
    x = np.array([int(r["capability_rank"]) for r in rows])
    y = np.array([float(r["c4_pooled_med"]) for r in rows])
    lo = y - np.array([float(r["c4_min"]) for r in rows])
    hi = np.array([float(r["c4_max"]) for r in rows]) - y
    split = float(np.median(x)) - 0.5  # the median rank sits in the lower-capability half

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(0, color="#5f5e5a", lw=0.8)
    ax.axvline(split, color="#5f5e5a", lw=0.8, ls="--")
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#b4b2a9", lw=1.2, capsize=3, zorder=1)
    ax.scatter(x, y, s=60, c=np.where(y > 0, "#e34948", "#2a78d6"), edgecolor="white", linewidth=1, zorder=3)
    for xi, yi, r in zip(x, y, rows):
        ax.annotate(r["component"], (xi, yi), xytext=(8, 4), textcoords="offset points", fontsize=8.5)
    kw = {"fontsize": 8, "color": "#888780", "transform": ax.transAxes}
    ax.text(0.02, 0.97, "high waste\nhigh capability", va="top", **kw)
    ax.text(0.98, 0.97, "high waste\nlow capability", va="top", ha="right", **kw)
    ax.text(0.02, 0.03, "low waste\nhigh capability", **kw)
    ax.text(0.98, 0.03, "low waste\nlow capability", ha="right", **kw)
    ax.set_xticks(range(1, x.max() + 1))
    ax.set_xlim(0.4, x.max() + 0.9)
    span = (y + hi).max() - (y - lo).min()
    ax.set_ylim((y - lo).min() - 0.18 * span, (y + hi).max() + 0.18 * span)  # room for the quadrant labels
    ax.set_xlabel("capability rank (1 = largest contribution to quality)", fontsize=9)
    ax.set_ylabel("C4 = C3 − C2 (median over configs; whisker = range)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("F8. Waste vs capability per component (red: energy share exceeds compute share)", fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F8.{ext}", dpi=200)
    plt.close(fig)


def f9(shifts_csv: Path = DATA / "derived" / "rank_shifts.csv") -> None:
    """F9: C5 rank shifts. One heatmap per moderator (scale, sequence length, precision); each column
    compares two configs differing only in that moderator. Cell = rank_from - rank_to:
    positive (red) = the component climbs toward rank 1, i.e. more waste relative to the others."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    with open(shifts_csv) as f:
        rows = list(csv.DictReader(f))
    mods = list(dict.fromkeys(r["moderator"] for r in rows))
    cols = {m: list(dict.fromkeys(r["held"] for r in rows if r["moderator"] == m)) for m in mods}
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    n = len(COMPONENTS)
    fig, axes = plt.subplots(1, len(mods), figsize=(15, 4.6), gridspec_kw={"width_ratios": [len(cols[m]) for m in mods]})
    for ax, m in zip(axes, mods):
        by = {(r["held"], r["component"]): int(r["shift"]) for r in rows if r["moderator"] == m}
        grid = np.array([[by[(h, c)] for h in cols[m]] for c in COMPONENTS])
        im = ax.imshow(grid, cmap=cmap, vmin=-(n - 1), vmax=n - 1, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                ax.text(j, i, f"{v:+d}" if v else "0", ha="center", va="center", fontsize=8,
                        color="#2c2c2a" if abs(v) < 4 else "white")
        r0 = next(r for r in rows if r["moderator"] == m)
        ax.set_title(f"{m}: {str(r0['from']).removeprefix('pythia-')} → {str(r0['to']).removeprefix('pythia-')}", fontsize=9)
        ax.set_xticks(range(len(cols[m])), cols[m], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n), COMPONENTS if ax is axes[0] else [""] * n, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("rank change (+ = climbs toward rank 1, more waste)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("F9. Shift in C4 rank under each configuration moderator (C5); columns: the other two factors held fixed", fontsize=10)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"F9.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
