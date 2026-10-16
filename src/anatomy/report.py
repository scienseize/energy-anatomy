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
                       linewidth=0.8, label=c, zorder=3)
        ax.set(xscale="log", yscale="log", title=prec.upper(), xlabel="C1 = FLOPs / bytes (FLOP/B)")
        ax.grid(True, which="major", color="#e5e4de", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(top=400)
    axes[0].set_ylabel("attainable TFLOP/s (analytical)")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("F6. Roofline position of baseline components, T4 (all models and sequence lengths)", fontsize=10)
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
            ax.set_title(f"{m.removeprefix('pythia-')} · {s} · {p.upper()}", fontsize=8.5)
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
    axes[0, 0].set_yticks(y, COMPONENTS)
    for ax in axes[-1]:
        ax.set_xlabel("share of model total", fontsize=8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("F5. Compute share vs energy share per component, Pythia on T4 (rows: model; columns: sequence length, precision)", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    labels = [c + (" *" if panels[0][1][c]["memory_bound"] == "True" else "") for c in COMPONENTS]
    axes[0, 0].set_yticks(y, labels)
    for ax in axes[-1]:
        ax.set_xlabel("C4 = C3 − C2", fontsize=8)
    fig.suptitle("F7. Waste score C4 per component (red: energy share exceeds compute share; * memory-bound in every config)", fontsize=10)
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
