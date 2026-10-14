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
