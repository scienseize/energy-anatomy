# energy-anatomy

Profiling and analysis code for a master's thesis on the energy anatomy of transformer pretraining: a roofline-based diagnostic waste score (component energy share minus compute share) for Pythia 160M, 1.4B and 6.9B on an NVIDIA T4.

- `DESIGN.md`: the specification and decision log (Section 11).
- `docs/methods.md`: what is implemented, every convention and constant, deviations, verification, and limitations. Written for the Methodology and Results and Discussion chapters.

## Layout

```
configs/                 Pythia configs, grid, T4 hardware, measurement protocol
src/anatomy/
  config.py              configs -> ModelSpec, RunSpec
  components.py          baseline units (GPT-NeoX semantics) and the full block
  substitutions.py       sdpa_core, linear_core, gqa_proj, factorized_embed, rmsnorm
  units.py               unit registry: build, inputs, one fwd+bwd step, counts per model
  analytical.py          FLOPs/bytes by tracing on the meta device; C1, ridge, C2, t_hat
  harness.py             Zeus energy windows, rotating buffers, calibration, resume
  analysis.py            shares C3, waste C4, decision rule, validation
  report.py              tables (CSV + Markdown) and figures
  cli.py                 the `anatomy` command
notebooks/kaggle_profile.ipynb   thin Kaggle runner
data/raw/                passes.csv, idle.csv (measured; from Kaggle)
data/derived/            analytical.csv
data/tables/, figures/   thesis tables and figures
tests/
```

## Use

```
uv sync
uv run pytest
uv run anatomy analytical          # data/derived/analytical.csv
uv run anatomy report              # T2, F6
uv run anatomy analyze             # shares, C4, decision rule: T4-T7, F5, F7 (needs data/raw/passes.csv)
uv run anatomy profile --dry-run   # harness end to end on CPU with fake energy (no real numbers)
```

Energy is measured only on the Kaggle T4. In Kaggle, open `notebooks/kaggle_profile.ipynb` with Accelerator = GPU T4 x2 and Internet on. For the first session, set `SMOKE = True` (about 10 min). After that, run with `SMOKE = False`. To resume after a 12 h session ends, attach the previous notebook version's output as an input; the notebook carries `passes.csv` over and runs `anatomy profile --resume`. Copy the resulting `raw/` folder into `data/raw/`.
