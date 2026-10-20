# Methods record: energy anatomy of transformer pretraining components

This file records how the code in this repository produces every number used in the thesis, at the level of detail needed to write the Methodology chapter and to defend the Results and Discussion chapter. DESIGN.md is the specification; this file records what was implemented, the conventions it fixes, where it departs from the specification, and its known limitations. Section and file references point at the code that implements each statement.

Status (2026-10-17): the full grid has been measured three times on the Kaggle T4, in three sessions on three different GPUs (Section 13). Run 3 is the main dataset (`data/raw/`); runs 1 and 2 (`data/raw/run-1/`, `data/raw/run-2/`) are kept as repeatability checks (Section 17). The analysis (Sections 14 to 18) and the dashboard export (Section 19) are implemented and tested. Every number in Section 9 is analytical (computed, not measured).

## 1. Scope and provenance

- **Hardware target.** A single NVIDIA T4 (16 GB GDDR6, 320 GB/s, 4 MiB L2, 70 W cap) on Kaggle. Peak rates in `configs/hardware/t4.yaml`: FP32 8.1 TFLOP/s (CUDA cores; Turing has no TF32, so FP32 matmuls also run here), FP16 16.2 TFLOP/s on CUDA cores (half2, twice FP32), FP16 65 TFLOP/s on tensor cores.
- **Models.** Pythia 160M, 1.4B and 6.9B (EleutherAI). The architecture fields were transcribed from EleutherAI/pythia `models/*.yml` into Hugging Face `config.json` form (`configs/pythia/`), because the Hugging Face Hub was unreachable from the build environment. No weights are used (Section 2.3).
- **Grid.** 3 models x sequence length {512, 1024, 2048} x precision {FP32, FP16} = 18 configurations, each processing 8,192 tokens per pass (batch = 8192 / seq_len, so 16, 8 or 4). Defined in `configs/grid.yaml`.
- **Software.** Python 3.11, PyTorch 2.14.1 (local analytical runs), Zeus 0.16.0 for energy. Dependencies are pinned in `uv.lock`. `transformers` is used only in tests.
- **Provenance per measurement.** Every row of `data/raw/passes.csv` carries the run id, timestamp, torch and CUDA versions, NVIDIA driver, and the git commit (suffix `+dirty` if tracked files had uncommitted changes).

## 2. Models and units

### 2.1 Architecture

All modules reproduce the GPT-NeoX variant used by Pythia: parallel residual (`x + attn(ln1(x)) + mlp(ln2(x))`), LayerNorm with eps 1e-5, rotary embeddings on 25% of each head's dimensions (base 10000), exact (erf) GELU, biases on all attention and MLP linears, untied input and output embeddings, and no bias on the output projection. `src/anatomy/config.py` refuses any config that does not match this variant.

| Model | Layers L | d_model | Heads | Head dim | Rotary dims | FFN | Vocab | Parameters (ours = published) | Non-embedding |
|---|---|---|---|---|---|---|---|---|---|
| Pythia 160M | 12 | 768 | 12 | 64 | 16 | 3072 | 50304 | 162,322,944 | 85,056,000 |
| Pythia 1.4B | 24 | 2048 | 16 | 128 | 32 | 8192 | 50304 | 1,414,647,808 | 1,208,602,624 |
| Pythia 6.9B | 32 | 4096 | 32 | 128 | 32 | 16384 | 50432 | 6,857,302,016 | 6,444,163,072 |

Parameter counts summed over our modules match the published counts exactly (test `test_param_counts_match_published`).

### 2.2 Units

A *unit* is anything measured on its own: a baseline component, a substitution, or the full block. Each unit runs forward and backward on its own inputs, with a random `grad_output` for every output (except `loss`, whose scalar output is back-propagated directly). Defined in `src/anatomy/units.py`, `components.py` and `substitutions.py`.

| Unit | What runs (forward; backward is autograd's) | Inputs | Count per model |
|---|---|---|---|
| embed_lookup | `Embedding(V, d)` gather | token ids (b, s) | 1 |
| ln | `LayerNorm(d)` | (b, s, d) | 2L + 1 (two per block plus the final norm, pooled) |
| attn_proj | fused QKV `Linear(d, 3d)` on the block input and output `Linear(d, d)` on the attention context | (b, s, d), (b, s, d) | L |
| rope | rotary on the first rotary dims of Q and K, eager ops as in HF (`x*cos + rotate_half(x)*sin`, then concatenate the pass-through dims) | Q, K (b, h, s, hd) | L |
| attn_core | `QK^T * hd^-0.5`, causal mask (`torch.where` with a boolean lower-triangular mask), softmax, `.V`, merge heads | Q, K, V (b, h, s, hd) | L |
| mlp | `Linear(d, 4d)`, GELU, `Linear(4d, d)` | (b, s, d) | L |
| out_proj | `Linear(d, V)` without bias | (b, s, d) | 1 |
| loss | cross-entropy over (b*s) positions, mean reduction | logits (b, s, V), targets | 1 |
| block | full parallel-residual block, including the QKV head split (views), rope and attn_core | (b, s, d) | L (denominator) |

Model energy per configuration is E_total = L * E_block + E_embed_lookup + E_ln (final norm, one instance) + E_out_proj + E_loss. Component energy is E_unit * count per model, and Unattributed is E_total minus the sum of components (DESIGN section 4). Unattributed captures work that exists in the block but in no isolated unit: the residual additions, the strided QKV views feeding rope and attention, and inter-kernel effects.

Isolated units receive fresh contiguous input tensors, while inside the block the same operations act on views. This difference is part of what Unattributed absorbs.

### 2.3 Weights and initialization

Weights are random (PyTorch default initialization). Energy depends on weight values only through small data-dependent power effects, which is stated as a limitation (Section 10).

### 2.4 Precision

- **FP32:** every tensor is float32.
- **FP16:** the whole module is cast with `.half()`. Parameters, activations, softmax, LayerNorm, loss and gradients are all float16. This matches the GPT-NeoX/DeepSpeed fp16 mode Pythia was trained with, and keeps one dtype per configuration in the bytes model. No autocast is used.

The current HF eager attention upcasts the softmax to float32 inside an otherwise fp16 model. Our `attn_core` keeps the softmax in float16, consistent with the pure-fp16 choice. The HF equivalence test therefore runs in FP32 (Section 8).

## 3. Substitutions

Each substitution replaces exactly one baseline unit, keeps its call signature, and is measured on every configuration with the same protocol (`src/anatomy/substitutions.py`).

| Substitution | Replaces | Implementation |
|---|---|---|
| sdpa_core | attn_core | Fused causal attention through the memory-efficient backend, called directly (`aten._scaled_dot_product_efficient_attention`), so it cannot silently fall back to the math path. FlashAttention-2 is unsupported on the T4. Followed by the same merge-heads copy as attn_core. |
| linear_core | attn_core | Causal linear attention (Katharopoulos et al., 2020), phi(x) = elu(x) + 1, with the normalizer phi(q).sum(phi(k)) and eps 1e-6. Chunked with C = 64: masked quadratic attention inside each chunk (including the diagonal), plus a prefix state over earlier chunks carried by an exclusive cumulative sum of per-chunk phi(K)^T V and phi(K) sums. |
| gqa_proj | attn_proj | Fused projection `Linear(d, d + 2 * kv_heads * hd)` with kv_heads = heads / 4 (3, 4 and 8 for the three models), the K/V repeat to all heads materialized as in HF `repeat_kv` (expand + reshape, a real copy), and the output `Linear(d, d)`. Biases as in the baseline. |
| factorized_embed (in) | embed_lookup | ALBERT style (Lan et al., 2020): `Embedding(V, E)` + `Linear(E, d)`, E = 128, no bias. |
| factorized_embed (out) | out_proj | Mirrored: `Linear(d, E)` + `Linear(E, V)`, E = 128, no bias (like Pythia's embed_out). **This output-side factorization is the thesis's own extension.** ALBERT factorizes only the input embedding, so no published result covers the mirrored output layer (limitation 11). |
| rmsnorm | ln (all 2L + 1) | Zhang and Sennrich (2019): `x * rsqrt(mean(x^2) + eps) * w`, eager ops in the run dtype (six separate operators, as in HF's Llama RMSNorm). |
| rope_fused | rope | The same rotary function compiled by `torch.compile` (Inductor, `fullgraph=True`) into fused kernels (added 2026-10-17, because rope is a priority component with no substitution; Section 18). The Kaggle notebook lists the CUDA kernels one forward + backward launches, to show whether Inductor fused it on the T4. |
| nope | rope | NoPE: the positional encoding removed (Haviv et al., 2022; Kazemnejad et al., 2023). Nothing runs in its place, so it is not profiled: the substituted model energy is the baseline minus rope's measured energy (Section 16). |
| rmsnorm_fused | ln (all 2L + 1) | The same function through `F.rms_norm`, which dispatches to PyTorch's fused CUDA kernel where the build has one (added 2026-10-17; Section 16). The Kaggle notebook logs which operators actually ran. |

The RQ5 metric is delta E_model = E_total(baseline) - E_total(with the substitution swapped in), as a percentage of baseline, per configuration.

## 4. Analytical model

Implemented in `src/anatomy/analytical.py`. Output: `data/derived/analytical.csv`, one row per (configuration, unit), 270 rows.

### 4.1 Counting method

FLOPs and bytes are not written out by hand per component. Each unit's real module runs one forward and backward step on PyTorch's `meta` device (shapes only, no data) under a dispatch-mode interceptor that sees every ATen operator autograd issues. This counts exactly the operator sequence the GPU will execute, for baseline units, substitutions and the block alike, with no separate formulas to keep in sync.

- View and aliasing operators (`is_view`, `_unsafe_view`) and pure allocations (`new_empty_strided`) move no data and are skipped.
- An operator without a FLOP rule stops the run with an error, so no operator is ever silently counted as zero.
- **Exception: rope_fused** is also counted by formula, because the meta device cannot see Inductor's fused kernels: one fused pass forward and one backward per tensor (Q and K), each reading and writing the full tensor once plus the cos/sin tables, with 3 FLOPs per rotated element forward and 3 backward. At 1.4B/2048/FP32 this gives 0.54 GB per call against 1.88 GB for the eager rope.
- **Exception: rmsnorm_fused** is counted by formula, because on the meta device `F.rms_norm` decomposes into eager operators and tracing cannot see the fused CUDA kernel. It is counted like `native_layer_norm`: one operator forward (reads x and w, writes y and an fp32 rstd per row) and one backward (reads dy, x, w and rstd, writes dx and dw), with 4 FLOPs per element forward and 9 backward.

### 4.2 Bytes

Bytes per operator = bytes read from every input tensor + bytes written to every output tensor, at the configuration's precision. No cache reuse is assumed between operators. This is the upper bound named in DESIGN section 7, and it matches the harness's rotating buffers, which defeat L2 reuse across repetitions.

- A tensor's bytes are min(number of elements, storage size) * element size. A broadcast (expanded) tensor reads only its storage, while a slice reads only its own elements.
- Write-only operators (`ones_like`, `scalar_tensor`) count no reads. `copy_` reads its source only. `nll_loss_backward` reads only the gradient and targets (its logits argument conveys a shape and is not read).
- Token ids are int64 (8 bytes). The attention mask is bool (1 byte). LayerNorm's saved mean and rstd are float32 in both precisions (as on CUDA).

### 4.3 FLOPs

- **Matrix multiplications** (`mm`, `bmm`, `addmm`): 2mkn, plus mn for the fused bias of `addmm`. The backward pass is counted from its own traced matmuls, so it comes out at exactly twice the forward pass, as DESIGN section 7 specifies.
- **Memory-efficient attention** (sdpa_core): forward 4 * b * h * s_q * s_k * hd, halved under the causal mask because the kernel skips fully masked tiles. Backward is 2.5x forward (five matmuls). Eager attn_core, by contrast, computes the full s x s scores.
- **Elementwise and normalization operators:** a fixed per-element constant, applied to the largest operand (the output for elementwise operators, the input for reductions). These constants are a convention and are listed in full:

| Operator | FLOPs per element | Basis |
|---|---|---|
| add, sub, mul, div, neg, where, pow, rsqrt, sum, mean, cumsum | 1 | |
| elu, elu backward | 2 | |
| gelu (exact) | 5 | 0.5 x (1 + erf(x / sqrt 2)) |
| gelu backward | 10 | dy (Phi(x) + x phi(x)) |
| softmax / log_softmax | 5 | max, subtract, exp, sum, divide |
| softmax / log_softmax backward | 4 | |
| layer norm | 8 | mean, variance (3), normalize (2), affine (2) |
| layer norm backward | 15 | dx (10) + dgamma, dbeta (5) |
| nll loss forward / backward | 1 | per target |
| embedding backward | 1 | per gathered gradient element |
| embedding (gather), cat, clone, copy, flip, slice backward, fills | 0 | data movement only |

Because the elementwise constants are conventions, only the order of magnitude of a memory-bound unit's C1 is meaningful. The flags are robust to them: purely elementwise units sit one to three orders of magnitude below their ridge, and the units closest to a ridge (attn_core, sdpa_core, factorized_embed) get their FLOPs almost entirely from exactly counted matmuls (for attn_core, elementwise operators are 0.8% of FLOPs).

### 4.4 Roofline quantities

- **C1** = FLOPs / bytes for one unit instance, forward + backward.
- **Peaks.** Matmul FLOPs use the tensor-core peak in FP16 and the FP32 CUDA-core peak in FP32. All other FLOPs use the CUDA-core peak of the precision.
- **Effective peak for mixed units** (decided 2026-10-15). A unit whose FLOPs split between matmul and elementwise work (attn_core, mlp, the block) gets P_eff = F / (F_mm / P_mm + F_ew / P_ew), the harmonic, FLOP-weighted peak.
- **Ridge** = P_eff / bandwidth. For pure-matmul units this is 25.3 FLOP/B in FP32 and 203 FLOP/B in FP16. For pure elementwise units it is 25.3 (FP32) and 50.6 (FP16).
- **Hardware parameters and their sources.** FP32 8.1 TFLOP/s, FP16 tensor 65 TFLOP/s, 16 GB, 70 W and 320 GB/s are NVIDIA's published T4 figures (product page, verified 2026-10-17); NVIDIA's own documents differ on bandwidth (the product page gives "320+ GB/s", the datasheet is reported as 300 GB/s). The FP16 CUDA-core rate (16.2 TFLOP/s, twice FP32) is an assumption: no NVIDIA document read for this study states it. **Sensitivity:** with FP16 elementwise at 8.1 TFLOP/s no memory-bound flag changes; with 300 GB/s exactly one changes, the full block at 160M/2048/FP32, which is the energy denominator and is not tested. No tested component's or substitution's flag depends on either value.
- **Memory-bound** if C1 < ridge. This flag decides which (component, configuration) pairs enter the H1 test.
- **Predicted time** t_hat = max(F / P_eff, bytes / bandwidth), per instance.
- **C2** = FLOPs_unit * count per model / total model FLOPs, where total model FLOPs is the sum over the eight baseline components (so the C2 values of the baseline components sum to 1 by construction). For a substitution unit, C2 is its share within the model that has that substitution swapped in. For factorized_embed, both the in and out units are swapped together. The block row has no C2.

## 5. Tables and figures

All are regenerated by `anatomy analytical`, `anatomy report` and `anatomy analyze` into `data/tables/` (CSV + Markdown) and `data/figures/` (PNG + SVG).

| Output | Content | Section |
|---|---|---|
| T2 | the 18 configurations, model dimensions, parameter counts, profiled flag | 1 |
| T4 | analytical vs measured C3 (component-normalized), absolute difference, Spearman rho | 14, 15 |
| T5 | C1, C2, C3 median and IQR per component and configuration, plus Unattributed | 14, 15 |
| T6 | H1 decision rule per tested component | 14, 15 |
| T7 | C4 median, IQR and rank per component and configuration | 14, 15 |
| T8 | capability rank and evidence, pooled C4, F8 quadrant | 18 |
| T9 | priority component, mechanism, substitution, source | 16 |
| T10 | baseline vs substituted model energy, delta %, published value | 16 |
| TA1, TA2 | appendix: shares and decision rule on dynamic energy | 14 |
| TA3 | appendix: decision rule with component-normalized C3 | 14 |
| F5 | C2 vs C3 paired bars, 3 x 6 panels | 15 |
| F6 | analytical roofline scatter of baseline components | 4, 5 |
| F7 | diverging C4 bars, 3 x 6 panels | 15 |
| F8 | waste x capability quadrant | 18 |
| F9 | C4 rank shifts under the C5 moderators (scale, sequence length, precision) | 18 |
| F10 | delta % of model energy per substitution | 16 |

F6 details: x axis is analytical C1; y axis is the analytical attainable rate F / t_hat. Roof lines show the CUDA-core and (FP16) tensor-core peaks with their ridge points. Colors follow a fixed categorical order per component, and each component also has its own marker shape, so identity does not depend on color alone.

## 6. Measurement protocol as implemented

Implemented in `src/anatomy/harness.py`; run with `anatomy profile [--resume]`. Protocol constants live in `configs/protocol.yaml`.

0. **Thermal pre-conditioning** (decided 2026-10-16, after the smoke test). At the start of each session, before the first idle window, the harness runs FP32 4096 x 4096 matmuls in 30 s steps until the GPU temperature changes by less than 1 C over one step, or for at most 5 minutes. Every measured unit, including the first of a session, then sees the hot steady state that a real pretraining run sits in. The plateau temperature and the time taken are printed to the session log.
1. **Order** (decided 2026-10-16). Configuration-major in grid order: all 15 units of a configuration before the next configuration, smallest model first (160M, then 1.4B, then 6.9B; within a model, seq 512 to 2048, FP32 then FP16). If a session ends early, every finished configuration is complete and can be analyzed on its own.
2. **Build.** The module is built once per (configuration, unit) on `cuda:0` and cast to the run dtype.
3. **Rotating buffers.** K independent (inputs, grad_outputs) sets with K = max(2, ceil(4 x 4 MiB / bytes per set)). Repetition i uses set i mod K, so consecutive repetitions never find their inputs in L2. Weights are not rotated.
4. **Warm-up.** 20 forward + backward iterations, discarded.
5. **Calibration** (decided 2026-10-16). A timed probe runs n = 1, 2, 4, ... repetitions between two device synchronizations until the probe lasts at least 0.2 s. Then R = ceil(1.1 x 2 s / t_rep), with a 10% margin. R is fixed for all passes of that unit in that session.
6. **Passes.** N = 10. Each pass is: Zeus `begin_window` (synchronizes the device, then reads the NVML energy counter), R x (forward on buffer i mod K, `backward(grad_output)`, then set every parameter and input `.grad` to None, so every repetition does identical work with no gradient accumulation), and `end_window` (synchronizes, reads the counter). The row stores energy_j, time_s, energy per repetition = energy_j / R, and average power = energy_j / time_s.
7. **Context** (revised 2026-10-16, after the smoke test). A background thread samples NVML every 0.1 s during each window. The row stores the mean SM and memory clocks over the window, the maximum temperature, the enforced power limit, and `throttle_reasons`: the union (bitwise OR) of NVML's clocks-event reasons over all samples, written as names (`sw_power_cap`, `sw_thermal`, `hw_slowdown`, `hw_thermal`, `hw_power_brake`, `gpu_idle`, ...). Kaggle grants no root, so clocks cannot be locked. A pass is flagged `throttled` if its mean SM clock is below 90% of the GPU's maximum SM clock (NVML; 1,590 MHz on the T4). Throttled passes are flagged, never dropped. The original rule (compare with pass 0 of the same unit) was replaced because the smoke test showed pass 0 is already power-capped, so that rule flagged 5 of 150 passes while every unit after the first ran at 27 to 76% of the maximum SM clock.
8. **Status.** `ok`; `short_window` if the measured window came in under 2 s despite calibration (kept, flagged for the analysis); `oom` if the unit ran out of GPU memory; `error:<ExceptionType>` if the unit failed for any other reason (for example torch.compile unavailable on the GPU; added 2026-10-17). Both write one row per remaining pass, free memory, and move on, so one failing unit cannot stop an unattended 12 h run. They are never silently dropped.
9. **Idle baseline** (decided 2026-10-16). After pre-conditioning, a 10 s idle window (device synchronized, then sleep inside a Zeus window) at the start of each session and whenever 30 minutes have passed since the last one, checked before every pass. Idle windows go to `data/raw/idle.csv`. Shares use total energy. The idle power x window time correction (dynamic energy) is an appendix sensitivity check only.
10. **Persistence and resume.** Every pass is appended to `passes.csv` and the file is closed immediately, so a killed session loses at most the pass in progress. With `--resume`, any (configuration, unit, pass index) already present, of any status, is skipped. A configuration is never split across sessions (decided 2026-10-16, after the second smoke test showed absolute energy shifting 16% between sessions; Section 12). If an earlier session left a configuration incomplete, the resumed session measures all of that configuration's units again from pass 0. The earlier partial rows stay in the append-only file but are not used: the analysis takes, per configuration, the one run id that covers every unit and pass. Without `--resume`, all passes are appended again under a new run id, for deliberate reruns. Before appending, the harness checks that an existing `passes.csv` or `idle.csv` has exactly the current column layout and stops with an error otherwise, so rows are never written under a stale header (for example, the earlier smoke-test schema).
11. **Device identity.** Kaggle provides two T4s. The harness sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` before CUDA initializes and asserts that NVML device 0 and CUDA device 0 have the same PCI bus id. That way the energy counter, the clock readings and the workload all refer to the same physical GPU.

### 6.1 Kaggle procedure

`notebooks/kaggle_profile.ipynb`:
1. Clone the repository at a chosen ref and run `pip install -e .[gpu]`.
2. Print `nvidia-smi` identity, driver, power limit and maximum SM clock.
3. Copy `passes.csv` / `idle.csv` from a previous session's output (attached as notebook input) into the output folder.
4. Check that Zeus reads nonzero energy over a matmul loop; abort otherwise.
5. Run `anatomy profile --resume --out /kaggle/working/raw`.
6. Print a status and per-configuration summary.

The first session runs a smoke test (`SMOKE = True`): one 160M configuration into a separate `smoke/` folder, so smoke rows never mix with the real run.

## 7. Data schemas

`data/raw/passes.csv` (append-only, one row per pass):
`run_id, ts, model, seq_len, precision, tokens, unit, unit_kind{component|substitution|block}, pass_idx, reps, energy_j, time_s, energy_per_rep_j, avg_power_w, sm_clock_mhz, mem_clock_mhz, temp_c, power_limit_w, throttled, throttle_reasons, status{ok|short_window|oom|error:<type>}, torch, cuda, driver, gpu_uuid, git_sha`

`sm_clock_mhz` and `mem_clock_mhz` are window means, and `temp_c` is the window maximum (Section 6, step 7). `throttle_reasons` was added on 2026-10-16 after the first smoke test. `gpu_uuid` (NVML's identifier of the physical GPU) was added on 2026-10-16 after the second smoke test, so rows can be grouped by device. `power_limit_w` was added to the DESIGN section 9 schema on 2026-10-16, because DESIGN section 6 requires recording the power limit. Substitution unit names in this file are `sdpa_core, linear_core, gqa_proj, factorized_embed_in, factorized_embed_out, rmsnorm`.

`data/raw/idle.csv`: `run_id, ts, energy_j, time_s, avg_power_w, sm_clock_mhz, mem_clock_mhz, temp_c, power_limit_w, throttle_reasons, gpu_uuid`

`data/raw/smoke/` holds the first smoke test (Section 12) in the earliest schema: no `throttle_reasons` or `gpu_uuid` column, clocks and temperature read once right after each window, and `throttled` computed against pass 0. `data/raw/smoke-2/` holds the second smoke test: current schema except `gpu_uuid`.

`data/derived/analytical.csv`: `model, seq_len, precision, component, count_per_model, flops, bytes, c1, ridge, memory_bound, c2, t_hat`. Here `flops`, `bytes` and `t_hat` are per unit instance, forward + backward. `component` holds the unit name, including substitutions and `block`.

## 8. Verification

`uv run pytest` (54 tests):

| Test | What it establishes |
|---|---|
| test_param_counts_match_published (x3) | Our modules assemble to exactly the published Pythia parameter counts, total and non-embedding. |
| test_grid | The grid has 18 configurations, each with batch x seq_len = 8,192 tokens. |
| test_linear_flops_and_bytes_by_hand | The traced count for out_proj equals the hand formula: 3 x 2nkd FLOPs and the exact reads and writes of the forward, dX and dW matmuls. |
| test_layernorm_flops_and_bytes_by_hand | The traced LayerNorm count equals the hand formula (23 FLOPs per element; exact bytes). |
| test_c2_sums_to_one | C2 over the baseline components sums to 1 in every configuration. |
| test_memory_bound_flags | Embedding, LN, RoPE, loss and RMSNorm are memory-bound everywhere; attn_proj, mlp and out_proj nowhere. |
| test_row_order_invariant | Results do not depend on row order. |
| test_every_unit_steps_on_cpu (x15) | Every unit runs forward + backward and leaves no stale gradients. |
| test_block_matches_hf_gptneox_layer | Our block matches HF `GPTNeoXLayer` (transformers 5.19, eager attention) within 1e-4 for identical weights, at 160M dimensions, FP32, CPU. |
| test_sdpa_core_matches_attn_core | The fused path equals eager attention within tolerance (CPU math backend; the GPU uses the memory-efficient kernel). |
| test_linear_core_matches_quadratic_reference | The chunked linear attention equals the direct quadratic causal formulation (float64). |
| test_substitution_shapes_and_gqa_heads | Each substitution produces outputs of the same total size as the unit it replaces; kv heads are 3, 4 and 8. |
| test_dry_run_full_grid_and_resume | The harness runs the full unit set on a tiny grid end to end with a fake monitor; resume adds nothing; a configuration left incomplete is measured again in full in one new session, and the other configurations are untouched. |
| test_oom_is_logged_and_run_continues | A unit that runs out of memory is logged as `oom` for each pass, and later units still run. |
| test_gpu_window_samples_clocks_and_reasons | With stub NVML and Zeus objects, the GPU window averages the sampled clocks, takes the maximum temperature, and ORs the throttle-reason bits. |
| test_decision_rule (x6) | The decision rule at its boundaries: 10/18 is a majority and 9/18 is not; median C4 equal to the IQR fails (strict); positive C4 inside the spread fails; a component never memory-bound is "not tested"; only memory-bound configurations are counted. |
| test_iqr_is_linear_type7 | The IQR uses linear (type 7) quantiles: IQR(1..10) = 4.5. |
| test_spearman | Spearman rho is +1 and -1 for monotone data and handles ties with average ranks. |
| test_select_session_takes_the_complete_run | Per configuration, the analysis keeps the session that covers every unit and pass, and fails if none does. |
| test_share_identities | On the measured data, per configuration and pass: sum C3 + Unattributed = 1, sum C4 = -Unattributed, and the component-normalized C3 sums to 1, for total and dynamic energy. |
| test_summary_invariant_to_row_order | Shuffling the raw rows changes no median and no rank. |
| test_substitution_delta_by_hand | The substituted model total and delta % equal the hand calculation, with negative meaning the substitution costs more; unmeasured substitutions are skipped. |
| test_rmsnorm_fused_matches_eager | The fused and eager RMSNorm units compute the same function. |
| test_quadrant_split | F8 quadrant boundaries: C4 = 0 is not waste; the median capability rank falls in the lower-capability half. |
| test_rank_shift_sign | A component moving from rank 5 to 2 has shift +3 (climbs toward more waste). |
| test_value_parsing, test_export_copies_derived_values_without_computing | The JSON export is valid (no NaN), covers 18 configurations, and copies values from `shares_summary.csv` unchanged. |
| test_rope_fused_matches_eager | The compiled rope computes the same function as the eager rope. |
| test_removal_substitution_costs_nothing | NoPE's substituted total is the baseline minus rope's energy; unmeasured substitutions are skipped. |
| test_failing_unit_is_logged_and_run_continues | A non-OOM failure is logged as `error:<Type>` for each pass, and later units still run. |
| test_refuses_to_append_under_a_different_header | An output file with a different column layout stops the run and is left untouched. |

Verified on the T4 by the smoke test (Section 12): the Zeus energy counter, the NVML/CUDA device pinning, the memory-efficient attention kernel, and the full harness on one configuration. Not yet verified on the T4: the in-window NVML sampling and the pre-conditioning added after the smoke test (next Kaggle session), and the GPU-specific operator decomposition.

## 9. Analytical outcome (before measurement)

All values are analytical (Section 4), across the 18 configurations. They are not measurements.

| Unit | Memory-bound configs | C1 range (FLOP/B) | C2 range |
|---|---|---|---|
| embed_lookup | 18/18 | 0.0175 to 0.035 | 9.5e-08 to 9.6e-07 |
| ln | 18/18 | 1.15 to 2.3 | 1.4e-04 to 5.5e-04 |
| attn_proj | 0/18 | 199 to 1760 | 0.175 to 0.316 |
| rope | 18/18 | 0.054 to 0.107 | 1.8e-05 to 6.9e-05 |
| attn_core | 18/18 | 9.19 to 38.5 | 0.020 to 0.237 |
| mlp | 0/18 | 161 to 1470 | 0.350 to 0.633 |
| out_proj | 0/18 | 346 to 2590 | 0.029 to 0.290 |
| loss | 18/18 | 0.321 to 0.643 | 1.1e-05 to 5.7e-04 |
| sdpa_core | 6/18 | 49.7 to 398 | 0.012 to 0.151 |
| linear_core | 18/18 | 3.26 to 8.63 | 0.008 to 0.019 |
| gqa_proj | 0/18 | 110 to 1050 | 0.117 to 0.224 |
| factorized_embed_in | 9/18 | 32.4 to 107 | 7.5e-05 to 9.7e-04 |
| factorized_embed_out | 9/18 | 62.7 to 125 | 1.0e-03 to 0.064 |
| rmsnorm | 18/18 | 0.125 to 0.25 | 9.3e-05 to 3.6e-04 |
| block | 4/18 | 26.2 to 600 | n/a |

Observations for the Results and Discussion chapter:
- **H1 coverage.** All four tested components (embedding lookup, layer normalization pooled with the final norm, RoPE, attn_core) are memory-bound in all 18 configurations, so the decision rule evaluates each on all 18. The attention split is therefore not confounded by sequence length: eager attn_core is memory-bound even at seq 512.
- **Near the ridge.** attn_core's C1 reaches 38.5 in FP16 against an effective ridge of 193 to 198. In FP32 it reaches 19.2 against 25.3, the closest any tested component comes to its ridge (factor 1.3).
- **C2 near zero.** Embedding, LN, RoPE and loss have C2 between 1e-7 and 6e-4. Any measured energy share for them will make C4 roughly equal to C3. This is the "C2 near 0" situation DESIGN section 8 guards against with energy per FLOP.
- **Substitutions move units across the ridge.** sdpa_core is compute-bound except in FP16 at seq 512 and 1024 (6/18). Both factorized_embed units are memory-bound exactly in the 9 FP16 configurations. linear_core stays memory-bound everywhere.
- **The block.** The full block is memory-bound in 4 FP16 configurations (160M at every sequence length, 1.4B at 2048), driven by the s x s attention traffic.

## 10. Limitations

1. **Bytes are an upper bound at operator level.** Operators implemented as several GPU kernels (for example LayerNorm backward reads x and dy twice) are undercounted relative to a kernel-level count. Tile re-reads inside fused kernels (memory-efficient attention) are not counted. Operators that fuse on the GPU but not on the meta device would be overcounted.
2. **Elementwise FLOP constants are conventions** (Section 4.3); C1 for memory-bound units is meaningful only to an order of magnitude.
3. **Analytical counts come from the meta device.** Autograd formulas are device-independent, but backend kernel choices are not observed. In particular, the stride-restoring copies autograd inserts for attention gradients are counted as traced.
4. **Random weights.** Data-dependent power effects of trained weights are not captured.
5. **The T4 is power-capped and its clocks cannot be locked.** The smoke test showed every unit drawing 63 to 70 W against the 70 W cap, with SM clocks at 27 to 76% of maximum after the first unit. Three consequences follow. Energy shares closely track time shares (power is nearly constant across units). Compute-bound units run well below the boost-clock peak that sets the analytical ridge, so measured times exceed t_hat by about 2.6 to 2.9x for matmul units. And clock state depends on recent thermal history, which pre-conditioning (Section 6, step 0) controls but cannot remove. Clocks and throttle reasons are recorded per pass, and throttled passes are flagged, not dropped.
6. **Isolation.** Units measured in isolation see cold inputs from rotating buffers. In a real training step some inputs are warm in L2. Unattributed energy (E_total minus the sum of components) bounds the combined effect of isolation and of the work outside every component.
7. **Measurement windows span R repetitions** (at least 2 s), because the NVML energy counter updates far more slowly than a single small kernel runs. Energy per repetition is an average over the window, not a per-kernel reading.
8. **(Did not occur; see Section 13.) Expected OOM at the largest configurations.** The 6.9B block in FP32 at seq 2048 holds several float32 attention-score tensors of about 2 GB each and is expected to exceed 16 GB. If it does, that configuration has no block measurement. It is logged as `oom` and must be reported as a gap, not interpolated.
9. **Absolute energy is not reproducible across sessions to better than about 16%.** Two smoke runs of the same configuration, with identical software, differed by 16% in total energy, with higher clocks in every unit of the second run (Section 12). The likely cause is a different physical T4 or different cooling; the first two runs did not record the device. Shares (C3, C4) were stable to within 0.011, so the thesis conclusions rest on shares, and each configuration is measured within one session (Section 6, step 10). Absolute joules are reported as session-specific.
10. **Unattributed energy can be negative.** It is negative in 5 of 18 configurations in run 3 (down to -2.8%), 3 in run 2 (down to -1.6%) and 8 in run 1 (down to -6.7%) (Section 17). The isolated components then sum to more than the model total. The most plausible causes are that isolated units read cold inputs from rotating buffers, while inside the block some inputs are still in L2, and that isolated units launch separate kernels for work the block fuses or overlaps. Since the C4 values sum to -Unattributed, a negative Unattributed raises the C4 values by |Unattributed| in total, but how that overcount splits across components is not observable. If it were split equally (|Unattributed| / 8 each, at most 0.0035 in run 3), it would exceed embed_lookup's margin over the decision threshold in 4 run-3 configurations (1.4B/1024/FP32, 1.4B/2048/FP32, 6.9B/512/FP32, 6.9B/2048/FP32) and ln's in 1 (6.9B/512/FP32), leaving ln supported in 17 of 18; rope and attn_core keep their margins. If it is split in proportion to each component's energy (TA3), all four tested components remain supported in 18 of 18 configurations in all three runs. The embed_lookup verdict is therefore the least robust of the four and should be reported with this caveat.

11. **The output-side factorized embedding is untested in the literature.** factorized_embed mirrors ALBERT's input factorization at the output (`Linear(d, 128)` + `Linear(128, V)`), which is this thesis's extension. ALBERT (Lan et al., 2020) reports the quality cost of input factorization only (-0.6 average at E = 128, Table 3). The quality cost of a rank-128 output layer for a decoder LM is not established here. Chung et al. (2021, Table 4) found that shrinking the output embedding from 768 to 128 cost 3.6 average points in a multilingual encoder, which suggests it may be larger than on the input side. The thesis measures only its energy.

## 11. Reproducing

```
uv sync                        # Python 3.11 env from uv.lock
uv run pytest                  # 54 tests
uv run anatomy analyze         # shares, C4, rule, substitutions, capability: T4-T10, TA1-TA3, F5, F7-F10
uv run anatomy export          # data/export/*.json for the dashboard
uv run anatomy analytical      # data/derived/analytical.csv
uv run anatomy report          # data/tables/T2.*, data/figures/F6.*
uv run anatomy profile --dry-run   # harness end to end on CPU, fake energy, temp dir
```
On Kaggle: run `notebooks/kaggle_profile.ipynb` (Section 6.1). Results land in `/kaggle/working/raw/` and are copied into `data/raw/` in the repository.

## 12. Smoke tests on the Kaggle T4 (2026-10-16)

### 12.1 First smoke test

One configuration (Pythia 160M, seq 512, FP32), all 15 units x 10 passes, run on Kaggle with commit `44ef812` and no uncommitted changes. Session context: torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04, Tesla T4, power limit 70 W. Raw files: `data/raw/smoke/passes.csv`, `data/raw/smoke/idle.csv` (run id `20cd96fc`). These are real measurements, but from one configuration only. They are a protocol check, not a thesis result.

**Pipeline.** All 150 passes have status `ok`. Windows lasted 2.11 to 2.72 s (target at least 2 s). R ranged from 4 (out_proj) to 2,810 (ln). Zeus read nonzero energy throughout. The single idle window measured 31.2 W at 585 MHz, 56 C. The run took 6 min 12 s.

**Repeatability.** The interquartile range of energy per repetition, divided by its median, ranged from 0.001 (loss, sdpa_core) to 0.049 (attn_proj) across units.

**Power cap.** Median average power was 64.6 to 68.9 W in every unit (cap 70 W). Temperature rose from 62 C to 80 C within 6 minutes. The first unit of the session (embed_lookup, on a cool GPU) ran at 1,500 to 1,560 MHz. After it, the SM clock depended on the unit:

| SM clock (median over passes) | Units |
|---|---|
| 1,035 to 1,208 MHz | rope, rmsnorm, sdpa_core, loss |
| 578 to 795 MHz | ln, attn_proj, attn_core, mlp, out_proj, linear_core, gqa_proj, factorized_embed_in, block |
| 450 MHz | factorized_embed_out |

With power pinned at the cap, the clock settles at whatever the unit's workload can sustain within 70 W, so matmul-heavy units get the lowest clocks. That makes clock rate an outcome of the power cap rather than an independent nuisance variable, which matters for interpreting C4 on this GPU. The cool start of the session and the inadequate throttle flag led to two protocol changes (Section 6, steps 0 and 7).

**Measured time vs analytical t_hat** (t_hat divided by the measured time per repetition, i.e. the fraction of the analytical roofline achieved): embed_lookup 0.65, ln 0.49, attn_proj 0.38, rope 0.58, attn_core 0.46, mlp 0.38, out_proj 0.35, loss 0.61, block 0.32; substitutions sdpa_core 0.18, linear_core 0.62, gqa_proj 0.37, factorized_embed_in 0.29, factorized_embed_out 0.33, rmsnorm 0.74. The matmul units' 0.35 to 0.38 is consistent with SM clocks at about 40% of the 1,590 MHz boost clock that the 8.1 TFLOP/s peak assumes.

**Energy closure.** E_total = 12 x E_block + E_embed + E_ln + E_out_proj + E_loss = 167.47 J per forward and backward over 8,192 tokens. The components sum to 164.58 J, so Unattributed is 1.7%.

**Preview of shares (one configuration, medians over 10 passes, not a result):**

| Component | C2 | C3 | C4 = C3 - C2 | Memory-bound |
|---|---|---|---|---|
| embed_lookup | 0.0000 | 0.0007 | +0.0007 | yes |
| ln (2L + 1) | 0.0006 | 0.0080 | +0.0075 | yes |
| attn_proj | 0.2123 | 0.1731 | -0.0392 | no |
| rope | 0.0001 | 0.0177 | +0.0177 | yes |
| attn_core | 0.0719 | 0.1369 | +0.0650 | yes |
| mlp | 0.4251 | 0.3575 | -0.0676 | no |
| out_proj | 0.2895 | 0.2657 | -0.0238 | no |
| loss | 0.0006 | 0.0231 | +0.0225 | yes |
| Unattributed | | 0.0172 | | |

In this configuration every memory-bound component has C4 > 0 and every compute-bound component has C4 < 0. The decision rule needs the full grid and the per-pass IQR (DESIGN section 8), so this is not yet evidence for H1.

### 12.2 Second smoke test (after the protocol changes)

The same configuration (Pythia 160M, seq 512, FP32), run on Kaggle with commit `6a80ad8` (no uncommitted changes), with thermal pre-conditioning and in-window NVML sampling. Same software as the first run: torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04. Raw files: `data/raw/smoke-2/` (run id `a59dc0ff`). The run took 6 min 5 s after pre-conditioning.

**Protocol changes behave as intended.** The idle window after pre-conditioning ran at 78 C (31.1 W), and every pass ran at 73 to 80 C, so the first unit no longer runs on a cool GPU. All 150 passes are `ok`, with windows of 2.16 to 2.64 s. NVML reported `sw_power_cap` as the throttle reason in all 150 passes, which confirms the power cap directly. Against the 90%-of-maximum rule, 141 of 150 passes are `throttled`. On the T4 this flag marks the normal operating state, and `throttle_reasons` together with the mean clock carry the information. The spread of energy per repetition across passes (IQR divided by median) was 0.001 to 0.047 per unit.

**Clocks averaged over each window are higher than the single post-window reading of the first run.** Per-unit median SM clocks were 715 to 1,494 MHz, against 450 to 1,522 MHz in the first run. Part of the difference is the measurement change: reading once after the window ends catches the GPU as it leaves the load.

**Cross-session shift in absolute energy.** E_total was 140.75 J against 167.47 J in the first run (-16%). Energy per repetition fell by about 15 to 21% for matmul-heavy units (attn_proj 0.853, mlp 0.845, out_proj 0.808, block 0.846, factorized_embed_out 0.793, as ratios of run 2 to run 1) and changed by -3% to +5% for embed_lookup, ln, rope, loss, sdpa_core and rmsnorm. Software was identical, so the shift is in the hardware or its cooling. The first two runs did not record which physical GPU they used, which led to the `gpu_uuid` column and to measuring each configuration within one session (Section 6, step 10).

**Shares were stable.** Unattributed was 0.7% (first run 1.7%). C4 per component, run 1 then run 2:

| Component | C2 | C4 run 1 | C4 run 2 |
|---|---|---|---|
| embed_lookup | 0.0000 | +0.0007 | +0.0008 |
| ln (2L + 1) | 0.0006 | +0.0075 | +0.0087 |
| attn_proj | 0.2123 | -0.0392 | -0.0365 |
| rope | 0.0001 | +0.0177 | +0.0220 |
| attn_core | 0.0719 | +0.0650 | +0.0696 |
| mlp | 0.4251 | -0.0676 | -0.0659 |
| out_proj | 0.2895 | -0.0238 | -0.0339 |
| loss | 0.0006 | +0.0225 | +0.0279 |

Every sign was preserved, and the largest change was 0.0101 (out_proj). This is the basis for treating shares, not absolute joules, as the reportable quantity (Section 10, limitation 9).

## 13. Full profiling runs

### 13.1 Run 1 (2026-10-16, now `data/raw/run-1/`)

The whole grid ran in one Kaggle session: run id `0cd31197`, one physical GPU (`gpu_uuid` ending `a4dce36505e2`), commit `52f319a` with no uncommitted changes, torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04. Passes ran from 10:38:16 to 12:50:12 UTC (2 h 12 min, against a DESIGN estimate of 4 to 6 h). Raw files: `data/raw/run-1/passes.csv` (2,700 rows), `data/raw/run-1/idle.csv` (5 rows). Run 1 had 15 units (no rmsnorm_fused).

**Coverage.** 18 configurations x 15 units x 10 passes = 2,700 rows, exactly 150 per configuration, with no duplicate (configuration, unit, pass) keys. No configuration was split across sessions.

**Status.** 2,699 `ok` and 1 `short_window`: factorized_embed_out at Pythia 1.4B, seq 2048, FP16, pass 3, with a 1.991 s window at R = 65 (target 2 s). There were no `oom` rows. The 6.9B block in FP32 at seq 2048 fit in 16 GB, contrary to the earlier expectation (Section 10, limitation 8, which is now moot).

**GPU state.** `throttle_reasons` was `sw_power_cap` in all 2,700 passes, and temperature stayed at 74 to 82 C. Windows lasted 2.00 to 4.91 s. The five idle windows (every 30 min) measured 34.2 to 36.3 W at 75 to 77 C.

**Repeatability.** Across the 270 (configuration, unit) groups, IQR / median of energy per repetition has a median of 0.014. The largest values are 0.108 (mlp, 6.9B, 512, FP16), 0.104 (ln, 1.4B, 1024, FP16), 0.083 (ln, 6.9B, 1024, FP16) and 0.078 (block, 160M, 2048, FP32).

### 13.2 Run 2 (2026-10-16, now `data/raw/run-2/`)

Rerun after adding rmsnorm_fused (Section 16, decision 27). One Kaggle session: run id `4360bc50`, a different physical GPU from run 1 (`gpu_uuid` ending `59154059ef9e`), commit `33799b8` with no uncommitted changes, torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04. Passes ran from 17:22:18 to 19:35:34 UTC (2 h 13 min).

**Coverage.** 18 configurations x 16 units x 10 passes = 2,880 rows, exactly 160 per configuration, with no duplicates and no `oom`.

**Status.** 2,877 `ok` and 3 `short_window`, all factorized_embed_in at 160M/512/FP16 (passes 3, 7 and 8, at 1.979, 1.9995 and 1.991 s). They are kept, flagged, under the same rule as decision 26.

**GPU state.** `sw_power_cap` in all 2,880 passes, at 73 to 82 C.

### 13.3 Run 3, the main dataset (2026-10-17, `data/raw/`)

Rerun after adding rope_fused (decision 32). A first attempt stopped 49 s in, before any measurement, at a notebook provenance cell that imported the package into a kernel started before `pip install -e` (fixed in commit `dc321d9`). The completed run is one Kaggle session: run id `25dd8e9d`, a third physical GPU (`gpu_uuid` ending `f55f66e52865`), commit `dc321d9` with no uncommitted changes, torch 2.11.0+cu128, CUDA 12.8, driver 580.178.04. Passes ran from 12:18:53 to 14:45:28 UTC (2 h 27 min).

**Coverage.** 18 configurations x 17 units x 10 passes = 3,060 rows, exactly 170 per configuration, with no duplicates, no `oom` and no `error` rows. rope_fused ran in all 18 configurations.

**GPU state.** `sw_power_cap` in all 3,060 passes, at 74 to 81 C. The five idle windows measured 36.2 to 37.8 W.

**Short windows.** 24 passes came in under 2 s (1.512 to 1.999 s): rope_fused at 160M/512/FP16 (all 10 passes) and 160M/1024/FP16 (2), factorized_embed_in at 160M/512/FP16 (8 of 10), and ln at 6.9B FP16 (4). They are the fastest units, where the calibration probe overestimates time per repetition and so sets R slightly low. They are kept, flagged (decision 26). **Sensitivity:** dropping every short-window pass (and its matched set) leaves all four H1 verdicts at 18 of 18, moves no median C4 by more than 0.005 and no substitution delta by more than 0.16 percentage points, and loses only rope_fused at 160M/512/FP16.

**rope_fused ran fused.** The notebook's kernel-listing cell shows Inductor compiled the rope into exactly two Triton kernels, one forward and one backward, in both precisions (Section 16.1; verbatim output in `data/raw/provenance/run3_rope_fused_kernels.txt`). Per instance, rope_fused used 0.22 to 0.45 times the energy of the eager rope (median 0.26).

## 14. Analysis method as implemented

Implemented in `src/anatomy/analysis.py`; run with `anatomy analyze`. Inputs: `data/raw/passes.csv`, `data/raw/idle.csv`, `data/derived/analytical.csv`. Outputs: `data/derived/shares.csv` (per pass, total energy), `shares_dynamic.csv` (per pass, dynamic energy), `shares_summary.csv` (per configuration and component), tables T4 to T7 and the appendix tables TA1 to TA3, and figures F5 and F7.

1. **Session selection.** Per configuration, the analysis uses the one run id whose rows cover every baseline component and the block for all 10 passes (the latest such run id if a configuration was deliberately rerun). It fails if no session covers a configuration. In the full run, every configuration came from run `0cd31197`.
2. **Usable passes.** Status `ok` and `short_window` (flagged, kept). `oom` rows carry no energy. If any unit lacks pass j, the whole matched set j of that configuration is dropped. No set was dropped in the full run.
3. **Per-pass shares** (DESIGN section 8). For pass j of configuration k, E_c,j = (energy per repetition of c in pass j) x count per model, and E_total,j = L x E_block,j + E_embed,j + E_ln,j + E_out_proj,j + E_loss,j, where the single E_ln instance is the final norm. Then C3 = E_c,j / E_total,j, Unattributed_j = 1 - sum of C3, and C4 = C3 - C2. This gives 10 values of C3 and C4 per component and configuration.
4. **Summary.** Median and IQR over the 10 passes. The IQR is Q3 - Q1 with linear (Hyndman-Fan type 7) quantiles, numpy's default (decided 2026-10-17). `shares_summary.csv` also holds C1, C2, the memory-bound flag, energy per FLOP (median energy per repetition / analytical FLOPs), the median measured power, and the rank of the component by median C4 within its configuration (1 = largest C4).
5. **Decision rule (T6).** For each tested component (embed_lookup; ln pooled with the final norm; rope; attn_core): among configurations where it is memory-bound, count those with median C4 > 0 and median C4 > IQR(C4), both strict. The component is *supported* if that count is a strict majority of its memory-bound configurations, *not supported* otherwise, and *not tested* if it is never memory-bound.
6. **Validation (T4).** The analytical estimate is t_hat x count x P_class, where P_class is the mean of the measured median powers of the configuration's components in the same roofline class (memory-bound or not). It is normalized over the 8 components and compared with the measured C3 renormalized over the 8 components, so both sides sum to 1 (decided 2026-10-17: the analytical model has no Unattributed term). T4 reports the absolute difference per component and Spearman rho across the 8 components per configuration (descriptive).
7. **Dynamic energy (appendix TA1, TA2).** Per pass, dynamic energy = energy_j - P_idle x time_s, where P_idle is the power of the nearest idle window at or before the pass in the same session (decided 2026-10-17). C3, C4 and the decision rule are recomputed on dynamic energy.
8. **Normalized sensitivity (appendix TA3).** The decision rule recomputed with C4 = (C3 renormalized over the 8 components) - C2. This spreads any Unattributed energy over the components in proportion to their energy (limitation 10).

## 15. Baseline results (run 3)

All values come from run 3 (`data/derived/shares_summary.csv`, tables T4 to T7). Section 17 compares them with runs 1 and 2.

**Decision rule (T6).** All four tested components are memory-bound in all 18 configurations, and each has median C4 > 0 and median C4 > IQR(C4) in all 18. Verdict: supported for embed_lookup, layer normalization (pooled), rope and attn_core. The result is identical on dynamic energy (TA2; switching moved the tested components' median C4 by -0.0115 to +0.0145), with C3 renormalized over the 8 components (TA3; smallest margins over the IQR: embed_lookup 5.3e-5, ln 2.2e-3, rope 3.9e-3, attn_core 5.5e-3), without the short-window passes (Section 13.3), and in runs 1 and 2 (Section 17).

**Waste score C4 by component (median over passes; range over the 18 configurations):**

| Component | Memory-bound | C4 range | Configs with C4 > 0 | Rank by C4 within config | Energy per FLOP (J) |
|---|---|---|---|---|---|
| attn_core | 18/18 | +0.0069 (6.9B/512/FP32) to +0.3118 (160M/2048/FP16) | 18 | 1 to 2 | 1.2e-11 to 5.1e-11 |
| rope | 18/18 | +0.0040 to +0.0424 | 18 | 2 to 3 | 3.8e-09 to 6.7e-09 |
| ln (pooled) | 18/18 | +0.0022 to +0.0232 | 18 | 3 to 4 | 2.7e-10 to 4.7e-10 |
| loss | 18/18 | +0.0004 to +0.0433 | 18 | 2 to 5 | 5.2e-10 to 1.0e-09 |
| embed_lookup | 18/18 | +0.00005 to +0.0020 | 18 | 5 to 6 | 1.1e-08 to 1.9e-08 |
| mlp | 0/18 | -0.1757 to +0.0679 | 3 | 1 to 8 | 3.7e-12 to 3.1e-11 |
| out_proj | 0/18 | -0.1229 to +0.0082 | 7 | 4 to 7 | 4.4e-12 to 2.4e-11 |
| attn_proj | 0/18 | -0.0958 to -0.0128 | 0 | 6 to 8 | 3.7e-12 to 2.3e-11 |

For every tested component, the smallest C4 occurs at Pythia 6.9B in FP32 (2048 for embed_lookup, ln, rope and loss; 512 for attn_core) and the largest at 160M in FP16.

Observations for the Results and Discussion chapter:
- **attn_core carries the largest waste among memory-bound components.** It ranks first by C4 in 15 of 18 configurations and second in the three 6.9B FP32 configurations. Its C4 grows with sequence length and is largest in FP16, reaching +0.312 at 160M/2048/FP16, where it takes 55% of measured energy against a 24% compute share (F5).
- **mlp at 6.9B FP32 is the exception.** There the compute-bound mlp has the largest C4 (+0.038 to +0.068, rank 1 in all three). The same happens in run 1 but not in run 2, where mlp's C4 at those configurations was -0.002 to +0.012 (Section 17). Its C4 is positive in 8 of the 9 run-configuration pairs at 6.9B FP32, so the sign is fairly consistent while the size and rank are not. It does not affect the H1 verdicts (mlp is not tested), but it shows C4 > 0 can arise for a compute-bound unit running below its ridge-implied efficiency at the power-capped clock.
- **out_proj** is slightly positive in 7 configurations (at most +0.008) and **attn_proj** is negative everywhere.
- **The C2 near 0 situation.** embed_lookup, rope, ln and loss have C2 between 1e-7 and 6e-4, so their C4 is almost entirely their C3. Energy per FLOP separates them from the matmul units by one to four orders of magnitude (for example ln 2.7e-10 to 4.7e-10 J/FLOP against mlp 3.7e-12 to 3.1e-11), so their positive C4 reflects energy spent per unit of work, not a division artifact.
- **Unattributed** ranges from -2.8% (6.9B/512/FP32) to +4.8% (1.4B/512/FP16), with a median of +1.0%. It is negative in 5 configurations (1.4B/1024/FP32, 1.4B/2048/FP32, 6.9B/512/FP32, 6.9B/2048/FP32, 6.9B/2048/FP16). See limitation 10.

**Validation (T4).** The analytical model ranks the 8 components almost exactly as measured. Spearman rho is 1.0 in 12 configurations, 0.976 in 5 (all FP16) and 0.952 at 6.9B/2048/FP16, identical in all three runs. The median absolute difference between analytical and measured component-normalized C3 is 0.009. The largest differences are for attn_core (median 0.041, at most 0.148 at 160M/1024/FP16) and mlp (median 0.052, at most 0.098). The analytical model gets the ordering right but not the size of attn_core's share, which is the quantity the measurement exists to provide.

## 16. Substitution analysis method

Implemented in `src/anatomy/analysis.py` (`substitution_deltas`, `summarize_substitutions`, `priority_table`); runs as part of `anatomy analyze`. Outputs: `data/derived/substitutions.csv` (per pass), tables T9 and T10, figure F10.

1. **Same session.** Substitution units are taken from the same session as the baseline of that configuration (Section 14, step 1). A substitution not measured in that session is skipped, not borrowed from another session, because absolute energy shifts between sessions (Section 12.2).
2. **Swapped model total, per matched pass** (DESIGN section 5). E_total,sub = E_total,base - sum over replaced units of (E_replaced x count) + sum over substitute units of (E_sub x count), where E_total,base is the baseline total of Section 14 for the same pass j. factorized_embed swaps its in and out units together (embed_lookup and out_proj). The rmsnorm variants replace all 2L + 1 norms.
3. **Saving.** delta % = (E_total,base - E_total,sub) / E_total,base x 100. Positive means the substitution saves energy. The table reports the median and IQR (linear) over passes, the replaced units' share of the baseline total, the substitute's share of the substituted total, and whether the substitute is memory-bound (analytical).
4. **Removals.** NoPE removes rope outright, so nothing is measured in its place: E_total,sub = E_total,base - E_rope x L for the same pass. Its quality cost is not measured here and comes from Haviv et al. (2022, Table 1): validation perplexity +0.55 (WikiText-103, 247M) and +0.05 (the Pile, 1.3B) against learned positions, for removing all positional encoding rather than Pythia's 25% partial rotary.
5. **Additivity assumption.** The substituted total is computed by swapping isolated-unit energies, not by measuring a full block with the substitution inside. Unattributed energy is therefore assumed unchanged by the substitution.
6. **Short window** (decided 2026-10-17). The one `short_window` pass (factorized_embed_out, 1.4B/2048/FP16, pass 3, 1.991 s) is kept, flagged. Its matched set is not dropped.
7. **T9** lists, per substitution, the priority component it replaces, that component's H1 verdict, its C4 range and C4 rank range (computed), and the mechanism and source (from `data/manual/substitutions.csv`, curated by hand). **T10's `published_value`** also comes from that file. It stays blank until a published figure is entered by hand, and no value is ever filled in by the code.

**Why rmsnorm_fused was added (2026-10-17).** In the first full run, the eager rmsnorm cost more model energy than LayerNorm in all 18 configurations (delta -0.5% to -4.4%). The analytical model shows why: the eager version moves about 6 times the bytes of the fused LayerNorm kernel (for example 2.01 GB against 0.34 GB per call at 1.4B/2048/FP32), because each of its six operators makes a full round-trip to HBM. That compares an eager implementation with a fused one, not the RMSNorm algorithm with the LayerNorm algorithm. rmsnorm_fused computes the same function in one kernel (0.34 GB analytically), so the pair separates the algorithm from the implementation. Adding it required rerunning the full grid in one session, because a unit measured in a later session cannot be compared with a baseline from an earlier one (Section 12.2).

### 16.1 Substitution results (run 3)

The published values in T10 are not comparable with this thesis's measurement, or with each other. Each is the source's own headline metric, measured on other hardware and models, and none is energy. They indicate each technique's claimed benefit, not a prediction of delta E on the T4.

From T10 (median over passes; positive = saving; "saves" counts configurations with median saving > 0 and > IQR):

| Substitution | Replaces | delta % of model energy | Saves | Costs more |
|---|---|---|---|---|
| sdpa_core | attn_core | +0.79 (6.9B/512/FP32) to +41.54 (160M/2048/FP16) | 18/18 | 0 |
| linear_core | attn_core | -1.71 (1.4B/512/FP16) to +45.24 (160M/2048/FP16) | 16/18 | 2 |
| gqa_proj | attn_proj | +2.12 (160M/2048/FP16) to +12.17 (6.9B/512/FP32) | 18/18 | 0 |
| factorized_embed | embed_lookup + out_proj | +1.93 (6.9B/2048/FP32) to +21.27 (160M/512/FP32) | 18/18 | 0 |
| rmsnorm (eager) | ln | -4.50 (160M/512/FP16) to -0.48 (6.9B/2048/FP32) | 0/18 | 18 |
| rmsnorm_fused | ln | -0.54 (1.4B/512/FP16) to +0.48 (160M/512/FP16) | 14/18 | 3 |
| rope_fused | rope | +0.29 (6.9B/2048/FP32) to +3.10 (160M/512/FP16) | 18/18 | 0 |
| nope (removal, computed) | rope | +0.40 (6.9B/2048/FP32) to +4.25 (160M/512/FP16) | 18/18 | 0 |

**Algorithm vs implementation (RMSNorm).** Per instance, within run 3, the eager rmsnorm used 1.7 to 3.7 times the energy of LayerNorm (median 3.0), while rmsnorm_fused used 0.77 to 1.32 times (median 0.93). The fused kernel ran on the T4 (dispatch log below). Replacing LayerNorm with fused RMSNorm changes model energy by -0.54% to +0.48%: it costs slightly more in three 1.4B FP16 configurations, and its saving is within the IQR at 1.4B/1024/FP32. In run 2 it saved 0.00 to 0.44% everywhere (Section 17). The algorithmic difference is therefore close to zero and not consistent in sign, while writing RMSNorm as six eager operators costs 0.5 to 4.5% in every configuration. For the thesis, the implementation of a normalization layer matters more for energy than the choice between LayerNorm and RMSNorm. This agrees in direction with Liger Kernel (Hsu et al., 2024, arXiv:2410.10989), which reports a fused Triton RMSNorm about 7 times faster than an eager HF-style RMSNorm on an A100 at hidden size 16384, with about 3 times less peak memory (Sec. 4.1). The baseline is identified from the repository's benchmark script, since the paper does not name it. That is a speed and memory result on different hardware, not energy.

**Provenance (run 2 and run 3 notebooks, Tesla T4).** ATen operators dispatched by `F.rms_norm` forward + backward, from the dispatch-mode log cell of the run 2 notebook (the run 3 notebook log has not been recorded here):

```
2.11.0+cu128 torch.float32 -> ['aten._fused_rms_norm', 'aten.ones_like', 'aten._fused_rms_norm_backward']
2.11.0+cu128 torch.float16 -> ['aten._fused_rms_norm', 'aten.ones_like', 'aten._fused_rms_norm_backward']
```

(`ones_like` is the test's own gradient seed.) This also confirms the analytical treatment of rmsnorm_fused as one operator forward and one backward (Section 4.1).

**rope (the priority component without a substitution before 2026-10-17).** Fusing rope (rope_fused) saves 0.29 to 3.10% of model energy, in all 18 configurations; per instance it uses 0.22 to 0.45 times the energy of the eager rope. Removing rope entirely (NoPE) would save 0.40 to 4.25%. The fused kernel therefore recovers 56 to 78% of the removal's saving (per configuration, median over matched passes) while keeping the positional encoding, so it carries no quality cost. NoPE's quality cost is not measured here: Haviv et al. (2022) report +0.55 (WikiText-103, 247M) and +0.05 (the Pile, 1.3B) validation perplexity against learned positions, for removing all positional encoding rather than Pythia's 25% partial rotary. Liger Kernel reports its fused RoPE about 8 times faster with about 3 times lower memory than HF's eager `apply_rotary_pos_emb` (A100, hidden 16384; baseline from the repository script), again a kernel speed result, not energy. As with RMSNorm, the eager implementation is the main source of rope's waste.

**Provenance (run 3 notebook, Tesla T4).** CUDA kernels launched by one forward + backward of rope_fused, from the kernel-listing cell (excerpt; full output in `data/raw/provenance/run3_rope_fused_kernels.txt`):

```
2.11.0+cu128 torch.float32 2 triton kernels of 12: [... 'triton_poi_fused_add_cat_mul_neg_slice_0', 'triton_poi_fused_add_mul_neg_slice_slice_backward_0', ...]
2.11.0+cu128 torch.float16 2 triton kernels of 12: [... 'triton_poi_fused_add_cat_mul_neg_slice_0', 'triton_poi_fused_add_mul_neg_slice_slice_backward_0', ...]
```

The rope itself is these two kernels: the forward one fuses the slicing, negation, multiply-add and concatenation, and the backward one fuses their gradients (including the slice-backward zero-padding). The remaining ten entries come from the check itself (`.sum()` reductions, an add, the gradient-seed fill, a copy) and CUDA runtime calls. This confirms the analytical treatment of rope_fused as one fused pass forward and one backward (Section 4.1).

**Attention substitutions** save the most at small models and long sequences in FP16, where attn_core's waste is largest (Section 15). sdpa_core saves energy everywhere. linear_core costs slightly more in 2 configurations (1.4B/512/FP16 and 6.9B/512/FP16) and changes the model, so its quality cost would need separate evidence.

## 17. Repeatability across runs (runs 1, 2 and 3)

The three full runs ran on three different physical T4 GPUs (Section 13) with identical software, apart from rmsnorm_fused (added for run 2) and rope_fused (added for run 3). Baseline shares are compared over the 144 (configuration, component) pairs.

- **Absolute energy is GPU-specific.** Between runs 1 and 2, the ratio of pass-0 energy per repetition had a median of 0.914 and ranged from 0.555 to 1.063 across (configuration, unit). Absolute joules are not comparable across sessions (limitation 9).
- **Verdicts are identical.** All four tested components are supported in 18 of 18 configurations in every run, under total energy, dynamic energy and the normalized check.
- **Tested components' C4 is stable.** The largest range of median C4 across the three runs is 0.0002 for embed_lookup, 0.0059 for ln, 0.0050 for rope, 0.0057 for loss and 0.0169 for attn_core.
- **Reported-only components near zero are not stable.** Across the three runs, C4 disagrees in sign for 9 (configuration, component) pairs, all of them mlp or out_proj. The largest ranges are mlp 0.069 (6.9B/512/FP32), attn_proj 0.040 and out_proj 0.011. At 6.9B FP32, mlp's C4 is positive and rank 1 in all three configurations in runs 1 and 3, but -0.002 to +0.012 in run 2. Statements about compute-bound components' C4 should report this spread.
- **Ranks.** attn_core is rank 1 in 15 configurations in runs 1 and 3 and in all 18 in run 2; where it is not first, mlp is (6.9B FP32).
- **Unattributed** ranges from -6.7% to +4.3% (8 negative) in run 1, -1.6% to +4.1% (3 negative) in run 2 and -2.8% to +4.8% (5 negative) in run 3.
- **Validation** (Spearman rho) is identical in all three runs: 1.0 in 12 configurations, 0.976 in 5 and 0.952 in 1.
- **RMSNorm.** rmsnorm_fused vs LayerNorm: +0.00 to +0.44% in run 2, -0.54 to +0.48% in run 3. The eager rmsnorm costs more in all 18 configurations in all three runs.

## 18. Capability (T8, F8) and configuration moderators (F9)

### 18.1 Method

- **Capability** (Chapter II) is a component's contribution to model quality: how much loss or accuracy suffers when it is removed, simplified, reduced or replaced. The ordinal ranks (1 = largest contribution) and their evidence are in `data/manual/capability.csv`, curated by hand. Every cited number was checked against its primary source by an independent verifier agent (2026-10-17), and caveats are in the `note` column. loss is the training objective and is not ranked.
- **Pooled waste.** For T8 and F8, each component's C4 is pooled as the median over the 18 configurations of its per-configuration median C4 (run 2), with the range over configurations as the whisker.
- **Quadrant** (decision 30). Waste is split at C4 = 0 (positive = high waste). Capability is split at the median rank of the ranked components. With 7 ranked components the median rank is 4 (embed_lookup), which is placed in the lower-capability half, so high capability = ranks 1 to 3.
- **C5 rank shifts** (decision 28; `rank_shifts` in `analysis.py`, `data/derived/rank_shifts.csv`). For each moderator, each pair of configurations that differ only in that moderator is compared: scale 160M to 6.9B (6 pairs), sequence length 512 to 2048 (6 pairs), precision FP32 to FP16 (9 pairs). shift = rank_from - rank_to, so a positive shift means the component climbs toward rank 1 (more waste relative to the others).

### 18.2 Results (run 3)

**Capability ranking with pooled waste (T8):**

| Rank | Component | Pooled median C4 | Range over configs | Quadrant |
|---|---|---|---|---|
| 1 | mlp | -0.0826 | -0.1757 to +0.0679 | low waste, high capability |
| 2 | attn_core | +0.0702 | +0.0069 to +0.3118 | high waste, high capability |
| 3 | out_proj | -0.0054 | -0.1229 to +0.0082 | low waste, high capability |
| 4 | embed_lookup | +0.0004 | +0.0001 to +0.0020 | high waste, low capability |
| 5 | attn_proj | -0.0504 | -0.0958 to -0.0128 | low waste, low capability |
| 6 | rope | +0.0155 | +0.0040 to +0.0424 | high waste, low capability |
| 7 | ln | +0.0105 | +0.0022 to +0.0232 | high waste, low capability |
| - | loss | +0.0037 | +0.0004 to +0.0433 | not ranked |

The quadrants are the same as in run 2.

Observations for the Results and Discussion chapter:
- **High waste, low capability** holds rope, ln and embed_lookup, the priority components of the first proposition. Each now has a matched substitution (Section 16.1): rope_fused and NoPE for rope, rmsnorm and rmsnorm_fused for ln, factorized_embed for embed_lookup. For rope and ln the result is the same: the waste is mostly implementation (eager operators), and a fused kernel removes most of it without changing the function.
- **attn_core is the only high-waste, high-capability component.** It carries the most waste among memory-bound components, but removing it costs quality, so its candidates are substitutes that preserve the function: sdpa_core computes the same attention and saved 0.8 to 41.5% of model energy, while linear_core changes the model.
- **mlp and attn_proj** sit in the low-waste quadrants, consistent with being compute-bound, though mlp's pooled C4 hides its positive values at 6.9B FP32 (Section 17).
- embed_lookup's position rests on a pooled C4 of +0.0004 and on the least robust H1 margin of the four tested components (limitation 10).

**C5 rank shifts (F9):**
- **attn_core barely moves.** It is unchanged in 15 of 21 comparisons; it drops one place under scale and rises one place under precision only where mlp overtakes it at 6.9B FP32.
- **Scale moves the ranking most.** 35 of 48 (component, pair) cells change. loss falls 3 places from 160M to 6.9B, attn_proj falls 1 to 2, out_proj rises up to 3 in FP16.
- **Sequence length barely moves it** (8 of 48 cells). Longer sequences raise attn_core's C4 (Section 15) but it is already rank 1 wherever mlp is not.
- **Precision** changes 22 of 72 cells, mostly by 1 place, and concentrated at 6.9B.
- **The large shifts are mlp at 6.9B FP32:** +7 under scale (from rank 8 at 160M to rank 1) and -7 under precision (from rank 1 in FP32 to rank 8 in FP16). They track mlp's positive C4 at 6.9B FP32, whose size is not stable between runs (Section 17), so they should be reported with that caveat rather than as a clean moderator effect.

## 19. Dashboard export

`anatomy export` (`src/anatomy/export.py`) writes four JSON files to `data/export/` for the dashboard (DESIGN section 9, Phase 7). Values are copied from `data/derived/` and the tables without any computation, so the dashboard shows exactly the thesis numbers.

| File | Content |
|---|---|
| configs.json | the 18 configurations (T2), each with an id `model/seq_len/precision` |
| profile.json | per configuration: for each component C1, C2, memory-bound, C3 median and IQR, dynamic C3, energy per FLOP, median power; plus Unattributed |
| ranking.json | per configuration: C4 median, IQR, dynamic C4 and rank; the decision rule (T6); capability with pooled C4 and quadrant (T8); the C5 rank shifts (F9) |
| meta.json | export time, code commit, GPU, and the measurement run's ids, GPU UUID, commit, software versions and time span |

Substitution results (T10) are not exported; DESIGN section 9 does not list them.
