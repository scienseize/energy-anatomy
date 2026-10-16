# Methods record: energy anatomy of transformer pretraining components

This file records how the code in this repository produces every number used in the thesis, at the level of detail needed to write the Methodology chapter and to defend the Results and Discussion chapter. DESIGN.md is the specification; this file records what was implemented, the conventions it fixes, where it departs from the specification, and its known limitations. Section and file references point at the code that implements each statement.

Status (2026-10-16): the analytical model, the tables T2 and figure F6 derived from it, and the profiling harness are implemented and tested. One smoke-test configuration has been measured on the Kaggle T4 (Section 12). The full grid has not been measured yet. Every number in Section 9 is analytical (computed, not measured).

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
| factorized_embed (out) | out_proj | Mirrored: `Linear(d, E)` + `Linear(E, V)`, E = 128, no bias (like Pythia's embed_out). |
| rmsnorm | ln (all 2L + 1) | Zhang and Sennrich (2019): `x * rsqrt(mean(x^2) + eps) * w`, eager ops in the run dtype. |

The RQ5 metric is delta E_model = E_total(baseline) - E_total(with the substitution swapped in), as a percentage of baseline, per configuration.

## 4. Analytical model

Implemented in `src/anatomy/analytical.py`. Output: `data/derived/analytical.csv`, one row per (configuration, unit), 270 rows.

### 4.1 Counting method

FLOPs and bytes are not written out by hand per component. Each unit's real module runs one forward and backward step on PyTorch's `meta` device (shapes only, no data) under a dispatch-mode interceptor that sees every ATen operator autograd issues. This counts exactly the operator sequence the GPU will execute, for baseline units, substitutions and the block alike, with no separate formulas to keep in sync.

- View and aliasing operators (`is_view`, `_unsafe_view`) and pure allocations (`new_empty_strided`) move no data and are skipped.
- An operator without a FLOP rule stops the run with an error, so no operator is ever silently counted as zero.

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
- **Memory-bound** if C1 < ridge. This flag decides which (component, configuration) pairs enter the H1 test.
- **Predicted time** t_hat = max(F / P_eff, bytes / bandwidth), per instance.
- **C2** = FLOPs_unit * count per model / total model FLOPs, where total model FLOPs is the sum over the eight baseline components (so the C2 values of the baseline components sum to 1 by construction). For a substitution unit, C2 is its share within the model that has that substitution swapped in. For factorized_embed, both the in and out units are swapped together. The block row has no C2.

## 5. Tables and figures produced so far

- **T2** (`data/tables/T2.csv`, `.md`): the 18 configurations with model dimensions, parameter counts, batch, tokens, precision, and a `profiled` flag. The flag becomes true once `data/raw/passes.csv` holds any `ok` pass for that configuration.
- **F6** (`data/figures/F6.png`, `.svg`): roofline scatter of the eight baseline components, one panel per precision. The x axis is analytical C1; the y axis is the analytical attainable rate F / t_hat. Roof lines show the CUDA-core and (FP16) tensor-core peaks with their ridge points. Colors follow a fixed categorical order per component, and each component also has its own marker shape, so identity does not depend on color alone. Measured rates will be overlaid once profiling data exist.

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
8. **Status.** `ok`; `short_window` if the measured window came in under 2 s despite calibration (kept, flagged for the analysis); `oom` if the unit ran out of GPU memory. An OOM writes one `oom` row per remaining pass, frees memory, and moves on. It is never silently dropped.
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
`run_id, ts, model, seq_len, precision, tokens, unit, unit_kind{component|substitution|block}, pass_idx, reps, energy_j, time_s, energy_per_rep_j, avg_power_w, sm_clock_mhz, mem_clock_mhz, temp_c, power_limit_w, throttled, throttle_reasons, status{ok|short_window|oom}, torch, cuda, driver, gpu_uuid, git_sha`

`sm_clock_mhz` and `mem_clock_mhz` are window means, and `temp_c` is the window maximum (Section 6, step 7). `throttle_reasons` was added on 2026-10-16 after the first smoke test. `gpu_uuid` (NVML's identifier of the physical GPU) was added on 2026-10-16 after the second smoke test, so rows can be grouped by device. `power_limit_w` was added to the DESIGN section 9 schema on 2026-10-16, because DESIGN section 6 requires recording the power limit. Substitution unit names in this file are `sdpa_core, linear_core, gqa_proj, factorized_embed_in, factorized_embed_out, rmsnorm`.

`data/raw/idle.csv`: `run_id, ts, energy_j, time_s, avg_power_w, sm_clock_mhz, mem_clock_mhz, temp_c, power_limit_w, throttle_reasons, gpu_uuid`

`data/raw/smoke/` holds the first smoke test (Section 12) in the earliest schema: no `throttle_reasons` or `gpu_uuid` column, clocks and temperature read once right after each window, and `throttled` computed against pass 0. `data/raw/smoke-2/` holds the second smoke test: current schema except `gpu_uuid`.

`data/derived/analytical.csv`: `model, seq_len, precision, component, count_per_model, flops, bytes, c1, ridge, memory_bound, c2, t_hat`. Here `flops`, `bytes` and `t_hat` are per unit instance, forward + backward. `component` holds the unit name, including substitutions and `block`.

## 8. Verification

`uv run pytest` (32 tests):

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
8. **Expected OOM at the largest configurations.** The 6.9B block in FP32 at seq 2048 holds several float32 attention-score tensors of about 2 GB each and is expected to exceed 16 GB. If it does, that configuration has no block measurement. It is logged as `oom` and must be reported as a gap, not interpolated.
9. **Absolute energy is not reproducible across sessions to better than about 16%.** Two smoke runs of the same configuration, with identical software, differed by 16% in total energy, with higher clocks in every unit of the second run (Section 12). The likely cause is a different physical T4 or different cooling; the first two runs did not record the device. Shares (C3, C4) were stable to within 0.011, so the thesis conclusions rest on shares, and each configuration is measured within one session (Section 6, step 10). Absolute joules are reported as session-specific.
10. **Only two smoke configurations have been measured.** All other quantitative statements above are analytical.

## 11. Reproducing

```
uv sync                        # Python 3.11 env from uv.lock
uv run pytest                  # 32 tests
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
