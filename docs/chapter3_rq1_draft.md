# Chapter III draft, part 1: RQ1 and the hypothesis test

Draft, 2026-10-19, for Bill's review. Nothing goes into the Google Doc until approved. Every number comes from run 3 (`data/tables/T4.csv`, `T5.csv`, `T6.csv`, `TA2_dynamic_rule.csv`, `TA3_normalized_rule.csv`) or from `docs/methods.md` sections 13, 15 and 17; the trace at the end lists the source of each.

The conceptual framework says the first research question "describes the distributions C2 and C3 measure, and with them tests the hypothesis", so this section reports the shares, the analytical check, and the H1 decision. The full waste profile (C4 ranks, Table 7, Figure 7) is left for RQ2.

## Open questions for Bill

1. **Table and figure numbers.** The draft uses the repo's numbers (Table 4, 5, 6; Figure 5). The manuscript currently has Table 1 and Figures 1 to 3 only. Table 3 (the screening record) is cited in Chapter II but is not in the doc yet, and nothing occupies Table 2 or Figure 4. Options: (a) add Table 2 (the 18 configurations, repo T2) under Population and Sampling and Table 3, and renumber the figures so Chapter III starts at Figure 4; (b) renumber everything in order of appearance.
2. **Body table.** The full T5 has 180 rows (18 configurations x 10 rows). The draft puts a condensed Table 5 in the body (one row per component, ranges over the 18 configurations) and moves the full T5 to an appendix. Alternatively, the body could show a smaller selection of configurations.
3. **Cross-entropy loss.** It is not one of the five RQ1 components but it is measured and memory-bound. The draft reports it in one sentence and leaves it out of the H1 test, as Chapter II does. Keep it that way?

---

## Draft text

### Research Question 1: How is pretraining energy distributed across the components?

All results in this chapter come from the main profiling run (run 3): eighteen configurations, ten matched passes each, on one T4 GPU in one session. Two earlier full runs on two other T4 GPUs are used to check that the results repeat. Shares are medians over the ten passes. Across passes, the interquartile range of a component's energy share was at most 3.8 percentage points (feedforward) and below 0.2 points for layer normalization, positional encoding, the embedding lookup and the loss, so the medians reported below are stable at the precision given. Table 5 summarizes the compute share (C2) and the measured energy share (C3) of each component across the eighteen configurations, and Figure 5 plots the two shares side by side for every configuration.

**Table 5.** Compute share (C2) and measured energy share (C3) per component, median of ten passes, range over the eighteen configurations (full table in Appendix [X])

| Component | Roofline class | C2 range | C3 range | C3 median | Configurations with C3 > C2 |
|---|---|---|---|---|---|
| Attention (projections + core) | mixed | 28.4% to 41.1% | 29.2% to 62.7% | 38.5% | 15 of 18 |
| Attention projections | compute-bound | 17.5% to 31.6% | 7.9% to 28.6% | 24.0% | 0 of 18 |
| Attention core | memory-bound | 2.0% to 23.7% | 2.7% to 54.9% | 16.2% | 18 of 18 |
| Feedforward | compute-bound | 35.0% to 63.3% | 17.4% to 70.1% | 47.0% | 3 of 18 |
| Layer normalization | memory-bound | 0.014% to 0.055% | 0.24% to 2.38% | 1.09% | 18 of 18 |
| Positional encoding (RoPE) | memory-bound | 0.002% to 0.007% | 0.40% to 4.25% | 1.56% | 18 of 18 |
| Token embeddings: input lookup | memory-bound | below 0.0001% | 0.005% to 0.20% | 0.037% | 18 of 18 |
| Token embeddings: output projection | compute-bound | 2.9% to 29.0% | 2.0% to 25.8% | 7.4% | 7 of 18 |
| Cross-entropy loss | memory-bound | 0.001% to 0.057% | 0.04% to 4.39% | 0.38% | 18 of 18 |
| Unattributed | n/a | n/a | -2.8% to +4.8% | 1.0% | n/a |

*Note.* C2 is computed from the architecture and does not change with precision. Unattributed energy is the model's total energy minus the sum of the components measured in isolation.

**Figure 5.** Compute share and energy share per component, for each of the eighteen configurations (repo `data/figures/F5`).

**1.1 Attention.** Taken as a whole, attention used 29.2% to 62.7% of measured energy against a compute share of 28.4% to 41.1%, and its energy share exceeded its compute share in 15 of the 18 configurations, all except the three Pythia 6.9B FP32 configurations. The whole-attention figure hides two opposite behaviors. The attention projections, the matrix multiplications that produce the queries, keys and values (fused into one in the Pythia models) and the attention output, used less energy than their compute share in every configuration (7.9% to 28.6% of energy against 17.5% to 31.6% of compute). The attention core, the score computation, mask, softmax and weighting, used more in every configuration, by a factor of 1.2 to 3.4. Its share grew with sequence length and was largest in FP16 and at the smallest model, peaking at 54.9% of all energy at Pythia 160M, sequence length 2,048, FP16, against a compute share of 23.7%. At that configuration and at two others (160M at 1,024 in FP16, and 160M at 2,048 in FP32) the attention core was the single largest consumer of energy in the model.

**1.2 Feedforward layers.** The feedforward sublayer performs most of the arithmetic (35.0% to 63.3% of compute) and was the largest single consumer of energy in 15 of the 18 configurations, using 17.4% to 70.1% of the total. Its energy share stayed below its compute share in 15 configurations. The exceptions are the three Pythia 6.9B FP32 configurations, where it rose slightly above (Section RQ2 reports how far, and that this excess did not repeat in every run). Attention as a whole used more energy than the feedforward sublayer in six configurations: five of the six Pythia 160M configurations and Pythia 1.4B at 2,048 tokens in FP16.

**1.3 Layer normalization.** Layer normalization (all 2L + 1 instances, pooled with the final norm) accounts for 0.014% to 0.055% of compute but used 0.24% to 2.38% of energy, 13 to 103 times its compute share.

**1.4 Positional encoding.** Rotary positional encoding is applied to a quarter of each head's dimensions in the Pythia models and performs almost no arithmetic (at most 0.007% of compute). It used 0.40% to 4.25% of energy, 221 to 860 times its compute share, and more energy than layer normalization in every configuration.

**1.5 Token embeddings.** The two sub-components behave in opposite ways, which is why they are reported separately. The input embedding lookup, a table read with essentially no arithmetic, used 0.005% to 0.20% of energy, the smallest share of any component, yet 573 to 2,488 times its compute share. The output projection, a matrix multiplication onto the vocabulary, used 2.0% to 25.8% of energy against 2.9% to 29.0% of compute, close to its compute share and below it in 11 of the 18 configurations. Its share was largest at Pythia 160M, where the vocabulary is large relative to the model width.

The cross-entropy loss, measured as its own component, used 0.04% to 4.39% of energy. Energy not attributed to any isolated component ranged from -2.8% to +4.8% of the total (median 1.0%). It was negative in five configurations, meaning the isolated components summed to slightly more than the full model; the likely causes are discussed below.

The configuration moderators shift these shares in a consistent direction. Larger models move energy toward the feedforward sublayer and away from the attention core, the output projection, positional encoding, the embedding lookup and the loss; layer normalization's share falls from 160M to 1.4B but not consistently beyond. Longer sequences move energy into the attention core: its work per sequence grows with the square of the sequence length, so at a fixed 8,192 tokens per pass its arithmetic doubles each time the sequence length doubles, while the other components' arithmetic stays the same. FP16 raises the share of every memory-bound component relative to FP32: halving the precision halves the bytes these components move, but the T4's tensor cores raise the matrix-multiplication peak about eightfold, so the matrix multiplications speed up far more than the memory-bound components do. How these shifts change the ranking of components is taken up under RQ3.

**Analytical check.** The analytical estimate, built from each component's predicted roofline time (Williams et al., 2009) and the measured power of its roofline class, ranks the eight components in nearly the same order as the measurement (Table 4). The Spearman rank correlation between the two orderings was 1.0 in 12 configurations, 0.976 in 5 (all FP16) and 0.952 in the remaining one (Pythia 6.9B, 2,048 tokens, FP16), and the same values were obtained in all three runs. The median absolute difference between estimated and measured shares was 0.9 percentage points. The largest differences were for the attention core (median 4.1 points, at most 14.8 points at Pythia 160M, 1,024 tokens, FP16) and the feedforward sublayer (median 5.2 points, at most 9.8). The analytical model therefore gets the order of the components right but not the size of the attention core's share, which is the quantity the profiling exists to measure.

**Hypothesis test.** Four units of the five components are memory-bound in all eighteen configurations: the attention core, layer normalization, positional encoding and the input embedding lookup. The attention projections, the feedforward sublayer and the output projection are compute-bound in all eighteen and are not tested. The cross-entropy loss is also memory-bound but is not one of the five components under study, so it is reported but not tested. Table 6 applies the decision rule. Each of the four memory-bound components had a positive diagnostic waste score (C4 > 0) that exceeded its interquartile range across passes in all eighteen configurations, so H1 is supported and H0 is rejected for each of them.

**Table 6.** Decision rule for the memory-bound components (run 3, total energy)

| Component | Memory-bound configurations | C4 > 0 | C4 > IQR | Both | Smallest median C4 | Verdict |
|---|---|---|---|---|---|---|
| Attention core | 18 | 18 | 18 | 18 | +0.0069 (6.9B, 512, FP32) | H1 supported |
| Positional encoding | 18 | 18 | 18 | 18 | +0.0040 (6.9B, 2,048, FP32) | H1 supported |
| Layer normalization | 18 | 18 | 18 | 18 | +0.0022 (6.9B, 2,048, FP32) | H1 supported |
| Input embedding lookup | 18 | 18 | 18 | 18 | +0.00005 (6.9B, 2,048, FP32) | H1 supported |

The verdict did not depend on the choices made in the analysis. It was the same, 18 of 18 for every component, when idle power was subtracted and the test rerun on dynamic energy (Appendix Table A2), when the unattributed energy was spread over the components in proportion to their energy (Appendix Table A3), when the 24 passes whose measurement window fell short of the 2-second target (1.5 to 2.0 seconds) were dropped, and in both earlier runs on different GPUs. Across the three runs, the median waste score of the tested components moved by at most 0.017 (attention core) and by at most 0.006 for the other three.

The input embedding lookup is the least robust of the four. Its waste score is small in absolute terms (as low as +0.00005), and in the configurations where unattributed energy is negative, the isolated components overstate the total by an amount whose split across components cannot be observed. If that overstatement were split equally among the eight components, it would exceed the lookup's margin in four configurations and layer normalization's in one. The lookup would then pass the rule in 14 of 18 configurations and layer normalization in 17, both still a majority, so the verdict holds, but the lookup's result rests on the smallest margins in the study.

### Discussion

The measured distribution confirms the central claim of the theoretical framework: energy does not follow arithmetic. The components the roofline places in the memory-bound region all used more energy than their compute share, in every configuration, and the components it places in the compute-bound region mostly used less. Jain et al. (2025) reported that energy is not proportional to computation across transformer components during inference. These results show the same divergence in the forward and backward passes of pretraining, and they show it is not random: its direction follows each component's roofline class.

The size of the divergence differs sharply between the two kinds of memory-bound component. Layer normalization, positional encoding and the embedding lookup do so little arithmetic that their energy runs 13 to 2,488 times their compute share, yet together they used at most about 7% of the model's energy. Chapter I described layer normalization as doing almost no arithmetic while still straining memory bandwidth, and the measurement bears this out. Positional encoding, which the literature rarely treats as an energy cost, used more energy than layer normalization in every configuration. The attention core is the component where the divergence is large in absolute terms, reaching more than half of all energy at long sequences in the smallest model. This agrees with the kernel-level evidence that attention's cost on GPUs is driven by memory traffic rather than arithmetic (Dao et al., 2022; Dao, 2024). Two qualifications apply. The baseline uses the unfused attention core, which writes the full score matrix to memory; a fused kernel avoids this, and its effect is measured under RQ5. And the T4 does not support FlashAttention-2 (Dao, 2024), so the fused alternative available on this GPU is the memory-efficient kernel.

The feedforward sublayer is the mirror image. It does most of the arithmetic and usually uses most of the energy, but less energy than its compute share, so on this evidence it is an efficient component rather than a wasteful one. This matters for how the totals are read: the largest consumer of energy is not the largest source of divergence.

The analytical check supports using the roofline to classify components, since it predicted the order of the components almost exactly, but not to replace measurement. Its largest single errors fell on the attention core, the component with the largest divergence, which is consistent with the framework's premise that arithmetic counts alone cannot recover where energy goes (Hooker, 2021).

Three features of the setup bound these conclusions. First, the T4 was held by its 70 W power limit in every measured pass, and the median power of every component lay between 57 and 68 W. Because power was nearly constant, a component's energy share closely tracks its share of running time, so the divergence measured here is, in effect, the gap between a component's time and its arithmetic. On a GPU not limited by power, components could draw different power and the shares could differ. Second, each component was measured in isolation, reading inputs that are not already in the GPU's cache; inside a full training step some inputs would be, and the GPU could overlap work that is measured separately here. The unattributed energy, between -2.8% and +4.8%, bounds the combined effect. Third, the models used random weights and processed 8,192 tokens per pass on a single GPU type, so the shares describe this hardware and these configurations; they are reported with their configuration rather than as fixed properties of the architecture.

---

## Trace of every number

| Number in the draft | Source |
|---|---|
| Table 5 ranges, medians and C3 > C2 counts | `data/tables/T5.csv`, columns c2 and c3_med, grouped by component |
| Maximum C3 IQR (3.8 points feedforward; < 0.2 points for ln, rope, embed_lookup, loss) | T5.csv c3_iqr maxima: mlp 0.0375, ln 0.0013, rope 0.0018, loss 0.0016, embed_lookup 0.00008 |
| Attention not above C2 at 6.9B FP32 | T5.csv, attention rows where c3_med <= c2 |
| Attention core 1.2 to 3.4x; 54.9% vs 23.7% at 160M/2048/FP16 | T5.csv c3_med / c2 for attn_core |
| Attention core largest single component at 160M/1024/FP16, 160M/2048/FP16, 160M/2048/FP32 | T5.csv, argmax of c3_med over the 8 baseline components |
| Feedforward largest in 15 of 18; attention above feedforward in 6 | T5.csv |
| Ratios 13 to 103 (ln), 221 to 860 (rope), 573 to 2,488 (lookup) | T5.csv c3_med / c2 |
| Output projection below C2 in 11 of 18 | T5.csv |
| Unattributed -2.8% to +4.8%, median 1.0%, negative in 5 | T5.csv unattributed rows; methods.md section 15 |
| Memory-bound small components at most about 7% together | T5.csv: ln + rope + embed_lookup max 0.0238 + 0.0425 + 0.0020 at 160M/512/FP16 = 6.8% |
| Tensor-core peak about eightfold | methods.md 4.4: FP16 tensor 65 TFLOP/s vs FP32 8.1 TFLOP/s |
| Spearman 1.0 / 0.976 / 0.952; median abs diff 0.009; attn_core 0.041, max 0.148; mlp 0.052, max 0.098 | `data/tables/T4.csv`; methods.md section 15 |
| Table 6 counts and verdicts | `data/tables/T6.csv` |
| Smallest median C4 per tested component | `data/derived/shares_summary.csv` c4_med (lookup 5.4e-05); methods.md section 15 |
| Same verdict on dynamic energy and normalized C3 | `TA2_dynamic_rule.csv`, `TA3_normalized_rule.csv` |
| 24 short-window passes; verdicts unchanged without them | methods.md section 13.3 |
| Same verdict in runs 1 and 2; C4 range across runs 0.017 / 0.006 | methods.md section 17 |
| Equal-split sensitivity: 4 configurations (lookup), 1 (ln) | methods.md section 10, limitation 10 |
| Median power 57 to 68 W per component | `data/derived/shares_summary.csv`, power_w_med |
