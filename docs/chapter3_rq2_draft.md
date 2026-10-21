# Chapter III draft, part 2: RQ2, the diagnostic waste score

**Applied to the Google Doc on 2026-10-20 after Bill approved it (both open questions kept as drafted)**, after the last RQ1 paragraph; text verified paragraph by paragraph against this file, Table 7 cell by cell, Figure 6 inserted from commit b3a85b8, rendered pages checked. Table 7 is set in 8 pt with 2 pt cell padding so that no word splits across lines. Draft, 2026-10-20. Every number comes from run 3 (`data/derived/shares_summary.csv`, `data/tables/T7.csv`) or from `docs/methods.md` sections 15 and 17; the trace at the end lists the source of each. Same conventions as RQ1: condensed body table (full T7 to Appendix D), repo F7 is Figure 6, C4 written as a share difference (as in Table 6).

The hypothesis verdicts were reported under RQ1; this section reports the full waste profile, including the compute-bound components, the ranks, and how stable the scores are.

## Open questions for Bill

1. **Energy per FLOP column.** Table 7 reports each component's energy per floating-point operation as a ratio to the feedforward sublayer in the same configuration (for example "15 to 116 times"), rather than in joules per FLOP (for example 2.7e-10 J for layer normalization at Pythia 1.4B, FP16). The ratio reads more easily and does not depend on the GPU's absolute energy, which differs between sessions. Keep the ratio?
2. **Unstable compute-bound scores.** The feedforward result at Pythia 6.9B FP32 (rank 1 in runs 1 and 3, not in run 2) is reported with its run-to-run spread, as the methods record recommends. The draft does not offer a cause, because the measurement does not identify one. Is that the right level of caution, or do you want a stated hypothesis (for example, clock behavior under the power cap)?

---

## Draft text

### Research Question 2: How far does each component's energy share diverge from its compute share?

The diagnostic waste score of a component, C4, is its energy share minus its compute share, computed pass by pass and summarized by the median over the ten passes. A positive score means the component used a larger share of the model's energy than of its arithmetic; a negative score means a smaller share. Because a component's energy is measured in isolation and the model's total is composed from those measurements, the energy shares, and therefore the scores, are estimates of the shares inside a full training step. The scores of all components in a configuration sum to minus the unattributed energy, so they are close to zero-sum: one component's positive divergence is matched by negative divergence elsewhere. Table 7 summarizes the scores across the eighteen configurations, and Figure 6 shows them for every configuration.

**Table 7.** Diagnostic waste score (C4) per component, median of ten passes, over the eighteen configurations (full table: Table D3 in Appendix D)

| Component | Roofline class | C4 range | C4 median | Configurations with C4 > 0 | Rank within configuration | Energy per FLOP, relative to feedforward |
|---|---|---|---|---|---|---|
| Attention core | memory-bound | +0.0069 to +0.3118 | +0.070 | 18 of 18 | 1 (15), 2 (3) | 1.2 to 5.0 times |
| Positional encoding | memory-bound | +0.0040 to +0.0424 | +0.016 | 18 of 18 | 2 to 3 | 201 to 1,040 times |
| Layer normalization | memory-bound | +0.0022 to +0.0232 | +0.011 | 18 of 18 | 3 to 4 | 15 to 116 times |
| Cross-entropy loss | memory-bound | +0.0004 to +0.0433 | +0.004 | 18 of 18 | 2 to 5 | 34 to 141 times |
| Input embedding lookup | memory-bound | +0.00005 to +0.0020 | +0.0004 | 18 of 18 | 5 to 6 | 526 to 3,138 times |
| Output projection | compute-bound | -0.1229 to +0.0082 | -0.005 | 7 of 18 | 4 to 7 | 0.62 to 1.39 times |
| Attention projections | compute-bound | -0.0958 to -0.0128 | -0.050 | 0 of 18 | 6 to 8 | 0.76 to 1.04 times |
| Feedforward | compute-bound | -0.1757 to +0.0679 | -0.083 | 3 of 18 | 1 (3), 8 (15) | 1 (reference) |

*Note.* Rank 1 is the largest score in the configuration; the number of configurations at a rank is given in parentheses. Energy per FLOP is the median energy per repetition divided by the analytical FLOPs, taken as a ratio to the feedforward sublayer in the same configuration.

**Figure 6.** Diagnostic waste score (C4) per component, for each of the eighteen configurations (repo `data/figures/F7`). *Note.* Bars show the median over ten passes, and whiskers extend half the interquartile range on each side of the median; red bars are positive scores and blue bars negative. An asterisk marks the components that are memory-bound in every configuration.

**Direction.** The sign of the score follows the roofline class. All five memory-bound components (the attention core, positional encoding, layer normalization, the input embedding lookup and the loss) had positive scores in all eighteen configurations, and the three compute-bound components had negative scores in 44 of their 54 configuration results. Across all 144 results, the sign matched the roofline class in 134. The ten exceptions are all compute-bound: the output projection in seven configurations, by at most +0.008, and the feedforward sublayer in the three Pythia 6.9B FP32 configurations, by +0.038 to +0.068.

**Size.** The attention core diverges most. Its score ranged from +0.0069 at Pythia 6.9B, 512 tokens, FP32 to +0.3118 at Pythia 160M, 2,048 tokens, FP16, where it used 31 percentage points more of the model's energy than of its arithmetic. It ranked first in 15 of the 18 configurations and second in the other three, the Pythia 6.9B FP32 configurations. Within every model and precision, its score rose with sequence length, and at every model size and sequence length it was higher in FP16 than in FP32. In a typical configuration it accounts for most of the model's positive divergence: a median of 68%, ranging from 8% at Pythia 6.9B, 512 tokens, FP32 to 82%. Positional encoding (+0.0040 to +0.0424) and layer normalization (+0.0022 to +0.0232) follow, ranking second to fourth. The loss ranked second in all six Pythia 160M configurations, where the vocabulary is large relative to the model, and fifth at the largest model. The input embedding lookup had the smallest positive scores (+0.00005 to +0.0020). Together, positional encoding, layer normalization, the loss and the lookup diverged by +0.007 to +0.111, about a third of the attention core's largest score.

**Two kinds of divergence.** The scores of the small memory-bound components are close to their energy shares, because their compute shares are below 0.06%. Their energy per floating-point operation shows that this is not an artifact of a small denominator: per operation, layer normalization used 15 to 116 times as much energy as the feedforward sublayer in the same configuration, the loss 34 to 141 times, positional encoding 201 to 1,040 times and the input embedding lookup 526 to 3,138 times. The attention core used only 1.2 to 5.0 times as much. The profile therefore contains two kinds of divergence. The small components are extremely inefficient per operation but use little energy in total, so their scores stay small. The attention core is only moderately inefficient per operation but performs a large and growing share of the arithmetic, so its score is the largest in absolute terms.

**Compute-bound components.** The attention projections had negative scores in every configuration (-0.0958 to -0.0128). The feedforward sublayer was the most negative component in 15 configurations, down to -0.1757, and absorbed 43% to 75% of the model's negative divergence wherever it was negative. The output projection's score was near zero in most configurations (median -0.005) but reached -0.1229 at Pythia 160M, where the projection onto the vocabulary is a large part of the model. The positive feedforward scores at Pythia 6.9B FP32 (+0.038 to +0.068, rank 1 in all three) are the least stable results in the profile. Their interquartile range across passes reached 0.0375 at 512 tokens, and across the three runs on different GPUs the feedforward score at these configurations was positive and ranked first in runs 1 and 3 but lay between -0.002 and +0.012 in run 2. Over the three runs, the score changed sign in nine configuration and component pairs, all of them feedforward or output projection results, and the largest run-to-run range of a median score was 0.069 for the feedforward sublayer. The sign of the feedforward score at Pythia 6.9B FP32 is fairly consistent (positive in 8 of the 9 run and configuration pairs), but its size and rank are not. The memory-bound components were far more stable: their median scores moved by at most 0.017 across the runs (attention core) and by at most 0.006 for the others, and subtracting idle power moved the four tested components' scores by -0.0115 to +0.0145, a range set by the attention core alone (the others moved by at most 0.0033).

### Discussion

The waste profile supports the framework's central claim in its sharper form: divergence is not just present but systematic, with its sign predicted by the roofline class in 134 of 144 results. The ten exceptions all sit on the compute-bound side; seven are small, and the three larger ones, the feedforward sublayer at Pythia 6.9B FP32, do not repeat reliably across runs. A positive score is therefore not exclusive to memory-bound components, but on this evidence it is characteristic of them.

The profile also reorders the components relative to raw energy. The feedforward sublayer used the most energy in 15 of 18 configurations (first research question), yet it had the most negative score in 15 configurations. The attention core, the largest consumer in only three configurations, had the largest score in fifteen. Jain et al. (2025) attributed raw energy to transformer components during inference; ranking by raw energy would put the feedforward sublayer first, while ranking by divergence puts the attention core first. That difference is the reason the study uses the score rather than raw energy to decide where to look: raw energy identifies the most expensive component, the score identifies the component whose expense is least explained by the work it does.

The growth of the attention core's score with sequence length agrees with the IO-aware account of attention (Dao et al., 2022). At a fixed number of tokens per pass, the unfused core's arithmetic grows in proportion to the sequence length and its memory traffic nearly as fast (1.8 to 1.9 times per doubling), so its share of the model's work grows; because each unit of that work costs more energy than the model's average, its excess energy grows with it. The higher scores in FP16 follow the same logic as the energy shares under the first research question: the tensor cores accelerate the matrix multiplications far more than they accelerate memory traffic, so every memory-bound component's share of the remaining time, and of the energy, rises. In the terms of Hooker (2021), the components that fit the hardware's strengths become cheaper and the ones that do not are left carrying the cost.

The two kinds of divergence matter for what comes next. A component with a large score, like the attention core, offers a large absolute saving if its cost mechanism can be addressed. A component with extreme energy per operation but a small score, like the embedding lookup, offers little to save in total no matter how inefficient it is. The score alone does not say whether a component can be changed without harming the model; that is the role of the capability overlay under the third research question.

Two limits apply to reading the scores. First, they are relative: because they sum to minus the unattributed energy, a component's score depends on how efficiently the rest of the model runs, and a change to one component moves the others. Second, under the T4's power cap the energy shares track time shares (first research question), so a score measures how far a component's running time departs from its arithmetic at nearly constant power. The compute-bound scores, which are small differences between large shares, are the most sensitive to this, which is consistent with their instability across runs.

---

## Trace of every number

| Number in the draft | Source |
|---|---|
| Table 7 C4 ranges, medians, positive counts, ranks | `data/derived/shares_summary.csv` (c4_med, rank_c4), 8 baseline components; matches `data/tables/T7.csv` |
| Lookup smallest C4 +0.00005 | shares_summary.csv c4_med 5.4e-05 at 6.9B/2048/FP32 |
| Energy per FLOP ratios | shares_summary.csv energy_per_flop_j divided by the mlp value of the same configuration |
| Sign matches class in 134 of 144 | 90 memory-bound results all positive; compute-bound negatives 18 + 15 + 11 = 44 |
| Attention core rises with sequence length in every model and precision; FP16 above FP32 | shares_summary.csv attn_core c4_med pivoted by model, precision, seq_len |
| Attention core share of positive divergence: median 68%, 8% (6.9B/512/FP32) to 82% | attn_core c4_med / sum of positive c4_med per configuration |
| Loss second in all 160M configurations, fifth at 6.9B | rank_c4 per configuration |
| Four small components together +0.007 to +0.111 | sum of c4_med for ln, rope, embed_lookup, loss per configuration (max 160M/512/FP16) |
| Feedforward absorbs 43% to 75% of negative divergence | mlp c4_med / sum of negative c4_med, configurations where mlp is negative |
| Feedforward IQR 0.0375 at 6.9B/512/FP32 | shares_summary.csv c4_iqr |
| Run-to-run: 9 sign changes (mlp, out_proj); ranges mlp 0.069 (recomputed: 0.0695); 6.9B FP32 mlp in run 2 -0.002 to +0.012; positive in 8 of 9; tested components within 0.017 / 0.006 | methods.md section 17 |
| Dynamic-energy shift -0.0115 to +0.0145 (attn_core); others at most 0.0033 | shares_summary.csv c4_dyn_med - c4_med; methods.md section 15 |
| Feedforward largest consumer in 15 of 18; attention core largest in 3 | RQ1 (T5.csv) |
| Attention-core bytes 1.8 to 1.9 times per doubling of sequence length; FLOPs exactly 2 times | `data/derived/analytical.csv`, attn_core flops and bytes x count_per_model |
