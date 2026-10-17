# Chapter II revisions: aligning the manuscript with the study as run

Draft, 2026-10-17. Each item gives the manuscript's current text (quoted exactly from the Google Doc "manuscript alignment"), the proposed replacement, and why. Nothing is applied to the Google Doc until approved. All citations were checked by independent verifier agents (2026-10-17). The T4 is cited from NVIDIA's product page, which gives no date (n.d.); its bandwidth figure is "320+ GB/s", while the datasheet is reported as 300 GB/s, a difference that changes no tested result (methods.md section 4.4).

Two citation-disambiguation notes for APA 7 (both apply wherever these appear):
- Two 2024 papers by authors named He: write **He and Hofmann (2024)** for *Simplifying Transformer Blocks* and **He, Sun, et al. (2024)** for *What Matters in Transformers?*
- Different first authors named Chung: write **H. W. Chung et al. (2021)** for *Rethinking Embedding Coupling*, to keep it apart from **J.-W. Chung et al. (2024, 2025)** already in the references.

---

## R1. Research Design: capability sources

**Current:** "...and component ablation and removal studies that supply the capability evidence (Haviv et al., 2022; Kazemnejad et al., 2023; He & Hofmann, 2024)."

**Proposed:** "...and component ablation and removal studies that supply the capability evidence (Lan et al., 2020; H. W. Chung et al., 2021; Haviv et al., 2022; Ainslie et al., 2023; Kazemnejad et al., 2023; He & Hofmann, 2024; He, Sun, et al., 2024; Zhu et al., 2025)."

**Why:** the capability ranking uses all eight; the current list names three (finding 3).

## R2. Research Design: what analytical estimation does

**Current:** "The same method extends to energy: applying the roofline model with the published hardware parameters of the profiled GPUs predicts how a component's energy behavior shifts when its arithmetic intensity or memory traffic changes. The study refers to this use as analytical estimation, and it is the procedure through which candidate substitutions are evaluated without a full training run."

**Proposed:** "The same method extends to energy: applying the roofline model with the published hardware parameters of the profiled GPU (NVIDIA, n.d.) predicts how a component's energy behavior shifts when its arithmetic intensity or memory traffic changes. The study refers to this use as analytical estimation. Because every sampled configuration and every candidate substitution fit the rented GPU, all of them are profiled directly, and analytical estimation serves as a check on the profiled energy shares rather than as a substitute for them."

**Why:** no configuration or substitution was evaluated by analytical estimation alone; T4 compares the two (finding 5).

## R3. Population and Sampling: the configurations actually profiled

**Current:** "Only configurations that fit the rented GPU instances are profiled directly; larger configurations enter through published evidence and analytical estimation, and the profiled configurations are used to check the analytical estimates where the two overlap."

**Proposed:** "The sampled configurations are the Pythia models at 160 million, 1.4 billion and 6.9 billion parameters (Biderman et al., 2023), decoder-only transformers released with their full architectural specifications, at sequence lengths of 512, 1,024 and 2,048 tokens and in 32-bit (FP32) and 16-bit (FP16) floating point: eighteen configurations, each processing 8,192 tokens per pass. All eighteen fit the rented GPU, an NVIDIA T4 with 16 GB of memory (NVIDIA, n.d.), and are profiled directly; their analytical estimates are computed alongside and checked against the profiled values."

**Why:** the manuscript never names the models or the GPU, and no larger configuration entered the study (finding 5).

## R4. Population and Sampling: seminal exceptions

**Current:** "Under these conditions: Williams et al. (2009) for the roofline model, Vaswani et al. (2017) for the transformer architecture, Shazeer et al. (2017) and Shazeer (2019) for mixture-of-experts and multi-query attention, Strubell et al. (2019) for training energy accounting, Kaplan et al. (2020) for scaling laws, Katharopoulos et al., Wang et al., and Lan et al. (2020) for linear-attention and factorized-embedding techniques, and Hennessy and Patterson (2019) for analytical modeling in computer architecture."

**Proposed:** "Under these conditions: Williams et al. (2009) for the roofline model, Vaswani et al. (2017) for the transformer architecture, Shazeer et al. (2017) and Shazeer (2019) for mixture-of-experts and multi-query attention, Strubell et al. (2019) for training energy accounting, **Zhang and Sennrich (2019) for root mean square layer normalization,** Kaplan et al. (2020) for scaling laws, Katharopoulos et al., Wang et al., and Lan et al. (2020) for linear-attention and factorized-embedding techniques, and Hennessy and Patterson (2019) for analytical modeling in computer architecture. **Lan et al. (2020) also supplies capability evidence, the quality cost of factorizing the embedding table; this falls within the rule, because the restriction concerns evidence about energy behavior and this evidence concerns model quality.**"

**Why:** Zhang and Sennrich is the RMSNorm method source and is pre-2021; ALBERT's quality numbers are used as capability evidence (decisions of 2026-10-17, finding 3).

## R5. Phase 1: the screening record

**Current:** "Sources that pass are logged with their configuration details: model scale, sequence length, and numerical precision."

**Proposed:** "Sources that pass are logged with their configuration details: model scale, sequence length, and numerical precision. **The screening record, with each source's decision against every criterion, is summarized in Table 3.**"

**Why:** the record now exists (`data/manual/sources.csv`, T3: 10 admitted, 4 admitted as method, 2 context only, 2 excluded) (finding 6).

## R6. Phase 2: the attention split

**Current:** "...with the input embedding lookup and the output projection counted as separate sub-components because the first is a memory-bound table lookup and the second a compute-bound matrix multiplication."

**Proposed:** "...with the input embedding lookup and the output projection counted as separate sub-components because the first is a memory-bound table lookup and the second a compute-bound matrix multiplication. **Attention is divided in the same way, into its projections (the query, key, value and output matrix multiplications, compute-bound) and its core (the score computation, causal mask, softmax and weighting, memory-bound); their shares are reported separately and summed to describe attention as a whole.**"

**Why:** RQ1.1 asks about attention as one component; the study profiles the two parts separately and now reports their sum (T5, finding 11).

## R7. Phase 3: profiling only

**Current:** "Phase 3, profiling, estimates each component's energy share (C3) through two tracks: extracting component-level energy, power, and timing figures from admitted sources, and filling the gaps with small-scale cloud GPU profiling of forward and backward passes, with analytical estimation covering configurations too large to profile directly. The profiling protocol is fixed in advance. [...] such as layer normali zation execute in microseconds, [...]"

**Proposed (first sentence replaced; one sentence added after the protocol; the rest kept, with the typo fixed):** "Phase 3, profiling, measures each component's energy share (C3) by small-scale cloud GPU profiling of forward and backward passes. Published sources rarely report component-level energy for pretraining, so published figures do not enter the energy share; they are used in Phase 6 to compare substitution effects. The profiling protocol is fixed in advance. Warm-up passes precede measurement, each pass is repeated and summarized by its median to resist outliers, and energy is read through the GPU's built-in power instrumentation (You et al., 2023). Because that instrumentation samples power at millisecond resolution while individual components such as layer normalization execute in microseconds, [...unchanged...] by dividing out the repetition count. **All components of a configuration are measured within one session, since absolute energy differs between sessions on different physical GPUs while energy shares do not, and the full set of configurations is profiled three times, on three different GPUs, to check that the shares and the decisions repeat.** The profile produced this way covers [...unchanged...]"

**Why:** C3 is 100% profiled; there is no published-figures track and no analytical-only configuration (finding 5). The session rule and the three runs are part of the protocol as run.

## R8. Phase 4: capability sources and the single ordering

**Current:** "The overlay is drawn from the component ablation and removal studies admitted in Phase 1, which report what happens to model quality when a component is removed, replaced, or shrunk (Haviv et al., 2022; Kazemnejad et al., 2023; He & Hofmann, 2024), read alongside the scaling-law relation between compute allocation and capability (Kaplan et al., 2020; Hoffmann et al., 2022)."

**Proposed:** "The overlay is drawn from the component ablation and removal studies admitted in Phase 1, which report what happens to model quality when a component is removed, replaced, or shrunk (the sources listed under the research design and recorded in Table 3). Because this evidence does not vary with the sampled configurations, the capability ordering is a single ordering applied to every configuration."

**Why:** Kaplan and Hoffmann were not used for the ordering (finding 3), and the ordering is pooled, not per configuration (finding 10). Kaplan and Hoffmann stay in R15 as moderator references.

## R9. Phase 5: candidates chosen in advance, then matched

**Current:** "Phase 5, matching, pairs each priority component, those combining the highest diagnostic waste score with the lowest capability contribution, with the candidate substitutions from the Chapter I catalog that most directly target its cost mechanism, answering the fourth question."

**Proposed:** "Phase 5, matching, pairs each priority component, those combining the highest diagnostic waste score with the lowest capability contribution, with the candidate substitutions from the Chapter I catalog that most directly target its cost mechanism, answering the fourth question. **An initial set of candidates, covering the attention core, the attention projections, the embeddings and layer normalization, was fixed before profiling so that it could be measured alongside the baseline. Once Phases 2 to 4 had identified the priority components, each was checked for a matched candidate. Positional encoding, a priority component the initial set did not cover, received two: the same rotary computation compiled into fused kernels, and its removal (Haviv et al., 2022; Kazemnejad et al., 2023). Layer normalization received a fused implementation of its substitute, so that the effect of the algorithm could be separated from that of its implementation. Candidates for components that are not priorities are kept and reported as comparisons.**"

**Why:** five of the six original candidates were fixed on 2026-10-14, before measurement; the priority components (rope, ln, embed_lookup) came out of Phases 2 to 4; rope had no candidate until 2026-10-17 (finding 1). This states that order plainly instead of implying all candidates followed the priorities.

## R10. Phase 6: no analytical-only substitutions; published evidence is qualitative

**Current:** "...under the same profiling protocol used in Phase 3, with analytical estimation supplying the effect of any substitution that cannot be profiled directly. The estimated energy reduction for each substitution is then checked against the published evidence admitted in Phase 1 for that technique, answering the fifth question."

**Proposed:** "...under the same profiling protocol used in Phase 3 and in the same profiling session as that baseline. A substitution that removes a component outright runs nothing in its place, so its effect is computed from the removed component's measured energy. The estimated energy reduction for each substitution is then set beside the published evidence admitted in Phase 1 for that technique, answering the fifth question. Because that evidence reports each technique's own metric (speed, memory or parameter count) on other hardware, rather than energy, the comparison is qualitative."

**Why:** every substitution except the removal was profiled (finding 5); published values are not energy and not comparable (finding 2 and Bill's T10 caveat).

## R11. Phase 7: capability ordering is shared

**Current:** "...(each component's compute share, energy share, arithmetic intensity, diagnostic waste score, and capability ordering)..."

**Proposed:** "...(each component's compute share, energy share, arithmetic intensity and diagnostic waste score, together with the capability ordering, which is the same for every configuration)..."

**Why:** finding 10.

## R12. Data Analysis, descriptive stage: how energy share is obtained

**Current:** "Energy share is estimated from published figures and profiling measurements, cross-checked against each other where multiple sources report the same component and supplemented by analytical estimation where neither is available, with profiled measurements summarized using the median across repeated passes to resist outliers (You et al., 2023)."

**Proposed:** "Energy share is measured by profiling and summarized using the median across repeated passes to resist outliers (You et al., 2023). The analytical estimate of each share is compared against the measured share as a check, by their absolute difference and by the rank correlation of the two orderings, both reported descriptively."

**Why:** finding 5; T4 reports absolute differences and Spearman rho.

## R13. Data Analysis, comparative stage: the attention example

**Current:** "...for example the memory-traffic reduction reported for IO-aware attention (Dao et al., 2022)."

**Proposed:** "...for example the memory reduction reported for memory-efficient attention (Rabe & Staats, 2021), the fused attention available on the profiled GPU, which does not support FlashAttention-2 (Dao, 2024)."

**Why:** the study's fused attention is the memory-efficient kernel, not FlashAttention (finding 8).

## R14. Decision rule: the denominator, and the configurations that enter

**Current:** "H1 is judged supported for a memory-bound component when its diagnostic waste score is positive (C4 > 0) in a majority of the profiled configurations and, in those configurations, the gap exceeds the spread of C4 across repeated profiling passes, taken as its interquartile range. Configurations whose energy shares rest only on published figures or analytical estimation have no repeated passes, so they are reported descriptively and do not enter the test."

**Proposed:** "H1 is judged supported for a memory-bound component when its diagnostic waste score is positive (C4 > 0) in a majority of the profiled configurations **in which it is memory-bound** and, in those configurations, the gap exceeds the spread of C4 across repeated profiling passes, taken as its interquartile range. **Every sampled configuration is profiled, so every configuration enters the test.**"

**Why:** the rule as coded counts memory-bound configurations; the result is the same either way, because all four tested components are memory-bound in all 18 (finding 9). The second sentence described configurations that do not exist.

## R15. Data Analysis, diagnostic stage: capability sources

**Current:** "...its ordering is taken from the ablation and removal evidence admitted in Phase 1 (Haviv et al., 2022; Kazemnejad et al., 2023; He & Hofmann, 2024)."

**Proposed:** "...its ordering is taken from the ablation and removal evidence admitted in Phase 1 and recorded in Table 3."

**Why:** same as R8; the full citation list sits in R1. The following sentence (Kaplan, Hoffmann, Peng et al.) stays as the moderator references.

---

## References to add (APA 7)

- Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 2397–2430). PMLR. https://proceedings.mlr.press/v202/biderman23a.html
- Chung, H. W., Févry, T., Tsai, H., Johnson, M., & Ruder, S. (2021). Rethinking embedding coupling in pre-trained language models. In *International Conference on Learning Representations (ICLR 2021)*.
- He, S., Sun, G., Shen, Z., & Li, A. (2024). What matters in transformers? Not all attention is needed. *arXiv preprint arXiv:2406.15786*. https://doi.org/10.48550/arXiv.2406.15786
- Hsu, P.-L., Dai, Y., Kothapalli, V., Song, Q., Tang, S., Zhu, S., Shimizu, S., Sahni, S., Ning, H., & Chen, Y. (2024). Liger Kernel: Efficient Triton kernels for LLM training. *arXiv preprint arXiv:2410.10989*. https://doi.org/10.48550/arXiv.2410.10989
- Lefaudeux, B., Massa, F., Liskovich, D., Xiong, W., Caggiano, V., Naren, S., Xu, M., Hu, J., Tintore, M., Zhang, S., Labatut, P., Haziza, D., Wehrstedt, L., Reizenstein, J., & Sizov, G. (2022). *xFormers: A modular and hackable Transformer modelling library* [Computer software]. https://github.com/facebookresearch/xformers
- NVIDIA. (n.d.). *NVIDIA T4 tensor core GPU*. Retrieved October 17, 2026, from https://www.nvidia.com/en-us/data-center/tesla-t4/
- Rabe, M. N., & Staats, C. (2021). Self-attention does not need O(n²) memory. *arXiv preprint arXiv:2112.05682*. https://doi.org/10.48550/arXiv.2112.05682
- Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. In *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
- Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). Transformers without normalization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)* (pp. 14901–14911).

Notes:
- Liger Kernel and xFormers are cited in Chapter III (substitution comparisons), not in these Chapter II revisions; they are listed so the References are complete once Chapter III is written.
- The CVPR page range for Zhu et al. comes from catalog metadata (the CVF page could not be opened); the ICLR 2021 venue for H. W. Chung et al. comes from the conference listing. Both should be checked against the PDFs before submission.
- Michel et al. (2019) and Press and Wolf (2017) are not added: they were excluded on 2026-10-17 (Table 1 recency).
