# Chapter II additions: Table 2, Table 3, and four references

**Applied to the Google Doc on 2026-10-19 after Bill approved all (A1 to A5).** Each insertion was verified by reading the doc back (text and every table cell match this file) and by a rendered PDF check. Draft, 2026-10-19. Follows the numbering decision of 2026-10-19 (option a): add the missing Tables 2 and 3 so the repo's table numbers T2 to T10 hold in the manuscript.

Sources: `data/tables/T2.csv` (configurations) and `data/tables/T3.csv` (screening record, from `data/manual/sources.csv`). One fix made while drafting: S17 (Zhang & Sennrich) still read "pre-2021, not a listed exception" in `sources.csv`, but R4 (applied 2026-10-17) added it to the seminal-exception list; `sources.csv` and T3 now say "listed seminal exception".

---

## A1. Population and Sampling: cite Tables 1 and 2

APA numbers tables by first mention. Table 1 is placed in Population and Sampling but is first cited in Phase 1, and Table 2 must be cited after it.

**Current:** "For the evidentiary dimension, sources are screened against the explicit inclusion and exclusion criteria."

**Proposed:** "For the evidentiary dimension, sources are screened against the explicit inclusion and exclusion criteria in Table 1. The sampled models and the eighteen configurations are listed in Table 2."

## A2. Table 2, inserted at the end of Population and Sampling

Placed after the paragraph beginning "Purely methodological references", before the Data Gathering heading, so it follows Table 1.

**Table 2.** Sampled Pythia models and profiling configurations

| Model | Parameters | Non-embedding parameters | Layers | Model width | Attention heads | Vocabulary |
|---|---|---|---|---|---|---|
| Pythia 160M | 162,322,944 | 85,056,000 | 12 | 768 | 12 | 50,304 |
| Pythia 1.4B | 1,414,647,808 | 1,208,602,624 | 24 | 2,048 | 16 | 50,304 |
| Pythia 6.9B | 6,857,302,016 | 6,444,163,072 | 32 | 4,096 | 32 | 50,432 |

*Note.* Each model is profiled at sequence lengths of 512, 1,024 and 2,048 tokens, with batch sizes of 16, 8 and 4 so that every pass processes 8,192 tokens, in 32-bit (FP32) and 16-bit (FP16) floating point: six configurations per model, eighteen in total, all profiled. All three models apply rotary positional encoding to 25% of each attention head's dimensions, compute the attention and feedforward sublayers in parallel, and use separate input and output embedding matrices. Dimensions from Biderman et al. (2023).

## A3. Table 3, inserted after the Phase 1 paragraph

Phase 1 ends "The screening record, with each source's decision against every criterion, is summarized in Table 3." The table follows that paragraph. Criterion numbers refer to Table 1 (content criteria 1 energy or timing data, 2 specifications, 3 capability effect). All 18 screened sources are listed. As applied, the Verifiability column is folded into the table note (17 of 18 sources share one value, and eight columns do not fit the page legibly), and the table is set in 8 pt.

**Table 3.** Screening record of the published sources

| Source | Used for | Content (Table 1) | Relevance to training | Recency | Model scale | Decision |
|---|---|---|---|---|---|---|
| He, Sun, et al. (2024) | Capability evidence | 3: removing a component | Adaptable: layers dropped from pretrained models | 2021 or later | Llama-2 13B and 70B, Mistral 7B | Admitted |
| He and Hofmann (2024) | Capability evidence | 3: simplifying a component | Training: simplified blocks trained from scratch | 2021 or later | 18-block, width-768 decoder; Crammed BERT | Admitted |
| Haviv et al. (2022) | Capability evidence; removing positional encoding | 3: removing a component | Training: language models trained without positional encoding | 2021 or later | 247M; 1.3B | Admitted |
| Kazemnejad et al. (2023) | Capability evidence (supporting); removing positional encoding | 3: removing a component | Training: models trained with each encoding | 2021 or later | 1.3B | Admitted |
| Zhu et al. (2025) | Capability evidence | 3: replacing a component | Training: LLaMA pretrained without normalization layers | 2021 or later | LLaMA 7B to 70B | Admitted |
| H. W. Chung et al. (2021) | Capability evidence | 3: shrinking a component | Training: multilingual pretraining | 2021 or later | 12-layer, width-768 encoder | Admitted |
| Ainslie et al. (2023) | Capability evidence; grouped-query attention | 3: reducing key-value heads; 1: timing | Inference timing, with a stated path to training (uptraining) | 2021 or later | T5-Large, T5-XXL | Admitted |
| Lan et al. (2020) | Capability evidence; factorized embedding | 3: factorizing a component | Training: pretraining | Before 2021, seminal exception | ALBERT-base | Admitted |
| Michel et al. (2019) | Capability evidence | 3: pruning attention heads | Adaptable: heads pruned from trained models | Before 2021, not an exception | BERT; WMT Transformer | Excluded (recency) |
| Press and Wolf (2017) | Capability evidence (supporting) | 3: tying the embeddings | Training | Before 2021, not an exception | LSTM language models | Excluded (recency) |
| Geva et al. (2021) | Context | None quantitative (mechanism only) | Not applicable | 2021 or later | Not reported | Context only |
| Brody et al. (2023) | Context | None quantitative for model quality | Not applicable | 2021 or later | Not reported | Context only |
| Rabe and Staats (2021) | Memory-efficient attention | 1: memory and runtime | Training: backward pass measured | 2021 or later | Not reported | Admitted |
| Lefaudeux et al. (2022) | Memory-efficient attention kernel | Kernel implementation | Training and inference | 2021 or later | Not reported | Admitted as method |
| Katharopoulos et al. (2020) | Linear attention | 1: timing | Training: time per epoch | Before 2021, seminal exception | Not reported | Admitted as method |
| Shazeer (2019) | Origin of grouped-query attention (multi-query attention) | 1: timing | Inference only (decoding) | Before 2021, seminal exception | WMT14 Transformer | Admitted as method |
| Zhang and Sennrich (2019) | Root mean square normalization | 1: running time | Training: running time per 1,000 steps | Before 2021, seminal exception | Transformer and others | Admitted as method |
| Hsu et al. (2024) | Fused normalization and rotary kernels | 1: kernel speed and memory | Adaptable: kernels for language-model training | 2021 or later | Kernel benchmark | Admitted |
*Note.* Admitted: the source's results enter the study (capability ordering or published substitution values). Admitted as method: the source defines a technique the study implements, but its numbers are not used as evidence. Context only: cited for background, no number used. Excluded sources are kept in the record so the screening can be audited. All sources are publicly accessible through scholarly repositories, one of them as a preprint (He, Sun, et al., 2024), except Lefaudeux et al. (2022), which is an official vendor code repository.

## A4. Four references added (61 to 65)

Verified 2026-10-19 by an independent agent against NeurIPS proceedings and the ACL Anthology (authors, year, title, venue, pages, DOI); the Press and Wolf DOI resolves to ACL Anthology E17-2025. Inserted in alphabetical order (Brody after Biderman; Geva after Fedus; Michel after Luccioni; Press after Poli).

- Brody, S., Alon, U., & Yahav, E. (2023). On the expressivity role of LayerNorm in transformers' attention. In Findings of the Association for Computational Linguistics: ACL 2023 (pp. 14211–14221). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-acl.895
- Geva, M., Schuster, R., Berant, J., & Levy, O. (2021). Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 5484–5495). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.446
- Michel, P., Levy, O., & Neubig, G. (2019). Are sixteen heads really better than one? In Advances in Neural Information Processing Systems 32 (NeurIPS 2019).
- Press, O., & Wolf, L. (2017). Using the output embedding to improve language models. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers (pp. 157–163). Association for Computational Linguistics. https://doi.org/10.18653/v1/E17-2025

## A5. List of Tables (front matter)

**Current:** "Table 1. Inclusion and exclusion criteria for published sources"

**Proposed:** add after it:
"Table 2. Sampled Pythia models and profiling configurations"
"Table 3. Screening record of the published sources"

Chapter III tables and Figures 4 onward are added to the lists when Chapter III goes into the doc.
