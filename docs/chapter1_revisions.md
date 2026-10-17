# Chapter I and Research Design revisions: one evidence model across the manuscript

**Applied to the Google Doc on 2026-10-18 after Bill approved all four** (exact-match replacements pinned to the revision read beforehand, each changing exactly one occurrence; verified by reading the doc back: none of the replaced phrasing remains). Draft, 2026-10-18. The Chapter II alignment of 2026-10-17 changed how the manuscript says energy shares are obtained (measured by profiling, with the analytical model as a check and published evidence used only for compute shares, capability and substitution comparisons). Four passages written on 2026-10-14 still describe the older model, in which energy shares were estimated partly from published benchmark data. Each item gives the current text (quoted exactly from the Google Doc), the proposed replacement, and why. Nothing is applied until approved.

**Left unchanged on purpose:** "estimated" in RQ2 ("estimated energy share"), RQ5 ("estimated energy reduction"), the second proposition, and "computable from collected and estimated data". Each component's energy is measured, but the model-level shares and the substitution savings are composed from isolated-component measurements under an additivity assumption (methods.md sections 14 and 16), so "estimated" remains accurate there.

---

## C1. Theoretical framework: the framework's limits

**Current (two sentences):** "Reliance on estimation rather than full-run instrumentation is also where the framework's limits sit, and they follow directly from the design rather than from a gap in the logic. Because energy shares are estimated from published benchmark data, small-scale component profiling, and roofline-based analytical estimation rather than measured at the socket during a full pretraining run, the resulting waste profile is a modeled profile validated against published evidence, not a from-scratch measurement, and it is reported alongside its configuration rather than presented as universally fixed."

**Proposed:** "Reliance on component-level profiling rather than full-run instrumentation is also where the framework's limits sit, and they follow directly from the design rather than from a gap in the logic. Because energy shares are measured by profiling each component in isolation on a single GPU type, rather than at the socket during a full pretraining run, the resulting waste profile describes components measured one at a time and cross-checked against a roofline-based analytical estimate, not a complete training run, and it is reported alongside its configuration rather than presented as universally fixed."

**Why:** energy shares are now measured, not modeled from published benchmarks (Chapter II, Phase 3), and the analytical estimate is a check rather than a source (Research Design, R2). The limits stay stated honestly, and more precisely: isolated components, one GPU type, no full training run. These are the limits Chapter III will discuss (the Unattributed share, the power-capped T4).

## C2. Conceptual framework: the definition of C3

**Current:** "Energy share (C3): the fraction of total energy a component uses, estimated from published benchmark data, small-scale profiling, and analytical estimation."

**Proposed:** "Energy share (C3): the fraction of total energy a component uses, measured by profiling the component's forward and backward passes on a GPU and summarized across repeated passes."

**Why:** Phase 3 now states that published figures do not enter the energy share. The following sentence ("Expected to drift away from FLOP share…") is unchanged.

## C3. Statement of the Problem: how the profile is built

**Current:** "The profile is estimated from published specifications and benchmark evidence, small-scale cloud GPU profiling of forward and backward passes, and analytical estimation, rather than from a full training run."

**Proposed:** "The profile is built from small-scale cloud GPU profiling of forward and backward passes, which measures each component's energy share, together with published architectural specifications for the compute shares and published ablation evidence for capability, rather than from a full training run."

**Why:** it assigns each evidence source to the quantity it actually supplies: profiling for energy (C3), specifications for compute (C2), ablation studies for capability. The research questions that follow are unchanged.

## C4. Research Design: what secondary data supplies

**Current:** "Secondary data analysis (Johnston, 2014) draws on technical reports, model cards, and benchmark results already published by developers such as OpenAI, Google DeepMind, Anthropic, and DeepSeek-AI, together with academic energy benchmarks and component attribution studies (You et al., 2023; Chung et al., 2025; Jain et al., 2025), and component ablation and removal studies that supply the capability evidence (Lan et al., 2020; H. W. Chung et al., 2021; Haviv et al., 2022; Ainslie et al., 2023; Kazemnejad et al., 2023; He & Hofmann, 2024; He, Sun, et al., 2024; Zhu et al., 2025)."

**Proposed:** "Secondary data analysis (Johnston, 2014) draws on the published architectural specifications of the sampled models (Biderman et al., 2023), academic energy benchmarks and component attribution studies (You et al., 2023; Chung et al., 2025; Jain et al., 2025), component ablation and removal studies that supply the capability evidence (Lan et al., 2020; H. W. Chung et al., 2021; Haviv et al., 2022; Ainslie et al., 2023; Kazemnejad et al., 2023; He & Hofmann, 2024; He, Sun, et al., 2024; Zhu et al., 2025), and the published results for each candidate substitution, against which the measured savings are compared."

**Why:** no number in the study comes from developer technical reports or model cards (OpenAI, Google DeepMind, Anthropic, DeepSeek-AI). Those works still appear in the Chapter I literature review as background, which this change does not touch. The new list matches what secondary data actually supplies: specifications (C2), capability evidence (the overlay) and published substitution results (T10). Biderman et al. (2023) is already in the References.
