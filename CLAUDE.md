# energy-anatomy

Profiling and analysis code for Bill's master's thesis: a roofline-based diagnostic waste score (component energy share minus compute share) for transformer pretraining. Deadline for the full manuscript: **Oct 30, 2026**.

## Read first
- `DESIGN.md`: the spec. Every design decision is closed and recorded in its Section 11. Don't reopen them; ask Bill if something truly conflicts.
- The thesis rules this code must satisfy: hypothesis H0/H1 (memory-bound components have C4 > 0), decision rule = C4 > 0 in a majority of profiled configs where the component is memory-bound **and** median C4 > IQR(C4) across passes; tested per component.

## Locked setup
- GPU: Kaggle free NVIDIA T4 (not P100). Precisions FP32 and FP16 (no BF16). Energy via Zeus `ZeusMonitor`.
- Models: Pythia 160M (L12, d768, h12, vocab 50304), 1.4B (L24, d2048, h16, vocab 50304), 6.9B (L32, d4096, h32, vocab 50432). Rotary pct 0.25, parallel residual, LayerNorm eps 1e-5, exact GELU, untied embeddings, no bias on embed_out. Source: EleutherAI/pythia `models/*.yml` (Hugging Face Hub was unreachable from the cloud session).
- Published param counts for tests: 160M = 162,322,944 (non-embedding 85,056,000); 1.4B = 1,414,647,808 (1,208,602,624); 6.9B = 6,857,302,016 (6,444,163,072).
- Grid: 3 sizes × seq 512/1024/2048 × FP32/FP16 = 18 configs, 8,192 tokens per pass.
- Units: embed_lookup, ln (count 2L+1, pooled with final norm), attn_proj, rope, attn_core (eager ops), mlp, out_proj, loss, block (denominator). Substitutions: sdpa_core (mem-efficient backend), linear_core (chunked causal, elu+1), gqa_proj (kv heads = heads/4, includes the kv repeat), factorized_embed in/out (E = 128), rmsnorm.
- Protocol: 20 warm-up, rotating input buffers > 4 MB L2, reps calibrated so each window ≥ 2 s, N = 10 passes, idle window every 30 min. Shares use total energy; dynamic energy is an appendix check.

## Stack
- Python 3.11, uv with a committed lockfile, `src/anatomy` package, `anatomy` CLI, pytest. `transformers` is a test-only dependency (HF GPTNeoXLayer equivalence test).
- Kaggle runs a thin notebook: clone, `pip install -e .[gpu]`, `anatomy profile --resume`.
- Dashboard later: React 19 + Vite + TypeScript, same stack as Bill's LKD dashboard, reading `export/*.json`.

## How Bill works
- Ask at every meaningful design choice instead of deciding. Keep replies short and skip basics.
- Never fabricate profiling numbers. If something can't run on the Mac, say so.

## Next milestone (Oct 15)
Scaffold, configs, component and substitution modules, analytical model (FLOPs, bytes, C1, C2, ridge, t_hat), and tests, then generate `data/tables/T2` and `data/derived/analytical.csv`. Harness and Kaggle notebook come after that (Oct 16).
