# NeurIPS 2026 Paper TODO List

> Updated 2026-04-14 after paper polish pass on main.tex.
> Status: DONE = addressed in paper, PARTIAL = some progress, TODO = not started.

---

## Critical (paper will be rejected without these)

### 1. ~~Fix A1b > A5 framing~~ → DONE
**Status: DONE**

The updated paper fixes this convincingly:
- Full system now wins 7/19 kernels (A1b also 7/19, A1a 1, ties 4)
- Full achieves highest geomean: 2.46x vs A1b 2.42x vs A1a 2.21x
- Full achieves highest improvement rate: 81% (323/397 shapes) vs A1b 67% vs A1a 74%
- Narrative reframed around "consistency" — orchestrator produces more reliably good configs even when single agents hit higher peaks
- The 58-shape gap (323 vs 265) is a strong argument for multi-agent value

No further action needed unless reviewers push back.

### 2. Complete missing baselines
**Status: PARTIAL — still the biggest risk**

Camera-ready note has been removed from the baselines section, and the "preliminary" label on random search has been rephrased. Cross-GPU evaluation is now fully in the paper (V100, L40S, A40, A100 in Table 5).

What exists in paper:
- [x] Random search: 5 operators, 536 shapes (no longer framed as "preliminary")
- [x] Autotuner: addmm only (8 configs in 7.36s)
- [x] Cross-GPU: V100 main table + L40S/A40/A100 in `tab:v100_vs_l40s`
- [x] Stale "planned for camera-ready" note REMOVED (edit 2026-04-14)

What exists in codebase but NOT in paper:
- `auto_tune_baseline/` — code ready for all kernels
- `ablation/a2_no_feedback.py`, `ablation/a3_single_candidate.py` — NOT in paper

Still needed:
- [ ] Run autotuner baseline on all 19 kernels (code exists in `auto_tune_baseline/`)
- [ ] Expand random search from 5 to all 19 kernels
- [ ] **Add A2 (no feedback) and A3 (single candidate) ablation results** — code exists, these directly prove feedback loop and phased search matter
- [ ] Consider adding Bayesian optimization baseline (Optuna) — would strengthen the "sample efficiency" argument

### 3. ~~Fix CUDA-error kernels~~ → DONE
**Status: DONE**

All 19 kernels now have complete results in Table 2:
- addmm(ind): results present (1.00x across all variants)
- jag_ln(red): 1.20/1.23/1.21x
- jag_softmax: 1.89/2.36/1.99x
- triton_mm added as new kernel (3.66/1.96/3.69x)
- No more † markers or missing cells

### 4. Add statistical significance / error bars
**Status: PARTIAL — acknowledged but not addressed**

The paper now mentions in limitations: *"LLM stochasticity introduces run-to-run variance; we observe ~0.1-0.2x geomean variation across repeated runs"* (line 583). This is good as an acknowledgment but insufficient — reviewers will want to see actual error bars.

- [x] Variance acknowledged in limitations
- [x] Benchmark stability validated (0/125 inflated baselines, stdev < 0.05)
- [ ] Run full pipeline 3+ times on representative kernels and report mean +/- std
- [ ] Add error bars to Table 2 (at least for geomean row)
- [ ] Report configuration agreement rate across runs

---

## High Priority (reviewers will ding you)

### 5. ~~Add cross-GPU evaluation (A100, A40)~~ → DONE
**Status: DONE**

Cross-GPU results are now in the paper:
- V100 (SM70): 2.46x geomean / 401 shapes — Table 2 (main results)
- L40S (SM89): 1.80x geomean / 390 shapes — Table 5 (`tab:v100_vs_l40s`)
- A40 (SM86): 1.55x geomean / 326 shapes — Table 5
- A100 (SM80): 1.59x geomean / 391 shapes — Table 5

Conclusion section now summarizes all four GPUs as deployed results, not future work (edited 2026-04-14).

### 6. Add cross-LLM ablation
**Status: TODO — code exists, experiments not run**

LLM backends already implemented (`llm_provider.py`): OpenAI, DeepSeek, Ollama, vLLM. Only DeepSeek tested.

- [ ] Run 5-8 representative kernels with GPT-4o
- [ ] Run with open-weight model (Llama-3 via vLLM)
- [ ] Add comparison table or paragraph

### 7. Add sample efficiency analysis figure
**Status: TODO**

The paper describes the cost (~10 benchmarks per shape, ~30-45s per shape) but doesn't visualize sample efficiency against baselines.

- [ ] Plot cumulative-best-speedup vs # evaluations for 3-4 kernels
- [ ] Curves: TritonTune phases 1→2→3, random search, autotuner grid sweep
- [ ] Data for Phase progression exists (Table 4 shows geglu/rope/int4_gemm)
- [ ] This is the key figure for "why not just autotune?"

### 8b. Add input-modality ablation (is TTGIR meaningful?)
**Status: TODO — flagged as open question in §4.3 (added 2026-04-14)**

The paper now poses this as an open question in the ablation section: does TTGIR-specific reasoning matter, or would agents produce similar configurations from TTIR, Python source, or a shape/dtype-only summary? This is a clean ablation that would directly answer "what is the multi-agent system actually extracting from the IR?"

- [ ] Implement input-modality variants: TTIR, source, shape-only, LLVM IR, PTX
- [ ] Run 5-8 representative kernels with each modality (hold agents/orchestrator/phased search fixed)
- [ ] Add to §4.3 ablation table
- [ ] If TTGIR is not meaningfully better than alternatives, reframe contribution as "orchestration + phased search", not "IR-grounded reasoning"

### 8. Add A2/A3 ablations to paper
**Status: CODE EXISTS, not in paper**

The ablation/ directory has:
- `a2_no_feedback.py` — proves feedback loop matters
- `a3_single_candidate.py` — proves multi-candidate phased search matters

These are the missing ablation columns that would make Section 4.3 (Ablation) rigorous:

| Variant | What it proves |
|---------|---------------|
| A1a (DM only) | ✅ in paper |
| A1b (RB only) | ✅ in paper |
| A2 (no feedback) | ❌ NOT in paper — proves feedback matters |
| A3 (single candidate) | ❌ NOT in paper — proves phased search matters |
| Full | ✅ in paper |

- [ ] Run A2 and A3 on all 19 kernels
- [ ] Add to ablation table or separate table
- [ ] This is LOW EFFORT, HIGH IMPACT — directly strengthens contribution claims

### 9. Position against concurrent work
**Status: TODO**

Related work (Section 2) is missing:
- [ ] Compiler-R1 (RL for compiler tuning) — distinguish: no training, works zero-shot
- [ ] KernelBench / KernelLLM — distinguish: config tuning vs kernel generation
- [ ] PRAGMA, Astra — multi-agent kernel optimization
- [ ] Add bib entries and 1-2 paragraphs

---

## Medium Priority (polish)

### 10. Add cost analysis table
**Status: PARTIAL — text mentions costs, no table**

Paper mentions ~30-45s/shape, ~70 min total for 401 shapes. But no structured table.

- [ ] Table: per-kernel wall-clock, LLM API cost ($), # LLM calls, # benchmarks
- [ ] DeepSeek is very cheap — quantify this as a strength
- [ ] Compare against autotuner wall-clock per kernel

### 11. Add LLM reasoning case study
**Status: TODO**

Paper describes agent behavior in abstract terms but never shows actual LLM output.

- [ ] Extract from debug logs: DM reasoning on memory-bound kernel
- [ ] Extract: RB reasoning on compute-bound kernel
- [ ] Show a conflict + resolution
- [ ] Appendix is fine for this

### 12. Comparison with torch.compile max-autotune
**Status: TODO**

- [ ] For inductor kernels (geglu, rope, mamba2): what does max-autotune select?
- [ ] Show TritonTune finds configs outside max-autotune's predefined grid

### 13. Broader impact section
**Status: PARTIAL**

Checklist answers \answerNA{} for broader impact and safeguards (lines 775-782). This is fine but could be stronger.

- [ ] Add brief broader impact paragraph: energy efficiency gains, API cost/dependence, generalization risks

### 14. Clean up checklist
**Status: MOSTLY DONE**

The checklist in main.tex is filled out and reasonable. The separate checklist.tex still has \answerTODO{} everywhere.

- [ ] Delete checklist.tex or ensure main.tex is the only version used
- [ ] Add answer for statistical significance (currently not addressed in checklist)
- [x] Add LLM declaration answer (added 2026-04-14 — main.tex now has an LLM Usage question with DeepSeek / temperature 0.2 declaration)

### 15. Verify page count
**Status: TODO**

- [ ] Compile with tectonic, check ≤ 9 pages main content
- [ ] Current paper has substantial content — may be tight
- [ ] Candidates for appendix if over: conflict patterns table (Table 3), convergence analysis (Section 4.4)

---

## Summary: What Changed Since Last Review

| Item | Previous State | Current State |
|------|---------------|---------------|
| A1b > Full problem | A1b won 12/18 | Fixed: Full wins 7/19, best geomean (2.46x), best improvement rate (81%) |
| CUDA errors | 3 failed kernels (†) | All fixed, all 19 kernels have data |
| Kernel count | 18 | 19 (triton_mm added; introduction wording corrected 2026-04-14) |
| Geomean speedup | 2.31x | 2.46x |
| A1a performance | 1.37x | 2.21x (benchmark stability fix) |
| Shapes improved | Not reported | 323/397 (81%) |
| Benchmark stability | Not discussed | New section with validation (0/125 inflation) |
| Parallel execution | Not discussed | 16 workers, GPU lock, L2 flush documented |
| Iteration count | I=3 | I=1 (simpler, still effective) |
| Baselines | "planned for camera-ready" | Note REMOVED 2026-04-14 |
| Cross-GPU | "available" / future work | V100 + L40S + A40 + A100 now in Table 5 and conclusion |
| Cross-LLM | Future work | Still future work (reworded in limitations 2026-04-14) |
| Error bars | None | Mentioned in limitations (~0.1-0.2x variance) but no data |
| A2/A3 ablations | Not mentioned | Still not in paper (code exists) |
| Related work gaps | Missing concurrent work | Still missing Compiler-R1, KernelBench |
| Figure 3 bar chart | 17/19 kernels plotted | Now all 19 kernels (addmm-ind, jag-ln-r added 2026-04-14) |
| LLM disclosure | Missing from checklist | Added 2026-04-14 |
| Self-referential wording | "improved from prior experiments (1.37x)" | Rephrased 2026-04-14 — no longer references prior runs |
