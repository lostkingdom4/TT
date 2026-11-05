# Baselines

Unified home for every non-LLM baseline used in the paper's Table 2.

```
baselines/
    common/          # shared search space, injector, verify, results, budget, discover
    random/          # one random config per shape (existing Table 2 baseline)
    autotune/        # Triton @autotune capture (existing Table 2 baseline)
    bo/              # BO (Optuna TPE). Fixed@9 or --budget-policy scaled
    random_search/   # NEW: uniform-random search at scaled budget
    auc_bandit/      # NEW: OpenTuner-style AUC-Bandit ensemble, from scratch
    analyze_comparison.py   # NEW: unified analyzer producing matched@9 + full-budget views
    results/         # centralized per-device output: <device>/<baseline>/...
```

All baselines tune over the same per-(operator, kernel) search space in
`baselines/common/search_space.py`, so the Table 2 numbers are directly
comparable.  The new scaled-budget baselines (`bo_scaled`, `random_search`,
`auc_bandit`) share a single budget policy in
`baselines/common/budget.py`:

```
budget(kernel) = clip(round(0.10 * |search_space|), 9, 150)
```

This keeps the LLM multi-agent system's fixed 9-eval budget competitive
on tiny spaces (11 of 19 kernels floor to 9) while giving BO and the
bandit a fair 150 evals on the 2304-config GEMMs.

## Quick reference

| Baseline | Command | Notes |
|----------|---------|-------|
| Random (existing) | `python -m baselines.random.run_all --ir-dir extract/ir_output/v100` | One random config per shape |
| Autotune | `python -m baselines.autotune.run_all --ir-dir extract/ir_output/v100` | Captures `@triton.autotune` best_config |
| BO fixed@9 | `python -m baselines.bo.run_all --ir-dir extract/ir_output/v100` | Optuna TPE, matches LLM budget |
| BO scaled | `python -m baselines.bo.run_all --ir-dir extract/ir_output/v100 --budget-policy scaled` | Optuna TPE at per-kernel scaled budget -> `bo_scaled/` |
| Random-search scaled | `python -m baselines.random_search.run_all --ir-dir extract/ir_output/v100` | Uniform random @ scaled budget -> `random_search/` |
| AUC-Bandit scaled | `python -m baselines.auc_bandit.run_all --ir-dir extract/ir_output/v100` | DE + greedy-mutation + uniform-random ensemble |

All scaled-budget runners default to `--run-list
run_lists/v100_threshold_include_no_jagged_ln.json` and take the same
core flags as the existing baselines.

### Benchmark harness

All three scaled-budget runners (`bo`, `random_search`, `auc_bandit`)
accept `--harness {compare_configs, tritonbench}`.  The default is
`compare_configs` -- it matches the paper's Full-system measurement
path (extracted kernel `.py` + per-input `*_shapes.json` + `do_bench`
with L2 flush), so the best-latency numbers produced by any scaled
baseline are directly comparable to the Full column of Table 2.

The `tritonbench` harness is the legacy path through
`op.get_example_inputs()`.  Its input distribution is different from
the paper's extracted shapes, so numbers measured under it are **not**
apples-to-apples with the Full column -- keep it only for reproducing
the pre-2026-04 behaviour.

Shared harness code lives in `baselines/common/cc_bench.py`
(`run_operator_cc`).

## Cache-replay (no-GPU mode)

The autotune baseline exhaustively measures every config in the common
search space per (kernel, shape).  That cache can replay BO / random
search / AUC-Bandit in-process, deterministically, without touching
the GPU:

```
python -m baselines.run_cached \
    --cache-dir baselines/results/v100/autotune \
    --strategy random_search bo auc_bandit \
    --seeds 0 1 2 3 4 \
    --output-dir baselines/results/v100
```

Output lands under `baselines/results/v100/<strategy>_cached/` with the
same envelope schema as the GPU runners.  Runtime for the full
15-cacheable-kernel × 3-strategy × 5-seed sweep is ~20 seconds (vs ~2
GPU-days).

**Coverage caveat**: four kernels (`addmm/_addmm_fwd`, `addmm/triton_mm`,
`addmm/triton_tem_fused_addmm_0`, `jagged_softmax`) wrote an
`@triton.autotune`-decorator config list (8-24 entries) rather than
the full common search space, so their cache cannot back a scaled-budget
replay.  Those kernels are skipped by default and still require GPU
runs via `baselines.{bo, random_search, auc_bandit}.run_all`.

## Analysis

After running one or more baselines, generate the matched@9 +
full-budget comparison:

```
python -m baselines.analyze_comparison \
    baselines/results/v100 \
    --output-dir baselines/results/v100/comparison \
    [--llm-dir path/to/multiagent/outputs]
```

Outputs: per-kernel convergence plots, cross-kernel normalized-speedup
heatmaps, and a nested `summary.json` with per-kernel-then-geomean
aggregates (the geomean convention used throughout the paper).

## Budget-scaled strategy internals

The three scaled-budget strategies scale their internal hyperparameters
honestly with the budget -- no hand-tuned per-kernel knobs:

- **BO (Optuna TPE)**: `n_startup_trials = min(5, max(3, budget // 3))`.
  At budget=9, TPE gets 3 random warmup points + 6 surrogate-guided
  trials (vs Optuna's default 10 startup trials, which would have made
  budget=9 entirely random).
- **AUC-Bandit**: window `W = min(budget, 20)`; DE population
  `pop = max(3, min(budget // 2, 10))`.  At budget=9: W=9, pop=4.
- **Random-search**: budget-agnostic.

## Search space

`baselines/common/search_space.py` exposes:

- `get_search_space(operator, kernel=None)` — per-kernel entries for
  the 19 run-list kernels, with a per-operator fallback for legacy
  TritonBench ops (softmax, sum, vector_add, ...).
- `CONFIG_LEVEL_PARAMS = {"num_warps", "num_stages", "num_ctas"}` —
  distinguishes launch-level params from kernel constexprs.

## Results layout

```
baselines/results/
    v100/
        random/          existing
        autotune/        existing
        bo/              existing fixed@9
        bo_scaled/       new: Optuna TPE @ scaled budget
        random_search/   new: uniform random @ scaled budget
        auc_bandit/      new: AUC-Bandit ensemble @ scaled budget
        comparison/      generated by analyze_comparison.py
```

Per-kernel envelopes are named `<operator>__<kernel>.json` to
disambiguate operators with multiple kernels (e.g. `addmm` has
`_addmm_fwd`, `triton_mm`, `triton_tem_fused_addmm_0`).  Legacy
per-operator files (from pre-reorg runs) are preserved without a
kernel suffix.

## Dependencies

`optuna>=3.5.0` is required for the BO baseline (listed in the
top-level `requirements.txt`).  Random and Autotune baselines have no
new dependencies beyond the existing Triton / TritonBench stack.
