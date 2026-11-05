# BO Baseline — Optuna TPE

Bayesian Optimization baseline for the NeurIPS paper's Table 2 "BO"
column.  Runs Optuna's TPE sampler per-shape for every kernel in
`run_lists/v100_improved_only.json` (the 19 kernels of the paper's main
results).

## Design

For each `(operator, kernel)`:

1. Load the TritonBench operator.
2. For every shape the operator produces, measure an **original-config**
   latency using `triton.testing.do_bench(warmup=5, rep=20, return_mode="all")`.
3. Run an Optuna TPE study with `n_trials` iterations (default **9** —
   matches the Full multi-agent system's ~9 benchmarks-per-shape budget).
   Each trial:
   - samples a categorical config from
     `baselines.common.search_space.get_search_space(operator, kernel)`;
   - installs it via `baselines.common.injector._FixedConfigInjector`;
   - benchmarks the shape again and reports the median latency back to TPE.
4. Record best config, `speedup = original / best_latency`, and the
   full trial history.

Results land under
`baselines/results/<device>/bo/<operator>__<kernel>.json`.

## Usage

```bash
# 19 kernels × 9 trials/shape (default)
python -m baselines.bo.run_all \
    --ir-dir extract/ir_output/v100 --export-csv

# Larger BO budget for a stronger baseline
python -m baselines.bo.run_all \
    --ir-dir extract/ir_output/v100 --n-trials 30

# Single kernel (smoke test)
python -m baselines.bo.run_all \
    --ir-dir extract/ir_output/v100 \
    --operator rope --kernel triton_poi_fused_add_cat_mul_0 \
    --n-trials 9
```

## Outputs

- `baselines/results/<device>/bo/<operator>__<kernel>.json` — per-kernel
  envelope with all per-shape BO runs.
- `baselines/results/<device>/bo/summary.json` — aggregate with
  per-kernel geomean speedup (the number that feeds Table 2's BO column).
- `baselines/results/<device>/bo/summary.csv` — flat per-shape CSV.

## Dependencies

- `optuna>=3.5.0` (added to `requirements.txt`).
- Same TritonBench / Triton stack as the other baselines.
