# Benchmark Measurement Stability

Measures how accurate and repeatable our Triton kernel benchmarks are by running the same configuration multiple times and computing variance, standard deviation, and coefficient of variation across runs.

Uses the same `triton.testing.do_bench()` benchmarking infrastructure as `multiagent_triton_runner.py` and `compare_configs.py` — no LLM involvement.

## Quick Start

Run from the **project root** (`PyTorch_Intel_GPU_MLIR-Extraction/`):

```bash
# Verified working command — all 9 registered kernels, 10 runs each, with plot
python measurement_experiment/measure_benchmark_stability.py --num-runs 10 --warmup 25 --rep 100 --plot
```

## All Commands

```bash
# All registered kernels, 10 runs each (default)
python measurement_experiment/measure_benchmark_stability.py

# Specific operators with visualization
python measurement_experiment/measure_benchmark_stability.py \
    --operators softmax,addmm,layer_norm --num-runs 20 --plot

# Quick scan, fewer runs
python measurement_experiment/measure_benchmark_stability.py \
    --num-runs 5 --warmup 10 --rep 50

# Verbose logging to see per-run details
python measurement_experiment/measure_benchmark_stability.py \
    --operators softmax --num-runs 10 --verbose

# Use a specific device's IR data (required when multiple device dirs exist)
python measurement_experiment/measure_benchmark_stability.py \
    --cache-dir extract/ir_output/a100

# Available device dirs: v100, a100, A40, l40s
python measurement_experiment/measure_benchmark_stability.py \
    --cache-dir extract/ir_output/v100 --num-runs 20 --plot
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--operators` | `all` | Comma-separated operator names, or `all` for all 9 registered kernels |
| `--num-runs` | `10` | Number of repeated benchmark runs per config |
| `--warmup` | `25` | Warmup time in ms per `do_bench` call |
| `--rep` | `100` | Repetition time in ms per `do_bench` call |
| `--max-shapes` | `1` | Max shapes files per operator |
| `--seed` | `42` | Random seed for input generation |
| `--device` | `cuda` | Device to benchmark on |
| `--cache-dir` | auto-detect | Root IR/shapes directory (e.g. `extract/ir_output/v100`). Auto-detects if only one device dir exists; errors with options if multiple are found |
| `--output-dir` | `measurement_experiment/results` | Where to save JSON results |
| `--plot` | off | Generate matplotlib box plot and CoV bar chart |
| `--verbose` | off | Show per-run timing details |

Note: `--warmup` and `--rep` are **durations in milliseconds**, not iteration counts. Faster kernels produce more samples per run.

## Registered Kernels

The script includes a built-in registry of 9 verified kernel configurations matching those used by `multiagent_triton_runner.py`:

| Operator | Kernel | Source |
|---|---|---|
| softmax | `Operator.softmax_kernel` | tritonbench/operators/softmax |
| addmm | `_addmm_fwd` | submodules/generative-recommenders |
| layer_norm (triton) | `_layer_norm_fwd_fused` | tritonbench/operators/layer_norm |
| layer_norm (fused_no_bias) | `_layer_norm_fwd_fused_no_bias` | tritonbench/operators/layer_norm |
| dropout (seeded) | `_seeded_triton_dropout` | tritonbench/operators/low_mem_dropout |
| dropout (triton) | `_triton_dropout` | tritonbench/operators/low_mem_dropout |
| sum | `triton_sum_kernel_scalar_result` | tritonbench/operators/sum |
| vector_add | `triton_add_kernel` | tritonbench/operators/vector_add |
| vector_exp | `triton_exp_kernel` | tritonbench/operators/vector_exp |

**Excluded kernels**: jagged_mean, jagged_softmax, jagged_sum — these require semantically valid "offsets" tensors that cannot be generated with random inputs.

## Sample Output (Tesla V100-SXM2-32GB)

```
----------------------------------------------------------------------------------------------------------------------------
Operator                           Kernel                          Runs    Med(ms)  StdDev(ms)   CoV(%)  Spread(%)  Samp/Run
----------------------------------------------------------------------------------------------------------------------------
softmax                            softmax_kernel                    10     0.0123      0.0008     6.80      16.67        51
addmm                              _addmm_fwd                        10     0.7352      0.0055     0.75       2.92        97
layer_norm (triton)                _layer_norm_fwd_fused             10     0.0594      0.0000     0.00       0.00       275
layer_norm (fused_no_bias)         _layer_norm_fwd_fused_no_bias     10     0.0584      0.0000     0.00       0.00       251
dropout (seeded)                   _seeded_triton_dropout            10     0.0051      0.0000     0.00       0.00       323
dropout (triton)                   _triton_dropout                   10     0.0041      0.0004    10.04      25.00       324
sum                                triton_sum_kernel_scalar_result    10     0.0051      0.0000     0.00       0.00       322
vector_add                         triton_add_kernel                 10     0.0051      0.0000     0.00       0.00       296
vector_exp                         triton_exp_kernel                 10     0.0051      0.0000     0.00       0.00       323
----------------------------------------------------------------------------------------------------------------------------

Overall: 9/9 operators benchmarked
Average CoV: 1.96%  |  Best: layer_norm (triton) (0.00%)  |  Worst: dropout (triton) (10.04%)
```

### JSON Results

Saved to `measurement_experiment/results/stability_results_{timestamp}.json` with:

- **metadata**: device, GPU name, run parameters, timestamp
- **results**: per-operator breakdown with:
  - `per_run_stats`: median/mean/min/max/std_dev for each individual run
  - `cross_run_stats.medians_analysis`: variance, std dev, CoV%, spread% across the N medians
  - `cross_run_stats.means_analysis`: same across the N means
  - `cross_run_stats.pooled_analysis`: all samples from all runs combined
- **summary**: avg/min/max CoV%, most/least stable operator

### Visualization (`--plot`)

Saved to `measurement_experiment/results/stability_plot.png`:
- Box plot of median distributions across runs per operator
- CoV bar chart with color coding (green < 1%, yellow 1-5%, red > 5%)

## Interpreting Results

| CoV (%) | Stability | Implication for Optimization |
|---|---|---|
| < 1% | Very stable | Speedups > 1-2% are trustworthy |
| 1-5% | Reasonable | Need speedup > 2x CoV to be confident |
| > 5% | Noisy | Reported speedups may be measurement noise |

**Key findings from V100 run:**
- **addmm** (0.75% CoV): Most meaningful kernel to optimize — large enough runtime (0.73ms) for stable measurement, and small CoV means speedup claims are reliable.
- **layer_norm** (0.00% CoV): Perfectly stable at 0.058ms, but be aware this may be timer resolution quantization.
- **softmax** (6.80% CoV): Moderate noise — a reported 5% speedup could be measurement artifact.
- **dropout (triton)** (10.04% CoV): High noise due to very fast execution (0.004ms). Speedup claims under 20% are unreliable.
- **Very fast kernels** (sum, vector_add, vector_exp at ~0.005ms): Show 0% CoV because they hit the timer resolution floor. Differences between configs may be invisible.

## Column Reference

| Column | Meaning |
|---|---|
| **Med(ms)** | Median of the N per-run medians — the "answer" the benchmark gives |
| **StdDev(ms)** | Standard deviation of the N medians — absolute wobble |
| **CoV(%)** | Coefficient of variation = StdDev/Mean x 100 — normalized stability |
| **Spread(%)** | (max - min) / median x 100 — worst-case gap between runs |
| **Samp/Run** | Samples per run (depends on kernel speed, not `--rep` directly) |
