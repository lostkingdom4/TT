"""Bayesian-Optimization (Optuna TPE) baseline.

For each kernel, the runner loops over all shapes and runs a
per-shape Optuna study with ``n_trials`` evaluations (default 9 —
matches the Full system's ~9 benchmarks/shape budget).  The best
config found for each shape is recorded together with a speedup
against the original launch config.

CLI::

    python -m baselines.bo.run_all \\
        --ir-dir extract/ir_output/v100 \\
        --run-list run_lists/v100_improved_only.json \\
        --n-trials 9 --export-csv
"""
