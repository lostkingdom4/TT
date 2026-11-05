"""Shared infrastructure for all Triton configuration baselines.

Public API
----------

search_space
    ``get_search_space(operator, kernel=None)`` returns the valid search
    space for a given (operator, kernel) pair.  Kernel-level entries are
    preferred; falls back to the per-operator entry, then the default.

budget
    ``scaled_budget(size)`` implements the scaled-budget policy used by
    ``bo_scaled``, ``random_search``, and ``auc_bandit``:
    ``clip(round(0.10 * size), 9, 150)``.  ``print_budget_table(entries)``
    produces a human-readable per-kernel budget table at run start.

injector
    ``_RandomConfigInjector`` and ``_FixedConfigInjector`` monkeypatch
    Triton to force a chosen kernel configuration during a benchmark run.

verify
    ``_verify_outputs`` performs post-benchmark cross-implementation
    output verification (baseline vs Triton).

results
    ``BaseResultsStore`` persists per-kernel JSON envelopes, writes
    ``index.json`` / ``summary.json`` / ``summary.csv`` and supports
    resume semantics.  Each baseline subclasses with its own CSV schema.

tritonbench_setup
    ``load_operator`` builds a TritonBench op-class instance with
    ``warmup=5, rep=20``, a fresh Triton cache, and the shared bench
    arguments used across the paper.

discover
    ``discover_operators(ir_dir)`` scans ``extract/ir_output/<gpu>/``
    for operator directories and their shape JSONs.  Also exports
    ``get_device_name``, ``print_discovery_summary``.
"""
