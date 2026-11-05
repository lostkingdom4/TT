"""Unified Triton kernel configuration baselines.

Subpackages:

* ``baselines.common`` — shared infrastructure (search space, config
  injector, output verification, results store, operator discovery,
  TritonBench setup). Every baseline imports from here.
* ``baselines.random``   — random search over the per-kernel space.
* ``baselines.autotune`` — Triton ``@triton.autotune`` capture.
* ``baselines.bo``       — Bayesian optimization (Optuna TPE).
"""
