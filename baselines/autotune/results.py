"""Autotune (exhaustive grid-search) baseline results store.

For every kernel in the run-list, every config in the space is benchmarked
and the fastest is recorded. Schema mirrors ``baselines/bo/results.py`` so
downstream analysis scripts can read both folders with the same parser.
"""

import math
from typing import Any, Dict, List

from baselines.common.results import BaseResultsStore


class BaselineResultsStore(BaseResultsStore):
    BASELINE_NAME = "autotune"
    FIELD_SCHEMA = [
        ("operator", "operator"),
        ("kernel", "kernel"),
        ("has_autotune", "has_autotune"),
        ("input_id", "input_id"),
        ("x_val", "x_val"),
        ("triton_impl_name", "impl_name"),
        ("n_configs", "n_configs"),
        ("config_source", "config_source"),
        ("original_config_latency_ms", "original_config_latency_ms"),
        ("best_latency_ms", "best_latency_ms"),
        ("latency_ms", "latency_ms"),
        ("speedup", "speedup"),
        ("best_config", "best_config"),
        ("best_trial_id", "best_trial_id"),
        ("trial_errors", "trial_errors"),
        ("verified", "verified"),
        ("verification_error", "verification_error"),
        ("status", "status"),
        ("error", "error"),
    ]

    def __init__(self, results_root, device):
        super().__init__(results_root, device, baseline_name="autotune")

    def _operator_summary_extras(
        self, data: Dict[str, Any], shapes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        speedups = [
            s["speedup"] for s in shapes
            if s.get("speedup") is not None and s["speedup"] > 0
        ]
        improved = sum(1 for v in speedups if v > 1.005)
        best = max(speedups) if speedups else None
        geomean = (
            math.exp(sum(math.log(v) for v in speedups) / len(speedups))
            if speedups else None
        )
        return {
            "shapes_with_speedup": len(speedups),
            "shapes_improved": improved,
            "best_speedup": best,
            "geomean_speedup": geomean,
        }
