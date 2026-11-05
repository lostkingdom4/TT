"""RandomSearch baseline results store.

Shares the speedup/geomean aggregation with the BO store -- structurally
identical, different CSV column names for the chosen config.
"""

import math
from typing import Any, Dict, List

from baselines.common.results import BaseResultsStore


class RandomSearchResultsStore(BaseResultsStore):
    BASELINE_NAME = "random_search"
    FIELD_SCHEMA = [
        ("operator", "operator"),
        ("kernel", "kernel"),
        ("has_autotune", "has_autotune"),
        ("input_id", "input_id"),
        ("x_val", "x_val"),
        ("triton_impl_name", "impl_name"),
        ("n_trials", "n_trials"),
        ("budget", "budget"),
        ("budget_policy", "budget_policy"),
        ("space_size", "space_size"),
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

    def __init__(
        self, results_root, device, n_trials=9, seed=42,
        baseline_name: str = "random_search",
    ):
        super().__init__(
            results_root, device, baseline_name=baseline_name,
            extra_index={"n_trials": n_trials, "seed": seed},
        )
        self.n_trials = n_trials
        self.seed = seed

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
