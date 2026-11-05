"""Random-config baseline results store.

Thin subclass of ``BaseResultsStore`` — adds the ``random_config`` and
``random_seed`` fields to the CSV schema.
"""

from baselines.common.results import BaseResultsStore


class RandomConfigResultsStore(BaseResultsStore):
    BASELINE_NAME = "random"
    FIELD_SCHEMA = [
        ("operator", "operator"),
        ("kernel", "kernel"),
        ("random_seed", "random_seed"),
        ("has_autotune", "has_autotune"),
        ("input_id", "input_id"),
        ("x_val", "x_val"),
        ("triton_impl_name", "impl_name"),
        ("latency_ms", "latency_ms"),
        ("random_config", "random_config"),
        ("verified", "verified"),
        ("verification_error", "verification_error"),
        ("status", "status"),
        ("error", "error"),
    ]

    def __init__(self, results_root, device, seed=42):
        super().__init__(
            results_root, device,
            baseline_name="random",
            extra_index={"random_seed": seed},
        )
        self.seed = seed

    def _operator_summary_extras(self, data, shapes):
        return {
            "with_random_config": sum(
                1 for s in shapes if s.get("random_config")
            ),
        }
