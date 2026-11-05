#!/usr/bin/env python3
"""
Tests for the closed-loop benchmark verification in the multi-agent optimizer.

Tests the benchmark_node, convergence regression handling, fallback logic,
and backward compatibility using mocked LLM and benchmark functions.
"""

import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

# Allow importing from the agent directory
AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_benchmark_result(
    orig_median: float = 10.0,
    rec_median: float = 8.0,
    error: str = None,
) -> Dict[str, Any]:
    """Build a benchmark_config_pair-style result dict."""
    if error:
        return {
            "error": error,
            "original_stats": {},
            "recommended_stats": {},
            "original_config": {},
            "recommended_config": {},
            "grid": [128],
        }
    speedup = orig_median / rec_median if rec_median > 0 else None
    improvement_pct = ((1 - rec_median / orig_median) * 100) if orig_median > 0 else None
    return {
        "original_stats": {"median": orig_median, "mean": orig_median, "min": orig_median, "max": orig_median, "std_dev": 0.0, "samples": 20},
        "recommended_stats": {"median": rec_median, "mean": rec_median, "min": rec_median, "max": rec_median, "std_dev": 0.0, "samples": 20},
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "original_config": {"num_warps": 4, "block_size": {"XBLOCK": 128}},
        "recommended_config": {"num_warps": 8, "block_size": {"XBLOCK": 256}},
        "grid": [128],
    }


def _make_optimization_result(
    num_warps: int = 8,
    block_size: Dict[str, int] = None,
    confidence: float = 0.9,
) -> Dict[str, Any]:
    """Build a coordinator-style optimization_result."""
    bs = block_size or {"XBLOCK": 256}
    return {
        "kernel_name": "test_kernel",
        "original_config": {"num_warps": 4, "num_stages": 2, "block_size": {"XBLOCK": 128}, "grid": {"x": 128}},
        "recommended_config": {"num_warps": num_warps, "num_stages": 2, "block_size": bs, "grid": {"x": 128}},
        "expected_improvement": "20%",
        "confidence_score": confidence,
        "warp_analysis": {},
        "sm_analysis": {},
        "gpu_analysis": {},
        "coordinator_reasoning": "test",
        "tradeoffs": [],
        "implementation_instructions": "",
        "risk_assessment": "low",
    }


def _make_initial_state(benchmark_config: Dict = None) -> Dict[str, Any]:
    """Build a minimal OptimizationState dict for testing."""
    from agent_system_langgraph.state import KernelConfig

    kc = KernelConfig(
        kernel_name="test_kernel",
        ttgir="module {}",
        num_warps=4,
        num_stages=2,
        block_size={"XBLOCK": 128},
        grid={"x": 128},
    )
    return {
        "kernel_config": kc,
        "messages": [],
        "warp_analysis": None,
        "sm_analysis": None,
        "gpu_analysis": None,
        "constraints": {},
        "warp_proposal": None,
        "sm_proposal": None,
        "gpu_proposal": None,
        "optimization_result": None,
        "iteration": 0,
        "max_iterations": 3,
        "converged": False,
        "error": None,
        "iteration_history": [],
        "refinement_focus": None,
        "previous_coordinator_feedback": None,
        "benchmark_results": None,
        "baseline_benchmark_result": None,
        "regression_detected": False,
        "benchmark_iterations": 0,
        "benchmark_config": benchmark_config,
        # Multi-candidate search fields (Approach 5)
        "candidate_configs": [],
        "benchmark_results_per_candidate": [],
    }


# ---------------------------------------------------------------------------
# Test: _benchmark_node
# ---------------------------------------------------------------------------

class TestBenchmarkNode:
    """Tests for GPUOptimizationOrchestrator._benchmark_candidates_node (single-candidate)."""

    def _make_orchestrator(self, benchmark_fn=None):
        """Create an orchestrator with mocked LLM and optional benchmark_fn."""
        with patch("agent_system_langgraph.orchestrator.LLMProvider"):
            with patch("agent_system_langgraph.orchestrator.WarpAgent"):
                with patch("agent_system_langgraph.orchestrator.SMAgent"):
                    with patch("agent_system_langgraph.orchestrator.GPUAgent"):
                        with patch("agent_system_langgraph.orchestrator.CoordinatorAgent"):
                            from agent_system_langgraph.orchestrator import GPUOptimizationOrchestrator
                            orch = GPUOptimizationOrchestrator.__new__(GPUOptimizationOrchestrator)
                            orch.benchmark_fn = benchmark_fn
                            orch.include_agent_proposals = False
                            return orch

    def test_passthrough_when_no_benchmark_fn(self):
        """Benchmark node should be a no-op when benchmark_fn is None."""
        orch = self._make_orchestrator(benchmark_fn=None)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["converged"] is True
        assert result["regression_detected"] is False
        assert result["benchmark_iterations"] == 0

    def test_passthrough_when_no_benchmark_config(self):
        """Benchmark node should be a no-op when benchmark_config is None."""
        mock_fn = MagicMock()
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=None)
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        mock_fn.assert_not_called()
        assert result["converged"] is True
        assert result["benchmark_iterations"] == 0

    def test_detects_improvement(self):
        """When recommended is faster, should not detect regression."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is False
        assert result["converged"] is True  # Left as-is from coordinator
        assert result["benchmark_iterations"] == 1
        assert result["benchmark_results"]["speedup"] == pytest.approx(1.25)

    def test_detects_regression(self):
        """When recommended is slower, should detect regression and unset converged."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=14.0))
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True  # Set by coordinator

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is True
        assert result["converged"] is False  # Overridden by benchmark node
        assert result["benchmark_iterations"] == 1
        assert result["benchmark_results"]["speedup"] == pytest.approx(10.0 / 14.0)

    def test_benchmark_exception_treated_as_regression(self):
        """When benchmark_fn raises, should treat as regression."""
        mock_fn = MagicMock(side_effect=RuntimeError("GPU OOM"))
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is True
        assert result["converged"] is False
        assert result["benchmark_iterations"] == 1
        assert "error" in result["benchmark_results"]
        # Per-candidate results should capture the specific error
        per_cand = result["benchmark_results_per_candidate"]
        assert len(per_cand) == 1
        assert "GPU OOM" in per_cand[0]["error"]

    def test_benchmark_error_in_result_treated_as_regression(self):
        """When benchmark returns an error dict, should treat as regression."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(error="config failed"))
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is True
        assert result["converged"] is False
        # Per-candidate results should capture the error
        per_cand = result["benchmark_results_per_candidate"]
        assert len(per_cand) == 1
        assert per_cand[0]["error"] == "config failed"

    def test_caches_baseline_on_first_run(self):
        """Baseline benchmark result should be cached after first benchmark."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = self._make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config={"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"})
        state["optimization_result"] = _make_optimization_result()

        result = orch._benchmark_candidates_node(state)

        assert result["baseline_benchmark_result"] is not None
        assert result["baseline_benchmark_result"]["median"] == 10.0


# ---------------------------------------------------------------------------
# Test: _check_convergence_node regression handling
# ---------------------------------------------------------------------------

class TestConvergenceRegressionHandling:
    """Tests for regression handling in _check_convergence_node."""

    def _make_orchestrator(self):
        with patch("agent_system_langgraph.orchestrator.LLMProvider"):
            with patch("agent_system_langgraph.orchestrator.WarpAgent"):
                with patch("agent_system_langgraph.orchestrator.SMAgent"):
                    with patch("agent_system_langgraph.orchestrator.GPUAgent"):
                        with patch("agent_system_langgraph.orchestrator.CoordinatorAgent"):
                            from agent_system_langgraph.orchestrator import GPUOptimizationOrchestrator
                            orch = GPUOptimizationOrchestrator.__new__(GPUOptimizationOrchestrator)
                            orch.benchmark_fn = None
                            orch.include_agent_proposals = False
                            return orch

    def test_regression_retry_with_feedback(self):
        """On regression with retries left, should set feedback and continue."""
        orch = self._make_orchestrator()
        state = _make_initial_state()
        state["optimization_result"] = _make_optimization_result()
        state["regression_detected"] = True
        state["benchmark_iterations"] = 1
        state["benchmark_results"] = _make_benchmark_result(orig_median=10.0, rec_median=14.0)

        result = orch._check_convergence_node(state)

        assert result["converged"] is False
        assert result["regression_detected"] is False  # Reset for next round
        assert result["refinement_focus"] == "benchmark_regression"
        assert "BENCHMARK REGRESSION" in result["previous_coordinator_feedback"]
        assert "14.0ms" in result["previous_coordinator_feedback"]
        assert "10.0ms" in result["previous_coordinator_feedback"]

    def test_regression_fallback_after_max_retries(self):
        """After MAX_BENCHMARK_RETRIES regressions, should fallback to original config."""
        orch = self._make_orchestrator()
        state = _make_initial_state()
        state["optimization_result"] = _make_optimization_result()
        state["regression_detected"] = True
        state["benchmark_iterations"] = 3  # MAX_BENCHMARK_RETRIES
        state["benchmark_results"] = _make_benchmark_result(orig_median=10.0, rec_median=14.0)

        result = orch._check_convergence_node(state)

        assert result["converged"] is True
        assert result["regression_detected"] is False
        opt = result["optimization_result"]
        assert opt["recommendation"] == "keep_original"
        assert opt["recommended_config"]["num_warps"] == 4  # Original
        assert opt["recommended_config"]["block_size"]["XBLOCK"] == 128  # Original
        assert opt["confidence_score"] == 1.0

    def test_no_regression_falls_through_to_standard_checks(self):
        """When no regression, standard convergence checks should apply."""
        orch = self._make_orchestrator()
        state = _make_initial_state()
        state["optimization_result"] = _make_optimization_result(confidence=0.9)
        state["regression_detected"] = False

        result = orch._check_convergence_node(state)

        # High confidence -> converged
        assert result["converged"] is True


# ---------------------------------------------------------------------------
# Test: benchmark_config_pair
# ---------------------------------------------------------------------------

class TestBenchmarkConfigPair:
    """Tests for the benchmark_config_pair function in compare_configs.py."""

    def test_returns_speedup_and_stats(self):
        """Verify the return dict structure from benchmark_config_pair."""
        from compare_configs import benchmark_config_pair

        # We can't run actual benchmarks without GPU, but we can verify the
        # function signature and error handling with mocked internals.
        with patch("compare_configs.load_tensor_shapes") as mock_load, \
             patch("compare_configs.create_kernel_inputs") as mock_inputs, \
             patch("compare_configs._load_kernel") as mock_kernel, \
             patch("compare_configs._benchmark_config") as mock_bench:

            mock_load.return_value = {"grid": [128], "kwargs": {}}
            mock_inputs.return_value = ([], {})
            mock_kernel.return_value = MagicMock()
            mock_bench.side_effect = [
                {"median": 10.0, "mean": 10.0, "min": 9.5, "max": 10.5, "std_dev": 0.3, "samples": 20},
                {"median": 8.0, "mean": 8.0, "min": 7.5, "max": 8.5, "std_dev": 0.2, "samples": 20},
            ]

            result = benchmark_config_pair(
                original_config={"num_warps": 4},
                recommended_config={"num_warps": 8},
                shapes_file=Path("/tmp/shapes.json"),
                kernel_module=Path("/tmp/kernel.py"),
                kernel_name="test_kernel",
            )

            assert result["speedup"] == pytest.approx(1.25)
            assert result["improvement_pct"] == pytest.approx(20.0)
            assert "error" not in result
            assert result["original_stats"]["median"] == 10.0
            assert result["recommended_stats"]["median"] == 8.0

    def test_returns_error_on_benchmark_failure(self):
        """When a benchmark fails, should return error in result dict."""
        from compare_configs import benchmark_config_pair

        with patch("compare_configs.load_tensor_shapes") as mock_load, \
             patch("compare_configs.create_kernel_inputs") as mock_inputs, \
             patch("compare_configs._load_kernel") as mock_kernel, \
             patch("compare_configs._benchmark_config") as mock_bench:

            mock_load.return_value = {"grid": [128], "kwargs": {}}
            mock_inputs.return_value = ([], {})
            mock_kernel.return_value = MagicMock()
            mock_bench.side_effect = [
                {"median": 10.0, "mean": 10.0, "min": 9.5, "max": 10.5, "std_dev": 0.3, "samples": 20},
                {"error": "kernel launch failed"},
            ]

            result = benchmark_config_pair(
                original_config={"num_warps": 4},
                recommended_config={"num_warps": 8},
                shapes_file=Path("/tmp/shapes.json"),
                kernel_module=Path("/tmp/kernel.py"),
                kernel_name="test_kernel",
            )

            assert "error" in result
            assert "recommended config benchmark failed" in result["error"]


# ---------------------------------------------------------------------------
# Test: backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that existing callers without benchmark_fn work unchanged."""

    def test_orchestrator_without_benchmark_fn(self):
        """Orchestrator created without benchmark_fn should have no benchmark side effects."""
        with patch("agent_system_langgraph.orchestrator.LLMProvider"):
            with patch("agent_system_langgraph.orchestrator.WarpAgent"):
                with patch("agent_system_langgraph.orchestrator.SMAgent"):
                    with patch("agent_system_langgraph.orchestrator.GPUAgent"):
                        with patch("agent_system_langgraph.orchestrator.CoordinatorAgent"):
                            from agent_system_langgraph.orchestrator import GPUOptimizationOrchestrator
                            orch = GPUOptimizationOrchestrator.__new__(GPUOptimizationOrchestrator)
                            orch.benchmark_fn = None

                            state = _make_initial_state()
                            state["optimization_result"] = _make_optimization_result()
                            state["converged"] = True

                            result = orch._benchmark_candidates_node(state)

                            assert result["converged"] is True
                            assert result["benchmark_iterations"] == 0
                            assert result["benchmark_results"] is None

    def test_state_has_all_new_fields(self):
        """Verify initial state includes all benchmark and multi-candidate fields with safe defaults."""
        state = _make_initial_state()

        assert state["benchmark_results"] is None
        assert state["baseline_benchmark_result"] is None
        assert state["regression_detected"] is False
        assert state["benchmark_iterations"] == 0
        assert state["benchmark_config"] is None
        # Approach 5 fields
        assert state["candidate_configs"] == []
        assert state["benchmark_results_per_candidate"] == []


# ---------------------------------------------------------------------------
# Test: submodule path detection in _load_kernel
# ---------------------------------------------------------------------------

class TestSubmodulePathDetection:
    """Tests for submodule sys.path handling in _load_kernel."""

    def test_detects_submodule_root(self):
        """_get_submodule_path returns submodule root for paths under submodules/."""
        from compare_configs import _get_submodule_path
        p = Path("/root/tritonbench/submodules/generative-recommenders/generative_recommenders/ops/triton/triton_addmm.py")
        with patch.object(Path, "is_dir", return_value=True):
            result = _get_submodule_path(p)
        assert result == Path("/root/tritonbench/submodules/generative-recommenders")

    def test_returns_none_for_operator_path(self):
        """_get_submodule_path returns None for standard operator paths."""
        from compare_configs import _get_submodule_path
        p = Path("/root/tritonbench/tritonbench/operators/softmax/operator.py")
        result = _get_submodule_path(p)
        assert result is None

    def test_handles_various_submodule_names(self):
        """_get_submodule_path works for any submodule directory name."""
        from compare_configs import _get_submodule_path
        for name in ["xformers", "flash-attention", "aiter", "ThunderKittens"]:
            p = Path(f"/root/tritonbench/submodules/{name}/some_pkg/kernel.py")
            with patch.object(Path, "is_dir", return_value=True):
                result = _get_submodule_path(p)
            assert result == Path(f"/root/tritonbench/submodules/{name}"), f"Failed for {name}"

    def test_load_kernel_adds_submodule_path(self):
        """_load_kernel wraps exec_module with _add_path for submodule kernels."""
        from compare_configs import _load_kernel

        fake_path = Path("/root/tritonbench/submodules/generative-recommenders/generative_recommenders/ops/triton/triton_addmm.py")

        with patch("compare_configs._get_submodule_path") as mock_detect, \
             patch("importlib.util.spec_from_file_location") as mock_spec, \
             patch("importlib.util.module_from_spec") as mock_from_spec, \
             patch("compare_configs._add_path") as mock_add_path:

            mock_detect.return_value = Path("/root/tritonbench/submodules/generative-recommenders")
            mock_loader = MagicMock()
            mock_spec.return_value = MagicMock(loader=mock_loader)
            mock_mod = MagicMock()
            mock_mod._addmm_fwd = MagicMock()
            mock_from_spec.return_value = mock_mod
            mock_add_path.return_value.__enter__ = MagicMock()
            mock_add_path.return_value.__exit__ = MagicMock(return_value=False)

            _load_kernel(fake_path, "_addmm_fwd")

            mock_add_path.assert_called_once_with(
                str(Path("/root/tritonbench/submodules/generative-recommenders"))
            )

    def test_load_kernel_no_path_addition_for_non_submodule(self):
        """_load_kernel should not wrap exec_module for non-submodule kernels."""
        from compare_configs import _load_kernel

        fake_path = Path("/tmp/some_kernel.py")

        with patch("compare_configs._get_submodule_path", return_value=None), \
             patch("importlib.util.spec_from_file_location") as mock_spec, \
             patch("importlib.util.module_from_spec") as mock_from_spec, \
             patch("compare_configs._add_path") as mock_add_path:

            mock_loader = MagicMock()
            mock_spec.return_value = MagicMock(loader=mock_loader)
            mock_mod = MagicMock()
            mock_mod.test_kernel = MagicMock()
            mock_from_spec.return_value = mock_mod

            _load_kernel(fake_path, "test_kernel")

            mock_add_path.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Autotuner unwrapping in _unwrap_autotuner
# ---------------------------------------------------------------------------

class TestAutotunerUnwrapping:
    """Tests for _unwrap_autotuner which bypasses Triton Autotuner wrappers."""

    def _make_mock_autotuner(self, inner_fn, config_kwargs_list):
        """Create a mock that looks like triton.runtime.autotuner.Autotuner."""
        from triton.runtime.autotuner import Autotuner
        mock = MagicMock(spec=Autotuner)
        mock.fn = inner_fn
        mock.configs = []
        for kw in config_kwargs_list:
            cfg = MagicMock()
            cfg.kwargs = kw
            mock.configs.append(cfg)
        # Make isinstance check work
        mock.__class__ = Autotuner
        return mock

    def test_unwrap_strips_autotuner(self):
        """Autotuner wrapping a JIT fn -> returns inner fn + config keys."""
        from compare_configs import _unwrap_autotuner

        inner_fn = MagicMock(name="jit_function")
        autotuner = self._make_mock_autotuner(
            inner_fn,
            [{"BLOCK_M": 32, "BLOCK_N": 64}, {"BLOCK_M": 64, "BLOCK_N": 128}],
        )
        result_fn, keys = _unwrap_autotuner(autotuner)
        assert result_fn is inner_fn
        assert keys == {"BLOCK_M", "BLOCK_N"}

    def test_unwrap_nested_autotuners(self):
        """Nested Autotuner layers -> both stripped, keys collected from both."""
        from compare_configs import _unwrap_autotuner

        inner_fn = MagicMock(name="jit_function")
        inner_autotuner = self._make_mock_autotuner(
            inner_fn, [{"BLOCK_K": 32}],
        )
        outer_autotuner = self._make_mock_autotuner(
            inner_autotuner, [{"BLOCK_M": 64, "BLOCK_N": 128}],
        )
        result_fn, keys = _unwrap_autotuner(outer_autotuner)
        assert result_fn is inner_fn
        assert keys == {"BLOCK_M", "BLOCK_N", "BLOCK_K"}

    def test_unwrap_preserves_non_autotuner(self):
        """Plain JIT fn -> returned unchanged, empty keys."""
        from compare_configs import _unwrap_autotuner

        jit_fn = MagicMock(name="jit_function")
        result_fn, keys = _unwrap_autotuner(jit_fn)
        assert result_fn is jit_fn
        assert keys == set()

    def test_unwrap_preserves_heuristics(self):
        """Autotuner wrapping Heuristics -> stops at Heuristics layer."""
        from compare_configs import _unwrap_autotuner

        heuristics_fn = MagicMock(name="heuristics")
        autotuner = self._make_mock_autotuner(
            heuristics_fn, [{"BLOCK_M": 64, "GROUP_M": 8}],
        )
        result_fn, keys = _unwrap_autotuner(autotuner)
        assert result_fn is heuristics_fn
        assert keys == {"BLOCK_M", "GROUP_M"}

    def test_unwrap_graceful_on_import_error(self):
        """If Autotuner import fails, returns input unchanged."""
        from compare_configs import _unwrap_autotuner

        kernel_fn = MagicMock(name="some_kernel")
        with patch.dict("sys.modules", {"triton.runtime.autotuner": None}):
            result_fn, keys = _unwrap_autotuner(kernel_fn)
        assert result_fn is kernel_fn
        assert keys == set()
