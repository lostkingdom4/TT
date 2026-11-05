#!/usr/bin/env python3
"""
Tests for Approach 5: Multi-Config Search with LLM-Guided Sampling.

Tests the multi-candidate benchmark node, coordinator multi-candidate output,
retry integration, and backward compatibility.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
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


def _make_candidate(
    num_warps: int = 8,
    block_size: Dict[str, int] = None,
    rationale: str = "test",
    confidence: float = 0.8,
    variant_type: str = "balanced",
) -> Dict[str, Any]:
    """Build a single candidate config entry."""
    bs = block_size or {"XBLOCK": 256}
    return {
        "config": {
            "num_warps": num_warps,
            "num_stages": 2,
            "block_size": bs,
            "grid": {"x": 128},
        },
        "rationale": rationale,
        "confidence": confidence,
        "variant_type": variant_type,
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
        # Approach 5 fields
        "candidate_configs": [],
        "benchmark_results_per_candidate": [],
    }


_BC = {"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"}


def _make_orchestrator(benchmark_fn=None):
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


# ---------------------------------------------------------------------------
# Test: _benchmark_candidates_node
# ---------------------------------------------------------------------------

class TestBenchmarkCandidatesNode:
    """Tests for GPUOptimizationOrchestrator._benchmark_candidates_node."""

    def test_passthrough_no_benchmark_fn(self):
        """No-op when benchmark_fn is None."""
        orch = _make_orchestrator(benchmark_fn=None)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["converged"] is True
        assert result["benchmark_iterations"] == 0
        assert result["benchmark_results_per_candidate"] == []

    def test_passthrough_no_benchmark_config(self):
        """No-op when benchmark_config is None."""
        mock_fn = MagicMock()
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=None)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        mock_fn.assert_not_called()
        assert result["converged"] is True
        assert result["benchmark_iterations"] == 0

    def test_fallback_no_candidates_uses_recommended(self):
        """When candidate_configs is empty, falls back to recommended_config."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = []  # Empty — should fall back
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        # Phase 1 calls benchmark once; Phase 3 (neighborhood search) may add more
        assert mock_fn.call_count >= 1
        assert result["regression_detected"] is False
        assert result["benchmark_iterations"] == 1
        assert len(result["benchmark_results_per_candidate"]) == 1

    def test_selects_fastest_candidate(self):
        """Given 3 candidates with different speedups, selects the fastest."""
        # Candidate 0: 1.2x, Candidate 1: 1.5x, Candidate 2: 1.1x
        mock_fn = MagicMock(side_effect=[
            _make_benchmark_result(orig_median=10.0, rec_median=8.33),   # ~1.2x
            _make_benchmark_result(orig_median=10.0, rec_median=6.67),   # ~1.5x
            _make_benchmark_result(orig_median=10.0, rec_median=9.09),   # ~1.1x
        ])
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=16),
            _make_candidate(num_warps=4, block_size={"XBLOCK": 512}),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is False
        assert result["converged"] is True
        assert result["benchmark_iterations"] == 1
        # Best is candidate 1 (num_warps=16, speedup ~1.5x)
        sel = result["optimization_result"]["benchmark_selection"]
        assert sel["best_candidate_index"] == 1
        assert sel["num_candidates_tested"] == 3
        assert sel["num_successful"] == 3
        # Recommended config updated to winning candidate
        assert result["optimization_result"]["recommended_config"]["num_warps"] == 16

    def test_all_regress_sets_regression_detected(self):
        """When all candidates are slower than original, triggers retry."""
        mock_fn = MagicMock(side_effect=[
            _make_benchmark_result(orig_median=10.0, rec_median=12.5),  # 0.8x
            _make_benchmark_result(orig_median=10.0, rec_median=11.1),  # 0.9x
        ])
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=16),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is True
        assert result["converged"] is False  # Overridden for retry
        assert result["benchmark_iterations"] == 1
        # benchmark_results should have the best (least bad) candidate
        assert result["benchmark_results"]["speedup"] < 1.0

    def test_all_fail_sets_regression_detected(self):
        """When all benchmark calls raise, triggers retry."""
        mock_fn = MagicMock(side_effect=RuntimeError("GPU OOM"))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=16),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is True
        assert result["converged"] is False
        assert result["benchmark_iterations"] == 1
        assert "error" in result["benchmark_results"]

    def test_skips_errors_selects_from_survivors(self):
        """One raises, one returns error, one succeeds — picks the survivor."""
        mock_fn = MagicMock(side_effect=[
            RuntimeError("GPU OOM"),
            _make_benchmark_result(error="config failed"),
            _make_benchmark_result(orig_median=10.0, rec_median=7.7),  # ~1.3x
        ])
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=16),
            _make_candidate(num_warps=6),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is False
        assert result["converged"] is True
        per_cand = result["benchmark_results_per_candidate"]
        assert len(per_cand) == 3
        # Two errors, one success
        errors = [r for r in per_cand if "error" in r]
        successes = [r for r in per_cand if "speedup" in r and "error" not in r]
        assert len(errors) == 2
        assert len(successes) == 1
        # Winner is candidate 2
        sel = result["optimization_result"]["benchmark_selection"]
        assert sel["best_candidate_index"] == 2

    def test_caches_baseline_on_first_benchmark(self):
        """baseline_benchmark_result should be populated from first successful run."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]

        result = orch._benchmark_candidates_node(state)

        assert result["baseline_benchmark_result"] is not None
        assert result["baseline_benchmark_result"]["median"] == 10.0

    def test_benchmark_results_per_candidate_populated(self):
        """Verify benchmark_results_per_candidate has correct structure."""
        mock_fn = MagicMock(side_effect=[
            _make_benchmark_result(orig_median=10.0, rec_median=8.0),
            _make_benchmark_result(orig_median=10.0, rec_median=9.0),
        ])
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=6),
        ]

        result = orch._benchmark_candidates_node(state)

        per_cand = result["benchmark_results_per_candidate"]
        assert len(per_cand) == 2
        for i, r in enumerate(per_cand):
            assert r["candidate_index"] == i
            assert "config" in r
            assert "speedup" in r
            assert "stats" in r
            assert "original_stats" in r

    def test_single_candidate_equivalent(self):
        """Single candidate should behave like old single-benchmark node."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate(num_warps=8)]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["regression_detected"] is False
        assert result["converged"] is True
        assert result["benchmark_iterations"] == 1
        assert result["benchmark_results"]["speedup"] == pytest.approx(1.25)

    def test_benchmark_selection_metadata(self):
        """optimization_result should have benchmark_selection metadata."""
        mock_fn = MagicMock(side_effect=[
            _make_benchmark_result(orig_median=10.0, rec_median=8.0),
            _make_benchmark_result(orig_median=10.0, rec_median=7.0),
        ])
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=6),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        sel = result["optimization_result"]["benchmark_selection"]
        assert sel["num_candidates_tested"] == 2
        assert sel["num_successful"] == 2
        assert sel["best_candidate_index"] == 1
        assert sel["best_speedup"] == pytest.approx(10.0 / 7.0)
        assert len(sel["all_results"]) == 2


# ---------------------------------------------------------------------------
# Test: CoordinatorAgent multi-candidate output
# ---------------------------------------------------------------------------

class TestCoordinatorMultiCandidate:
    """Tests for CoordinatorAgent producing candidate_configs list."""

    def _make_coordinator_state(self):
        """Build state with all analyses populated for coordinator."""
        state = _make_initial_state()
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 90}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        state["warp_proposal"] = [{"agent": "WarpAgent", "proposal": {"changes": {}}, "confidence": 0.7}]
        state["sm_proposal"] = [{"agent": "SMAgent", "proposal": {"changes": {}}, "confidence": 0.6}]
        state["gpu_proposal"] = [{"agent": "GPUAgent", "proposal": {"changes": {}}, "confidence": 0.5}]
        return state

    def _make_mock_llm(self, response_json: Dict):
        """Create a mock LLM that returns the given JSON response."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        return mock_llm

    def test_produces_candidate_configs_list(self):
        """When LLM returns candidate_configs, they should be stored in state."""
        from agent_system_langgraph.agents import CoordinatorAgent
        from agent_system_langgraph.knowledge_base import get_architecture

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "More warps for latency hiding",
                    "confidence": 0.8,
                    "variant_type": "conservative",
                },
                {
                    "config": {"num_warps": 4, "num_stages": 3, "block_size": {"XBLOCK": 512}, "grid": {"x": 64, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "Larger blocks for coalescing",
                    "confidence": 0.6,
                    "variant_type": "aggressive",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "10-20%", "confidence_score": 0.7, "risk_level": "medium", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = self._make_mock_llm(response_json)
        arch = get_architecture("cuda:80")
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        result = agent.coordinate(state)

        assert len(result["candidate_configs"]) == 2
        assert result["candidate_configs"][0]["config"]["num_warps"] == 8
        assert result["candidate_configs"][1]["config"]["num_warps"] == 4
        # Top candidate used as recommended_config
        assert result["optimization_result"]["recommended_config"]["num_warps"] == 8

    def test_fallback_single_recommended_config(self):
        """When LLM returns old format, it's wrapped into a 1-element list."""
        from agent_system_langgraph.agents import CoordinatorAgent
        from agent_system_langgraph.knowledge_base import get_architecture

        response_json = {
            "recommended_config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "10%", "confidence_score": 0.7, "risk_level": "medium", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = self._make_mock_llm(response_json)
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        result = agent.coordinate(state)

        # Wrapped into a 1-element candidate_configs list
        assert len(result["candidate_configs"]) == 1
        assert result["candidate_configs"][0]["config"]["num_warps"] == 8
        assert result["candidate_configs"][0]["variant_type"] == "balanced"

    def test_all_candidates_block_sizes_validated(self):
        """Block sizes should be validated in all candidates (Approach 4 integration)."""
        from agent_system_langgraph.agents import CoordinatorAgent
        from agent_system_langgraph.state import KernelConfig

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "test1", "confidence": 0.8, "variant_type": "balanced",
                },
                {
                    "config": {"num_warps": 4, "num_stages": 2, "block_size": {"XBLOCK": 512}, "grid": {"x": 64, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "test2", "confidence": 0.6, "variant_type": "aggressive",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "10%", "confidence_score": 0.7, "risk_level": "medium", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        # Set constexpr_param_names to BLOCK_SIZE — should remap XBLOCK
        state["kernel_config"] = KernelConfig(
            kernel_name="test_kernel",
            ttgir="module {}",
            num_warps=4,
            num_stages=2,
            block_size={"BLOCK_SIZE": 128},
            grid={"x": 128},
            constexpr_param_names=["BLOCK_SIZE"],
        )

        result = agent.coordinate(state)

        # Both candidates should have BLOCK_SIZE, not XBLOCK
        for cand in result["candidate_configs"]:
            assert "BLOCK_SIZE" in cand["config"]["block_size"]
            assert "XBLOCK" not in cand["config"]["block_size"]

    def test_top_candidate_set_as_recommended(self):
        """optimization_result['recommended_config'] should be the first candidate."""
        from agent_system_langgraph.agents import CoordinatorAgent

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 6, "num_stages": 2, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "top pick", "confidence": 0.9, "variant_type": "balanced",
                },
                {
                    "config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 512}, "grid": {"x": 64, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "alternative", "confidence": 0.6, "variant_type": "aggressive",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "15%", "confidence_score": 0.9, "risk_level": "low", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        result = agent.coordinate(state)

        assert result["optimization_result"]["recommended_config"]["num_warps"] == 6


# ---------------------------------------------------------------------------
# Test: Retry integration
# ---------------------------------------------------------------------------

class TestRetryIntegration:
    """Tests for multi-candidate node integration with the existing retry loop."""

    def test_regression_triggers_retry_via_check_convergence(self):
        """All candidates regress -> check_convergence feeds timing back for retry."""
        orch = _make_orchestrator()

        # First: benchmark_candidates_node sets regression_detected
        mock_fn = MagicMock(side_effect=[
            _make_benchmark_result(orig_median=10.0, rec_median=12.0),  # 0.83x
            _make_benchmark_result(orig_median=10.0, rec_median=11.0),  # 0.91x
        ])
        orch.benchmark_fn = mock_fn
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            _make_candidate(num_warps=8),
            _make_candidate(num_warps=16),
        ]
        state["converged"] = True

        state = orch._benchmark_candidates_node(state)
        assert state["regression_detected"] is True
        assert state["converged"] is False

        # Then: check_convergence_node should set feedback for retry
        state = orch._check_convergence_node(state)
        assert state["converged"] is False
        assert "BENCHMARK REGRESSION" in state["previous_coordinator_feedback"]
        assert state["refinement_focus"] == "benchmark_regression"

    def test_retry_exhaustion_falls_back_to_original(self):
        """After MAX_BENCHMARK_RETRIES, check_convergence falls back to original."""
        from agent_system_langgraph.orchestrator import MAX_BENCHMARK_RETRIES
        orch = _make_orchestrator()

        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["regression_detected"] = True
        state["benchmark_iterations"] = MAX_BENCHMARK_RETRIES  # Exhausted
        state["benchmark_results"] = _make_benchmark_result(orig_median=10.0, rec_median=14.0)

        state = orch._check_convergence_node(state)

        assert state["converged"] is True
        opt = state["optimization_result"]
        assert opt["recommendation"] == "keep_original"
        assert opt["recommended_config"]["num_warps"] == 4  # Original

    def test_benchmark_feedback_includes_candidate_results(self):
        """On retry, coordinator prompt should include per-candidate results."""
        from agent_system_langgraph.agents import CoordinatorAgent

        # Simulate state after all candidates regressed
        state = _make_initial_state(benchmark_config=_BC)
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 90}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        state["warp_proposal"] = [{"agent": "WarpAgent", "proposal": {"changes": {}}, "confidence": 0.7}]
        state["sm_proposal"] = [{"agent": "SMAgent", "proposal": {"changes": {}}, "confidence": 0.6}]
        state["gpu_proposal"] = [{"agent": "GPUAgent", "proposal": {"changes": {}}, "confidence": 0.5}]
        state["benchmark_iterations"] = 1
        state["benchmark_results"] = {
            "original_stats": {"median": 10.0},
            "recommended_stats": {"median": 12.0},
            "speedup": 0.83,
            "recommended_config": {"num_warps": 8},
        }
        state["benchmark_results_per_candidate"] = [
            {"candidate_index": 0, "speedup": 0.83, "config": {"num_warps": 8}},
            {"candidate_index": 1, "speedup": 0.91, "config": {"num_warps": 16}},
        ]

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 4, "num_stages": 3, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "revised", "confidence": 0.7, "variant_type": "conservative",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "minor", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "5%", "confidence_score": 0.7, "risk_level": "low", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        result = agent.coordinate(state)

        # Verify the LLM was called with a prompt containing candidate feedback
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]  # Second positional arg is user_prompt
        assert "CRITICAL - BENCHMARK FEEDBACK" in user_prompt
        assert "candidate" in user_prompt.lower()
        assert "0.83" in user_prompt  # Best speedup referenced


# ---------------------------------------------------------------------------
# Test: Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """Verify backward compatibility with systems not using multi-candidate."""

    def test_initial_state_has_new_fields(self):
        """Verify initial state includes multi-candidate fields."""
        state = _make_initial_state()
        assert state["candidate_configs"] == []
        assert state["benchmark_results_per_candidate"] == []

    def test_no_benchmark_fn_runs_without_error(self):
        """Graph should complete without benchmark_fn (no benchmarking)."""
        orch = _make_orchestrator(benchmark_fn=None)
        state = _make_initial_state()
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        assert result["converged"] is True
        assert result["benchmark_results"] is None
        assert result["benchmark_results_per_candidate"] == []

    def test_empty_candidate_config_skipped(self):
        """Candidates with empty config dicts are skipped gracefully."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [
            {"config": {}, "rationale": "empty", "confidence": 0.1},
            _make_candidate(num_warps=6),
        ]
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        # Phase 1: 1 call (empty skipped); Phase 3 (neighborhood search) may add more
        assert mock_fn.call_count >= 1
        per_cand = result["benchmark_results_per_candidate"]
        assert len(per_cand) == 2
        assert per_cand[0]["error"] == "empty config"
        assert "speedup" in per_cand[1]


# ---------------------------------------------------------------------------
# Test: _sanitize_candidate_config
# ---------------------------------------------------------------------------

class TestSanitizeCandidateConfig:
    """Tests for num_warps power-of-2 sanitization."""

    def test_valid_num_warps_unchanged(self):
        """Power-of-2 num_warps should pass through unchanged."""
        from agent_system_langgraph.orchestrator import _sanitize_candidate_config
        for nw in [1, 2, 4, 8, 16, 32]:
            config = {"num_warps": nw, "block_size": {"XBLOCK": 256}}
            result = _sanitize_candidate_config(config)
            assert result["num_warps"] == nw

    def test_non_power_of_2_rounded(self):
        """Non-power-of-2 num_warps should be rounded to nearest valid value."""
        from agent_system_langgraph.orchestrator import _sanitize_candidate_config
        # 6 -> 4 (equidistant between 4 and 8; min picks first = 4)
        assert _sanitize_candidate_config({"num_warps": 6})["num_warps"] == 4
        # 3 -> 2 (equidistant between 2 and 4; min picks first = 2)
        assert _sanitize_candidate_config({"num_warps": 3})["num_warps"] == 2
        # 5 -> 4 (equidistant, min picks 4)
        assert _sanitize_candidate_config({"num_warps": 5})["num_warps"] == 4
        # 12 -> 8 (equidistant between 8 and 16; min picks first = 8)
        assert _sanitize_candidate_config({"num_warps": 12})["num_warps"] == 8
        # 7 -> 8 (closer to 8 than 4)
        assert _sanitize_candidate_config({"num_warps": 7})["num_warps"] == 8
        # 10 -> 8 (closer to 8 than 16)
        assert _sanitize_candidate_config({"num_warps": 10})["num_warps"] == 8

    def test_no_num_warps_unchanged(self):
        """Config without num_warps should pass through unchanged."""
        from agent_system_langgraph.orchestrator import _sanitize_candidate_config
        config = {"block_size": {"XBLOCK": 256}}
        result = _sanitize_candidate_config(config)
        assert "num_warps" not in result

    def test_sanitization_applied_in_benchmark_loop(self):
        """Non-power-of-2 num_warps should be fixed before benchmark call."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate(num_warps=6)]  # Invalid!
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        # benchmark_fn should be called with sanitized num_warps=4 (6 rounds to 4)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["recommended_config"]["num_warps"] == 4


# ---------------------------------------------------------------------------
# Test: num_candidates parameter
# ---------------------------------------------------------------------------

class TestNumCandidates:
    """Tests for configurable num_candidates parameter."""

    def _make_coordinator_state(self):
        state = _make_initial_state()
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 90}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        state["warp_proposal"] = [{"agent": "WarpAgent", "proposal": {"changes": {}}, "confidence": 0.7}]
        state["sm_proposal"] = [{"agent": "SMAgent", "proposal": {"changes": {}}, "confidence": 0.6}]
        state["gpu_proposal"] = [{"agent": "GPUAgent", "proposal": {"changes": {}}, "confidence": 0.5}]
        return state

    def test_num_candidates_in_prompt(self):
        """Coordinator prompt should include the requested num_candidates."""
        from agent_system_langgraph.agents import CoordinatorAgent

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}, "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "test", "confidence": 0.8, "variant_type": "balanced",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "10%", "confidence_score": 0.7, "risk_level": "medium", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        agent.coordinate(state, num_candidates=7)

        # Check the prompt sent to the LLM contains "exactly 7"
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]
        assert "exactly 7 candidate configs" in user_prompt

    def test_default_num_candidates_is_3(self):
        """Default num_candidates should be 3."""
        from agent_system_langgraph.agents import CoordinatorAgent

        response_json = {
            "candidate_configs": [],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "minor", "optimization_strategy": "test", "conflicts_resolved": [], "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "0%", "confidence_score": 0.5, "risk_level": "low", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [], "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.3, "sm_proposal_weight": 0.3, "gpu_proposal_weight": 0.3, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [], "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = self._make_coordinator_state()
        agent.coordinate(state)  # Uses default num_candidates=3

        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]
        assert "exactly 3 candidate configs" in user_prompt


# ---------------------------------------------------------------------------
# Test: All agents generate multiple proposals
# ---------------------------------------------------------------------------

def _make_agent_state():
    """Build a minimal state dict for agent analysis tests."""
    from agent_system_langgraph.state import KernelConfig

    kc = KernelConfig(
        kernel_name="test_kernel",
        ttgir="module {}",
        num_warps=4,
        num_stages=2,
        block_size={"XBLOCK": 128},
        grid={"x": 128},
        source_code="@triton.jit\ndef test_kernel(): pass",
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
        "benchmark_config": None,
        "candidate_configs": [],
        "benchmark_results_per_candidate": [],
    }


def _mock_llm_with_proposals(proposals: List[Dict], extra_fields: Dict = None):
    """Create a mock LLM that returns a JSON response with given optimization_proposals."""
    response = {
        "coalescing_efficiency": {"estimated_percentage": 85},
        "bank_conflicts": {"severity": "none", "estimated_conflicts_per_warp": 0, "conflict_sources": []},
        "warp_divergence": {"present": False, "divergence_points": [], "impact": "low"},
        "cache_utilization": {"l1_hit_rate_estimate": 0.8, "cache_line_utilization": "good"},
        # SM-specific fields
        "occupancy_analysis": {"current_occupancy_pct": 80, "limiting_factor": "threads", "target_occupancy_pct": 90},
        "shared_memory_analysis": {"current_usage_bytes": 0, "utilization_efficiency": "good"},
        "register_pressure": {"estimated_registers_per_thread": 32, "spilling_risk": "none"},
        "warp_scheduling": {"latency_hiding_effectiveness": "good", "memory_bound_stalls": False, "suggestions": []},
        # GPU-specific fields
        "grid_parallelism": {"current_ctas": 128, "ctas_per_sm": 1.6, "saturation_level": "good"},
        "l2_cache_analysis": {"data_locality_pattern": "medium", "estimated_l2_hit_rate": 0.5},
        "memory_bandwidth": {"estimated_utilization_pct": 60, "bottleneck_type": "bandwidth_bound"},
        "load_balancing": {"grid_uniformity": "good", "tail_effect_risk": "low"},
        # proposals
        "optimization_proposals": proposals,
        "key_insights": ["insight1"],
    }
    if extra_fields:
        response.update(extra_fields)

    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps(response)
    mock_llm.parse_json_response.side_effect = lambda r: json.loads(r)
    return mock_llm


class TestAgentMultiProposal:
    """Tests for multi-proposal output from all three specialist agents."""

    def _three_proposals(self):
        return [
            {"strategy": "conservative", "num_warps": 4, "changes": {"XBLOCK": 256},
             "reasoning": "test", "expected_improvement": "5%", "confidence": 0.8,
             "variant_type": "conservative", "risks": []},
            {"strategy": "balanced", "num_warps": 8, "changes": {"XBLOCK": 512},
             "reasoning": "test", "expected_improvement": "15%", "confidence": 0.7,
             "variant_type": "balanced", "risks": []},
            {"strategy": "aggressive", "num_warps": 16, "changes": {"XBLOCK": 1024},
             "reasoning": "test", "expected_improvement": "30%", "confidence": 0.5,
             "variant_type": "aggressive", "risks": ["occupancy drop"]},
        ]

    def test_warp_agent_stores_list(self):
        """WarpAgent should store all proposals as a list."""
        from agent_system_langgraph.agents import WarpAgent

        mock_llm = _mock_llm_with_proposals(self._three_proposals())
        agent = WarpAgent(mock_llm, "cuda:80")
        state = _make_agent_state()

        result = agent.analyze(state, num_candidates=3)

        assert isinstance(result["warp_proposal"], list)
        assert len(result["warp_proposal"]) == 3
        for p in result["warp_proposal"]:
            assert p["agent"] == "WarpAgent"
            assert "proposal" in p
            assert "confidence" in p

    def test_sm_agent_accepts_num_candidates(self):
        """SMAgent.analyze() should accept num_candidates without crashing."""
        from agent_system_langgraph.agents import SMAgent

        mock_llm = _mock_llm_with_proposals(self._three_proposals())
        agent = SMAgent(mock_llm, "cuda:80")
        state = _make_agent_state()

        # Should not raise TypeError
        result = agent.analyze(state, num_candidates=5)
        assert result["sm_analysis"] is not None

    def test_sm_agent_stores_list(self):
        """SMAgent should store all proposals as a list."""
        from agent_system_langgraph.agents import SMAgent

        mock_llm = _mock_llm_with_proposals(self._three_proposals())
        agent = SMAgent(mock_llm, "cuda:80")
        state = _make_agent_state()

        result = agent.analyze(state, num_candidates=3)

        assert isinstance(result["sm_proposal"], list)
        assert len(result["sm_proposal"]) == 3
        for p in result["sm_proposal"]:
            assert p["agent"] == "SMAgent"
            assert "proposal" in p
            assert "confidence" in p

    def test_gpu_agent_accepts_num_candidates(self):
        """GPUAgent.analyze() should accept num_candidates without crashing."""
        from agent_system_langgraph.agents import GPUAgent

        mock_llm = _mock_llm_with_proposals(self._three_proposals())
        agent = GPUAgent(mock_llm, "cuda:80")
        state = _make_agent_state()

        # Should not raise TypeError
        result = agent.analyze(state, num_candidates=5)
        assert result["gpu_analysis"] is not None

    def test_gpu_agent_stores_list(self):
        """GPUAgent should store all proposals as a list."""
        from agent_system_langgraph.agents import GPUAgent

        # GPUAgent uses block_size_changes instead of changes
        proposals = [
            {"strategy": "conservative", "num_warps": 4,
             "grid_dimensions": {"x": 128, "y": 1, "z": 1},
             "block_size_changes": {"XBLOCK": 256},
             "expected_cta_count": 128, "reasoning": "test",
             "expected_improvement": "5%", "confidence": 0.8,
             "variant_type": "conservative"},
            {"strategy": "aggressive", "num_warps": 16,
             "grid_dimensions": {"x": 64, "y": 1, "z": 1},
             "block_size_changes": {"XBLOCK": 1024},
             "expected_cta_count": 64, "reasoning": "test",
             "expected_improvement": "25%", "confidence": 0.5,
             "variant_type": "aggressive"},
        ]
        mock_llm = _mock_llm_with_proposals(proposals)
        agent = GPUAgent(mock_llm, "cuda:80")
        state = _make_agent_state()

        result = agent.analyze(state, num_candidates=2)

        assert isinstance(result["gpu_proposal"], list)
        assert len(result["gpu_proposal"]) == 2
        for p in result["gpu_proposal"]:
            assert p["agent"] == "GPUAgent"
            assert "proposal" in p
            assert "confidence" in p

    def test_coordinator_handles_list_proposals(self):
        """Coordinator should handle list-based proposals from all agents."""
        from agent_system_langgraph.agents import CoordinatorAgent

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256},
                               "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "test", "confidence": 0.8, "variant_type": "balanced",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "moderate",
                                   "optimization_strategy": "test", "conflicts_resolved": [],
                                   "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "10%", "confidence_score": 0.7,
                                     "risk_level": "medium", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [],
                                             "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.5, "sm_proposal_weight": 0.3,
                                       "gpu_proposal_weight": 0.2, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [],
                                         "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = _make_agent_state()
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 90}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        # All proposals as lists (new format)
        state["warp_proposal"] = [
            {"agent": "WarpAgent", "proposal": {"changes": {"XBLOCK": 256}}, "confidence": 0.8},
            {"agent": "WarpAgent", "proposal": {"changes": {"XBLOCK": 512}}, "confidence": 0.6},
        ]
        state["sm_proposal"] = [
            {"agent": "SMAgent", "proposal": {"num_warps": 8}, "confidence": 0.7},
        ]
        state["gpu_proposal"] = [
            {"agent": "GPUAgent", "proposal": {"grid_dimensions": {"x": 64}}, "confidence": 0.5},
        ]

        result = agent.coordinate(state)

        assert result["optimization_result"] is not None
        # Prompt should contain the list proposals serialized as JSON
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]
        assert "WarpAgent" in user_prompt
        assert "SMAgent" in user_prompt

    def test_coordinator_handles_mixed_formats(self):
        """Coordinator should handle a mix of old (dict) and new (list) proposal formats."""
        from agent_system_langgraph.agents import CoordinatorAgent

        response_json = {
            "candidate_configs": [
                {
                    "config": {"num_warps": 4, "num_stages": 2, "block_size": {"XBLOCK": 128},
                               "grid": {"x": 128, "y": 1, "z": 1}, "eviction_policy": "evict_last"},
                    "rationale": "test", "confidence": 0.9, "variant_type": "conservative",
                },
            ],
            "decision_rationale": {"primary_bottleneck": "warp", "bottleneck_severity": "minor",
                                   "optimization_strategy": "test", "conflicts_resolved": [],
                                   "tradeoffs_accepted": []},
            "expected_improvement": {"speedup_estimate": "5%", "confidence_score": 0.9,
                                     "risk_level": "low", "risk_factors": []},
            "implementation_instructions": {"step_by_step": [], "critical_changes": [],
                                             "validation_steps": []},
            "agent_weight_analysis": {"warp_proposal_weight": 0.3, "sm_proposal_weight": 0.3,
                                       "gpu_proposal_weight": 0.3, "weighting_rationale": "test"},
            "comparison_with_current": {"improvements": [], "regressions": [],
                                         "net_assessment": "test"},
        }

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps(response_json)
        mock_llm.parse_json_response.return_value = response_json
        agent = CoordinatorAgent(mock_llm, "cuda:80")

        state = _make_agent_state()
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 90}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        # Mixed formats: list, old single dict, None
        state["warp_proposal"] = [{"agent": "WarpAgent", "proposal": {}, "confidence": 0.7}]
        state["sm_proposal"] = {"agent": "SMAgent", "proposal": {}, "confidence": 0.6}  # Old format
        state["gpu_proposal"] = None  # Missing

        result = agent.coordinate(state)

        # Should complete without error
        assert result["optimization_result"] is not None
        assert len(result["candidate_configs"]) >= 1
