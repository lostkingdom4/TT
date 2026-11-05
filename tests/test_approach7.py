#!/usr/bin/env python3
"""
Tests for Approach 7: Negotiation-based agent architecture.

Covers TODO items 7.14-7.19:
- 7.14: New agents (DataMovementAgent, ResourceBudgetAgent)
- 7.15: compute_grid() pure function
- 7.16: Negotiation protocol (conflict detection, OrchestratorAgent)
- 7.17: Feedback loop (construct_agent_feedback, feedback mechanics)
- 7.18: Integration with benchmark infrastructure
- 7.19: Backward compatibility
- NCU profiler internal tests
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# Allow importing from the agent directory
AGENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BC = {"kernel_module": "/tmp/k.py", "kernel_name": "k", "shapes_file": "/tmp/s.json"}


def _make_kernel_config(
    block_size: Optional[Dict[str, int]] = None,
    constexpr_param_names: Optional[List[str]] = None,
    kernel_name: str = "test_kernel",
    ttgir: str = "module {}",
    tensor_shapes: Optional[Dict] = None,
    tensor_dtypes: Optional[Dict] = None,
    num_warps: int = 4,
    num_stages: int = 2,
    shared_memory_bytes: int = 0,
):
    from agent_system_langgraph.state import KernelConfig

    return KernelConfig(
        kernel_name=kernel_name,
        ttgir=ttgir,
        num_warps=num_warps,
        num_stages=num_stages,
        block_size=block_size or {"XBLOCK": 128},
        grid={"x": 128},
        constexpr_param_names=constexpr_param_names or [],
        tensor_shapes=tensor_shapes or {"x": [4096, 256], "output": [4096, 256]},
        tensor_dtypes=tensor_dtypes or {"x": "float32", "output": "float32"},
        shared_memory_bytes=shared_memory_bytes,
    )


def _make_initial_state_v7(benchmark_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a minimal OptimizationState dict with ALL Approach 7 fields."""
    kc = _make_kernel_config()
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
        # Approach 7 fields
        "data_movement_proposal": None,
        "resource_budget_proposal": None,
        "computed_grid": None,
        "execution_metrics": None,
        "conflict_detection": None,
        "negotiation_history": None,
        "consensus_reached": False,
        "negotiation_round": 0,
        "max_negotiation_rounds": 2,
        "negotiation_in_progress": False,
        "agent_feedback": {},
        "feedback_iteration": 0,
        "max_feedback_iterations": 2,
        "profiling_data": None,
    }


def _make_benchmark_result(
    orig_median: float = 10.0,
    rec_median: float = 8.0,
    error: Optional[str] = None,
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
    return {
        "original_stats": {"median": orig_median, "mean": orig_median, "min": orig_median, "max": orig_median, "std_dev": 0.0, "samples": 20},
        "recommended_stats": {"median": rec_median, "mean": rec_median, "min": rec_median, "max": rec_median, "std_dev": 0.0, "samples": 20},
        "speedup": speedup,
        "original_config": {"num_warps": 4, "block_size": {"XBLOCK": 128}},
        "recommended_config": {"num_warps": 8, "block_size": {"XBLOCK": 256}},
        "grid": [128],
    }


def _make_candidate(
    num_warps: int = 8,
    block_size: Optional[Dict[str, int]] = None,
    num_stages: int = 2,
    rationale: str = "test",
    confidence: float = 0.8,
    variant_type: str = "balanced",
) -> Dict[str, Any]:
    bs = block_size or {"XBLOCK": 256}
    return {
        "config": {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "block_size": bs,
            "grid": {"x": 128},
        },
        "rationale": rationale,
        "confidence": confidence,
        "variant_type": variant_type,
    }


def _make_optimization_result(
    num_warps: int = 8,
    block_size: Optional[Dict[str, int]] = None,
    confidence: float = 0.9,
) -> Dict[str, Any]:
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


def _make_dm_proposal(
    block_size: Optional[Dict[str, int]] = None,
    num_stages: int = 2,
    num_warps: int = 4,
    shared_memory_bytes: int = 0,
    what_I_need: str = "",
) -> Dict[str, Any]:
    bs = block_size or {"XBLOCK": 256}
    return {
        "preferred": {"block_size": bs, "num_stages": num_stages, "num_warps": num_warps, "shared_memory_bytes": shared_memory_bytes},
        "alternatives": [{"block_size": {"XBLOCK": 512}, "num_stages": 2, "num_warps": num_warps, "shared_memory_bytes": shared_memory_bytes, "tradeoff": "larger tile"}],
        "hard_constraints": ["coalescing alignment"],
        "soft_constraints": ["L2 cache residency"],
        "rationale": "test",
        "what_I_need_from_others": what_I_need,
        "confidence": 0.8,
        "analysis": {"coalescing_pattern": "contiguous", "reuse_factor": 4.0, "working_set_bytes": 1024, "pipeline_assessment": "good", "cache_behavior": "good"},
    }


def _make_rb_proposal(
    num_warps: int = 8,
    shared_memory_bytes: int = 0,
    block_size: Optional[Dict[str, int]] = None,
    num_stages: int = 2,
    what_I_need: str = "",
    register_total: int = 0,
) -> Dict[str, Any]:
    bs = block_size or {"XBLOCK": 256}
    return {
        "preferred": {"num_warps": num_warps, "shared_memory_bytes": shared_memory_bytes, "block_size": bs, "num_stages": num_stages},
        "alternatives": [{"num_warps": 4, "shared_memory_bytes": 0, "block_size": bs, "num_stages": num_stages, "tradeoff": "lower occupancy"}],
        "hard_constraints": ["register file limit"],
        "soft_constraints": ["occupancy > 50%"],
        "rationale": "test",
        "what_I_need_from_others": what_I_need,
        "confidence": 0.7,
        "analysis": {
            "register_estimate": {"data_registers": 30, "compiler_overhead": 25, "total": register_total or 55},
            "occupancy_analysis": {"current_occupancy_pct": 50.0, "limiting_factor": "threads", "blocks_per_sm": 4},
            "warp_scheduling": "good",
            "sram_utilization": "minimal",
        },
    }


def _make_orchestrator(benchmark_fn=None):
    """Create an orchestrator with mocked LLM using __new__ (bypasses __init__)."""
    with patch("agent_system_langgraph.orchestrator.LLMProvider"):
        with patch("agent_system_langgraph.orchestrator.WarpAgent"):
            with patch("agent_system_langgraph.orchestrator.SMAgent"):
                with patch("agent_system_langgraph.orchestrator.GPUAgent"):
                    with patch("agent_system_langgraph.orchestrator.CoordinatorAgent"):
                        from agent_system_langgraph.orchestrator import GPUOptimizationOrchestrator
                        orch = GPUOptimizationOrchestrator.__new__(GPUOptimizationOrchestrator)
                        orch.benchmark_fn = benchmark_fn
                        # Set up mock agents needed by _detect_conflicts_node and _data_movement_node
                        from agent_system_langgraph.agents import OrchestratorAgent
                        mock_orch_agent = OrchestratorAgent.__new__(OrchestratorAgent)
                        mock_orch_agent.agent_name = "OrchestratorAgent"
                        from agent_system_langgraph.knowledge_base import get_architecture
                        mock_orch_agent.architecture = get_architecture("cuda:70")
                        orch.orchestrator_agent = mock_orch_agent
                        orch.data_movement_agent = MagicMock()
                        orch.resource_budget_agent = MagicMock()
                        orch.num_candidates = 3
                        orch.disable_negotiation = False
                        orch.include_agent_proposals = False
                        return orch


def _make_mock_llm(response_json: Dict[str, Any]) -> MagicMock:
    """Create a mock LLM provider that returns a fixed JSON response."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps(response_json)
    mock_llm.parse_json_response.side_effect = lambda r: json.loads(r)
    return mock_llm


# ===========================================================================
# 7.14: Tests for New Agents
# ===========================================================================

class TestDataMovementAgent:
    """Tests for DataMovementAgent output scoping and behavior."""

    def _make_agent(self, response_json: Dict[str, Any], arch: str = "cuda:70"):
        from agent_system_langgraph.agents import DataMovementAgent
        mock_llm = _make_mock_llm(response_json)
        agent = DataMovementAgent(mock_llm, arch)
        return agent, mock_llm

    def test_output_stored_in_data_movement_proposal(self):
        """LLM response should be stored in state['data_movement_proposal']."""
        proposal = _make_dm_proposal(block_size={"XBLOCK": 512}, num_stages=3)
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()

        result = agent.analyze(state)

        assert result["data_movement_proposal"] is not None
        assert result["data_movement_proposal"]["preferred"]["block_size"] == {"XBLOCK": 512}
        assert result["data_movement_proposal"]["preferred"]["num_stages"] == 3
        assert "hard_constraints" in result["data_movement_proposal"]
        assert "soft_constraints" in result["data_movement_proposal"]

    def test_output_includes_all_params(self):
        """DataMovementAgent proposal's preferred config must include all 4 params."""
        proposal = _make_dm_proposal(block_size={"XBLOCK": 256})
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()

        result = agent.analyze(state)

        preferred = result["data_movement_proposal"]["preferred"]
        assert "num_warps" in preferred
        assert "shared_memory_bytes" in preferred
        assert "block_size" in preferred
        assert "num_stages" in preferred

    def test_validates_block_size_names(self):
        """Block size names should be auto-corrected to match constexpr_param_names."""
        # LLM returns XBLOCK but kernel uses BLOCK_SIZE
        proposal = _make_dm_proposal(block_size={"XBLOCK": 256})
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()
        state["kernel_config"] = _make_kernel_config(
            block_size={"BLOCK_SIZE": 128},
            constexpr_param_names=["BLOCK_SIZE"],
        )

        result = agent.analyze(state)

        preferred_bs = result["data_movement_proposal"]["preferred"]["block_size"]
        assert "BLOCK_SIZE" in preferred_bs
        assert "XBLOCK" not in preferred_bs

    def test_feedback_appended_to_prompt(self):
        """When agent_feedback has data_movement_feedback, it should appear in the LLM prompt."""
        proposal = _make_dm_proposal()
        agent, mock_llm = self._make_agent(proposal)
        state = _make_initial_state_v7()
        state["agent_feedback"] = {
            "data_movement_feedback": "BENCHMARK FEEDBACK: your BLOCK_SIZE=4096 regressed"
        }

        agent.analyze(state)

        # The second positional arg to generate() is the user prompt
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]
        assert "BENCHMARK FEEDBACK: your BLOCK_SIZE=4096 regressed" in user_prompt


class TestResourceBudgetAgent:
    """Tests for ResourceBudgetAgent output scoping and behavior."""

    def _make_agent(self, response_json: Dict[str, Any], arch: str = "cuda:70"):
        from agent_system_langgraph.agents import ResourceBudgetAgent
        mock_llm = _make_mock_llm(response_json)
        agent = ResourceBudgetAgent(mock_llm, arch)
        return agent, mock_llm

    def test_output_stored_in_resource_budget_proposal(self):
        """LLM response should be stored in state['resource_budget_proposal']."""
        proposal = _make_rb_proposal(num_warps=8, shared_memory_bytes=4096)
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()

        result = agent.analyze(state)

        assert result["resource_budget_proposal"] is not None
        assert result["resource_budget_proposal"]["preferred"]["num_warps"] == 8
        assert result["resource_budget_proposal"]["preferred"]["shared_memory_bytes"] == 4096

    def test_output_includes_all_params(self):
        """ResourceBudgetAgent proposal's preferred config must include all 4 params."""
        proposal = _make_rb_proposal(num_warps=8)
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()

        result = agent.analyze(state)

        preferred = result["resource_budget_proposal"]["preferred"]
        assert "block_size" in preferred
        assert "num_stages" in preferred
        assert "num_warps" in preferred
        assert "shared_memory_bytes" in preferred

    def test_corrects_non_power_of_2_num_warps(self):
        """LLM returning num_warps=6 should be corrected to 4."""
        proposal = _make_rb_proposal(num_warps=6)
        agent, _ = self._make_agent(proposal)
        state = _make_initial_state_v7()

        result = agent.analyze(state)

        assert result["resource_budget_proposal"]["preferred"]["num_warps"] in [1, 2, 4, 8, 16, 32]
        # 6 is closest to 4 (|6-4|=2) vs 8 (|6-8|=2), tie -> smaller (4)
        assert result["resource_budget_proposal"]["preferred"]["num_warps"] == 4

    def test_feedback_appended_to_prompt(self):
        """When agent_feedback has resource_budget_feedback, it should appear in the LLM prompt."""
        proposal = _make_rb_proposal()
        agent, mock_llm = self._make_agent(proposal)
        state = _make_initial_state_v7()
        state["agent_feedback"] = {
            "resource_budget_feedback": "BENCHMARK FEEDBACK: your num_warps=16 had low occupancy"
        }

        agent.analyze(state)

        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1]
        assert "BENCHMARK FEEDBACK: your num_warps=16 had low occupancy" in user_prompt


# ===========================================================================
# 7.15: Tests for compute_grid()
# ===========================================================================

class TestComputeGrid:
    """Tests for knowledge_base.compute_grid() pure function."""

    def test_elementwise_1d(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 256},
            tensor_shapes={"x": [1024]},
            kernel_type="elementwise",
            num_sms=80,
        )
        assert result["grid"]["x"] == math.ceil(1024 / 256)  # 4
        assert result["grid"]["y"] == 1
        assert result["grid"]["z"] == 1

    def test_elementwise_2d(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 512},
            tensor_shapes={"x": [4096, 512]},
            kernel_type="elementwise",
            num_sms=80,
        )
        total_elements = 4096 * 512  # 2097152
        assert result["grid"]["x"] == math.ceil(total_elements / 512)  # 4096

    def test_reduction_one_cta_per_row(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 4096},
            tensor_shapes={"x": [4096, 256]},
            kernel_type="reduction",
            num_sms=80,
        )
        assert result["grid"]["x"] == 4096  # One CTA per row

    def test_reduction_independent_of_block_size(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        for bs_val in [256, 512, 2048]:
            result = compute_grid(
                block_size={"BLOCK_SIZE": bs_val},
                tensor_shapes={"x": [4096, 256]},
                kernel_type="reduction",
                num_sms=80,
            )
            assert result["grid"]["x"] == 4096

    def test_gemm_grid(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_M": 64, "BLOCK_N": 128},
            tensor_shapes={"A": [512, 256], "B": [256, 512]},
            kernel_type="gemm",
            num_sms=80,
        )
        assert result["grid"]["x"] == math.ceil(512 / 64)  # 8
        assert result["grid"]["y"] == math.ceil(512 / 128)  # 4
        assert result["total_ctas"] == 8 * 4  # 32

    def test_sm_saturation_excellent(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 1},
            tensor_shapes={"x": [320]},
            kernel_type="elementwise",
            num_sms=80,
        )
        # 320 CTAs / 80 SMs = 4.0 CTAs/SM -> "excellent"
        assert result["total_ctas"] == 320
        assert result["sm_saturation"] == "excellent"

    def test_sm_saturation_poor(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 1},
            tensor_shapes={"x": [60]},
            kernel_type="elementwise",
            num_sms=80,
        )
        # 60 CTAs / 80 SMs = 0.75 CTAs/SM -> "poor"
        assert result["total_ctas"] == 60
        assert result["sm_saturation"] == "poor"

    def test_wave_quantization_waste(self):
        from agent_system_langgraph.knowledge_base import compute_grid

        result = compute_grid(
            block_size={"BLOCK_SIZE": 1},
            tensor_shapes={"x": [100]},
            kernel_type="elementwise",
            num_sms=80,
        )
        # 100 CTAs, 80 SMs: full_waves=1, partial=20
        assert result["total_ctas"] == 100
        assert result["full_waves"] == 1
        assert result["partial_wave_ctas"] == 20
        # waste = 1.0 - 20/80 = 0.75
        assert result["wave_quantization_waste"] == pytest.approx(0.75)


# ===========================================================================
# 7.16: Tests for Negotiation Protocol
# ===========================================================================

class TestConflictDetection:
    """Tests for OrchestratorAgent._detect_conflicts() programmatic checks."""

    def _make_agent(self):
        from agent_system_langgraph.agents import OrchestratorAgent
        mock_llm = MagicMock()
        return OrchestratorAgent(mock_llm, "cuda:70")

    def test_no_conflicts_when_no_needs(self):
        """No conflicts when both agents have empty 'what_I_need_from_others' and matching params."""
        agent = self._make_agent()
        dm = _make_dm_proposal(what_I_need="", num_warps=8)
        rb = _make_rb_proposal(what_I_need="", num_warps=8)
        grid_info = {"ctas_per_sm": 4.0}

        conflicts = agent._detect_conflicts(dm, rb, grid_info)

        assert len(conflicts) == 0

    def test_conflict_when_dm_has_needs(self):
        """Conflict detected when DataMovement has needs from ResourceBudget."""
        agent = self._make_agent()
        dm = _make_dm_proposal(what_I_need="need low num_warps for register headroom")
        rb = _make_rb_proposal(what_I_need="")
        grid_info = {"ctas_per_sm": 4.0}

        conflicts = agent._detect_conflicts(dm, rb, grid_info)

        assert len(conflicts) >= 1
        assert any("DataMovement needs" in c["parameter"] for c in conflicts)

    def test_sm_saturation_conflict(self):
        """Conflict detected when combined config has poor SM saturation."""
        agent = self._make_agent()
        dm = _make_dm_proposal(what_I_need="")
        rb = _make_rb_proposal(what_I_need="")
        grid_info = {"ctas_per_sm": 0.5}  # < 1.0 triggers conflict

        conflicts = agent._detect_conflicts(dm, rb, grid_info)

        assert any("SM saturation" in c["parameter"] for c in conflicts)

    def test_register_overflow_conflict(self):
        """Conflict detected when estimated regs/block exceeds SM register file."""
        agent = self._make_agent()
        dm = _make_dm_proposal(what_I_need="")
        # 200 regs/thread * 8 warps * 32 threads/warp = 51200 < 65536 (V100)
        # But 255 * 8 * 32 = 65280 is close, and 256 would overflow
        # Use a value that clearly overflows: total=260, nw=8 -> 260*8*32 = 66560 > 65536
        rb = _make_rb_proposal(num_warps=8, what_I_need="", register_total=260)
        grid_info = {"ctas_per_sm": 4.0}

        conflicts = agent._detect_conflicts(dm, rb, grid_info)

        assert any("register" in c["parameter"].lower() for c in conflicts)


class TestOrchestratorCoordination:
    """Tests for OrchestratorAgent.coordinate() method."""

    def _make_agent(self, response_json: Dict[str, Any]):
        from agent_system_langgraph.agents import OrchestratorAgent
        mock_llm = _make_mock_llm(response_json)
        return OrchestratorAgent(mock_llm, "cuda:70"), mock_llm

    def test_candidates_emitted_from_coordination(self):
        """Orchestrator should populate state['candidate_configs'] and set converged."""
        response = {
            "agreements": {"num_stages": 2},
            "conflicts": [],
            "candidate_configs": [
                {"config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}}, "rationale": "balanced", "confidence": 0.8, "variant_type": "balanced"},
                {"config": {"num_warps": 4, "num_stages": 2, "block_size": {"XBLOCK": 512}}, "rationale": "conservative", "confidence": 0.7, "variant_type": "conservative"},
            ],
            "resolution": "agents agree",
        }
        agent, _ = self._make_agent(response)
        state = _make_initial_state_v7()
        state["data_movement_proposal"] = _make_dm_proposal()
        state["resource_budget_proposal"] = _make_rb_proposal()

        result = agent.coordinate(state)

        assert len(result["candidate_configs"]) == 2
        assert result["converged"] is True
        assert result["optimization_result"] is not None

    def test_block_size_names_validated_in_candidates(self):
        """Candidates with wrong block names should be auto-corrected."""
        response = {
            "agreements": {},
            "conflicts": [],
            "candidate_configs": [
                {"config": {"num_warps": 8, "num_stages": 2, "block_size": {"XBLOCK": 256}}, "rationale": "test", "confidence": 0.8, "variant_type": "balanced"},
            ],
            "resolution": "ok",
        }
        agent, _ = self._make_agent(response)
        state = _make_initial_state_v7()
        state["kernel_config"] = _make_kernel_config(
            block_size={"BLOCK_SIZE": 128},
            constexpr_param_names=["BLOCK_SIZE"],
        )
        state["data_movement_proposal"] = _make_dm_proposal()
        state["resource_budget_proposal"] = _make_rb_proposal()

        result = agent.coordinate(state)

        cand_bs = result["candidate_configs"][0]["config"]["block_size"]
        assert "BLOCK_SIZE" in cand_bs
        assert "XBLOCK" not in cand_bs


# ===========================================================================
# 7.17: Tests for Feedback Loop
# ===========================================================================

class TestConstructAgentFeedback:
    """Tests for construct_agent_feedback() pure function."""

    def test_empty_results_returns_no_data_message(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        result = construct_agent_feedback(
            benchmark_results_per_candidate=[],
            candidate_configs=[],
            data_movement_proposal={},
            resource_budget_proposal={},
        )
        assert "No benchmark data" in result["data_movement_feedback"]
        assert "No benchmark data" in result["resource_budget_feedback"]

    def test_dm_feedback_attributes_block_size(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal(block_size={"XBLOCK": 256})
        rb = _make_rb_proposal(num_warps=8)
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
            {"candidate_index": 1, "config": {"block_size": {"XBLOCK": 512}, "num_warps": 4, "num_stages": 2}, "speedup": 0.9},
        ]

        feedback = construct_agent_feedback(results, [], dm, rb)

        # Both candidates with full configs appear in DM feedback
        assert "block_size=" in feedback["data_movement_feedback"]
        assert "num_warps=" in feedback["data_movement_feedback"]
        assert "Candidate 0" in feedback["data_movement_feedback"]
        assert "Candidate 1" in feedback["data_movement_feedback"]

    def test_rb_feedback_attributes_num_warps(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal(block_size={"XBLOCK": 256})
        rb = _make_rb_proposal(num_warps=8)
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
            {"candidate_index": 1, "config": {"block_size": {"XBLOCK": 512}, "num_warps": 4, "num_stages": 2}, "speedup": 0.9},
        ]

        feedback = construct_agent_feedback(results, [], dm, rb)

        # Both candidates with full configs appear in RB feedback
        assert "num_warps=" in feedback["resource_budget_feedback"]
        assert "block_size=" in feedback["resource_budget_feedback"]
        assert "Candidate 0" in feedback["resource_budget_feedback"]
        assert "Candidate 1" in feedback["resource_budget_feedback"]

    def test_dm_feedback_includes_full_config(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal()
        rb = _make_rb_proposal()
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
        ]

        feedback = construct_agent_feedback(results, [], dm, rb)

        # DM feedback should include ALL parameters (full config visibility)
        assert "block_size" in feedback["data_movement_feedback"]
        assert "num_warps" in feedback["data_movement_feedback"]
        assert "num_stages" in feedback["data_movement_feedback"]

    def test_rb_feedback_includes_full_config(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal()
        rb = _make_rb_proposal()
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
        ]

        feedback = construct_agent_feedback(results, [], dm, rb)

        # RB feedback should include ALL parameters (full config visibility)
        assert "num_warps" in feedback["resource_budget_feedback"]
        assert "block_size" in feedback["resource_budget_feedback"]
        assert "num_stages" in feedback["resource_budget_feedback"]

    def test_profiling_data_shared_with_both_agents(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal()
        rb = _make_rb_proposal()
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
        ]
        profiling = {
            "global_memory_throughput_pct": 72.5,
            "l2_hit_rate": 15.0,
            "l1_hit_rate": 92.0,
            "achieved_occupancy_pct": 45.0,
            "registers_per_thread": 73,
            "warp_stall_reasons": {"memory": "60.0%", "barrier": "25.0%"},
        }

        feedback = construct_agent_feedback(results, [], dm, rb, profiling_data=profiling)

        # BOTH agents get ALL metrics (no more scoping)
        for key in ("data_movement_feedback", "resource_budget_feedback"):
            fb = feedback[key]
            assert "72.5%" in fb  # global memory throughput
            assert "L2 cache hit rate: 15.0%" in fb
            assert "L1 cache hit rate: 92.0%" in fb
            assert "45.0%" in fb  # achieved occupancy
            assert "73" in fb  # registers per thread
            assert "stall" in fb.lower()

    def test_revision_instructions_appended(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        dm = _make_dm_proposal()
        rb = _make_rb_proposal()
        results = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 1.1},
        ]

        feedback = construct_agent_feedback(results, [], dm, rb)

        assert "INSTRUCTION: Revise" in feedback["data_movement_feedback"]
        assert "block_size" in feedback["data_movement_feedback"]
        assert "INSTRUCTION: Revise" in feedback["resource_budget_feedback"]
        assert "num_warps" in feedback["resource_budget_feedback"]


class TestFeedbackLoopMechanics:
    """Tests for feedback loop node mechanics in the orchestrator."""

    def test_construct_feedback_node_fires_on_regression(self):
        """_construct_feedback_node should populate agent_feedback when regression_detected."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["regression_detected"] = True
        state["benchmark_results_per_candidate"] = [
            {"candidate_index": 0, "config": {"block_size": {"XBLOCK": 256}, "num_warps": 8, "num_stages": 2}, "speedup": 0.9},
        ]
        state["data_movement_proposal"] = _make_dm_proposal()
        state["resource_budget_proposal"] = _make_rb_proposal()

        result = orch._construct_feedback_node(state)

        assert result["agent_feedback"] is not None
        assert "data_movement_feedback" in result["agent_feedback"]
        assert "resource_budget_feedback" in result["agent_feedback"]

    def test_construct_feedback_node_noop_without_regression(self):
        """_construct_feedback_node should be a no-op when regression_detected is False."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["regression_detected"] = False

        result = orch._construct_feedback_node(state)

        # agent_feedback should remain as the initial empty dict (unchanged)
        assert result["agent_feedback"] == {}

    def test_convergence_increments_feedback_iteration_on_regression(self):
        """_check_convergence_node should increment feedback_iteration on regression."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["optimization_result"] = _make_optimization_result()
        state["regression_detected"] = True
        state["benchmark_iterations"] = 1
        state["feedback_iteration"] = 0
        state["benchmark_results"] = _make_benchmark_result(orig_median=10.0, rec_median=14.0)

        result = orch._check_convergence_node(state)

        assert result["converged"] is False
        assert result["feedback_iteration"] == 1
        assert result["regression_detected"] is False  # Reset for next round


# ===========================================================================
# 7.18: Integration with Benchmark Infrastructure
# ===========================================================================

class TestBenchmarkIntegration:
    """Tests for integration between Approach 7 and benchmark infrastructure."""

    def test_candidates_benchmarked_with_sanitization(self):
        """Candidate with num_warps=6 should be sanitized to 4 before benchmarking."""
        mock_fn = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_fn)
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate(num_warps=6)]

        orch._benchmark_candidates_node(state)

        # Verify benchmark_fn received sanitized config
        call_kwargs = mock_fn.call_args
        rec_config = call_kwargs[1]["recommended_config"] if "recommended_config" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        # The positional args are (original_config=, recommended_config=, ...)
        # but it may also be kwargs
        # Let's check the actual call
        assert mock_fn.called
        actual_call = mock_fn.call_args
        # recommended_config is the second positional arg or a kwarg
        if actual_call.kwargs and "recommended_config" in actual_call.kwargs:
            rec = actual_call.kwargs["recommended_config"]
        else:
            rec = actual_call.args[1] if len(actual_call.args) > 1 else None
        # Actually the benchmark_fn is called with keyword args
        # Let's just check that per-candidate results have sanitized config
        per_cand = state["benchmark_results_per_candidate"]
        assert len(per_cand) == 1
        assert per_cand[0]["config"]["num_warps"] == 4  # 6 -> 4

    def test_profiler_fn_called_on_improvement(self):
        """When profiler_fn is set and benchmark shows improvement, profiler should be called."""
        mock_bench = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        mock_profiler = MagicMock(return_value={"achieved_occupancy_pct": 75.0})
        orch = _make_orchestrator(benchmark_fn=mock_bench)
        orch.profiler_fn = mock_profiler
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]

        orch._benchmark_candidates_node(state)

        assert mock_profiler.called
        assert state["profiling_data"] == {"achieved_occupancy_pct": 75.0}

    def test_profiler_fn_not_called_without_flag(self):
        """When profiler_fn is None (default), no profiling should happen."""
        mock_bench = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        orch = _make_orchestrator(benchmark_fn=mock_bench)
        # profiler_fn not set (default behavior via __new__)
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]

        result = orch._benchmark_candidates_node(state)

        assert result["profiling_data"] is None

    def test_profiler_failure_is_nonfatal(self):
        """Profiler exception should not crash the benchmark node."""
        mock_bench = MagicMock(return_value=_make_benchmark_result(orig_median=10.0, rec_median=8.0))
        mock_profiler = MagicMock(side_effect=RuntimeError("NCU crash"))
        orch = _make_orchestrator(benchmark_fn=mock_bench)
        orch.profiler_fn = mock_profiler
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["candidate_configs"] = [_make_candidate()]

        result = orch._benchmark_candidates_node(state)

        # Should complete without raising
        assert result["regression_detected"] is False
        assert result["profiling_data"] is None

    def test_approach7_state_fields_present(self):
        """Verify OptimizationState TypedDict has all Approach 7 fields."""
        from agent_system_langgraph.state import OptimizationState

        annotations = OptimizationState.__annotations__
        expected_fields = [
            "data_movement_proposal",
            "resource_budget_proposal",
            "computed_grid",
            "execution_metrics",
            "conflict_detection",
            "negotiation_history",
            "consensus_reached",
            "agent_feedback",
            "feedback_iteration",
            "max_feedback_iterations",
            "profiling_data",
        ]
        for field in expected_fields:
            assert field in annotations, f"Missing field: {field}"

    def test_feedback_fallback_after_max_retries(self):
        """After max retries + max feedback iterations, should fallback to keep_original."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["optimization_result"] = _make_optimization_result()
        state["regression_detected"] = True
        state["benchmark_iterations"] = 3  # MAX_BENCHMARK_RETRIES
        state["feedback_iteration"] = 2  # max_feedback_iterations
        state["benchmark_results"] = _make_benchmark_result(orig_median=10.0, rec_median=14.0)

        result = orch._check_convergence_node(state)

        assert result["converged"] is True
        assert result["optimization_result"]["recommendation"] == "keep_original"
        assert result["optimization_result"]["confidence_score"] == 1.0


# ===========================================================================
# 7.19: Backward Compatibility
# ===========================================================================

class TestBackwardCompat:
    """Tests verifying backward compatibility with existing infrastructure."""

    def test_orchestrator_accepts_old_init_args(self):
        """GPUOptimizationOrchestrator should work without profiler_fn (Phase E backward compat)."""
        with patch("agent_system_langgraph.orchestrator.LLMProvider") as MockLLM:
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance
            from agent_system_langgraph.orchestrator import GPUOptimizationOrchestrator
            # Should not raise — profiler_fn defaults to None
            orch = GPUOptimizationOrchestrator(
                llm_provider="openai",
                llm_model="gpt-4",
                architecture_code="cuda:70",
                benchmark_fn=None,
            )
            assert orch.profiler_fn is None

    def test_state_has_all_approach7_fields(self):
        """Full initial state dict should include all Approach 7 fields."""
        state = _make_initial_state_v7()
        assert "data_movement_proposal" in state
        assert "resource_budget_proposal" in state
        assert "computed_grid" in state
        assert "execution_metrics" in state
        assert "agent_feedback" in state
        assert "feedback_iteration" in state
        assert "max_feedback_iterations" in state
        assert "profiling_data" in state

    def test_v7_state_compatible_with_benchmark_node(self):
        """State built with v7 helper should pass through benchmark node without crash."""
        orch = _make_orchestrator(benchmark_fn=None)
        state = _make_initial_state_v7(benchmark_config=_BC)
        state["optimization_result"] = _make_optimization_result()
        state["converged"] = True

        result = orch._benchmark_candidates_node(state)

        # Should be a no-op pass-through (benchmark_fn=None)
        assert result["converged"] is True

    def test_v7_state_compatible_with_convergence_node(self):
        """State built with v7 helper should pass through convergence node."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["optimization_result"] = _make_optimization_result(confidence=0.95)

        result = orch._check_convergence_node(state)

        assert result["converged"] is True  # High confidence -> converge


# ===========================================================================
# NCU Profiler Tests
# ===========================================================================

class TestNCUProfiler:
    """Tests for ncu_profiler.py internal functions."""

    def test_parse_ncu_csv_extracts_metrics(self):
        from ncu_profiler import _parse_ncu_csv

        csv_text = (
            '"Metric Name","Metric Value"\n'
            '"sm__warps_active.avg.pct_of_peak_sustained_active","75.5"\n'
            '"launch__registers_per_thread","64"\n'
            '"dram__bytes.sum.per_second","450000000000"\n'
        )
        result = _parse_ncu_csv(csv_text)

        assert result["achieved_occupancy_pct"] == pytest.approx(75.5)
        assert result["registers_per_thread"] == pytest.approx(64.0)
        assert result["_dram_bytes_per_sec"] == pytest.approx(450e9)

    def test_parse_ncu_csv_handles_comments(self):
        from ncu_profiler import _parse_ncu_csv

        csv_text = (
            "==PROF== Profiling result:\n"
            "==PROF== Disconnected\n"
            '"Metric Name","Metric Value"\n'
            '"sm__warps_active.avg.pct_of_peak_sustained_active","80.0"\n'
        )
        result = _parse_ncu_csv(csv_text)

        assert result["achieved_occupancy_pct"] == pytest.approx(80.0)

    def test_parse_ncu_csv_empty_input(self):
        from ncu_profiler import _parse_ncu_csv

        result = _parse_ncu_csv("")
        assert result == {}

    def test_compute_derived_metrics_throughput(self):
        from ncu_profiler import _compute_derived_metrics

        raw = {"_dram_bytes_per_sec": 450e9}
        result = _compute_derived_metrics(raw, peak_bw_gb_s=900.0)

        assert result["global_memory_throughput_pct"] == pytest.approx(50.0)

    def test_compute_derived_metrics_cache_rates(self):
        from ncu_profiler import _compute_derived_metrics

        raw = {"_l2_hit": 75.0, "_l2_miss": 25.0, "_l1_hit": 90.0, "_l1_miss": 10.0}
        result = _compute_derived_metrics(raw, peak_bw_gb_s=900.0)

        assert result["l2_hit_rate"] == pytest.approx(75.0)
        assert result["l1_hit_rate"] == pytest.approx(90.0)

    def test_compute_derived_metrics_stalls(self):
        from ncu_profiler import _compute_derived_metrics

        raw = {
            "_stall_memory": 60.0,
            "_stall_short_scoreboard": 10.0,
            "_stall_barrier": 25.0,
            "_stall_not_selected": 5.0,
        }
        result = _compute_derived_metrics(raw, peak_bw_gb_s=900.0)

        assert "warp_stall_reasons" in result
        stalls = result["warp_stall_reasons"]
        assert stalls["memory"] == "60.0%"
        assert stalls["barrier"] == "25.0%"
        assert stalls["short_scoreboard"] == "10.0%"
        assert stalls["not_selected"] == "5.0%"


# ---------------------------------------------------------------------------
# Negotiation Loop Tests
# ---------------------------------------------------------------------------


class TestNegotiationLoop:
    """Tests for the detect_conflicts node and negotiation sub-loop."""

    def test_high_severity_conflict_triggers_negotiation(self):
        """When block_size disagrees, _detect_conflicts_node sets negotiation_in_progress."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["data_movement_proposal"] = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=4)
        state["resource_budget_proposal"] = _make_rb_proposal(block_size={"XBLOCK": 128}, num_warps=8)
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}

        result = orch._detect_conflicts_node(state)

        assert result["negotiation_in_progress"] is True
        assert result["negotiation_round"] == 1
        assert result["consensus_reached"] is False

    def test_should_negotiate_returns_negotiate(self):
        """_should_negotiate returns 'negotiate' when negotiation_in_progress."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["negotiation_in_progress"] = True

        assert orch._should_negotiate(state) == "negotiate"

    def test_should_negotiate_returns_proceed(self):
        """_should_negotiate returns 'proceed' when not in negotiation."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["negotiation_in_progress"] = False

        assert orch._should_negotiate(state) == "proceed"

    def test_no_conflict_proceeds(self):
        """When both agents agree on all params, negotiation is not triggered."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        # Same block_size and num_warps
        state["data_movement_proposal"] = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=8)
        state["resource_budget_proposal"] = _make_rb_proposal(block_size={"XBLOCK": 256}, num_warps=8)
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}

        result = orch._detect_conflicts_node(state)

        assert result["negotiation_in_progress"] is False
        assert result["consensus_reached"] is True

    def test_max_rounds_exhaustion_proceeds(self):
        """After max_negotiation_rounds, proceed even with conflicts."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["data_movement_proposal"] = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=4)
        state["resource_budget_proposal"] = _make_rb_proposal(block_size={"XBLOCK": 128}, num_warps=8)
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}
        state["negotiation_round"] = 2  # Already at max
        state["max_negotiation_rounds"] = 2

        result = orch._detect_conflicts_node(state)

        assert result["negotiation_in_progress"] is False  # Proceeds despite conflicts

    def test_negotiation_round_increments(self):
        """Each negotiation pass increments negotiation_round."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["data_movement_proposal"] = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=4)
        state["resource_budget_proposal"] = _make_rb_proposal(block_size={"XBLOCK": 128}, num_warps=8)
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}
        state["negotiation_round"] = 0

        result = orch._detect_conflicts_node(state)
        assert result["negotiation_round"] == 1

    def test_negotiation_history_populated(self):
        """Negotiation history grows each round."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        state["data_movement_proposal"] = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=4)
        state["resource_budget_proposal"] = _make_rb_proposal(block_size={"XBLOCK": 128}, num_warps=8)
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}

        result = orch._detect_conflicts_node(state)

        history = result["negotiation_history"]
        assert history is not None
        assert len(history) == 1
        assert history[0]["round"] == 1
        assert "dm_preferred" in history[0]
        assert "rb_preferred" in history[0]
        assert "conflicts" in history[0]

    def test_fresh_iteration_resets_negotiation(self):
        """When entering a fresh iteration (not negotiation), reset negotiation state."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        # Simulate leftover negotiation state from previous iteration
        state["negotiation_round"] = 2
        state["negotiation_history"] = [{"round": 1}, {"round": 2}]
        state["negotiation_in_progress"] = False  # Fresh iteration
        state["consensus_reached"] = True

        # Mock data_movement_agent.analyze to return the state as-is
        orch.data_movement_agent.analyze.side_effect = lambda s, **kw: s

        result = orch._data_movement_node(state)

        assert result["negotiation_round"] == 0
        assert result["negotiation_history"] is None
        assert result["negotiation_in_progress"] is False
        assert result["consensus_reached"] is False

    def test_consensus_reached_set_true(self):
        """When no high-severity conflicts, consensus_reached is True."""
        orch = _make_orchestrator()
        state = _make_initial_state_v7()
        # Only medium-severity conflict (num_stages differ)
        state["data_movement_proposal"] = _make_dm_proposal(
            block_size={"XBLOCK": 256}, num_warps=8, num_stages=3,
        )
        state["resource_budget_proposal"] = _make_rb_proposal(
            block_size={"XBLOCK": 256}, num_warps=8, num_stages=2,
        )
        state["computed_grid"] = {"grid_x": 128, "total_ctas": 128, "ctas_per_sm": 1.6}

        result = orch._detect_conflicts_node(state)

        assert result["consensus_reached"] is True
        assert result["negotiation_in_progress"] is False


class TestUpdatedConflictDetection:
    """Tests for the enhanced _detect_conflicts with parameter comparison."""

    def test_block_size_disagreement_detected(self):
        from agent_system_langgraph.agents import OrchestratorAgent as OrchestratorAgentV2

        agent = OrchestratorAgentV2.__new__(OrchestratorAgentV2)
        agent.agent_name = "OrchestratorAgent"
        from agent_system_langgraph.knowledge_base import get_architecture
        agent.architecture = get_architecture("cuda:70")

        dm = _make_dm_proposal(block_size={"XBLOCK": 256})
        rb = _make_rb_proposal(block_size={"XBLOCK": 128})

        conflicts = agent._detect_conflicts(dm, rb, {})

        param_names = [c["parameter"] for c in conflicts]
        assert "block_size" in param_names
        bs_conflict = next(c for c in conflicts if c["parameter"] == "block_size")
        assert bs_conflict["severity"] == "high"

    def test_num_warps_disagreement_detected(self):
        from agent_system_langgraph.agents import OrchestratorAgent as OrchestratorAgentV2

        agent = OrchestratorAgentV2.__new__(OrchestratorAgentV2)
        agent.agent_name = "OrchestratorAgent"
        from agent_system_langgraph.knowledge_base import get_architecture
        agent.architecture = get_architecture("cuda:70")

        dm = _make_dm_proposal(num_warps=4)
        rb = _make_rb_proposal(num_warps=8)

        conflicts = agent._detect_conflicts(dm, rb, {})

        param_names = [c["parameter"] for c in conflicts]
        assert "num_warps" in param_names
        nw_conflict = next(c for c in conflicts if c["parameter"] == "num_warps")
        assert nw_conflict["severity"] == "high"

    def test_full_agreement_no_high_severity(self):
        from agent_system_langgraph.agents import OrchestratorAgent as OrchestratorAgentV2

        agent = OrchestratorAgentV2.__new__(OrchestratorAgentV2)
        agent.agent_name = "OrchestratorAgent"
        from agent_system_langgraph.knowledge_base import get_architecture
        agent.architecture = get_architecture("cuda:70")

        dm = _make_dm_proposal(block_size={"XBLOCK": 256}, num_warps=8, num_stages=2)
        rb = _make_rb_proposal(block_size={"XBLOCK": 256}, num_warps=8, num_stages=2)

        conflicts = agent._detect_conflicts(dm, rb, {})

        high = [c for c in conflicts if c.get("severity") == "high"]
        assert len(high) == 0
