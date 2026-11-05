#!/usr/bin/env python3
"""
Tests for Approach 4: Config Parameter Name Validation.

Covers:
- _format_block_size_instructions / _format_block_size_schema helpers
- _validate_block_size_names auto-correction
- _get_valid_block_names resolution order
- BLOCK_NAME_ALIASES alias matching in compare_configs._map_block_sizes
- Integration: agent prompts use actual param names from KernelConfig
- Integration: CoordinatorAgent validates block names in LLM output
"""

import json
import logging
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

def _make_kernel_config(
    block_size: Dict[str, int] = None,
    constexpr_param_names: list = None,
    kernel_name: str = "test_kernel",
):
    """Build a minimal KernelConfig for testing."""
    from agent_system_langgraph.state import KernelConfig

    return KernelConfig(
        kernel_name=kernel_name,
        ttgir="module {}",
        num_warps=4,
        num_stages=2,
        block_size=block_size or {},
        grid={"x": 128},
        constexpr_param_names=constexpr_param_names or [],
    )


# ===========================================================================
# Tests for _get_valid_block_names
# ===========================================================================

class TestGetValidBlockNames:
    def test_prefers_constexpr_param_names(self):
        from agent_system_langgraph.agents import _get_valid_block_names

        config = _make_kernel_config(
            block_size={"XBLOCK": 128},
            constexpr_param_names=["BLOCK_SIZE"],
        )
        assert _get_valid_block_names(config) == ["BLOCK_SIZE"]

    def test_falls_back_to_block_size_keys(self):
        from agent_system_langgraph.agents import _get_valid_block_names

        config = _make_kernel_config(
            block_size={"BLOCK_M": 64, "BLOCK_N": 32},
            constexpr_param_names=[],
        )
        assert _get_valid_block_names(config) == ["BLOCK_M", "BLOCK_N"]

    def test_ultimate_fallback_to_xblock(self):
        from agent_system_langgraph.agents import _get_valid_block_names

        config = _make_kernel_config(block_size={}, constexpr_param_names=[])
        assert _get_valid_block_names(config) == ["XBLOCK"]


# ===========================================================================
# Tests for _format_block_size_schema
# ===========================================================================

class TestFormatBlockSizeSchema:
    def test_single_param(self):
        from agent_system_langgraph.agents import _format_block_size_schema

        config = _make_kernel_config(constexpr_param_names=["BLOCK_SIZE"])
        schema = _format_block_size_schema(config)
        assert schema == '{"BLOCK_SIZE": <int>}'

    def test_two_params(self):
        from agent_system_langgraph.agents import _format_block_size_schema

        config = _make_kernel_config(constexpr_param_names=["XBLOCK", "YBLOCK"])
        schema = _format_block_size_schema(config)
        assert schema == '{"XBLOCK": <int>, "YBLOCK": <int>}'


# ===========================================================================
# Tests for _format_block_size_instructions
# ===========================================================================

class TestFormatBlockSizeInstructions:
    def test_block_size_kernel(self):
        from agent_system_langgraph.agents import _format_block_size_instructions

        config = _make_kernel_config(constexpr_param_names=["BLOCK_SIZE"])
        text = _format_block_size_instructions(config)
        assert "BLOCK_SIZE" in text
        assert "XBLOCK" not in text
        assert "Use ONLY these exact parameter names" in text

    def test_xblock_yblock_kernel(self):
        from agent_system_langgraph.agents import _format_block_size_instructions

        config = _make_kernel_config(constexpr_param_names=["XBLOCK", "YBLOCK"])
        text = _format_block_size_instructions(config)
        assert "XBLOCK" in text
        assert "YBLOCK" in text

    def test_empty_falls_back(self):
        from agent_system_langgraph.agents import _format_block_size_instructions

        config = _make_kernel_config(block_size={}, constexpr_param_names=[])
        text = _format_block_size_instructions(config)
        assert "XBLOCK" in text  # fallback


# ===========================================================================
# Tests for _validate_block_size_names
# ===========================================================================

class TestValidateBlockSizeNames:
    def test_direct_match_pass_through(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"BLOCK_SIZE": 256},
            ["BLOCK_SIZE"],
        )
        assert result == {"BLOCK_SIZE": 256}

    def test_remap_xblock_to_block_size(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"XBLOCK": 256},
            ["BLOCK_SIZE"],
        )
        assert result == {"BLOCK_SIZE": 256}

    def test_multi_param_pass_through(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"XBLOCK": 128, "YBLOCK": 64},
            ["XBLOCK", "YBLOCK"],
        )
        assert result == {"XBLOCK": 128, "YBLOCK": 64}

    def test_remap_multi_params(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"XBLOCK": 128, "YBLOCK": 64},
            ["BLOCK_M", "BLOCK_N"],
        )
        assert result == {"BLOCK_M": 128, "BLOCK_N": 64}

    def test_drops_extra_keys(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"XBLOCK": 128, "ZBLOCK": 64},
            ["BLOCK_SIZE"],
        )
        # Only XBLOCK maps (position 0 -> BLOCK_SIZE); ZBLOCK dropped
        assert result == {"BLOCK_SIZE": 128}

    def test_empty_proposed(self):
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names({}, ["BLOCK_SIZE"])
        assert result == {}

    def test_partial_direct_match(self):
        """One key matches directly, another needs remapping."""
        from agent_system_langgraph.agents import _validate_block_size_names

        result = _validate_block_size_names(
            {"XBLOCK": 128, "YBLOCK": 64},
            ["XBLOCK", "BLOCK_N"],
        )
        # XBLOCK matches directly; YBLOCK -> BLOCK_N by position
        assert result["XBLOCK"] == 128
        assert result["BLOCK_N"] == 64


# ===========================================================================
# Tests for compare_configs._map_block_sizes alias matching
# ===========================================================================

class TestMapBlockSizesAliases:
    def test_direct_match(self):
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {"BLOCK_SIZE": 256},
            {"BLOCK_SIZE": {"type": "scalar", "value": 256}},
        )
        assert result == {"BLOCK_SIZE": 256}

    def test_alias_xblock_to_block_size(self):
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {"XBLOCK": 512},
            {"BLOCK_SIZE": {"type": "scalar", "value": 256}},
        )
        assert result == {"BLOCK_SIZE": 512}

    def test_alias_multi_params(self):
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {"XBLOCK": 128, "YBLOCK": 64},
            {
                "BLOCK_M": {"type": "scalar", "value": 64},
                "BLOCK_N": {"type": "scalar", "value": 32},
            },
        )
        assert result == {"BLOCK_M": 128, "BLOCK_N": 64}

    def test_no_block_params_in_shapes(self):
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {"XBLOCK": 128},
            {"warmup": {"type": "scalar", "value": False}},
        )
        # Falls through to pass-through
        assert result == {"XBLOCK": 128}

    def test_empty_recommended(self):
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {},
            {"BLOCK_SIZE": {"type": "scalar", "value": 256}},
        )
        assert result == {}

    def test_positional_fallback_for_unknown_names(self):
        """Keys not in the alias table fall back to positional mapping."""
        from compare_configs import _map_block_sizes

        result = _map_block_sizes(
            {"MY_CUSTOM_BLOCK": 512},
            {"BLOCK_SIZE": {"type": "scalar", "value": 256}},
        )
        assert result == {"BLOCK_SIZE": 512}


# ===========================================================================
# Integration: agent prompts use actual param names
# ===========================================================================

class TestAgentPromptIntegration:
    """Verify that agent prompts dynamically include correct parameter names."""

    def _make_state(self, block_size, constexpr_names):
        """Build a minimal OptimizationState dict for testing."""
        from agent_system_langgraph.state import KernelConfig

        kc = KernelConfig(
            kernel_name="softmax_kernel",
            ttgir="module {}",
            num_warps=4,
            num_stages=2,
            block_size=block_size,
            grid={"x": 4096},
            tensor_shapes={"arg0": [4096, 256], "arg1": [4096, 256]},
            tensor_dtypes={"arg0": "torch.float16", "arg1": "torch.float16"},
            constexpr_param_names=constexpr_names,
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

    def test_warp_agent_prompt_uses_block_size(self):
        """WarpAgent prompt should mention BLOCK_SIZE, not XBLOCK, for softmax."""
        from agent_system_langgraph.agents import WarpAgent

        mock_llm = MagicMock()
        # Make LLM return a valid JSON response
        mock_llm.generate.return_value = json.dumps({
            "coalescing_efficiency": {"estimated_percentage": 85},
            "bank_conflicts": {"severity": "none"},
            "warp_divergence": {"present": False},
            "cache_utilization": {"l1_hit_rate_estimate": 0.8},
            "optimization_proposals": [],
            "key_insights": [],
        })
        mock_llm.parse_json_response.side_effect = lambda r: json.loads(r)

        agent = WarpAgent(mock_llm, "cuda:70")
        state = self._make_state({"BLOCK_SIZE": 256}, ["BLOCK_SIZE"])
        agent.analyze(state)

        # Check the user prompt passed to the LLM
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
        assert "BLOCK_SIZE" in user_prompt
        # The hardcoded "XBLOCK should divide" text should NOT appear
        assert "XBLOCK should divide" not in user_prompt

    def test_coordinator_prompt_uses_block_size(self):
        """CoordinatorAgent prompt should include BLOCK_SIZE in output schema."""
        from agent_system_langgraph.agents import CoordinatorAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "recommended_config": {
                "num_warps": 8,
                "num_stages": 2,
                "block_size": {"BLOCK_SIZE": 512},
                "grid": {"x": 4096, "y": 1, "z": 1},
                "eviction_policy": "evict_last",
            },
            "decision_rationale": {},
            "expected_improvement": {"speedup_estimate": "20%", "confidence_score": 0.8, "risk_level": "low"},
            "implementation_instructions": {"step_by_step": []},
        })
        mock_llm.parse_json_response.side_effect = lambda r: json.loads(r)

        agent = CoordinatorAgent(mock_llm, "cuda:70")
        state = self._make_state({"BLOCK_SIZE": 256}, ["BLOCK_SIZE"])
        # Need analyses for coordinator
        state["warp_analysis"] = {"coalescing_efficiency": {"estimated_percentage": 85}}
        state["sm_analysis"] = {"occupancy_analysis": {"current_occupancy_pct": 50}}
        state["gpu_analysis"] = {"grid_analysis": {"sm_saturation_pct": 80}}
        state["warp_proposal"] = [{"agent": "WarpAgent", "proposal": {}, "confidence": 0.5}]
        state["sm_proposal"] = [{"agent": "SMAgent", "proposal": {}, "confidence": 0.5}]
        state["gpu_proposal"] = [{"agent": "GPUAgent", "proposal": {}, "confidence": 0.5}]

        agent.coordinate(state)

        # Check the prompt
        call_args = mock_llm.generate.call_args
        user_prompt = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("user_prompt", "")
        assert '"BLOCK_SIZE": <int>' in user_prompt
        # Should NOT have the old generic schema
        assert '"XBLOCK": <int>, "YBLOCK": <int>' not in user_prompt


# ===========================================================================
# Integration: CoordinatorAgent validates block names in LLM output
# ===========================================================================

class TestCoordinatorValidation:
    """Verify that CoordinatorAgent auto-corrects wrong block names in LLM output."""

    def test_corrects_xblock_to_block_size_in_result(self):
        """Even if the LLM ignores instructions and uses XBLOCK, the result
        should be corrected to BLOCK_SIZE."""
        from agent_system_langgraph.agents import CoordinatorAgent
        from agent_system_langgraph.state import KernelConfig

        mock_llm = MagicMock()
        # LLM returns XBLOCK instead of BLOCK_SIZE
        mock_llm.generate.return_value = json.dumps({
            "recommended_config": {
                "num_warps": 8,
                "num_stages": 2,
                "block_size": {"XBLOCK": 512},
                "grid": {"x": 4096, "y": 1, "z": 1},
                "eviction_policy": "evict_last",
            },
            "decision_rationale": {},
            "expected_improvement": {"speedup_estimate": "20%", "confidence_score": 0.8, "risk_level": "low"},
            "implementation_instructions": {"step_by_step": []},
        })
        mock_llm.parse_json_response.side_effect = lambda r: json.loads(r)

        kc = KernelConfig(
            kernel_name="softmax_kernel",
            ttgir="module {}",
            num_warps=4,
            block_size={"BLOCK_SIZE": 256},
            constexpr_param_names=["BLOCK_SIZE"],
        )

        state = {
            "kernel_config": kc,
            "messages": [],
            "warp_analysis": {"coalescing_efficiency": {"estimated_percentage": 85}},
            "sm_analysis": {"occupancy_analysis": {"current_occupancy_pct": 50}},
            "gpu_analysis": {"grid_analysis": {"sm_saturation_pct": 80}},
            "warp_proposal": [{"agent": "WarpAgent", "proposal": {}, "confidence": 0.5}],
            "sm_proposal": [{"agent": "SMAgent", "proposal": {}, "confidence": 0.5}],
            "gpu_proposal": [{"agent": "GPUAgent", "proposal": {}, "confidence": 0.5}],
            "constraints": {},
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

        agent = CoordinatorAgent(mock_llm, "cuda:70")
        result_state = agent.coordinate(state)

        opt_result = result_state["optimization_result"]
        assert opt_result is not None
        rec_block = opt_result["recommended_config"]["block_size"]
        # Should be corrected to BLOCK_SIZE
        assert "BLOCK_SIZE" in rec_block
        assert rec_block["BLOCK_SIZE"] == 512
        assert "XBLOCK" not in rec_block


# ===========================================================================
# Tests for KernelConfig.constexpr_param_names field
# ===========================================================================

class TestKernelConfigField:
    def test_default_empty(self):
        from agent_system_langgraph.state import KernelConfig

        kc = KernelConfig(kernel_name="k", ttgir="m")
        assert kc.constexpr_param_names == []

    def test_roundtrip_to_dict(self):
        from agent_system_langgraph.state import KernelConfig

        kc = KernelConfig(
            kernel_name="k", ttgir="m",
            constexpr_param_names=["BLOCK_SIZE"],
        )
        d = kc.to_dict()
        assert d["constexpr_param_names"] == ["BLOCK_SIZE"]
