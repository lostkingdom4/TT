#!/usr/bin/env python3
"""
Tests for the output correctness validation in compare_configs.py.

Verifies that _compare_tensors, _get_dtype_tolerances, _clone_tensor_args,
_validate_outputs, and the validate_outputs parameter in benchmark_config_pair
all behave correctly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# Allow importing from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from compare_configs import (
    _clone_tensor_args,
    _compare_tensors,
    _get_dtype_tolerances,
    _validate_outputs,
)


# ---------------------------------------------------------------------------
# Tests: _get_dtype_tolerances
# ---------------------------------------------------------------------------

class TestGetDtypeTolerances:
    def test_fp16(self):
        rtol, atol = _get_dtype_tolerances(torch.float16)
        assert rtol == 1e-2
        assert atol == 1e-2

    def test_bf16(self):
        rtol, atol = _get_dtype_tolerances(torch.bfloat16)
        assert rtol == 1e-2
        assert atol == 1e-2

    def test_fp32(self):
        rtol, atol = _get_dtype_tolerances(torch.float32)
        assert rtol == 1e-4
        assert atol == 1e-4

    def test_fp64(self):
        rtol, atol = _get_dtype_tolerances(torch.float64)
        assert rtol == 1e-5
        assert atol == 1e-5

    def test_bool(self):
        rtol, atol = _get_dtype_tolerances(torch.bool)
        assert rtol == 0.0
        assert atol == 0.0

    def test_int32(self):
        rtol, atol = _get_dtype_tolerances(torch.int32)
        assert rtol == 0.0
        assert atol == 0.0

    def test_int64(self):
        rtol, atol = _get_dtype_tolerances(torch.int64)
        assert rtol == 0.0
        assert atol == 0.0


# ---------------------------------------------------------------------------
# Tests: _clone_tensor_args
# ---------------------------------------------------------------------------

class TestCloneTensorArgs:
    def test_cloned_tensors_are_independent(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        args = [t, 42]
        kwargs = {"out": torch.zeros(3)}

        cloned_args, cloned_kwargs = _clone_tensor_args(args, kwargs)

        # Modify clone — original should be unaffected
        cloned_args[0].fill_(99.0)
        cloned_kwargs["out"].fill_(99.0)

        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(args[0], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(kwargs["out"], torch.zeros(3))

    def test_scalars_pass_through(self):
        args = [42, 3.14, "hello"]
        cloned_args, _ = _clone_tensor_args(args, {})
        assert cloned_args == [42, 3.14, "hello"]

    def test_kwargs_tensors_cloned(self):
        kw = {"a": torch.ones(2), "b": 5}
        _, cloned_kw = _clone_tensor_args([], kw)
        assert isinstance(cloned_kw["a"], torch.Tensor)
        assert cloned_kw["b"] == 5
        cloned_kw["a"].fill_(0.0)
        assert torch.equal(kw["a"], torch.ones(2))

    def test_empty_args(self):
        cloned_args, cloned_kw = _clone_tensor_args([], {})
        assert cloned_args == []
        assert cloned_kw == {}


# ---------------------------------------------------------------------------
# Tests: _compare_tensors
# ---------------------------------------------------------------------------

class TestCompareTensors:
    def test_exact_match_int(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        b = torch.tensor([1, 2, 3], dtype=torch.int32)
        assert _compare_tensors(a, b, "test") is None

    def test_exact_match_fp32(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = a.clone()
        assert _compare_tensors(a, b, "test") is None

    def test_small_diff_within_tolerance_fp16(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
        # Add a tiny perturbation within fp16 tolerance (1e-2)
        b = a + 1e-3
        result = _compare_tensors(a, b, "test")
        assert result is None

    def test_large_diff_outside_tolerance_fp32(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32)
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "value_mismatch"
        assert result["max_abs_diff"] == pytest.approx(0.5, abs=1e-6)
        assert result["num_mismatched"] >= 1

    def test_int_mismatch_exact(self):
        a = torch.tensor([1, 2, 3], dtype=torch.int32)
        b = torch.tensor([1, 2, 4], dtype=torch.int32)
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "value_mismatch"

    def test_shape_mismatch(self):
        a = torch.zeros(3)
        b = torch.zeros(4)
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "shape_mismatch"
        assert result["original_shape"] == [3]
        assert result["recommended_shape"] == [4]

    def test_nan_in_recommended_only(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, float("nan"), 3.0])
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "nan_mismatch"
        assert result["recommended_nan_count"] == 1
        assert result["original_nan_count"] == 0

    def test_nan_in_both_same_positions(self):
        a = torch.tensor([1.0, float("nan"), 3.0])
        b = torch.tensor([1.0, float("nan"), 3.0])
        result = _compare_tensors(a, b, "test")
        assert result is None

    def test_nan_in_both_different_positions(self):
        a = torch.tensor([float("nan"), 2.0, 3.0])
        b = torch.tensor([1.0, float("nan"), 3.0])
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "nan_mismatch"

    def test_inf_matching(self):
        a = torch.tensor([float("inf"), -float("inf"), 1.0])
        b = torch.tensor([float("inf"), -float("inf"), 1.0])
        result = _compare_tensors(a, b, "test")
        assert result is None

    def test_bool_exact_match(self):
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, False, True])
        result = _compare_tensors(a, b, "test")
        assert result is None

    def test_bool_mismatch(self):
        a = torch.tensor([True, False, True])
        b = torch.tensor([True, True, True])
        result = _compare_tensors(a, b, "test")
        assert result is not None
        assert result["reason"] == "value_mismatch"

    def test_result_includes_name(self):
        a = torch.tensor([1.0])
        b = torch.tensor([999.0])
        result = _compare_tensors(a, b, "arg[2]")
        assert result["name"] == "arg[2]"

    def test_all_nan_tensors_match(self):
        a = torch.tensor([float("nan"), float("nan")])
        b = torch.tensor([float("nan"), float("nan")])
        result = _compare_tensors(a, b, "test")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    """Tests using mock kernel functions (no GPU required)."""

    def _make_kernel_fn(self, side_effect=None):
        """Create a mock that behaves like kernel_fn[grid](*args, **kwargs)."""
        inner = MagicMock(side_effect=side_effect)
        kernel = MagicMock()
        kernel.__getitem__ = MagicMock(return_value=inner)
        return kernel

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_valid_when_kernel_is_deterministic(self):
        """Both configs write the same constant — outputs should match."""
        def write_ones(*args, **kwargs):
            for a in args:
                if isinstance(a, torch.Tensor):
                    a.fill_(1.0)
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    v.fill_(1.0)

        kernel = self._make_kernel_fn(side_effect=write_ones)
        args = [torch.zeros(4), 42]
        orig_kw = {"out": torch.zeros(4), "BLOCK_SIZE": 128}
        rec_kw = {"out": torch.zeros(4), "BLOCK_SIZE": 256}

        result = _validate_outputs(
            kernel, (1,), (1,), args, orig_kw, rec_kw, device="cpu",
        )
        assert result["valid"] is True
        assert result["mismatches"] == []
        assert result["num_tensors_checked"] == 2  # args[0] + kwargs["out"]

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_invalid_when_recommended_produces_wrong_output(self):
        """Recommended config writes different values — should fail."""
        call_count = [0]

        def config_dependent(*args, **kwargs):
            call_count[0] += 1
            for a in args:
                if isinstance(a, torch.Tensor):
                    if call_count[0] == 1:
                        a.fill_(1.0)  # original
                    else:
                        a.fill_(999.0)  # recommended — different!

        kernel = self._make_kernel_fn(side_effect=config_dependent)
        args = [torch.zeros(4)]

        result = _validate_outputs(
            kernel, (1,), (1,), args, {}, {}, device="cpu",
        )
        assert result["valid"] is False
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["name"] == "arg[0]"
        assert result["mismatches"][0]["reason"] == "value_mismatch"

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_error_when_original_crashes(self):
        """Original config raises — validation should report error."""
        kernel = self._make_kernel_fn(side_effect=RuntimeError("segfault"))
        args = [torch.zeros(4)]

        result = _validate_outputs(
            kernel, (1,), (1,), args, {}, {}, device="cpu",
        )
        assert result["valid"] is False
        assert "original config execution failed" in result["error"]

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_error_when_recommended_crashes(self):
        """Recommended config raises — validation should report error."""
        call_count = [0]

        def fail_second_time(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError("out of shared memory")

        kernel = self._make_kernel_fn(side_effect=fail_second_time)
        args = [torch.zeros(4)]

        result = _validate_outputs(
            kernel, (1,), (1,), args, {}, {}, device="cpu",
        )
        assert result["valid"] is False
        assert "recommended config execution failed" in result["error"]

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_mixed_tensor_and_scalar_args(self):
        """Only tensors should be compared; scalars are ignored."""
        def noop(*args, **kwargs):
            pass

        kernel = self._make_kernel_fn(side_effect=noop)
        args = [torch.zeros(4), 42, torch.ones(3), "stride"]

        result = _validate_outputs(
            kernel, (1,), (1,), args, {}, {}, device="cpu",
        )
        assert result["valid"] is True
        assert result["num_tensors_checked"] == 2  # Only the 2 tensors

    @patch("compare_configs.device_sync", new=MagicMock())
    def test_no_tensors_still_valid(self):
        """If there are no tensor args, validation passes trivially."""
        kernel = self._make_kernel_fn(side_effect=lambda *a, **k: None)
        result = _validate_outputs(
            kernel, (1,), (1,), [42, 3.14], {"N": 1024}, {"N": 1024}, device="cpu",
        )
        assert result["valid"] is True
        assert result["num_tensors_checked"] == 0


# ---------------------------------------------------------------------------
# Tests: benchmark_config_pair with validate_outputs
# ---------------------------------------------------------------------------

class TestBenchmarkConfigPairValidation:
    """Integration tests verifying validate_outputs flows through benchmark_config_pair."""

    def _patch_internals(self, validation_result=None, benchmark_error=False):
        """Return patches for all heavy functions inside benchmark_config_pair."""
        patches = {
            "load_tensor_shapes": patch(
                "compare_configs.load_tensor_shapes",
                return_value={"kwargs": {}, "tensor_args": [], "grid": [128]},
            ),
            "create_kernel_inputs": patch(
                "compare_configs.create_kernel_inputs",
                return_value=([torch.zeros(4)], {}),
            ),
            "_load_kernel": patch(
                "compare_configs._load_kernel",
                return_value=MagicMock(),
            ),
            "_unwrap_autotuner": patch(
                "compare_configs._unwrap_autotuner",
                return_value=(MagicMock(), []),
            ),
            "_build_kwargs": patch(
                "compare_configs._build_kwargs",
                return_value={"BLOCK_SIZE": 128},
            ),
            "_map_block_sizes": patch(
                "compare_configs._map_block_sizes",
                return_value={},
            ),
            "_resolve_grid": patch(
                "compare_configs._resolve_grid",
                return_value=(128,),
            ),
            "_benchmark_config": patch(
                "compare_configs._benchmark_config",
                return_value=(
                    {"error": "boom"} if benchmark_error
                    else {"median": 10.0, "mean": 10.0, "min": 10.0, "max": 10.0, "std_dev": 0.0, "samples": 20}
                ),
            ),
        }
        if validation_result is not None:
            patches["_validate_outputs"] = patch(
                "compare_configs._validate_outputs",
                return_value=validation_result,
            )
        return patches

    def test_validation_result_included_on_success(self):
        validation_ok = {
            "valid": True,
            "mismatches": [],
            "error": None,
            "num_tensors_checked": 1,
        }
        patches = self._patch_internals(validation_result=validation_ok)
        with patches["load_tensor_shapes"], patches["create_kernel_inputs"], \
             patches["_load_kernel"], patches["_unwrap_autotuner"], \
             patches["_build_kwargs"], patches["_map_block_sizes"], \
             patches["_resolve_grid"], patches["_benchmark_config"], \
             patches["_validate_outputs"]:
            from compare_configs import benchmark_config_pair
            result = benchmark_config_pair(
                original_config={"num_warps": 4},
                recommended_config={"num_warps": 8},
                shapes_file=Path("/tmp/shapes.json"),
                kernel_module=Path("/tmp/kernel.py"),
                kernel_name="test_kernel",
                validate_outputs=True,
            )
        assert "error" not in result
        assert result["validation"] == validation_ok
        assert result["speedup"] is not None

    def test_validation_failure_returns_error(self):
        validation_fail = {
            "valid": False,
            "mismatches": [{"name": "arg[0]", "reason": "value_mismatch", "max_abs_diff": 5.0}],
            "error": None,
            "num_tensors_checked": 1,
        }
        patches = self._patch_internals(validation_result=validation_fail)
        with patches["load_tensor_shapes"], patches["create_kernel_inputs"], \
             patches["_load_kernel"], patches["_unwrap_autotuner"], \
             patches["_build_kwargs"], patches["_map_block_sizes"], \
             patches["_resolve_grid"], patches["_benchmark_config"], \
             patches["_validate_outputs"]:
            from compare_configs import benchmark_config_pair
            result = benchmark_config_pair(
                original_config={"num_warps": 4},
                recommended_config={"num_warps": 8},
                shapes_file=Path("/tmp/shapes.json"),
                kernel_module=Path("/tmp/kernel.py"),
                kernel_name="test_kernel",
                validate_outputs=True,
            )
        assert "error" in result
        assert "output validation failed" in result["error"]
        assert result["validation_result"] == validation_fail

    def test_validate_outputs_false_skips_validation(self):
        patches = self._patch_internals()
        with patches["load_tensor_shapes"], patches["create_kernel_inputs"], \
             patches["_load_kernel"], patches["_unwrap_autotuner"], \
             patches["_build_kwargs"], patches["_map_block_sizes"], \
             patches["_resolve_grid"], patches["_benchmark_config"]:
            from compare_configs import benchmark_config_pair
            result = benchmark_config_pair(
                original_config={"num_warps": 4},
                recommended_config={"num_warps": 8},
                shapes_file=Path("/tmp/shapes.json"),
                kernel_module=Path("/tmp/kernel.py"),
                kernel_name="test_kernel",
                validate_outputs=False,
            )
        assert "error" not in result
        assert result["validation"] is None


# ---------------------------------------------------------------------------
# Tests: Orchestrator feedback labeling
# ---------------------------------------------------------------------------

class TestOrchestratorFeedbackLabeling:
    """Verify that construct_agent_feedback labels validation errors distinctly."""

    def test_correctness_error_labeled(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        results = [{
            "candidate_index": 0,
            "config": {"num_warps": 8, "block_size": {"XBLOCK": 256}, "num_stages": 2},
            "error": "output validation failed: output mismatch in 1 tensor(s)",
        }]
        candidates = [{
            "config": {"num_warps": 8, "block_size": {"XBLOCK": 256}},
            "rationale": "test",
            "confidence": 0.9,
        }]
        dm_proposal = {"block_size": {"XBLOCK": 256}, "num_stages": 2}
        rb_proposal = {"num_warps": 8}

        feedback = construct_agent_feedback(
            results, candidates, dm_proposal, rb_proposal,
        )
        assert "CORRECTNESS ERROR" in feedback["data_movement_feedback"]
        assert "CORRECTNESS ERROR" in feedback["resource_budget_feedback"]

    def test_regular_error_not_labeled_correctness(self):
        from agent_system_langgraph.orchestrator import construct_agent_feedback

        results = [{
            "candidate_index": 0,
            "config": {"num_warps": 8, "block_size": {"XBLOCK": 256}, "num_stages": 2},
            "error": "benchmark crashed: CUDA OOM",
        }]
        candidates = [{
            "config": {"num_warps": 8, "block_size": {"XBLOCK": 256}},
            "rationale": "test",
            "confidence": 0.9,
        }]
        dm_proposal = {"block_size": {"XBLOCK": 256}, "num_stages": 2}
        rb_proposal = {"num_warps": 8}

        feedback = construct_agent_feedback(
            results, candidates, dm_proposal, rb_proposal,
        )
        assert "CORRECTNESS ERROR" not in feedback["data_movement_feedback"]
        assert "ERROR" in feedback["data_movement_feedback"]
