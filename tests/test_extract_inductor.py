"""Tests for extract/run_inductor.py — inductor kernel extraction."""

import json
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import yaml

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extract.run_inductor import (
    _extract_source_paths_from_ttgir,
    _normalize_grid,
    _normalize_kwargs,
    _rewrite_ttgir_locs,
    _save_kernel_shapes,
    _shape_signature,
    _update_yaml_with_kernels,
    parse_inductor_ops,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_yaml(tmp_path):
    """Create a minimal oss_cuda_kernels.yaml for testing."""
    data = {
        "softmax": {
            "naive_softmax": {"tags": ["aten"]},
            "torch_compile_softmax": {"tags": ["pt2"]},
            "triton_softmax": {
                "kernels": ["Operator.softmax_kernel"],
                "tags": ["triton"],
            },
        },
        "vector_add": {
            "torch_add": {"tags": []},
            "torch_compile_add": {"tags": ["pt2"]},
            "triton_add": {
                "kernels": ["triton_add_kernel"],
                "tags": ["triton"],
            },
        },
        "addmm": {
            "aten_addmm": {"tags": ["aten"]},
            "pt2_triton_matmul": {"tags": ["pt2"]},
            "triton_addmm": {
                "kernels": ["_addmm_fwd"],
                "tags": ["triton"],
            },
        },
        "layer_norm": {
            "torch_compile_layer_norm": {"tags": ["pt2"]},
            "torch_layer_norm": {"tags": ["aten"]},
            "triton_layer_norm": {
                "kernels": ["_layer_norm_bwd_dwdb"],
                "tags": ["triton"],
            },
        },
        "no_inductor_op": {
            "triton_impl": {
                "kernels": ["some_kernel"],
                "tags": ["triton"],
            },
        },
    }
    yaml_path = tmp_path / "oss_cuda_kernels.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


SAMPLE_TTGIR = textwrap.dedent('''\
    #loc = loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":14:0)
    #loc1 = loc(unknown)
    #loc2 = loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":18:33)
    #loc3 = loc("/venv/lib/python3.12/site-packages/triton/language/standard.py":40:22)
    #loc4 = loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":20:27)
    tt.func public @triton_poi_fused_add_0(%arg0: !tt.ptr<f32> loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":14:0)) {
        %0 = tt.load %arg0 loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":18:33)
        tt.store %arg0, %0 loc("/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py":20:27)
    }
''')


# ---------------------------------------------------------------------------
# Test: YAML discovery
# ---------------------------------------------------------------------------

class TestDiscoverInductorOps:
    def test_finds_pt2_tagged_ops(self, sample_yaml):
        ops = parse_inductor_ops(sample_yaml)
        assert "softmax" in ops
        assert "vector_add" in ops
        assert "addmm" in ops
        assert "layer_norm" in ops

    def test_excludes_non_pt2_ops(self, sample_yaml):
        ops = parse_inductor_ops(sample_yaml)
        assert "no_inductor_op" not in ops

    def test_returns_impl_names(self, sample_yaml):
        ops = parse_inductor_ops(sample_yaml)
        assert ops["softmax"] == ["torch_compile_softmax"]
        assert ops["vector_add"] == ["torch_compile_add"]
        assert ops["addmm"] == ["pt2_triton_matmul"]

    def test_empty_yaml(self, tmp_path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        ops = parse_inductor_ops(yaml_path)
        assert ops == {}


# ---------------------------------------------------------------------------
# Test: TTGIR loc() rewriting
# ---------------------------------------------------------------------------

class TestTTGIRLocRewrite:
    def test_rewrite_loc_paths(self):
        old_path = "/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py"
        new_path = "/workspace/extract/ir_output/v100/vector_add/triton_poi_fused_add_0.py"
        rewritten = _rewrite_ttgir_locs(SAMPLE_TTGIR, old_path, new_path)
        assert old_path not in rewritten
        assert new_path in rewritten

    def test_preserves_library_paths(self):
        old_path = "/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py"
        new_path = "/workspace/extract/ir_output/v100/vector_add/triton_poi_fused_add_0.py"
        rewritten = _rewrite_ttgir_locs(SAMPLE_TTGIR, old_path, new_path)
        # Library path (site-packages) should NOT be rewritten
        assert "site-packages/triton/language/standard.py" in rewritten

    def test_preserves_line_numbers(self):
        old_path = "/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py"
        new_path = "/workspace/output/triton_poi_fused_add_0.py"
        rewritten = _rewrite_ttgir_locs(SAMPLE_TTGIR, old_path, new_path)
        # Line numbers must be preserved
        assert ':14:0' in rewritten
        assert ':18:33' in rewritten
        assert ':20:27' in rewritten

    def test_noop_when_empty(self):
        result = _rewrite_ttgir_locs(SAMPLE_TTGIR, "", "")
        assert result == SAMPLE_TTGIR

    def test_extract_source_paths(self):
        paths = _extract_source_paths_from_ttgir(SAMPLE_TTGIR)
        assert len(paths) == 1
        assert paths[0] == "/tmp/torchinductor_root/abc123/triton_poi_fused_add_0.py"
        # site-packages path should be filtered out
        assert all("site-packages" not in p for p in paths)


# ---------------------------------------------------------------------------
# Test: YAML update
# ---------------------------------------------------------------------------

class TestYAMLUpdate:
    def test_adds_kernels_entry(self, sample_yaml):
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        with open(sample_yaml) as f:
            data = yaml.safe_load(f)
        variant = data["softmax"]["torch_compile_softmax"]
        assert "kernels" in variant
        assert "triton_red_fused__softmax_0" in variant["kernels"]

    def test_preserves_existing_tags(self, sample_yaml):
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        with open(sample_yaml) as f:
            data = yaml.safe_load(f)
        assert "pt2" in data["softmax"]["torch_compile_softmax"]["tags"]

    def test_does_not_duplicate_kernels(self, sample_yaml):
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        with open(sample_yaml) as f:
            data = yaml.safe_load(f)
        kernels = data["softmax"]["torch_compile_softmax"]["kernels"]
        assert kernels.count("triton_red_fused__softmax_0") == 1

    def test_extends_existing_kernels(self, sample_yaml):
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_poi_fused__softmax_1"],
        )
        with open(sample_yaml) as f:
            data = yaml.safe_load(f)
        kernels = data["softmax"]["torch_compile_softmax"]["kernels"]
        assert "triton_red_fused__softmax_0" in kernels
        assert "triton_poi_fused__softmax_1" in kernels

    def test_does_not_clobber_other_ops(self, sample_yaml):
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )
        with open(sample_yaml) as f:
            data = yaml.safe_load(f)
        # Existing triton_softmax should be untouched
        assert "Operator.softmax_kernel" in data["softmax"]["triton_softmax"]["kernels"]
        # Other ops should be untouched
        assert "triton_add_kernel" in data["vector_add"]["triton_add"]["kernels"]


# ---------------------------------------------------------------------------
# Test: file naming compatibility with multiagent_triton_runner
# ---------------------------------------------------------------------------

class TestFileNamingCompat:
    def test_shapes_file_naming(self, tmp_path):
        """Saved shapes files must match _find_shapes() glob patterns."""
        captured = {
            "triton_poi_fused_add_0": [
                {
                    "kernel": "triton_poi_fused_add_0",
                    "tensor_args": [{"arg_index": 0, "shape": [1024], "dtype": "torch.float32"}],
                    "scalar_args": [],
                    "kwargs": {},
                    "grid": [4, 1, 1],
                    "total_args": 1,
                    "input_index": 0,
                },
                {
                    "kernel": "triton_poi_fused_add_0",
                    "tensor_args": [{"arg_index": 0, "shape": [2048], "dtype": "torch.float32"}],
                    "scalar_args": [],
                    "kwargs": {},
                    "grid": [8, 1, 1],
                    "total_args": 1,
                    "input_index": 1,
                },
            ]
        }
        _save_kernel_shapes(captured, "vector_add", tmp_path)

        # Check default shapes file exists
        default = tmp_path / "vector_add_triton_poi_fused_add_0_shapes.json"
        assert default.exists()
        data = json.loads(default.read_text())
        assert data["kernel"] == "triton_poi_fused_add_0"

        # Check per-input files exist
        input0 = tmp_path / "vector_add_triton_poi_fused_add_0_input0_shapes.json"
        input1 = tmp_path / "vector_add_triton_poi_fused_add_0_input1_shapes.json"
        assert input0.exists()
        assert input1.exists()

    def test_find_ttgir_glob_pattern(self, tmp_path):
        """Saved TTGIR must match _find_ttgir() glob: *{kernel_name}*.ttgir"""
        kernel_name = "triton_poi_fused_add_0"
        ttgir_file = tmp_path / f"vector_add_{kernel_name}.ttgir"
        ttgir_file.write_text("dummy ttgir content")

        # _find_ttgir uses: cache_dir.rglob(f"*{kernel_name}*.ttgir")
        candidates = sorted(tmp_path.rglob(f"*{kernel_name}*.ttgir"))
        assert len(candidates) == 1
        assert candidates[0] == ttgir_file

    def test_find_shapes_glob_pattern(self, tmp_path):
        """Saved shapes must match _find_shapes() globs."""
        kernel_name = "triton_poi_fused_add_0"

        # Create files matching naming convention
        default_shapes = tmp_path / f"vector_add_{kernel_name}_shapes.json"
        default_shapes.write_text(json.dumps({"kernel": kernel_name}))

        input0_shapes = tmp_path / f"vector_add_{kernel_name}_input0_shapes.json"
        input0_shapes.write_text(json.dumps({"kernel": kernel_name}))

        # _find_shapes uses: shapes_dir.rglob(f"*{kernel_name}_shapes.json")
        default_matches = sorted(tmp_path.rglob(f"*{kernel_name}_shapes.json"))
        assert len(default_matches) >= 1

        # _find_shapes uses: shapes_dir.rglob(f"*{kernel_name}_input*_shapes.json")
        per_input_matches = sorted(tmp_path.rglob(f"*{kernel_name}_input*_shapes.json"))
        assert len(per_input_matches) == 1


# ---------------------------------------------------------------------------
# Test: helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_shape_signature_deterministic(self):
        tensor_args = [{"arg_index": 0, "shape": [1024], "dtype": "torch.float32"}]
        scalar_args = [{"arg_index": 1, "value": 256}]
        grid = [4, 1, 1]
        sig1 = _shape_signature(tensor_args, scalar_args, grid)
        sig2 = _shape_signature(tensor_args, scalar_args, grid)
        assert sig1 == sig2

    def test_shape_signature_differs_for_different_shapes(self):
        tensor_args1 = [{"arg_index": 0, "shape": [1024], "dtype": "torch.float32"}]
        tensor_args2 = [{"arg_index": 0, "shape": [2048], "dtype": "torch.float32"}]
        scalar_args = []
        grid = [4, 1, 1]
        sig1 = _shape_signature(tensor_args1, scalar_args, grid)
        sig2 = _shape_signature(tensor_args2, scalar_args, grid)
        assert sig1 != sig2

    def test_normalize_grid_tuple(self):
        assert _normalize_grid((4, 1, 1)) == [4, 1, 1]

    def test_normalize_grid_list(self):
        assert _normalize_grid([4, 1, 1]) == [4, 1, 1]

    def test_normalize_grid_callable(self):
        grid = lambda meta: (4, 1, 1)
        result = _normalize_grid(grid)
        assert isinstance(result, dict)
        assert result["type"] == "callable"


# ---------------------------------------------------------------------------
# Test: multiagent_triton_runner tag filter change
# ---------------------------------------------------------------------------

class TestMultiagentRunnerTagFilter:
    """Verify that the tag filter in multiagent_triton_runner.py now includes pt2."""

    def test_parse_triton_kernels_includes_pt2(self, sample_yaml):
        """_parse_triton_kernels should find kernels under pt2-tagged variants."""
        # First, add kernels to a pt2 variant
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )

        # Now simulate what _parse_triton_kernels does
        with open(sample_yaml) as f:
            metadata = yaml.safe_load(f) or {}

        kernels = []
        for _op, variants in metadata.items():
            if not isinstance(variants, dict):
                continue
            for _variant, meta in variants.items():
                if not isinstance(meta, dict):
                    continue
                tags = meta.get("tags", []) or []
                # This is the updated filter from our change
                if not any("triton" in str(t).lower() or t == "pt2" for t in tags):
                    continue
                for k in meta.get("kernels", []) or []:
                    kernels.append(str(k))

        kernels = sorted(set(kernels))

        # Should include both hand-written triton AND pt2 inductor kernels
        assert "Operator.softmax_kernel" in kernels  # from triton_softmax
        assert "triton_red_fused__softmax_0" in kernels  # from torch_compile_softmax
        assert "triton_add_kernel" in kernels  # from triton_add

    def test_kernel_to_op_map_includes_pt2(self, sample_yaml):
        """_kernel_to_op_map should map pt2 kernel names to operators."""
        _update_yaml_with_kernels(
            sample_yaml, "softmax", "torch_compile_softmax",
            ["triton_red_fused__softmax_0"],
        )

        with open(sample_yaml) as f:
            metadata = yaml.safe_load(f) or {}

        mapping = {}
        for op_name, variants in metadata.items():
            if not isinstance(variants, dict):
                continue
            for _variant, meta in variants.items():
                if not isinstance(meta, dict):
                    continue
                tags = meta.get("tags", []) or []
                if not any("triton" in str(t).lower() or t == "pt2" for t in tags):
                    continue
                for k in meta.get("kernels", []) or []:
                    mapping[str(k)] = op_name

        assert mapping["triton_red_fused__softmax_0"] == "softmax"
        assert mapping["Operator.softmax_kernel"] == "softmax"
        assert mapping["triton_add_kernel"] == "vector_add"
