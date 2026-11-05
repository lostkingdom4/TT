#!/usr/bin/env python3
"""
Verification script to compare multi-agent recommended configurations
against TritonBench's default autotune configurations.

Usage:
    # Verify specific kernel with baseline CSV
    python verify_multiagent_config.py \\
      --operator addmm \\
      --kernel _addmm_fwd \\
      --cache-dir irs/new \\
      --input-shapes shapes/addmm__addmm_fwd_shapes.json \\
      --baseline-csv baseline_addmm.csv \\
      --output results/verification.json

    # Auto-discover all Triton kernels for operator
    python verify_multiagent_config.py \\
      --operator addmm \\
      --cache-dir irs/new \\
      --input-shapes-dir ./shapes \\
      --baseline-csv baseline_addmm.csv \\
      --output-dir ./results

    # Generate baseline CSV first using benchmark_baseline.py:
    python benchmark_baseline.py --op addmm --output baseline_addmm.csv
"""

import argparse
import ast
import json
import re
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from optimizer_functions import optimize_kernel_from_cache, extract_best_config, test_recommended_config
from tensor_utils import load_tensor_shapes

# Import TritonBench utilities for backend checking
from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS, override_args

try:
    from triton.runtime.jit import JITFunction
except Exception:
    JITFunction = None

try:
    import torch
except Exception:
    torch = None


def _metadata_dir() -> Path:
    from paths import METADATA_DIR
    return METADATA_DIR


def _get_triton_benchmarks_for_op(operator_name: str) -> List[str]:
    """Get all Triton benchmark names for an operator from metadata."""
    yaml_path = _metadata_dir().joinpath("oss_cuda_kernels.yaml")
    if not yaml_path.exists():
        return []
    with open(yaml_path, "r", encoding="utf-8") as fp:
        metadata = yaml.safe_load(fp) or {}

    benchmarks = []
    if operator_name in metadata:
        variants = metadata[operator_name]
        if isinstance(variants, dict):
            for variant_name, meta in variants.items():
                if not isinstance(meta, dict):
                    continue
                tags = meta.get("tags", []) or []
                if any("triton" in str(t).lower() for t in tags):
                    benchmarks.append(variant_name)
    return sorted(benchmarks)


def _get_triton_kernels_for_op(operator_name: str) -> Dict[str, str]:
    """Get all Triton kernels for an operator from metadata.

    Returns:
        Dict mapping kernel_name -> benchmark_name
    """
    yaml_path = _metadata_dir().joinpath("oss_cuda_kernels.yaml")
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as fp:
        metadata = yaml.safe_load(fp) or {}

    kernel_to_benchmark = {}
    if operator_name in metadata:
        variants = metadata[operator_name]
        if isinstance(variants, dict):
            for variant_name, meta in variants.items():
                if not isinstance(meta, dict):
                    continue
                tags = meta.get("tags", []) or []
                if not any("triton" in str(t).lower() for t in tags):
                    continue
                for k in meta.get("kernels", []) or []:
                    kernel_to_benchmark[str(k)] = variant_name
    return kernel_to_benchmark


def _select_triton_backend(op_name: str, torch_model: Optional[str] = None) -> Optional[str]:
    """Select a Triton backend for an operator."""
    if op_name not in REGISTERED_BENCHMARKS:
        return None
    backends = REGISTERED_BENCHMARKS[op_name]

    if torch_model:
        tm = torch_model.strip()
        candidates = [tm, tm.replace("triton_", ""), f"triton_{tm}", tm.replace("_kernel", "")]
        for name in backends.keys():
            if name in candidates or any(c in name for c in candidates):
                return name

    tagged = [name for name, cfg in backends.items() if getattr(cfg, "tags", None) and any("triton" in t.lower() for t in cfg.tags)]
    named = [name for name in backends if "triton" in name]

    if tagged:
        return tagged[0]
    if named:
        return named[0]
    return None


def _generate_ttgir(op_name: str, cache_dir: Path, device: str = "cuda") -> bool:
    """Generate TTGIR files by running the operator with --dump-ir."""
    try:
        print(f"Generating TTGIR files for operator '{op_name}'...")
        op_cls = load_opbench_by_name(op_name)
        if op_cls is None:
            print(f"Error: Could not load operator '{op_name}'")
            return False

        backend = _select_triton_backend(op_name)
        if not backend:
            print(f"Error: No Triton backend found for '{op_name}'")
            return False

        ir_dir = cache_dir / op_name
        arg_list = [
            "--op", op_name,
            "--device", device,
            "--only", backend,
            "--num-inputs", "1",
            "--input-id", "0",
            "--warmup", "0",
            "--rep", "1",
            "--repcnt", "1",
            "--dump-ir", str(ir_dir),
            "--skip-print",
            "--keep-going",
        ]

        tb_args, extra_args = override_args(arg_list)
        op = op_cls(tb_args=tb_args, extra_args=extra_args)
        op.run(warmup=0, rep=1, sleep=0)
        print(f"Generated TTGIR files in: {ir_dir}")
        return True
    except Exception as e:
        print(f"Error generating TTGIR: {e}")
        import traceback
        traceback.print_exc()
        return False


def _capture_inputs_for_kernel(
    op_name: str,
    kernel_name: str,
    shapes_dir: Path,
    device: str = "cuda",
    num_inputs: Optional[int] = None,
    cache_dir: Optional[Path] = None
) -> Optional[Path]:
    """Capture input shapes for a specific kernel by running the operator."""
    if JITFunction is None:
        print(f"Warning: Triton JIT not available, cannot capture shapes")
        return None

    op_cls = load_opbench_by_name(op_name)
    if op_cls is None:
        print(f"Warning: Could not load operator '{op_name}'")
        return None

    print(kernel_name)

    kernel_mapping = _get_triton_kernels_for_op(op_name)
    if kernel_name not in kernel_mapping:
        print(f"Warning: Kernel '{kernel_name}' not found for operator '{op_name}'")
        return None

    backend = kernel_mapping[kernel_name]

    shapes_dir.mkdir(parents=True, exist_ok=True)

    shape_mapping = {
        "kernel_name": kernel_name,
        "backend": backend,
        "operator": op_name,
        "inputs": []
    }

    original_run = JITFunction.run

    # Determine how many inputs to process
    if num_inputs is None:
        # Create temporary operator to query available inputs
        temp_arg_list = ["--op", op_name, "--device", device]
        temp_args, temp_extra = override_args(temp_arg_list)
        temp_op = op_cls(tb_args=temp_args, extra_args=temp_extra)
        num_inputs = temp_op.get_available_num_inputs()
        print(f"Auto-detected {num_inputs} available inputs for operator '{op_name}'")
    else:
        print(f"Processing {num_inputs} inputs as specified")

    # Ensure num_inputs is valid
    if num_inputs <= 0:
        print(f"Warning: No inputs available for operator '{op_name}'")
        return None

    for input_id in range(num_inputs):
        print(f"Capturing shapes for input {input_id}...")

        captured: Dict[str, Dict] = {}

        # Set up IR directory - use cache_dir if provided, otherwise use shapes_dir/ir
        if cache_dir:
            ir_dir = cache_dir / op_name
        else:
            ir_dir = shapes_dir / "ir" / op_name

        arg_list = [
            "--op", op_name,
            "--device", device,
            "--only", backend,
            "--input-id", str(input_id),
            "--num-inputs", "1",
            "--warmup", "0",
            "--rep", "1",
            "--repcnt", "1",
            "--dump-ir", str(ir_dir),
            "--skip-print",
            "--keep-going",
        ]
        tb_args, extra_args = override_args(arg_list)
        op = op_cls(tb_args=tb_args, extra_args=extra_args)

        def patched_run(self, *args, grid, **kwargs):
            try:
                kname = getattr(getattr(self, "fn", None), "__name__", "unknown")
                kernel_matches = (kname == kernel_name or
                                kernel_name.endswith(f".{kname}") or
                                kname == kernel_name.split('.')[-1])
                if kernel_matches:
                    tensor_args = []
                    scalar_args = []
                    for idx, arg in enumerate(args):
                        if torch is not None and isinstance(arg, getattr(torch, "Tensor", object)):
                            tensor_args.append({
                                "arg_index": idx,
                                "shape": list(arg.shape),
                                "dtype": str(arg.dtype),
                                "device": str(arg.device),
                                "numel": int(arg.numel()),
                                "stride": list(arg.stride()),
                                "is_contiguous": bool(arg.is_contiguous()),
                            })
                        elif isinstance(arg, (int, float, bool)):
                            scalar_args.append({"arg_index": idx, "value": arg, "type": type(arg).__name__})

                    kwargs_info = {}
                    for key, val in kwargs.items():
                        if key == "grid":
                            continue
                        if torch is not None and isinstance(val, getattr(torch, "Tensor", object)):
                            kwargs_info[key] = {"type": "tensor", "shape": list(val.shape), "dtype": str(val.dtype), "device": str(val.device)}
                        elif isinstance(val, (int, float, bool, str)):
                            kwargs_info[key] = {"type": "scalar", "value": val, "value_type": type(val).__name__}
                        else:
                            kwargs_info[key] = {"type": "other", "value": str(val), "value_type": type(val).__name__}

                    grid_info = None
                    if isinstance(grid, (tuple, list)):
                        grid_info = list(grid)
                    elif callable(grid):
                        try:
                            import inspect
                            grid_info = {"type": "callable", "source": inspect.getsource(grid)}
                        except Exception:
                            grid_info = {"type": "callable", "repr": str(grid)}
                    else:
                        grid_info = str(grid)

                    captured[kname] = {
                        "kernel": kname,
                        "tensor_args": tensor_args,
                        "scalar_args": scalar_args,
                        "kwargs": kwargs_info,
                        "grid": grid_info,
                        "total_args": len(args),
                    }
            except Exception as e:
                import traceback
                print(f"Warning: Exception during kernel capture: {e}")
                traceback.print_exc()
            return original_run(self, *args, grid=grid, **kwargs)

        try:
            JITFunction.run = patched_run
            op.run(warmup=0, rep=1, sleep=0)
        finally:
            JITFunction.run = original_run

        if kernel_name in captured or kernel_name.split('.')[-1] in captured:
            actual_key = kernel_name if kernel_name in captured else kernel_name.split('.')[-1]

            # Create nested directory structure: shapes_dir/{op_name}/{backend}/{kernel_name}/
            kernel_shapes_dir = shapes_dir / op_name / backend / kernel_name.split('.')[-1]
            kernel_shapes_dir.mkdir(parents=True, exist_ok=True)

            out_path = kernel_shapes_dir / f"shapes_{input_id}.json"
            out_path.write_text(json.dumps(captured[actual_key], indent=2))
            print(f"Captured shapes for input {input_id} to {out_path}")

            tensor_shapes = captured[actual_key]["tensor_args"]
            shape_hash = "x".join(str(s) for arg in tensor_shapes for s in arg["shape"]) if tensor_shapes else "unknown"

            shape_mapping["inputs"].append({
                "input_id": input_id,
                "shape_hash": shape_hash,
                "shape_file": str(out_path.relative_to(shapes_dir)),
                "tensor_shapes": [arg["shape"] for arg in tensor_shapes],
                "scalar_args": captured[actual_key]["scalar_args"],
                "ir_dir": str(ir_dir / backend)
            })
        else:
            print(f"Warning: Kernel '{kernel_name}' was not executed for input {input_id}")

    if not shape_mapping["inputs"]:
        print(f"Warning: No shapes captured for kernel '{kernel_name}'")
        return None

    # Save mapping file in the same nested structure
    kernel_shapes_dir = shapes_dir / op_name / backend / kernel_name.split('.')[-1]
    kernel_shapes_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = kernel_shapes_dir / "shape_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(shape_mapping, f, indent=2)

    print(f"Generated shape mapping for {len(shape_mapping['inputs'])} inputs")
    print(f"Mapping saved to: {mapping_file}")

    return mapping_file


def check_triton_backend(operator_name: str, torch_model: str) -> bool:
    """
    Check if the operator has a Triton backend implementation.

    Args:
        operator_name: TritonBench operator name
        torch_model: Source model name (used to match backend)

    Returns:
        True if Triton backend exists, False otherwise
    """
    # Load operator to populate REGISTERED_BENCHMARKS
    try:
        op_cls = load_opbench_by_name(operator_name)
        if op_cls is None:
            print(f"Warning: Could not load operator '{operator_name}'")
            return False
    except Exception as e:
        print(f"Warning: Error loading operator '{operator_name}': {e}")
        return False

    # Check if operator has registered backends
    if operator_name not in REGISTERED_BENCHMARKS:
        print(f"Warning: Operator '{operator_name}' not in REGISTERED_BENCHMARKS")
        return False

    backends = REGISTERED_BENCHMARKS[operator_name]

    # Look for Triton backends (check both tags and names)
    triton_backends = []
    for name, cfg in backends.items():
        # Check tags first
        tags = getattr(cfg, 'tags', None)
        if tags and any('triton' in str(t).lower() for t in tags):
            triton_backends.append(name)
        # Also check if 'triton' is in the backend name
        elif 'triton' in name.lower():
            triton_backends.append(name)

    if not triton_backends:
        print(f"Warning: No Triton backends found for operator '{operator_name}'")
        print(f"Available backends: {list(backends.keys())}")
        return False

    print(f"Found Triton backends for '{operator_name}': {triton_backends}")

    # Check if torch_model matches any Triton backend
    torch_model_normalized = torch_model.strip()
    candidates = [
        torch_model_normalized,
        torch_model_normalized.replace("triton_", ""),
        f"triton_{torch_model_normalized}",
        torch_model_normalized.replace("_kernel", ""),
    ]

    matched_backend = None
    for backend_name in triton_backends:
        if backend_name in candidates or any(c in backend_name for c in candidates):
            matched_backend = backend_name
            break

    if matched_backend:
        print(f"Matched Triton backend: {matched_backend}")
        return True
    else:
        print(f"Warning: torch_model '{torch_model}' does not match any Triton backend")
        print(f"Triton backends: {triton_backends}")
        print(f"Continuing anyway - will use first Triton backend: {triton_backends[0]}")
        return True  # Still return True if any Triton backend exists


def extract_tritonbench_configs(operator_name: str, kernel_name: str) -> List[Dict[str, Any]]:
    """
    Extract autotune configs from TritonBench operator source code.

    Args:
        operator_name: Name of the TritonBench operator (e.g., 'addmm')
        kernel_name: Name of the kernel function (e.g., '_addmm_fwd')

    Returns:
        List of config dictionaries with block sizes, num_warps, num_stages
    """
    # Find operator directory
    from paths import TRITONBENCH_PACKAGE_DIR
    operator_dir = TRITONBENCH_PACKAGE_DIR / "operators" / operator_name

    if not operator_dir.exists():
        print(f"Warning: Operator directory not found: {operator_dir}")
        return []

    # Search for kernel definition in operator files
    configs = []
    for py_file in operator_dir.rglob("*.py"):
        try:
            content = py_file.read_text()

            # Skip if kernel name not in file
            if kernel_name not in content:
                continue

            # Look for @triton.autotune decorator
            autotune_pattern = r'@triton\.autotune\s*\(\s*configs\s*=\s*\[(.*?)\]'
            matches = re.findall(autotune_pattern, content, re.DOTALL)

            for match in matches:
                # Parse triton.Config entries
                config_pattern = r'triton\.Config\s*\(\s*({[^}]+})\s*,?\s*(?:num_warps\s*=\s*(\d+))?\s*,?\s*(?:num_stages\s*=\s*(\d+))?'
                config_matches = re.findall(config_pattern, match)

                for block_dict_str, num_warps, num_stages in config_matches:
                    try:
                        # Parse block size dict
                        block_dict = ast.literal_eval(block_dict_str)

                        config = {
                            "block_size": block_dict,
                            "num_warps": int(num_warps) if num_warps else 4,
                            "num_stages": int(num_stages) if num_stages else 2,
                        }
                        configs.append(config)
                    except Exception as e:
                        print(f"Warning: Failed to parse config: {e}")
                        continue

        except Exception as e:
            print(f"Warning: Error reading {py_file}: {e}")
            continue

    if configs:
        print(f"Extracted {len(configs)} TritonBench configs for {kernel_name}")
    else:
        print(f"Warning: No autotune configs found for {kernel_name} in {operator_dir}")

    return configs


def run_multiagent_optimization(
    cache_dir: Path,
    kernel_name: str,
    torch_model: str,
    input_shapes_json: Path,
    operator_name: str,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Run multi-agent optimization and extract recommended config.

    Args:
        cache_dir: Directory containing TTGIR files
        kernel_name: Kernel function name
        torch_model: Source model name
        input_shapes_json: Path to shapes JSON file
        **kwargs: Additional arguments for optimization

    Returns:
        Optimization result dictionary
    """
    print(f"\nRunning multi-agent optimization for {kernel_name}...")

    result = optimize_kernel_from_cache(
        cache_dir=cache_dir,
        kernel_name=kernel_name,
        torch_model=torch_model,
        input_shapes_json=input_shapes_json,
        operator_name=operator_name,
        **kwargs
    )

    if result and result.get("success"):
        print("Multi-agent optimization completed successfully")
        return result
    else:
        print(f"Multi-agent optimization failed: {result.get('error') if result else 'Unknown error'}")
        return None


def benchmark_config(
    kernel_path: Path,
    kernel_name: str,
    input_shapes_json: Path,
    config: Dict[str, Any],
    device: str = "cuda",
    num_iterations: int = 100
) -> Optional[Dict[str, Any]]:
    """
    Benchmark a specific configuration.

    Args:
        kernel_path: Path to kernel source file
        kernel_name: Kernel function name
        input_shapes_json: Path to shapes JSON file
        config: Configuration dict with block_size, num_warps, num_stages
        device: Device to run on
        num_iterations: Number of benchmark iterations

    Returns:
        Benchmark result dictionary
    """
    try:
        result = test_recommended_config(
            kernel_name=kernel_name,
            kernel_path=kernel_path,
            input_shapes_json=input_shapes_json,
            config=config,
            device=device,
            num_iterations=num_iterations
        )
        return result
    except Exception as e:
        print(f"Warning: Benchmarking failed for config {config}: {e}")
        return None


def compare_configs(
    tritonbench_results: List[Dict[str, Any]],
    multiagent_result: Optional[Dict[str, Any]],
    baseline_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comparison report.

    Args:
        tritonbench_results: List of TritonBench config benchmark results
        multiagent_result: Multi-agent config benchmark result
        baseline_result: Baseline config benchmark result

    Returns:
        Comparison report dictionary
    """
    comparison = {
        "best_tritonbench": None,
        "multiagent": None,
        "baseline": None,
        "winner": None,
        "improvement_over_tritonbench": None
    }

    # Find best TritonBench config
    valid_tb_results = [r for r in tritonbench_results if r and r.get("success")]
    if valid_tb_results:
        best_tb = min(valid_tb_results, key=lambda r: r["execution_time_ms"]["median"])
        comparison["best_tritonbench"] = {
            "config": best_tb["config"],
            "time_ms": best_tb["execution_time_ms"]["median"],
            "speedup_vs_baseline": None
        }

        if baseline_result and baseline_result.get("success"):
            baseline_time = baseline_result["execution_time_ms"]["median"]
            comparison["best_tritonbench"]["speedup_vs_baseline"] = baseline_time / best_tb["execution_time_ms"]["median"]

    # Multi-agent result
    if multiagent_result and multiagent_result.get("success"):
        ma_time = multiagent_result["execution_time_ms"]["median"]
        comparison["multiagent"] = {
            "config": multiagent_result["config"],
            "time_ms": ma_time,
            "speedup_vs_baseline": None,
            "speedup_vs_best_tritonbench": None
        }

        if baseline_result and baseline_result.get("success"):
            baseline_time = baseline_result["execution_time_ms"]["median"]
            comparison["multiagent"]["speedup_vs_baseline"] = baseline_time / ma_time

        if comparison["best_tritonbench"]:
            best_tb_time = comparison["best_tritonbench"]["time_ms"]
            comparison["multiagent"]["speedup_vs_best_tritonbench"] = best_tb_time / ma_time

            # Determine winner
            if ma_time < best_tb_time:
                comparison["winner"] = "multiagent"
                improvement = ((best_tb_time - ma_time) / best_tb_time) * 100
                comparison["improvement_over_tritonbench"] = f"{improvement:.1f}%"
            else:
                comparison["winner"] = "tritonbench"
                improvement = ((ma_time - best_tb_time) / ma_time) * 100
                comparison["improvement_over_tritonbench"] = f"-{improvement:.1f}%"

    # Baseline result
    if baseline_result and baseline_result.get("success"):
        comparison["baseline"] = {
            "config": baseline_result.get("config", {}),
            "time_ms": baseline_result["execution_time_ms"]["median"]
        }

    return comparison


def verify_multiagent_config(
    operator_name: str,
    backend_name: str,
    kernel_name: str,
    cache_dir: Path,
    input_shapes_json: Path,
    output_file: Optional[Path] = None,
    baseline_csv: Optional[Path] = None,
    device: str = "cuda",
    num_iterations: int = 100,
    benchmark_tritonbench: bool = True,
    **multiagent_kwargs
) -> Dict[str, Any]:
    """
    Complete verification workflow.

    Args:
        operator_name: TritonBench operator name
        kernel_name: Kernel function name
        cache_dir: Directory containing TTGIR files
        input_shapes_json: Path to shapes JSON file
        output_file: Optional path to save results JSON
        baseline_csv: Optional path to baseline CSV file (from benchmark_baseline.py)
        device: Device to run on
        num_iterations: Number of benchmark iterations
        benchmark_tritonbench: Whether to benchmark TritonBench configs
        **multiagent_kwargs: Additional arguments for multi-agent optimization

    Returns:
        Complete verification results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Verifying Multi-Agent Configuration")
    print(f"{'='*80}")
    print(f"Operator: {operator_name}")
    print(f"Kernel: {kernel_name}")
    print(f"Input shapes: {input_shapes_json}")
    print(f"{'='*80}\n")

    # Load input shapes
    shapes_info = load_tensor_shapes(input_shapes_json)

    # Extract index from filename like "shapes_0.json"
    index = int(Path(input_shapes_json).stem.split('_')[-1])

    # Find kernel source file
    # Try both full kernel name and without class prefix
    kernel_search_names = [kernel_name, kernel_name.split('.')[-1]]
    ttgir_files = []
    operator_cache_dir = cache_dir / operator_name
    for search_name in kernel_search_names:
        ttgir_files = list(operator_cache_dir.rglob(f"*{search_name}*.ttgir"))
        if ttgir_files:
            break

    if not ttgir_files:
        print(f"No TTGIR found for kernel '{kernel_name}' in {cache_dir}")
        print(f"Attempting to generate TTGIR files...")
        if _generate_ttgir(operator_name, cache_dir, device):
            # Try finding TTGIR again with both names
            for search_name in kernel_search_names:
                ttgir_files = list(cache_dir.rglob(f"*{search_name}*.ttgir"))
                if ttgir_files:
                    break
            if not ttgir_files:
                raise FileNotFoundError(f"Failed to generate TTGIR for kernel '{kernel_name}' in {cache_dir}")
        else:
            raise FileNotFoundError(f"Could not generate TTGIR for kernel '{kernel_name}' in {cache_dir}")

    ttgir_path = ttgir_files[0]
    metadata_path = ttgir_path.with_suffix('.json')

    # Extract source location
    ttgir_content = ttgir_path.read_text()
    from configuration_optimizer import ConfigurationOptimizer
    source_file, _ = ConfigurationOptimizer._extract_source_location(ttgir_content)

    if not source_file:
        raise ValueError(f"Could not extract source location from {ttgir_path}")

    kernel_path = Path(source_file)
    torch_model = kernel_path.stem

    print(f"Kernel source: {kernel_path}")

    # Verify Triton backend exists
    print("Checking for Triton backend...")
    has_triton = check_triton_backend(operator_name, backend_name)
    if not has_triton:
        raise ValueError(
            f"Operator '{operator_name}' does not have a Triton backend implementation. "
            f"Cannot proceed with verification. Please ensure the operator has Triton kernels."
        )
    print()

    # Step 1: Extract TritonBench configs
    tritonbench_configs = []
    if benchmark_tritonbench:
        tritonbench_configs = extract_tritonbench_configs(operator_name, kernel_name)

    # Step 2: Run multi-agent optimization
    multiagent_result = run_multiagent_optimization(
        cache_dir=operator_cache_dir,
        kernel_name=kernel_name,
        torch_model=torch_model,
        input_shapes_json=input_shapes_json,
        operator_name=operator_name,
        **multiagent_kwargs
    )

    # Extract recommended config
    multiagent_config = None
    if multiagent_result:
        multiagent_config = extract_best_config(multiagent_result)

    # Step 3: Load baseline config from CSV (if provided)
    # NOTE: benchmark_baseline_config() has been removed due to parser bug.
    # Use benchmark_baseline.py to generate baseline CSV, then pass via --baseline-csv
    baseline_result = None
    if baseline_csv and baseline_csv.exists():
        print(f"\nLoading baseline results from CSV: {baseline_csv}")
        import csv
        with open(baseline_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('kernel_name') == kernel_name:
                    baseline_result = row
                    break
        if baseline_result:
            print(f"Found baseline for kernel: {kernel_name}")
        else:
            print(f"Warning: No baseline found for kernel {kernel_name} in CSV")
    else:
        print(f"\nNo baseline CSV provided. Use benchmark_baseline.py to generate one.")
        print(f"Example: python benchmark_baseline.py --op {operator_name} --output baseline.csv")

    # Step 4: Benchmark TritonBench configs
    tritonbench_results = []
    if benchmark_tritonbench and tritonbench_configs:
        print(f"\nBenchmarking {len(tritonbench_configs)} TritonBench configurations...")
        for i, config in enumerate(tritonbench_configs):
            print(f"  Config {i+1}/{len(tritonbench_configs)}: {config}")
            result = benchmark_config(
                kernel_path=kernel_path,
                kernel_name=kernel_name,
                input_shapes_json=input_shapes_json,
                config=config,
                device=device,
                num_iterations=num_iterations
            )
            if result:
                result["config"] = config
                tritonbench_results.append(result)

    # Step 5: Benchmark multi-agent config
    multiagent_benchmark = None
    if multiagent_config:
        print(f"\nBenchmarking multi-agent configuration...")
        print(f"  Config: {multiagent_config}")
        multiagent_benchmark = benchmark_config(
            kernel_path=kernel_path,
            kernel_name=kernel_name,
            input_shapes_json=input_shapes_json,
            config=multiagent_config,
            device=device,
            num_iterations=num_iterations
        )
        if multiagent_benchmark:
            multiagent_benchmark["config"] = multiagent_config

    # Step 6: Generate comparison report
    comparison = compare_configs(
        tritonbench_results=tritonbench_results,
        multiagent_result=multiagent_benchmark,
        baseline_result=baseline_result
    )

    # Build final results
    results = {
        "kernel": kernel_name,
        "operator": operator_name,
        "input_shapes": shapes_info,
        "tritonbench_configs": tritonbench_results,
        "multiagent_config": multiagent_benchmark,
        "baseline_config": baseline_result,
        "comparison": comparison
    }

    # Save to file if requested
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")

    if baseline_result and baseline_result.get("success"):
        print(f"Baseline: {baseline_result['execution_time_ms']['median']:.3f} ms")

    if comparison["best_tritonbench"]:
        print(f"Best TritonBench: {comparison['best_tritonbench']['time_ms']:.3f} ms "
              f"(speedup: {comparison['best_tritonbench'].get('speedup_vs_baseline', 'N/A')}x)")

    if comparison["multiagent"]:
        print(f"Multi-Agent: {comparison['multiagent']['time_ms']:.3f} ms "
              f"(speedup: {comparison['multiagent'].get('speedup_vs_baseline', 'N/A')}x)")

        if comparison["winner"]:
            print(f"\nWinner: {comparison['winner'].upper()}")
            if comparison["improvement_over_tritonbench"]:
                print(f"Improvement: {comparison['improvement_over_tritonbench']}")

    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Verify multi-agent configurations against TritonBench defaults"
    )
    parser.add_argument("--operator", type=str, required=True, help="TritonBench operator name")
    parser.add_argument("--kernel", type=str, help="Kernel function name (optional, will discover all if not provided)")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory containing TTGIR files")
    parser.add_argument("--input-shapes", type=Path, help="Path to shapes JSON file (optional, will auto-capture if not provided)")
    parser.add_argument("--input-shapes-dir", type=Path, default=Path("./shapes"), help="Directory to save/load captured shapes")
    parser.add_argument("--baseline-csv", type=Path, help="Path to baseline CSV file (from benchmark_baseline.py)")
    parser.add_argument("--output", type=Path, help="Output JSON file path")
    parser.add_argument("--output-dir", type=Path, help="Output directory for multiple results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--skip-tritonbench", action="store_true", help="Skip benchmarking TritonBench configs")
    parser.add_argument("--llm-provider", type=str, default="deepseek", help="LLM provider for multi-agent")
    parser.add_argument("--debug-mode", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Discover kernels if not specified
    kernels_to_verify = []
    if args.kernel:
        kernels_to_verify = [args.kernel]
    else:
        print(f"Discovering Triton kernels for operator '{args.operator}'...")
        kernel_mapping = _get_triton_kernels_for_op(args.operator)
        if not kernel_mapping:
            print(f"Error: No Triton kernels found for operator '{args.operator}'")
            sys.exit(1)
        kernels_to_verify = list(kernel_mapping.keys())
        print(f"Found {len(kernels_to_verify)} Triton kernels: {kernels_to_verify}")

    # Process each kernel
    all_results = []
    for kernel_name in kernels_to_verify:
        print(f"\n{'='*80}")
        print(f"Processing kernel: {kernel_name}")
        print(f"{'='*80}\n")
        backend = kernel_mapping[kernel_name]

        # Find or capture input shapes
        input_shapes_json = args.input_shapes
        kernel_cache_dir = args.cache_dir  # Default to args.cache_dir

        if not input_shapes_json:
            # Try to find existing shapes in new nested structure
            shapes_dir = args.input_shapes_dir
            # Search for shapes files in nested structure: {op_name}/*/{kernel_name}/shapes_*.json
            kernel_short_name = kernel_name.split('.')[-1]
            shapes_pattern = f"{args.operator}/{backend}/{kernel_short_name}/shapes_*.json"
            existing_shapes = list(shapes_dir.glob(shapes_pattern))

            if existing_shapes:
                input_shapes_json = existing_shapes[0]
                print(f"Found existing shapes: {input_shapes_json}")

                # Try to load mapping file to get IR directory
                mapping_file = input_shapes_json.parent / "shape_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file) as f:
                        shape_mapping = json.load(f)
                    if shape_mapping.get("inputs"):
                        first_input = shape_mapping["inputs"][0]
                        if "ir_dir" in first_input and not kernel_cache_dir:
                            kernel_cache_dir = Path(first_input["ir_dir"]).parent
                        print(f"Using input_id={first_input['input_id']} with shape {first_input.get('shape_hash', 'unknown')}")
            else:
                # Capture shapes
                print(f"No shapes found, capturing for kernel '{kernel_name}'...")
                mapping_file = _capture_inputs_for_kernel(
                    args.operator, kernel_name, shapes_dir, args.device,
                    cache_dir=kernel_cache_dir
                )
                if not mapping_file:
                    print(f"Error: Could not capture shapes for kernel '{kernel_name}', skipping...")
                    continue

                # Load mapping and select first input
                with open(mapping_file) as f:
                    shape_mapping = json.load(f)

                if not shape_mapping.get("inputs"):
                    print(f"Error: No inputs captured for kernel '{kernel_name}', skipping...")
                    continue

                # Use first input's shape file
                first_input = shape_mapping["inputs"][0]
                input_shapes_json = shapes_dir / first_input["shape_file"]
                if "ir_dir" in first_input and not kernel_cache_dir:
                    kernel_cache_dir = Path(first_input["ir_dir"]).parent
                print(f"Using input_id={first_input['input_id']} with shape {first_input.get('shape_hash', 'unknown')}")

        # Determine output file
        output_file = args.output
        if not output_file and args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = args.output_dir / f"{args.operator}_{kernel_name}_verification.json"

        try:
            result = verify_multiagent_config(
                operator_name=args.operator,
                backend_name=backend,
                kernel_name=kernel_name,
                cache_dir=kernel_cache_dir,
                input_shapes_json=input_shapes_json,
                output_file=output_file,
                baseline_csv=args.baseline_csv,
                device=args.device,
                num_iterations=args.num_iterations,
                benchmark_tritonbench=not args.skip_tritonbench,
                llm_provider=args.llm_provider,
                debug_mode=args.debug_mode
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error verifying kernel '{kernel_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary if multiple kernels
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Verified {len(all_results)} kernels")
        print(f"{'='*80}")
        for result in all_results:
            kernel = result.get("kernel", "unknown")
            comparison = result.get("comparison", {})
            winner = comparison.get("winner", "N/A")
            improvement = comparison.get("improvement_over_tritonbench", "N/A")
            print(f"  {kernel}: Winner={winner}, Improvement={improvement}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
