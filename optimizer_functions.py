#!/usr/bin/env python3
"""
TTGIR Optimizer Functions

Collection of utility functions for analyzing kernels from cache,
extracting configurations, benchmarking, and testing configurations.
"""

import json
import math
import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from configuration_optimizer import ConfigurationOptimizer
from tensor_utils import create_kernel_inputs, load_tensor_shapes, print_tensor_info
from grid_utils import compute_grid_from_numel, infer_numel_for_kernel
from benchmark_utils import benchmark_kernel
from ttgir_parser import TTGIRParser
from compare_configs import benchmark_config_pair


def _autotune_first_config_for_kernel(
	kernel_name: str,
	source_file: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
	"""Return the first entry of an autotune candidate list for *kernel_name*.

	Used when BASELINE_USE_AUTOTUNE_FIRST=1 is set: the first config in a
	kernel's @triton.autotune decorator list becomes the "original" baseline,
	so speedup numbers for kernels that share the same decorator (e.g. the
	three addmm kernels) are measured against the same starting point.

	Lookup order:
	  1. The hard-coded ``KERNEL_AUTOTUNE_CONFIGS`` table in
	     ``baselines/common/search_space.py`` (TritonBench).
	  2. A captured JSON next to *source_file* named
	     ``{kernel_name}_autotune_configs.json`` — produced for KernelBench
	     templates by ``extract/capture_kb_autotune_configs.py``.
	"""
	try:
		from baselines.common.search_space import (
			KERNEL_AUTOTUNE_CONFIGS,
			CONFIG_LEVEL_PARAMS,
		)
	except Exception:
		KERNEL_AUTOTUNE_CONFIGS = {}
		CONFIG_LEVEL_PARAMS = {"num_warps", "num_stages", "num_ctas"}
	flat = None
	for (_op, kern), cfgs in KERNEL_AUTOTUNE_CONFIGS.items():
		if kern == kernel_name and cfgs:
			flat = cfgs[0]
			break
	# Fallback: JSON next to the kernel's captured .py (KernelBench path)
	if flat is None and source_file:
		cfg_json = Path(source_file).parent / f"{kernel_name}_autotune_configs.json"
		if cfg_json.exists():
			try:
				with open(cfg_json, 'r') as fp:
					payload = json.load(fp)
				cands = payload.get("candidates") or []
				if cands:
					flat = cands[0]
			except (OSError, json.JSONDecodeError):
				pass
	if flat is None:
		return None
	top: Dict[str, Any] = {}
	block: Dict[str, int] = {}
	for k, v in flat.items():
		if k in CONFIG_LEVEL_PARAMS:
			top[k] = v
		else:
			block[k] = v
	top["block_size"] = block
	return top


def _extract_input_id_from_shapes_path(input_shapes_json: Path) -> Optional[int]:
	"""Extract the input_id from a shapes filename or its JSON contents.

	Naming convention: ``*_input{N}_shapes.json`` -> N.
	Fallback: reads the ``input_index`` field from the JSON.
	Returns ``None`` for legacy files without an input identifier.
	"""
	if not input_shapes_json:
		return None
	m = re.search(r'input(\d+)_shapes\.json$', input_shapes_json.name)
	if m:
		return int(m.group(1))
	if input_shapes_json.exists():
		try:
			with open(input_shapes_json, 'r') as f:
				data = json.load(f)
			if isinstance(data, dict) and 'input_index' in data:
				return int(data['input_index'])
		except (json.JSONDecodeError, ValueError, OSError):
			pass
	return None


def _select_source_for_input(
	base_source: Path,
	kernel_name: str,
	input_id: Optional[int],
) -> Path:
	"""Return the per-input captured source when present, else *base_source*.

	Inductor template kernels bake ``M``/``N``/``K``/strides into the
	``@triton.jit`` body as literals, so each captured call-site needs its own
	``{kernel_name}_input{input_id}.py`` snapshot — produced by
	``extract/run_inductor.py`` adjacent to the base source. This resolver
	mirrors ``_select_ttgir_for_input``: prefer the per-input file, silently
	fall back to the base path when ``input_id`` is ``None`` or the snapshot
	is missing (callers decide whether the fallback is acceptable).
	"""
	if input_id is None:
		return base_source
	per_shape = base_source.parent / f"{kernel_name}_input{input_id}.py"
	return per_shape if per_shape.exists() else base_source


def _select_ttgir_for_input(
	ttgir_files: List[Path],
	torch_model: str,
	input_id: Optional[int],
) -> Optional[Path]:
	"""Select the TTGIR file matching *torch_model* and, when available, *input_id*.

	Collects **all** TTGIR files whose embedded source location matches
	*torch_model*, then prefers the one whose filename contains
	``input_x{input_id}``.  Falls back to the first match when no
	*input_id* is given or no filename matches.
	"""
	matching: List[Path] = []
	for f in ttgir_files:
		with open(f, 'r') as fh:
			content = fh.read()
		source_file, _ = ConfigurationOptimizer._extract_source_location(content)
		if not source_file:
			continue
		source_file = source_file.split('/')[-1].split('.')[0]
		if source_file == torch_model:
			matching.append(f)

	if not matching:
		return None

	if input_id is not None:
		# Use regex to avoid input_x1 matching input_x10, etc.
		pat = re.compile(rf'input_x{input_id}(?:\D|$)')
		for f in matching:
			if pat.search(f.stem):
				return f

	return matching[0]


def _evaluate_callable_grid(
	grid_info: Dict[str, Any], shapes_info: Dict[str, Any]
) -> Optional[List[int]]:
	"""Evaluate a callable grid from its captured source + runtime args.

	Triton grids can be lambdas like::

	    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),
	                         triton.cdiv(N, meta["BLOCK_N"]))

	We reconstruct a namespace from the shapes JSON (scalar args, kwargs,
	tensor dimensions) and eval the lambda to get concrete grid values.

	Returns a list of ints ``[grid_x, grid_y, ...]`` or ``None`` on failure.
	"""
	source = grid_info.get("source", "")
	if not source:
		return None

	# Build meta dict from kwargs (what Triton passes as meta["BLOCK_M"] etc.)
	meta: Dict[str, Any] = {}
	for k, v in shapes_info.get("kwargs", {}).items():
		if isinstance(v, dict) and v.get("type") == "scalar":
			meta[k] = v["value"]

	# Build eval namespace — start with kwargs scalars (BLOCK_M, GROUP_M, ...)
	namespace: Dict[str, Any] = dict(meta)

	# Add positional scalar args (these are closure variables like M, N, K)
	for sa in shapes_info.get("scalar_args", []):
		namespace[f"arg{sa['arg_index']}"] = sa["value"]

	# Infer common dimension names from tensor shapes
	tensor_args = shapes_info.get("tensor_args", [])
	if len(tensor_args) >= 2:
		shapes = [t["shape"] for t in tensor_args]
		# Common gemm pattern: A[M,K] x B[K,N]
		if len(shapes[0]) == 2 and len(shapes[1]) == 2:
			namespace.setdefault("M", shapes[0][0])
			namespace.setdefault("N", shapes[1][1])
			namespace.setdefault("K", shapes[0][1])
	if len(tensor_args) >= 1:
		first_shape = tensor_args[0]["shape"]
		if len(first_shape) == 1:
			namespace.setdefault("N", first_shape[0])
			# Common alias used in elementwise kernels
			namespace.setdefault("n_elements", first_shape[0])
		# For any shape, total numel is a common grid variable
		numel = 1
		for d in first_shape:
			numel *= d
		namespace.setdefault("n_elements", numel)

	# triton.cdiv shim
	class _triton:
		@staticmethod
		def cdiv(a, b):
			return math.ceil(a / b)

	namespace["triton"] = _triton()
	namespace["math"] = math

	try:
		# Extract the lambda expression from the source
		idx = source.find("lambda")
		if idx < 0:
			return None
		lambda_src = source[idx:].strip()
		# Strip inline comments that break parsing (e.g. "# noqa E731")
		lines = lambda_src.split("\n")
		cleaned = []
		for line in lines:
			# Remove trailing comments but preserve string content
			comment_idx = line.find("#")
			if comment_idx >= 0:
				line = line[:comment_idx]
			cleaned.append(line)
		lambda_src = "\n".join(cleaned).rstrip()
		# Eval the lambda — namespace must be globals so free variables resolve
		eval_globals = {"__builtins__": {}}
		eval_globals.update(namespace)
		grid_fn = eval(lambda_src, eval_globals)  # noqa: S307
		result = grid_fn(meta)
		if isinstance(result, (tuple, list)):
			return [int(v) for v in result]
		return [int(result)]
	except Exception:
		return None


def _extract_block_sizes_from_source(kernel_source: str, kernel_name: str) -> Dict[str, int]:
	"""Extract block size parameters from kernel call site"""
	import ast
	block_sizes = {}

	# Find kernel call pattern: kernel_name[grid](args...)
	call_pattern = rf'{kernel_name}\s*\[([^\]]+)\]\s*\(([^)]+)\)'
	call_match = re.search(call_pattern, kernel_source, re.DOTALL)

	if not call_match:
		return block_sizes

	args_str = call_match.group(2)

	# Parse kernel signature to get parameter names
	try:
		tree = ast.parse(kernel_source)
	except:
		return block_sizes

	kernel_def = None
	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
			kernel_def = node
			break

	if not kernel_def:
		return block_sizes

	param_names = [arg.arg for arg in kernel_def.args.args]

	# Split arguments carefully (handle nested structures)
	args_parts = []
	paren_depth = 0
	current_arg = []
	for char in args_str:
		if char in '([':
			paren_depth += 1
		elif char in ')]':
			paren_depth -= 1
		elif char == ',' and paren_depth == 0:
			args_parts.append(''.join(current_arg).strip())
			current_arg = []
			continue
		current_arg.append(char)
	if current_arg:
		args_parts.append(''.join(current_arg).strip())

	# Match positional args to parameters
	positional_args = [arg for arg in args_parts if '=' not in arg]
	for i, param_name in enumerate(param_names):
		if i < len(positional_args) and 'BLOCK' in param_name.upper():
			arg_value = positional_args[i].strip()
			if arg_value.isdigit():
				block_sizes[param_name] = int(arg_value)

	# Also parse keyword arguments for block sizes
	keyword_args = [arg for arg in args_parts if '=' in arg]
	for kwarg in keyword_args:
		parts = kwarg.split('=', 1)
		if len(parts) == 2:
			key = parts[0].strip()
			value = parts[1].strip()
			# Check if this is a block size parameter
			if 'BLOCK' in key.upper():
				# Try to parse as integer literal
				if value.isdigit():
					block_sizes[key] = int(value)
				# If it's a variable reference, we can't extract the value from source
				# This will be handled by TTGIR metadata extraction

	return block_sizes


def regenerate_ir_with_shapes(
	operator_name: str,
	kernel_name: str,
	output_dir: Path,
	num_inputs: int = 5,
	device: str = "cuda"
) -> Path:
	"""Regenerate IR and shapes for all inputs"""
	from tritonbench.operators import load_opbench_by_name
	from tritonbench.utils.triton_op import override_args
	from triton.runtime.jit import JITFunction

	print(f"Regenerating IR for operator '{operator_name}', kernel '{kernel_name}'")

	# Clear existing IR files
	if output_dir.exists():
		for f in output_dir.glob("*.ttgir"):
			f.unlink()
		for f in output_dir.glob("*.json"):
			f.unlink()

	output_dir.mkdir(parents=True, exist_ok=True)

	# Prepare mapping
	shape_mapping = {
		"kernel_name": kernel_name,
		"operator": operator_name,
		"inputs": []
	}

	# Run for each input
	for input_id in range(num_inputs):
		print(f"Processing input {input_id}...")

		# Setup arguments
		arg_list = [
			"--op", operator_name,
			"--device", device,
			"--input-id", str(input_id),
			"--num-inputs", "1",
			"--warmup", "0",
			"--rep", "1",
			"--keep-going",
		]

		# Use --only flag to run specific kernel
		benchmark_name = kernel_name.split('.')[-1] if '.' in kernel_name else kernel_name
		arg_list.extend(["--only", benchmark_name])

		# Load operator
		tb_args, extra_args = override_args(arg_list)
		OpClass = load_opbench_by_name(operator_name)
		op = OpClass(tb_args=tb_args, extra_args=extra_args)

		# Reset captured_shapes for each input to avoid overwriting
		captured_shapes = {}
		original_run = JITFunction.run

		def capture_run(self, *args, grid, **kwargs):
			kname = getattr(getattr(self, "fn", None), "__name__", "unknown")
			if kname == kernel_name or kernel_name in kname:
				# Capture tensor and scalar args
				captured_shapes[kname] = {
					"tensor_shapes": [list(arg.shape) for arg in args if hasattr(arg, 'shape')],
					"scalar_args": [arg for arg in args if not hasattr(arg, 'shape')],
					"grid": grid
				}
			return original_run(self, *args, grid=grid, **kwargs)

		JITFunction.run = capture_run

		try:
			# Run operator
			op.run(warmup=0, rep=1)

			# Dump IR
			fn = op._get_bm_func(benchmark_name)
			op.dump_ir(input_id=input_id, fn=fn, output_dir=output_dir)

			# Save shape info
			if captured_shapes:
				shape_file = output_dir / f"{operator_name}_{kernel_name}_shapes_{input_id}.json"
				with open(shape_file, 'w') as f:
					json.dump(captured_shapes, f, indent=2)

				# Add to mapping
				kname = list(captured_shapes.keys())[0]
				tensor_shapes = captured_shapes[kname]["tensor_shapes"]
				shape_hash = "x".join(str(s) for shapes in tensor_shapes for s in shapes) if tensor_shapes else "unknown"

				shape_mapping["inputs"].append({
					"input_id": input_id,
					"shape_hash": shape_hash,
					"ttgir_file": f"{kernel_name}_k0_input_x{input_id}.ttgir",
					"metadata_file": f"{kernel_name}_k0_input_x{input_id}.json",
					"shape_file": shape_file.name,
					"tensor_shapes": tensor_shapes,
					"scalar_args": captured_shapes[kname]["scalar_args"]
				})

		except Exception as e:
			print(f"Warning: Failed to process input {input_id}: {e}")
		finally:
			JITFunction.run = original_run

	# Save mapping
	mapping_file = output_dir / "shape_mapping.json"
	with open(mapping_file, 'w') as f:
		json.dump(shape_mapping, f, indent=2)

	print(f"Generated IR for {len(shape_mapping['inputs'])} inputs")
	print(f"Mapping saved to: {mapping_file}")

	return mapping_file


def analyze_kernel_from_cache(cache_dir: Path, kernel_name: str, torch_model: str,
							  input_shapes_json: Path,
							  target_gpu: str = "cuda:70",
							  triton_modules_dir: Optional[Path] = None,
							  use_cached_response: bool = False,
							  cached_response_path: Optional[Path] = None,
							  model: str = "deepseek-chat",
							  prompt_output_dir: Optional[Path] = None,) -> Optional[Dict[str, Any]]:
	"""Analyze a specific kernel from the Triton cache directory"""

	optimizer = ConfigurationOptimizer(use_cached_response=use_cached_response, model=model, prompt_output_dir=prompt_output_dir)

	# Extract input_id from the shapes filename for per-input TTGIR matching
	input_id = _extract_input_id_from_shapes_path(input_shapes_json)

	# Find the kernel's TTGIR file — try exact suffix first, then substring
	ttgir_files = list(cache_dir.rglob(f"*{kernel_name}.ttgir"))
	if not ttgir_files:
		# Fall back to substring match (covers per-input files like _k0_input_x5.ttgir)
		ttgir_files = list(cache_dir.rglob(f"*{kernel_name}*.ttgir"))

	if not ttgir_files:
		raise ValueError(f"Error: Could not find TTGIR file for kernel '{kernel_name}' with source '{torch_model}'")

	ttgir_path = _select_ttgir_for_input(ttgir_files, torch_model, input_id)

	if not ttgir_path:
		raise ValueError(f"Error: Could not find TTGIR file for kernel '{kernel_name}' with source '{torch_model}'")

	print(f"Matched TTGIR file: {ttgir_path} for kernel source: {torch_model}"
		  + (f" (input_id={input_id})" if input_id is not None else ""))

	metadata_path = ttgir_path.with_suffix('.json')

	if not metadata_path.exists():
		print(f"Metadata file not found: {metadata_path}. Creating fake V100 metadata.")
		metadata = {
			"name": kernel_name,
			"arch": target_gpu or "cuda:70",
			"shared": 0,
		}
	else:
		# Load metadata
		with open(metadata_path, 'r') as f:
			metadata = json.load(f)


	print(f"Analyzing kernel: {kernel_name}")
	print(f"Triton model source: {torch_model}")
	print(f"TTGIR: {ttgir_path}")
	# print(f"Metadata: {metadata_path}")
	if triton_modules_dir:
		print(f"Triton modules dir: {triton_modules_dir}")
	if use_cached_response and cached_response_path:
		print(f"Using cached response: {cached_response_path}")
	print()

	# Create optimizer and analyze
	result = optimizer.analyze_and_optimize(
		ttgir_path, torch_model, metadata, input_shapes_json, target_gpu, triton_modules_dir, cached_response_path
	)

	return result


def optimize_kernel_from_cache(
	cache_dir: Path,
	kernel_name: str,
	torch_model: str,
	input_shapes_json: Path,
	operator_name: Optional[str] = None,
	target_gpu: str = "cuda:70",
	triton_modules_dir: Optional[Path] = None,
	llm_provider: str = "deepseek",
	llm_model: Optional[str] = None,
	temperature: float = 0.2,
	max_iterations: int = 1,
	num_candidates: int = 3,
	optimization_target: str = "throughput",
	debug_mode: bool = False,
	debug_dir: Optional[Path] = None,
	use_cached_prompts: bool = False,
	cached_prompts_dir: Optional[Path] = None,
	use_profiler: bool = False,
	skip_resource_budget: bool = False,
	skip_data_movement: bool = False,
	include_agent_proposals: bool = False,
	disable_negotiation: bool = False,
) -> Optional[Dict[str, Any]]:
	"""
	Optimize a kernel using the LangGraph multi-agent system from cache directory.

	This is the LangGraph equivalent of analyze_kernel_from_cache. It extracts all
	necessary information from the cache and uses the multi-agent optimization system.

	Args:
		cache_dir: Path to Triton cache directory containing TTGIR files
		kernel_name: Name of the kernel function
		torch_model: Name of the source Triton model file (without .py extension)
		input_shapes_json: Path to *_shapes.json file with captured tensor shapes
		target_gpu: Target GPU architecture (e.g., "cuda:70" for V100)
		triton_modules_dir: Path to triton_modules directory (optional, for source code)
		llm_provider: LLM provider ("deepseek", "openai", "ollama", "vllm")
		llm_model: Model name (default depends on provider)
		temperature: LLM sampling temperature
		max_iterations: Max refinement iterations for multi-agent system
		optimization_target: "throughput", "latency", or "energy"
		debug_mode: Enable detailed logging of prompts and responses
		debug_dir: Directory to save debug artifacts
		use_cached_prompts: If True, replay responses from a previous debug run
		cached_prompts_dir: Path to debug run directory with cached response files
		use_profiler: If True, enable NCU profiling for detailed hardware metrics
			in the feedback loop (requires ncu on PATH)
		skip_resource_budget: Ablation A1a — remove ResourceBudgetAgent
		skip_data_movement: Ablation A1b — remove DataMovementAgent
		include_agent_proposals: Inject each agent's preferred config as
			benchmark candidates alongside orchestrator-synthesised ones
		disable_negotiation: Skip negotiation sub-loop

	Returns:
		Dictionary with optimization results from multi-agent system
	"""
	try:
		from agent_system_langgraph import GPUOptimizationOrchestrator
	except ImportError:
		raise ImportError(
			"agent_system_langgraph package not found. "
			"Make sure it's installed or add the parent directory to PYTHONPATH."
		)

	# Check if shape.json exists
	input_id = None
	if not input_shapes_json or not input_shapes_json.exists():
		print(f"ERROR: Shape file not found: {input_shapes_json}")
		if not operator_name:
			raise RuntimeError(
				f"Shape file not found: {input_shapes_json}. "
				f"Cannot regenerate without operator_name."
			)
		print("Attempting to regenerate IR with shapes...")

		# Regenerate using provided operator_name
		mapping_file = regenerate_ir_with_shapes(
			operator_name=operator_name,
			kernel_name=kernel_name,
			output_dir=cache_dir,
			num_inputs=5,
			device="cuda"
		)

		# Load mapping and use first input
		with open(mapping_file) as f:
			shape_mapping = json.load(f)

		if not shape_mapping["inputs"]:
			raise RuntimeError("Failed to generate any IR files")

		# Use first input as default
		first_input = shape_mapping["inputs"][0]
		ttgir_path = cache_dir / first_input["ttgir_file"]
		input_shapes_json = cache_dir / first_input["shape_file"]

		print(f"Using input_id={first_input['input_id']} with shape {first_input['shape_hash']}")

		# Skip the TTGIR search below since we already have the path
		metadata_path = ttgir_path.with_suffix('.json')
		if not metadata_path.exists():
			raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
		with open(metadata_path, 'r') as f:
			metadata = json.load(f)
	else:
		# Extract input_id from the shapes filename so we can select the
		# corresponding TTGIR (compiled with matching constexpr values).
		input_id = _extract_input_id_from_shapes_path(input_shapes_json)
		if input_id is not None:
			print(f"Detected input_id={input_id} from shapes file: {input_shapes_json.name}")

		# Find the kernel's TTGIR file matching both kernel_name and torch_model
		# Try both full kernel name and without class prefix (e.g., "Operator.softmax_kernel" -> "softmax_kernel")
		kernel_search_names = [kernel_name, kernel_name.split('.')[-1]]
		ttgir_files = []
		for search_name in kernel_search_names:
			ttgir_files = list(cache_dir.rglob(f"*{search_name}*.ttgir"))
			if ttgir_files:
				break

		if not ttgir_files:
			raise ValueError(f"Error: Could not find TTGIR file for kernel '{kernel_name}'")

		ttgir_path = _select_ttgir_for_input(ttgir_files, torch_model, input_id)

		if not ttgir_path:
			raise ValueError(
				f"Error: Could not find TTGIR file for kernel '{kernel_name}' "
				f"with source '{torch_model}'"
			)

		print(f"Matched TTGIR file: {ttgir_path} for kernel source: {torch_model}"
			  + (f" (input_id={input_id})" if input_id is not None else ""))

		metadata_path = ttgir_path.with_suffix('.json')

		if not metadata_path.exists():
			print(f"Metadata file not found: {metadata_path}. Creating fake metadata with target GPU defaults.")
			metadata = {
				"name": kernel_name,
				"arch": target_gpu or "cuda:70",
				"shared": 0,
			}
		else:
			# Load metadata
			with open(metadata_path, 'r') as f:
				metadata = json.load(f)

	print(f"\n{'='*80}")
	print(f"Optimizing kernel with Multi-Agent System: {kernel_name}")
	print(f"{'='*80}")
	print(f"Triton model source: {torch_model}")
	print(f"TTGIR: {ttgir_path}")
	print(f"Metadata: {metadata_path}")
	print(f"Input shapes: {input_shapes_json}")
	print(f"Target GPU: {target_gpu}")
	print(f"LLM Provider: {llm_provider}")
	if triton_modules_dir:
		print(f"Triton modules dir: {triton_modules_dir}")

	# Parse TTGIR
	ttgir_analysis = TTGIRParser.parse_ttgir_file(ttgir_path)

	# Update analysis with metadata
	ttgir_analysis.shared_memory_bytes = metadata.get('shared', 0)
	ttgir_analysis.compute_capability = metadata.get('arch', '')

	# Read TTGIR content
	with open(ttgir_path, 'r') as f:
		ttgir_content = f.read()

	# Read TTIR content (original Triton IR)
	ttir_path = ttgir_path.with_suffix('.ttir')
	ttir_content = None
	if ttir_path.exists():
		with open(ttir_path, 'r') as f:
			ttir_content = f.read()

	# Extract source code
	source_file, line_range = ConfigurationOptimizer._extract_source_location(ttgir_content)
	triton_source = None
	if source_file:
		print(f"Found source location: {source_file}, lines {line_range}")
		triton_source = ConfigurationOptimizer._extract_source_from_file(source_file, line_range)
	# Load tensor shapes
	tensor_shapes = {}
	tensor_dtypes = {}
	if input_shapes_json and input_shapes_json.exists():
		try:
			shapes_info = load_tensor_shapes(input_shapes_json)
			print(f"Loaded tensor shapes from: {input_shapes_json}")

			# Extract tensor shapes and dtypes
			if "tensor_args" in shapes_info:
				for arg in shapes_info["tensor_args"]:
					arg_idx = arg.get("arg_index")
					shape = arg.get("shape")
					dtype = arg.get("dtype")
					if arg_idx is not None:
						arg_name = f"arg{arg_idx}"
						if shape:
							tensor_shapes[arg_name] = shape
						if dtype:
							tensor_dtypes[arg_name] = dtype

			print(f"Extracted {len(tensor_shapes)} tensor shapes")
			print(f"Extracted {len(tensor_dtypes)} tensor dtypes")
		except Exception as e:
			print(f"Warning: Could not load tensor shapes from {input_shapes_json}: {e}")

	# Extract constexpr block param names from two sources and merge:
	# 1. Shapes JSON kwargs (runtime-captured, e.g. XBLOCK, RBLOCK)
	# 2. Kernel source tl.constexpr declarations (e.g. BLOCK_M, BLOCK_K, RBLOCK)
	# Source constexprs catch reduction dims (BLOCK_K, RBLOCK) that kwargs may miss.
	_kwargs_block_names = []
	_kwargs_block_vals = {}  # name -> int value from runtime kwargs
	if input_shapes_json and input_shapes_json.exists():
		try:
			shapes_info = load_tensor_shapes(input_shapes_json)
			kwargs = shapes_info.get("kwargs", {})
			for k in kwargs:
				if "BLOCK" in k.upper():
					_kwargs_block_names.append(k)
					v = kwargs[k]
					_kwargs_block_vals[k] = v.get("value") if isinstance(v, dict) else v
			if _kwargs_block_names:
				print(f"Found constexpr block params from shapes kwargs: {_kwargs_block_names}")
		except Exception as e:
			print(f"Warning: Could not extract constexpr param names: {e}")

	_src_block_names = []
	if triton_source:
		_src_constexprs = re.findall(
			r'(\w+)\s*:\s*tl\.constexpr\s*=\s*(\d+)', triton_source
		)
		_src_block_names = [
			name for name, _ in _src_constexprs
			if "BLOCK" in name.upper()
		]
		if _src_block_names:
			print(f"Found constexpr block params from source: {_src_block_names}")

	# Merge: source constexprs are authoritative (they define what the kernel
	# actually accepts).  Add any kwargs-only names that source doesn't have.
	if _src_block_names:
		constexpr_param_names = list(_src_block_names)
		for name in _kwargs_block_names:
			if name not in constexpr_param_names:
				constexpr_param_names.append(name)
	else:
		constexpr_param_names = _kwargs_block_names

	# Extract num_warps/num_stages from shapes kwargs as fallback
	# (inductor kernels store these in kwargs, not in top-level metadata)
	_shapes_kwargs = {}
	if input_shapes_json and input_shapes_json.exists():
		try:
			_si = load_tensor_shapes(input_shapes_json)
			_shapes_kwargs = _si.get("kwargs", {})
		except Exception:
			pass

	def _kwarg_val(key, default):
		"""Extract scalar value from shapes kwargs dict."""
		v = _shapes_kwargs.get(key)
		if isinstance(v, dict) and "value" in v:
			return v["value"]
		return default

	# Build current configuration
	current_config = {
		"num_warps": metadata.get('num_warps') or ttgir_analysis.num_warps or _kwarg_val("num_warps", 4),
		"num_stages": metadata.get('num_stages') or _kwarg_val("num_stages", 2),
		"num_ctas": metadata.get('num_ctas', ttgir_analysis.num_ctas),
		"block_size": {},  # Will be extracted from source if available
		"grid": {"x": 1, "y": 1, "z": 1},  # Will be extracted from captured invocation if available
		"constexpr_param_names": constexpr_param_names,
		# "shared_memory_bytes": metadata.get('shared', ttgir_analysis.shared_memory_bytes),
		# "global_loads": ttgir_analysis.global_loads,
		# "global_stores": ttgir_analysis.global_stores,
	}

	# Build block_size: start with TTGIR, then override/augment with source constexprs.
	# TTGIR uses generic names (XBLOCK, YBLOCK, BLOCK_SIZE); source constexprs use
	# the kernel's actual names (BLOCK_M, BLOCK_K, RBLOCK).  When source constexprs
	# exist, they replace the TTGIR values entirely (they are more accurate).
	block_sizes = ConfigurationOptimizer._extract_block_sizes_from_ttgir(ttgir_content)
	if block_sizes:
		print(f"Extracted block sizes from TTGIR: {block_sizes}")

	if _src_block_names:
		# Source constexprs are authoritative — replace TTGIR generic names
		_src_vals = dict(re.findall(
			r'(\w+)\s*:\s*tl\.constexpr\s*=\s*(\d+)', triton_source
		))
		_merged = {
			name: int(_src_vals[name]) for name in _src_block_names if name in _src_vals
		}
		# Also include kwargs-only params not found in source (e.g. XBLOCK=1)
		for name in _kwargs_block_names:
			if name not in _merged and name in _kwargs_block_vals:
				_merged[name] = _kwargs_block_vals[name]
		current_config['block_size'] = _merged
		print(f"Overrode with source constexprs: {current_config['block_size']}")
	elif _kwargs_block_names and block_sizes:
		# Kwargs provide the runtime param names (e.g. XBLOCK, RBLOCK) but
		# TTGIR uses generic names (BLOCK_SIZE).  Use kwargs values directly
		# since they were captured at runtime and are authoritative.
		# For names not in kwargs (shouldn't happen), fall back to TTGIR.
		_mapped = {}
		for name in _kwargs_block_names:
			if name in _kwargs_block_vals:
				_mapped[name] = _kwargs_block_vals[name]
		current_config['block_size'] = _mapped
		print(f"Mapped kwargs block values: {_mapped}")
	elif block_sizes:
		current_config['block_size'] = block_sizes

	# Extract grid from captured kernel invocation
	if input_shapes_json and input_shapes_json.exists():
		try:
			shapes_info = load_tensor_shapes(input_shapes_json)
			if "grid" in shapes_info and shapes_info["grid"]:
				raw_grid = shapes_info["grid"]
				grid_list = None

				if isinstance(raw_grid, list):
					grid_list = raw_grid
				elif isinstance(raw_grid, dict) and raw_grid.get("type") == "callable":
					# Store the formula source for agent prompts
					current_config["grid_formula"] = raw_grid.get("source", "")
					# Try to evaluate the callable grid
					grid_list = _evaluate_callable_grid(raw_grid, shapes_info)
					if grid_list:
						print(f"Evaluated callable grid: {grid_list}")
					else:
						print(f"Could not evaluate callable grid; formula stored for prompts")

				if grid_list:
					if len(grid_list) >= 1:
						current_config['grid']['x'] = grid_list[0]
					if len(grid_list) >= 2:
						current_config['grid']['y'] = grid_list[1]
					if len(grid_list) >= 3:
						current_config['grid']['z'] = grid_list[2]
					print(f"Extracted grid from captured invocation: {grid_list}")
		except Exception as e:
			print(f"Warning: Could not extract grid: {e}")

	# Optional override: for kernels shipping with @triton.autotune, use the
	# first entry of the decorator's config list as the "original" baseline
	# (keeps block_size/num_warps/num_stages; preserves grid / constexpr info).
	if os.environ.get("BASELINE_USE_AUTOTUNE_FIRST") == "1":
		override = _autotune_first_config_for_kernel(
			kernel_name.split('.')[-1],
			source_file=source_file,
		)
		if override:
			print(
				"[BASELINE_USE_AUTOTUNE_FIRST=1] replacing current_config "
				"with first @triton.autotune entry for "
				f"{kernel_name}: {override}"
			)
			for k, v in override.items():
				current_config[k] = v

	print(f"\nCurrent configuration:")
	print(json.dumps(current_config, indent=2))

	# Build benchmark config for closed-loop verification
	benchmark_config = None
	if source_file and input_shapes_json and input_shapes_json.exists():
		# Parse kernel class from kernel_name if present (e.g. "Operator.softmax_kernel")
		if "." in kernel_name:
			bench_kernel_class, bench_kernel_name = kernel_name.rsplit(".", 1)
		else:
			bench_kernel_class, bench_kernel_name = None, kernel_name

		# Prefer per-input .py file that matches the shapes file's input index.
		# This ensures the kernel source (with hardcoded xnumel/rnumel) matches
		# the tensor sizes in the shapes file.
		bench_source_file = source_file
		bench_shapes_file = str(input_shapes_json)
		resolved_source = _select_source_for_input(
			Path(source_file), bench_kernel_name, input_id,
		)
		if str(resolved_source) != source_file:
			bench_source_file = str(resolved_source)
			print(f"Using per-input kernel source: {resolved_source.name}")
		else:
			# Fell back to the base source. If it's an Inductor template
			# kernel, the baked M/N/K/strides won't match this shape and the
			# benchmark will silently produce OOB writes — fail loud.
			try:
				base_src = Path(source_file).read_text()
			except (OSError, IOError):
				base_src = ""
			if "triton_heuristics.template(" in base_src:
				raise RuntimeError(
					f"Per-shape source missing for template kernel "
					f"{bench_kernel_name} (input_id={input_id}, base={source_file}); "
					f"rerun extract/run_inductor.py to regenerate per-input captures."
				)
			# Non-template fallback: find the shapes file that matches the
			# kernel's hardcoded dimensions (legacy handwritten kernels).
			if input_id is not None:
				try:
					hardcoded = {}
					for name in ("xnumel", "rnumel", "M", "N", "K"):
						m = re.search(rf'^\s+{name}\s*=\s*(\d+)', base_src, re.MULTILINE)
						if m:
							hardcoded[name] = int(m.group(1))
					if hardcoded:
						shapes_dir = Path(source_file).parent
						op_name = shapes_dir.name
						for sf in sorted(shapes_dir.glob(f"{op_name}_{bench_kernel_name}_input*_shapes.json")):
							sd = json.load(open(sf))
							scalar_vals = [s["value"] for s in sorted(sd.get("scalar_args", []), key=lambda x: x["arg_index"])]
							tensor_dims = [dim for t in sd.get("tensor_args", []) for dim in t.get("shape", [])]
							all_vals = scalar_vals + tensor_dims
							if all(v in all_vals for v in hardcoded.values()):
								bench_shapes_file = str(sf)
								print(f"Matched shapes to kernel dims {hardcoded}: {sf.name}")
								break
				except Exception as e:
					print(f"Warning: shapes matching failed: {e}")

		benchmark_config = {
			"kernel_module": bench_source_file,
			"kernel_name": bench_kernel_name,
			"kernel_class": bench_kernel_class,
			"shapes_file": bench_shapes_file,
			"device": "cuda",
			"warmup": 5,
			"rep": 20,
		}
		print(f"Closed-loop benchmarking enabled (kernel={bench_kernel_name}, class={bench_kernel_class})")
	else:
		print("Closed-loop benchmarking disabled (missing source_file or shapes)")

	# Build optional profiler function for NCU integration
	profiler_fn = None
	if use_profiler:
		try:
			from ncu_profiler import profile_with_ncu, is_ncu_available
			if is_ncu_available():
				from agent_system_langgraph.knowledge_base import get_architecture
				arch = get_architecture(target_gpu)
				peak_bw = arch.memory_bandwidth_gb_s

				def _profiler_fn(**kw):
					return profile_with_ncu(**kw, peak_memory_bandwidth_gb_s=peak_bw)

				profiler_fn = _profiler_fn
				print("NCU profiling enabled (ncu found on PATH)")
			else:
				print("WARNING: --enable-profiling requested but 'ncu' not found on PATH. Continuing without profiling.")
		except ImportError as e:
			print(f"WARNING: NCU profiler module not available: {e}. Continuing without profiling.")

	# Create orchestrator
	orchestrator = GPUOptimizationOrchestrator(
		llm_provider=llm_provider,
		llm_model=llm_model,
		architecture_code=target_gpu,
		temperature=temperature,
		max_iterations=max_iterations,
		num_candidates=num_candidates,
		debug_mode=debug_mode,
		debug_dir=str(debug_dir) if debug_dir else None,
		use_cached_prompts=use_cached_prompts,
		cached_prompts_dir=str(cached_prompts_dir) if cached_prompts_dir else None,
		benchmark_fn=benchmark_config_pair if benchmark_config else None,
		profiler_fn=profiler_fn,
		skip_resource_budget=skip_resource_budget,
		skip_data_movement=skip_data_movement,
		include_agent_proposals=include_agent_proposals,
		disable_negotiation=disable_negotiation,
	)

	# Run optimization
	print(f"\n{'='*80}")
	print("Running Multi-Agent Optimization...")
	print(f"{'='*80}\n")

	result = orchestrator.optimize_from_ttgir(
		kernel_name=kernel_name,
		ttgir=ttgir_content,
		current_config=current_config,
		tensor_shapes=tensor_shapes,
		tensor_dtypes=tensor_dtypes,
		ttir=ttir_content,
		source_code=triton_source,
		optimization_target=optimization_target,
		benchmark_config=benchmark_config,
	)

	# Add metadata to result for consistency with analyze_kernel_from_cache
	if result.get("success"):
		result["kernel_name"] = kernel_name
		result["torch_model"] = torch_model
		result["target_architecture"] = target_gpu
		result["current_config"] = current_config
		result["tensor_shapes"] = tensor_shapes
		result["tensor_dtypes"] = tensor_dtypes

	return result


def extract_best_config(analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	"""
	Extract the best recommended configuration from LLM analysis result.

	Works with both ConfigurationOptimizer results and multi-agent system results.

	Args:
		analysis_result: The result dictionary from analyze_and_optimize or optimize_kernel_from_cache

	Returns:
		Dictionary with the best config parameters, or None if not found
	"""
	try:
		# Check if this is a multi-agent result (has "result" key with OptimizationResult)
		if "result" in analysis_result and isinstance(analysis_result["result"], dict):
			opt_result = analysis_result["result"]
			recommended = opt_result.get("recommended_config", {})

			if not recommended:
				print("Warning: No recommended configuration found in multi-agent result")
				return None

			# Extract from multi-agent result
			config_params = {
				"num_warps": recommended.get("num_warps"),
				"block_size": recommended.get("block_size", {}),
				"num_stages": recommended.get("num_stages", 2),
				"num_ctas": recommended.get("num_ctas", 1),
				"eviction_policy": recommended.get("eviction_policy", "evict_last"),
				"expected_improvement": opt_result.get("expected_improvement", "unknown"),
				"reasoning": opt_result.get("coordinator_reasoning", ""),
				"confidence_score": opt_result.get("confidence_score", 0.0),
				"risk_assessment": opt_result.get("risk_assessment", "unknown"),
			}

			print(f"\n{'='*80}")
			print("Best Recommended Configuration (Multi-Agent):")
			print(f"  - num_warps: {config_params['num_warps']}")
			print(f"  - block_size: {config_params['block_size']}")
			print(f"  - num_stages: {config_params['num_stages']}")
			print(f"  - num_ctas: {config_params['num_ctas']}")
			print(f"  - eviction_policy: {config_params['eviction_policy']}")
			print(f"  - Expected improvement: {config_params['expected_improvement']}")
			print(f"  - Confidence: {config_params['confidence_score']:.1%}")
			print(f"  - Risk: {config_params['risk_assessment']}")
			print(f"  - Reasoning: {config_params['reasoning'][:200]}...")
			print(f"{'='*80}\n")

			return config_params

		# Otherwise, assume it's a ConfigurationOptimizer result
		recommendations = analysis_result.get("recommendations", {})
		configs = recommendations.get("recommended_configurations", [])

		if not configs:
			print("Warning: No recommended configurations found in analysis result")
			return None

		# Get the first recommended config (typically the best one)
		best_config = configs[0]

		# Extract key parameters
		config_params = {
			"num_warps": best_config.get("num_warps"),
			"block_size": best_config.get("block_size", {}),
			"num_stages": best_config.get("num_stages", 2),
			"eviction_policy": best_config.get("eviction_policy", "evict_last"),
			"expected_improvement": best_config.get("expected_improvement", "unknown"),
			"reasoning": best_config.get("reasoning", ""),
			"includes_current": best_config.get("includes_current", False)
		}

		print(f"\n{'='*80}")
		print("Best Recommended Configuration:")
		print(f"  - num_warps: {config_params['num_warps']}")
		print(f"  - block_size: {config_params['block_size']}")
		print(f"  - num_stages: {config_params['num_stages']}")
		print(f"  - eviction_policy: {config_params['eviction_policy']}")
		print(f"  - Expected improvement: {config_params['expected_improvement']}")
		print(f"  - Reasoning: {config_params['reasoning']}")
		print(f"{'='*80}\n")

		return config_params

	except Exception as e:
		print(f"Error extracting best config: {e}")
		return None



# REMOVED: benchmark_baseline_config() function (lines 386-706)
#
# Reason: Critical bug in regex parser at line 509 that failed with nested parentheses.
# The pattern ([^)]+) stopped at the first ) instead of matching the full argument list
# when kernel calls contained nested parentheses like: kernel[grid](arg1, func(x, y), num_warps=4)
#
# Replacement: Use benchmark_baseline.py which orchestrates run.py directly.
# This provides a more reliable way to benchmark baseline configurations using
# TritonBench's official benchmarking infrastructure.
#
# Migration:
#   Old: baseline_result = benchmark_baseline_config(kernel_name, kernel_path, ...)
#   New: 1. Run: python benchmark_baseline.py --op <operator> --output baseline.csv
#        2. Load CSV and find kernel results
#
# See benchmark_baseline.py and compare_configs.py for the new workflow.



def test_recommended_config(
	kernel_name: str,
	kernel_path: Path,
	input_shapes_json: Path,
	config: Dict[str, Any],
	device: str = "cuda",
	num_iterations: int = 100
) -> Dict[str, Any]:
	"""
	Test the recommended configuration by executing the kernel with actual inputs.

	Uses captured tensor shape information from capture_all_kernel_ir.py to recreate
	the exact input scenario for accurate benchmarking.

	Args:
		kernel_path: Path to the Triton kernel Python file
		input_shapes_json: Path to the *_shapes.json file from capture_all_kernel_ir.py
		config: Configuration dictionary from extract_best_config
		device: Device to run on ("cuda", "xpu", or "cpu")
		num_iterations: Number of iterations for benchmarking

	Returns:
		Dictionary with benchmark results including execution time and throughput
	"""
	print(f"\n{'='*80}")
	print(f"Testing Recommended Configuration on {device}")
	print(f"{'='*80}")

	try:
		# Check device availability
		if device == "cuda" and not torch.cuda.is_available():
			print("CUDA not available, falling back to CPU")
			device = "cpu"
		elif device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
			print("XPU not available, falling back to CPU")
			device = "cpu"

		# Load and display tensor shape information
		print(f"\nLoading tensor shapes from: {input_shapes_json}")
		if not input_shapes_json.exists():
			raise FileNotFoundError(f"Tensor shapes file not found: {input_shapes_json}")

		print("\n" + "-"*80)
		print("Captured Tensor Information:")
		print("-"*80)
		print_tensor_info(input_shapes_json)
		print("-"*80)

		# Recreate kernel inputs from captured shapes
		print(f"\nRecreating kernel inputs with random values...")
		input_args, input_kwargs = create_kernel_inputs(
			input_shapes_json,
			device=device,
			values='random',
			seed=42  # Fixed seed for reproducibility
		)
		print(f"Created {len(input_args)} positional arguments")
		if input_kwargs:
			print(f"Created {len(input_kwargs)} keyword arguments")

		# Display argument details
		print("\nArgument details:")
		for i, arg in enumerate(input_args):
			if isinstance(arg, torch.Tensor):
				print(f"  arg[{i}]: Tensor(shape={tuple(arg.shape)}, dtype={arg.dtype}, device={arg.device})")
			else:
				print(f"  arg[{i}]: {type(arg).__name__}({arg})")

		if input_kwargs:
			print("\nKeyword argument details:")
			# Filter out any warmup entries from input_kwargs if they exist
			input_kwargs = {k: v for k, v in input_kwargs.items() if 'warmup' not in k.lower()}

			for key, value in input_kwargs.items():
				if isinstance(value, torch.Tensor):
					print(f"  {key}: Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})")
				else:
					print(f"  {key}: {type(value).__name__}({value})")


		# Dynamically import the kernel module
		print(f"\nLoading kernel from: {kernel_path}")
		if not kernel_path.exists():
			raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

		spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
		kernel_module = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(kernel_module)

		# Find the Triton JIT kernel function
		kernel_func = None
		if hasattr(kernel_module, kernel_name):
			kernel_func = getattr(kernel_module, kernel_name)
			print(f"Found kernel function: {kernel_name}")

		if kernel_func is None:
			# List available functions for debugging
			available_funcs = [name for name in dir(kernel_module) if not name.startswith('_')]
			raise ValueError(f"Could not find kernel function. Available: {available_funcs}")

		# Check if this is a Triton JIT kernel
		try:
			import triton
			from triton.runtime.jit import JITFunction
			is_triton_kernel = isinstance(kernel_func, JITFunction)
		except ImportError:
			print("Warning: Triton not available, assuming regular Python function")
			is_triton_kernel = False

		print(f"\nKernel type: {'Triton JIT kernel' if is_triton_kernel else 'Regular Python function'}")

		# Compute new grid based on recommended config and input tensor shapes
		print(f"\nComputing grid configuration based on new block sizes...")

		# Load metadata to get numel values
		metadata = load_tensor_shapes(input_shapes_json)

		# Try to extract numel values from the kernel source code
		numel_dict = {}
		try:
			# Read kernel source code
			with open(kernel_path, 'r') as f:
				kernel_source = f.read()

			# Use infer_numel_for_kernel from tune_one_file.py to extract numel values
			import ast
			numel_param_names, numel_values, other_param_names = infer_numel_for_kernel(kernel_source, kernel_name)

			if numel_param_names and numel_values:
				numel_dict = dict(zip(numel_param_names, numel_values))
				print(f"Extracted numel values from kernel source: {numel_dict}")
		except Exception as e:
			raise ValueError()

		if not numel_dict:
			raise ValueError("Cannot extract numel values from source, metadata, or tensors")

		# Get block sizes from recommended config
		block_size = config.get('block_size', {})
		XBLOCK = block_size.get('XBLOCK', None)
		YBLOCK = block_size.get('YBLOCK', None)
		R0_BLOCK = block_size.get('R0_BLOCK', None)

		if XBLOCK is None:
			raise ValueError("XBLOCK not found in recommended configuration")

		print(f"Recommended block sizes: XBLOCK={XBLOCK}, YBLOCK={YBLOCK}, R0_BLOCK={R0_BLOCK}")

		# Build block config for grid computation
		block_config = {'XBLOCK': XBLOCK}
		if YBLOCK is not None:
			block_config['YBLOCK'] = YBLOCK
		if R0_BLOCK is not None:
			block_config['R0_BLOCK'] = R0_BLOCK

		# Compute grid using utility function
		try:
			grid = compute_grid_from_numel(numel_dict, block_config)
			print(f"Computed grid: {grid} from numel={numel_dict}, block_config={block_config}")
		except ValueError as e:
			raise ValueError(f"Failed to compute grid: {e}")

		# Apply configuration to kernel if it's a Triton JIT kernel
		if is_triton_kernel:
			print(f"\nApplying Triton kernel configuration:")
			num_warps = config.get('num_warps', None)
			num_stages = config.get('num_stages', None)
			eviction_policy = config.get('eviction_policy', None)

			print(f"  num_warps: {num_warps}")
			print(f"  num_stages: {num_stages}")
			print(f"  eviction_policy: {eviction_policy}")
			print(f"  block_size: {config.get('block_size')}")
			print(f"  grid: {grid}")

			# Parse kernel signature to understand parameter order
			# Read kernel source to get exact parameter ordering
			print(f"\nParsing kernel signature to understand parameter order...")
			with open(kernel_path, 'r') as f:
				kernel_source = f.read()

			# Extract kernel function signature
			import ast
			import inspect

			tree = ast.parse(kernel_source)
			kernel_def = None
			for node in ast.walk(tree):
				if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
					kernel_def = node
					break

			if not kernel_def:
				raise ValueError(f"Could not find kernel definition for {kernel_name}")

			# Get parameter names from kernel signature
			param_names = [arg.arg for arg in kernel_def.args.args]
			print(f"  Kernel signature parameters: {param_names}")

			# Identify which parameters are tensors (pointer types) vs scalars
			# Tensors: parameters that don't end with 'numel' and aren't block sizes
			# Scalars: numel parameters (xnumel, ynumel, etc.) and block sizes (XBLOCK, YBLOCK, etc.)

			# Count how many tensor pointers we have in input_args
			num_tensor_args = sum(1 for arg in input_args if isinstance(arg, torch.Tensor))
			print(f"  Number of tensor arguments: {num_tensor_args}")

			# Build mapping of parameter names to expected values
			param_value_map = {}

			# Add numel values
			for numel_name, numel_value in numel_dict.items():
				param_value_map[numel_name] = numel_value

			# Add block size values (these may be passed as positional args, not kwargs!)
			if XBLOCK is not None:
				param_value_map['XBLOCK'] = XBLOCK
			if YBLOCK is not None:
				param_value_map['YBLOCK'] = YBLOCK
			if R0_BLOCK is not None:
				param_value_map['R0_BLOCK'] = R0_BLOCK

			print(f"  Parameter value map: {param_value_map}")

			# Reconstruct input_args with correct values
			updated_args = []

			for i, param_name in enumerate(param_names):
				if i < len(input_args):
					original_arg = input_args[i]

					# If it's a tensor, keep it as-is
					if isinstance(original_arg, torch.Tensor):
						updated_args.append(original_arg)
						print(f"  [{i}] {param_name}: Tensor {tuple(original_arg.shape)} (unchanged)")
					# If it's a scalar parameter we have a new value for, update it
					elif param_name in param_value_map:
						new_value = param_value_map[param_name]
						updated_args.append(new_value)
						print(f"  [{i}] {param_name}: {original_arg} -> {new_value}")
					# Otherwise keep original scalar value
					else:
						updated_args.append(original_arg)
						print(f"  [{i}] {param_name}: {original_arg} (unchanged)")
				else:
					# If we don't have enough args, this might be a constexpr that should be in kwargs
					if param_name in param_value_map:
						# Don't add to args, will be in kwargs
						print(f"  [{i}] {param_name}: Will be in kwargs")
						break

			input_args = updated_args

			# For Triton kernels, constexpr parameters can be passed either as:
			# 1. Positional arguments (as in the original call)
			# 2. Keyword arguments (more explicit)
			#
			# Since the original call passes them as positional, we already handled them above.
			# Only add to kwargs if they weren't already added to args.
			if input_kwargs is None:
				input_kwargs = {}

			if num_warps:
				input_kwargs['num_warps'] = num_warps
			if num_stages:
				input_kwargs['num_stages'] = num_stages

			# Print all input_kwargs for debugging
			print(f"\nFinal input_kwargs passed to kernel:")
			if input_kwargs:
				for key, value in input_kwargs.items():
					if isinstance(value, torch.Tensor):
						print(f"  {key}: Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})")
					else:
						print(f"  {key}: {type(value).__name__}({value})")
			else:
				print("  No keyword arguments")

			# Add compile-time configuration (num_warps, num_stages) to kwargs
			# These are NOT kernel parameters but Triton JIT compilation options
			# Note: These should NOT be passed to the kernel call itself

		else:
			raise ValueError("Kernel function is not a Triton JIT kernel; cannot apply configuration")

		# Use benchmark_kernel from benchmark_utils for robust multiprocessing-based timing
		print(f"\nBenchmarking with warmup={25} and rep={num_iterations}...")

		# benchmark_kernel returns median time in ms, or float('inf') on failure
		median_time, min_time, max_time, mean_time, std_dev = benchmark_kernel(
			module_path=kernel_path,
			kernel_name=kernel_name,
			grid=grid,
			args=input_args,
			kwargs=input_kwargs,
			warmup=25,
			rep=num_iterations,
			timeout=60  # 60s timeout for safety
		)

		if median_time == float('inf'):
			raise RuntimeError("Benchmark failed - kernel execution timed out or errored")

		results = {
			"success": True,
			"device": device,
			"num_iterations": num_iterations,
			"execution_time_ms": {
				"mean": mean_time,
				"median": median_time,
				"min": min_time,
				"max": max_time,
				"std_dev": std_dev
			},
			"config_tested": config,
			"throughput_items_per_sec": 1000.0 / mean_time if mean_time > 0 else 0
		}

		print(f"\n{'='*80}")
		print("Benchmark Results:")
		print(f"  Mean execution time: {mean_time:.4f} ms")
		print(f"  Median execution time: {median_time:.4f} ms")
		print(f"  Min execution time: {min_time:.4f} ms")
		print(f"  Max execution time: {max_time:.4f} ms")
		print(f"  Std deviation: {std_dev:.4f} ms")
		print(f"  Throughput: {results['throughput_items_per_sec']:.2f} calls/sec")
		print(f"{'='*80}\n")

		return results

	except Exception as e:
		print(f"\nError testing configuration: {e}")
		import traceback
		traceback.print_exc()

		return {
			"success": False,
			"error": str(e),
			"device": device,
			"config_tested": config
		}