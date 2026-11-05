#!/usr/bin/env python3
"""
Model-free multi-agent runner.

Discovers Triton kernels from metadata, maps them to TTGIR and input shape
captures, and runs the multi-agent optimizer (or single-agent fallback) per
kernel/shape. Defaults to the first TTGIR and first shapes file per kernel;
`--all-shapes` processes every shapes match. When shapes are missing and
`--auto-capture-shapes` is enabled, this script runs the corresponding
TritonBench operator once with a Triton backend and hooks Triton JIT to capture
the actual runtime tensor/scalar arguments for each launched kernel, storing
them as `*_shapes.json`. Per-shape failures are recorded and the script
continues. A summary JSON is emitted with success/failure counts and best
speedup per kernel when benchmarking is enabled.
"""

import argparse
import contextlib
import json
import logging
import sys
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import yaml
from tritonbench.utils.triton_op import REGISTERED_BENCHMARKS, override_args
from tritonbench.operators import load_opbench_by_name

try:  # optional, only needed when capturing inputs
	from triton.runtime.jit import JITFunction  # type: ignore
except Exception:
	JITFunction = None  # type: ignore

try:  # optional, enrich capture metadata
	import torch  # type: ignore
except Exception:
	torch = None  # type: ignore

from configuration_optimizer import ConfigurationOptimizer
from optimizer_functions import (
	_extract_input_id_from_shapes_path,
	_select_source_for_input,
	analyze_kernel_from_cache,
	# benchmark_baseline_config,
	extract_best_config,
	optimize_kernel_from_cache,
	test_recommended_config,
)

logger = logging.getLogger("multiagent_triton_runner")


@dataclass
class KernelWorkItem:
	name: str
	ttgir: Path
	torch_model: str
	source_path: Path
	shapes: List[Path]

	def source_path_for(self, shape_path: Path) -> Path:
		"""Resolve the per-shape captured source for *shape_path*.

		Returns the per-input snapshot when present; falls back to
		``self.source_path`` for legacy captures and non-template kernels.
		Callers that require per-shape source (template kernels) must check
		and fail when the fallback equals the base path.
		"""
		input_id = _extract_input_id_from_shapes_path(shape_path)
		return _select_source_for_input(self.source_path, self.name, input_id)


@dataclass
class RunResult:
	kernel: str
	shape_path: Path
	success: bool
	error: Optional[str] = None
	payload: Dict = field(default_factory=dict)
	baseline_ms: Optional[float] = None
	optimized_ms: Optional[float] = None
	speedup: Optional[float] = None


def _metadata_dir() -> Path:
	from paths import METADATA_DIR
	return METADATA_DIR


def _iter_metadata_yamls(metadata_dir: Path, extra_yamls: Optional[List[Path]] = None):
	"""Yield (path, parsed_dict) for the main YAML plus any --metadata-extra files."""
	primary = metadata_dir.joinpath("oss_cuda_kernels.yaml")
	if not primary.exists():
		raise FileNotFoundError(f"Metadata file not found: {primary}")
	with open(primary, "r", encoding="utf-8") as fp:
		yield primary, (yaml.safe_load(fp) or {})
	for extra in (extra_yamls or []):
		if not extra.exists():
			print(f"[warn] --metadata-extra path missing, skipping: {extra}")
			continue
		with open(extra, "r", encoding="utf-8") as fp:
			yield extra, (yaml.safe_load(fp) or {})


def _parse_triton_kernels(metadata_dir: Path, extra_yamls: Optional[List[Path]] = None) -> List[str]:
	kernels: List[str] = []
	for _path, metadata in _iter_metadata_yamls(metadata_dir, extra_yamls):
		for _op, variants in metadata.items():
			if not isinstance(variants, dict):
				continue
			for _variant, meta in variants.items():
				if not isinstance(meta, dict):
					continue
				tags = meta.get("tags", []) or []
				if not any("triton" in str(t).lower() or t == "pt2" for t in tags):
					continue
				for k in meta.get("kernels", []) or []:
					kernels.append(str(k))
	return sorted({k for k in kernels if k})


def _kernel_to_op_map(metadata_dir: Path, extra_yamls: Optional[List[Path]] = None) -> Dict[str, str]:
	"""Kernel name -> op name mapping from metadata YAMLs (triton/pt2-tagged)."""
	mapping: Dict[str, str] = {}
	try:
		sources = list(_iter_metadata_yamls(metadata_dir, extra_yamls))
	except FileNotFoundError:
		return {}
	for _path, metadata in sources:
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
	return mapping


def _find_ttgir(cache_dir: Path, kernel_name: str) -> Path:
	# Match kernels whose TTGIR filenames may include suffixes (e.g., _k0_input_x0).
	candidates = sorted(cache_dir.rglob(f"*{kernel_name}*.ttgir"))
	if not candidates:
		raise FileNotFoundError(f"No TTGIR found for kernel '{kernel_name}' under {cache_dir}")
	return candidates[0]


def _extract_source_info(ttgir_path: Path) -> Tuple[str, Path]:
	content = ttgir_path.read_text()
	source_file, _ = ConfigurationOptimizer._extract_source_location(content)
	if source_file:
		return Path(source_file).stem, Path(source_file)
	raise ValueError(f"Could not extract source location for {ttgir_path}")


def _dedup_paths(paths: List[Path]) -> List[Path]:
	"""Deduplicate paths while preserving order."""
	seen: set = set()
	result: List[Path] = []
	for p in paths:
		if p not in seen:
			result.append(p)
			seen.add(p)
	return result


def _filter_shapes_by_kernel(paths: List[Path], kernel_name: str) -> List[Path]:
	"""Filter shapes files to only those whose JSON ``kernel`` field matches *kernel_name*.

	Glob patterns like ``*_triton_dropout_shapes.json`` can accidentally match
	``_seeded_triton_dropout_shapes.json``.  This post-filter reads the ``kernel``
	field from each candidate JSON and keeps only exact matches.
	"""
	# The kernel_name from YAML may have an "Operator." prefix (e.g.
	# "Operator.softmax_kernel") while the JSON "kernel" field stores the
	# bare function name ("softmax_kernel").  Accept either form.
	bare_name = kernel_name.split(".", 1)[-1] if "." in kernel_name else kernel_name
	accept = {kernel_name, bare_name}

	verified: List[Path] = []
	for p in paths:
		try:
			with open(p, "r") as f:
				data = json.load(f)
			if isinstance(data, dict) and data.get("kernel") in accept:
				verified.append(p)
		except (json.JSONDecodeError, OSError):
			pass
	return verified


def _find_shapes(shapes_dir: Path, kernel_name: str, all_shapes: bool, op_name: Optional[str] = None, num_shapes: int = 1) -> List[Path]:
	# New consolidated format: {op_name}/input_shapes.json
	if op_name:
		consolidated = shapes_dir / op_name / "input_shapes.json"
		if consolidated.exists():
			return [consolidated]

	# When multiple shapes are needed, try per-input files first
	if all_shapes or num_shapes > 1:
		per_input_patterns = [f"*{kernel_name}_input*_shapes.json"]
		per_input_matches: List[Path] = []
		for pattern in per_input_patterns:
			per_input_matches.extend(sorted(shapes_dir.rglob(pattern)))
		per_input_matches = _dedup_paths(per_input_matches)
		per_input_matches = _filter_shapes_by_kernel(per_input_matches, kernel_name)
		if per_input_matches:
			return per_input_matches if all_shapes else per_input_matches[:num_shapes]

	# Fall back to default per-kernel shapes pattern (backward compat)
	default_patterns = [f"*{kernel_name}_shapes.json", f"*_{kernel_name}_shapes.json", f"{kernel_name}_shapes.json"]
	matches: List[Path] = []
	for pattern in default_patterns:
		matches.extend(sorted(shapes_dir.rglob(pattern)))
	deduped = _dedup_paths(matches)
	deduped = _filter_shapes_by_kernel(deduped, kernel_name)
	if not deduped:
		raise FileNotFoundError(f"No shapes found for kernel '{kernel_name}' under {shapes_dir}")
	return deduped if all_shapes else deduped[:num_shapes]


def _select_triton_backend(op_name: str, torch_model: Optional[str] = None) -> Optional[str]:
	"""Select a Triton backend for an operator.

	Preference order:
	1) Exact/normalized match against `torch_model` if provided
	2) Any backend tagged with 'triton'
	3) Any backend whose name contains 'triton'
	"""
	if op_name not in REGISTERED_BENCHMARKS:
		return None
	backends = REGISTERED_BENCHMARKS[op_name]

	# Try to match backend name to the torch_model hint
	if torch_model:
		tm = torch_model.strip()
		# Normalize common variations
		candidates = [
			tm,
			tm.replace("triton_", ""),
			f"triton_{tm}",
			tm.replace("_kernel", ""),
		]
		# Exact match
		for name in backends.keys():
			if name in candidates:
				return name
		# Substring match
		for name in backends.keys():
			if any(c in name for c in candidates):
				return name

	# Fallbacks
	tagged = [name for name, cfg in backends.items() if getattr(cfg, "tags", None) and any("triton" in t.lower() for t in cfg.tags)]
	named = [name for name in backends if "triton" in name]

	if tagged:
		return tagged[0]
	if named:
		return named[0]
	return None


def _shape_signature(tensor_args: list, scalar_args: list, grid) -> str:
	"""Compute a hashable signature for deduplication of repeated kernel calls."""
	parts = []
	for t in tensor_args:
		parts.append(f"{t['arg_index']}:{t['shape']}:{t['dtype']}")
	for s in scalar_args:
		parts.append(f"{s['arg_index']}:{s['value']}")
	parts.append(str(grid))
	return "|".join(parts)


def _normalize_grid(grid):
	"""Convert a Triton grid value to a JSON-serializable format."""
	if isinstance(grid, (tuple, list)):
		return list(grid)
	if callable(grid):
		try:
			import inspect
			return {"type": "callable", "source": inspect.getsource(grid)}
		except Exception:
			return {"type": "callable", "repr": str(grid)}
	return str(grid)


def _capture_inputs_for_op(
	op_name: str,
	torch_model: str,
	shapes_dir: Path,
	device: str = "cuda",
	num_inputs: int = 1,
	start_input_id: int = 0,
) -> None:
	"""Run the operator and capture Triton kernel inputs via JIT hook.

	Captures all unique input configurations (deduplicated by shape signature).
	Writes per-input files ``<op>_<kernel>_input<i>_shapes.json`` and a
	backward-compatible default ``<op>_<kernel>_shapes.json`` (= input 0).
	"""
	op_cls = load_opbench_by_name(op_name)
	if op_cls is None:
		raise RuntimeError(f"operator class not found for {op_name}")
	backend = _select_triton_backend(op_name, torch_model)
	if not backend:
		raise RuntimeError(f"no triton backend registered for {op_name}")
	if JITFunction is None:
		raise RuntimeError("Triton JIT not available for capture")

	shapes_dir.mkdir(parents=True, exist_ok=True)

	arg_list = [
		"--op", op_name,
		"--device", device,
		"--only", backend,
		"--num-inputs", str(num_inputs),
		"--input-id", str(start_input_id),
		"--warmup", "0",
		"--rep", "1",
		"--repcnt", "1",
		"--skip-print",
		"--keep-going",
	]
	tb_args, extra_args = override_args(arg_list)
	op = op_cls(tb_args=tb_args, extra_args=extra_args)

	captured: Dict[str, List[Dict]] = {}
	seen_sigs: Dict[str, set] = {}
	original_run = JITFunction.run  # type: ignore[attr-defined]

	def patched_run(self, *args, grid, **kwargs):  # type: ignore[no-redef]
		try:
			kname = getattr(getattr(self, "fn", None), "__name__", "unknown")
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

			grid_info = _normalize_grid(grid)
			sig = _shape_signature(tensor_args, scalar_args, grid_info)

			if kname not in seen_sigs:
				seen_sigs[kname] = set()
				captured[kname] = []

			if sig not in seen_sigs[kname]:
				seen_sigs[kname].add(sig)
				captured[kname].append({
					"kernel": kname,
					"tensor_args": tensor_args,
					"scalar_args": scalar_args,
					"kwargs": kwargs_info,
					"grid": grid_info,
					"total_args": len(args),
					"input_index": len(captured[kname]),
				})
		except Exception:
			pass
		return original_run(self, *args, grid=grid, **kwargs)

	try:
		JITFunction.run = patched_run  # type: ignore[assignment]
		op.run(warmup=0, rep=1, sleep=0)
	finally:
		JITFunction.run = original_run  # type: ignore[assignment]

	for kname, shape_list in captured.items():
		if not shape_list:
			continue
		# Backward-compatible default file (= input 0)
		default_path = shapes_dir / f"{op_name}_{kname}_shapes.json"
		default_path.write_text(json.dumps(shape_list[0], indent=2))
		logger.info(f"Captured default shapes for kernel '{kname}' of op '{op_name}' to {default_path}")
		# Per-input files
		for i, meta in enumerate(shape_list):
			per_input_path = shapes_dir / f"{op_name}_{kname}_input{i}_shapes.json"
			per_input_path.write_text(json.dumps(meta, indent=2))
		logger.info(f"Captured {len(shape_list)} input shape(s) for kernel '{kname}'")


def _run_capture_shapes(target_dir: Path, device: str = "cuda:0") -> int:
	"""Invoke capture_all_kernel_ir.py to generate real input shapes."""
	script = Path(__file__).resolve().parent.joinpath("capture_all_kernel_ir.py")
	if not script.exists():
		logger.warning(f"capture_all_kernel_ir.py not found at {script}")
		return 1
	cmd = [
		sys.executable,
		str(script),
		"--output-dir",
		str(target_dir),
		"--device",
		device,
	]
	logger.info(f"Capturing shapes by running: {' '.join(cmd)}")
	proc = subprocess.run(cmd, capture_output=True, text=True)
	if proc.returncode != 0:
		logger.error("Shape capture failed:")
		logger.error(proc.stdout)
		logger.error(proc.stderr)
	return proc.returncode


def _prepare_work_items(
	cache_dir: Path,
	shapes_dir: Path,
	kernels: List[str],
	all_shapes: bool,
	device: str,
	auto_capture_shapes: bool,
	kernel_to_op: Optional[Dict[str, str]] = None,
	capture_num_inputs: int = 1,
	capture_start_input_id: int = 0,
	num_shapes: int = 1,
) -> List[KernelWorkItem]:
	items: List[KernelWorkItem] = []
	tried_capture = False
	logger.info(f"Processing {len(kernels)} kernels: {kernels[:5]}{'...' if len(kernels) > 5 else ''}")
	for kernel in kernels:
		try:
			ttgir_path = _find_ttgir(cache_dir, kernel)
			torch_model, source_path = _extract_source_info(ttgir_path)
			op_name = kernel_to_op.get(kernel) if kernel_to_op else None
			try:
				shapes = _find_shapes(shapes_dir, kernel, all_shapes, op_name=op_name, num_shapes=num_shapes)
				items.append(KernelWorkItem(kernel, ttgir_path, torch_model, source_path, shapes))
			except FileNotFoundError as e:
				# First try a targeted operator run to capture inputs for this kernel's op
				if auto_capture_shapes and kernel_to_op and kernel in kernel_to_op:
					op_name = kernel_to_op[kernel]
					try:
						_capture_inputs_for_op(
							op_name, torch_model, shapes_dir, device=device,
							num_inputs=capture_num_inputs,
							start_input_id=capture_start_input_id,
						)
						shapes = _find_shapes(shapes_dir, kernel, all_shapes, op_name=op_name, num_shapes=num_shapes)
						items.append(KernelWorkItem(kernel, ttgir_path, torch_model, source_path, shapes))
						continue
					except Exception as cap_err:
						logger.warning(f"Auto-capture via operator failed for '{kernel}' ({op_name}): {cap_err}")
				elif auto_capture_shapes and kernel_to_op and kernel not in kernel_to_op:
					logger.warning(f"No op mapping for kernel '{kernel}' in metadata; cannot auto-capture shapes")
				# Fallback to the broader capture script once
				if auto_capture_shapes and not tried_capture:
					ret = _run_capture_shapes(shapes_dir, device=device)
					tried_capture = True
					if ret == 0:
						try:
							shapes = _find_shapes(shapes_dir, kernel, all_shapes, op_name=op_name, num_shapes=num_shapes)
							items.append(KernelWorkItem(kernel, ttgir_path, torch_model, source_path, shapes))
							continue
						except Exception as retry_err:
							logger.warning(f"Retrying shapes after capture failed for '{kernel}': {retry_err}")
				logger.warning(f"Skipping kernel '{kernel}': {e}")
		except Exception as e:  # noqa: BLE001
			logger.warning(f"Skipping kernel '{kernel}' due to TTGIR/source error: {e}")
	return items


def _run_single(
	item: KernelWorkItem,
	shape_path: Path,
	args: argparse.Namespace,
) -> RunResult:
	try:
		if args.single_agent:
			result = analyze_kernel_from_cache(
				cache_dir=args.cache_dir,
				kernel_name=item.name,
				torch_model=item.torch_model,
				input_shapes_json=shape_path,
				target_gpu=args.target_gpu,
				triton_modules_dir=args.triton_modules_dir,
				use_cached_response=args.use_cached_response,
				cached_response_path=args.cached_response_file,
				model=args.model,
				prompt_output_dir=args.prompt_output_dir,
			)
		else:
			# When debug mode is on, isolate prompts/responses per kernel/shape
			# since different inputs produce different LLM interactions.
			debug_dir = args.debug_dir
			cached_prompts_dir = args.use_cached_prompts
			if args.debug_mode and args.debug_dir:
				sub_folder = f"{item.name}__{shape_path.stem}"
				debug_dir = Path(args.debug_dir) / sub_folder
			if args.use_cached_prompts:
				sub_folder = f"{item.name}__{shape_path.stem}"
				cached_prompts_dir = Path(args.use_cached_prompts) / sub_folder

			result = optimize_kernel_from_cache(
				cache_dir=args.cache_dir,
				kernel_name=item.name,
				torch_model=item.torch_model,
				input_shapes_json=shape_path,
				target_gpu=args.target_gpu,
				triton_modules_dir=args.triton_modules_dir,
				llm_provider=args.llm_provider,
				llm_model=args.llm_model,
				temperature=args.temperature,
				max_iterations=args.max_iterations,
				num_candidates=args.num_candidates,
				optimization_target=args.optimization_target,
				debug_mode=args.debug_mode,
				debug_dir=debug_dir,
				use_cached_prompts=bool(args.use_cached_prompts),
				cached_prompts_dir=cached_prompts_dir,
				use_profiler=args.enable_profiling,
				skip_resource_budget=args.skip_resource_budget,
				skip_data_movement=args.skip_data_movement,
				include_agent_proposals=args.include_agent_proposals,
				disable_negotiation=args.disable_negotiation,
			)
		if not result:
			return RunResult(item.name, shape_path, success=False, error="empty result")
		run_result = RunResult(item.name, shape_path, success=True, payload=result)
	except Exception as e:  # noqa: BLE001
		return RunResult(item.name, shape_path, success=False, error=str(e))

	if not args.test_config:
		return run_result

	# Template kernels (@triton_heuristics.template) bake M/N/K/strides as
	# literals, so each captured shape needs its own per-input source.
	kernel_file = item.source_path_for(shape_path)
	if not kernel_file.exists():
		run_result.success = False
		run_result.error = f"kernel source not found: {kernel_file}"
		return run_result

	if kernel_file == item.source_path:
		try:
			base_src = item.source_path.read_text()
		except (OSError, IOError):
			base_src = ""
		if "triton_heuristics.template(" in base_src:
			run_result.success = False
			run_result.error = (
				f"per-shape source missing for template kernel {item.name} "
				f"@ {shape_path.name}; rerun extract/run_inductor.py"
			)
			return run_result

	best_config = extract_best_config(result)
	if not best_config:
		run_result.success = False
		run_result.error = "no best config"
		return run_result

	baseline_result = None
	# if args.benchmark_baseline:
	# 	try:
	# 		baseline_result = benchmark_baseline_config(
	# 			kernel_name=item.name,
	# 			kernel_path=kernel_file,
	# 			input_shapes_json=shape_path,
	# 			device=args.test_device,
	# 			num_iterations=args.num_iterations,
	# 		)
	# 		run_result.payload["baseline_benchmark"] = baseline_result
	# 	except Exception as e:  # noqa: BLE001
	# 		run_result.error = f"baseline benchmark failed: {e}"

	try:
		optimized_result = test_recommended_config(
			kernel_name=item.name,
			kernel_path=kernel_file,
			input_shapes_json=shape_path,
			config=best_config,
			device=args.test_device,
			num_iterations=args.num_iterations,
		)
		run_result.payload["optimized_benchmark"] = optimized_result
		if args.benchmark_baseline and baseline_result and baseline_result.get("success") and optimized_result.get("success"):
			baseline_time = baseline_result["execution_time_ms"]["median"]
			optimized_time = optimized_result["execution_time_ms"]["median"]
			if optimized_time > 0:
				speedup = baseline_time / optimized_time
				run_result.baseline_ms = baseline_time
				run_result.optimized_ms = optimized_time
				run_result.speedup = speedup
				run_result.payload["performance_comparison"] = {
					"baseline_time_ms": baseline_time,
					"optimized_time_ms": optimized_time,
					"speedup": speedup,
					"improvement_percent": (speedup - 1) * 100,
				}
	except Exception as e:  # noqa: BLE001
		run_result.error = f"optimized benchmark failed: {e}"

	return run_result


def _summarize(results: List[RunResult], summary_path: Path) -> None:
	success = [r for r in results if r.success]
	failures = [r for r in results if not r.success]
	best_speedup: Dict[str, Dict[str, float]] = {}
	for r in results:
		if r.speedup is None:
			continue
		current = best_speedup.get(r.kernel)
		if current is None or r.speedup > current.get("speedup", 0):
			best_speedup[r.kernel] = {
				"speedup": r.speedup,
				"baseline_time_ms": r.baseline_ms,
				"optimized_time_ms": r.optimized_ms,
				"shape_path": str(r.shape_path),
			}
	summary = {
		"total_runs": len(results),
		"successes": len(success),
		"failures": len(failures),
		"best_speedup_per_kernel": best_speedup,
		"errors": [
			{
				"kernel": r.kernel,
				"shape_path": str(r.shape_path),
				"error": r.error,
			}
			for r in failures
			if r.error
		],
	}
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.write_text(json.dumps(summary, indent=2))


def _update_live_summary(
	results: List[RunResult],
	output_dir: Path,
	total_shapes: int,
	start_time: datetime,
) -> None:
	"""Write incremental summary to output_dir/live_summary.json.

	Called after every shape so users can monitor progress in real time.
	"""
	now = datetime.now()
	elapsed = (now - start_time).total_seconds()
	completed = len(results)
	remaining = total_shapes - completed
	avg_per_shape = elapsed / completed if completed else 0

	per_shape = []
	speedups: List[float] = []
	improved = 0
	unchanged = 0
	regressed = 0
	kept_original = 0
	successes = 0
	failures = 0

	for r in results:
		entry: Dict[str, Any] = {
			"kernel": r.kernel,
			"shape": Path(r.shape_path).stem,
			"shape_path": str(r.shape_path),
			"success": r.success,
			"error": r.error,
		}

		if not r.success:
			failures += 1
			per_shape.append(entry)
			continue

		successes += 1
		payload = r.payload or {}
		result_data = payload.get("result", {})
		bench = payload.get("benchmark", {})

		recommendation = result_data.get("recommendation", "optimize")
		entry["recommendation"] = recommendation
		entry["confidence_score"] = result_data.get("confidence_score")

		orig_stats = bench.get("original_stats", {})
		rec_stats = bench.get("recommended_stats", {})
		speedup = bench.get("speedup")

		entry["speedup"] = speedup
		entry["original_median_ms"] = orig_stats.get("median")
		entry["recommended_median_ms"] = rec_stats.get("median")
		entry["benchmark_iterations"] = payload.get("benchmark_iterations", 0)

		entry["original_config"] = (
			payload.get("current_config")
			or result_data.get("original_config")
		)
		entry["recommended_config"] = result_data.get("recommended_config")

		if recommendation == "keep_original":
			kept_original += 1
		elif speedup is not None:
			if speedup > 1.0:
				improved += 1
			elif speedup < 1.0:
				regressed += 1
			else:
				unchanged += 1

		if speedup is not None:
			speedups.append(speedup)

		per_shape.append(entry)

	speedup_stats: Dict[str, Any] = {
		"average_speedup": sum(speedups) / len(speedups) if speedups else None,
		"max_speedup": max(speedups) if speedups else None,
		"min_speedup": min(speedups) if speedups else None,
		"num_with_speedup_data": len(speedups),
	}

	summary = {
		"last_updated": now.isoformat(),
		"elapsed_seconds": round(elapsed, 1),
		"progress": {
			"total_shapes": total_shapes,
			"completed": completed,
			"remaining": remaining,
			"percent_complete": round(completed / total_shapes * 100, 1) if total_shapes else 0,
			"estimated_remaining_seconds": round(avg_per_shape * remaining, 1),
		},
		"counts": {
			"success": successes,
			"failure": failures,
			"improved": improved,
			"unchanged": unchanged,
			"regressed": regressed,
			"kept_original": kept_original,
		},
		"speedup_stats": speedup_stats,
		"per_shape": per_shape,
	}

	live_path = output_dir / "live_summary.json"
	live_path.write_text(json.dumps(summary, indent=2))


def _print_final_results(results: List[RunResult]) -> None:
	"""Print a summary table of all optimization results to stdout."""
	if not results:
		return

	# Compute overall stats first
	total = len(results)
	successes = sum(1 for r in results if r.success)
	kept_original = sum(
		1 for r in results
		if r.success and (r.payload or {}).get("result", {}).get("recommendation") == "keep_original"
	)
	improved = sum(
		1 for r in results
		if r.success and (r.payload or {}).get("benchmark", {}).get("speedup", 0) and
		(r.payload or {}).get("benchmark", {}).get("speedup", 0) >= 1.0 and
		(r.payload or {}).get("result", {}).get("recommendation") != "keep_original"
	)

	print("\n" + "=" * 80)
	print("FINAL RESULTS")
	print("=" * 80)
	print(f"  Total: {total}  |  Success: {successes}  |  Improved: {improved}  |  Kept original: {kept_original}  |  Failed: {total - successes}")
	print(f"{'─' * 80}")

	for r in results:
		shape_label = Path(r.shape_path).stem
		print(f"\n  Kernel: {r.kernel}  |  Shape: {shape_label}")
		print(f"  {'─' * 70}")

		if not r.success:
			print(f"  Status: FAILED — {r.error}")
			continue

		payload = r.payload or {}
		bench = payload.get("benchmark")
		bench_iters = payload.get("benchmark_iterations", 0)
		result_data = payload.get("result", {})
		recommendation = result_data.get("recommendation", "")

		if recommendation == "keep_original":
			print(f"  Status: KEEP ORIGINAL (all {bench_iters} optimization attempts regressed)")
			if bench:
				orig_stats = bench.get("original_stats", {})
				if orig_stats:
					print(f"  Original median: {orig_stats.get('median', 'N/A'):.4f} ms")
			print(f"  Outcome: No regression (original config preserved)")
		elif bench:
			orig_stats = bench.get("original_stats", {})
			rec_stats = bench.get("recommended_stats", {})
			speedup = bench.get("speedup", None)
			orig_med = orig_stats.get("median")
			rec_med = rec_stats.get("median")

			print(f"  Benchmark iterations: {bench_iters}")
			if orig_med is not None:
				print(f"  Original median:    {orig_med:.4f} ms")
			if rec_med is not None:
				print(f"  Recommended median: {rec_med:.4f} ms")
			if speedup is not None:
				improvement = (speedup - 1) * 100
				label = "IMPROVEMENT" if speedup >= 1.0 else "REGRESSION"
				print(f"  Speedup: {speedup:.3f}x ({improvement:+.1f}% — {label})")
		else:
			print(f"  Status: Completed (no benchmark data)")

	print("=" * 80 + "\n")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Model-free multi-agent TTGIR optimizer")
	parser.add_argument("--cache-dir", type=Path, default=Path("irs/new"), help="Directory containing TTGIR files")
	parser.add_argument("--input-shapes-dir", type=Path, help="Directory containing *_shapes.json; if omitted or empty, minimal shapes will be generated")
	parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-run results")
	parser.add_argument("--auto-capture-shapes", action="store_true", help="Run capture_all_kernel_ir.py to generate real shapes if none are found")
	parser.add_argument("--prepare-cache", action="store_true", help="Run extract/run.py to populate TTGIR/IR into --cache-dir before optimizing")
	parser.add_argument("--all-shapes", action="store_true", help="Process all matching shapes per kernel instead of first only")
	parser.add_argument("--num-shapes", type=int, default=1, help="Number of input shapes to process per kernel when --all-shapes is not set (default: 1)")
	parser.add_argument("--capture-num-inputs", type=int, default=1, help="Number of input configs to capture during auto-capture (default: 1)")
	parser.add_argument("--capture-start-input-id", type=int, default=0, help="Starting input ID for auto-capture (default: 0)")
	parser.add_argument("--kernels", type=str, help="Comma-separated kernel name filter")
	parser.add_argument("--llm-provider", type=str, default="deepseek", choices=["deepseek", "openai", "ollama", "vllm"], help="LLM provider")
	parser.add_argument("--llm-model", type=str, help="LLM model name override")
	parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
	parser.add_argument("--max-iterations", type=int, default=1, help="Max refinement iterations")
	parser.add_argument("--num-candidates", type=int, default=3, help="Number of candidate configs per round (default: 3)")
	parser.add_argument("--skip-resource-budget", action="store_true", help="Ablation A1a: remove ResourceBudgetAgent (single-agent DM only)")
	parser.add_argument("--skip-data-movement", action="store_true", help="Ablation A1b: remove DataMovementAgent (single-agent RB only)")
	parser.add_argument("--include-agent-proposals", action="store_true", help="Inject each agent's preferred config as benchmark candidates alongside orchestrator synthesis")
	parser.add_argument("--disable-negotiation", action="store_true", help="Skip negotiation sub-loop between agents")
	parser.add_argument("--optimization-target", type=str, default="throughput", choices=["throughput", "latency", "energy"], help="Optimization target")
	parser.add_argument("--debug-mode", action="store_true", help="Enable debug logging for multi-agent")
	parser.add_argument("--debug-dir", type=Path, default=Path("./debug_multiagent"), help="Debug output directory")
	parser.add_argument("--single-agent", action="store_true", help="Use single-agent ConfigurationOptimizer instead of multi-agent")
	parser.add_argument("--use-cached-response", action="store_true", help="Use cached response for single-agent")
	parser.add_argument("--cached-response-file", type=Path, help="Path to cached response JSON for single-agent")
	parser.add_argument("--model", type=str, default="deepseek-chat", help="LLM model for single-agent path")
	parser.add_argument("--prompt-output-dir", type=Path, help="Directory to save prompts for single-agent")
	parser.add_argument("--target-gpu", type=str, default="cuda:70", help="Target GPU arch code: cuda:70 (V100), cuda:80 (A100), cuda:86 (RTX3090), cuda:a40 (A40), cuda:l40s (L40S), cuda:90 (H100)")
	parser.add_argument("--triton-modules-dir", type=Path, help="Path to triton modules for source lookup")
	parser.add_argument("--test-config", action="store_true", help="Benchmark recommended config")
	parser.add_argument("--benchmark-baseline", action="store_true", help="Also benchmark baseline config")
	parser.add_argument("--test-device", type=str, default="cuda", choices=["cuda", "xpu", "cpu"], help="Device for benchmarking")
	parser.add_argument("--num-iterations", type=int, default=100, help="Iterations for benchmarking")
	parser.add_argument("--summary-file", type=Path, help="Summary JSON output path (default: <output-dir>/summary.json)")
	parser.add_argument("--use-cached-prompts", type=Path, metavar="DEBUG_DIR", help="Replay LLM responses from a previous debug run directory instead of calling the API (e.g. debug_multiagent/20260203_204657)")
	parser.add_argument("--enable-profiling", action="store_true", help="Enable NCU profiling for detailed hardware metrics in the feedback loop (requires ncu on PATH)")
	parser.add_argument("--shape-file", type=Path, help="Process exactly this one shape file (used by batch_runner --parallel-shapes)")
	parser.add_argument("--metadata-extra", type=Path, action="append", default=None, help="Additional metadata YAML files to merge alongside oss_cuda_kernels.yaml (may be repeated)")
	return parser.parse_args()


def main() -> int:
	args = _parse_args()

	if not args.cache_dir.exists():
		print(f"Cache dir does not exist: {args.cache_dir}")
		return 1
	if args.input_shapes_dir is None:
		args.input_shapes_dir = args.output_dir.joinpath("generated_shapes")
		print(f"No input shapes dir provided; will generate minimal shapes under {args.input_shapes_dir}")
	if args.input_shapes_dir.is_file():
		print(f"Input shapes path is a file, expected a directory: {args.input_shapes_dir}")
		return 1
	if not args.input_shapes_dir.exists():
		print(f"Input shapes dir does not exist; creating: {args.input_shapes_dir}")
		args.input_shapes_dir.mkdir(parents=True, exist_ok=True)

	# Optional: generate TTGIR/IR by running the extract runner
	if getattr(args, "prepare_cache", False):
		extract_runner = Path(__file__).resolve().parent.joinpath("extract", "run.py")
		if not extract_runner.exists():
			print(f"Cannot prepare cache; missing extract runner at {extract_runner}")
			return 1
		cmd = [
			sys.executable,
			str(extract_runner),
			"--output-dir",
			str(args.cache_dir),
		]
		if args.kernels:
			# The extract runner filters by operator names; pass through verbatim to narrow scope
			cmd.extend(["--ops", args.kernels])
		print(f"Preparing cache by running: {' '.join(cmd)}")
		prep = subprocess.run(cmd, capture_output=True, text=True)
		if prep.returncode != 0:
			print("Cache preparation failed:")
			print(prep.stdout)
			print(prep.stderr)
			return prep.returncode
		print("Cache preparation complete")

	metadata_dir = _metadata_dir()
	extra_yamls = list(args.metadata_extra or [])
	kernels = _parse_triton_kernels(metadata_dir, extra_yamls=extra_yamls)
	k2op = _kernel_to_op_map(metadata_dir, extra_yamls=extra_yamls)

	if args.kernels:
		wanted = {k.strip() for k in args.kernels.split(',') if k.strip()}
		kernels = [k for k in kernels if k in wanted]

		# Fallback: if requested kernels aren't registered in the YAML
		# (e.g., inductor-generated kernels), check if they have matching
		# TTGIR files directly in --cache-dir.
		if not kernels:
			for kname in sorted(wanted):
				ttgir_candidates = sorted(args.cache_dir.rglob(f"*{kname}*.ttgir"))
				if ttgir_candidates:
					kernels.append(kname)
					logger.info(f"Kernel '{kname}' not in YAML but TTGIR found in cache dir (direct mode)")

	if not kernels:
		print("No kernels to process after filtering")
		return 1

	try:
		work_items = _prepare_work_items(
			args.cache_dir,
			args.input_shapes_dir,
			kernels,
			args.all_shapes,
			args.test_device,
			args.auto_capture_shapes,
			k2op,
			capture_num_inputs=args.capture_num_inputs,
			capture_start_input_id=args.capture_start_input_id,
			num_shapes=args.num_shapes,
		)
	except Exception as e:  # noqa: BLE001
		print(f"Preparation failed: {e}")
		return 1

	# --shape-file: override discovered shapes to process exactly one file
	if args.shape_file:
		sf = args.shape_file.resolve()
		if not sf.exists():
			print(f"Shape file does not exist: {sf}")
			return 1
		for item in work_items:
			item.shapes = [sf]

	args.output_dir.mkdir(parents=True, exist_ok=True)

	# Configure file-only logging for per-kernel details
	log_path = args.output_dir / "runner.log"
	file_handler = logging.FileHandler(log_path, mode="w")
	file_handler.setLevel(logging.DEBUG)
	file_handler.setFormatter(logging.Formatter(
		"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
	))
	logger.addHandler(file_handler)
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	# Also redirect dependency loggers to the same file (suppress terminal)
	for dep_logger_name in ("compare_configs", "optimizer_functions", "orchestrator"):
		dep = logging.getLogger(dep_logger_name)
		dep.addHandler(file_handler)
		dep.setLevel(logging.DEBUG)
		dep.propagate = False

	# Create a timestamped debug directory for this launch
	launch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	if args.debug_mode and args.debug_dir:
		args.debug_dir = Path(args.debug_dir) / launch_timestamp
		print(f"Debug output directory: {args.debug_dir}")
	if args.use_cached_prompts:
		args.use_cached_prompts = Path(args.use_cached_prompts) / launch_timestamp

	results: List[RunResult] = []
	optimization_start_time = datetime.now()

	total_shapes = sum(len(item.shapes) for item in work_items)
	log_stream = open(log_path, "a")
	try:
		with tqdm(total=total_shapes, desc="Optimizing", unit="shape") as pbar:
			for item in work_items:
				for shape_path in item.shapes:
					pbar.set_description(f"{item.name}")
					with contextlib.redirect_stdout(log_stream):
						run_result = _run_single(item, shape_path, args)
					results.append(run_result)
					out_name = f"{item.name}__{shape_path.stem}.json"
					# Truncate if filename exceeds filesystem limit (255 bytes)
					if len(out_name.encode()) > 255:
						import hashlib
						suffix = hashlib.md5(out_name.encode()).hexdigest()[:12]
						max_base = 255 - len(".json") - len(suffix) - 1  # 1 for '_'
						out_name = out_name[:max_base].rstrip("_") + f"_{suffix}.json"
					out_path = args.output_dir.joinpath(out_name)
					out_payload = {
						"kernel": item.name,
						"shape_path": str(shape_path),
						"success": run_result.success,
						"error": run_result.error,
						"result": run_result.payload,
					}
					out_path.write_text(json.dumps(out_payload, indent=2))
					_update_live_summary(results, args.output_dir, total_shapes, optimization_start_time)
					pbar.update(1)
	finally:
		log_stream.close()

	summary_path = args.summary_file or args.output_dir.joinpath("summary.json")
	_summarize(results, summary_path)
	print(f"Summary written to {summary_path}")
	print(f"Detailed log written to {log_path}")
	_print_final_results(results)
	# Exit 0 if at least one shape succeeded (partial success is still success).
	# Per-shape failures are recorded in live_summary.json for inspection.
	return 0 if any(r.success for r in results) else 1


if __name__ == "__main__":
	sys.exit(main())
