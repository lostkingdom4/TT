#!/usr/bin/env python3
"""
Capture IR for all Triton kernels by running models and recording actual inputs.

This script:
1. Iterates through all Triton module files in data_64/triton_modules
2. Runs the corresponding PyTorch model to capture real kernel input shapes
3. Generates dummy tensors with the same shapes
4. Executes each kernel to trigger compilation and capture IR
5. Stores IR, metadata, and input shapes for later analysis
"""

import sys
import os
import json
import torch
import logging
import gc
import importlib.util
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add Data directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "Data"))

try:
	from tune_one_file import (
		install_inductor_shims,
		_install_triton_grid_shim
	)
	TUNE_UTILS_AVAILABLE = True
	# Install shims immediately
	_install_triton_grid_shim()
	install_inductor_shims()
except ImportError:
	TUNE_UTILS_AVAILABLE = False
	print("Warning: tune_one_file utilities not available")

# Configure logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	handlers=[
		logging.StreamHandler(),
		logging.FileHandler('capture_kernel_ir.log')
	]
)
log = logging.getLogger(__name__)

# Triton imports
try:
	import triton
	import triton.language as tl
	from triton.runtime.jit import JITFunction
	TRITON_AVAILABLE = True
except ImportError:
	TRITON_AVAILABLE = False
	log.error("Triton not available - cannot capture IR")
	sys.exit(1)


@dataclass
class KernelIRCapture:
	"""Metadata and IR for a captured kernel"""
	kernel_name: str
	module_file: str
	ir_type: str  # 'ttir', 'ttgir', 'llir', etc.
	ir_content: str
	input_shapes: List[Tuple[int, ...]]
	input_dtypes: List[str]
	numel_params: Dict[str, int]
	other_params: List[str]
	grid: Tuple[int, ...]
	config: Dict[str, Any]
	success: bool
	error: Optional[str] = None


class KernelIRCaptureSystem:
	"""System for capturing IR from all Triton kernels"""

	def __init__(self, data_dir: Path, output_dir: Path, device: str = "cuda:1"):
		self.data_dir = data_dir
		self.triton_modules_dir = data_dir / "triton_modules"
		self.pytorch_modules_dir = data_dir / "pytorch_modules"
		self.output_dir = output_dir
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.device = torch.device(device)

		# Set up Triton cache directory
		self.triton_cache_dir = output_dir / "triton_cache"
		self.triton_cache_dir.mkdir(parents=True, exist_ok=True)
		os.environ['TRITON_CACHE_DIR'] = str(self.triton_cache_dir)
		log.info(f"Triton cache directory: {self.triton_cache_dir}")

		# Create subdirectories for organized storage
		self.ir_dir = output_dir / "ir_files"
		self.metadata_dir = output_dir / "metadata"
		self.shapes_dir = output_dir / "input_shapes"

		for d in [self.ir_dir, self.metadata_dir, self.shapes_dir]:
			d.mkdir(parents=True, exist_ok=True)

		self.captured_inputs = {}  # Store captured input shapes
		self.results = []

	def cleanup_memory(self):
		"""Aggressive memory cleanup"""
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		elif hasattr(torch, "xpu") and torch.xpu.is_available():
			torch.xpu.synchronize()
			if hasattr(torch.xpu, 'empty_cache'):
				torch.xpu.empty_cache()

	def extract_input_shapes_from_call(self, source_code: str) -> List[Tuple[int, ...]]:
		"""
		Parse the call() function to extract input shapes from assert_size_stride calls.
		Only extracts the first assertion for each primals_N variable (the initial args).
		Returns list of tuples representing tensor shapes.
		"""
		import re

		shapes = []
		seen_vars = set()

		# Find assert_size_stride calls - pattern: assert_size_stride(var, (shape), (stride))
		# Example: assert_size_stride(primals_1, (64, 64, 64, 64), (262144, 4096, 64, 1))
		# Also handle single dimension: assert_size_stride(primals_3, (64,), (1,))
		pattern = r'assert_size_stride\s*\((\w+)\s*,\s*\(([^)]+)\)\s*,'

		matches = re.findall(pattern, source_code)
		for var_name, shape_str in matches:
			# Only capture primals_* variables (the actual function inputs)
			# Skip buf* variables (intermediate buffers)
			if not var_name.startswith('primals_'):
				continue

			# Only capture first occurrence of each variable
			if var_name in seen_vars:
				continue
			seen_vars.add(var_name)

			try:
				# Parse the shape tuple
				# shape_str looks like: "64, 64, 64, 64" or "64,"
				shape_str = shape_str.strip()
				if shape_str:
					# Remove trailing comma and split
					shape_str = shape_str.rstrip(',').strip()
					if ',' in shape_str:
						# Multiple dimensions
						shape = tuple(int(x.strip()) for x in shape_str.split(',') if x.strip())
					else:
						# Single dimension
						shape = (int(shape_str),)
					shapes.append(shape)
					log.debug(f"    Extracted shape for {var_name}: {shape}")
			except Exception as e:
				log.debug(f"    Failed to parse shape from: {shape_str} - {e}")

		return shapes

	def capture_model_inputs(self, triton_file: Path) -> Tuple[Dict[str, Dict[str, Any]], Any]:
		"""
		Run the Triton module's model and capture actual kernel inputs using hooks.
		Returns tuple of (captured_kernels dict, module object with cached kernels).
		"""
		# Derive module name from Triton file
		fname = triton_file.name
		if not fname.endswith("_triton.py"):
			log.warning(f"Expected '*_triton.py' file, got: {fname}")
			return ({}, None)

		op_name = fname[:-len("_triton.py")]

		try:
			# Parse the Triton file to extract input shapes from call() function
			source_code = triton_file.read_text(encoding="utf-8", errors="ignore")
			input_shapes = self.extract_input_shapes_from_call(source_code)

			if not input_shapes:
				log.warning(f"Could not extract input shapes from {triton_file}")
				return ({}, None)

			log.info(f"  Extracted {len(input_shapes)} input shapes from call() function")

			# Import Triton module (which contains the model class that calls kernels)
			spec = importlib.util.spec_from_file_location(f"triton_{op_name}", str(triton_file))
			if spec is None or spec.loader is None:
				return ({}, None)

			mod = importlib.util.module_from_spec(spec)
			sys.modules[f"triton_{op_name}"] = mod
			spec.loader.exec_module(mod)

			# Create input tensors based on extracted shapes
			device = self.device
			inputs = []
			for shape in input_shapes:
				if len(shape) > 0:
					# Create tensor with the extracted shape
					inputs.append(torch.rand(shape, device=device))
				else:
					# Scalar or empty shape - skip
					log.debug(f"  Skipping empty shape: {shape}")

			if not inputs:
				log.warning(f"No valid input tensors created")
				return ({}, None)

			log.info(f"  Created {len(inputs)} input tensors")

			# Get the call() function from Triton module
			if not hasattr(mod, 'call'):
				log.warning(f"No 'call' function found in {triton_file}")
				return ({}, None)

			call_fn = mod.call            # Storage for captured kernel data
			captured_kernels = {}

			# call_fn(inputs)  # Warm-up run to ensure JITFunction is populated
			# exit()

			# Monkey-patch Triton JITFunction to capture kernel execution
			original_run = JITFunction.run

			def patched_run(self, *args, grid, **kwargs):
				"""Patched run method to capture kernel metadata and tensor shapes"""
				try:
					kernel_name = self.fn.__name__ if hasattr(self, 'fn') else 'unknown'

					# Extract signature from the compiled kernel
					# The signature is in self.signature if available
					signature = {}
					constants = {}

					if hasattr(self, 'signature'):
						# Convert Signature object to dict
						sig_obj = self.signature
						if hasattr(sig_obj, 'params'):
							signature = {i: str(p) for i, p in enumerate(sig_obj.params)}
						else:
							signature = str(sig_obj)
					if hasattr(self, 'constants'):
						const_obj = self.constants
						if isinstance(const_obj, dict):
							constants = const_obj
						else:
							constants = {str(k): str(v) for k, v in enumerate(const_obj)} if const_obj else {}

					# Capture tensor shapes and dtypes from actual arguments
					tensor_args = []
					scalar_args = []
					
					for idx, arg in enumerate(args):
						if isinstance(arg, torch.Tensor):
							tensor_args.append({
								'arg_index': idx,
								'shape': list(arg.shape),
								'dtype': str(arg.dtype),
								'device': str(arg.device),
								'numel': arg.numel(),
								'stride': list(arg.stride()),
								'is_contiguous': arg.is_contiguous()
							})
						elif isinstance(arg, (int, float, bool)):
							scalar_args.append({
								'arg_index': idx,
								'value': arg,
								'type': type(arg).__name__
							})
					
					# Capture kwargs (additional keyword arguments passed to kernel)
					kwargs_info = {}
					for key, value in kwargs.items():
						if key == 'grid':  # Skip grid as it's already captured
							continue
						if isinstance(value, torch.Tensor):
							kwargs_info[key] = {
								'type': 'tensor',
								'shape': list(value.shape),
								'dtype': str(value.dtype),
								'device': str(value.device)
							}
						elif isinstance(value, (int, float, bool, str)):
							kwargs_info[key] = {
								'type': 'scalar',
								'value': value,
								'value_type': type(value).__name__
							}
						else:
							kwargs_info[key] = {
								'type': 'other',
								'value': str(value),
								'value_type': type(value).__name__
							}

					# Store kernel metadata
					if kernel_name not in captured_kernels:
						captured_kernels[kernel_name] = {
							'signature': signature,
							'constants': constants,
							'grid': list(grid) if isinstance(grid, tuple) else grid,
							'args_count': len(args),
							'tensor_args': tensor_args,
							'scalar_args': scalar_args,
							'kwargs': kwargs_info,
						}
						log.debug(f"  Captured kernel call: {kernel_name}")
						log.debug(f"    Grid: {grid}")
						log.debug(f"    Args: {len(args)} ({len(tensor_args)} tensors, {len(scalar_args)} scalars)")

				except Exception as e:
					log.error(f"Error in patched run: {e}")

				# Call original run method
				return original_run(self, *args, grid=grid, **kwargs)

			# Install hook and run model
			try:
				# Clear JIT function caches to force recompilation
				# This ensures our patched run is called
				for name in dir(mod):
					obj = getattr(mod, name)
					if hasattr(obj, 'cache'):
						obj.cache.clear()

				JITFunction.run = patched_run                # Run call() function to trigger kernel compilation
				log.info(f"  Running call() function to capture kernel inputs...")
				with torch.no_grad():
					output = call_fn(inputs)

				log.info(f"  Successfully captured {len(captured_kernels)} kernel calls")

			finally:
				# Restore original run method
				JITFunction.run = original_run
				if 'inputs' in locals():
					del inputs
				self.cleanup_memory()

			return (captured_kernels, mod)  # Return both the metadata and module

		except Exception as e:
			log.error(f"Failed to run model {op_name}: {e}")
			log.debug(traceback.format_exc())
			return ({}, None)

	def capture_kernel_ir(self, triton_file: Path, kernel_name: str,
						 kernel_fn: Any, kernel_source: str,
						 captured_metadata: Dict[str, Any]) -> Optional[KernelIRCapture]:
		"""
		Extract IR from Triton's file-based cache.
		After kernels are executed, Triton saves compiled versions to disk.
		"""
		try:
			if not captured_metadata:
				log.warning(f"  No captured metadata for {kernel_name}, skipping")
				return KernelIRCapture(
					kernel_name=kernel_name,
					module_file=str(triton_file),
					ir_type="",
					ir_content="",
					input_shapes=[],
					input_dtypes=[],
					numel_params={},
					other_params=[],
					grid=(),
					config={},
					success=False,
					error="No captured metadata from model run"
				)

			# Extract IR from Triton's file cache
			# The cache directory contains .ttir, .ttgir, .llir files
			# Files are in subdirectories named with hashes, but filenames contain kernel names
			ir_content = ""
			ir_type = ""

			# Look for IR files matching this specific kernel name
			for ir_format in ['ttir', 'ttgir', 'llir']:
				matching_files = list(self.triton_cache_dir.rglob(f"*{kernel_name}.{ir_format}"))
				if matching_files:
					# Use the most recent file if there are multiple
					latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
					ir_content = latest_file.read_text()
					ir_type = ir_format
					log.info(f"    Extracted {ir_type} from cache: {latest_file.name} ({len(ir_content)} bytes)")
					break

			if not ir_content:
				log.warning(f"  No IR files found in Triton cache for {kernel_name}")
				cache_files = list(self.triton_cache_dir.rglob("*"))
				log.debug(f"    Cache directory: {self.triton_cache_dir}")
				log.debug(f"    Total files in cache: {len(cache_files)}")
				log.debug(f"    Sample files: {[f.name for f in cache_files[:10]]}")
				return KernelIRCapture(
					kernel_name=kernel_name,
					module_file=str(triton_file),
					ir_type="",
					ir_content="",
					input_shapes=[],
					input_dtypes=[],
					numel_params={},
					other_params=[],
					grid=captured_metadata.get('grid', []),
					config={},
					success=False,
					error="No IR files in Triton cache"
				)

			# Extract tensor information from captured metadata
			tensor_args = captured_metadata.get('tensor_args', [])
			input_shapes = [t['shape'] for t in tensor_args]
			input_dtypes = [t['dtype'] for t in tensor_args]
			
			return KernelIRCapture(
				kernel_name=kernel_name,
				module_file=str(triton_file),
				ir_type=ir_type,
				ir_content=ir_content,
				input_shapes=input_shapes,
				input_dtypes=input_dtypes,
				numel_params={},
				other_params=[],
				grid=captured_metadata.get('grid', []),
				config={},
				success=True,
				error=""
			)

		except Exception as e:
			log.error(f"  Error capturing IR for {kernel_name}: {e}")
			log.debug(traceback.format_exc())
			return KernelIRCapture(
				kernel_name=kernel_name,
				module_file=str(triton_file),
				ir_type="",
				ir_content="",
				input_shapes=[],
				input_dtypes=[],
				numel_params={},
				other_params=[],
				grid=(),
				config={},
				success=False,
				error=str(e)
			)

	def process_triton_file(self, triton_file: Path) -> List[KernelIRCapture]:
		"""Process a single Triton module file and capture IR for all kernels"""
		log.info(f"Processing: {triton_file.name}")
		results = []

		try:
			# STEP 1: Run the model to capture kernel metadata AND get the module with cached kernels
			log.info(f"Step 1: Running model to capture kernel metadata...")
			captured_kernels, compiled_module = self.capture_model_inputs(triton_file)

			if not captured_kernels or not compiled_module:
				log.warning(f"  No kernels captured from model run")
			else:
				log.info(f"  Captured metadata for {len(captured_kernels)} kernels")

			# STEP 2: Extract IR from the compiled module's kernel caches
			log.info(f"Step 2: Extracting IR from compiled kernels...")

			if not compiled_module:
				log.warning(f"  No compiled module available")
				return results

			# Find all Triton kernels in the compiled module
			kernel_count = 0
			for name in dir(compiled_module):
				obj = getattr(compiled_module, name)
				if callable(obj) and name.startswith("triton_"):
					kernel_count += 1
					log.info(f"  Kernel {kernel_count}: {name}")

					# Get captured metadata for this kernel
					kernel_metadata = captured_kernels.get(name, {})

					# Capture IR for this kernel from the cache
					ir_capture = self.capture_kernel_ir(
						triton_file, name, obj, "", kernel_metadata
					)

					if ir_capture:
						results.append(ir_capture)

						# Save IR to file
						ir_filename = None
						if ir_capture.success and ir_capture.ir_content:
							ir_filename = f"{triton_file.stem}_{name}.{ir_capture.ir_type}"
							ir_path = self.ir_dir / ir_filename
							ir_path.write_text(ir_capture.ir_content)
							log.info(f"    Saved IR to: {ir_filename}")

						# Save tensor shapes separately for easy reuse
						if ir_capture.input_shapes:
							shapes_filename = f"{triton_file.stem}_{name}_shapes.json"
							shapes_path = self.shapes_dir / shapes_filename
							
							# Get full tensor metadata from captured metadata
							tensor_metadata = {
								'kernel_name': name,
								'module_file': triton_file.stem,
								'tensor_args': kernel_metadata.get('tensor_args', []),
								'scalar_args': kernel_metadata.get('scalar_args', []),
								'kwargs': kernel_metadata.get('kwargs', {}),
								'grid': kernel_metadata.get('grid', []),
								'total_args': kernel_metadata.get('args_count', 0),
								# Quick reference for shape creation
								'shapes': ir_capture.input_shapes,
								'dtypes': ir_capture.input_dtypes,
							}
							
							with open(shapes_path, 'w') as f:
								json.dump(tensor_metadata, f, indent=2)
							log.info(f"    Saved tensor shapes to: {shapes_filename}")

						# Save metadata
						meta_filename = f"{triton_file.stem}_{name}.json"
						meta_path = self.metadata_dir / meta_filename
						metadata = asdict(ir_capture)
						if ir_filename:
							metadata['ir_content'] = f"See ir_files/{ir_filename}"
						else:
							metadata['ir_content'] = "No IR captured"

						# Add captured kernel metadata to the output
						metadata['captured_from_model'] = kernel_metadata

						with open(meta_path, 'w') as f:
							json.dump(metadata, f, indent=2)

					self.cleanup_memory()

			log.info(f"  Found {kernel_count} kernels, captured {len(results)} IR files")

		except Exception as e:
			log.error(f"Failed to process {triton_file}: {e}")
			log.debug(traceback.format_exc())

		return results

	def process_all_files(self, limit: Optional[int] = None):
		"""Process all Triton module files"""
		triton_files = sorted(self.triton_modules_dir.glob("*_triton.py"))

		if limit:
			triton_files = triton_files[:limit]

		log.info(f"Found {len(triton_files)} Triton module files to process")

		total_kernels = 0
		successful_captures = 0

		for i, triton_file in enumerate(triton_files, 1):
			log.info(f"\n[{i}/{len(triton_files)}] {triton_file.name}")

			file_results = self.process_triton_file(triton_file)
			self.results.extend(file_results)

			total_kernels += len(file_results)
			successful_captures += sum(1 for r in file_results if r.success)

			log.info(f"  Progress: {successful_captures}/{total_kernels} kernels captured")

			self.cleanup_memory()

		# Save summary
		summary = {
			'total_files': len(triton_files),
			'total_kernels': total_kernels,
			'successful_captures': successful_captures,
			'failed_captures': total_kernels - successful_captures,
			'output_directory': str(self.output_dir),
			'results': [
				{
					'kernel_name': r.kernel_name,
					'module_file': r.module_file,
					'ir_type': r.ir_type,
					'success': r.success,
					'error': r.error
				}
				for r in self.results
			]
		}

		summary_path = self.output_dir / "capture_summary.json"
		with open(summary_path, 'w') as f:
			json.dump(summary, f, indent=2)

		log.info(f"\n{'='*60}")
		log.info(f"SUMMARY")
		log.info(f"{'='*60}")
		log.info(f"Files processed: {len(triton_files)}")
		log.info(f"Kernels found: {total_kernels}")
		log.info(f"IR captured: {successful_captures}")
		log.info(f"Failed: {total_kernels - successful_captures}")
		log.info(f"Output directory: {self.output_dir}")
		log.info(f"Summary saved to: {summary_path}")


def main():
	import argparse
	parser = argparse.ArgumentParser(
		description="Capture IR for all Triton kernels by executing them with real-shaped inputs"
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path(__file__).parent / "Data" / "data_64",
		help="Data directory containing triton_modules and pytorch_modules"
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).parent / "captured_ir",
		help="Output directory for captured IR and metadata"
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit number of files to process (for testing)"
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda:0",
		help="Device to use for kernel execution (default: cuda:0)"
	)

	args = parser.parse_args()

	if not TUNE_UTILS_AVAILABLE:
		log.error("tune_one_file utilities not available - cannot proceed")
		sys.exit(1)

	log.info("Starting Kernel IR Capture System")
	log.info(f"Data directory: {args.data_dir}")
	log.info(f"Output directory: {args.output_dir}")
	log.info(f"Device: {args.device}")

	system = KernelIRCaptureSystem(args.data_dir, args.output_dir, args.device)
	system.process_all_files(limit=args.limit)

	log.info("\nCapture complete!")


if __name__ == "__main__":
	main()
