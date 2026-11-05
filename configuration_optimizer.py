#!/usr/bin/env python3
"""
Configuration Optimizer Class

Uses DeepSeek API to analyze TTGIR and propose optimal kernel configurations.
"""

import json
import os
import re
import time
import importlib.util
import tempfile
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from openai import OpenAI
from dotenv import load_dotenv

from ttgir_analysis import TTGIRAnalysis
from ttgir_parser import TTGIRParser
from tensor_utils import create_kernel_inputs, load_tensor_shapes, print_tensor_info
from grid_utils import compute_grid_from_numel, infer_numel_for_kernel
from benchmark_utils import benchmark_kernel, benchmark_with_statistics, detect_kernel_type


class ConfigurationOptimizer:
	"""Uses DeepSeek API to analyze TTGIR and propose optimal configurations"""

	def __init__(self, api_key: Optional[str] = None, use_cached_response: bool = False, model: str = "deepseek-chat", prompt_output_dir: Optional[Path] = None):
		load_dotenv()
		# Prefer explicit key arg, then env vars (DEEPSEEK_API_KEY or OPENAI_API_KEY)
		resolved_api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
		self.use_cached_response = use_cached_response
		self.model = model
		self.prompt_output_dir = prompt_output_dir

		# Create prompt output directory if specified
		if self.prompt_output_dir:
			self.prompt_output_dir = Path(self.prompt_output_dir)
			self.prompt_output_dir.mkdir(parents=True, exist_ok=True)
			print(f"Prompt output directory: {self.prompt_output_dir}")

		if not use_cached_response:
			if not resolved_api_key:
				raise ValueError("Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY, or pass api_key explicitly.")

			self.api_key = resolved_api_key
			# Also export to OPENAI_API_KEY so downstream OpenAI() clients without explicit keys work
			os.environ.setdefault("OPENAI_API_KEY", self.api_key)
			os.environ.setdefault("DEEPSEEK_API_KEY", self.api_key)

			self.client = OpenAI(
				api_key=self.api_key,
				base_url="https://api.deepseek.com"
			)
		else:
			self.client = None

	def analyze_and_optimize(self,
							ttgir_path: Path,
							torch_model: str,
							metadata: Dict[str, Any],
							input_shapes_json: Path,
							target_gpu: str = "cuda:70",
							triton_modules_dir: Optional[Path] = None,
							cached_response_path: Optional[Path] = None,
							model: Optional[str] = None) -> Dict[str, Any]:
		"""
		Analyze TTGIR and metadata, then propose optimal configurations

		Args:
			ttgir_path: Path to .ttgir file
			metadata_path: Path to .json metadata file
			target_gpu: Target GPU architecture (e.g., "cuda:70" for V100)
			triton_modules_dir: Path to triton_modules directory (optional)
			cached_response_path: Path to cached JSON response file (for debugging)

		Returns:
			Dictionary with analysis and recommended configurations
		"""
		# Parse TTGIR
		ttgir_analysis = TTGIRParser.parse_ttgir_file(ttgir_path)

		# Update analysis with metadata
		ttgir_analysis.shared_memory_bytes = metadata.get('shared', 0)
		ttgir_analysis.compute_capability = metadata.get('arch', '')

		# Read TTGIR content for context
		with open(ttgir_path, 'r') as f:
			ttgir_content = f.read()

		# Read TTIR content (original Triton IR before GPU-specific lowering)
		ttir_path = ttgir_path.with_suffix('.ttir')
		ttir_content = ""
		if ttir_path.exists():
			with open(ttir_path, 'r') as f:
				ttir_content = f.read()

		# Try to find original Triton source code using loc information from TTGIR
		triton_source = ""
		kernel_name = metadata.get('name', '')
		triton_content = ""

		# Always try loc-based extraction from TTGIR first (does not need triton_modules_dir)
		source_file, line_range = ConfigurationOptimizer._extract_source_location(ttgir_content)
		if source_file:
			print(f"Found source location from TTGIR: {source_file}, lines {line_range}")
			triton_source = ConfigurationOptimizer._extract_source_from_file(source_file, line_range)
			if triton_source and Path(source_file).exists():
				with open(source_file, 'r') as f:
					triton_content = f.read()

		# Fallback: search triton_modules_dir if loc-based extraction failed
		if not triton_source and triton_modules_dir and kernel_name:
			triton_source = self._find_triton_source(triton_modules_dir, kernel_name)

		if not triton_source:
			raise RuntimeError(f"Could not extract source location from TTGIR for kernel {kernel_name}")

		# Load tensor shapes from input_shapes_json
		tensor_shapes_info = None
		if input_shapes_json and input_shapes_json.exists():
			try:
				tensor_shapes_info = load_tensor_shapes(input_shapes_json)
				print(f"Loaded tensor shapes from: {input_shapes_json}")
			except Exception as e:
				print(f"Warning: Could not load tensor shapes from {input_shapes_json}: {e}")

		input_tensors_shape = self._format_tensor_shapes(tensor_shapes_info) if tensor_shapes_info else "Tensor shape information not available"

		# Generate prompt
		prompt = self._generate_analysis_prompt(
			ttgir_analysis, metadata, ttgir_content, ttir_content,
			triton_source, triton_content, target_gpu, input_tensors_shape
		)		# Save prompt to file if output directory specified
		if self.prompt_output_dir and torch_model and kernel_name:
			# Create subdirectory for this torch_model
			model_dir = self.prompt_output_dir / torch_model
			model_dir.mkdir(parents=True, exist_ok=True)

			# Save prompt with kernel_name as filename
			prompt_file = model_dir / f"{kernel_name}_prompt.txt"
			with open(prompt_file, 'w') as f:
				f.write(prompt)
			print(f"Prompt saved to: {prompt_file}")

		# Get LLM response - either from cache or API
		# Use model parameter if provided, otherwise use instance model
		current_model = model if model is not None else self.model

		if self.use_cached_response and cached_response_path:
			print(f"Loading cached response from: {cached_response_path}")
			try:
				with open(cached_response_path, 'r') as f:
					cached_data = json.load(f)
					# Extract just the recommendations part if it's a full result
					if "recommendations" in cached_data:
						config_recommendations = cached_data["recommendations"]
					else:
						config_recommendations = cached_data
				print("Successfully loaded cached response")
			except Exception as e:
				print(f"Error loading cached response: {e}")
				print("Falling back to API call...")
				response = self._call_deepseek(prompt, model=current_model)
				try:
					config_recommendations = json.loads(response)
				except json.JSONDecodeError:
					config_recommendations = {"raw_response": response}
		else:
			print(f"Analyzing with DeepSeek API (model: {current_model})...")
			response = self._call_deepseek(prompt, model=current_model)

			# Parse response
			try:
				config_recommendations = json.loads(response)
			except json.JSONDecodeError:
				# If not valid JSON, wrap it
				config_recommendations = {"raw_response": response}

		result = {
			"kernel_name": metadata.get('name', 'unknown'),
			"current_config": {
				"num_warps": ttgir_analysis.num_warps,
				"num_ctas": ttgir_analysis.num_ctas,
				"shared_memory_bytes": ttgir_analysis.shared_memory_bytes,
				"blocked_layouts": ttgir_analysis.blocked_layouts
			},
			"analysis": {
				"memory_characteristics": {
					"global_loads": ttgir_analysis.global_loads,
					"global_stores": ttgir_analysis.global_stores,
					"shared_memory_bytes": ttgir_analysis.shared_memory_bytes,
					"eviction_policies": ttgir_analysis.eviction_policies
				},
				"compute_characteristics": ttgir_analysis.operations,
				"input_tensor_shapes": tensor_shapes_info,
				"parallelism": {
					"num_warps": ttgir_analysis.num_warps,
					"threads_per_warp": ttgir_analysis.threads_per_warp,
					"num_ctas": ttgir_analysis.num_ctas
				}
			},
			"recommendations": config_recommendations,
			"target_architecture": target_gpu
		}

		# Save result to file if output directory specified
		if self.prompt_output_dir and torch_model and kernel_name:
			# Save result with kernel_name as filename
			result_file = model_dir / f"{kernel_name}_result.json"
			with open(result_file, 'w') as f:
				json.dump(result, f, indent=2)
			print(f"Result saved to: {result_file}")
			# from rich import print_json
			# print_json(data=result)

		return result

	@staticmethod
	def _format_tensor_shapes(tensor_shapes_info: Dict) -> str:
		"""Format tensor shapes information for the prompt"""
		if not tensor_shapes_info:
			return "No tensor shape information available"

		formatted = []

		# Format tensor arguments
		if "tensor_args" in tensor_shapes_info:
			formatted.append("Tensor Arguments:")
			for arg in tensor_shapes_info["tensor_args"]:
				arg_idx = arg.get("arg_index", "?")
				shape = arg.get("shape", "N/A")
				dtype = arg.get("dtype", "N/A")
				device = arg.get("device", "N/A")
				numel = arg.get("numel", "N/A")
				stride = arg.get("stride", "N/A")
				is_contiguous = arg.get("is_contiguous", "N/A")

				if isinstance(numel, int):
					formatted.append(f"  arg{arg_idx}: shape={shape}, dtype={dtype}, numel={numel:,}, stride={stride}, contiguous={is_contiguous}")
				else:
					formatted.append(f"  arg{arg_idx}: shape={shape}, dtype={dtype}, numel={numel}, stride={stride}, contiguous={is_contiguous}")

		# Format scalar arguments
		if "scalar_args" in tensor_shapes_info:
			formatted.append("\nScalar Arguments:")
			for arg in tensor_shapes_info["scalar_args"]:
				arg_idx = arg.get("arg_index", "?")
				value = arg.get("value", "N/A")
				arg_type = arg.get("type", "N/A")
				formatted.append(f"  arg{arg_idx}: value={value}, type={arg_type}")

		# Format kwargs
		if "kwargs" in tensor_shapes_info:
			formatted.append("\nKeyword Arguments:")
			for key, value in tensor_shapes_info["kwargs"].items():
				if key == "warmup":
					continue  # Skip warmup
				if isinstance(value, dict):
					val_type = value.get("type", "N/A")
					val = value.get("value", "N/A")
					val_type_detail = value.get("value_type", "N/A")
					formatted.append(f"  {key}: {val} (type={val_type}, value_type={val_type_detail})")
				else:
					formatted.append(f"  {key}: {value}")

		# Add grid information
		if "grid" in tensor_shapes_info:
			grid = tensor_shapes_info["grid"]
			formatted.append(f"\nGrid Dimensions: {grid}")

		# Add summary statistics
		total_elements = 0
		total_memory_bytes = 0

		if "tensor_args" in tensor_shapes_info:
			for arg in tensor_shapes_info["tensor_args"]:
				if "numel" in arg and isinstance(arg["numel"], int):
					total_elements += arg["numel"]

					# Estimate memory based on dtype
					dtype = arg.get("dtype", "torch.float32")
					bytes_per_element = 4  # Default for float32
					if "float16" in str(dtype) or "half" in str(dtype):
						bytes_per_element = 2
					elif "float64" in str(dtype) or "double" in str(dtype):
						bytes_per_element = 8
					elif "int64" in str(dtype) or "long" in str(dtype):
						bytes_per_element = 8
					elif "int32" in str(dtype) or "int" in str(dtype):
						bytes_per_element = 4
					elif "int16" in str(dtype) or "short" in str(dtype):
						bytes_per_element = 2
					elif "int8" in str(dtype) or "byte" in str(dtype):
						bytes_per_element = 1

					total_memory_bytes += arg["numel"] * bytes_per_element

		if total_elements > 0:
			formatted.append(f"\nSummary:")
			formatted.append(f"  Total tensor elements: {total_elements:,}")
			formatted.append(f"  Estimated total memory: {total_memory_bytes / (1024**2):.2f} MB ({total_memory_bytes / (1024**3):.4f} GB)")
			if "tensor_args" in tensor_shapes_info:
				formatted.append(f"  Number of tensor arguments: {len(tensor_shapes_info['tensor_args'])}")
			if "scalar_args" in tensor_shapes_info:
				formatted.append(f"  Number of scalar arguments: {len(tensor_shapes_info['scalar_args'])}")

		return "\n".join(formatted)

	@staticmethod
	def _extract_source_location(ttgir_content: str) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
		"""Extract source file path and line range from TTGIR loc attributes"""
		try:
			# Look for loc attributes in TTGIR that point to original source
			# Format: loc("path/to/file.py":line:column)
			loc_pattern = r'loc\("([^"]+\.py)":(\d+):(\d+)\)'
			matches = re.findall(loc_pattern, ttgir_content)

			if matches:
				# Get the most common file path (the actual kernel file)
				file_paths = {}
				for path, line, col in matches:
					if path not in file_paths:
						file_paths[path] = []
					file_paths[path].append(int(line))

				# Filter out framework/library files (site-packages, dist-packages,
				# stdlib) — these are inlined dependencies (e.g. triton.language.random)
				# not the actual user kernel.
				_LIBRARY_MARKERS = ("site-packages", "dist-packages", "/lib/python")
				user_paths = {
					p: lines for p, lines in file_paths.items()
					if not any(m in p for m in _LIBRARY_MARKERS)
				}
				# Prefer user files; fall back to all if none remain
				candidates = user_paths if user_paths else file_paths

				# Find the file with the most references
				if candidates:
					most_common_file = max(candidates.items(), key=lambda x: len(x[1]))[0]
					lines = candidates[most_common_file]
					min_line = min(lines)
					max_line = max(lines)
					# Add some padding to capture the full function
					return (most_common_file, (max(1, min_line - 5), max_line))

		except Exception as e:
			print(f"Error extracting source location from TTGIR: {e}")

		return (None, None)

	@staticmethod
	def _extract_source_from_file(file_path: str, line_range: Optional[Tuple[int, int]]) -> str:
		"""Extract source code from a file given a line range"""
		try:
			# Convert to Path and check if file exists
			source_path = Path(file_path)
			if not source_path.exists():
				print(f"Source file not found: {file_path}")
				return ""

			with open(source_path, 'r') as f:
				lines = f.readlines()

			if line_range:
				start_line, end_line = line_range
				# Adjust for 0-based indexing
				start_idx = max(0, start_line - 1)
				end_idx = min(len(lines), end_line)
				source = ''.join(lines[start_idx:end_idx])
			else:
				source = ''.join(lines)

			print(f"Extracted {end_idx - start_idx} lines from {file_path}")
			return source

		except Exception as e:
			print(f"Error reading source file {file_path}: {e}")
			return ""

	def _find_triton_source(self, triton_modules_dir: Path, kernel_name: str) -> str:
		"""Find and extract original Triton source code for a kernel"""
		# Extract base module name from kernel name
		# e.g., "triton_poi_fused_add_convolution_2" -> look in files for this function

		# Search all Python files in triton_modules directory
		for py_file in triton_modules_dir.glob("*.py"):
			try:
				with open(py_file, 'r') as f:
					content = f.read()

				# Check if this file contains the kernel function
				if f"def {kernel_name}(" in content:
					# Extract the function definition
					# Match from def to the next def/class or end of string
					pattern = rf'(def {kernel_name}\([^)]*\).*?)(?=\n(?:def |class |$))'
					match = re.search(pattern, content, re.DOTALL)
					if match:
						source = match.group(1).strip()
						# If multiple matches (same kernel name in different files),
						# prefer longer definitions or first match
						print(f"Found Triton source in: {py_file}")
						return source
			except Exception:
				continue

		return ""

	@staticmethod
	def _extract_block_sizes_from_ttgir(ttgir_content: str) -> Dict[str, int]:
		"""Extract block sizes from TTGIR make_range patterns"""
		block_sizes = {}
		# Pattern: tt.make_range {end = N : i32, start = 0 : i32}
		pattern = r'tt\.make_range\s*\{end\s*=\s*(\d+)\s*:\s*i32'
		matches = re.findall(pattern, ttgir_content)

		if not matches:
			return block_sizes

		# Get unique values while preserving order
		unique_matches = []
		seen = set()
		for match in matches:
			if match not in seen:
				unique_matches.append(match)
				seen.add(match)

		# Assign block sizes based on number of unique dimensions
		if len(unique_matches) == 1:
			# 1D kernel: use BLOCK_SIZE
			block_sizes['BLOCK_SIZE'] = int(unique_matches[0])
		elif len(unique_matches) == 2:
			# 2D kernel: XBLOCK and YBLOCK
			block_sizes['XBLOCK'] = int(unique_matches[0])
			block_sizes['YBLOCK'] = int(unique_matches[1])
		elif len(unique_matches) >= 3:
			# 3D or reduction kernel: XBLOCK, YBLOCK, and ZBLOCK/RBLOCK
			block_sizes['XBLOCK'] = int(unique_matches[0])
			block_sizes['YBLOCK'] = int(unique_matches[1])
			block_sizes['ZBLOCK'] = int(unique_matches[2])
			# If there are more, add them as R0_BLOCK, R1_BLOCK, etc.
			for i, match in enumerate(unique_matches[3:]):
				block_sizes[f'R{i}_BLOCK'] = int(match)

		return block_sizes

	def _generate_analysis_prompt(self,
								  analysis: TTGIRAnalysis,
								  metadata: Dict,
								  ttgir_content: str,
								  ttir_content: str,
								  triton_source: str,
								  triton_content: str,
								  target_gpu: str,
								  input_tensors_shape: Optional[str] = None) -> str:
		"""Generate comprehensive analysis prompt for DeepSeek"""
		# GPU architecture database
		gpu_specs = {
			"cuda:70": {
				"name": "Tesla V100",
				"compute_capability": "7.0",
				"shared_memory_per_sm": 96 * 1024,  # 96 KB
				"l2_cache": 6 * 1024 * 1024,  # 6 MB
				"memory_bandwidth": 900,  # GB/s
				"max_warps_per_sm": 64,
				"max_threads_per_sm": 2048
			},
			"cuda:80": {
				"name": "A100",
				"compute_capability": "8.0",
				"shared_memory_per_sm": 164 * 1024,  # 164 KB
				"l2_cache": 40 * 1024 * 1024,  # 40 MB
				"memory_bandwidth": 1555,  # GB/s
				"max_warps_per_sm": 64,
				"max_threads_per_sm": 2048
			},
			"cuda:86": {
				"name": "RTX 3090",
				"compute_capability": "8.6",
				"shared_memory_per_sm": 100 * 1024,  # 100 KB
				"l2_cache": 6 * 1024 * 1024,  # 6 MB
				"memory_bandwidth": 936,  # GB/s
				"max_warps_per_sm": 48,
				"max_threads_per_sm": 1536
			},
			"cuda:a40": {
				"name": "A40",
				"compute_capability": "8.6",
				"shared_memory_per_sm": 100 * 1024,  # 100 KB
				"l2_cache": 6 * 1024 * 1024,  # 6 MB
				"memory_bandwidth": 696,  # GB/s
				"max_warps_per_sm": 48,
				"max_threads_per_sm": 1536
			},
			"cuda:l40s": {
				"name": "L40S",
				"compute_capability": "8.9",
				"shared_memory_per_sm": 100 * 1024,  # 100 KB
				"l2_cache": 96 * 1024 * 1024,  # 96 MB
				"memory_bandwidth": 864,  # GB/s
				"max_warps_per_sm": 48,
				"max_threads_per_sm": 1536
			}
		}

		gpu_info = gpu_specs.get(target_gpu, gpu_specs["cuda:70"])

		# Extract numel information from kernel source or metadata
		numel_info = ""

		# Initialize grid_str with default value
		grid_str = "Grid dimensions not available"

		# Initialize kernel_type_specific_block_format with default
		kernel_type_specific_block_format = '{"XBLOCK": <int>, "YBLOCK": <int if applicable>}'

		if triton_source:
			# Try to extract numel from kernel source
			try:
				numel_param_names, numel_values, other_param_names = infer_numel_for_kernel(triton_source, metadata.get('name', ''))
				if numel_param_names and numel_values:
					numel_dict = dict(zip(numel_param_names, numel_values))
					numel_info = "\n# Data Volume (from kernel source)\n"
					for name, value in numel_dict.items():
						numel_info += f"- {name}: {value:,} elements\n"

					# Detect kernel type for dynamic prompt generation
					all_param_names = numel_param_names + (other_param_names or [])
					kernel_type = detect_kernel_type(all_param_names)
					numel_info += f"\n# Kernel Type: {kernel_type}\n"

					# Generate kernel type specific block format
					if kernel_type == '1d':
						kernel_type_specific_block_format = '{"XBLOCK": <int>}'
					elif kernel_type == '2d':
						kernel_type_specific_block_format = '{"XBLOCK": <int>, "YBLOCK": <int>}'
					elif kernel_type in ['reduction', 'reduction_fixed']:
						kernel_type_specific_block_format = '{"XBLOCK": <int>, "R0_BLOCK": <int>}'
					else:
						kernel_type_specific_block_format = '{"XBLOCK": <int>, "YBLOCK": <int if applicable>}'
			except:
				raise RuntimeError(f"Could not infer numel from kernel source for kernel {metadata.get('name', '')}")


		kernel_name = metadata.get("name", "")
		call_pattern = rf'{kernel_name}\s*\[([^\]]+)\]\s*\(([^)]+)\)'
		call_match = re.search(call_pattern, triton_content, re.DOTALL)

		if call_match:
			grid_str = call_match.group(1)
		else:
			raise RuntimeError(f"Could not find kernel call pattern in Triton source for kernel {kernel_name}")

		prompt = f"""You are an expert in GPU kernel optimization and memory hierarchy analysis.# Task
Analyze the Triton GPU IR (TTGIR) and propose optimal kernel configurations for the target GPU architecture.
Focus on memory access patterns, cache behavior (L1/L2 cache hits/misses), shared memory usage, thread occupancy,
and **CTA-level parallelism via grid dimensions**.

# Target GPU Architecture
- Name: {gpu_info['name']}
- Compute Capability: {gpu_info['compute_capability']}
- Shared Memory per SM: {gpu_info['shared_memory_per_sm'] / 1024:.0f} KB
- L2 Cache: {gpu_info['l2_cache'] / (1024*1024):.0f} MB
- Memory Bandwidth: {gpu_info['memory_bandwidth']} GB/s
- Max Warps per SM: {gpu_info['max_warps_per_sm']}
- Max Threads per SM: {gpu_info['max_threads_per_sm']}
{numel_info}
# Current Kernel Configuration
- Kernel Name: {metadata.get('name', 'unknown')}
- Grid Dimensions: {grid_str} (parsed from kernel launch call)
- Number of Warps per CTA: {analysis.num_warps}
- Number of CTAs: {analysis.num_ctas}
- Threads per Warp: {analysis.threads_per_warp}
- Shared Memory Usage per CTA: {analysis.shared_memory_bytes} bytes ({analysis.shared_memory_bytes / 1024:.2f} KB)

# Memory Access Analysis
- Global Memory Loads: {analysis.global_loads}
- Global Memory Stores: {analysis.global_stores}
- Eviction Policies: {', '.join(analysis.eviction_policies) if analysis.eviction_policies else 'None specified'}

# Input Tensor Shapes (from actual kernel invocation)
{input_tensors_shape}

# Block Layout Information
{json.dumps(analysis.blocked_layouts, indent=2)}

Blocked layouts show:
- sizePerThread: Elements processed per thread
- threadsPerWarp: Thread arrangement within warp
- warpsPerCTA: Warp arrangement within CTA
- order: Memory access order (affects coalescing)

# Operation Counts
- Arithmetic Operations: {analysis.operations['arithmetic']}
- Memory Operations: {analysis.operations['memory']}
- Tensor Operations: {analysis.operations['tensor']}
- Compute Intensity: {'High' if analysis.operations['arithmetic'] > analysis.operations['memory'] * 10 else 'Low' if analysis.operations['arithmetic'] < analysis.operations['memory'] else 'Medium'}

# Original Triton Kernel Source Code
{f'''
```python
{triton_source}
```
''' if triton_source else "Original Triton source not available - analyze from IR only"}

# TTIR (Triton IR - High-level representation before GPU-specific lowering)
{f'''
```mlir
{ttir_content[:3000]}
{"..." if len(ttir_content) > 3000 else ""}
```
''' if ttir_content else "TTIR not available"}

# TTGIR (Triton GPU IR - GPU-specific with blocked layouts and memory hierarchy)
```mlir
{ttgir_content[:3000]}
{"..." if len(ttgir_content) > 3000 else ""}
```

# Analysis Framework

Please analyze:

1. **Memory Access Patterns**:
   - Are memory accesses coalesced? (based on blocked layout order and threadsPerWarp)
   - Stride patterns from sizePerThread and tensor shapes
   - Potential cache line utilization

2. **Cache Behavior Prediction**:
   - L2 cache hit probability (data reuse across CTAs)
   - L1 cache effectiveness (data reuse within SM)
   - Shared memory vs L1 cache trade-offs

3. **Shared Memory Analysis**:
   - Current usage: {analysis.shared_memory_bytes / 1024:.2f} KB / {gpu_info['shared_memory_per_sm'] / 1024:.0f} KB available
   - Bank conflicts potential (based on threadsPerWarp arrangement)
   - Opportunity for increased shared memory usage

4. **Thread Occupancy Within Each CTA**:
   - Current: {analysis.num_warps} warps, {analysis.num_warps * analysis.threads_per_warp} threads per CTA
   - Shared memory limits occupancy
   - Register pressure (estimated from operation complexity)

5. **CTA-Level Parallelism and Grid Dimensions** (CRITICAL):
   - **Total data volume**: Consider the numel values above - this determines how many CTAs will be launched
   - **Grid dimensions**: Calculated as ceil(numel / BLOCK_SIZE) for each dimension
   - **Block size trade-offs**:
     * Larger BLOCK → Fewer CTAs → Less parallelism but better resource utilization per CTA
     * Smaller BLOCK → More CTAs → More parallelism but potential resource waste
   - **GPU saturation**: Different block sizes create different grid sizes based on numel values
   - **Example**: For xnumel=N, XBLOCK=128 creates ceil(N/128) CTAs, XBLOCK=256 creates ceil(N/256) CTAs
   - **Optimal strategy**: Balance between:
     * Enough CTAs to saturate GPU (hide memory latency, utilize all SMs)
     * Enough work per CTA to amortize launch overhead
     * Fit within resource constraints (shared memory, registers)

6. **Configuration Recommendations**:
   - Optimal num_warps (balance occupancy vs resources per CTA)
   - **Block size adjustments considering TOTAL DATA VOLUME and grid parallelism**:
     * XBLOCK, YBLOCK: Impact grid dimensions directly (grid = ceil(numel/BLOCK))
     * Consider how many CTAs will be created with different block sizes
     * Ensure sufficient CTAs to saturate the GPU
   - Eviction policy recommendations
   - Potential layout reordering for better coalescing

# Output Format

Provide your analysis as JSON:

{{
  "memory_analysis": {{
	"coalescing_efficiency": "high/medium/low",
	"predicted_l2_hit_rate": "percentage estimate with reasoning",
	"predicted_l1_hit_rate": "percentage estimate with reasoning",
	"shared_memory_efficiency": "description",
	"memory_bottleneck": "yes/no with explanation"
  }},
  "occupancy_analysis": {{
	"current_occupancy_percentage": "estimate",
	"occupancy_limited_by": "shared_memory/registers/warps",
	"recommended_occupancy": "target percentage"
  }},
  "grid_parallelism_analysis": {{
	"total_elements": <total numel value>,
	"current_grid_estimate": "estimated CTAs with current block sizes",
	"grid_saturation": "sufficient/insufficient - does grid create enough CTAs to saturate GPU?",
	"block_size_impact": "how different block sizes affect grid dimensions and parallelism"
  }},
  "recommended_configurations": [
	{{
	  "config_id": 1,
	  "num_warps": <int>,
	  "block_size": {kernel_type_specific_block_format},
	  "num_stages": <int>,
	  "eviction_policy": "<policy>",
	  "expected_grid_size": "estimated number of CTAs that will be launched",
	  "grid_saturation_level": "high/medium/low - how well this saturates the GPU",
	  "expected_improvement": "percentage and bottleneck addressed",
	  "reasoning": "why this configuration is better, INCLUDING grid-level parallelism considerations",
	  "includes_current": <true/false>
	}},
	{{
	  "config_id": 2,
	  "...": "2-3 additional configurations to try"
	}}
  ],
  "current_config_assessment": {{
	"is_near_optimal": <true/false>,
	"bottlenecks": ["list of current bottlenecks"],
	"strengths": ["what current config does well"]
  }},
  "key_insights": [
	"insight 1: specific observation about memory/compute patterns",
	"insight 2: architecture-specific optimization opportunity",
	"insight 3: trade-off consideration"
  ]
}}

IMPORTANT:
- **Consider the total data volume (numel) when recommending block sizes**
- **Grid dimensions = ceil(numel / BLOCK_SIZE) - this is the primary source of parallelism**
- Different block sizes create different numbers of CTAs, affecting overall GPU utilization
- A configuration with better per-CTA efficiency but insufficient grid parallelism may underperform
- Balance: enough CTAs to saturate GPU + enough work per CTA to be efficient
- If the current configuration is already good/near-optimal, include it as one of the recommended configurations
- Provide 3-5 configurations total (including current if good)
- Focus on configurations that address different potential bottlenecks
- Be specific about expected improvements and reasoning
- **Always mention how block size changes affect grid size and overall parallelism**
"""

		return prompt

	def _call_deepseek(self, prompt: str,
					   model: str = "deepseek-chat",
					   temperature: float = 0.1) -> str:
		"""Call DeepSeek API with reasoning model support"""

		try:
			# Build messages
			messages = [
				{
					"role": "system",
					"content": "You are an expert GPU performance engineer specializing in memory hierarchy optimization and kernel tuning. Provide detailed, actionable analysis in JSON format."
				},
				{
					"role": "user",
					"content": prompt
				}
			]

			# Use response_format for all models
			response = self.client.chat.completions.create(
				model=model,
				messages=messages,
				temperature=temperature,
				response_format={"type": "json_object"}
			)

			return response.choices[0].message.content

		except Exception as e:
			print(f"Error calling DeepSeek API: {e}")
			return json.dumps({"error": str(e)})