#!/usr/bin/env python3
"""
TTGIR Configuration Optimizer using DeepSeek API

Analyzes Triton GPU IR (TTGIR) and metadata to propose optimal kernel configurations
for a given GPU architecture, including memory access patterns and cache behavior.
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
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import torch

# Import tensor utilities for recreating inputs from captured shapes
# sys.path.insert(0, str(Path(__file__).parent))
from tensor_utils import create_kernel_inputs, load_tensor_shapes, print_tensor_info
from grid_utils import compute_grid_from_numel, infer_numel_for_kernel
from benchmark_utils import benchmark_kernel, benchmark_with_statistics, detect_kernel_type


@dataclass
class TTGIRAnalysis:
	"""Analysis results from TTGIR"""
	# Block layout information
	blocked_layouts: List[Dict[str, Any]]

	# Memory characteristics
	shared_memory_bytes: int
	global_loads: int
	global_stores: int

	# Thread configuration
	num_warps: int
	num_ctas: int
	threads_per_warp: int

	# Architecture info
	target_arch: str
	compute_capability: str

	# Memory access patterns
	eviction_policies: List[str]
	tensor_shapes: List[str]

	# Operations
	operations: Dict[str, int]


class TTGIRParser:
	"""Parse TTGIR files to extract optimization-relevant information"""

	@staticmethod
	def parse_ttgir_file(ttgir_path: Path) -> TTGIRAnalysis:
		"""Parse TTGIR file and extract key information"""
		with open(ttgir_path, 'r') as f:
			content = f.read()

		# Extract blocked layouts
		blocked_layouts = TTGIRParser._extract_blocked_layouts(content)

		# Extract tensor shapes
		tensor_shapes = TTGIRParser._extract_tensor_shapes(content)

		# Count operations
		operations = TTGIRParser._count_operations(content)

		# Extract eviction policies
		eviction_policies = re.findall(r'evictionPolicy = (\w+)', content)

		# Extract memory operations
		global_loads = content.count('tt.load')
		global_stores = content.count('tt.store')

		# Get module attributes
		module_attrs = TTGIRParser._extract_module_attributes(content)

		return TTGIRAnalysis(
			blocked_layouts=blocked_layouts,
			shared_memory_bytes=0,  # Will be filled from metadata
			global_loads=global_loads,
			global_stores=global_stores,
			num_warps=module_attrs.get('num_warps', 0),
			num_ctas=module_attrs.get('num_ctas', 1),
			threads_per_warp=module_attrs.get('threads_per_warp', 32),
			target_arch=module_attrs.get('target', 'unknown'),
			compute_capability='',  # Will be filled from metadata
			eviction_policies=list(set(eviction_policies)),
			tensor_shapes=tensor_shapes,
			operations=operations
		)

	@staticmethod
	def _extract_blocked_layouts(content: str) -> List[Dict[str, Any]]:
		"""Extract blocked layout information"""
		layouts = []
		pattern = r'#blocked\d* = #ttg\.blocked<{([^}]+)}>'

		for match in re.finditer(pattern, content):
			layout_str = match.group(1)
			layout = {}

			# Parse layout attributes
			for attr in ['sizePerThread', 'threadsPerWarp', 'warpsPerCTA', 'order']:
				attr_pattern = rf'{attr}\s*=\s*\[([^\]]+)\]'
				attr_match = re.search(attr_pattern, layout_str)
				if attr_match:
					values = [int(x.strip()) for x in attr_match.group(1).split(',')]
					layout[attr] = values

			if layout:
				layouts.append(layout)

		return layouts

	@staticmethod
	def _extract_tensor_shapes(content: str) -> List[str]:
		"""Extract tensor shapes from the IR"""
		shapes = set()
		pattern = r'tensor<([^>]+)>'

		for match in re.finditer(pattern, content):
			shape_str = match.group(1)
			# Only keep shape dimensions, not full type info
			if 'x' in shape_str:
				shape_parts = shape_str.split(',')[0].strip()
				shapes.add(shape_parts)

		return list(shapes)

	@staticmethod
	def _count_operations(content: str) -> Dict[str, int]:
		"""Count different types of operations"""
		ops = {
			'arithmetic': 0,
			'memory': 0,
			'tensor': 0,
			'control_flow': 0
		}

		# Arithmetic operations
		arithmetic_ops = ['arith.addi', 'arith.muli', 'arith.addf', 'arith.mulf',
						 'arith.divsi', 'arith.remsi']
		for op in arithmetic_ops:
			ops['arithmetic'] += content.count(op)

		# Memory operations
		ops['memory'] = content.count('tt.load') + content.count('tt.store')

		# Tensor operations
		tensor_ops = ['tt.broadcast', 'tt.expand_dims', 'tt.addptr',
					 'ttg.convert_layout', 'tt.splat']
		for op in tensor_ops:
			ops['tensor'] += content.count(op)

		# Control flow
		ops['control_flow'] = content.count('scf.') + content.count('cf.')

		return ops

	@staticmethod
	def _extract_module_attributes(content: str) -> Dict[str, Any]:
		"""Extract module-level attributes"""
		attrs = {}

		# Extract module attributes
		module_pattern = r'module attributes \{([^}]+)\}'
		match = re.search(module_pattern, content)
		if match:
			attrs_str = match.group(1)

			# Parse num-warps
			warps_match = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', attrs_str)
			if warps_match:
				attrs['num_warps'] = int(warps_match.group(1))

			# Parse num-ctas
			ctas_match = re.search(r'"ttg\.num-ctas"\s*=\s*(\d+)', attrs_str)
			if ctas_match:
				attrs['num_ctas'] = int(ctas_match.group(1))

			# Parse threads-per-warp
			threads_match = re.search(r'"ttg\.threads-per-warp"\s*=\s*(\d+)', attrs_str)
			if threads_match:
				attrs['threads_per_warp'] = int(threads_match.group(1))

			# Parse target
			target_match = re.search(r'ttg\.target\s*=\s*"([^"]+)"', attrs_str)
			if target_match:
				attrs['target'] = target_match.group(1)

		return attrs


class ConfigurationOptimizer:
	"""Uses DeepSeek API to analyze TTGIR and propose optimal configurations"""

	def __init__(self, api_key: Optional[str] = None, use_cached_response: bool = False, model: str = "deepseek-chat", prompt_output_dir: Optional[Path] = None):
		load_dotenv()
		self.use_cached_response = use_cached_response
		self.model = model
		self.prompt_output_dir = prompt_output_dir

		# Create prompt output directory if specified
		if self.prompt_output_dir:
			self.prompt_output_dir = Path(self.prompt_output_dir)
			self.prompt_output_dir.mkdir(parents=True, exist_ok=True)
			print(f"Prompt output directory: {self.prompt_output_dir}")

		if not use_cached_response:
			self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
			if not self.api_key:
				raise ValueError("DEEPSEEK_API_KEY not found in environment or arguments")

			self.client = OpenAI(
				api_key=self.api_key,
				base_url="https://api.deepseek.com"
			)
		else:
			self.client = None

	def analyze_and_optimize(self,
							ttgir_path: Path,
							torch_model: str,
							metadata_path: Path,
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

		# Load metadata
		with open(metadata_path, 'r') as f:
			metadata = json.load(f)

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
		if triton_modules_dir and kernel_name:
			# First try to extract source location from TTGIR loc attributes
			source_file, line_range = self._extract_source_location(ttgir_content)
			if source_file:
				print(f"Found source location from TTGIR: {source_file}, lines {line_range}")
				triton_source = self._extract_source_from_file(source_file, line_range)
				with open(source_file, 'r') as f:
					triton_content = f.read()

			if not triton_source:
				raise RuntimeError(f"Could not extract source location from TTGIR for kernel {kernel_name}")

		# Generate prompt
		prompt = self._generate_analysis_prompt(
			ttgir_analysis, metadata, ttgir_content, ttir_content,
			triton_source, triton_content, target_gpu
		)

		# Save prompt to file if output directory specified
		if self.prompt_output_dir and torch_model and kernel_name:
			# Create subdirectory for this torch_model
			model_dir = self.prompt_output_dir / torch_model
			model_dir.mkdir(parents=True, exist_ok=True)

			# Save prompt with kernel_name as filename
			prompt_file = model_dir / f"{kernel_name}_prompt.txt"
			with open(prompt_file, 'w') as f:
				f.write(prompt)
			print(f"Prompt saved to: {prompt_file}")

		exit()

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
				"tensor_shapes": ttgir_analysis.tensor_shapes,
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

				# Find the file with the most references
				if file_paths:
					most_common_file = max(file_paths.items(), key=lambda x: len(x[1]))[0]
					lines = file_paths[most_common_file]
					min_line = min(lines)
					max_line = max(lines)
					# Add some padding to capture the full function
					return (most_common_file, (max(1, min_line - 5), max_line + 10))

		except Exception as e:
			print(f"Error extracting source location from TTGIR: {e}")

		return (None, None)

	def _extract_source_from_file(self, file_path: str, line_range: Optional[Tuple[int, int]]) -> str:
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

	def _generate_analysis_prompt(self,
								  analysis: TTGIRAnalysis,
								  metadata: Dict,
								  ttgir_content: str,
								  ttir_content: str,
								  triton_source: str,
								  triton_content: str,
								  target_gpu: str) -> str:
		"""Generate comprehensive analysis prompt for DeepSeek"""		# GPU architecture database
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
			print(metadata.get("name", ""))
			call_pattern = rf'{kernel_name}\s*\[([^\]]+)\]\s*\(([^)]+)\)'
			call_match = re.search(call_pattern, triton_content, re.DOTALL)

			if call_match:
				grid_str = call_match.group(1)
			else:
				raise RuntimeError(f"Could not find kernel call pattern in Triton source for kernel {kernel_name}")

		prompt = f"""You are an expert in GPU kernel optimization and memory hierarchy analysis.

# Task
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
- Tensor Shapes: {', '.join(analysis.tensor_shapes)}

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


def analyze_kernel_from_cache(cache_dir: Path, kernel_name: str, torch_model: str, target_gpu: str = "cuda:70",
							  triton_modules_dir: Optional[Path] = None,
							  use_cached_response: bool = False,
							  cached_response_path: Optional[Path] = None,
							  model: str = "deepseek-chat",
							  prompt_output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
	"""Analyze a specific kernel from the Triton cache directory"""

	optimizer = ConfigurationOptimizer(use_cached_response=use_cached_response, model=model, prompt_output_dir=prompt_output_dir)

	# Find the kernel's cache subdirectory
	ttgir_files = list(cache_dir.rglob(f"*{kernel_name}.ttgir"))

	if not ttgir_files:
		raise ValueError(f"Error: Could not find TTGIR file for kernel '{kernel_name}' with source '{torch_model}'")

	ttgir_path = None

	for f in ttgir_files:
		with open(f, 'r') as file:
			content = file.read()
		source_file, _ = optimizer._extract_source_location(content)
		if not source_file:
			raise ValueError(f"Error: Could not extract source location from TTGIR file: {f}")
		source_file = source_file.split('/')[-1].split('.')[0]
		if source_file == torch_model:
			ttgir_path = f
			print(f"Matched TTGIR file: {ttgir_path} for kernel source: {source_file}")
			break

	if not ttgir_path:
		raise ValueError(f"Error: Could not find TTGIR file for kernel '{kernel_name}' with source '{torch_model}'")

	metadata_path = ttgir_path.with_suffix('.json')

	if not metadata_path.exists():
		print(f"Error: Metadata file not found: {metadata_path}")
		return None

	print(f"Analyzing kernel: {kernel_name}")
	print(f"Triton model source: {torch_model}")
	print(f"TTGIR: {ttgir_path}")
	print(f"Metadata: {metadata_path}")
	if triton_modules_dir:
		print(f"Triton modules dir: {triton_modules_dir}")
	if use_cached_response and cached_response_path:
		print(f"Using cached response: {cached_response_path}")
	print()

	# Create optimizer and analyze
	result = optimizer.analyze_and_optimize(
		ttgir_path, torch_model, metadata_path, target_gpu, triton_modules_dir, cached_response_path
	)

	return result


def extract_best_config(analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
	"""
	Extract the best recommended configuration from LLM analysis result.

	Args:
		analysis_result: The result dictionary from analyze_and_optimize

	Returns:
		Dictionary with the best config parameters, or None if not found
	"""
	try:
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


def benchmark_baseline_config(
	kernel_name: str,
	kernel_path: Path,
	input_shapes_json: Path,
	device: str = "cuda",
	num_iterations: int = 100
) -> Dict[str, Any]:
	"""
	Benchmark the baseline (default) kernel configuration.

	This extracts the original configuration from the captured metadata and benchmarks it
	to establish a performance baseline for comparison with optimized configurations.

	Args:
		kernel_name: Name of the kernel function
		kernel_path: Path to the Triton kernel Python file
		input_shapes_json: Path to the *_shapes.json file with captured inputs
		device: Device to run on ("cuda", "xpu", or "cpu")
		num_iterations: Number of benchmark iterations

	Returns:
		Dictionary with baseline benchmark results
	"""
	print(f"\n{'='*80}")
	print(f"Benchmarking Baseline Configuration on {device}")
	print(f"{'='*80}")

	try:
		# Check device availability
		if device == "cuda" and not torch.cuda.is_available():
			print("CUDA not available, falling back to CPU")
			device = "cpu"
		elif device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
			print("XPU not available, falling back to CPU")
			device = "cpu"

		# Load tensor shape information
		print(f"\nLoading tensor shapes from: {input_shapes_json}")
		if not input_shapes_json.exists():
			raise FileNotFoundError(f"Tensor shapes file not found: {input_shapes_json}")

		metadata = load_tensor_shapes(input_shapes_json)

		# Validate metadata structure
		if not metadata:
			raise ValueError(f"Empty or invalid metadata loaded from {input_shapes_json}")

		if not isinstance(metadata, dict):
			raise TypeError(f"Metadata must be a dictionary, got {type(metadata)}")

		kw = metadata.get('kwargs', {})
		if not kw:
			raise ValueError(
				f"No 'kwargs' found in tensor shapes metadata.\n"
				f"Metadata keys available: {list(metadata.keys())}\n"
				f"This likely means the shapes file was not captured correctly.\n"
				f"Expected structure: {{'kwargs': {{'num_warps': ..., 'num_stages': ..., ...}}}}"
			)

		if not isinstance(kw, dict):
			raise TypeError(f"'kwargs' must be a dictionary, got {type(kw)}")

		# Extract baseline configuration from metadata
		num_warps_entry = kw.get("num_warps", None)
		num_stages_entry = kw.get("num_stages", None)

		if num_warps_entry is None:
			raise ValueError(
				f"'num_warps' not found in kwargs.\n"
				f"Available kwargs keys: {list(kw.keys())}\n"
				f"Cannot proceed without baseline configuration."
			)

		if num_stages_entry is None:
			raise ValueError(
				f"'num_stages' not found in kwargs.\n"
				f"Available kwargs keys: {list(kw.keys())}\n"
				f"Cannot proceed without baseline configuration."
			)

		# Validate structure of num_warps and num_stages entries
		if not isinstance(num_warps_entry, dict) or "value" not in num_warps_entry:
			raise ValueError(
				f"Invalid 'num_warps' structure in metadata.\n"
				f"Expected: {{'value': <int>}}, got: {num_warps_entry}"
			)

		if not isinstance(num_stages_entry, dict) or "value" not in num_stages_entry:
			raise ValueError(
				f"Invalid 'num_stages' structure in metadata.\n"
				f"Expected: {{'value': <int>}}, got: {num_stages_entry}"
			)

		baseline_config = {
			"num_warps": num_warps_entry["value"],
			"num_stages": num_stages_entry["value"],
			"block_size": {}
		}

		# Validate extracted values
		if not isinstance(baseline_config["num_warps"], int) or baseline_config["num_warps"] <= 0:
			raise ValueError(f"Invalid num_warps value: {baseline_config['num_warps']} (must be positive integer)")

		if not isinstance(baseline_config["num_stages"], int) or baseline_config["num_stages"] <= 0:
			raise ValueError(f"Invalid num_stages value: {baseline_config['num_stages']} (must be positive integer)")

		# Parse the kernel source to find the original kernel call and extract block sizes
		print("Extracting baseline configuration from kernel source...")
		if not kernel_path.exists():
			raise FileNotFoundError(f"Kernel file not found: {kernel_path}")

		with open(kernel_path, 'r') as f:
			kernel_source = f.read()

		print(f"Reading kernel source from: {kernel_path}")

		if not kernel_source or not kernel_source.strip():
			raise ValueError(f"Kernel source file is empty: {kernel_path}")

		# Find the kernel call in the source code
		# Pattern: kernel_name[grid](args..., num_warps=X, num_stages=Y)

		# Search for the kernel call pattern
		call_pattern = rf'{kernel_name}\s*\[([^\]]+)\]\s*\(([^)]+)\)'
		call_match = re.search(call_pattern, kernel_source, re.DOTALL)

		if call_match:
			grid_str = call_match.group(1)
			args_str = call_match.group(2)

			print(f"Found kernel call: {kernel_name}[{grid_str}](...)")

			# Parse the arguments
			# Split by comma but be careful with nested structures
			args_parts = []
			paren_depth = 0
			current_arg = []

			for char in args_str:
				if char == '(' or char == '[':
					paren_depth += 1
				elif char == ')' or char == ']':
					paren_depth -= 1
				elif char == ',' and paren_depth == 0:
					args_parts.append(''.join(current_arg).strip())
					current_arg = []
					continue
				current_arg.append(char)

			if current_arg:
				args_parts.append(''.join(current_arg).strip())

			# Extract num_warps and num_stages from keyword arguments
			for arg in args_parts:
				if 'num_warps' in arg:
					match = re.search(r'num_warps\s*=\s*(\d+)', arg)
					if match:
						baseline_config['num_warps'] = int(match.group(1))
				elif 'num_stages' in arg:
					match = re.search(r'num_stages\s*=\s*(\d+)', arg)
					if match:
						baseline_config['num_stages'] = int(match.group(1))

			# Now parse the kernel signature to understand positional arguments
			import ast
			tree = ast.parse(kernel_source)
			kernel_def = None
			for node in ast.walk(tree):
				if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
					kernel_def = node
					break

			if kernel_def:
				param_names = [arg.arg for arg in kernel_def.args.args]
				print(f"Kernel parameters: {param_names}")

				# Filter out keyword arguments from positional arguments
				positional_args = [arg for arg in args_parts if '=' not in arg]

				print(f"Found {len(positional_args)} positional arguments in kernel call")

				# Match positional args to parameter names
				# Skip tensor pointers, focus on scalar values
				for i, param_name in enumerate(param_names):
					if i < len(positional_args):
						arg_value_str = positional_args[i].strip()

						# Check if this is a block size parameter
						if 'BLOCK' in param_name.upper():
							# Try to parse the value as an integer
							try:
								# Handle simple integer literals
								if arg_value_str.isdigit():
									baseline_config['block_size'][param_name] = int(arg_value_str)
									print(f"  {param_name} = {arg_value_str} (from positional arg {i})")
							except:
								pass

		if not baseline_config['block_size']:
			raise ValueError("Could not extract block size parameters from kernel source")

		print(f"\nBaseline configuration:")
		print(f"  num_warps: {baseline_config['num_warps']}")
		print(f"  num_stages: {baseline_config['num_stages']}")
		print(f"  block_size: {baseline_config['block_size']}")

		# Extract grid from the original kernel call
		# Grid was already captured in grid_str from the regex match above
		print(f"\nExtracting grid from baseline kernel call...")
		if call_match:
			grid_str = call_match.group(1).strip()
			print(f"Found grid string: [{grid_str}]")

			# Parse grid tuple - format can be: "64", "64, 64", or "grid(64)"
			grid_parts = [x.strip() for x in grid_str.split(',')]
			try:
				if len(grid_parts) == 1:
					grid = (int(grid_parts[0]),)
				elif len(grid_parts) == 2:
					grid = (int(grid_parts[0]), int(grid_parts[1]))
				elif len(grid_parts) == 3:
					grid = (int(grid_parts[0]), int(grid_parts[1]), int(grid_parts[2]))
				else:
					raise ValueError(f"Unexpected grid format: {grid_str}")
				print(f"Parsed grid: {grid}")
			except ValueError as e:
				print(f"Warning: Could not parse grid from string '{grid_str}', computing from block sizes")
				# Fallback: compute grid from block sizes
				with open(kernel_path, 'r') as f:
					kernel_source = f.read()
				numel_param_names, numel_values, _ = infer_numel_for_kernel(kernel_source, kernel_name)
				if numel_param_names and numel_values:
					numel_dict = dict(zip(numel_param_names, numel_values))
					grid = compute_grid_from_numel(numel_dict, baseline_config['block_size'])
					print(f"Computed grid: {grid}")
				else:
					raise ValueError("Cannot parse grid and cannot compute from numel values")
		else:
			raise ValueError("Could not find kernel call to extract grid")

		# Load baseline arguments from captured shapes (these are the original baseline arguments)
		print(f"\nLoading baseline arguments from: {input_shapes_json}")
		input_args, input_kwargs = create_kernel_inputs(
			input_shapes_json,
			device=device,
			values='random',
			seed=42  # Fixed seed for reproducibility
		)
		print(f"Loaded {len(input_args)} positional arguments (baseline)")

		# Filter out warmup entries and set compilation options
		if input_kwargs is None:
			input_kwargs = {}
		input_kwargs = {k: v for k, v in input_kwargs.items() if 'warmup' not in k.lower()}
		input_kwargs['num_warps'] = baseline_config['num_warps']
		input_kwargs['num_stages'] = baseline_config['num_stages']

		print(f"\nBenchmarking baseline configuration with warmup=25, rep={num_iterations}...")

		# Directly call benchmark_kernel
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
			raise RuntimeError("Baseline benchmark failed - kernel execution timed out or errored")

		result = {
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
			"config_tested": baseline_config,
			"throughput_items_per_sec": 1000.0 / mean_time if mean_time > 0 else 0,
			"is_baseline": True,
			"config_type": "baseline"
		}

		print(f"\n{'='*80}")
		print("Baseline Benchmark Results:")
		print(f"  Mean execution time: {mean_time:.4f} ms")
		print(f"  Median execution time: {median_time:.4f} ms")
		print(f"  Min execution time: {min_time:.4f} ms")
		print(f"  Max execution time: {max_time:.4f} ms")
		print(f"  Std deviation: {std_dev:.4f} ms")
		print(f"  Throughput: {result['throughput_items_per_sec']:.2f} calls/sec")
		print(f"{'='*80}\n")

		return result

	except Exception as e:
		print(f"\nError benchmarking baseline: {e}")
		import traceback
		traceback.print_exc()

		return {
			"success": False,
			"error": str(e),
			"device": device,
			"is_baseline": True,
			"config_type": "baseline"
		}


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


def main():
	parser = argparse.ArgumentParser(
		description="Analyze TTGIR and propose optimal kernel configurations"
	)
	parser.add_argument(
		"--cache-dir",
		type=Path,
		default=Path("./captured_ir/triton_cache"),
		help="Path to Triton cache directory"
	)
	parser.add_argument(
		"--triton-modules-dir",
		type=Path,
		default=Path("./Data/data_64/triton_modules"),
		help="Path to Triton modules directory with original source code"
	)
	parser.add_argument(
		"--target-gpu",
		type=str,
		default="cuda:70",
		choices=["cuda:70", "cuda:80", "cuda:86", "cuda:a40", "cuda:l40s", "cuda:90"],
		help="Target GPU architecture (V100/A100/RTX3090/A40/L40S/H100)"
	)
	parser.add_argument(
		"--output",
		type=Path,
		help="Output JSON file for results (default: print to stdout)"
	)
	parser.add_argument(
		"--analyze-all",
		action="store_true",
		help="Analyze all kernels in cache directory"
	)
	parser.add_argument(
		"--test-config",
		action="store_true",
		help="Test the best recommended configuration with actual kernel execution"
	)
	parser.add_argument(
		"--input-shapes-json",
		type=Path,
		help="Path to *_shapes.json file from capture_all_kernel_ir.py (required for --test-config)"
	)
	parser.add_argument(
		"--test-device",
		type=str,
		default="cuda",
		choices=["cuda", "xpu", "cpu"],
		help="Device to use for testing (default: cuda)"
	)
	parser.add_argument(
		"--num-iterations",
		type=int,
		default=100,
		help="Number of benchmark iterations when testing config (default: 100)"
	)
	parser.add_argument(
		"--benchmark-baseline",
		action="store_true",
		help="Benchmark the baseline (default) configuration before testing recommended config"
	)
	parser.add_argument(
		"--input-shapes-dir",
		type=Path,
		default=Path("./captured_ir/input_shapes"),
		help="Directory containing *_shapes.json files (default: ./captured_ir/input_shapes)"
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Directory to save individual kernel results (only with --analyze-all)"
	)
	parser.add_argument(
		"--limit",
		type=int,
		help="Limit number of kernels to process when using --analyze-all (default: process all)"
	)
	parser.add_argument(
		"--model",
		type=str,
		default="deepseek-chat",
		help="DeepSeek model to use (default: deepseek-chat, alternatives: deepseek-reasoner)"
	)

	# Cached response group - all three must be provided together
	cached_group = parser.add_argument_group(
		'cached response options',
		'All three arguments must be provided together when using cached responses'
	)
	cached_group.add_argument(
		"--kernel-name",
		type=str,
		help="Kernel name to analyze (e.g., 'triton_poi_fused_add_convolution_2')"
	)
	cached_group.add_argument(
		"--kernel-file",
		type=Path,
		help="Path to the kernel Python file (optional, will search if not provided)"
	)
	cached_group.add_argument(
		"--torch-model-name",
		type=str,
		help="Name of the Torch model (required with cached response)"
	)
	cached_group.add_argument(
		"--use-cached-response",
		action="store_true",
		help="Use cached JSON response instead of calling API"
	)
	cached_group.add_argument(
		"--cached-response-file",
		type=Path,
		help="Path to cached JSON response file (required with --use-cached-response)"
	)
	cached_group.add_argument(
		"--prompt-output-dir",
		type=Path,
		default=Path("./generated_prompts"),
		help="Directory to save generated prompts organized by torch_model/kernel_name"
	)

	args = parser.parse_args()

	if args.analyze_all:
		# Find all TTGIR files
		ttgir_files = sorted(list(args.cache_dir.rglob("*.ttgir")))
		print(f"Found {len(ttgir_files)} TTGIR files to analyze")

		# Apply limit if specified
		if args.limit and args.limit > 0:
			ttgir_files = ttgir_files[:args.limit]
			print(f"Limiting to first {args.limit} kernels\n")
		else:
			print()

		# Create output directory if specified
		if args.output_dir:
			args.output_dir.mkdir(parents=True, exist_ok=True)
			print(f"Results will be saved to: {args.output_dir}\n")
			prompt_output_dir = args.output_dir / "generated_prompts"
			prompt_output_dir.mkdir(parents=True, exist_ok=True)
		else:
			raise ValueError("When using --analyze-all, --output-dir must be specified to save results")

		results = []
		success_count = 0
		failure_count = 0

		for idx, ttgir_path in enumerate(ttgir_files, 1):
			kernel_name = ttgir_path.stem

			# Extract source file info early for tracking
			source_file_info = None
			ttgir_content = ttgir_path.read_text()
			optimizer_temp = ConfigurationOptimizer(use_cached_response=True)
			source_file, line_range = optimizer_temp._extract_source_location(ttgir_content)
			optimizer_temp = None  # Free memory
			if source_file:
				source_file_info = {
					"source_file": str(source_file),
					"line_range": line_range
				}

			if source_file_info:
				# Hash source file path to create unique suffix
				source_model = source_file_info["source_file"].split('/')[-1].split('.')[0]
				unique_kernel_id = f"{kernel_name}_{source_model}"
			else:
				raise ValueError(f"Could not extract source file info from TTGIR: {ttgir_path}")

			print(f"\n{'='*80}")
			print(f"[{idx}/{len(ttgir_files)}] Processing: {kernel_name}")
			if source_file_info:
				print(f"Source: {source_model}")
				if source_file_info.get('line_range'):
					print(f"Lines: {source_file_info['line_range'][0]}-{source_file_info['line_range'][1]}")
				print(f"Unique ID: {unique_kernel_id}")
			print('='*80)

			try:
				# Analyze kernel
				result = analyze_kernel_from_cache(
					args.cache_dir, kernel_name, source_model, args.target_gpu, args.triton_modules_dir,
					args.use_cached_response, args.cached_response_file, args.model, prompt_output_dir
				)				# Add source file info to result
				if result and source_file_info:
					result["source_file_info"] = source_file_info
					result["unique_kernel_id"] = unique_kernel_id

				if not result:
					print(f"Failed to analyze {kernel_name}")
					failure_count += 1
					continue

				# Test configuration if requested
				if args.test_config:
					# Try to find input shapes JSON file
					input_shapes_json = None

					# Search for matching shapes file in input_shapes_dir
					if args.input_shapes_dir and args.input_shapes_dir.exists():
						# Try multiple naming patterns
						patterns = [
							f"*{kernel_name}_shapes.json",
							f"{kernel_name}_shapes.json",
							f"*_{kernel_name}_shapes.json"
						]

						for pattern in patterns:
							matches = list(args.input_shapes_dir.glob(pattern))
							if matches:
								input_shapes_json = matches[0]
								print(f"\nFound input shapes: {input_shapes_json}")
								break

					if not input_shapes_json:
						print(f"\nWarning: No input shapes JSON found for {kernel_name}, skipping benchmarks")
						results.append(result)
						continue

					# Find kernel source file
					kernel_file = None

					# Strategy 1: Extract from TTGIR
					if source_file:
						source_path = Path(source_file)
						if source_path.exists():
							kernel_file = source_path
							print(f"Found kernel file: {kernel_file}")

					if not kernel_file:
						print(f"\nWarning: Could not find kernel source file for {kernel_name}, skipping benchmarks")
						results.append(result)
						continue

					# Extract best config
					best_config = extract_best_config(result)

					if not best_config:
						print(f"\nWarning: Could not extract best config for {kernel_name}, skipping benchmarks")
						results.append(result)
						continue

					# Benchmark baseline if requested
					if args.benchmark_baseline:
						try:
							print(f"\nBenchmarking baseline configuration...")
							baseline_result = benchmark_baseline_config(
								kernel_name=kernel_name,
								kernel_path=kernel_file,
								input_shapes_json=input_shapes_json,
								device=args.test_device,
								num_iterations=args.num_iterations
							)
							result["baseline_benchmark"] = baseline_result
						except Exception as e:
							print(f"Error benchmarking baseline: {e}")
							result["baseline_benchmark"] = {"success": False, "error": str(e)}

					# Test recommended configuration
					try:
						print(f"\nBenchmarking recommended configuration...")
						test_result = test_recommended_config(
							kernel_name=kernel_name,
							kernel_path=kernel_file,
							input_shapes_json=input_shapes_json,
							config=best_config,
							device=args.test_device,
							num_iterations=args.num_iterations
						)
						result["optimized_benchmark"] = test_result

						# Calculate speedup if baseline was benchmarked
						if args.benchmark_baseline and result.get("baseline_benchmark", {}).get("success") and test_result.get("success"):
							baseline_time = result["baseline_benchmark"]["execution_time_ms"]["median"]
							optimized_time = test_result["execution_time_ms"]["median"]
							speedup = baseline_time / optimized_time if optimized_time > 0 else 0

							result["performance_comparison"] = {
								"baseline_time_ms": baseline_time,
								"optimized_time_ms": optimized_time,
								"speedup": speedup,
								"improvement_percent": (speedup - 1) * 100
							}

							print(f"\n{'='*80}")
							print("Performance Comparison:")
							print(f"  Baseline time: {baseline_time:.4f} ms")
							print(f"  Optimized time: {optimized_time:.4f} ms")
							print(f"  Speedup: {speedup:.2f}x")
							print(f"  Improvement: {(speedup - 1) * 100:.2f}%")
							print(f"{'='*80}\n")
					except Exception as e:
						print(f"Error benchmarking optimized config: {e}")
						result["optimized_benchmark"] = {"success": False, "error": str(e)}

				results.append(result)
				success_count += 1

				# Save individual result if output_dir specified
				if args.output_dir:
					output_file = args.output_dir / f"{unique_kernel_id}_result.json"
					with open(output_file, 'w') as f:
						json.dump(result, f, indent=2)
					print(f"Saved result to: {output_file}")

			except Exception as e:
				print(f"\nError processing {kernel_name}: {e}")
				import traceback
				traceback.print_exc()
				failure_count += 1

				# Save error result
				error_result = {
					"kernel_name": kernel_name,
					"unique_kernel_id": unique_kernel_id,
					"success": False,
					"error": str(e)
				}
				if source_file_info:
					error_result["source_file_info"] = source_file_info
				results.append(error_result)

				if args.output_dir:
					output_file = args.output_dir / f"{unique_kernel_id}_error.json"
					with open(output_file, 'w') as f:
						json.dump(error_result, f, indent=2)

		# Summary
		print(f"\n{'='*80}")
		print(f"Batch Processing Complete")
		print(f"{'='*80}")
		print(f"Total kernels: {len(ttgir_files)}")
		print(f"Successful: {success_count}")
		print(f"Failed: {failure_count}")
		print(f"{'='*80}\n")

		# Save combined results if output file specified
		if args.output:
			with open(args.output, 'w') as f:
				json.dump(results, f, indent=2)
			print(f"Combined results saved to: {args.output}\n")

	elif args.kernel_name and args.torch_model_name:
		# Validate cached response arguments - all three must be provided together
		if args.use_cached_response and not args.cached_response_file:
			print("Error: --cached-response-file is required when using --use-cached-response")
			print("\nWhen using cached responses, you must provide:")
			print("  --torch-model-name <name>")
			print("  --use-cached-response")
			print("  --cached-response-file <path>")
			return 1

		if not args.cache_dir:
			print("Error: --cache-dir is required when analyzing a specific kernel")
			print(f"\nPlease provide the directory containing TTGIR files for model '{args.torch_model_name}'")
			print("Example: --cache-dir ./triton_cache/{args.torch_model_name}/")
			return 1

		if not args.cache_dir.exists():
			print(f"Error: Cache directory does not exist: {args.cache_dir}")
			print(f"\nExpected directory for model '{args.torch_model_name}': {args.cache_dir}")
			print("Make sure you've captured TTGIR files for this model first.")
			return 1

		result = analyze_kernel_from_cache(
			args.cache_dir, args.kernel_name, args.torch_model_name, args.target_gpu, args.triton_modules_dir,
			args.use_cached_response, args.cached_response_file, args.model, args.prompt_output_dir
		)
		if result:
			if args.output:
				with open(args.output, 'w') as f:
					json.dump(result, f, indent=2)
				print(f"Results saved to: {args.output}")
			else:
				print("\n" + json.dumps(result, indent=2))

			# Test the recommended configuration if requested
			if args.test_config:
				# Try to auto-find input_shapes_json if not provided
				input_shapes_json = args.input_shapes_json

				if not input_shapes_json:
					print(f"\nAttempting to auto-find input shapes JSON...")
					if args.input_shapes_dir and args.input_shapes_dir.exists():
						patterns = [
							f"*{args.kernel_name}_shapes.json",
							f"{args.kernel_name}_shapes.json",
							f"*_{args.kernel_name}_shapes.json"
						]

						for pattern in patterns:
							matches = list(args.input_shapes_dir.glob(pattern))
							if matches:
								input_shapes_json = matches[0]
								print(f"Found input shapes: {input_shapes_json}")
								break

				if not input_shapes_json:
					print("\nError: --input-shapes-json is required when using --test-config")
					print("Please provide the path to the *_shapes.json file from capture_all_kernel_ir.py")
					print(f"Example: captured_ir/input_shapes/{args.kernel_name}_shapes.json")
					return 1

				print(f"\nUsing input shapes JSON: {input_shapes_json}")

				# Extract best config
				best_config = extract_best_config(result)

				if best_config:
					# Find the kernel Python file - try multiple search strategies
					kernel_file = None

					# If user provided kernel file path, use it
					if args.kernel_file:
						if args.kernel_file.exists():
							kernel_file = args.kernel_file
							print(f"Using provided kernel file: {kernel_file}")
						else:
							print(f"Error: Provided kernel file does not exist: {args.kernel_file}")
							return 1
					else:
						# Strategy 1: Extract source location from TTGIR file
						print(f"\nFinding kernel source file from TTGIR...")
						ttgir_files = list(args.cache_dir.rglob(f"*{args.kernel_name}.ttgir"))
						if ttgir_files:
							ttgir_path = ttgir_files[0]
							print(f"Reading TTGIR from: {ttgir_path}")

							with open(ttgir_path, 'r') as f:
								ttgir_content = f.read()

							# Use ConfigurationOptimizer's method to extract source location
							optimizer = ConfigurationOptimizer(use_cached_response=True)
							source_file, line_range = optimizer._extract_source_location(ttgir_content)

							if source_file:
								source_path = Path(source_file)
								if source_path.exists():
									kernel_file = source_path
									print(f"Found kernel from TTGIR loc info: {kernel_file}")
									if line_range:
										print(f"  Line range: {line_range[0]}-{line_range[1]}")
								else:
									print(f"Source file from TTGIR doesn't exist: {source_file}")
							else:
								print("Could not extract source location from TTGIR")


					if not kernel_file:
						print(f"\nError: Could not find kernel file for '{args.kernel_name}'")
						print(f"Searched in:")
						print(f"  - TTGIR loc attributes")
						print(f"  - {args.triton_modules_dir}")
						print(f"  - {args.triton_modules_dir.parent}")
						print("\nPlease provide the kernel file path manually using --kernel-file")
						print("or ensure the kernel file exists in the expected location.")
						return 1

					# Benchmark baseline configuration if requested
					if args.benchmark_baseline:
						baseline_result = benchmark_baseline_config(
							kernel_name=args.kernel_name,
							kernel_path=kernel_file,
							input_shapes_json=input_shapes_json,
							device=args.test_device,
							num_iterations=args.num_iterations
						)
						result["baseline_benchmark"] = baseline_result

					# Test the recommended configuration
					test_result = test_recommended_config(
						kernel_name=args.kernel_name,
						kernel_path=kernel_file,
						input_shapes_json=input_shapes_json,
						config=best_config,
						device=args.test_device,
						num_iterations=args.num_iterations
					)

					# Add test results to the overall result
					result["optimized_benchmark"] = test_result

					# Calculate speedup if baseline was benchmarked
					if args.benchmark_baseline and baseline_result.get("success") and test_result.get("success"):
						baseline_time = baseline_result["execution_time_ms"]["median"]
						optimized_time = test_result["execution_time_ms"]["median"]
						speedup = baseline_time / optimized_time if optimized_time > 0 else 0

						result["performance_comparison"] = {
							"baseline_time_ms": baseline_time,
							"optimized_time_ms": optimized_time,
							"speedup": speedup,
							"improvement_percent": (speedup - 1) * 100
						}

						print(f"\n{'='*80}")
						print("Performance Comparison:")
						print(f"  Baseline time: {baseline_time:.4f} ms")
						print(f"  Optimized time: {optimized_time:.4f} ms")
						print(f"  Speedup: {speedup:.2f}x")
						print(f"  Improvement: {(speedup - 1) * 100:.2f}%")
						print(f"{'='*80}\n")

					# Save updated results if output file specified
					if args.output:
						with open(args.output, 'w') as f:
							json.dump(result, f, indent=2)
						print(f"\nUpdated results (including benchmark) saved to: {args.output}")
				else:
					print("\nCould not extract best configuration. Skipping test.")

	else:
		parser.print_help()
		print("\nError: Must specify either (--kernel-name and --torch-model-name) or --analyze-all")
		return 1

	return 0


if __name__ == "__main__":
	exit(main())
