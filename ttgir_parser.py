#!/usr/bin/env python3
"""
TTGIR Parser Class

Parses Triton GPU IR (TTGIR) files to extract optimization-relevant information.
"""

import re
from pathlib import Path
from typing import Dict, Any, List

from ttgir_analysis import TTGIRAnalysis


class TTGIRParser:
	"""Parse TTGIR files to extract optimization-relevant information"""

	@staticmethod
	def parse_ttgir_file(ttgir_path: Path) -> TTGIRAnalysis:
		"""Parse TTGIR file and extract key information"""
		with open(ttgir_path, 'r') as f:
			content = f.read()

		# Extract blocked layouts
		blocked_layouts = TTGIRParser._extract_blocked_layouts(content)

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
			print(shape_str)
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