#!/usr/bin/env python3
"""
TTGIR Analysis Data Class

Contains the TTGIRAnalysis dataclass for storing TTGIR analysis results.
"""

from dataclasses import dataclass
from typing import Dict, Any, List


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

	# Operations
	operations: Dict[str, int]