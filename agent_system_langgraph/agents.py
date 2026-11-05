"""
Specialized GPU optimization agents using LangGraph

Each agent operates at a different level of the GPU hierarchy:
- WarpAgent: 32-thread warp level (memory coalescing, bank conflicts)
- SMAgent: Streaming Multiprocessor level (occupancy, shared memory)
- GPUAgent: Whole GPU level (grid parallelism, L2 cache)
- CoordinatorAgent: System orchestrator (conflict resolution, final decision)
"""

import json
import logging
import math
from typing import Dict, Any, List, Optional
from .state import OptimizationState, AgentMessage, KernelConfig
from .knowledge_base import get_architecture, get_optimization_strategy, detect_kernel_type, compute_grid
from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _compute_total_elements(shape):
    """
    Compute total number of elements in a tensor from its shape.

    Args:
        shape: Tensor shape (list of dimensions or scalar)

    Returns:
        Total number of elements (product of all dimensions)
    """
    if isinstance(shape, list):
        return math.prod(shape) if shape else 0
    return shape


def _analyze_tensor_layout(tensor_shapes: Dict, tensor_dtypes: Dict) -> Dict[str, Any]:
    """
    Analyze tensor dimensions to provide optimization guidance.

    Args:
        tensor_shapes: Dictionary mapping tensor names to shapes
        tensor_dtypes: Dictionary mapping tensor names to dtypes

    Returns:
        Dictionary with tensor layout analysis and optimization hints
    """
    analysis = {
        "dimensionality": {},
        "access_patterns": {},
        "reduction_dims": [],
        "parallel_dims": [],
        "memory_layout_hints": []
    }

    for name, shape in tensor_shapes.items():
        if not isinstance(shape, list):
            continue

        ndim = len(shape)
        total_elements = _compute_total_elements(shape)

        analysis["dimensionality"][name] = {
            "ndim": ndim,
            "shape": shape,
            "total_elements": total_elements,
            "dtype": tensor_dtypes.get(name, "unknown")
        }

        # Infer access pattern based on shape
        if ndim == 1:
            analysis["access_patterns"][name] = "1D_sequential"
            analysis["parallel_dims"].append((name, 0))
        elif ndim == 2:
            # For 2D, innermost dimension is typically contiguous
            analysis["access_patterns"][name] = "2D_row_major"
            analysis["parallel_dims"].extend([(name, 0), (name, 1)])
        elif ndim >= 3:
            analysis["access_patterns"][name] = "ND_complex"

    # Generate optimization hints -- emitted for every kernel regardless of
    # access pattern.  The old gate ``if any(1D_sequential)`` meant the
    # hint never fired for mamba2 / int4_gemm / jag_ln-class kernels (all
    # tensors >= 2D), which is exactly where the agents' scale-up prior
    # was systematically picking XBLOCK >= 64 / num_warps >= 4 while a
    # uniform-random baseline won with XBLOCK in {1, 2, 16, 32} and
    # num_warps in {1, 2}.  The wording below is kernel-class-agnostic.
    analysis["memory_layout_hints"].append(
        "Choose XBLOCK based on the access pattern. "
        "Larger XBLOCK (128-1024) helps coalescing when loads are dense and contiguous. "
        "Smaller XBLOCK (2-32) can be preferable when access is offset-indirect, "
        "when rows are short, when a large reduction dim needs the register budget, "
        "or when per-element work is cheap (tensor copies, bit-ops, trivial pointwise). "
        "For memory-bound kernels, deeper pipelines (num_stages=2-4) and lower "
        "num_warps (1-2) often win.  "
        "Do not assume XBLOCK must be a multiple of 32. "
        "**SMALL-GRID HEURISTIC**: If the launched grid is small "
        "(grid.x * grid.y * grid.z < ~1e6 program instances) AND the kernel is "
        "pointwise / bmm / clone-like (no large reduction inner dim), the OPTIMUM is "
        "frequently num_warps=1 with XBLOCK in {8, 16, 32}. This is NOT a "
        "low-occupancy anti-pattern — at small grid sizes the SMs aren't the bottleneck; "
        "register pressure and barrier overhead are. ALWAYS include at least one "
        "num_warps=1 candidate with XBLOCK <= 32 in your slate when the grid is small."
    )

    return analysis


# ---------------------------------------------------------------------------
# Canonical ordering for generic block-size names produced by LLMs
# ---------------------------------------------------------------------------
_GENERIC_BLOCK_ORDER = ["XBLOCK", "YBLOCK", "ZBLOCK"] + [f"R{i}_BLOCK" for i in range(10)]


def _get_valid_block_names(config: KernelConfig) -> List[str]:
    """Return the authoritative list of block-size parameter names for *config*.

    Prefers ``constexpr_param_names`` (extracted from shapes JSON kwargs at
    capture time).  Falls back to the keys already present in
    ``config.block_size``.  Returns ``["XBLOCK"]`` as a last resort.
    """
    names = config.constexpr_param_names or list(config.block_size.keys())
    return names if names else ["XBLOCK"]


def _format_block_size_schema(config: KernelConfig) -> str:
    """Build the JSON object pattern the LLM should use for block sizes.

    Example return value::

        '{"BLOCK_SIZE": <int>}'
    """
    names = _get_valid_block_names(config)
    inner = ", ".join(f'"{p}": <int>' for p in names)
    return "{" + inner + "}"


def _format_block_size_instructions(config: KernelConfig) -> str:
    """Generate a prompt snippet that tells the LLM which block-size names to use."""
    names = _get_valid_block_names(config)
    names_str = ", ".join(names)
    schema = _format_block_size_schema(config)
    return (
        f"This kernel uses these configurable block-size parameters: {names_str}.\n"
        f"Use ONLY these exact parameter names in your recommendations.\n"
        f"Output format for block sizes: {schema}"
    )


def _validate_block_size_names(
    proposed: Dict[str, Any],
    valid_names: List[str],
) -> Dict[str, Any]:
    """Validate and auto-correct block-size parameter names in LLM output.

    If the LLM produced generic names (XBLOCK, YBLOCK, …) but the kernel
    uses different names (e.g. BLOCK_SIZE), remap positionally using the
    canonical ordering defined by ``_GENERIC_BLOCK_ORDER``.

    Returns:
        A new dict with corrected keys.  Keys that cannot be mapped are
        dropped with a warning.
    """
    if not proposed:
        return {}

    # Fast path: all proposed keys are already valid
    if set(proposed.keys()).issubset(set(valid_names)):
        return dict(proposed)

    # First pass: collect direct matches
    corrected: Dict[str, Any] = {}
    needs_remap: Dict[str, Any] = {}
    for pkey, pval in proposed.items():
        if pkey in valid_names:
            corrected[pkey] = pval
        else:
            needs_remap[pkey] = pval

    # Remaining valid names that haven't been matched yet
    remaining_valid = sorted(n for n in valid_names if n not in corrected)

    # Sort unresolved keys by canonical generic order
    def _sort_key(k: str) -> int:
        try:
            return _GENERIC_BLOCK_ORDER.index(k)
        except ValueError:
            return 999

    remap_sorted = sorted(needs_remap.keys(), key=_sort_key)

    for i, pkey in enumerate(remap_sorted):
        if i < len(remaining_valid):
            actual = remaining_valid[i]
            corrected[actual] = needs_remap[pkey]
            logger.warning(
                "Remapped LLM block param %s -> %s (value=%s)",
                pkey, actual, needs_remap[pkey],
            )
        else:
            logger.warning(
                "Dropping unmappable LLM block param %s=%s (no matching kernel param)",
                pkey, needs_remap[pkey],
            )

    return corrected


def _normalize_proposals_for_prompt(proposals):
    """Normalize agent proposals for the coordinator prompt.

    Handles backward compatibility:
    - ``None`` → empty list
    - Single dict (old format) → 1-element list
    - List (new format) → returned as-is
    """
    if proposals is None:
        return []
    if isinstance(proposals, dict):
        return [proposals]
    return proposals


class BaseOptimizationAgent:
    """Base class for GPU optimization agents"""

    def __init__(
        self,
        agent_name: str,
        llm_provider: LLMProvider,
        architecture_code: str = "cuda:70"
    ):
        self.agent_name = agent_name
        self.llm = llm_provider
        self.architecture = get_architecture(architecture_code)
        logger.info(f"Initialized {agent_name} for {self.architecture.name}")

    def create_message(
        self,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 5
    ) -> AgentMessage:
        """Create an agent message"""
        return AgentMessage(
            sender=self.agent_name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            priority=priority
        )

    def analyze(self, state: OptimizationState) -> OptimizationState:
        """Analyze kernel and update state"""
        raise NotImplementedError


class WarpAgent(BaseOptimizationAgent):
    """
    Warp-level optimization agent (32-thread level)

    Focuses on:
    - Memory coalescing efficiency
    - Shared memory bank conflicts
    - Warp divergence
    - Cache line utilization
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("WarpAgent", llm_provider, architecture_code)

    def analyze(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Analyze warp-level characteristics"""
        logger.info(f"{self.agent_name}: Starting warp-level analysis")

        config = state["kernel_config"]
        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)

        # Analyze tensor layout for better optimization guidance
        tensor_layout = _analyze_tensor_layout(config.tensor_shapes, config.tensor_dtypes)

        # Build system prompt
        system_prompt = f"""You are a GPU warp-level optimization expert specializing in {self.architecture.name} architecture.

Your role is to analyze memory access patterns at the 32-thread warp granularity and suggest optimizations for:
1. Memory coalescing (ideal: all threads in a warp access contiguous memory)
2. Shared memory bank conflicts (32 banks, avoid same-bank access by multiple threads)
3. Warp divergence (branching that causes threads to execute serially)
4. Cache line utilization (128-byte cache lines)

Architecture constraints:
- Warp size: {self.architecture.warp_size} threads
- L1 cache line: 128 bytes
- Shared memory banks: 32
- Target coalescing efficiency: >{self.architecture.coalescing_efficiency_threshold * 100}%

Respond with JSON analysis only."""

        # Build user prompt
        user_prompt = f"""Analyze the following Triton kernel for warp-level optimizations:

**Kernel**: {config.kernel_name}
**Type**: {kernel_type}
**Current Configuration**:
- num_warps: {config.num_warps}
- block_size: {config.block_size}
- Threads per CTA: {config.num_warps * self.architecture.warp_size}

**Tensor Shapes**: {json.dumps(config.tensor_shapes, indent=2)}
**Tensor Dtypes**: {json.dumps(config.tensor_dtypes, indent=2)}
**Total elements per tensor**: {json.dumps({k: _compute_total_elements(v) for k, v in config.tensor_shapes.items()}, indent=2)}

**Tensor Layout Analysis**:
{json.dumps(tensor_layout, indent=2)}

**CRITICAL**: Consider how tensor dimensions affect block size selection:
{_format_block_size_instructions(config)}
- Larger blocks -> better coalescing but may reduce occupancy
- Smaller blocks -> better occupancy but more memory transactions
- **Use the tensor layout analysis above to understand access patterns and optimize accordingly**

**Memory Operations**:
- Global loads: {config.global_loads}
- Global stores: {config.global_stores}
- Shared memory: {config.shared_memory_bytes} bytes

**Triton Kernel Source Code**:
```python
{config.source_code if config.source_code else 'Source code not available'}
```

**TTGIR**:
```
{config.ttgir}
```

**Analysis Requirements**:
Provide JSON with this structure:
{{
  "coalescing_efficiency": {{
    "estimated_percentage": <float 0-100>,
    "bottlenecks": ["list of coalescing issues"],
    "access_pattern": "description of current memory access pattern",
    "tensor_dimension_impact": "how tensor shapes affect memory access"
  }},
  "bank_conflicts": {{
    "severity": "none|low|medium|high",
    "estimated_conflicts_per_warp": <int>,
    "conflict_sources": ["list of conflict sources"]
  }},
  "warp_divergence": {{
    "present": <bool>,
    "divergence_points": ["list of divergence locations"],
    "impact": "low|medium|high"
  }},
  "cache_utilization": {{
    "l1_hit_rate_estimate": <float 0-1>,
    "cache_line_utilization": "poor|fair|good|excellent"
  }},
  "optimization_proposals": [
    {{
      "strategy": "string description",
      "num_warps": <int>,
      "changes": {_format_block_size_schema(config)},
      "reasoning": "explain why these block sizes work well with tensor dimensions",
      "expected_improvement": "percentage or description",
      "confidence": <float 0-1>,
      "variant_type": "aggressive|conservative|balanced|exploratory",
      "risks": ["potential risks"]
    }}
  ],
  "key_insights": ["insight 1 about tensor dimension impact", "insight 2", "insight 3"]
}}

**IMPORTANT**: Provide exactly {num_candidates} distinct optimization proposals in the "optimization_proposals" array.
Each proposal should explore a different trade-off point (e.g., one conservative, one aggressive).
Ensure diversity — do NOT duplicate proposals."""

        # Add iteration feedback if available
        if state.get("iteration", 0) > 0 and state.get("previous_coordinator_feedback"):
            user_prompt += f"""

**ITERATION {state['iteration']} - REFINEMENT MODE**:
{state['previous_coordinator_feedback']}

**Previous Proposals**:
{json.dumps(state.get('iteration_history', []), indent=2)}

**Task**: Refine your analysis based on coordinator feedback. Focus on addressing the identified bottleneck.
"""

        # Add benchmark feedback if available (closed-loop verification)
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            if speedup is not None and speedup < 1.0:
                user_prompt += f"""

**BENCHMARK DATA FROM PREVIOUS ATTEMPT**:
- Original config execution time: {bench.get('original_stats', {}).get('median', 'N/A')}ms
- Previous recommendation execution time: {bench.get('recommended_stats', {}).get('median', 'N/A')}ms
- Speedup: {speedup:.3f}x (this was a REGRESSION)
Focus your analysis on why the previous recommendation failed and what should change.
"""

        # Call LLM
        try:
            response = self.llm.generate(system_prompt, user_prompt, response_format="json", agent_name=self.agent_name)
            analysis = self.llm.parse_json_response(response)

            # Validate block-size names in proposals
            valid_names = _get_valid_block_names(config)
            for prop in analysis.get("optimization_proposals", []):
                raw = prop.get("changes", {})
                if raw:
                    prop["changes"] = _validate_block_size_names(raw, valid_names)

            # Update state
            state["warp_analysis"] = analysis

            # Store all proposals as a list (multi-candidate support)
            if analysis.get("optimization_proposals"):
                state["warp_proposal"] = [
                    {
                        "agent": self.agent_name,
                        "proposal": p,
                        "confidence": p.get("confidence", 0.5),
                    }
                    for p in analysis["optimization_proposals"]
                ]

            # Send message to coordinator
            message = self.create_message(
                recipient="coordinator",
                message_type="analysis",
                content={
                    "analysis": analysis,
                    "bottleneck_severity": self._assess_severity(analysis)
                },
                priority=3
            )
            state["messages"].append(message)

            logger.info(f"{self.agent_name}: Analysis complete")

        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            state["error"] = f"WarpAgent error: {e}"

        return state

    def _assess_severity(self, analysis: Dict[str, Any]) -> str:
        """Assess bottleneck severity"""
        coalescing = analysis.get("coalescing_efficiency", {}).get("estimated_percentage", 100)
        bank_conflicts = analysis.get("bank_conflicts", {}).get("severity", "none")

        if coalescing < 50 or bank_conflicts == "high":
            return "critical"
        elif coalescing < 70 or bank_conflicts == "medium":
            return "moderate"
        else:
            return "minor"

    def _get_severity_score(self, analysis: Dict[str, Any]) -> float:
        """Convert severity assessment to numeric score for weighting"""
        coalescing = analysis.get("coalescing_efficiency", {}).get("estimated_percentage", 100)
        bank_conflicts = analysis.get("bank_conflicts", {}).get("severity", "none")

        # Calculate score based on severity
        if coalescing < 50 or bank_conflicts == "high":
            return 1.0  # Critical
        elif coalescing < 70 or bank_conflicts == "medium":
            return 0.6  # Moderate
        else:
            return 0.3  # Minor


class SMAgent(BaseOptimizationAgent):
    """
    SM-level optimization agent (Streaming Multiprocessor level)

    Focuses on:
    - Thread occupancy
    - Shared memory allocation
    - Register pressure
    - Warp scheduling efficiency
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("SMAgent", llm_provider, architecture_code)

    def analyze(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Analyze SM-level characteristics"""
        logger.info(f"{self.agent_name}: Starting SM-level analysis")

        config = state["kernel_config"]
        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)
        strategy = get_optimization_strategy(kernel_type, self.architecture)

        # Analyze tensor layout for better optimization guidance
        tensor_layout = _analyze_tensor_layout(config.tensor_shapes, config.tensor_dtypes)

        # Calculate current occupancy
        threads_per_block = config.num_warps * self.architecture.warp_size
        max_blocks_per_sm_threads = self.architecture.max_threads_per_sm // threads_per_block
        max_blocks_per_sm_smem = (self.architecture.shared_memory_per_sm_kb * 1024) // max(config.shared_memory_bytes, 1)
        max_blocks_per_sm = min(max_blocks_per_sm_threads, max_blocks_per_sm_smem, self.architecture.max_warps_per_sm // config.num_warps)
        theoretical_occupancy = (max_blocks_per_sm * config.num_warps) / self.architecture.max_warps_per_sm

        system_prompt = f"""You are a GPU SM-level optimization expert for {self.architecture.name}.

Your role is to optimize SM (Streaming Multiprocessor) resource utilization:
1. Thread occupancy (balance between parallelism and resource usage)
2. Shared memory allocation (avoid waste, maximize reuse)
3. Register pressure (minimize spilling)
4. Warp scheduling (hide memory latency)

SM constraints:
- Max warps per SM: {self.architecture.max_warps_per_sm}
- Max threads per SM: {self.architecture.max_threads_per_sm}
- Shared memory per SM: {self.architecture.shared_memory_per_sm_kb} KB
- Registers per SM: {self.architecture.registers_per_sm}
- Target occupancy: >{self.architecture.min_occupancy_target * 100}%

Kernel type: {kernel_type}
Optimization strategy: {strategy['characteristic']} workload

Respond with JSON analysis only."""

        user_prompt = f"""Analyze SM-level resource utilization for this kernel:

**Kernel**: {config.kernel_name}
**Current Configuration**:
- num_warps: {config.num_warps}
- num_ctas: {config.num_ctas}
- Threads per CTA: {threads_per_block}
- Shared memory: {config.shared_memory_bytes} bytes ({config.shared_memory_bytes / 1024:.2f} KB)

**Calculated Occupancy**:
- Max blocks per SM (threads): {max_blocks_per_sm_threads}
- Max blocks per SM (shared mem): {max_blocks_per_sm_smem}
- Effective max blocks: {max_blocks_per_sm}
- Theoretical occupancy: {theoretical_occupancy * 100:.1f}%

**Tensor Shapes**: {json.dumps(config.tensor_shapes, indent=2)}
**Tensor Dtypes**: {json.dumps(config.tensor_dtypes, indent=2)}
**Grid**: {config.grid}{f"""
**Grid Formula**: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}

**Tensor Layout Analysis**:
{json.dumps(tensor_layout, indent=2)}

**CRITICAL**: Balance num_warps with tensor dimensions:
- Larger num_warps -> more parallelism, lower occupancy, higher shared memory
- Consider total work (tensor elements) vs. available parallelism
- Adjust block sizes to match tensor dimensions for even distribution
- **Use the tensor layout analysis above to understand how tensor size affects optimal warp count**

{_format_block_size_instructions(config)}

**Triton Kernel Source Code**:
```python
{config.source_code if config.source_code else 'Source code not available'}
```

**TTGIR**:
```
{config.ttgir}
```

**Analysis Requirements**:
Provide JSON:
{{
  "occupancy_analysis": {{
    "current_occupancy_pct": {theoretical_occupancy * 100:.1f},
    "limiting_factor": "threads|shared_memory|registers|warps",
    "target_occupancy_pct": <float>,
    "occupancy_gap_explanation": "why current != target",
    "tensor_dimension_considerations": "how tensor size affects optimal warp count"
  }},
  "shared_memory_analysis": {{
    "current_usage_bytes": {config.shared_memory_bytes},
    "utilization_efficiency": "poor|fair|good|excellent",
    "bank_conflict_potential": "low|medium|high",
    "optimization_opportunities": ["list"]
  }},
  "register_pressure": {{
    "estimated_registers_per_thread": <int>,
    "spilling_risk": "none|low|medium|high",
    "impact_on_occupancy": "description"
  }},
  "warp_scheduling": {{
    "latency_hiding_effectiveness": "poor|fair|good|excellent",
    "memory_bound_stalls": <bool>,
    "suggestions": ["list"]
  }},
  "optimization_proposals": [
    {{
      "strategy": "description",
      "num_warps": <int>,
      "changes": {_format_block_size_schema(config)},
      "shared_memory_config": "details",
      "expected_occupancy": <float>,
      "reasoning": "explain relationship between num_warps and tensor dimensions",
      "expected_improvement": "percentage",
      "confidence": <float 0-1>,
      "variant_type": "aggressive|conservative|balanced|exploratory",
      "tradeoffs": ["list"]
    }}
  ],
  "key_insights": ["insight 1 about tensor/warp relationship", "insight 2"]
}}

**IMPORTANT**: Provide exactly {num_candidates} distinct optimization proposals in the "optimization_proposals" array.
Each proposal should explore a different trade-off point (e.g., one conservative, one aggressive).
Ensure diversity — do NOT duplicate proposals."""

        # Add iteration feedback if available
        if state.get("iteration", 0) > 0 and state.get("previous_coordinator_feedback"):
            user_prompt += f"""

**ITERATION {state['iteration']} - REFINEMENT MODE**:
{state['previous_coordinator_feedback']}

**Task**: Refine your analysis based on coordinator feedback. Focus on addressing the identified bottleneck.
"""

        # Add benchmark feedback if available (closed-loop verification)
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            if speedup is not None and speedup < 1.0:
                user_prompt += f"""

**BENCHMARK DATA FROM PREVIOUS ATTEMPT**:
- Original config execution time: {bench.get('original_stats', {}).get('median', 'N/A')}ms
- Previous recommendation execution time: {bench.get('recommended_stats', {}).get('median', 'N/A')}ms
- Speedup: {speedup:.3f}x (this was a REGRESSION)
Focus your analysis on why the previous recommendation failed and what should change.
"""

        try:
            response = self.llm.generate(system_prompt, user_prompt, response_format="json", agent_name=self.agent_name)
            analysis = self.llm.parse_json_response(response)

            # Validate block-size names in proposals
            valid_names = _get_valid_block_names(config)
            for prop in analysis.get("optimization_proposals", []):
                raw = prop.get("changes", {})
                if raw:
                    prop["changes"] = _validate_block_size_names(raw, valid_names)

            state["sm_analysis"] = analysis

            if analysis.get("optimization_proposals"):
                state["sm_proposal"] = [
                    {
                        "agent": self.agent_name,
                        "proposal": p,
                        "confidence": p.get("confidence", 0.5),
                    }
                    for p in analysis["optimization_proposals"]
                ]

            message = self.create_message(
                recipient="coordinator",
                message_type="analysis",
                content={
                    "analysis": analysis,
                    "occupancy": theoretical_occupancy,
                    "bottleneck_severity": self._assess_severity(analysis, theoretical_occupancy)
                },
                priority=2
            )
            state["messages"].append(message)

            logger.info(f"{self.agent_name}: Analysis complete (occupancy={theoretical_occupancy * 100:.1f}%)")

        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            state["error"] = f"SMAgent error: {e}"

        return state

    def _assess_severity(self, analysis: Dict[str, Any], occupancy: float) -> str:
        """Assess bottleneck severity"""
        if occupancy < 0.25:
            return "critical"
        elif occupancy < self.architecture.min_occupancy_target:
            return "moderate"
        else:
            return "minor"

    def _get_severity_score(self, analysis: Dict[str, Any], occupancy: float) -> float:
        """Convert severity assessment to numeric score for weighting"""
        if occupancy < 0.25:
            return 1.0  # Critical
        elif occupancy < self.architecture.min_occupancy_target:
            return 0.6  # Moderate
        else:
            return 0.3  # Minor


class GPUAgent(BaseOptimizationAgent):
    """
    GPU-level optimization agent (Whole GPU level)

    Focuses on:
    - Grid-level parallelism
    - L2 cache behavior
    - Memory bandwidth utilization
    - Multi-SM load balancing
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("GPUAgent", llm_provider, architecture_code)

    def analyze(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Analyze GPU-level characteristics"""
        logger.info(f"{self.agent_name}: Starting GPU-level analysis")

        config = state["kernel_config"]
        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)

        # Analyze tensor layout for better optimization guidance
        tensor_layout = _analyze_tensor_layout(config.tensor_shapes, config.tensor_dtypes)

        # Calculate grid metrics
        total_ctas = (config.grid.get("x") or 1) * (config.grid.get("y") or 1) * (config.grid.get("z") or 1)
        ctas_per_sm = total_ctas / self.architecture.num_sms if total_ctas > 0 else 0

        system_prompt = f"""You are a GPU system-level optimization expert for {self.architecture.name}.

Your role is to optimize GPU-wide characteristics:
1. Grid parallelism (ensure sufficient CTAs to saturate all {self.architecture.num_sms} SMs)
2. L2 cache behavior (data reuse across CTAs, {self.architecture.l2_cache_size_mb} MB available)
3. Memory bandwidth ({self.architecture.memory_bandwidth_gb_s} GB/s)
4. Load balancing across SMs

Grid sizing strategy:
- Need enough CTAs to keep all SMs busy
- But not so many that each CTA does trivial work
- Typical target: 2-4x more CTAs than SMs for latency hiding

Respond with JSON analysis only."""

        user_prompt = f"""Analyze GPU-level parallelism and cache behavior:

**Kernel**: {config.kernel_name}
**Type**: {kernel_type}
**Current Grid Configuration**:
- Grid dimensions: {config.grid}{f"""
- Grid formula: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}
- Total CTAs: {total_ctas}
- CTAs per SM: {ctas_per_sm:.2f}
- Available SMs: {self.architecture.num_sms}

**Block Configuration**:
- Block size: {config.block_size}
- num_warps: {config.num_warps}
- Threads per CTA: {config.num_warps * self.architecture.warp_size}

**Memory Characteristics**:
- Global loads: {config.global_loads}
- Global stores: {config.global_stores}
- L2 cache: {self.architecture.l2_cache_size_mb} MB
- Bandwidth: {self.architecture.memory_bandwidth_gb_s} GB/s

**Tensor Shapes**: {json.dumps(config.tensor_shapes, indent=2)}
**Tensor Dtypes**: {json.dumps(config.tensor_dtypes, indent=2)}

**Tensor Layout Analysis**:
{json.dumps(tensor_layout, indent=2)}

**CRITICAL**: Grid dimensions must match tensor dimensions:
- Total grid work = total tensor elements / block_size
- Grid should saturate all {self.architecture.num_sms} SMs
- For 1D tensors: grid_x = ceil(total_elements / block_size)
- For 2D tensors: grid needs both x and y dimensions
- Consider data reuse across CTAs for L2 cache optimization
- **Use the tensor layout analysis above to understand how tensor dimensions map to grid configuration**

{_format_block_size_instructions(config)}

**Triton Kernel Source Code**:
```python
{config.source_code if config.source_code else 'Source code not available'}
```

**TTGIR**:
```
{config.ttgir}
```

**Analysis Requirements**:
Provide JSON:
{{
  "grid_parallelism": {{
    "current_ctas": {total_ctas},
    "ctas_per_sm": {ctas_per_sm:.2f},
    "saturation_level": "insufficient|adequate|good|excellent",
    "parallelism_bottleneck": <bool>,
    "recommended_cta_count": <int>,
    "tensor_dimension_alignment": "how grid matches tensor shapes"
  }},
  "l2_cache_analysis": {{
    "data_locality_pattern": "none|low|medium|high",
    "estimated_l2_hit_rate": <float 0-1>,
    "reuse_distance": "description",
    "cache_optimization_opportunities": ["list"]
  }},
  "memory_bandwidth": {{
    "estimated_utilization_pct": <float 0-100>,
    "bottleneck_type": "bandwidth_bound|latency_bound|compute_bound",
    "bytes_per_thread": <float>,
    "arithmetic_intensity": <float>
  }},
  "load_balancing": {{
    "grid_uniformity": "poor|fair|good|excellent",
    "tail_effect_risk": "none|low|medium|high",
    "load_imbalance_sources": ["list"]
  }},
  "optimization_proposals": [
    {{
      "strategy": "description",
      "num_warps": <int>,
      "grid_dimensions": {{"x": <int>, "y": <int>, "z": <int>}},
      "block_size_changes": {_format_block_size_schema(config)},
      "expected_cta_count": <int>,
      "reasoning": "explain grid sizing based on tensor dimensions",
      "expected_improvement": "percentage",
      "confidence": <float 0-1>,
      "variant_type": "aggressive|conservative|balanced|exploratory"
    }}
  ],
  "key_insights": ["insight 1 about grid/tensor relationship", "insight 2"]
}}

**IMPORTANT**: Provide exactly {num_candidates} distinct optimization proposals in the "optimization_proposals" array.
Each proposal should explore a different trade-off point (e.g., one conservative, one aggressive).
Ensure diversity — do NOT duplicate proposals."""

        # Add iteration feedback if available
        if state.get("iteration", 0) > 0 and state.get("previous_coordinator_feedback"):
            user_prompt += f"""

**ITERATION {state['iteration']} - REFINEMENT MODE**:
{state['previous_coordinator_feedback']}

**Task**: Refine your analysis based on coordinator feedback. Focus on addressing the identified bottleneck.
"""

        # Add benchmark feedback if available (closed-loop verification)
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            if speedup is not None and speedup < 1.0:
                user_prompt += f"""

**BENCHMARK DATA FROM PREVIOUS ATTEMPT**:
- Original config execution time: {bench.get('original_stats', {}).get('median', 'N/A')}ms
- Previous recommendation execution time: {bench.get('recommended_stats', {}).get('median', 'N/A')}ms
- Speedup: {speedup:.3f}x (this was a REGRESSION)
Focus your analysis on why the previous recommendation failed and what should change.
"""

        try:
            response = self.llm.generate(system_prompt, user_prompt, response_format="json", agent_name=self.agent_name)
            analysis = self.llm.parse_json_response(response)

            # Validate block-size names in proposals
            valid_names = _get_valid_block_names(config)
            for prop in analysis.get("optimization_proposals", []):
                raw = prop.get("block_size_changes", {})
                if raw:
                    prop["block_size_changes"] = _validate_block_size_names(raw, valid_names)

            state["gpu_analysis"] = analysis

            if analysis.get("optimization_proposals"):
                state["gpu_proposal"] = [
                    {
                        "agent": self.agent_name,
                        "proposal": p,
                        "confidence": p.get("confidence", 0.5),
                    }
                    for p in analysis["optimization_proposals"]
                ]

            message = self.create_message(
                recipient="coordinator",
                message_type="analysis",
                content={
                    "analysis": analysis,
                    "grid_saturation": ctas_per_sm,
                    "bottleneck_severity": self._assess_severity(analysis, total_ctas)
                },
                priority=1
            )
            state["messages"].append(message)

            logger.info(f"{self.agent_name}: Analysis complete (CTAs={total_ctas}, per_SM={ctas_per_sm:.2f})")

        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            state["error"] = f"GPUAgent error: {e}"

        return state

    def _assess_severity(self, analysis: Dict[str, Any], total_ctas: int) -> str:
        """Assess bottleneck severity"""
        # Check if grid is undersaturated
        if total_ctas < self.architecture.num_sms:
            return "critical"
        elif total_ctas < self.architecture.num_sms * 2:
            return "moderate"
        else:
            return "minor"

    def _get_severity_score(self, analysis: Dict[str, Any], total_ctas: int) -> float:
        """Convert severity assessment to numeric score for weighting"""
        if total_ctas < self.architecture.num_sms:
            return 1.0  # Critical
        elif total_ctas < self.architecture.num_sms * 2:
            return 0.6  # Moderate
        else:
            return 0.3  # Minor


class CoordinatorAgent(BaseOptimizationAgent):
    """
    Coordinator agent for hierarchical optimization

    Responsibilities:
    - Aggregate analyses from all agents
    - Resolve conflicts between proposals
    - Make final configuration decisions
    - Generate implementation instructions
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("CoordinatorAgent", llm_provider, architecture_code)

    def _get_severity_score(self, analysis: Dict, analysis_type: str) -> float:
        """Convert severity assessment to numeric score for any agent type"""
        if analysis_type == "warp":
            coalescing = analysis.get("coalescing_efficiency", {}).get("estimated_percentage", 100)
            if coalescing < 50:
                return 1.0  # Critical
            elif coalescing < 70:
                return 0.6  # Moderate
            else:
                return 0.3  # Minor
        elif analysis_type == "sm":
            occupancy = analysis.get("occupancy_analysis", {}).get("current_occupancy_pct", 100)
            if occupancy < 25:
                return 1.0
            elif occupancy < 50:
                return 0.6
            else:
                return 0.3
        elif analysis_type == "gpu":
            saturation = analysis.get("grid_analysis", {}).get("sm_saturation_pct", 100)
            if saturation < 50:
                return 1.0
            elif saturation < 75:
                return 0.6
            else:
                return 0.3
        return 0.5  # Default

    def _compute_proposal_weights(self, state: OptimizationState) -> Dict[str, float]:
        """Compute weights for each agent's proposal based on bottleneck severity"""
        weights = {"warp": 0.33, "sm": 0.33, "gpu": 0.33}  # Default equal

        # Extract severity scores
        warp_severity = self._get_severity_score(state.get("warp_analysis", {}), "warp")
        sm_severity = self._get_severity_score(state.get("sm_analysis", {}), "sm")
        gpu_severity = self._get_severity_score(state.get("gpu_analysis", {}), "gpu")

        total_severity = warp_severity + sm_severity + gpu_severity

        if total_severity > 0:
            weights["warp"] = warp_severity / total_severity
            weights["sm"] = sm_severity / total_severity
            weights["gpu"] = gpu_severity / total_severity

        return weights

    def coordinate(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Coordinate all agent analyses and produce final result"""
        logger.info(f"{self.agent_name}: Coordinating optimization decisions")

        config = state["kernel_config"]

        # Check if all analyses are complete
        if not all([state.get("warp_analysis"), state.get("sm_analysis"), state.get("gpu_analysis")]):
            logger.warning(f"{self.agent_name}: Not all analyses complete, cannot coordinate")
            return state

        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)

        system_prompt = f"""You are the chief GPU optimization coordinator for {self.architecture.name}.

Your role is to:
1. Integrate analyses from Warp, SM, and GPU level agents
2. Resolve conflicts between their proposals
3. Make final configuration decisions balancing all trade-offs
4. Provide clear implementation instructions

Decision-making principles:
- Prioritize bottleneck with highest severity
- Balance occupancy vs parallelism vs memory efficiency
- Consider kernel type ({kernel_type}) characteristics
- Ensure configuration is within hardware limits
- Provide clear reasoning for all decisions

Respond with JSON containing final optimized configuration."""

        # Gather all proposals (normalize for backward compat)
        proposals_summary = {
            "warp": _normalize_proposals_for_prompt(state.get("warp_proposal")),
            "sm": _normalize_proposals_for_prompt(state.get("sm_proposal")),
            "gpu": _normalize_proposals_for_prompt(state.get("gpu_proposal")),
        }

        # Compute proposal weights based on bottleneck severity
        proposal_weights = self._compute_proposal_weights(state)

        user_prompt = f"""Coordinate optimization decisions for kernel: {config.kernel_name}

**Current Configuration**:
- num_warps: {config.num_warps}
- num_stages: {config.num_stages}
- block_size: {config.block_size}
- grid: {config.grid}{f"""
- grid_formula: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}
- shared_memory: {config.shared_memory_bytes} bytes

**Agent Analyses**:

**Warp Agent Analysis**:
{json.dumps(state["warp_analysis"], indent=2)}

**SM Agent Analysis**:
{json.dumps(state["sm_analysis"], indent=2)}

**GPU Agent Analysis**:
{json.dumps(state["gpu_analysis"], indent=2)}

**Agent Proposals** (each agent provides up to {num_candidates} ranked proposals):
{json.dumps(proposals_summary, indent=2)}

**Computed Proposal Weights** (based on bottleneck severity):
- Warp Agent: {proposal_weights['warp']:.2f}
- SM Agent: {proposal_weights['sm']:.2f}
- GPU Agent: {proposal_weights['gpu']:.2f}

**Conflict Resolution Strategy**:
1. Prioritize proposals from agents with higher weights (more severe bottlenecks)
2. Synthesize complementary insights from multiple agents
3. When proposals conflict, favor the one addressing the most severe bottleneck
4. Ensure final config doesn't regress any dimension significantly

**Task**: Synthesize these analyses into multiple candidate configurations ranked by expected improvement.
All candidates will be benchmarked on actual hardware — the empirically fastest one will be selected.

{_format_block_size_instructions(config)}

**CANDIDATE GENERATION RULES**:
1. Provide exactly {num_candidates} candidate configs, ranked by expected improvement (most promising first).
2. Do NOT include the original config as a candidate — it will be automatically included as baseline.
3. At least one candidate MUST be conservative: change only ONE parameter (e.g., only num_warps, or only block_size).
4. At least one candidate should be more aggressive: change multiple parameters.
5. Each candidate must be a COMPLETE, self-consistent configuration (all fields filled).
6. Ensure diversity: candidates should explore different trade-off points.

**Output JSON Format**:
{{
  "candidate_configs": [
    {{
      "config": {{
        "num_warps": <int>,
        "num_stages": <int>,
        "block_size": {_format_block_size_schema(config)},
        "grid": {{"x": <int>, "y": <int>, "z": <int>}},
        "eviction_policy": "evict_last|evict_first"
      }},
      "rationale": "Why this particular config variant should improve performance",
      "confidence": <float 0-1>,
      "variant_type": "aggressive|conservative|balanced|exploratory"
    }}
  ],
  "decision_rationale": {{
    "primary_bottleneck": "warp|sm|gpu level bottleneck",
    "bottleneck_severity": "critical|moderate|minor",
    "optimization_strategy": "high-level strategy chosen",
    "conflicts_resolved": ["list of conflicts and how resolved"],
    "tradeoffs_accepted": ["list of tradeoffs"]
  }},
  "expected_improvement": {{
    "speedup_estimate": "percentage range across candidates",
    "confidence_score": <float 0-1>,
    "risk_level": "low|medium|high",
    "risk_factors": ["list"]
  }},
  "implementation_instructions": {{
    "step_by_step": ["instruction 1", "instruction 2", ...],
    "critical_changes": ["most important changes"],
    "validation_steps": ["how to verify improvement"]
  }},
  "agent_weight_analysis": {{
    "warp_proposal_weight": <float 0-1>,
    "sm_proposal_weight": <float 0-1>,
    "gpu_proposal_weight": <float 0-1>,
    "weighting_rationale": "why these weights"
  }},
  "comparison_with_current": {{
    "improvements": ["list of improvements"],
    "regressions": ["potential regressions if any"],
    "net_assessment": "overall assessment"
  }}
}}"""

        # Add benchmark feedback if available (closed-loop verification)
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            orig_ms = bench.get("original_stats", {}).get("median", "N/A")
            rec_ms = bench.get("recommended_stats", {}).get("median", "N/A")
            prev_rec_config = bench.get("recommended_config", {})

            if speedup is not None and speedup < 1.0:
                regression_pct = (1.0 - speedup) * 100

                # Summarise all candidates that were tried (Approach 5)
                prev_candidates = state.get("benchmark_results_per_candidate", [])
                if prev_candidates:
                    tried_summary = "\n".join(
                        f"  - Candidate {r.get('candidate_index', '?')}: "
                        f"speedup={r.get('speedup', 'error')}"
                        + (f"x" if isinstance(r.get("speedup"), (int, float)) else "")
                        + (f" ({r.get('error', '')})" if r.get("error") else "")
                        for r in prev_candidates
                    )
                    candidates_note = f"""
All {len(prev_candidates)} candidate configurations were benchmarked — none beat the original.
Results per candidate:
{tried_summary}

Propose DIFFERENT approaches — vary parameters the previous candidates did not explore.
"""
                else:
                    candidates_note = ""

                user_prompt += f"""

**CRITICAL - BENCHMARK FEEDBACK FROM PREVIOUS ITERATION**:
Your previous candidates were MEASURED on actual hardware:
- Original config execution time: {orig_ms}ms
- Best candidate execution time: {rec_ms}ms
- Best speedup: {speedup:.3f}x
- Result: REGRESSION — best candidate was {regression_pct:.1f}% SLOWER
{candidates_note}
Previous best recommended config that FAILED:
{json.dumps(prev_rec_config, indent=2)}

**INSTRUCTION**: All new candidates MUST be measurably faster than {orig_ms}ms.
If you cannot confidently propose an improvement, set all candidate parameters
to match the original configuration.
"""
            else:
                user_prompt += f"""

**BENCHMARK FEEDBACK FROM PREVIOUS ITERATION**:
Your previous recommendation was MEASURED on actual hardware:
- Original config execution time: {orig_ms}ms
- Your recommended config execution time: {rec_ms}ms
- Speedup: {speedup:.3f}x
- Result: SUCCESS — improvement confirmed
"""

        try:
            # Scale max_tokens with candidate count (~500 tokens per candidate + 1500 base)
            max_tokens = 1500 + num_candidates * 500
            response = self.llm.generate(system_prompt, user_prompt, response_format="json", max_tokens=max_tokens, agent_name=self.agent_name)
            coordination_result = self.llm.parse_json_response(response)

            # --- Extract candidate configs (Approach 5) ---
            valid_names = _get_valid_block_names(config)

            if "candidate_configs" in coordination_result:
                # New multi-candidate format
                candidates = coordination_result["candidate_configs"]
            elif "recommended_config" in coordination_result:
                # Fallback: old single-config format — wrap into a 1-element list
                candidates = [{
                    "config": coordination_result["recommended_config"],
                    "rationale": coordination_result.get("decision_rationale", {}).get("optimization_strategy", ""),
                    "confidence": coordination_result.get("expected_improvement", {}).get("confidence_score", 0.5),
                    "variant_type": "balanced",
                }]
            else:
                candidates = []

            # Validate and correct block-size parameter names in ALL candidates
            for candidate in candidates:
                raw_block = candidate.get("config", {}).get("block_size", {})
                if raw_block:
                    candidate["config"]["block_size"] = _validate_block_size_names(raw_block, valid_names)

            state["candidate_configs"] = candidates

            # Use top candidate as the recommended_config for backward compat
            top_config = candidates[0]["config"] if candidates else {}

            # Build final optimization result
            optimization_result = {
                "kernel_name": config.kernel_name,
                "original_config": {
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                    "block_size": config.block_size,
                    "grid": config.grid,
                },
                "recommended_config": top_config,
                "expected_improvement": coordination_result.get("expected_improvement", {}).get("speedup_estimate", "unknown"),
                "confidence_score": coordination_result.get("expected_improvement", {}).get("confidence_score", 0.5),
                "warp_analysis": state["warp_analysis"],
                "sm_analysis": state["sm_analysis"],
                "gpu_analysis": state["gpu_analysis"],
                "coordinator_reasoning": json.dumps(coordination_result.get("decision_rationale", {}), indent=2),
                "tradeoffs": coordination_result.get("decision_rationale", {}).get("tradeoffs_accepted", []),
                "implementation_instructions": "\n".join(coordination_result.get("implementation_instructions", {}).get("step_by_step", [])),
                "predicted_speedup": None,  # Would require performance model
                "risk_assessment": coordination_result.get("expected_improvement", {}).get("risk_level", "medium"),
            }

            state["optimization_result"] = optimization_result
            state["converged"] = True

            logger.info(f"{self.agent_name}: Coordination complete — {len(candidates)} candidate(s)")
            logger.info(f"Top candidate config: {top_config}")

        except Exception as e:
            logger.error(f"{self.agent_name}: Coordination failed: {e}")
            state["error"] = f"CoordinatorAgent error: {e}"

        return state


# ---------------------------------------------------------------------------
# Approach 7 agents  (new architecture — added alongside the original four)
# ---------------------------------------------------------------------------

# Valid power-of-2 warp counts (Triton constraint).
_VALID_NUM_WARPS = [1, 2, 4, 8, 16, 32]

# Byte width lookup for common dtypes.
_DTYPE_BYTES: Dict[str, int] = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "int32": 4, "int64": 8, "int16": 2, "int8": 1,
    "float64": 8, "bool": 1,
}


class DataMovementAgent(BaseOptimizationAgent):
    """Data Movement Agent — analyses HBM -> L2 -> L1 -> Register data path.

    Owns: ``block_size`` (tile dimensions), ``num_stages`` (pipeline depth).
    Does **not** touch: ``num_warps``, grid dimensions.
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("DataMovementAgent", llm_provider, architecture_code)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def analyze(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Analyse memory hierarchy and propose block_size / num_stages."""
        logger.info(f"{self.agent_name}: Starting data-movement analysis")

        config: KernelConfig = state["kernel_config"]
        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)
        strategy = get_optimization_strategy(kernel_type, self.architecture)
        tensor_layout = _analyze_tensor_layout(config.tensor_shapes, config.tensor_dtypes)
        valid_names = _get_valid_block_names(config)

        # ---- Pre-compute data-movement metrics for the LLM ----
        total_tensor_bytes = sum(
            _compute_total_elements(s)
            * _DTYPE_BYTES.get(config.tensor_dtypes.get(n, "float32"), 4)
            for n, s in config.tensor_shapes.items()
        )
        first_block_val = next(iter(config.block_size.values()), 128)
        first_total = next(
            (_compute_total_elements(s) for s in config.tensor_shapes.values()), 1
        )
        reuse_factor = first_total / max(first_block_val, 1)
        pipeline_opportunity = config.global_loads >= 2 and config.num_stages <= 1

        # ---- Pre-compute occupancy for current config ----
        threads_per_block = config.num_warps * self.architecture.warp_size
        max_blocks_threads = self.architecture.max_threads_per_sm // max(threads_per_block, 1)
        max_blocks_warps = self.architecture.max_warps_per_sm // max(config.num_warps, 1)
        max_blocks_smem = (
            (self.architecture.shared_memory_per_sm_kb * 1024) // max(config.shared_memory_bytes, 1)
            if config.shared_memory_bytes > 0 else 999
        )
        current_max_blocks = min(max_blocks_threads, max_blocks_smem, max_blocks_warps)
        current_occupancy_pct = (
            current_max_blocks * config.num_warps
            / self.architecture.max_warps_per_sm * 100
            if self.architecture.max_warps_per_sm > 0 else 0
        )

        # ---- System prompt ----
        system_prompt = f"""You are the Data Movement Agent analysing the GPU memory hierarchy on {self.architecture.name}.

Your PRIMARY LENS is data movement — you trace the data path from HBM -> L2 -> L1 -> Registers
and decide how data should be tiled and pipelined to maximize bandwidth utilization.

You propose a COMPLETE configuration: block_size, num_stages, num_warps, shared_memory_bytes.
Grid dimensions are computed automatically from your block_size.

Analysis approach:
1. Memory access patterns — coalesced vs strided, transaction count.
2. Optimal tile size — maximise reuse at the fastest memory level.
3. Pipeline stages — overlap the next tile's load with the current tile's compute.
4. Cache behaviour — L1/L2 hit rates, cache-line utilisation.
5. Feasibility check — verify your proposal doesn't violate resource limits below.
   Compute occupancy for your proposed config. If occupancy drops below 25%,
   reconsider num_warps or block_size.

Think about total bytes moved and transaction efficiency. Wide XBLOCK improves coalescing
when loads are dense and contiguous; small XBLOCK (2-32) can minimize wasted transactions
when access is offset-indirect, when rows are short, or when per-lane work is dominated by
the reduction dim. Deep pipelines (num_stages=2-4) overlap the next tile's load with the
current tile's compute — prefer them when the kernel does meaningful work per tile.

Architecture specs:
Memory hierarchy:
- Memory bandwidth: {self.architecture.memory_bandwidth_gb_s} GB/s
- L2 cache: {self.architecture.l2_cache_size_mb} MB
- L1 cache: {self.architecture.l1_cache_size_kb} KB per SM
- Shared memory: {self.architecture.shared_memory_per_sm_kb} KB per SM
- Coalescing threshold: >{self.architecture.coalescing_efficiency_threshold * 100}%

Compute resources:
- Number of SMs: {self.architecture.num_sms}
- Max warps/SM: {self.architecture.max_warps_per_sm}
- Max threads/SM: {self.architecture.max_threads_per_sm}
- Max threads/block: {self.architecture.max_threads_per_block}
- Warp size: {self.architecture.warp_size}
- Registers/SM: {self.architecture.registers_per_sm}
- Max registers/thread: {self.architecture.max_registers_per_thread}

Current config occupancy: {current_occupancy_pct:.0f}% ({current_max_blocks} blocks/SM)

Kernel type: {kernel_type} ({strategy['characteristic']} workload)

Respond with JSON only."""

        # ---- User prompt ----
        block_instr = _format_block_size_instructions(config)
        block_schema = _format_block_size_schema(config)

        user_prompt = f"""Analyse the following Triton kernel for data-movement optimisations:

**Kernel**: {config.kernel_name}
**Type**: {kernel_type}
**Current config** (all parameters):
- block_size: {config.block_size}
- num_stages: {config.num_stages}
- num_warps: {config.num_warps}
- shared_memory: {config.shared_memory_bytes} bytes
- grid: {config.grid}{f"""
- grid_formula: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}

**Pre-computed data-movement metrics**:
- Total tensor bytes: {total_tensor_bytes:,}
- Reuse factor (total elements / block_size): {reuse_factor:.1f}
- Pipeline opportunity: {"yes" if pipeline_opportunity else "no"} (>= 2 global loads & num_stages <= 1)

**Tensor Shapes**: {json.dumps(config.tensor_shapes, indent=2)}
**Tensor Dtypes**: {json.dumps(config.tensor_dtypes, indent=2)}
**Total elements per tensor**: {json.dumps({k: _compute_total_elements(v) for k, v in config.tensor_shapes.items()}, indent=2)}

**Tensor Layout Analysis**:
{json.dumps(tensor_layout, indent=2)}

{block_instr}

**Memory Operations**:
- Global loads: {config.global_loads}
- Global stores: {config.global_stores}
- Shared memory: {config.shared_memory_bytes} bytes

**Source Code**:
```python
{config.source_code if config.source_code else 'Not available'}
```

**TTGIR**:
```
{config.ttgir}
```

Provide JSON matching this schema:
{{
  "preferred": {{
    "block_size": {block_schema},
    "num_stages": <int>,
    "num_warps": <int, must be power of 2: 1|2|4|8|16|32>,
    "shared_memory_bytes": <int>
  }},
  "alternatives": [
    {{
      "block_size": {block_schema},
      "num_stages": <int>,
      "num_warps": <int>,
      "shared_memory_bytes": <int>,
      "tradeoff": "explanation of why this differs from preferred"
    }}
  ],
  "hard_constraints": ["physics-based, non-negotiable constraints"],
  "soft_constraints": ["preferences, negotiable"],
  "rationale": "explanation of analysis",
  "what_I_need_from_others": "what ResourceBudget decisions would affect your analysis",
  "confidence": <float 0-1>,
  "analysis": {{
    "coalescing_pattern": "description",
    "reuse_factor": <float>,
    "working_set_bytes": <int>,
    "pipeline_assessment": "description",
    "cache_behavior": "description"
  }}
}}

**IMPORTANT**: num_warps MUST be a power of 2 (1, 2, 4, 8, 16, or 32).
Provide exactly {num_candidates} entries in "alternatives" (distinct trade-off points).

**Breadth requirement**: Do NOT emit "XBLOCK must be a multiple of 32", "XBLOCK >= 64",
"num_warps >= 4", or "num_stages = 1" as HARD constraints. These are preferences, not
physical limits. Your alternatives MUST span at least:
- a 4x range in XBLOCK (e.g. one small <= 16, one medium 32-128, one large >= 256);
- two distinct num_warps values (nw=2 is a valid first-class option);
- two distinct num_stages values (ns >= 2 is valid whenever there's non-trivial reduction work).

**Small-grid mandate**: If the kernel's grid is small relative to {self.architecture.num_sms} SMs *
{self.architecture.max_warps_per_sm} warps/SM = {self.architecture.num_sms * self.architecture.max_warps_per_sm} concurrent warps
(specifically, if `grid.x * grid.y * grid.z * num_warps` < {self.architecture.num_sms * self.architecture.max_warps_per_sm}), low-occupancy configs
(num_warps=1 with XBLOCK in 8/16/32) are NOT under-utilization — they minimize register
pressure and barrier overhead and are often the global optimum on this GPU. In such cases
your slate MUST include at least one (num_warps=1, XBLOCK<=32) candidate.

**{self.architecture.name} architecture rules of thumb**:
- {self.architecture.num_sms} SMs * {self.architecture.max_warps_per_sm} warps/SM = {self.architecture.num_sms * self.architecture.max_warps_per_sm} concurrent warps total. If `grid.x * grid.y * grid.z * num_warps` already
  exceeds 4x that ({self.architecture.num_sms * self.architecture.max_warps_per_sm * 4}), going to num_warps=8 or 16 buys no extra parallelism.
- Each SM has {self.architecture.shared_memory_per_sm_kb} KB shared memory and {self.architecture.registers_per_sm} registers. A config of
  `XBLOCK * RBLOCK * 4 bytes/elem` exceeding {self.architecture.shared_memory_per_sm_kb // 2} KB will likely fail with
  "out of resource: shared memory" — back off to smaller blocks.
- Element-wise pointwise/bmm/clone kernels rarely benefit from num_warps>=4 because
  there is no inter-warp data sharing to amortize the warp overhead.
"""

        # ---- Append negotiation context (if in negotiation sub-loop) ----
        if state.get("negotiation_in_progress"):
            rb_prop = state.get("resource_budget_proposal", {})
            conflict_info = state.get("conflict_detection") or {}
            neg_round = state.get("negotiation_round", 0)
            user_prompt += f"""
**NEGOTIATION ROUND {neg_round}**:
The ResourceBudgetAgent proposed this config:
{json.dumps(rb_prop.get('preferred', {}), indent=2)}

Their rationale: {rb_prop.get('rationale', 'N/A')}
Their hard constraints: {rb_prop.get('hard_constraints', [])}

Detected conflicts:
{json.dumps(conflict_info.get('conflicts', []), indent=2)}

INSTRUCTION: Revise your proposal to address the conflicts.
Explain which constraints you are accommodating and which you cannot.
"""

        # ---- Append agent-scoped feedback (Approach 7 feedback loop) ----
        dm_feedback = state.get("agent_feedback", {}).get("data_movement_feedback")
        if dm_feedback:
            user_prompt += f"\n\n{dm_feedback}\n"

        # ---- Append generic benchmark regression feedback (backward compat) ----
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            if speedup is not None and speedup < 1.0:
                user_prompt += f"""
**BENCHMARK DATA FROM PREVIOUS ATTEMPT**:
- Original config time: {bench.get('original_stats', {}).get('median', 'N/A')}ms
- Previous recommendation time: {bench.get('recommended_stats', {}).get('median', 'N/A')}ms
- Speedup: {speedup:.3f}x (REGRESSION)
- Failed config: {bench.get('recommended_config', {})}
Analyze why the FULL configuration failed — consider all parameter interactions.
"""

        # ---- Append iteration refinement feedback ----
        if state.get("iteration", 0) > 0 and state.get("previous_coordinator_feedback"):
            user_prompt += f"""
**ITERATION {state['iteration']} — REFINEMENT**:
{state['previous_coordinator_feedback']}
"""

        # ---- LLM call ----
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.generate(
                        system_prompt, user_prompt,
                        response_format="json", agent_name=self.agent_name,
                    )
                    analysis = self.llm.parse_json_response(response)
                    break  # success
                except ValueError:
                    if attempt < max_retries - 1:
                        logger.warning(f"{self.agent_name}: JSON parse failed, retrying ({attempt + 1}/{max_retries})")
                        continue
                    raise

            # Validate block-size names in preferred + alternatives
            raw_preferred_bs = analysis.get("preferred", {}).get("block_size", {})
            if raw_preferred_bs:
                analysis["preferred"]["block_size"] = _validate_block_size_names(
                    raw_preferred_bs, valid_names,
                )
            for alt in analysis.get("alternatives", []):
                raw_alt_bs = alt.get("block_size", {})
                if raw_alt_bs:
                    alt["block_size"] = _validate_block_size_names(raw_alt_bs, valid_names)

            # Validate num_warps is power-of-2 in preferred + alternatives
            pref_nw = analysis.get("preferred", {}).get("num_warps")
            if pref_nw is not None and pref_nw not in _VALID_NUM_WARPS:
                closest = min(_VALID_NUM_WARPS, key=lambda v: abs(v - pref_nw))
                logger.warning(f"DM: Corrected preferred num_warps={pref_nw} -> {closest}")
                analysis["preferred"]["num_warps"] = closest
            for alt in analysis.get("alternatives", []):
                alt_nw = alt.get("num_warps")
                if alt_nw is not None and alt_nw not in _VALID_NUM_WARPS:
                    alt["num_warps"] = min(_VALID_NUM_WARPS, key=lambda v: abs(v - alt_nw))

            # Fill defaults for any missing params (LLM may omit unchanged params)
            pref = analysis.get("preferred", {})
            pref.setdefault("num_warps", config.num_warps)
            pref.setdefault("shared_memory_bytes", config.shared_memory_bytes)
            pref.setdefault("num_stages", config.num_stages)

            # Store in state
            state["data_movement_proposal"] = analysis

            message = self.create_message(
                recipient="orchestrator",
                message_type="proposal",
                content={"proposal": analysis, "severity": self._assess_severity(analysis)},
                priority=3,
            )
            state["messages"].append(message)
            logger.info(f"{self.agent_name}: Analysis complete")

        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            print(f"    {self.agent_name} ERROR: {e}", flush=True)
            state["error"] = f"DataMovementAgent error: {e}"

        return state

    # ------------------------------------------------------------------
    # severity helpers
    # ------------------------------------------------------------------

    def _assess_severity(self, analysis: Dict[str, Any]) -> str:
        inner = analysis.get("analysis", {})
        reuse = inner.get("reuse_factor", 1.0)
        coalescing = inner.get("coalescing_pattern", "")
        if reuse > 16 or "strided" in coalescing.lower():
            return "critical"
        if reuse > 4 or "partial" in coalescing.lower():
            return "moderate"
        return "minor"

    def _get_severity_score(self, analysis: Dict[str, Any]) -> float:
        sev = self._assess_severity(analysis)
        return {"critical": 1.0, "moderate": 0.6, "minor": 0.3}.get(sev, 0.3)


class ResourceBudgetAgent(BaseOptimizationAgent):
    """Resource Budget Agent — analyses SM-level resource allocation.

    Owns: ``num_warps``, ``shared_memory`` allocation.
    Does **not** touch: ``block_size``, grid dimensions.
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("ResourceBudgetAgent", llm_provider, architecture_code)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _compute_occupancy(self, num_warps: int, shared_memory_bytes: int) -> Dict[str, Any]:
        """Compute SM occupancy breakdown for a given num_warps / shared-memory pair."""
        threads_per_block = num_warps * self.architecture.warp_size
        max_blocks_threads = self.architecture.max_threads_per_sm // max(threads_per_block, 1)
        max_blocks_smem = (
            (self.architecture.shared_memory_per_sm_kb * 1024) // max(shared_memory_bytes, 1)
            if shared_memory_bytes > 0
            else 999
        )
        max_blocks_warps = self.architecture.max_warps_per_sm // max(num_warps, 1)
        max_blocks_per_sm = min(max_blocks_threads, max_blocks_smem, max_blocks_warps)
        theoretical_occupancy = (
            (max_blocks_per_sm * num_warps) / self.architecture.max_warps_per_sm
            if self.architecture.max_warps_per_sm > 0
            else 0.0
        )
        return {
            "threads_per_block": threads_per_block,
            "max_blocks_threads": max_blocks_threads,
            "max_blocks_smem": max_blocks_smem,
            "max_blocks_warps": max_blocks_warps,
            "max_blocks_per_sm": max_blocks_per_sm,
            "theoretical_occupancy": theoretical_occupancy,
            "theoretical_occupancy_pct": theoretical_occupancy * 100,
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def analyze(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Analyse SM resource budget and propose num_warps / shared_memory."""
        logger.info(f"{self.agent_name}: Starting resource-budget analysis")

        config: KernelConfig = state["kernel_config"]
        kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)
        strategy = get_optimization_strategy(kernel_type, self.architecture)

        occ = self._compute_occupancy(config.num_warps, config.shared_memory_bytes)

        # ---- System prompt ----
        system_prompt = f"""You are the Resource Budget Agent analysing SM-level resource allocation on {self.architecture.name}.

Your PRIMARY LENS is compute resource budgeting — you analyse how registers, warps, shared
memory, and occupancy interact to determine the optimal resource allocation for this kernel.

You propose a COMPLETE configuration: num_warps, shared_memory_bytes, block_size, num_stages.

Analysis approach:
1. Register pressure — estimate regs/thread (IMPORTANT: add 20-30 registers for compiler temporaries).
2. Occupancy — compute max blocks/SM from registers, shared memory, and thread limits.
3. Shared memory — explicit SRAM needed? Bank-conflict risk?
4. Warp scheduling — enough warps to hide latency? Too many for a short kernel?
5. Feasibility check — verify your proposal achieves good memory bandwidth utilization.
   Check whether your block_size enables coalesced memory access.

CRITICAL: When estimating registers, add 20-30 for compiler temporaries beyond
what the kernel data requires.  Underestimating register usage is the #1 source
of bad occupancy predictions.

Think about the shared register/SRAM budget across XBLOCK x RBLOCK x num_warps. High
occupancy via scale-up is one valid operating point; small XBLOCK that frees register
budget for large RBLOCK, paired with low num_warps (2), is another valid operating point,
especially when per-row work is small.

Architecture specs:
Compute resources:
- Max warps/SM: {self.architecture.max_warps_per_sm}
- Max threads/SM: {self.architecture.max_threads_per_sm}
- Max threads/block: {self.architecture.max_threads_per_block}
- Warp size: {self.architecture.warp_size}
- Shared memory/SM: {self.architecture.shared_memory_per_sm_kb} KB
- Registers/SM: {self.architecture.registers_per_sm}
- Max registers/thread: {self.architecture.max_registers_per_thread}
- Number of SMs: {self.architecture.num_sms}
- Target occupancy: >{self.architecture.min_occupancy_target * 100}%

Memory hierarchy:
- Memory bandwidth: {self.architecture.memory_bandwidth_gb_s} GB/s
- L2 cache: {self.architecture.l2_cache_size_mb} MB
- L1 cache: {self.architecture.l1_cache_size_kb} KB per SM
- Coalescing threshold: >{self.architecture.coalescing_efficiency_threshold * 100}%

Kernel type: {kernel_type} ({strategy['characteristic']} workload)

Respond with JSON only."""

        # ---- User prompt ----
        valid_names = _get_valid_block_names(config)
        block_instr = _format_block_size_instructions(config)
        block_schema = _format_block_size_schema(config)

        user_prompt = f"""Analyse SM resource utilisation for this kernel:

**Kernel**: {config.kernel_name}
**Type**: {kernel_type}
**Current config** (all parameters):
- num_warps: {config.num_warps}
- shared_memory: {config.shared_memory_bytes} bytes ({config.shared_memory_bytes / 1024:.2f} KB)
- block_size: {config.block_size}
- num_stages: {config.num_stages}
- grid: {config.grid}{f"""
- grid_formula: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}

**Calculated Occupancy** (current config):
- Threads per block: {occ['threads_per_block']}
- Max blocks/SM (threads): {occ['max_blocks_threads']}
- Max blocks/SM (shared mem): {occ['max_blocks_smem']}
- Max blocks/SM (warps): {occ['max_blocks_warps']}
- Effective max blocks: {occ['max_blocks_per_sm']}
- Theoretical occupancy: {occ['theoretical_occupancy_pct']:.1f}%

**Tensor Shapes**: {json.dumps(config.tensor_shapes, indent=2)}
**Tensor Dtypes**: {json.dumps(config.tensor_dtypes, indent=2)}

{block_instr}

**Source Code**:
```python
{config.source_code if config.source_code else 'Not available'}
```

**TTGIR**:
```
{config.ttgir}
```

Provide JSON matching this schema:
{{
  "preferred": {{
    "num_warps": <int, must be power of 2: 1|2|4|8|16|32>,
    "shared_memory_bytes": <int>,
    "block_size": {block_schema},
    "num_stages": <int>
  }},
  "alternatives": [
    {{
      "num_warps": <int>,
      "shared_memory_bytes": <int>,
      "block_size": {block_schema},
      "num_stages": <int>,
      "tradeoff": "explanation"
    }}
  ],
  "hard_constraints": ["physics-based, non-negotiable"],
  "soft_constraints": ["preferences, negotiable"],
  "rationale": "explanation",
  "what_I_need_from_others": "what DataMovement decisions would affect your analysis",
  "confidence": <float 0-1>,
  "analysis": {{
    "register_estimate": {{
      "data_registers": <int>,
      "compiler_overhead": <int>,
      "total": <int>
    }},
    "occupancy_analysis": {{
      "current_occupancy_pct": {occ['theoretical_occupancy_pct']:.1f},
      "limiting_factor": "threads|shared_memory|registers|warps",
      "blocks_per_sm": {occ['max_blocks_per_sm']}
    }},
    "warp_scheduling": "description",
    "sram_utilization": "description"
  }}
}}

**IMPORTANT**: num_warps MUST be a power of 2 (1, 2, 4, 8, 16, or 32).
Provide exactly {num_candidates} entries in "alternatives" (distinct trade-off points).

**Breadth requirement**: Do NOT emit "XBLOCK must be a multiple of 32", "XBLOCK >= 64",
"num_warps >= 4", or "num_stages = 1" as HARD constraints. These are preferences, not
physical limits. Your alternatives MUST span at least:
- a 4x range in XBLOCK (e.g. one small <= 16, one medium 32-128, one large >= 256);
- two distinct num_warps values (nw=2 is a valid first-class option);
- two distinct num_stages values (ns >= 2 is valid whenever there's non-trivial reduction work).

**Small-grid mandate**: If the kernel's grid is small relative to {self.architecture.num_sms} SMs *
{self.architecture.max_warps_per_sm} warps/SM = {self.architecture.num_sms * self.architecture.max_warps_per_sm} concurrent warps
(specifically, if `grid.x * grid.y * grid.z * num_warps` < {self.architecture.num_sms * self.architecture.max_warps_per_sm}), low-occupancy configs
(num_warps=1 with XBLOCK in 8/16/32) are NOT under-utilization — they minimize register
pressure and barrier overhead and are often the global optimum on this GPU. In such cases
your slate MUST include at least one (num_warps=1, XBLOCK<=32) candidate. Do NOT discount
such candidates for "low theoretical occupancy" — at small grid sizes the SMs aren't the
bottleneck.

**{self.architecture.name} architecture rules of thumb**:
- {self.architecture.num_sms} SMs * {self.architecture.max_warps_per_sm} warps/SM = {self.architecture.num_sms * self.architecture.max_warps_per_sm} concurrent warps total. If `grid.x * grid.y * grid.z * num_warps` already
  exceeds 4x that ({self.architecture.num_sms * self.architecture.max_warps_per_sm * 4}), going to num_warps=8 or 16 buys no extra parallelism.
- Each SM has {self.architecture.shared_memory_per_sm_kb} KB shared memory and {self.architecture.registers_per_sm} registers. A config of
  `XBLOCK * RBLOCK * 4 bytes/elem` exceeding {self.architecture.shared_memory_per_sm_kb // 2} KB will likely fail with
  "out of resource: shared memory" — back off to smaller blocks.
- Element-wise pointwise/bmm/clone kernels rarely benefit from num_warps>=4 because
  there is no inter-warp data sharing to amortize the warp overhead.
"""

        # ---- Append DataMovement proposal (sequential awareness) ----
        dm_proposal = state.get("data_movement_proposal")
        if dm_proposal:
            dm_pref = dm_proposal.get("preferred", {})
            user_prompt += f"""
**DataMovementAgent's Proposal** (proposed before you):
- Config: {json.dumps(dm_pref, indent=2)}
- Rationale: {dm_proposal.get('rationale', 'N/A')}
- Hard constraints: {dm_proposal.get('hard_constraints', [])}
- What they need from you: "{dm_proposal.get('what_I_need_from_others', '')}"

Consider how your choices interact with theirs. If you disagree with any parameter,
explain why in your rationale.
"""

        # ---- Append negotiation context (if in negotiation sub-loop) ----
        if state.get("negotiation_in_progress"):
            dm_prop = state.get("data_movement_proposal") or {}
            conflict_info = state.get("conflict_detection") or {}
            neg_round = state.get("negotiation_round", 0)
            if dm_prop:
                user_prompt += f"""
**NEGOTIATION ROUND {neg_round}**:
The DataMovementAgent revised their proposal to:
{json.dumps(dm_prop.get('preferred', {}), indent=2)}

Their rationale: {dm_prop.get('rationale', 'N/A')}
Their hard constraints: {dm_prop.get('hard_constraints', [])}

Detected conflicts:
{json.dumps(conflict_info.get('conflicts', []), indent=2)}

INSTRUCTION: Revise your proposal to address the conflicts.
Explain which constraints you are accommodating and which you cannot.
"""
            else:
                # DM proposal masked — show conflict summary without
                # revealing DM's exact values to prevent anchoring bias.
                raw_conflicts = conflict_info.get('conflicts', [])
                sanitized = []
                dm_keys = {'dm_position', 'dm_value', 'dm_wants',
                           'what_dm_needs', 'resolution_hint'}
                for c in raw_conflicts:
                    s = {k: v for k, v in c.items() if k not in dm_keys}
                    sanitized.append(s)
                user_prompt += f"""
**NEGOTIATION ROUND {neg_round}**:
Detected conflicts between your proposal and the other agent:
{json.dumps(sanitized, indent=2)}

INSTRUCTION: Revise your proposal to address the conflicts.
Explain which constraints you are accommodating and which you cannot.
"""

        # ---- Append agent-scoped feedback (Approach 7) ----
        rb_feedback = state.get("agent_feedback", {}).get("resource_budget_feedback")
        if rb_feedback:
            user_prompt += f"\n\n{rb_feedback}\n"

        # ---- Append generic benchmark regression feedback ----
        if state.get("benchmark_results") and state.get("benchmark_iterations", 0) > 0:
            bench = state["benchmark_results"]
            speedup = bench.get("speedup", 1.0)
            if speedup is not None and speedup < 1.0:
                user_prompt += f"""
**BENCHMARK DATA FROM PREVIOUS ATTEMPT**:
- Original config time: {bench.get('original_stats', {}).get('median', 'N/A')}ms
- Previous recommendation time: {bench.get('recommended_stats', {}).get('median', 'N/A')}ms
- Speedup: {speedup:.3f}x (REGRESSION)
- Failed config: {bench.get('recommended_config', {})}
Analyze why the FULL configuration failed — consider all parameter interactions.
"""

        # ---- Iteration refinement ----
        if state.get("iteration", 0) > 0 and state.get("previous_coordinator_feedback"):
            user_prompt += f"""
**ITERATION {state['iteration']} — REFINEMENT**:
{state['previous_coordinator_feedback']}
"""

        # ---- LLM call ----
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.generate(
                        system_prompt, user_prompt,
                        response_format="json", agent_name=self.agent_name,
                    )
                    analysis = self.llm.parse_json_response(response)
                    break  # success
                except ValueError:
                    if attempt < max_retries - 1:
                        logger.warning(f"{self.agent_name}: JSON parse failed, retrying ({attempt + 1}/{max_retries})")
                        continue
                    raise

            # Validate num_warps is power-of-2 in preferred + alternatives
            pref_nw = analysis.get("preferred", {}).get("num_warps")
            if pref_nw is not None and pref_nw not in _VALID_NUM_WARPS:
                closest = min(_VALID_NUM_WARPS, key=lambda v: abs(v - pref_nw))
                logger.warning(f"Corrected preferred num_warps={pref_nw} -> {closest}")
                analysis["preferred"]["num_warps"] = closest
            for alt in analysis.get("alternatives", []):
                alt_nw = alt.get("num_warps")
                if alt_nw is not None and alt_nw not in _VALID_NUM_WARPS:
                    alt["num_warps"] = min(_VALID_NUM_WARPS, key=lambda v: abs(v - alt_nw))

            # Validate block-size names in preferred + alternatives
            raw_preferred_bs = analysis.get("preferred", {}).get("block_size", {})
            if raw_preferred_bs:
                analysis["preferred"]["block_size"] = _validate_block_size_names(
                    raw_preferred_bs, valid_names,
                )
            for alt in analysis.get("alternatives", []):
                raw_alt_bs = alt.get("block_size", {})
                if raw_alt_bs:
                    alt["block_size"] = _validate_block_size_names(raw_alt_bs, valid_names)

            # Fill defaults for any missing params (LLM may omit unchanged params)
            pref = analysis.get("preferred", {})
            pref.setdefault("block_size", dict(config.block_size) if config.block_size else {})
            pref.setdefault("num_stages", config.num_stages)
            pref.setdefault("num_warps", config.num_warps)
            pref.setdefault("shared_memory_bytes", config.shared_memory_bytes)

            # Store in state
            state["resource_budget_proposal"] = analysis

            message = self.create_message(
                recipient="orchestrator",
                message_type="proposal",
                content={"proposal": analysis, "severity": self._assess_severity(analysis)},
                priority=3,
            )
            state["messages"].append(message)
            logger.info(f"{self.agent_name}: Analysis complete")

        except Exception as e:
            logger.error(f"{self.agent_name}: Analysis failed: {e}")
            print(f"    {self.agent_name} ERROR: {e}", flush=True)
            state["error"] = f"ResourceBudgetAgent error: {e}"

        return state

    # ------------------------------------------------------------------
    # severity helpers
    # ------------------------------------------------------------------

    def _assess_severity(self, analysis: Dict[str, Any]) -> str:
        occ_pct = (
            analysis.get("analysis", {})
            .get("occupancy_analysis", {})
            .get("current_occupancy_pct", 100)
        )
        if occ_pct < 25:
            return "critical"
        if occ_pct < self.architecture.min_occupancy_target * 100:
            return "moderate"
        return "minor"

    def _get_severity_score(self, analysis: Dict[str, Any]) -> float:
        sev = self._assess_severity(analysis)
        return {"critical": 1.0, "moderate": 0.6, "minor": 0.3}.get(sev, 0.3)


class OrchestratorAgent(BaseOptimizationAgent):
    """Orchestrator Agent — synthesises proposals from DataMovement and
    ResourceBudget agents, detects conflicts, and emits candidate configs.

    Replaces ``CoordinatorAgent`` in the Approach 7 architecture.
    """

    def __init__(self, llm_provider: LLMProvider, architecture_code: str = "cuda:70"):
        super().__init__("OrchestratorAgent", llm_provider, architecture_code)

    # ------------------------------------------------------------------
    # conflict detection (programmatic — no LLM)
    # ------------------------------------------------------------------

    def _detect_conflicts(
        self,
        dm_proposal: Dict[str, Any],
        rb_proposal: Dict[str, Any],
        grid_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check whether the two agents agree with each other's implementations.

        Compares each agent's ``what_I_need_from_others`` against what the
        other agent actually proposed and checks whether the *combined* config
        causes problems visible in *grid_info* execution metrics.

        Returns a list of conflict dicts (empty list means consensus).
        """
        conflicts: List[Dict[str, Any]] = []

        dm_preferred = dm_proposal.get("preferred", {})
        rb_preferred = rb_proposal.get("preferred", {})
        dm_needs = dm_proposal.get("what_I_need_from_others", "")
        rb_needs = rb_proposal.get("what_I_need_from_others", "")

        # 1. Direct parameter comparison (both agents now propose full configs)
        dm_bs = dm_preferred.get("block_size", {})
        rb_bs = rb_preferred.get("block_size", {})
        if dm_bs and rb_bs and dm_bs != rb_bs:
            conflicts.append({
                "parameter": "block_size",
                "dm_wants": dm_bs,
                "rb_wants": rb_bs,
                "severity": "high",
                "resolution_hint": (
                    "DM has data-movement expertise; consider their block_size "
                    "unless RB cites register pressure."
                ),
            })

        dm_nw = dm_preferred.get("num_warps")
        rb_nw = rb_preferred.get("num_warps")
        if dm_nw is not None and rb_nw is not None and dm_nw != rb_nw:
            conflicts.append({
                "parameter": "num_warps",
                "dm_wants": dm_nw,
                "rb_wants": rb_nw,
                "severity": "high",
                "resolution_hint": (
                    "RB has SM-resource expertise; consider their num_warps "
                    "unless DM cites latency hiding."
                ),
            })

        dm_ns = dm_preferred.get("num_stages")
        rb_ns = rb_preferred.get("num_stages")
        if dm_ns is not None and rb_ns is not None and dm_ns != rb_ns:
            conflicts.append({
                "parameter": "num_stages",
                "dm_wants": dm_ns,
                "rb_wants": rb_ns,
                "severity": "medium",
                "resolution_hint": (
                    "DM has pipelining expertise; favor their num_stages "
                    "unless RB cites shared memory overflow."
                ),
            })

        # 2. Check mutual needs (demoted to medium severity) -------------------
        if dm_needs:
            conflicts.append({
                "parameter": "DataMovement needs vs ResourceBudget proposal",
                "dm_position": json.dumps(dm_preferred),
                "rb_position": json.dumps(rb_preferred),
                "what_dm_needs": dm_needs,
                "what_rb_needs": "(see below)",
                "severity": "medium",
                "resolution_hint": (
                    "Check whether ResourceBudget's preferred config "
                    "satisfies DataMovement's stated needs."
                ),
            })
        if rb_needs:
            conflicts.append({
                "parameter": "ResourceBudget needs vs DataMovement proposal",
                "dm_position": json.dumps(dm_preferred),
                "rb_position": json.dumps(rb_preferred),
                "what_dm_needs": "(see above)",
                "what_rb_needs": rb_needs,
                "severity": "medium",
                "resolution_hint": (
                    "Check whether DataMovement's preferred config "
                    "satisfies ResourceBudget's stated needs."
                ),
            })

        # 4. Check combined-config execution quality ---------------------------
        ctas_per_sm = grid_info.get("ctas_per_sm", 0)
        if 0 < ctas_per_sm < 1.0:
            conflicts.append({
                "parameter": "Combined config -> SM saturation",
                "dm_position": json.dumps(dm_preferred),
                "rb_position": json.dumps(rb_preferred),
                "severity": "medium",
                "resolution_hint": (
                    f"Only {ctas_per_sm:.2f} CTAs/SM — SMs will be under-utilised. "
                    "Consider a smaller block_size to increase grid dimension."
                ),
            })

        # 5. Check register overflow for combined config -----------------------
        reg_estimate = (
            rb_proposal.get("analysis", {})
            .get("register_estimate", {})
            .get("total", 0)
        )
        nw = rb_preferred.get("num_warps", 4)
        if reg_estimate > 0:
            regs_per_block = reg_estimate * nw * self.architecture.warp_size
            if regs_per_block > self.architecture.registers_per_sm:
                conflicts.append({
                    "parameter": "num_warps vs register pressure",
                    "dm_position": json.dumps(dm_preferred),
                    "rb_position": json.dumps(rb_preferred),
                    "severity": "high",
                    "resolution_hint": (
                        f"Estimated {regs_per_block} regs/block > "
                        f"{self.architecture.registers_per_sm} regs/SM. "
                        "Reduce num_warps or accept lower occupancy."
                    ),
                })

        return conflicts

    # ------------------------------------------------------------------
    # public API (matches CoordinatorAgent.coordinate interface)
    # ------------------------------------------------------------------

    def coordinate(self, state: OptimizationState, num_candidates: int = 3) -> OptimizationState:
        """Synthesise proposals, resolve conflicts, emit candidate configs."""
        logger.info(f"{self.agent_name}: Starting orchestration")

        config: KernelConfig = state["kernel_config"]
        dm_proposal = state.get("data_movement_proposal", {})
        rb_proposal = state.get("resource_budget_proposal", {})

        if not dm_proposal and not rb_proposal:
            logger.warning(f"{self.agent_name}: No proposals found — skipping")
            return state

        # Handle partial proposals: if one agent failed, use the other's full
        # proposal as a fallback so the LLM gets coherent input.
        if not dm_proposal or not dm_proposal.get("preferred"):
            logger.warning(f"{self.agent_name}: DataMovement proposal missing — using ResourceBudget proposal as fallback")
            dm_proposal = rb_proposal
        elif not rb_proposal or not rb_proposal.get("preferred"):
            logger.warning(f"{self.agent_name}: ResourceBudget proposal missing — using DataMovement proposal as fallback")
            rb_proposal = dm_proposal

        # Grid / execution metrics (populated by compute_grid node in Step 6,
        # but may be empty during standalone testing).
        grid_info = state.get("computed_grid") or state.get("execution_metrics") or {}

        # If grid_info is empty, compute it inline as a fallback.
        if not grid_info and dm_proposal.get("preferred", {}).get("block_size"):
            kernel_type = detect_kernel_type(config.ttgir, config.kernel_name)
            grid_info = compute_grid(
                dm_proposal["preferred"]["block_size"],
                config.tensor_shapes,
                kernel_type,
                self.architecture.num_sms,
            )

        # ---- Programmatic conflict detection ----
        conflicts = self._detect_conflicts(dm_proposal, rb_proposal, grid_info)
        state["conflict_detection"] = {"conflicts": conflicts}

        valid_names = _get_valid_block_names(config)
        block_instr = _format_block_size_instructions(config)
        block_schema = _format_block_size_schema(config)

        # ---- System prompt ----
        if num_candidates == 1:
            # Agent-proposals mode: orchestrator produces 1 preferred + 2 alternatives
            system_prompt = f"""You are the Orchestrator Agent for {self.architecture.name}.
You synthesise proposals from two specialist agents and resolve any conflicts.

STEPS:
1. Review both proposals (DataMovement and ResourceBudget).
2. Review the computed grid and execution metrics for each proposed block_size.
3. Identify agreements (parameters both agents accept).
4. Resolve conflicts — hard constraints always win over soft constraints.
5. Generate ONE preferred config and TWO alternative configs.
   - The preferred config is your best synthesis of both agents' proposals.
   - Alternative 1: a different trade-off point (e.g., favor one agent's reasoning more).
   - Alternative 2: a more exploratory option (e.g., different block size or warp count).
   Each MUST include block_size, num_warps, num_stages, and optionally shared_memory_bytes.

Rules:
- The preferred config should have the HIGHEST confidence.
- Alternatives should have LOWER confidence but explore DIFFERENT regions of the config space.
- Do NOT include the original config (it is added automatically as baseline).
- Every candidate must be internally consistent.
- Sort by confidence: preferred first, then alternatives in decreasing confidence.

{block_instr}

Respond with JSON only."""
        else:
            system_prompt = f"""You are the Orchestrator Agent for {self.architecture.name}.
You synthesise proposals from two specialist agents and resolve any conflicts.

STEPS:
1. Review both proposals (DataMovement and ResourceBudget).
2. Review the computed grid and execution metrics for each proposed block_size.
3. Identify agreements (parameters both agents accept).
4. Resolve conflicts — hard constraints always win over soft constraints.
   Where two hard constraints conflict, find a compromise value.
5. Generate {num_candidates} candidate configs.  Each MUST include block_size,
   num_warps, num_stages, and optionally shared_memory_bytes.

Rules:
- At least one conservative candidate (change only the highest-confidence parameter).
- At least one that uses an agent's alternative value.
- Do NOT include the original config (it is added automatically as baseline).
- Every candidate must be internally consistent.
- Sort candidates by confidence, highest first.

{block_instr}

Respond with JSON only."""

        # ---- User prompt ----
        user_prompt = f"""**DataMovement Proposal**:
{json.dumps(dm_proposal, indent=2)}

**ResourceBudget Proposal**:
{json.dumps(rb_proposal, indent=2)}

**Computed Grid & Execution Metrics** (for DataMovement's preferred block_size):
{json.dumps(grid_info, indent=2)}

**Detected Potential Conflicts** (programmatic cross-check of mutual needs):
{json.dumps(conflicts, indent=2) if conflicts else "None — agents appear to agree."}

**Original Config** (for reference, do NOT include as a candidate):
- num_warps: {config.num_warps}
- num_stages: {config.num_stages}
- block_size: {config.block_size}
- grid: {config.grid}{f"""
- grid_formula: `{config.grid_formula.strip()}`""" if config.grid_formula else ""}

Output a JSON object with this structure:
{{
  "agreements": {{"parameter": "agreed value", ...}},
  "conflicts": [
    {{
      "parameter": "...",
      "data_movement_wants": "...",
      "resource_budget_wants": "...",
      "resolution": "..."
    }}
  ],"""

        if num_candidates == 1:
            user_prompt += f"""
  "preferred": {{
    "config": {{
      "num_warps": <int>,
      "num_stages": <int>,
      "block_size": {block_schema}
    }},
    "rationale": "...",
    "confidence": <float 0-1>
  }},
  "alternatives": [
    {{
      "config": {{
        "num_warps": <int>,
        "num_stages": <int>,
        "block_size": {block_schema}
      }},
      "rationale": "...",
      "confidence": <float 0-1>
    }}
  ],
  "resolution": "overall resolution reasoning"
}}

Generate 1 preferred config and 2 alternatives. Sort by confidence (preferred highest).
"""
        else:
            user_prompt += f"""
  "candidate_configs": [
    {{
      "config": {{
        "num_warps": <int>,
        "num_stages": <int>,
        "block_size": {block_schema}
      }},
      "rationale": "...",
      "confidence": <float 0-1>,
      "variant_type": "aggressive|conservative|balanced|exploratory"
    }}
  ],
  "resolution": "overall resolution reasoning"
}}

Generate exactly {num_candidates} candidates. Sort by confidence, highest first.
"""

        # Negotiation history (if multi-round, wired in Step 6).
        negotiation_history = state.get("negotiation_history", [])
        if negotiation_history:
            user_prompt += f"""
**Previous Negotiation Rounds**:
{json.dumps(negotiation_history, indent=2)}
"""

        # ---- LLM call ----
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.generate(
                        system_prompt, user_prompt,
                        response_format="json",
                        max_tokens=1500 + num_candidates * 500,
                        agent_name=self.agent_name,
                    )
                    result = self.llm.parse_json_response(response)
                    break  # success
                except ValueError:
                    if attempt < max_retries - 1:
                        logger.warning(f"{self.agent_name}: JSON parse failed, retrying ({attempt + 1}/{max_retries})")
                        continue
                    raise

            # Extract candidates (with backward-compat fallback)
            if "candidate_configs" in result:
                candidates = result["candidate_configs"]
            elif "preferred" in result:
                # New preferred + alternatives format (agent-proposals mode)
                pref = result["preferred"]
                candidates = [{
                    "config": pref.get("config", pref),
                    "rationale": pref.get("rationale", ""),
                    "confidence": pref.get("confidence", 0.85),
                    "variant_type": "orchestrator",
                }]
                # Store orchestrator alternatives for Phase 2
                orch_alts = []
                for alt in result.get("alternatives", []):
                    alt_entry = {
                        "config": alt.get("config", alt),
                        "rationale": alt.get("rationale", ""),
                        "confidence": alt.get("confidence", 0.7),
                        "variant_type": "orchestrator",
                    }
                    candidates.append(alt_entry)  # include all in candidates for now
                    orch_alts.append(alt.get("config", alt))
                state["orchestrator_alternatives"] = orch_alts
            elif "recommended_config" in result:
                candidates = [{
                    "config": result["recommended_config"],
                    "rationale": result.get("decision_rationale", ""),
                    "confidence": result.get("confidence", 0.5),
                    "variant_type": "balanced",
                }]
            else:
                candidates = []

            # Validate block-size names in all candidates
            for cand in candidates:
                raw_bs = cand.get("config", {}).get("block_size", {})
                if raw_bs:
                    cand["config"]["block_size"] = _validate_block_size_names(raw_bs, valid_names)

            # When agent-proposals mode: only keep the preferred as Phase 1 candidate
            # (alternatives are stored separately for Phase 2)
            if num_candidates == 1 and len(candidates) > 1:
                state["orchestrator_alternatives"] = [
                    c["config"] for c in candidates[1:]
                ]
                candidates = candidates[:1]

            state["candidate_configs"] = candidates

            # Build optimization_result (backward compat with benchmark node)
            top_config = candidates[0]["config"] if candidates else {}
            state["optimization_result"] = {
                "kernel_name": config.kernel_name,
                "original_config": {
                    "num_warps": config.num_warps,
                    "num_stages": config.num_stages,
                    "block_size": config.block_size,
                    "grid": config.grid,
                },
                "recommended_config": top_config,
                "confidence_score": candidates[0].get("confidence", 0.5) if candidates else 0.0,
                "data_movement_proposal": dm_proposal,
                "resource_budget_proposal": rb_proposal,
                "conflict_detection": result.get("conflicts", []),
                "coordinator_reasoning": result.get("resolution", ""),
            }
            state["converged"] = True

            logger.info(
                f"{self.agent_name}: Orchestration complete — "
                f"{len(candidates)} candidate(s)"
            )

        except Exception as e:
            logger.error(f"{self.agent_name}: Orchestration failed: {e}")
            print(f"    {self.agent_name} ERROR: {e}", flush=True)
            state["error"] = f"OrchestratorAgent error: {e}"

        return state
