import json
import inspect
import triton
import triton.language as tl
from typing import Dict, List, Any, Optional

class KernelAnalysis:
    """Analyzes Triton kernel code to extract optimization-relevant features"""

    @staticmethod
    def analyze_kernel_code(kernel_source: str) -> Dict[str, Any]:
        """Extract key characteristics from kernel source code"""
        analysis = {
            "operation_type": KernelAnalysis._detect_operation_type(kernel_source),
            "memory_access_patterns": KernelAnalysis._analyze_memory_access(kernel_source),
            "compute_characteristics": KernelAnalysis._analyze_compute(kernel_source),
            "parallelism_structure": KernelAnalysis._analyze_parallelism(kernel_source),
            "data_types": KernelAnalysis._infer_data_types(kernel_source)
        }
        return analysis

    @staticmethod
    def _detect_operation_type(source: str) -> str:
        """Detect the type of operation (elementwise, reduction, matmul, etc.)"""
        source_lower = source.lower()
        if 'tl.where' in source and 'tl.load' in source:
            return 'elementwise_fusion'
        elif 'tl.sum' in source or 'tl.max' in source:
            return 'reduction'
        elif 'tl.dot' in source:
            return 'matrix_operation'
        elif 'convolution' in source:
            return 'convolution'
        else:
            return 'elementwise'

    @staticmethod
    def _analyze_memory_access(source: str) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        patterns = {
            "has_contiguous_access": 'xindex' in source,
            "has_strided_access": 'x0' in source and 'x1' in source,
            "has_conditional_access": 'tl.where' in source,
            "load_count": source.count('tl.load'),
            "store_count": source.count('tl.store'),
            "eviction_policies": []
        }

        # Extract eviction policies
        import re
        eviction_matches = re.findall(r"eviction_policy='([^']*)'", source)
        patterns["eviction_policies"] = list(set(eviction_matches))

        return patterns

    @staticmethod
    def _analyze_compute(source: str) -> Dict[str, Any]:
        """Analyze computational characteristics"""
        compute_ops = {
            'arithmetic_ops': ['+', '-', '*', '/'],
            'comparison_ops': ['==', '!=', '>', '<'],
            'special_ops': ['tl.where', 'tl.sum', 'tl.dot']
        }

        counts = {}
        for op_type, ops in compute_ops.items():
            counts[op_type] = sum(source.count(op) for op in ops)

        return {
            **counts,
            "compute_intensity": "low" if sum(counts.values()) < 5 else "medium",
            "has_conditionals": 'tl.where' in source or 'if' in source
        }

    @staticmethod
    def _analyze_parallelism(source: str) -> Dict[str, Any]:
        """Analyze parallelism structure"""
        return {
            "has_xnumel": 'xnumel' in source,
            "has_xblock": 'XBLOCK' in source,
            "has_multiple_dims": any(dim in source for dim in ['RBLOCK', 'YBLOCK', 'ZBLOCK']),
            "total_work_estimate": KernelAnalysis._estimate_work_size(source)
        }

    @staticmethod
    def _estimate_work_size(source: str) -> str:
        """Estimate the problem size category"""
        if 'xnumel = ' in source:
            # Extract the xnumel value
            import re
            match = re.search(r'xnumel\s*=\s*(\d+)', source)
            if match:
                size = int(match.group(1))
                if size <= 32: return "very_small"
                elif size <= 1024: return "small"
                elif size <= 10000: return "medium"
                else: return "large"
        return "unknown"

    @staticmethod
    def _infer_data_types(source: str) -> List[str]:
        """Infer data types from operations"""
        types = []
        if 'float' in source or '.0' in source:
            types.append('fp32')
        if 'int32' in source:
            types.append('int32')
        if 'half' in source or 'fp16' in source:
            types.append('fp16')
        return types if types else ['likely_fp32']

class GPUArchitecture:
    """Provides GPU architecture information"""

    GPU_DATABASE = {
        "v100": {
            "compute_capability": "7.0",
            "sm_count": 80,
            "tensor_cores": True,
            "tensor_cores_count": 640,
            "shared_memory_per_sm_kb": 96,
            "l2_cache_size_mb": 6,
            "memory_bandwidth_gbs": 900,
            "max_threads_per_sm": 2048,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "register_file_size_kb": 256
        },
        "a100": {
            "compute_capability": "8.0",
            "sm_count": 108,
            "tensor_cores": True,
            "tensor_cores_count": 432,
            "shared_memory_per_sm_kb": 164,
            "l2_cache_size_mb": 40,
            "memory_bandwidth_gbs": 1555,
            "max_threads_per_sm": 2048,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "register_file_size_kb": 256
        },
        "rtx_3090": {
            "compute_capability": "8.6",
            "sm_count": 82,
            "tensor_cores": True,
            "tensor_cores_count": 328,
            "shared_memory_per_sm_kb": 128,
            "l2_cache_size_mb": 6,
            "memory_bandwidth_gbs": 936,
            "max_threads_per_sm": 1536,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "register_file_size_kb": 256
        }
    }

    @staticmethod
    def get_architecture(gpu_name: str) -> Dict[str, Any]:
        """Get architecture details for a specific GPU"""
        gpu_name = gpu_name.lower().replace(' ', '_')
        return GPUArchitecture.GPU_DATABASE.get(gpu_name, {})

class OptimizationPromptGenerator:
    """Generates high-quality prompts for kernel optimization"""

    def __init__(self):
        self.kernel_analyzer = KernelAnalysis()
        self.gpu_arch = GPUArchitecture()

    def generate_prompt(self, kernel_source: str, gpu_model: str,
                       tuning_parameters: Optional[Dict] = None) -> str:
        """Generate a comprehensive optimization prompt"""

        # Analyze kernel
        kernel_analysis = self.kernel_analyzer.analyze_kernel_code(kernel_source)
        gpu_architecture = self.gpu_arch.get_architecture(gpu_model)

        # Default tuning parameters
        if tuning_parameters is None:
            tuning_parameters = {
                "XBLOCK_candidates": [16, 32, 64, 128, 256],
                "num_warps_candidates": [1, 2, 4, 8],
                "num_stages_candidates": [1, 2, 3],
                "eviction_policy_candidates": ["evict_first", "evict_last", "normal"]
            }

        prompt_structure = {
            "task_description": "Optimize Triton kernel configuration for maximum performance",
            "gpu_architecture": gpu_architecture,
            "kernel_characteristics": kernel_analysis,
            "tuning_parameters": tuning_parameters,
            "optimization_guidelines": self._get_optimization_guidelines(),
            "reasoning_framework": self._get_reasoning_framework(),
            "kernel_source": kernel_source,
            "expected_output_format": {
                "recommended_configuration": {
                    "XBLOCK": "integer value",
                    "num_warps": "integer value",
                    "num_stages": "integer value",
                    "eviction_policy": "string value",
                    "additional_notes": "string explaining the reasoning"
                },
                "reasoning": "step-by-step explanation",
                "expected_improvement": "what performance gain is expected and why"
            }
        }

        return self._format_prompt(prompt_structure)

    def _get_optimization_guidelines(self) -> List[str]:
        """Provide optimization guidelines for the LLM"""
        return [
            "Match parallelism to problem size - avoid oversubscription for small problems",
            "Consider memory access patterns when choosing eviction policies",
            "Balance thread occupancy with resource utilization",
            "Use Tensor Cores for appropriate data types and problem sizes",
            "Consider cache hierarchy sizes for memory-bound kernels",
            "Optimize for latency vs throughput based on problem scale"
        ]

    def _get_reasoning_framework(self) -> List[str]:
        """Provide a reasoning framework for the LLM"""
        return [
            "1. Analyze kernel compute vs memory characteristics",
            "2. Map kernel requirements to GPU architecture strengths",
            "3. Consider problem size vs parallel resources balance",
            "4. Evaluate memory hierarchy utilization opportunities",
            "5. Select configuration that minimizes bottlenecks"
        ]

    def _format_prompt(self, prompt_data: Dict) -> str:
        """Format the prompt data into a readable string"""
        prompt = "# GPU Kernel Optimization Analysis\n\n"

        prompt += "## Task Description\n"
        prompt += f"{prompt_data['task_description']}\n\n"

        prompt += "## GPU Architecture\n"
        for key, value in prompt_data['gpu_architecture'].items():
            prompt += f"- {key}: {value}\n"
        prompt += "\n"

        prompt += "## Kernel Characteristics\n"
        for category, details in prompt_data['kernel_characteristics'].items():
            prompt += f"### {category.replace('_', ' ').title()}\n"
            if isinstance(details, dict):
                for k, v in details.items():
                    prompt += f"  - {k}: {v}\n"
            else:
                prompt += f"  - {details}\n"
            prompt += "\n"

        prompt += "## Tuning Parameters\n"
        for param, values in prompt_data['tuning_parameters'].items():
            prompt += f"- {param}: {values}\n"
        prompt += "\n"

        prompt += "## Optimization Guidelines\n"
        for guideline in prompt_data['optimization_guidelines']:
            prompt += f"- {guideline}\n"
        prompt += "\n"

        prompt += "## Reasoning Framework\n"
        for step in prompt_data['reasoning_framework']:
            prompt += f"{step}\n"
        prompt += "\n"

        prompt +="## Kernel Source\n"
        prompt +="Use the following Triton kernel as context for optimization. Do not modify the code.\n\n"
        prompt +="```python\n"
        prompt += f"{prompt_data['kernel_source']}```\n\n"

        prompt += "## Expected Output Format\n"
        prompt += "Please provide your analysis in the following format:\n"
        prompt += json.dumps(prompt_data['expected_output_format'], indent=2)

        return prompt

# Example usage and demonstration
def demo():
    """Demonstrate the prompt generator with your kernel"""

    # Your kernel source
    kernel_code = """
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = xindex % 4
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + x2, xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tl.where(tmp2, tmp5, tmp6)
    tl.store(out_ptr0 + x2, tmp7, xmask)
"""

    # Create generator and generate prompt
    generator = OptimizationPromptGenerator()
    prompt = generator.generate_prompt(kernel_code, "v100")

    print("Generated Prompt:")
    print("=" * 80)
    print(prompt)

    # Also show the analysis separately
    print("\n" + "=" * 80)
    print("Kernel Analysis Summary:")
    analyzer = KernelAnalysis()
    analysis = analyzer.analyze_kernel_code(kernel_code)
    print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    demo()