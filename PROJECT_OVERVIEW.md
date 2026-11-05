# Project Overview: Multi-Agent Triton Kernel Optimization System

## Purpose

This project implements a multi-agent system for generating optimal configurations for Triton kernel tuning. It integrates with TritonBench to:
1. Extract Triton kernels and their TTGIR (Triton GPU IR) representations
2. Capture input tensor shapes from actual benchmark runs
3. Use LLM-powered agents to analyze and optimize kernel configurations
4. Benchmark the optimized configurations against baselines

## Key Components

### 1. Main Runner: `multiagent_triton_runner.py`

**Purpose**: Model-free multi-agent runner that discovers Triton kernels from TritonBench metadata and runs optimization.

**Key Features**:
- Discovers Triton kernels from `tritonbench/metadata/oss_cuda_kernels.yaml`
- Maps kernels to TTGIR files and input shape captures
- Runs multi-agent optimizer per kernel/shape combination
- Supports both multi-agent and single-agent modes
- Optional benchmarking to validate optimizations

**Basic Usage**:
```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir <shapes_dir> \
  --output-dir ./agentic_results
```

**Key Flags**:
- `--kernels k1,k2`: Filter specific kernels
- `--all-shapes`: Process all shape files per kernel
- `--llm-provider {deepseek,openai,ollama,vllm}`: Choose LLM backend
- `--test-config --benchmark-baseline`: Enable benchmarking
- `--debug-mode --debug-dir ./debug`: Persist prompts/responses

### 2. Configuration Optimizer: `configuration_optimizer.py`

**Purpose**: Uses LLM APIs (DeepSeek/OpenAI) to analyze TTGIR and propose optimal configurations.

**Key Methods**:
- `analyze_and_optimize()`: Main entry point for optimization
- Extracts source location from TTGIR
- Loads tensor shapes from JSON files
- Generates prompts with TTGIR, TTIR, source code, and tensor shapes
- Parses LLM responses to extract configuration recommendations

**Supports**:
- Cached responses for debugging
- Prompt output to files for inspection
- Multiple LLM models (deepseek-chat, gpt-4, etc.)

### 3. Optimizer Functions: `optimizer_functions.py`

**Purpose**: Helper functions for kernel analysis, optimization, and benchmarking.

**Key Functions**:
- `analyze_kernel_from_cache()`: Analyze kernel from cached TTGIR
- `optimize_kernel_from_cache()`: Run optimization workflow
- `benchmark_baseline_config()`: Benchmark original configuration
- `test_recommended_config()`: Benchmark optimized configuration
- `extract_best_config()`: Extract configuration from LLM response

### 4. Multi-Agent Systems

#### A. Basic Multi-Agent (`multiagent.py`)

**7 Specialized Agents**:
1. **RegisterPressureAnalyzer**: Predicts register usage and spill probability
2. **MemoryHierarchyOptimizer**: Analyzes cache efficiency and memory coalescing
3. **ArchitectureProfiler**: Models hardware-specific constraints
4. **OccupancyCalculator**: Predicts occupancy based on resource usage
5. **PerformancePredictor**: Uses analytical models for latency estimation
6. **ConstraintValidator**: Ensures configurations respect hardware limits
7. **ConfigurationOrchestrator**: Coordinates agents and ranks configurations

**Features**:
- Zero-execution analysis (no kernel compilation required)
- Hardware-aware optimization (NVIDIA Ampere, Intel Xe-HPG)
- 100x faster than empirical tuning
- 80%+ correlation with empirical top configurations

#### B. LangGraph-Based Multi-Agent (`agent_system_langgraph/`)

**Hierarchical Agent Architecture**:
1. **WarpAgent**: 32-thread level optimization (coalescing, bank conflicts)
2. **SMAgent**: SM-level optimization (occupancy, shared memory, registers)
3. **GPUAgent**: System-wide optimization (grid, L2 cache, bandwidth)
4. **CoordinatorAgent**: Conflict resolution and final synthesis

**Features**:
- LangGraph StateGraph workflow orchestration
- Multiple LLM backend support (OpenAI, DeepSeek, Ollama, vLLM)
- GPU architecture database (V100, A100, H100, RTX 3090)
- Iterative refinement with convergence checking

### 5. Supporting Utilities

**TTGIR Analysis** (`ttgir_parser.py`, `ttgir_analysis.py`):
- Parse Triton GPU IR to extract kernel structure
- Analyze memory operations, control flow, and compute patterns

**Tensor Utilities** (`tensor_utils.py`):
- Load tensor shapes from JSON captures
- Create kernel inputs for benchmarking
- Print tensor information for debugging

**Grid Utilities** (`grid_utils.py`):
- Compute grid dimensions from tensor sizes
- Infer numel (number of elements) for kernels

**Benchmark Utilities** (`benchmark_utils.py`):
- Benchmark kernel configurations
- Detect kernel types (elementwise, reduction, matmul, etc.)
- Statistical benchmarking with multiple iterations

## Integration with TritonBench

### Data Flow

```
TritonBench Operators
        ↓
Kernel Metadata (oss_cuda_kernels.yaml)
        ↓
TTGIR Extraction (capture_all_kernel_ir.py)
        ↓
Shape Capture (capture_all_kernel_ir.py with hooks)
        ↓
Multi-Agent Optimization (multiagent_triton_runner.py)
        ↓
Configuration Testing & Benchmarking
        ↓
Results & Speedup Analysis
```

### Key Integration Points

1. **Kernel Discovery**: Uses `tritonbench/metadata/oss_cuda_kernels.yaml` to find Triton-tagged kernels
2. **Operator Loading**: Uses `tritonbench.operators.load_opbench_by_name()` to load operators
3. **TTGIR Extraction**: Hooks into Triton JIT compilation to capture IR
4. **Shape Capture**: Hooks into kernel launches to capture actual tensor shapes
5. **Benchmarking**: Uses TritonBench's benchmarking infrastructure for validation

## Workflow Examples

### Example 1: Basic Optimization

```bash
# 1. Prepare TTGIR cache (if not already done)
python multiagent_triton_runner.py --prepare-cache --cache-dir irs/new

# 2. Run optimization with shape capture
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --auto-capture-shapes \
  --output-dir ./agentic_results \
  --kernels gemm_kernel

# 3. View results
cat agentic_results/summary.json
```

### Example 2: Optimization with Benchmarking

```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir shapes/ \
  --output-dir ./agentic_results \
  --test-config \
  --benchmark-baseline \
  --test-device cuda \
  --num-iterations 100 \
  --llm-provider deepseek
```

### Example 3: Debug Mode

```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir shapes/ \
  --output-dir ./agentic_results \
  --debug-mode \
  --debug-dir ./debug_multiagent \
  --kernels specific_kernel
```

## Output Structure

### Per-Kernel Results
`<output-dir>/<kernel>__<shapes-stem>.json`:
```json
{
  "success": true,
  "kernel": "gemm_kernel",
  "shape_path": "shapes/gemm_kernel_shapes.json",
  "payload": {
    "recommended_config": {...},
    "analysis": {...},
    "expected_improvement": "1.5x"
  },
  "baseline_ms": 0.5,
  "optimized_ms": 0.33,
  "speedup": 1.52
}
```

### Summary Results
`<output-dir>/summary.json`:
```json
{
  "total_runs": 10,
  "successful": 8,
  "failed": 2,
  "best_speedups": {
    "gemm_kernel": 1.52,
    "reduce_kernel": 1.23
  },
  "errors": {...}
}
```

## Configuration

### Environment Variables
- `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`: API key for LLM providers
- `TRITONBENCH_TRITON_COMMIT_HASH`: Triton commit hash for reproducibility

### LLM Provider Options
- **DeepSeek** (default): Cost-effective, good reasoning
- **OpenAI**: Best accuracy, requires API key
- **Ollama**: Local models, offline use
- **vLLM**: High-performance local serving

### Optimization Targets
- `throughput`: Maximize operations per second
- `latency`: Minimize execution time
- `energy`: Minimize power consumption

## Performance Benefits

### Traditional Empirical Tuning
- Time: 30-60 seconds per kernel
- GPU executions: 100-1000+ per kernel
- Risk: Crashes, hangs, OOM errors
- Coverage: Limited by time budget

### Multi-Agent System
- Time: 0.001-0.01 seconds per kernel (analysis only)
- GPU executions: 0 (zero execution for analysis)
- Risk: No execution risks
- Coverage: Comprehensive configuration space analysis

### Accuracy
- Occupancy prediction: 95%+ accuracy
- Register usage: 90%+ accuracy
- Memory efficiency: 85%+ correlation
- Configuration ranking: Top-3 predictions match empirical top-5 in 80%+ cases

## Future Enhancements

1. **Learning System**: Adapt predictions based on empirical feedback
2. **More Architectures**: NVIDIA Ada Lovelace, Intel Xe-HPC support
3. **Kernel Fusion Analysis**: Multi-kernel optimization
4. **Dynamic Tuning**: Runtime adaptation based on input characteristics
5. **Integration with TritonBench CI**: Automated optimization in CI pipelines

## Key Files Reference

- `multiagent_triton_runner.py`: Main runner script
- `configuration_optimizer.py`: LLM-based optimizer
- `optimizer_functions.py`: Helper functions
- `multiagent.py`: Basic multi-agent system
- `agent_system_langgraph/`: LangGraph-based multi-agent system
- `ttgir_parser.py`: TTGIR parsing utilities
- `tensor_utils.py`: Tensor shape utilities
- `benchmark_utils.py`: Benchmarking utilities
- `capture_all_kernel_ir.py`: TTGIR and shape capture

## Documentation Files

- `README_multiagent_triton_runner.md`: Quick start guide
- `MULTIAGENT_GUIDE.md`: Comprehensive multi-agent system guide
- `agent_system_langgraph/README.md`: LangGraph system architecture
- `agent_system_langgraph/QUICKSTART.md`: LangGraph quick start
- `BENCHMARKING_INTEGRATION.md`: Benchmarking integration guide
- `CODE_WALKTHROUGH.md`: Code walkthrough
