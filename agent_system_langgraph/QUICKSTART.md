# Quick Start Guide - LangGraph Multi-Agent GPU Optimization

## Installation (5 minutes)

### 1. Install Dependencies

```bash
cd agent_system_langgraph
pip install -r requirements.txt
```

### 2. Set Up API Keys

Choose one LLM provider:

**Option A: DeepSeek (Recommended - Cost-effective)**
```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```
Get key at: https://platform.deepseek.com/

**Option B: OpenAI (Best accuracy)**
```bash
export OPENAI_API_KEY="your_openai_api_key"
```
Get key at: https://platform.openai.com/

**Option C: Ollama (Local/Offline)**
```bash
# Install Ollama: https://ollama.ai/
ollama pull llama3
export OLLAMA_BASE_URL="http://localhost:11434/v1"
export OLLAMA_MODEL="llama3"
```

**Option D: vLLM (High-Performance Serving)**
```bash
# Install vLLM: pip install vllm
# Start vLLM server with your model:
vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8000

# In another terminal, set environment variables:
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
```

### 3. Verify Installation

```bash
python -c "from agent_system_langgraph import GPUOptimizationOrchestrator; print('✓ Installation successful')"
```

---

## Your First Optimization (10 minutes)

### Example 1: Simple Kernel

Create `test_optimization.py`:

```python
from agent_system_langgraph import optimize_kernel

# Sample Triton kernel IR (simplified)
ttgir = """
module {
  tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    // ... Triton GPU IR ...
  }
}
"""

# Current configuration
config = {
    "num_warps": 4,
    "num_stages": 2,
    "block_size": {"XBLOCK": 1024},
    "grid": {"x": 256},
}

# ✨ NEW: Tensor dimensions are CRITICAL for accurate optimization
tensor_shapes = {
    "arg0": [262144],  # Input tensor 1: 256K elements
    "arg1": [262144],  # Input tensor 2: 256K elements
    "arg2": [262144],  # Output tensor: 256K elements
}

tensor_dtypes = {
    "arg0": "float32",
    "arg1": "float32",
    "arg2": "float32",
}

# Run optimization
result = optimize_kernel(
    kernel_name="add_kernel",
    ttgir=ttgir,
    current_config=config,
    tensor_shapes=tensor_shapes,  # ← IMPORTANT: Include tensor dimensions!
    tensor_dtypes=tensor_dtypes,  # ← Helps calculate memory bandwidth
    architecture="cuda:70",  # Tesla V100
    llm_provider="deepseek",  # or "openai" or "ollama" or "vllm"
)

# Print results
if result["success"]:
    print(f"✓ Expected improvement: {result['result']['expected_improvement']}")
    print(f"✓ Recommended config: {result['result']['recommended_config']}")
else:
    print(f"✗ Error: {result['error']}")
```

Run it:
```bash
python test_optimization.py
```

**Why tensor dimensions matter:**
- Agents use tensor sizes to calculate optimal block sizes
- Without dimensions, recommendations are generic and less accurate
- See `TENSOR_DIMENSIONS_GUIDE.md` for detailed explanation

Expected output:
```
✓ Expected improvement: 20-25%
✓ Recommended config: {'num_warps': 8, 'block_size': {'XBLOCK': 512}, ...}
```

---

## Real Kernel from Cache (15 minutes)

### Example 2: Optimize from TTGIR Cache

```python
from agent_system_langgraph import GPUOptimizationOrchestrator
from pathlib import Path
import json

# Load kernel from triton_cache
ttgir_path = Path("./captured_ir/triton_cache/some_kernel.ttgir")
metadata_path = ttgir_path.with_suffix('.json')

# Read files
with open(ttgir_path) as f:
    ttgir = f.read()

with open(metadata_path) as f:
    metadata = json.load(f)

# Create orchestrator
orchestrator = GPUOptimizationOrchestrator(
    llm_provider="deepseek",
    architecture_code="cuda:70",
)

# Optimize
result = orchestrator.optimize_from_ttgir(
    kernel_name=metadata["name"],
    ttgir=ttgir,
    current_config={
        "num_warps": metadata.get("num_warps", 4),
        "num_stages": metadata.get("num_stages", 2),
        "shared_memory_bytes": metadata.get("shared", 0),
    },
)

# Save result
with open("optimization_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)

print("✓ Results saved to optimization_result.json")
```

---

## Integration with Existing Pipeline (20 minutes)

### Example 3: Full Pipeline with Benchmarking

```python
from pathlib import Path
import json

# Step 1: Parse TTGIR (using existing tools)
from ttgir_config_optimizer import TTGIRParser

ttgir_path = Path("./captured_ir/triton_cache/kernel.ttgir")
ttgir_analysis = TTGIRParser.parse_ttgir_file(ttgir_path)

# Step 2: Get LLM optimization
from agent_system_langgraph import GPUOptimizationOrchestrator

orchestrator = GPUOptimizationOrchestrator(llm_provider="deepseek")

with open(ttgir_path) as f:
    ttgir_content = f.read()

llm_result = orchestrator.optimize_from_ttgir(
    kernel_name=ttgir_analysis.kernel_name,
    ttgir=ttgir_content,
    current_config={
        "num_warps": ttgir_analysis.num_warps,
        "num_stages": 2,
        "shared_memory_bytes": ttgir_analysis.shared_memory_bytes,
    }
)

if not llm_result["success"]:
    print(f"✗ LLM optimization failed: {llm_result['error']}")
    exit(1)

recommended_config = llm_result["result"]["recommended_config"]

# Step 3: Benchmark baseline
from ttgir_config_optimizer import benchmark_baseline_config

baseline = benchmark_baseline_config(
    kernel_name=ttgir_analysis.kernel_name,
    kernel_path=Path("./Data/data_64/triton_modules/kernel.py"),
    input_shapes_json=Path("./captured_ir/input_shapes/kernel_shapes.json"),
    device="cuda",
)

# Step 4: Benchmark optimized config
from ttgir_config_optimizer import test_recommended_config

optimized = test_recommended_config(
    kernel_name=ttgir_analysis.kernel_name,
    kernel_path=Path("./Data/data_64/triton_modules/kernel.py"),
    input_shapes_json=Path("./captured_ir/input_shapes/kernel_shapes.json"),
    config=recommended_config,
    device="cuda",
)

# Step 5: Compare results
baseline_time = baseline["execution_time_ms"]["median"]
optimized_time = optimized["execution_time_ms"]["median"]
actual_speedup = baseline_time / optimized_time

print(f"\n{'='*80}")
print("Optimization Results")
print(f"{'='*80}")
print(f"Baseline time:     {baseline_time:.3f} ms")
print(f"Optimized time:    {optimized_time:.3f} ms")
print(f"Actual speedup:    {actual_speedup:.2f}x")
print(f"Predicted:         {llm_result['result']['expected_improvement']}")
print(f"{'='*80}\n")

# Save comparison
comparison = {
    "kernel": ttgir_analysis.kernel_name,
    "baseline": baseline,
    "optimized": optimized,
    "llm_analysis": llm_result["result"],
    "actual_speedup": actual_speedup,
}

with open("optimization_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2, default=str)

print("✓ Full comparison saved to optimization_comparison.json")
```

---

## Batch Processing (30 minutes)

### Example 4: Optimize All Kernels in Cache

```python
from pathlib import Path
from agent_system_langgraph import GPUOptimizationOrchestrator
import json
from tqdm import tqdm

# Setup
cache_dir = Path("./captured_ir/triton_cache")
output_dir = Path("./optimized_configs")
output_dir.mkdir(exist_ok=True)

orchestrator = GPUOptimizationOrchestrator(
    llm_provider="deepseek",
    architecture_code="cuda:70",
    max_iterations=1,  # Fast mode
)

# Find all TTGIR files
ttgir_files = list(cache_dir.rglob("*.ttgir"))
print(f"Found {len(ttgir_files)} kernels to optimize")

results = []

for ttgir_path in tqdm(ttgir_files):
    metadata_path = ttgir_path.with_suffix('.json')

    if not metadata_path.exists():
        continue

    try:
        # Read files
        with open(ttgir_path) as f:
            ttgir = f.read()

        with open(metadata_path) as f:
            metadata = json.load(f)

        kernel_name = metadata.get("name", ttgir_path.stem)

        # Optimize
        result = orchestrator.optimize_from_ttgir(
            kernel_name=kernel_name,
            ttgir=ttgir,
            current_config={
                "num_warps": metadata.get("num_warps", 4),
                "num_stages": metadata.get("num_stages", 2),
                "shared_memory_bytes": metadata.get("shared", 0),
            }
        )

        if result["success"]:
            # Save individual result
            output_file = output_dir / f"{kernel_name}_optimized.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            results.append({
                "kernel": kernel_name,
                "status": "success",
                "improvement": result["result"]["expected_improvement"],
            })
        else:
            results.append({
                "kernel": kernel_name,
                "status": "failed",
                "error": result["error"],
            })

    except Exception as e:
        print(f"✗ Error processing {ttgir_path.name}: {e}")
        results.append({
            "kernel": ttgir_path.stem,
            "status": "error",
            "error": str(e),
        })

# Save summary
summary = {
    "total": len(results),
    "successful": sum(1 for r in results if r["status"] == "success"),
    "failed": sum(1 for r in results if r["status"] != "success"),
    "results": results,
}

with open(output_dir / "optimization_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Optimized {summary['successful']}/{summary['total']} kernels")
print(f"✓ Results saved to {output_dir}/")
```

---

## Troubleshooting

### Issue: "Import error: No module named 'langgraph'"
**Solution:**
```bash
pip install langgraph langchain langchain-core
```

### Issue: "API key not found"
**Solution:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export DEEPSEEK_API_KEY="your_key"
# OR create .env file in project root
echo "DEEPSEEK_API_KEY=your_key" > .env
```

### Issue: "LLM timeout or rate limit"
**Solution:**
```python
# Use local model instead (Ollama)
orchestrator = GPUOptimizationOrchestrator(
    llm_provider="ollama",
    llm_model="llama3",
)

# OR use vLLM for faster local inference
orchestrator = GPUOptimizationOrchestrator(
    llm_provider="vllm",
    llm_model="meta-llama/Meta-Llama-3-8B-Instruct",
)
```

### Issue: "Invalid JSON response from LLM"
**Solution:**
```python
# Increase temperature for more robust responses
orchestrator = GPUOptimizationOrchestrator(
    temperature=0.3,  # Default is 0.2
)
```

---

## Next Steps

1. **Run examples**: Try `python example_usage.py`
2. **Read docs**: Check `README.md` for detailed API reference
3. **Integration**: See `COMPARISON.md` for integration patterns
4. **Customize**: Extend agents in `agents.py` for custom analysis

---

## Quick Reference

### Import Patterns
```python
# Simple optimization
from agent_system_langgraph import optimize_kernel

# Full control
from agent_system_langgraph import GPUOptimizationOrchestrator, KernelConfig

# Architecture database
from agent_system_langgraph import get_architecture, ARCHITECTURES
```

### Architectures
- `cuda:70` - Tesla V100
- `cuda:80` - A100
- `cuda:86` - RTX 3090
- `cuda:90` - H100

### LLM Providers
- `openai` - GPT-4/GPT-4o (best accuracy)
- `deepseek` - DeepSeek (cost-effective)
- `ollama` - Local models (offline)
- `vllm` - High-performance local serving (fastest)

---

## Getting Help

- **Documentation**: See `README.md`
- **Examples**: See `example_usage.py`
- **Comparison**: See `COMPARISON.md` for original vs LangGraph
- **Issues**: Check error logs in console output
