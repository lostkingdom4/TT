# Tensor Dimensions in Multi-Agent Optimization

## Critical Update: Tensor Shape Awareness

The multi-agent optimization system now **requires and uses tensor dimensions** to make informed optimization decisions. This guide explains why tensor dimensions are crucial and how to provide them.

## Why Tensor Dimensions Matter

### 1. **Block Size Selection (WarpAgent)**
```python
# Example: 1D tensor with 262,144 elements
tensor_shapes = {"input": [262144]}

# Without tensor info: Agent might suggest XBLOCK=256
# Problem: 262144 / 256 = 1024 blocks (good)

# But if tensor was [100,000]:
# 100000 / 256 = 390.625 blocks → uneven work distribution!
# Better: XBLOCK=128 → 100000/128 = 781.25 (still not ideal)
# Optimal: XBLOCK=125 → 100000/125 = 800 blocks (even distribution)
```

**Impact**: Proper block sizing based on tensor dimensions improves:
- Memory coalescing efficiency
- Cache line utilization
- Load balancing across CTAs

### 2. **Warp Count Tuning (SMAgent)**
```python
# Example: Small vs large tensors
small_tensor = {"x": [1024]}    # Only 1K elements
large_tensor = {"x": [1048576]} # 1M elements

# Small tensor:
# - num_warps=16 is overkill (each CTA processes tiny chunk)
# - Better: num_warps=2 for higher occupancy

# Large tensor:
# - num_warps=16 is appropriate (more parallelism needed)
# - Helps saturate memory bandwidth
```

**Impact**: Tensor size influences optimal parallelism level:
- Small tensors → fewer warps, higher occupancy
- Large tensors → more warps, better throughput

### 3. **Grid Dimensioning (GPUAgent)**
```python
# Example: 2D tensor [1024, 1024]
tensor_shapes = {"matrix": [1024, 1024]}

# Grid calculation:
# grid_x = ceil(1024 / XBLOCK)
# grid_y = ceil(1024 / YBLOCK)

# If XBLOCK=128, YBLOCK=128:
# grid = (8, 8) → 64 CTAs total

# Need 2-4x more CTAs than SMs for good saturation
# V100 has 80 SMs → target 160-320 CTAs
# Current grid (64) is insufficient!
```

**Impact**: Grid size must match tensor dimensions to:
- Saturate all SMs
- Avoid tail effects
- Balance work distribution

## How to Provide Tensor Dimensions

### Method 1: Direct Specification
```python
from agent_system_langgraph import optimize_kernel

result = optimize_kernel(
    kernel_name="my_kernel",
    ttgir=ttgir_content,
    current_config={
        "num_warps": 4,
        "block_size": {"XBLOCK": 1024},
        # ... other config
    },
    # ✓ IMPORTANT: Add tensor dimensions
    tensor_shapes={
        "arg0": [262144],           # 1D tensor: 256K elements
        "arg1": [1024, 1024],       # 2D tensor: 1M elements
        "arg2": [8, 128, 128],      # 3D tensor: 128K elements
    },
    tensor_dtypes={
        "arg0": "float32",
        "arg1": "float16",
        "arg2": "int32",
    },
)
```

### Method 2: Extract from Metadata
```python
import json
from pathlib import Path

# Read captured IR metadata
metadata_path = Path("captured_ir/triton_cache/kernel_metadata.json")
with open(metadata_path) as f:
    metadata = json.load(f)

# Extract shapes
tensor_shapes = {}
tensor_dtypes = {}
if "input_shapes" in metadata:
    for i, shape in enumerate(metadata["input_shapes"]):
        arg_name = f"arg{i}"
        tensor_shapes[arg_name] = shape
        if "input_dtypes" in metadata:
            tensor_dtypes[arg_name] = metadata["input_dtypes"][i]

# Use in optimization
result = optimize_kernel(
    kernel_name=kernel_name,
    ttgir=ttgir,
    current_config=config,
    tensor_shapes=tensor_shapes,  # From metadata
    tensor_dtypes=tensor_dtypes,
)
```

### Method 3: Parse from PyTorch Model
```python
import torch

# Get shapes from actual model inputs
model_inputs = {
    "x": torch.randn(1024, 1024),
    "y": torch.randn(1024),
}

tensor_shapes = {
    name: list(tensor.shape)
    for name, tensor in model_inputs.items()
}

tensor_dtypes = {
    name: str(tensor.dtype).replace("torch.", "")
    for name, tensor in model_inputs.items()
}
```

## What Agents Do With Tensor Dimensions

### WarpAgent Analysis
- Calculates **bytes per memory transaction** based on dtype
- Estimates **coalescing efficiency** from access patterns
- Suggests block sizes that align with **cache line boundaries** (128 bytes)
- Example:
  ```
  float32 tensor [262144]
  Current XBLOCK=1024 → 1024 * 4 bytes = 4096 bytes per CTA
  L1 cache line = 128 bytes → 32 elements per line
  Suggestion: XBLOCK should be multiple of 32 for alignment
  ```

### SMAgent Analysis
- Computes **work per CTA** = tensor_elements / (grid_size * XBLOCK)
- Estimates **register usage** based on intermediate values
- Balances **occupancy vs parallelism** for tensor size
- Example:
  ```
  Tensor size: 1M elements
  Current: num_warps=4, XBLOCK=256 → 256 threads per CTA
  CTAs needed: 1M / 256 = 3906 CTAs
  If only 80 SMs → 48 CTAs per SM (excellent saturation)
  Recommendation: num_warps=4 is appropriate
  ```

### GPUAgent Analysis
- Calculates **total CTAs** needed: ceil(tensor_elements / XBLOCK)
- Checks **SM saturation**: total_CTAs / num_SMs
- Analyzes **L2 cache reuse** across CTAs processing adjacent data
- Example:
  ```
  2D tensor [1024, 1024], XBLOCK=128, YBLOCK=128
  Grid = (8, 8) = 64 CTAs
  V100 has 80 SMs → only 64 CTAs → undersaturated!
  Recommendation: Reduce XBLOCK to 64 → grid (16, 16) = 256 CTAs
  ```

## Common Mistakes Without Tensor Dimensions

### ❌ Blind Optimization
```python
# Agent suggests num_warps=32 without knowing tensor size
# If tensor is tiny [128], this creates severe underutilization
```

### ❌ Wrong Block Sizes
```python
# Agent suggests XBLOCK=1024 based on "general best practices"
# If tensor is [100,000], this creates uneven distribution
```

### ❌ Poor Grid Sizing
```python
# Agent can't calculate correct grid dimensions
# May create too few CTAs → wasted SMs
# Or too many tiny CTAs → excessive overhead
```

## Expected Improvements With Tensor Dimensions

### Before (No Tensor Info)
```
WarpAgent: "Suggest XBLOCK=256 (general purpose)"
SMAgent: "num_warps=8 for balanced occupancy"
GPUAgent: "Unable to determine optimal grid size"
Confidence: 40%
```

### After (With Tensor Info)
```
WarpAgent: "Tensor [262144] → XBLOCK=512 for even division (512 CTAs)"
SMAgent: "512 CTAs on 80 SMs = 6.4 per SM → num_warps=8 optimal"
GPUAgent: "Grid (512, 1) saturates all SMs, good L2 reuse"
Confidence: 85%
Expected improvement: 35-40%
```

## Integration Examples

### Example 1: Full Pipeline
```python
# 1. Extract PyTorch module (operator_fusion.py)
# 2. Generate Triton kernel (create_dataset.py)
# 3. Capture TTGIR + metadata (capture_all_kernel_ir.py)
# 4. Optimize with tensor dimensions

from agent_system_langgraph import GPUOptimizationOrchestrator

orchestrator = GPUOptimizationOrchestrator(llm_provider="deepseek")

result = orchestrator.optimize_from_ttgir(
    kernel_name="triton_red_fused_add_sum_0",
    ttgir=ttgir_content,
    current_config={
        "num_warps": 4,
        "num_stages": 2,
        "block_size": {"XBLOCK": 1024, "RBLOCK": 256},
    },
    tensor_shapes={
        "input": [1024, 1024],
        "output": [1024],
    },
    tensor_dtypes={
        "input": "float32",
        "output": "float32",
    },
)
```

### Example 2: Batch Processing
```python
from pathlib import Path
import json

ttgir_dir = Path("captured_ir/triton_cache")

for ttgir_file in ttgir_dir.rglob("*.ttgir"):
    metadata_file = ttgir_file.with_suffix('.json')

    # Load metadata with tensor info
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Extract tensor shapes
    tensor_shapes = {
        f"arg{i}": shape
        for i, shape in enumerate(metadata.get("input_shapes", []))
    }

    # Optimize
    result = optimize_kernel(
        kernel_name=ttgir_file.stem,
        ttgir=ttgir_file.read_text(),
        current_config=metadata["config"],
        tensor_shapes=tensor_shapes,
    )

    if result["success"]:
        print(f"✓ {ttgir_file.stem}: {result['result']['expected_improvement']}")
```

## Best Practices

1. **Always provide tensor shapes** when available
2. **Include dtypes** for accurate memory calculations
3. **Use actual input dimensions** from your workload, not synthetic data
4. **Validate recommendations** with empirical benchmarking
5. **Iterate**: Use agent feedback to refine tensor preprocessing

## See Also

- `example_usage.py` - Updated examples with tensor dimensions
- `QUICKSTART.md` - Setup guide
- `README.md` - Full system documentation
