# Multi-Agent System Debugging Guide

## Overview

The multi-agent optimization system now includes comprehensive debugging capabilities that allow you to observe all prompts sent to LLMs and their responses.

## Quick Start: Enable Debug Mode

### Command Line

```bash
# Enable debug mode with default directory (./debug_multiagent/)
python ttgir_config_optimizer_main.py \
  --kernel-name triton_red_fused__softmax_1 \
  --torch-model-name A2Block_triton \
  --use-multiagent \
  --debug-mode \
  --target-gpu cuda:70

# Specify custom debug directory
python ttgir_config_optimizer_main.py \
  --kernel-name triton_poi_fused_add_2 \
  --torch-model-name MyModel_triton \
  --use-multiagent \
  --debug-mode \
  --debug-dir ./my_debug_output/ \
  --llm-provider deepseek
```

### Python API

```python
from agent_system_langgraph import optimize_kernel

result = optimize_kernel(
    kernel_name="my_kernel",
    ttgir=ttgir_content,
    current_config=config,
    tensor_shapes=tensor_shapes,
    architecture="cuda:70",
    llm_provider="deepseek",
    debug_mode=True,  # ← Enable debugging
    debug_dir="./debug_output/",  # ← Optional: custom directory
)
```

## What Gets Logged

### 1. **Console Output**

When debug mode is enabled, you'll see detailed logs in the console:

```
================================================================================
LLM Call #1 (deepseek/deepseek-chat)
System Prompt (2458 chars):
You are a GPU warp-level optimization expert specializing in NVIDIA Tesla V100 architecture.

Your role is to analyze memory access patterns at the 32-thread warp granularity...

User Prompt (3842 chars):
Analyze the following Triton kernel for warp-level optimizations:

**Kernel**: triton_red_fused__softmax_1
**Type**: reduction
**Current Configuration**:
- num_warps: 4
...
================================================================================

================================================================================
LLM Response #1
Response (2156 chars):
{
  "coalescing_efficiency": {
    "estimated_percentage": 75.0,
    "bottlenecks": ["Non-contiguous access in reduction"],
    ...
  }
}
================================================================================
```

### 2. **Debug Files**

All prompts and responses are saved to timestamped files in the debug directory:

```
debug_multiagent/
├── 20251216_143025_call001_prompt.txt      # WarpAgent prompt
├── 20251216_143025_call001_response.txt    # WarpAgent response
├── 20251216_143025_call001_response.json   # Parsed JSON (if valid)
├── 20251216_143028_call002_prompt.txt      # SMAgent prompt
├── 20251216_143028_call002_response.txt    # SMAgent response
├── 20251216_143028_call002_response.json
├── 20251216_143031_call003_prompt.txt      # GPUAgent prompt
├── 20251216_143031_call003_response.txt    # GPUAgent response
├── 20251216_143031_call003_response.json
├── 20251216_143034_call004_prompt.txt      # CoordinatorAgent prompt
├── 20251216_143034_call004_response.txt    # CoordinatorAgent response
└── 20251216_143034_call004_response.json
```

## File Format

### Prompt Files (`*_prompt.txt`)

```
================================================================================
LLM Call #1
Provider: deepseek
Model: deepseek-chat
Temperature: 0.2
Timestamp: 20251216_143025
================================================================================

SYSTEM PROMPT:
--------------------------------------------------------------------------------
You are a GPU warp-level optimization expert specializing in NVIDIA Tesla V100...
[Full system prompt here]

USER PROMPT:
--------------------------------------------------------------------------------
Analyze the following Triton kernel for warp-level optimizations:
[Full user prompt here with TTGIR, tensor shapes, etc.]
```

### Response Files (`*_response.txt`)

```
================================================================================
LLM Response #1
Provider: deepseek
Model: deepseek-chat
Timestamp: 20251216_143025
Tokens - Prompt: 1234, Completion: 567, Total: 1801
================================================================================

RESPONSE CONTENT:
--------------------------------------------------------------------------------
{
  "coalescing_efficiency": {
    "estimated_percentage": 75.0,
    ...
  }
}
```

### JSON Files (`*_response.json`)

Pretty-printed JSON for easy inspection:

```json
{
  "coalescing_efficiency": {
    "estimated_percentage": 75.0,
    "bottlenecks": [
      "Non-contiguous access in reduction"
    ],
    "access_pattern": "Sequential loads from global memory with stride patterns",
    "tensor_dimension_impact": "RBLOCK=512 aligns with reduction dimension"
  },
  "bank_conflicts": {
    "severity": "low",
    "estimated_conflicts_per_warp": 2,
    "conflict_sources": [
      "Shared memory reduction accumulation"
    ]
  },
  ...
}
```

## Typical Agent Workflow

For a kernel optimization, you'll see this sequence:

1. **Call #1: WarpAgent** - Analyzes warp-level memory coalescing
2. **Call #2: SMAgent** - Analyzes SM resource utilization
3. **Call #3: GPUAgent** - Analyzes GPU-level parallelism
4. **Call #4: CoordinatorAgent** - Synthesizes all analyses

Each call's prompt and response are saved separately for debugging.

## Debugging Scenarios

### Scenario 1: Agent Recommends Wrong Block Size

**Check:**
1. Open the WarpAgent prompt file (`call001_prompt.txt`)
2. Verify tensor shapes are correct in the prompt
3. Check if the agent's response (`call001_response.json`) shows understanding of tensor dimensions
4. Look for the `tensor_dimension_impact` field in the response

### Scenario 2: Low Confidence Score

**Check:**
1. Open CoordinatorAgent files (`call004_*`)
2. Look for conflicts between agent recommendations
3. Check the `confidence` field in each agent's proposals
4. Review coordinator's reasoning in the response

### Scenario 3: JSON Parsing Errors

**Check:**
1. Open the response `.txt` file (not `.json`)
2. Look for malformed JSON or markdown code blocks
3. Check console warnings about JSON extraction
4. The system attempts to auto-fix common issues

### Scenario 4: Unexpected Recommendations

**Check:**
1. Review all prompts to ensure correct information is passed
2. Verify tensor shapes in prompt match your expectations
3. Check if TTGIR is truncated (only first 2000 chars sent)
4. Compare agent analyses to see if they agree or conflict

## Advanced Debugging

### Compare Multiple Runs

```bash
# Run 1: With tensor shapes
python ttgir_config_optimizer_main.py \
  --kernel-name my_kernel \
  --torch-model-name my_model \
  --use-multiagent \
  --debug-mode \
  --debug-dir ./debug_with_shapes/

# Run 2: Without tensor shapes (to see the difference)
# Temporarily remove input_shapes_json to compare
```

Then diff the prompt files to see what changed.

### Analyze Token Usage

```bash
# Extract token counts from all response files
grep "Tokens -" debug_multiagent/*.txt

# Example output:
# Tokens - Prompt: 1234, Completion: 567, Total: 1801
# Tokens - Prompt: 1456, Completion: 623, Total: 2079
# Tokens - Prompt: 1389, Completion: 701, Total: 2090
# Tokens - Prompt: 2134, Completion: 892, Total: 3026
```

### Search for Specific Patterns

```bash
# Find all mentions of "occupancy" in responses
grep -i "occupancy" debug_multiagent/*_response.txt

# Find all confidence scores
grep -i "confidence" debug_multiagent/*_response.json

# Check which agent suggested reducing num_warps
grep -i "num_warps.*reduce\|num_warps.*decrease" debug_multiagent/*.txt
```

## Logging Configuration

Set log level for more/less detail:

```python
import logging

# Set to DEBUG for maximum verbosity
logging.basicConfig(level=logging.DEBUG)

# Or set per-module
logging.getLogger('agent_system_langgraph').setLevel(logging.DEBUG)
logging.getLogger('agent_system_langgraph.llm_provider').setLevel(logging.INFO)
```

## Performance Impact

Debug mode adds minimal overhead:
- **File I/O**: ~10-20ms per LLM call to write files
- **Console logging**: ~5-10ms per call
- **Total impact**: <5% of overall runtime (LLM calls dominate)

It's safe to use in production if disk space allows.

## Troubleshooting

### Debug files not created

**Check:**
1. Directory permissions: `ls -ld ./debug_multiagent/`
2. Disk space: `df -h`
3. Look for warnings in console output

### Files are empty or incomplete

**Check:**
1. Python process has write permissions
2. No exceptions during file writing (check console)
3. Files may be written during execution, not all at once

### Too many debug files

**Clean up:**
```bash
# Remove old debug files
rm -rf ./debug_multiagent/*

# Or keep only recent files (last hour)
find ./debug_multiagent -type f -mmin +60 -delete
```

## Best Practices

1. **Use descriptive debug directories** for different experiments:
   ```bash
   --debug-dir ./debug_softmax_v100/
   --debug-dir ./debug_conv_a100/
   ```

2. **Save debug output with results**:
   ```bash
   # Tarball debug directory for later analysis
   tar -czf debug_$(date +%Y%m%d_%H%M%S).tar.gz debug_multiagent/
   ```

3. **Compare before/after** when tuning prompts or system parameters

4. **Share debug files** when reporting issues - they contain complete context

## Example: Full Debugging Session

```bash
# 1. Run with debug mode
python ttgir_config_optimizer_main.py \
  --kernel-name triton_red_fused__softmax_1 \
  --torch-model-name A2Block_triton \
  --use-multiagent \
  --debug-mode \
  --debug-dir ./debug_softmax/ \
  --llm-provider deepseek \
  --output result.json

# 2. Check what happened
ls -lh ./debug_softmax/

# 3. Review agent interactions
cat ./debug_softmax/*call001_prompt.txt  # WarpAgent input
cat ./debug_softmax/*call001_response.json  # WarpAgent output
cat ./debug_softmax/*call004_response.json  # Final coordinator decision

# 4. Look for issues
grep -i "error\|warning\|failed" ./debug_softmax/*.txt

# 5. Archive for later
tar -czf debug_softmax_$(date +%Y%m%d).tar.gz ./debug_softmax/
```

## See Also

- `TENSOR_DIMENSIONS_GUIDE.md` - Understanding tensor shape impact
- `QUICKSTART.md` - Basic usage without debugging
- `README.md` - Full system documentation
