# multiagent_triton_runner.py

Model-free multi-agent runner for optimizing Triton kernel configurations via TTGIR analysis.

## Overview

Discovers Triton kernels from `tritonbench/metadata/oss_cuda_kernels.yaml`, locates their TTGIR files and input shape captures, then runs either the multi-agent optimizer or single-agent fallback per kernel/shape combination. Per-shape failures are recorded individually and do not halt the pipeline. A summary JSON is emitted at the end with success/failure counts and best speedup per kernel.

## Architecture

### Data flow

1. **Kernel discovery** — Parses `oss_cuda_kernels.yaml` for triton-tagged kernels and builds a kernel-name-to-operator mapping.
2. **TTGIR resolution** — For each kernel, finds `*.ttgir` files under `--cache-dir` via recursive glob (`*{kernel_name}*.ttgir`).
3. **Source extraction** — Reads the TTGIR content and extracts the original Triton source file path using `ConfigurationOptimizer._extract_source_location`.
4. **Shape resolution** — Locates `*_shapes.json` files under `--input-shapes-dir`. Supports three formats:
   - Consolidated: `{op_name}/input_shapes.json`
   - Per-input: `*{kernel}_input*_shapes.json` (used with `--all-shapes`)
   - Default: `*{kernel}_shapes.json` (backward-compatible)
5. **Auto-capture** (optional) — When shapes are missing and `--auto-capture-shapes` is set, the runner:
   - First attempts a targeted capture by running the operator's Triton backend with a patched `JITFunction.run` hook that records tensor shapes, scalar args, kwargs, and grid info.
   - Falls back to running `capture_all_kernel_ir.py` once if the targeted capture fails.
6. **Optimization** — Dispatches to `optimize_kernel_from_cache` (multi-agent) or `analyze_kernel_from_cache` (single-agent).
7. **Benchmarking** (optional) — Tests the recommended config via `test_recommended_config` and computes speedup against baseline.
8. **Output** — Writes per-kernel/shape JSON results and a summary file.

### Key classes

- **`KernelWorkItem`** — Groups a kernel name with its TTGIR path, torch model name, source path, and list of shape files.
- **`RunResult`** — Captures per-shape execution outcome: success/error, optimizer payload, and optional benchmark metrics (baseline_ms, optimized_ms, speedup).

## Inputs

| Input | Flag | Default | Description |
|-------|------|---------|-------------|
| TTGIR cache | `--cache-dir` | `irs/new` | Directory containing `*.ttgir` files (searched recursively) |
| Input shapes | `--input-shapes-dir` | `<output-dir>/generated_shapes` | Directory containing `*_shapes.json` files |
| Kernel metadata | — | `tritonbench/metadata/oss_cuda_kernels.yaml` | Triton-tagged kernel list (auto-discovered) |

### Cache preparation

Pass `--prepare-cache` to run `extract/run.py` before optimization, populating TTGIR/IR files into `--cache-dir`. When combined with `--kernels`, the operator filter is forwarded to the extract runner.

## Usage

### Basic multi-agent run
```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir <shapes_dir> \
  --output-dir ./results
```

### Single-agent fallback
```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir <shapes_dir> \
  --output-dir ./results \
  --single-agent \
  --model deepseek-chat
```

### Filter to specific kernels
```bash
python multiagent_triton_runner.py \
  --cache-dir extract/ir_output/softmax \
  --input-shapes-dir extract/ir_output \
  --kernels Operator.softmax_kernel \
  --output-dir output/fp16_softmax_fwd
```

### Process multiple shape variants with debug logging
```bash
python multiagent_triton_runner.py \
  --cache-dir extract/ir_output/softmax \
  --input-shapes-dir extract/ir_output \
  --kernels Operator.softmax_kernel \
  --output-dir output/fp16_softmax_fwd_3inputs \
  --num-shapes 2 \
  --debug-mode
```

### Process all shape variants
```bash
python multiagent_triton_runner.py \
  --cache-dir extract/ir_output/softmax \
  --input-shapes-dir extract/ir_output \
  --kernels Operator.softmax_kernel \
  --output-dir output/fp16_softmax_fwd_all \
  --all-shapes \
  --debug-mode
```

### Replay cached LLM prompts (no API calls)
```bash
python multiagent_triton_runner.py \
  --cache-dir extract/ir_output/softmax \
  --input-shapes-dir extract/ir_output \
  --kernels Operator.softmax_kernel \
  --output-dir output/fp16_softmax_fwd \
  --use-cached-prompts debug_multiagent/20260203_204657
```

### With benchmarking
```bash
python multiagent_triton_runner.py \
  --cache-dir irs/new \
  --input-shapes-dir <shapes_dir> \
  --output-dir ./results \
  --test-config \
  --benchmark-baseline \
  --test-device cuda \
  --num-iterations 100
```

### Compare configs after optimization

Benchmarks original vs LLM-recommended configs for every kernel/shape pair in an output directory. Each output JSON's `shape_path` is resolved automatically, so `--shapes-file` is optional.

```bash
# Auto-resolve shapes from each output JSON (recommended):
python compare_configs.py \
  --output-dir output/fp16_softmax_fwd \
  --kernel-module ../../tritonbench/operators/softmax/operator.py \
  --kernel-name softmax_kernel \
  --kernel-class Operator \
  --output-json comparison_results.json

# Multi-input run (benchmarks all shape variants):
python compare_configs.py \
  --output-dir output/fp16_softmax_fwd_3inputs \
  --kernel-module ../../tritonbench/operators/softmax/operator.py \
  --kernel-name softmax_kernel \
  --kernel-class Operator \
  --output-json comparison_3inputs.json

# Override with a single shapes file (all pairs use the same shapes):
python compare_configs.py \
  --output-dir output/fp16_softmax_fwd \
  --shapes-file extract/ir_output/softmax/softmax_softmax_kernel_shapes.json \
  --kernel-module ../../tritonbench/operators/softmax/operator.py \
  --kernel-name softmax_kernel \
  --kernel-class Operator \
  --output-json comparison_results.json
```

#### compare_configs.py CLI flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | Path | *required* | Directory containing multiagent output JSONs |
| `--shapes-file` | Path | — | Global shapes file override. When omitted, each output JSON's `shape_path` is resolved relative to the script directory |
| `--kernel-module` | Path | *required* | Path to the Python file containing the Triton kernel |
| `--kernel-name` | str | *required* | Name of the `@triton.jit` kernel function |
| `--kernel-class` | str | — | Class name if the kernel is a class method (e.g. `Operator`) |
| `--device` | str | `cuda` | Device for benchmarking |
| `--warmup` | int | `25` | Warmup iterations for `triton.testing.do_bench` |
| `--rep` | int | `100` | Repetitions for `triton.testing.do_bench` |
| `--seed` | int | `42` | Random seed for input tensor generation |
| `--output-json` | Path | — | Path to write JSON comparison results |

#### compare_configs.py output format

The output JSON contains an envelope with a list of per-pair results:

```json
{
  "kernel": "softmax_kernel",
  "kernel_class": "Operator",
  "device": "cuda",
  "timestamp": "...",
  "num_pairs_total": 5,
  "num_pairs_benchmarked": 4,
  "num_pairs_skipped": 1,
  "results": [
    {
      "tensor_shape": [4096, 256],
      "tensor_dtype": "torch.float16",
      "original_config": { ... },
      "recommended_config": { ... },
      "original_stats": { "median": 0.013, "mean": 0.013, ... },
      "recommended_stats": { "median": 0.018, "mean": 0.018, ... },
      "speedup": 0.72,
      "improvement_pct": -38.5
    }
  ]
}
```

Pairs whose `shape_path` references a missing file are skipped with a warning.

## CLI flags

### Core
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache-dir` | Path | `irs/new` | Directory containing TTGIR files |
| `--input-shapes-dir` | Path | `<output-dir>/generated_shapes` | Directory containing `*_shapes.json` files |
| `--output-dir` | Path | *required* | Directory for per-run result JSON files |
| `--kernels` | str | all | Comma-separated kernel name filter |
| `--all-shapes` | flag | off | Process all matching shapes per kernel instead of first only |
| `--num-shapes` | int | 1 | Number of input shapes to process per kernel when `--all-shapes` is not set |
| `--prepare-cache` | flag | off | Run `extract/run.py` to populate TTGIR/IR before optimizing |

### Shape auto-capture
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--auto-capture-shapes` | flag | off | Capture real shapes via Triton JIT hook when none are found |
| `--capture-num-inputs` | int | 1 | Number of input configs to capture |
| `--capture-start-input-id` | int | 0 | Starting input ID for capture |

### LLM configuration (multi-agent)
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--llm-provider` | str | `deepseek` | Provider: `deepseek`, `openai`, `ollama`, `vllm` |
| `--llm-model` | str | — | Model name override |
| `--temperature` | float | 0.2 | LLM sampling temperature |
| `--max-iterations` | int | 1 | Max refinement iterations |
| `--optimization-target` | str | `throughput` | Target: `throughput`, `latency`, `energy` |

### Single-agent mode
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--single-agent` | flag | off | Use single-agent `ConfigurationOptimizer` instead of multi-agent |
| `--model` | str | `deepseek-chat` | LLM model for single-agent path |
| `--use-cached-response` | flag | off | Use cached LLM response |
| `--cached-response-file` | Path | — | Path to cached response JSON |
| `--prompt-output-dir` | Path | — | Directory to save prompts |

### Debug and replay
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--debug-mode` | flag | off | Enable debug logging; isolates prompts/responses per kernel/shape |
| `--debug-dir` | Path | `./debug_multiagent` | Debug output directory |
| `--use-cached-prompts` | Path | — | Replay LLM responses from a previous debug run directory |

### Benchmarking
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--test-config` | flag | off | Benchmark the recommended config |
| `--benchmark-baseline` | flag | off | Also benchmark the baseline config |
| `--test-device` | str | `cuda` | Device for benchmarking: `cuda`, `xpu`, `cpu` |
| `--num-iterations` | int | 100 | Benchmark iterations |
| `--target-gpu` | str | `cuda:70` | Target GPU arch: `cuda:70` (V100), `cuda:80` (A100), `cuda:86` (RTX3090), `cuda:a40` (A40), `cuda:l40s` (L40S), `cuda:90` (H100) |
| `--triton-modules-dir` | Path | — | Path to Triton modules for source lookup |

### Output
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--summary-file` | Path | `<output-dir>/summary.json` | Summary JSON output path |

## Outputs

### Per-run results
`<output-dir>/<kernel>__<shapes-stem>.json` — one file per kernel/shape combination:
```json
{
  "kernel": "<kernel_name>",
  "shape_path": "<path_to_shapes_json>",
  "success": true,
  "error": null,
  "result": { ... }
}
```

### Summary
`<output-dir>/summary.json` (or `--summary-file`):
```json
{
  "total_runs": 5,
  "successes": 4,
  "failures": 1,
  "best_speedup_per_kernel": {
    "kernel_name": {
      "speedup": 1.23,
      "baseline_time_ms": 1.5,
      "optimized_time_ms": 1.22,
      "shape_path": "..."
    }
  },
  "errors": [
    {
      "kernel": "kernel_name",
      "shape_path": "...",
      "error": "error message"
    }
  ]
}
```

### Captured shapes (auto-capture)
When `--auto-capture-shapes` captures inputs:
- Default: `<shapes_dir>/<op>_<kernel>_shapes.json` (input 0)
- Per-input: `<shapes_dir>/<op>_<kernel>_input<i>_shapes.json`

Each file contains tensor args (shape, dtype, device, numel, stride, contiguity), scalar args, kwargs, and grid info.

## Failure handling

- Missing TTGIR or unextractable source location causes the kernel to be skipped with a warning.
- Missing shapes triggers auto-capture (if enabled) or skips the kernel.
- Per-shape optimization failures are recorded in the result JSON; processing continues to the next shape.
- The exit code is 0 only if all kernel/shape combinations succeed; otherwise 1.
- The summary is written incrementally after each work item completes.
