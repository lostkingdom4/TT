# Auto-Tune Baseline

Captures Triton's built-in autotuning results for every TritonBench operator and shape that has captured IR. This provides a brute-force search baseline to compare against the multi-agent LLM optimizer.

## How It Works

1. **Discover** operators from `extract/ir_output/<device>/` (shapes captured during IR extraction)
2. **Run** each operator through TritonBench's benchmark framework in an isolated subprocess
3. **Record** `best_config` + latency for every shape — if the kernel has `@triton.autotune`, Triton sweeps its predefined config grid; otherwise, only the default config latency is recorded

Each operator runs in a separate subprocess to prevent GPU memory leaks and Triton cache cross-contamination.

## Usage

```bash
# Run all operators on V100
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/v100

# Single operator
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/v100 --operator softmax

# Resume after interruption (default behavior)
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/v100 --resume

# Force re-run all
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/v100 --no-resume

# Different device
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/A40

# Export CSV summary after run
python -m auto_tune_baseline.run_all --ir-dir extract/ir_output/v100 --export-csv
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--ir-dir` | required | IR output directory (determines operators & device) |
| `--output-dir` | `auto_tune_baseline/results` | Where to store results |
| `--operator` | all | Run only this operator |
| `--resume` / `--no-resume` | resume | Skip already-completed operators |
| `--warmup` | 25 | Warmup iterations |
| `--rep` | 100 | Benchmark repetitions |
| `--timeout` | 600 | Per-operator timeout (seconds) |
| `--export-csv` | off | Export flat CSV summary |
| `--log-level` | INFO | Logging verbosity |

## Module Structure

| File | Purpose |
|------|---------|
| `discover.py` | Scan IR output dir → list of available operators and shapes |
| `operator_runner.py` | Load a single TritonBench operator, run all shapes, capture results (runs as subprocess) |
| `results.py` | `BaselineResultsStore` — persist per-operator JSON, track completion for resume, export CSV/summary |
| `run_all.py` | CLI entry point — orchestrates discovery → run → store pipeline |

## Results Layout

```
results/<device>/
├── index.json              # Progress tracker (completed/failed operators)
├── summary.json            # Aggregate statistics
├── summary.csv             # Flat CSV (one row per operator+shape)
├── addmm.json              # Per-operator results
├── softmax.json
├── ...
```

### Per-Operator JSON Format

```json
{
  "operator": "addmm",
  "device": "v100",
  "has_autotune": false,
  "num_shapes": 126,
  "timestamp": "2026-02-24T...",
  "status": "completed",
  "shapes": [
    {
      "input_id": [4096, 256],
      "x_val": "(4096, 256)",
      "has_autotune": false,
      "best_config": {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 4, ...},
      "latency_ms": 0.756,
      "triton_impl_name": "triton_addmm",
      "status": "success"
    }
  ]
}
```

`best_config` is non-null only when `@triton.autotune` triggers for that shape. Operators without autotune still get their default-config latency recorded.

## V100 Results Summary

| Operator | Shapes | With Config | Autotune Active |
|----------|--------|-------------|-----------------|
| addmm | 126 | 42 | via `max_autotune` |
| jagged_softmax | 330 | 330 | `@triton.autotune` |
| jagged_sum | 150 | 150 | `@triton.autotune` |
| layer_norm | 120 | 0 | not triggered |
| softmax | 294 | 0 | no decorator |
| sum | 42 | 0 | 1D inputs skip autotune |
| vector_add | 48 | 0 | no decorator |
| low_mem_dropout | 32 | 0 | no decorator |

522 of 1,142 shapes produced a `best_config`, all from 3 operators (addmm, jagged_softmax, jagged_sum). The remaining 5 operators have no autotune on their forward-pass benchmark path.
