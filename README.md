# Multi-Agent GPU Kernel Optimization Framework

How to install, reproduce the paper's V100 numbers, and run the framework on
your own GPU.

- [Setup](#setup)
- [Reproducing the Paper Results](#reproducing-the-paper-results)
- [Usage Reference](#usage-reference)
- [Tests](#tests)

---

## Setup

### 1. Clone and create the conda environment

```bash
git clone <repo-url> PyTorch_Intel_GPU_MLIR-Extraction
cd PyTorch_Intel_GPU_MLIR-Extraction
conda env create -f environment.yml
conda activate tb_multi_agent
```

Stack: Python 3.12, PyTorch 2.6.0+cu124, Triton 3.2.0, LangGraph,
Optuna, OpenAI/DeepSeek SDK clients.

### 2. Set up tritonbench

Tritonbench lives under a git-ignored `tb/` directory.

```bash
# Clone from upstream (default)
python setup_tb.py

# Or symlink an existing checkout
python setup_tb.py --link /path/to/existing/tritonbench

# Clear the Triton JIT cache (needed after re-cloning tritonbench, since
# cached IR embeds absolute source paths from compile time)
python setup_tb.py --clear-triton-cache
```

The script clones `tb/`, initializes the `generative-recommenders`
submodule (used by addmm), and runs `pip install -e ./tb`.

To point at a non-standard tritonbench checkout:

```bash
export TRITONBENCH_ROOT=/path/to/my/tritonbench
```

### 3. LLM provider

```bash
export OPENAI_API_KEY=<your-key>      # OpenAI
export DEEPSEEK_API_KEY=<your-key>    # DeepSeek (used in the paper)
ollama serve                          # local Ollama
export VLLM_MODEL=<model-name>        # local vLLM
```

---

## Reproducing the Paper Results

End-to-end recipe for the V100 numbers (Table 2 + Figures). Steps 2 and
3 are the GPU-heavy phases; step 4 is post-processing. Run from the repo
root with `tb_multi_agent` activated.

The canonical kernel set is `run_lists/v100_threshold_include_no_jagged_ln.json`
(19 kernels, 401 shapes). The benchmark harness throughout is
`compare_configs` (extracted kernel `.py` + per-input `*_shapes.json` +
`do_bench` with L2 flush) — this is what the paper's "Full" column
measures.

### Step 1 — Extract kernel IR and per-input shapes

```bash
python extract/run.py --gpu v100 --clear-cache
```

Writes `extract/ir_output/v100/<operator>/<kernel>_input{N}.py` plus
`<kernel>_input{N}_shapes.json` for every TritonBench operator tagged in
`oss_cuda_kernels.yaml`. For Inductor template kernels (`triton_mm`,
`triton_tem_fused_addmm_0`, `triton_bmm_*`) one `.py` is captured per
input shape — required because those templates bake `M`/`N`/`K` and
strides as integer literals.

For Inductor-only operators (those tagged `pt2:` in
`oss_cuda_kernels.yaml`):

```bash
python extract/run_inductor.py --gpu v100 --num-inputs 25
```

### Step 2 — Run the baselines

Each command is independent and writes under
`baselines/results/v100/<baseline>/`. Run whichever baselines you need.

```bash
# Random (one config per shape)            -> baselines/results/v100/random/
python -m baselines.random.run_all --ir-dir extract/ir_output/v100

# Triton @autotune (decorator capture)     -> baselines/results/v100/autotune/
python -m baselines.autotune.run_all --ir-dir extract/ir_output/v100

# Bayesian Optimization (Optuna TPE)
#   fixed=9 evals  -> baselines/results/v100/bo/
#   scaled budget  -> baselines/results/v100/bo_scaled/
python -m baselines.bo.run_all --ir-dir extract/ir_output/v100
python -m baselines.bo.run_all --ir-dir extract/ir_output/v100 --budget-policy scaled

# Random-search at scaled budget           -> baselines/results/v100/random_search/
python -m baselines.random_search.run_all --ir-dir extract/ir_output/v100

# AUC-Bandit (DE + greedy + uniform)       -> baselines/results/v100/auc_bandit/
python -m baselines.auc_bandit.run_all --ir-dir extract/ir_output/v100 \
    --per-shape-isolation
```

`--per-shape-isolation` (auc_bandit, bo) spawns one subprocess per
(kernel, shape) so a poisoned CUDA context from one config cannot
contaminate later shapes.

**Cache-replay (no-GPU mode).** Once `autotune/` is populated, you can
replay BO / random-search / AUC-Bandit in-process from its cache:

```bash
python -m baselines.run_cached \
    --cache-dir baselines/results/v100/autotune \
    --strategy random_search bo auc_bandit \
    --seeds 0 1 2 3 4 \
    --output-dir baselines/results/v100
```

Finishes in ~20 s vs ~2 GPU-days, but covers only the 15 kernels whose
autotune cache spans the full common search space (the four
GEMM/jagged_softmax kernels with decorator-only configs still need a
GPU run).

### Step 3 — Run the multi-agent system

```bash
export DEEPSEEK_API_KEY=<your-key>      # or OPENAI_API_KEY

python batch_runner.py \
    --gpu v100 \
    --target-gpu-arch cuda:70 \
    --output-dir output/batch_v100_paper \
    --run-list run_lists/v100_threshold_include_no_jagged_ln.json \
    --all-shapes \
    --num-candidates 3 \
    --max-iterations 3 \
    --include-agent-proposals \
    --parallel-kernels 19 \
    --llm-provider deepseek
```

Each kernel produces a per-shape JSON under
`output/batch_v100_paper/<operator>/<kernel>/`. With
`--parallel-kernels 19` the full 19-kernel run takes ~6 min on a single
V100 (LLM calls overlap; GPU benchmarks serialize via a file lock).

For other devices, swap `--gpu`, `--target-gpu-arch`, and the step-1
extract:

```bash
# A100
python extract/run.py --gpu a100 --clear-cache
python batch_runner.py --gpu a100 --target-gpu-arch cuda:80 \
    --output-dir output/batch_a100_paper \
    --run-list run_lists/v100_threshold_include_no_jagged_ln.json \
    --all-shapes --num-candidates 3 --max-iterations 3 \
    --include-agent-proposals --parallel-kernels 19

# L40S
python extract/run.py --gpu l40s --clear-cache
python batch_runner.py --gpu l40s --target-gpu-arch cuda:l40s \
    --output-dir output/batch_l40s_paper \
    --run-list run_lists/v100_threshold_include_no_jagged_ln.json \
    --all-shapes --num-candidates 3 --max-iterations 3 \
    --include-agent-proposals --parallel-kernels 19
```

### Step 4 — Aggregate results and generate figures

```bash
python -m baselines.analyze_comparison \
    baselines/results/v100 \
    --output-dir baselines/results/v100/comparison \
    --llm-dir output/batch_v100_paper
```

Produces per-kernel convergence plots, the cross-kernel
normalized-speedup heatmap, and `summary.json` with per-kernel-first
geomean aggregates. Additional analysis scripts live under `Figures/`:

```bash
python Figures/analyze_phase_distribution.py --per-kernel
```

### Reproduction notes

- **Geomean convention**: cross-kernel aggregates are computed as
  per-kernel geomean first, then geomean across kernels — never flat
  per-shape. This avoids over-weighting kernels with many shapes.
- **Determinism**: GPU latency has run-to-run jitter at the microsecond
  scale. Per-kernel speedups are stable to ~1% across seeds; aggregate
  numbers are stable to <0.1x.
- **Budget**: the multi-agent system uses a fixed 9-eval budget per
  shape (3 candidates × 3 iterations). Scaled-budget baselines use
  `clip(round(0.10 * |search_space|), 9, 150)`.
- **Run-list**: `v100_threshold_include_no_jagged_ln.json` is the
  19-kernel set above the 0.006 ms latency threshold, with
  `jagged_layer_norm` excluded (reported separately).

---

## Usage Reference

### Extract kernels (`extract/run.py`)

```bash
# All triton-tagged operators
python extract/run.py --gpu v100

# Specific operators
python extract/run.py --gpu v100 --ops addmm,gemm

# Limit to first N input configurations
python extract/run.py --gpu v100 --num-inputs 3

# Clear the Triton JIT cache before extraction
python extract/run.py --gpu v100 --clear-cache
```

Output layout: `extract/ir_output/<gpu>/<operator>/`
- `*.ttgir` — Triton GPU IR (one per kernel variant × input)
- `*_shapes.json` — captured tensor shapes, scalar args, kwargs, grid
- `*_input{i}.py` — per-shape kernel source (Inductor templates only)

### Multi-agent batch runner (`batch_runner.py`)

```bash
# Recommended: run-list filter + parallel kernels + agent-proposal injection
python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
    --parallel-kernels 19 \
    --num-shapes 3 \
    --include-agent-proposals \
    --run-list run_lists/v100_improved_only.json
```

`--include-agent-proposals` adds each agent's preferred config as a
benchmark candidate alongside the orchestrator's synthesis.

**Available run-lists** (`run_lists/`):

| File | Description | Kernels |
|------|-------------|---------|
| `v100_improved_only.json` | Showed speedup in prior runs | 19 |
| `v100_threshold_include.json` | Above 0.006 ms latency | 23 |
| `v100_threshold_include_no_jagged_ln.json` | Above threshold, ex jagged_layer_norm | 19 |
| `v100_kernelbench_L1.json` / `L2.json` / `L3.json` | KernelBench tiers | varies |

**Parallel execution** (19 kernels, V100):

| Workers | Wall-clock | vs Sequential |
|---------|-----------|---------------|
| 1 | 21.8 min | 1.0x |
| 4 | 9.8 min | 2.2x |
| 19 | 5.7 min | 3.8x |

The bottleneck at high parallelism is GPU benchmark serialization
(one benchmark at a time via `gpu_lock.py`).

**Other useful flags:**

```bash
# Preview without running
python batch_runner.py --gpu v100 --output-dir output/x --dry-run

# Specific operators
python batch_runner.py --gpu v100 --output-dir output/x \
    --operators softmax addmm layer_norm

# Cap Inductor variant explosion (e.g., jagged_mean has 1500+)
python batch_runner.py --gpu v100 --output-dir output/x \
    --max-kernels-per-op 5

# Force re-run (resume is the default)
python batch_runner.py --gpu v100 --output-dir output/x --no-resume

# Different LLM provider / candidate count
python batch_runner.py --gpu v100 --output-dir output/x \
    --llm-provider openai --num-candidates 5 --max-iterations 2
```

Progress is tracked in `<output-dir>/batch_manifest.json`. Watch:

```bash
watch -n5 'python -c "
import json
m = json.load(open(\"output/batch_v100/batch_manifest.json\"))
s = m[\"summary\"]
print(f\"Total: {s[\"total_kernels\"]} | Success: {s[\"success\"]} | Failed: {s[\"failed\"]} | Running: {s[\"running\"]} | Pending: {s[\"pending\"]}\")"'
```

When `--num-shapes N` is used, kernels with some failing shapes are
still marked successful — failed shapes are recorded under
`per_shape_results` and the speedup is computed from the successful
shapes only.

### Per-kernel manual run (`multiagent_triton_runner.py`)

For debugging or re-running individual kernels:

```bash
python multiagent_triton_runner.py \
    --cache-dir extract/ir_output/v100/addmm \
    --input-shapes-dir extract/ir_output/v100/addmm \
    --kernels _addmm_fwd \
    --output-dir output/fp16_addmm_fwd \
    --all-shape \
    --debug-mode
```

For Inductor template kernels with per-shape sources, point
`--cache-dir` at the kernel's leaf directory (e.g.
`extract/ir_output/v100/jagged_mean/fp32_jagged_mean_fwd/triton_jagged_mean_simple_fused`).

### Standalone benchmark comparison (`compare_configs.py`)

Compare two configs (or a config against the captured baseline) without
the LLM loop:

```bash
python compare_configs.py \
    --kernel-py extract/ir_output/v100/addmm/_addmm_fwd_input0.py \
    --shapes extract/ir_output/v100/addmm/_addmm_fwd_input0_shapes.json \
    --recommended '{"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3}'
```

### Cache replay (no API calls)

Replay the multi-agent system's prior LLM responses against a fresh
benchmark — useful for paper reruns after harness changes:

```bash
python multiagent_triton_runner.py \
    --cache-dir extract/ir_output/v100/addmm \
    --input-shapes-dir extract/ir_output \
    --output-dir output/replay \
    --use-cached-prompts debug_multiagent/<run-id>
```

### Analysis and figures (`Figures/`)

```bash
# Phase 1 / 2 / 3 attribution of the chosen config
python Figures/analyze_phase_distribution.py --per-kernel
```

Outputs (CSV, PNG) are written next to the script. `Figures/` is
git-ignored — write new analysis there.

---

## Tests

```bash
pytest tests/                          # full suite
pytest tests/test_approach7.py -v      # current architecture
pytest tests/test_benchmark_loop.py    # closed-loop benchmarking
pytest tests/test_extract_inductor.py  # Inductor extraction
```
