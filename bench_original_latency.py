#!/usr/bin/env python3
"""
Benchmark original kernel latency for all kernels in a run-list JSON.

Each kernel is benchmarked in a separate subprocess to isolate CUDA errors.
Matches per-input .py files with their corresponding _input{i}_shapes.json.
Falls back to numel-matching for older extractions without per-input .py files.

Usage:
    conda run -n tb_multi_agent python bench_original_latency.py \
        --run-list run_lists/v100_inductor_above_0.01ms.json \
        --gpu v100 --warmup 5 --rep 20
"""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Worker script executed as a subprocess for each kernel
WORKER_SCRIPT = textwrap.dedent("""\
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "{project_root}")

import torch
import triton

from compare_configs import (
    _benchmark_config,
    _build_kwargs,
    _filter_kwargs_for_kernel,
    _load_kernel,
    _resolve_grid,
    _unwrap_autotuner,
)
from tensor_utils import create_kernel_inputs, load_tensor_shapes


def extract_hardcoded_numels(kernel_py):
    source = Path(kernel_py).read_text()
    numels = {{}}
    for name in ("xnumel", "rnumel", "ynumel", "znumel"):
        m = re.search(rf'^\\s+{{name}}\\s*=\\s*(\\d+)', source, re.MULTILINE)
        if m:
            numels[name] = int(m.group(1))
    return numels


def get_scalar_args(shapes_file):
    d = json.load(open(shapes_file))
    scalars = {{}}
    for s in d.get("scalar_args", []):
        scalars[s["arg_index"]] = s["value"]
    return scalars, d


def find_per_input_pairs(ir_dir, operator, kernel):
    \"\"\"Find all (kernel_py, shapes_file) pairs using per-input .py files.\"\"\"
    op_dir = ir_dir / operator
    pairs = []
    i = 0
    while True:
        py_file = op_dir / f"{{kernel}}_input{{i}}.py"
        shapes_file = op_dir / f"{{operator}}_{{kernel}}_input{{i}}_shapes.json"
        if py_file.exists() and shapes_file.exists():
            pairs.append((py_file, shapes_file, f"input{{i}}"))
            i += 1
        else:
            break
    return pairs


def find_numel_matched_pair(ir_dir, operator, kernel):
    \"\"\"Fallback: match single .py to shapes file by hardcoded numels.\"\"\"
    op_dir = ir_dir / operator

    # Find kernel module
    kernel_py = op_dir / f"{{kernel}}.py"
    if not kernel_py.exists():
        candidates = sorted(op_dir.glob(f"*{{kernel}}*.py"))
        candidates = [c for c in candidates if "_input" not in c.stem]
        if candidates:
            kernel_py = candidates[0]
        else:
            raise FileNotFoundError(f"No kernel module for {{operator}}/{{kernel}}")

    numels = extract_hardcoded_numels(kernel_py)
    input_shapes = sorted(op_dir.glob(f"{{operator}}_{{kernel}}_input*_shapes.json"))

    if numels and input_shapes:
        numel_values = [numels[k] for k in ("xnumel", "rnumel", "ynumel", "znumel") if k in numels]
        for sf in input_shapes:
            scalars, _ = get_scalar_args(sf)
            scalar_vals = [scalars[k] for k in sorted(scalars.keys())]
            if all(v in scalar_vals for v in numel_values):
                return [(kernel_py, sf, sf.stem.split("_input")[-1].replace("_shapes", ""))]

    # Default shapes file
    default_sf = op_dir / f"{{operator}}_{{kernel}}_shapes.json"
    if default_sf.exists():
        return [(kernel_py, default_sf, "default")]
    if input_shapes:
        return [(kernel_py, input_shapes[0], "input0")]
    raise FileNotFoundError(f"No shapes file for {{operator}}/{{kernel}}")


def extract_original_config(metadata):
    kwargs = metadata.get("kwargs", {{}})
    config = {{}}
    block_size = {{}}
    for key, val in kwargs.items():
        if key in ("warmup",):
            continue
        v = val.get("value", val) if isinstance(val, dict) else val
        if key in ("num_warps", "num_stages"):
            config[key] = v
        elif isinstance(v, (int, float)):
            block_size[key] = int(v)
    if block_size:
        config["block_size"] = block_size
    return config


ir_dir = Path("{ir_dir}")
operator = "{operator}"
kernel = "{kernel}"
device = "{device}"
warmup = {warmup}
rep = {rep}
input_index = {input_index}  # -1 = auto (try per-input pairs, then fallback)

try:
    # Find matching (py, shapes) pairs
    pairs = find_per_input_pairs(ir_dir, operator, kernel)
    if not pairs:
        pairs = find_numel_matched_pair(ir_dir, operator, kernel)

    if input_index >= 0:
        # Benchmark specific input
        pairs = [p for p in pairs if p[2] == f"input{{input_index}}"] or pairs[:1]

    results = []
    for kernel_py, shapes_file, label in pairs:
        try:
            metadata = load_tensor_shapes(shapes_file)
            input_args, input_kwargs = create_kernel_inputs(
                shapes_file, device=device, values="random", seed=42,
            )

            kernel_fn = _load_kernel(kernel_py, kernel, None)
            kernel_fn, _ = _unwrap_autotuner(kernel_fn)

            config = extract_original_config(metadata)
            shapes_kwargs = metadata.get("kwargs", {{}})
            orig_kwargs = _build_kwargs(config, input_kwargs, shapes_kwargs, skip_block_sizes=True)
            orig_kwargs = _filter_kwargs_for_kernel(kernel_fn, orig_kwargs)
            grid = _resolve_grid(metadata, config, orig_kwargs)

            stats = _benchmark_config(kernel_fn, grid, input_args, orig_kwargs, warmup, rep, "original")
            stats["input_label"] = label
            stats["shapes_file"] = str(shapes_file.name)
            stats["kernel_py"] = str(kernel_py.name)
            results.append(stats)
        except Exception as e:
            results.append({{"error": str(e), "input_label": label, "shapes_file": str(shapes_file.name), "kernel_py": str(kernel_py.name)}})
        finally:
            torch.cuda.empty_cache()

    print(json.dumps(results))
except Exception as e:
    print(json.dumps([{{"error": str(e)}}]))
""")


def bench_kernel_subprocess(ir_dir, operator, kernel, device, warmup, rep, timeout=120, input_index=-1):
    """Run benchmark for one kernel in an isolated subprocess."""
    script = WORKER_SCRIPT.format(
        project_root=str(PROJECT_ROOT),
        ir_dir=str(ir_dir),
        operator=operator,
        kernel=kernel,
        device=device,
        warmup=warmup,
        rep=rep,
        input_index=input_index,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        # Find the last line that looks like a JSON array
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("["):
                return json.loads(line)
        if result.returncode != 0:
            err = result.stderr.strip().split("\n")[-1][:80] if result.stderr else "unknown error"
            return [{"error": err}]
        return [{"error": "no output from worker"}]
    except subprocess.TimeoutExpired:
        return [{"error": "timeout"}]
    except Exception as e:
        return [{"error": str(e)}]


def main():
    parser = argparse.ArgumentParser(description="Benchmark original kernel latency")
    parser.add_argument("--run-list", required=True, help="Path to run-list JSON")
    parser.add_argument("--gpu", default="v100", help="GPU directory name")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=20, help="Benchmark repetitions")
    parser.add_argument("--device", default="cuda", help="Device string")
    parser.add_argument("--timeout", type=int, default=120, help="Per-kernel timeout (s)")
    parser.add_argument("--input-index", type=int, default=0,
                        help="Which input to benchmark (-1=all, 0=first, etc)")
    args = parser.parse_args()

    ir_dir = PROJECT_ROOT / "extract" / "ir_output" / args.gpu

    with open(args.run_list) as f:
        run_list = json.load(f)
    kernels = run_list["kernels"]

    print(f"\nBenchmarking {len(kernels)} kernels (warmup={args.warmup}, rep={args.rep}, input={args.input_index})")
    print(f"IR directory: {ir_dir}")
    print(f"{'='*145}")
    print(f"{'#':<4} {'Operator':<26} {'Kernel':<45} {'Input':<8} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>8}  {'Kernel .py'}")
    print(f"{'-'*145}")

    results = []
    for i, entry in enumerate(kernels, 1):
        operator = entry["operator"]
        kernel = entry["kernel"]
        kd = kernel[:43] + ".." if len(kernel) > 45 else kernel

        stats_list = bench_kernel_subprocess(
            ir_dir, operator, kernel, args.device,
            args.warmup, args.rep, args.timeout, args.input_index,
        )

        for stats in stats_list:
            label = stats.get("input_label", "?")
            py_name = stats.get("kernel_py", "?")
            if "error" in stats:
                print(f"{i:<4} {operator:<26} {kd:<45} {label:<8} {'ERROR':>10}  {stats['error'][:55]}")
                results.append({"operator": operator, "kernel": kernel,
                                "input_label": label, "error": stats["error"]})
            else:
                m = stats["median"]
                print(f"{i:<4} {operator:<26} {kd:<45} {label:<8} {m:>9.4f}ms {stats['mean']:>9.4f}ms {stats['min']:>9.4f}ms {stats['max']:>9.4f}ms {stats['std_dev']:>7.4f}  {py_name}")
                results.append({
                    "operator": operator, "kernel": kernel,
                    "input_label": label,
                    "median_ms": m, "mean_ms": stats["mean"],
                    "min_ms": stats["min"], "max_ms": stats["max"],
                    "std_dev_ms": stats["std_dev"], "samples": stats["samples"],
                    "shapes_file": stats.get("shapes_file"),
                    "kernel_py": py_name,
                })

    print(f"{'='*145}")

    ok = [r for r in results if "median_ms" in r]
    fail = [r for r in results if "error" in r]
    print(f"\nDone: {len(ok)} benchmarked, {len(fail)} failed/skipped")
    if ok:
        avg = sum(r["median_ms"] for r in ok) / len(ok)
        print(f"Average median latency: {avg:.4f} ms")
    if fail:
        print(f"\nFailed kernels:")
        for r in fail:
            print(f"  {r['operator']}/{r['kernel']} ({r.get('input_label','?')}): {r['error'][:70]}")

    out = args.output or f"extract/kernel_bench_latency_{args.gpu}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
