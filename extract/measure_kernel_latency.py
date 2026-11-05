"""Measure latency of captured inductor kernels by hooking CachingAutotuner
and timing each kernel launch during operator replay.

Usage:
    conda run --no-capture-output -n tb_multi_agent python extract/measure_kernel_latency.py --gpu v100
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from paths import TRITONBENCH_ROOT as REPO_PATH

# Ensure tritonbench is importable
sys.path.insert(0, str(REPO_PATH))

import torch
import torch._inductor.config as inductor_config
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.parser import get_parser

# Import skip list and precision overrides from run_inductor
sys.path.insert(0, str(SCRIPT_DIR))
from run_inductor import _GPU_SKIP, _PRECISION_OVERRIDE, parse_inductor_ops

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}


def measure_op_kernel_latencies(op_name, impl_name, device="cuda", num_inputs=1):
    """Run an operator and measure per-kernel latency via CachingAutotuner hook."""
    kernel_latencies = defaultdict(list)
    original_run = CachingAutotuner.run

    def timed_run(self, *args, grid, stream, benchmark_run=False, **kwargs):
        kname = getattr(getattr(self, "fn", None), "__name__", "unknown")

        # Time the kernel
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_run(self, *args, grid=grid, stream=stream,
                              benchmark_run=benchmark_run, **kwargs)
        end.record()
        torch.cuda.synchronize()
        latency = start.elapsed_time(end)
        kernel_latencies[kname].append(latency)
        return result

    precision = _PRECISION_OVERRIDE.get(op_name, "fp16")
    needs_autotune = "maxautotune" in impl_name.lower()

    parser = get_parser()
    tb_args_list = [
        "--device", device,
        "--precision", precision,
        "--metrics", "latency",
        "--only", impl_name,
        "--num-inputs", str(num_inputs),
    ]
    if needs_autotune:
        tb_args_list.append("--force")
    tb_args, extra_args = parser.parse_known_args(tb_args_list)
    tb_args.warmup = 0
    tb_args.rep = 1

    OpClass = load_opbench_by_name(op_name)
    if OpClass is None:
        return {}

    CachingAutotuner.run = timed_run
    try:
        torch._dynamo.reset()
        with inductor_config.patch(
            max_autotune=needs_autotune,
            max_autotune_gemm_backends="TRITON" if needs_autotune else "",
        ):
            op = OpClass(tb_args=tb_args, extra_args=extra_args)
            op.run(warmup=0, rep=1)
    except Exception as e:
        print(f"    ERROR running {op_name}.{impl_name}: {e}", flush=True)
    finally:
        CachingAutotuner.run = original_run

    return dict(kernel_latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--ops", default=None, help="Comma-separated ops")
    parser.add_argument("--num-inputs", type=int, default=1,
                        help="Number of inputs to measure (default: 1)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    metadata_path = args.metadata or str(
        REPO_PATH / "tritonbench" / "metadata" / "oss_cuda_kernels.yaml"
    )

    inductor_ops = parse_inductor_ops(Path(metadata_path))
    gpu_skip = _GPU_SKIP.get(args.gpu, {})
    inductor_ops = {k: v for k, v in inductor_ops.items() if k not in gpu_skip}

    if args.ops:
        requested = set(args.ops.split(","))
        inductor_ops = {k: v for k, v in inductor_ops.items() if k in requested}

    print(f"Measuring kernel latencies for {len(inductor_ops)} operators...", flush=True)

    all_results = []
    total = 0
    above_threshold = 0

    for op_name, impls in sorted(inductor_ops.items()):
        for impl_name in impls:
            print(f"\n  [{op_name}.{impl_name}]", flush=True)
            latencies = measure_op_kernel_latencies(
                op_name, impl_name,
                num_inputs=args.num_inputs,
            )
            for kname, times in latencies.items():
                median_ms = sorted(times)[len(times) // 2]
                total += 1
                if median_ms > 0.01:
                    above_threshold += 1
                all_results.append({
                    "operator": op_name,
                    "impl": impl_name,
                    "kernel": kname,
                    "latency_ms": round(median_ms, 6),
                    "n_calls": len(times),
                    "min_ms": round(min(times), 6),
                    "max_ms": round(max(times), 6),
                })
                print(f"    {kname}: {median_ms:.4f} ms ({len(times)} calls)", flush=True)

    print(f"\n{'='*60}")
    print(f"Total unique kernels measured: {total}")
    print(f"Kernels with latency > 0.01ms: {above_threshold}")
    print(f"Kernels with latency <= 0.01ms: {total - above_threshold}")

    if all_results:
        all_results.sort(key=lambda x: x["latency_ms"], reverse=True)
        print(f"\nTop 10 slowest:")
        for r in all_results[:10]:
            print(f"  {r['latency_ms']:.4f} ms  {r['operator']}/{r['kernel']}")
        print(f"\nTop 10 fastest:")
        for r in all_results[-10:]:
            print(f"  {r['latency_ms']:.4f} ms  {r['operator']}/{r['kernel']}")

    out_path = args.output or str(SCRIPT_DIR / f"kernel_latency_{args.gpu}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
