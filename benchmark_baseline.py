#!/usr/bin/env python3
"""
Benchmark baseline configurations using TritonBench run.py

This script orchestrates run.py to benchmark operators with their default
configurations and exports results to CSV for comparison with multiagent
recommendations.

Usage:
    # Benchmark all kernels for an operator
    python benchmark_baseline.py --op addmm --output baseline_results.csv

    # Benchmark specific kernel
    python benchmark_baseline.py --op addmm --only _addmm_fwd --output baseline_addmm_fwd.csv

    # Specify device
    python benchmark_baseline.py --op gemm --device cuda --output baseline_gemm.csv
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


def run_benchmark(
    operator: str,
    output_csv: Path,
    only_kernels: Optional[str] = None,
    device: str = "cuda",
    num_inputs: Optional[int] = None,
    rep: Optional[int] = None,
) -> bool:
    """
    Run TritonBench benchmark using run.py and export to CSV.

    Args:
        operator: Operator name (e.g., "addmm", "gemm")
        output_csv: Path to output CSV file
        only_kernels: Comma-separated list of kernel names to benchmark
        device: Device to run on ("cuda", "xpu", "cpu")
        num_inputs: Number of input shapes to test
        rep: Number of repetitions per input

    Returns:
        True if benchmark succeeded, False otherwise
    """
    # Find run.py in the tritonbench root
    from paths import TRITONBENCH_RUN_PY
    run_py = TRITONBENCH_RUN_PY

    if not run_py.exists():
        print(f"Error: run.py not found at {run_py}")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(run_py),
        "--op", operator,
        "--device", device,
        "--csv",
        "--output", str(output_csv),
    ]

    if only_kernels:
        cmd.extend(["--only", only_kernels])

    if num_inputs is not None:
        cmd.extend(["--num-inputs", str(num_inputs)])

    if rep is not None:
        cmd.extend(["--rep", str(rep)])

    print(f"Running benchmark command:")
    print(f"  {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False


def parse_csv_results(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Parse CSV results from run.py.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of dictionaries with benchmark results
    """
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return []

    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(dict(row))

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of benchmark results."""
    if not results:
        print("No results to summarize")
        return

    print(f"\n{'='*80}")
    print("Benchmark Summary")
    print(f"{'='*80}")
    print(f"Total kernels benchmarked: {len(results)}")
    print()

    # Get all metric columns (exclude metadata columns)
    metadata_cols = {'op_name', 'kernel_name', 'device', 'precision', 'input_id'}
    if results:
        metric_cols = [k for k in results[0].keys() if k not in metadata_cols]
    else:
        metric_cols = []

    for result in results:
        kernel_name = result.get('kernel_name', 'unknown')
        print(f"Kernel: {kernel_name}")
        for col in metric_cols:
            value = result.get(col, 'N/A')
            print(f"  {col}: {value}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark baseline configurations using TritonBench run.py"
    )
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator name (e.g., addmm, gemm, softmax)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of kernel names to benchmark"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "xpu", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=None,
        help="Number of input shapes to test"
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=None,
        help="Number of repetitions per input"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking operator: {args.op}")
    print(f"Output CSV: {output_path}")
    if args.only:
        print(f"Filtering kernels: {args.only}")
    print()

    # Run benchmark
    success = run_benchmark(
        operator=args.op,
        output_csv=output_path,
        only_kernels=args.only,
        device=args.device,
        num_inputs=args.num_inputs,
        rep=args.rep,
    )

    if not success:
        print("Benchmark failed")
        return 1

    # Parse and summarize results
    if not args.no_summary:
        results = parse_csv_results(output_path)
        print_summary(results)

    print(f"\nBaseline benchmark complete. Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
