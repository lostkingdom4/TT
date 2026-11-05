#!/usr/bin/env python3
"""
Kernel Statistics Analyzer

This script analyzes Triton kernels in a directory to determine:
1. Kernel types (1D, 2D, reduction, unknown)
2. Numel parameters found in each kernel
3. Statistical summary of kernel characteristics

Usage:
    python stats_kernel.py [--path PATH] [--output OUTPUT.json]
"""

import argparse
import json
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def read_text(p: Path) -> str:
    """Read text file with error handling."""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        log.warning(f"Failed to read {p}: {e}")
        return ""


def extract_kernel_info(src: str, kernel_name: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[str]]:
    """
    Extract kernel parameter information and detect kernel type.

    Returns:
        (numel_params, other_params, kernel_body)
    """
    # Find the kernel function definition
    pat = rf"@triton\.jit\s*def\s+{re.escape(kernel_name)}\s*\((?P<params>.*?)\)\s*:(?P<body>.+?)(?=\n@triton\.jit|\ndef\s|\Z)"
    m = re.search(pat, src, flags=re.DOTALL)
    if not m:
        return None, None, None

    params_src = m.group("params").strip()
    body = m.group("body")

    # Parse parameter names using AST
    all_param_names: Optional[List[str]] = None
    try:
        dummy = f"def __tmp__({params_src}):\n    pass\n"
        mod = ast.parse(dummy)
        f = mod.body[0]
        if isinstance(f, ast.FunctionDef):
            posonly = getattr(f.args, "posonlyargs", [])
            args = f.args.args
            all_param_names = [a.arg for a in (*posonly, *args)]
    except Exception:
        # Fallback: naive split
        if params_src:
            all_param_names = [p.split(":", 1)[0].split("=", 1)[0].strip()
                              for p in params_src.split(",") if p.strip()]

    if not all_param_names:
        return None, None, None

    # Separate numel parameters from others
    numel_params = [name for name in all_param_names if name.endswith("numel")]
    other_params = [name for name in all_param_names if not name.endswith("numel")]

    return numel_params, other_params, body


def detect_kernel_type(numel_params: List[str], other_params: List[str], body: str) -> str:
    """
    Detect kernel type based on parameters and body content.

    Returns: '1d', '2d', 'reduction', 'helper', 'unknown'
    """
    all_params = numel_params + other_params

    # Check for helper functions (simple scalar operations with no numel parameters)
    if (len(numel_params) == 0 and
        len(all_params) <= 3 and
        all(param.startswith('arg') for param in all_params) and
        len(body.strip().split('\n')) <= 5):  # Simple body with few lines
        return "helper"

    has_ynumel = "ynumel" in numel_params
    has_xnumel = "xnumel" in numel_params
    has_znumel = "znumel" in numel_params
    has_r0_numel = "r0_numel" in numel_params
    has_r1_numel = "r1_numel" in numel_params
    has_r2_numel = "r2_numel" in numel_params

    # Check for block parameters in the body or parameter names
    has_yblock = any("YBLOCK" in p for p in all_params) or "YBLOCK" in body
    has_xblock = any("XBLOCK" in p for p in all_params) or "XBLOCK" in body
    has_r0_block = any("R0_BLOCK" in p for p in all_params) or "R0_BLOCK" in body

    # Check for reduction patterns in body
    has_reduction_ops = any(pattern in body for pattern in [
        "tl.reduce", "tl.sum", "tl.max", "tl.min", "tl.mean",
        "reduce", "reduction"
    ])

    # Classify kernel type
    if has_r0_numel or has_r1_numel or has_r2_numel or has_r0_block or has_reduction_ops:
        return "reduction"
    elif has_ynumel and has_xnumel and (has_yblock or has_xblock):
        if has_znumel:
            return "3d"
        return "2d"
    elif has_xnumel and (has_xblock or "XBLOCK" in body):
        return "1d"
    else:
        return "unknown"


def extract_numel_values(body: str, numel_params: List[str]) -> Dict[str, Optional[int]]:
    """
    Try to extract actual numel values from kernel body.
    """
    numel_values = {}

    for param in numel_params:
        # Look for literal assignments like "xnumel = 1024"
        pattern = rf"\b{re.escape(param)}\s*=\s*(\d+)"
        m = re.search(pattern, body)
        if m:
            numel_values[param] = int(m.group(1))
        else:
            numel_values[param] = None

    return numel_values


def analyze_single_file(file_path: Path) -> Dict:
    """
    Analyze a single Triton kernel file.

    Returns:
        Dictionary with analysis results for this file
    """
    log.debug(f"Analyzing file: {file_path}")

    src = read_text(file_path)
    if not src:
        return {
            "file": str(file_path),
            "error": "Could not read file",
            "kernels": []
        }

    # Find all triton kernels in the file
    kernel_pattern = r"@triton\.jit\s*def\s+(\w+)\s*\("
    kernel_names = re.findall(kernel_pattern, src)

    file_results = {
        "file": str(file_path),
        "total_kernels": len(kernel_names),
        "kernels": []
    }

    for kernel_name in kernel_names:
        log.debug(f"  Analyzing kernel: {kernel_name}")

        numel_params, other_params, body = extract_kernel_info(src, kernel_name)

        if numel_params is None or other_params is None:
            kernel_info = {
                "name": kernel_name,
                "error": "Could not parse kernel parameters"
            }
        else:
            kernel_type = detect_kernel_type(numel_params, other_params, body or "")
            numel_values = extract_numel_values(body or "", numel_params)

            kernel_info = {
                "name": kernel_name,
                "type": kernel_type,
                "numel_params": numel_params,
                "other_params": other_params,
                "total_params": len(numel_params) + len(other_params),
                "numel_values": numel_values,
                "has_literal_values": any(v is not None for v in numel_values.values())
            }

            # Add detailed info for unknown kernels (but not helper kernels)
            if kernel_type == "unknown":
                kernel_info.update({
                    "file_path": str(file_path),
                    "body_snippet": (body or "")[:500] + "..." if body and len(body) > 500 else body,
                    "all_params": numel_params + other_params,
                    "debug_info": {
                        "has_ynumel": "ynumel" in numel_params,
                        "has_xnumel": "xnumel" in numel_params,
                        "has_znumel": "znumel" in numel_params,
                        "has_r0_numel": "r0_numel" in numel_params,
                        "has_yblock_in_params": any("YBLOCK" in p for p in numel_params + other_params),
                        "has_xblock_in_params": any("XBLOCK" in p for p in numel_params + other_params),
                        "has_yblock_in_body": "YBLOCK" in (body or ""),
                        "has_xblock_in_body": "XBLOCK" in (body or ""),
                        "has_reduction_ops": any(pattern in (body or "") for pattern in [
                            "tl.reduce", "tl.sum", "tl.max", "tl.min", "tl.mean", "reduce", "reduction"
                        ])
                    }
                })

        file_results["kernels"].append(kernel_info)

    return file_results


def compute_statistics(all_results: List[Dict]) -> Dict:
    """
    Compute overall statistics from all analyzed files.
    """
    stats = {
        "total_files": len(all_results),
        "total_kernels": 0,
        "successful_kernels": 0,
        "failed_kernels": 0,
        "kernel_types": Counter(),
        "numel_param_frequency": Counter(),
        "unique_numel_params": set(),
        "param_count_distribution": Counter(),
        "files_with_errors": 0,
        "kernels_with_literal_values": 0
    }

    for file_result in all_results:
        if "error" in file_result:
            stats["files_with_errors"] += 1
            continue

        stats["total_kernels"] += file_result["total_kernels"]

        for kernel in file_result["kernels"]:
            if "error" in kernel:
                stats["failed_kernels"] += 1
                continue

            stats["successful_kernels"] += 1
            stats["kernel_types"][kernel["type"]] += 1

            # Count numel parameter occurrences
            for param in kernel["numel_params"]:
                stats["numel_param_frequency"][param] += 1
                stats["unique_numel_params"].add(param)

            # Parameter count distribution
            stats["param_count_distribution"][kernel["total_params"]] += 1

            # Kernels with literal numel values
            if kernel.get("has_literal_values", False):
                stats["kernels_with_literal_values"] += 1

    # Convert set to list for JSON serialization
    stats["unique_numel_params"] = sorted(list(stats["unique_numel_params"]))

    return stats


def print_summary(stats: Dict):
    """Print a human-readable summary of the statistics."""
    print("\n" + "="*60)
    print("TRITON KERNEL ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nFiles analyzed: {stats['total_files']}")
    print(f"Files with errors: {stats['files_with_errors']}")
    print(f"Total kernels found: {stats['total_kernels']}")
    print(f"Successfully analyzed: {stats['successful_kernels']}")
    print(f"Failed to analyze: {stats['failed_kernels']}")

    print(f"\nKERNEL TYPES:")
    for kernel_type, count in stats['kernel_types'].most_common():
        percentage = (count / stats['successful_kernels'] * 100) if stats['successful_kernels'] > 0 else 0
        print(f"  {kernel_type:>10}: {count:>4} ({percentage:5.1f}%)")

    print(f"\nNUMEL PARAMETERS (frequency):")
    for param, count in stats['numel_param_frequency'].most_common():
        percentage = (count / stats['successful_kernels'] * 100) if stats['successful_kernels'] > 0 else 0
        print(f"  {param:>12}: {count:>4} ({percentage:5.1f}%)")

    print(f"\nPARAMETER COUNT DISTRIBUTION:")
    for param_count, count in sorted(stats['param_count_distribution'].items()):
        percentage = (count / stats['successful_kernels'] * 100) if stats['successful_kernels'] > 0 else 0
        print(f"  {param_count:>2} params: {count:>4} kernels ({percentage:5.1f}%)")

    print(f"\nUNIQUE NUMEL PARAMETERS FOUND:")
    print(f"  {', '.join(stats['unique_numel_params'])}")

    print(f"\nKernels with literal numel values: {stats['kernels_with_literal_values']}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Analyze Triton kernels for type and parameter statistics")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./data_64/triton_modules"),
        help="Path to directory containing Triton kernel files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./kernel_analysis_results.json"),
        help="Output JSON file for detailed results (optional)"
    )
    parser.add_argument(
        "--pattern",
        default="*.py",
        help="File pattern to match (default: *.py)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Check if path exists
    if not args.path.exists():
        log.error(f"Path does not exist: {args.path}")
        return 1

    if not args.path.is_dir():
        log.error(f"Path is not a directory: {args.path}")
        return 1

    # Find all Python files in the directory
    kernel_files = list(args.path.glob(args.pattern))

    if not kernel_files:
        log.warning(f"No files matching pattern '{args.pattern}' found in {args.path}")
        return 1

    log.info(f"Found {len(kernel_files)} files to analyze")

    # Analyze all files
    all_results = []
    for i, file_path in enumerate(kernel_files, 1):
        log.info(f"Processing {i}/{len(kernel_files)}: {file_path.name}")
        result = analyze_single_file(file_path)
        all_results.append(result)

    # Compute statistics
    stats = compute_statistics(all_results)

    # Print summary
    print_summary(stats)

    # Collect helper and unknown kernels for detailed analysis
    helper_kernels = []
    unknown_kernels = []
    for file_result in all_results:
        for kernel in file_result.get("kernels", []):
            if kernel.get("type") == "helper":
                helper_kernels.append(kernel)
            elif kernel.get("type") == "unknown":
                unknown_kernels.append(kernel)

    # Print summary of helper kernels
    if helper_kernels:
        print(f"\n{'='*80}")
        print(f"HELPER KERNELS SUMMARY ({len(helper_kernels)} found)")
        print(f"{'='*80}")
        print("Helper kernels are scalar utility functions generated by PyTorch compilation.")
        print("They perform simple operations like addition, multiplication, etc.")

        # Group by file
        helper_by_file = {}
        for kernel in helper_kernels:
            file_path = kernel.get('file_path', 'Unknown')
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path
            if file_name not in helper_by_file:
                helper_by_file[file_name] = []
            helper_by_file[file_name].append(kernel['name'])

        print(f"\nFiles containing helper kernels ({len(helper_by_file)} files):")
        for file_name, kernel_names in helper_by_file.items():
            print(f"  {file_name}: {len(kernel_names)} helpers")

    # Print detailed info about unknown kernels
    if unknown_kernels:
        print(f"\n{'='*80}")
        print(f"DETAILED ANALYSIS OF {len(unknown_kernels)} UNKNOWN KERNELS")
        print(f"{'='*80}")

        for i, kernel in enumerate(unknown_kernels[:20], 1):  # Show first 20
            print(f"\n--- Unknown Kernel #{i}: {kernel['name']} ---")
            print(f"File: {kernel.get('file_path', 'N/A')}")
            print(f"All parameters: {kernel.get('all_params', [])}")
            print(f"Numel parameters: {kernel['numel_params']}")
            print(f"Other parameters: {kernel['other_params']}")

            debug = kernel.get('debug_info', {})
            if debug:
                print(f"Debug info:")
                for key, value in debug.items():
                    print(f"  - {key}: {value}")

            if kernel.get('body_snippet'):
                print(f"Body snippet: {kernel['body_snippet'][:200]}...")
            print("-" * 60)

        if len(unknown_kernels) > 20:
            print(f"\n... and {len(unknown_kernels) - 20} more unknown kernels")    # Save detailed results if requested
    if args.output:
        output_data = {
            "summary": stats,
            "detailed_results": all_results,
            "helper_kernels": helper_kernels,
            "unknown_kernels_analysis": unknown_kernels
        }

        # Convert Counter objects to regular dicts for JSON serialization
        stats["kernel_types"] = dict(stats["kernel_types"])
        stats["numel_param_frequency"] = dict(stats["numel_param_frequency"])
        stats["param_count_distribution"] = dict(stats["param_count_distribution"])

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Detailed results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
