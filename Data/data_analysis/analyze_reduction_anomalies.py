#!/usr/bin/env python3
"""
Analyze reduction kernels that don't have R0_BLOCK parameters.
This helps identify potential classification issues or special reduction patterns.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

def load_analysis_results(json_file: Path) -> Dict:
    """Load the analysis results from JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        sys.exit(1)

def find_reduction_kernels_without_r0_block(analysis_data: Dict) -> Dict:
    """
    Find reduction kernels that don't have R0_BLOCK in their configuration.

    Returns:
        Dict with categories of anomalous reduction kernels
    """
    results = {
        "reduction_without_r0_block": [],
        "reduction_with_only_xblock": [],
        "reduction_with_unexpected_params": [],
        "summary": {
            "total_reduction_kernels": 0,
            "anomalous_kernels": 0,
            "files_affected": set(),
            "common_patterns": Counter()
        }
    }

    kernel_configs = analysis_data.get("kernel_configs", {})

    for kernel_id, kernel_info in kernel_configs.items():
        if kernel_info.get("kernel_type") == "reduction":
            results["summary"]["total_reduction_kernels"] += 1

            default_config = kernel_info.get("default_config", {})
            file_name = kernel_info.get("file", "unknown")
            kernel_name = kernel_info.get("kernel_name", "unknown")

            if not default_config:
                continue

            # Check if R0_BLOCK is missing
            has_r0_block = any("R0_BLOCK" in str(key) for key in default_config.keys())
            has_xblock = any("XBLOCK" in str(key) for key in default_config.keys())

            if not has_r0_block:
                results["summary"]["anomalous_kernels"] += 1
                results["summary"]["files_affected"].add(file_name)

                anomaly_info = {
                    "kernel_id": kernel_id,
                    "file": file_name,
                    "kernel_name": kernel_name,
                    "config": default_config,
                    "config_keys": list(default_config.keys()),
                    "has_xblock": has_xblock,
                    "param_count": len(default_config)
                }

                results["reduction_without_r0_block"].append(anomaly_info)

                # Categorize further
                if has_xblock and len(default_config) == 3:  # XBLOCK + num_warps + num_stages
                    results["reduction_with_only_xblock"].append(anomaly_info)
                    results["summary"]["common_patterns"]["xblock_only"] += 1
                elif not has_xblock and not has_r0_block:
                    results["reduction_with_unexpected_params"].append(anomaly_info)
                    results["summary"]["common_patterns"]["no_block_params"] += 1
                else:
                    results["summary"]["common_patterns"]["other_anomaly"] += 1

    # Convert sets to lists for JSON serialization
    results["summary"]["files_affected"] = list(results["summary"]["files_affected"])
    results["summary"]["common_patterns"] = dict(results["summary"]["common_patterns"])

    return results

def analyze_config_patterns(anomalous_kernels: List[Dict]) -> Dict:
    """Analyze common configuration patterns in anomalous kernels."""
    pattern_analysis = {
        "config_signatures": Counter(),
        "parameter_frequencies": defaultdict(Counter),
        "file_patterns": defaultdict(list)
    }

    for kernel in anomalous_kernels:
        config = kernel["config"]
        file_name = kernel["file"]

        # Create config signature (sorted keys)
        config_signature = tuple(sorted(config.keys()))
        pattern_analysis["config_signatures"][config_signature] += 1

        # Track parameter values
        for param, value in config.items():
            pattern_analysis["parameter_frequencies"][param][value] += 1

        # Group by file patterns
        pattern_analysis["file_patterns"][file_name].append(kernel["kernel_name"])

    # Convert to regular dicts for JSON serialization (convert tuple keys to strings)
    pattern_analysis["config_signatures"] = {
        str(sig): count for sig, count in pattern_analysis["config_signatures"].items()
    }
    pattern_analysis["parameter_frequencies"] = {
        param: dict(values) for param, values in pattern_analysis["parameter_frequencies"].items()
    }
    pattern_analysis["file_patterns"] = dict(pattern_analysis["file_patterns"])

    return pattern_analysis

def generate_detailed_report(results: Dict, pattern_analysis: Dict) -> str:
    """Generate a detailed text report of the findings."""
    report_lines = [
        "=" * 80,
        "REDUCTION KERNEL ANOMALY ANALYSIS REPORT",
        "=" * 80,
        "",
        "SUMMARY:",
        f"  Total Reduction Kernels: {results['summary']['total_reduction_kernels']}",
        f"  Anomalous Kernels (no R0_BLOCK): {results['summary']['anomalous_kernels']}",
        f"  Anomaly Rate: {results['summary']['anomalous_kernels'] / max(1, results['summary']['total_reduction_kernels']) * 100:.1f}%",
        f"  Files Affected: {len(results['summary']['files_affected'])}",
        "",
        "ANOMALY PATTERNS:",
    ]

    for pattern, count in results['summary']['common_patterns'].items():
        report_lines.append(f"  {pattern}: {count} kernels")

    report_lines.extend([
        "",
        "TOP CONFIGURATION SIGNATURES:",
    ])

    for signature, count in sorted(pattern_analysis['config_signatures'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
        report_lines.append(f"  {signature}: {count} kernels")

    report_lines.extend([
        "",
        "PARAMETER VALUE FREQUENCIES:",
    ])

    for param, value_counts in pattern_analysis['parameter_frequencies'].items():
        report_lines.append(f"  {param}:")
        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"    {value}: {count}")

    report_lines.extend([
        "",
        "FILES WITH MOST ANOMALIES:",
    ])

    file_counts = [(f, len(kernels)) for f, kernels in pattern_analysis['file_patterns'].items()]
    for file_name, count in sorted(file_counts, key=lambda x: x[1], reverse=True)[:10]:
        report_lines.append(f"  {file_name}: {count} anomalous kernels")

    report_lines.extend([
        "",
        "SAMPLE ANOMALOUS KERNELS:",
    ])

    for i, kernel in enumerate(results['reduction_without_r0_block'][:5]):
        report_lines.extend([
            f"  {i+1}. {kernel['kernel_id']}",
            f"     File: {kernel['file']}",
            f"     Config: {kernel['config']}",
            ""
        ])

    if len(results['reduction_without_r0_block']) > 5:
        report_lines.append(f"     ... and {len(results['reduction_without_r0_block']) - 5} more")

    report_lines.append("=" * 80)

    return "\n".join(report_lines)

def main():
    """Main analysis function."""
    # Input and output paths
    json_file = Path("default_config_analysis.json")
    output_json = Path("reduction_anomaly_analysis.json")
    output_report = Path("reduction_anomaly_report.txt")

    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        print("Please run the default config analyzer first.")
        sys.exit(1)

    print("Loading analysis data...")
    analysis_data = load_analysis_results(json_file)

    print("Finding reduction kernels without R0_BLOCK...")
    results = find_reduction_kernels_without_r0_block(analysis_data)

    print("Analyzing configuration patterns...")
    pattern_analysis = analyze_config_patterns(results["reduction_without_r0_block"])

    # Combine results
    full_results = {
        "analysis_metadata": {
            "source_file": str(json_file),
            "analysis_date": "2025-11-23",
            "total_kernels_analyzed": len(analysis_data.get("kernel_configs", {}))
        },
        "anomaly_results": results,
        "pattern_analysis": pattern_analysis
    }

    print("Generating reports...")

    # Save JSON results
    with open(output_json, 'w') as f:
        json.dump(full_results, f, indent=2)

    # Generate and save text report
    text_report = generate_detailed_report(results, pattern_analysis)
    with open(output_report, 'w') as f:
        f.write(text_report)

    # Print summary to console
    print("\nANALYSIS COMPLETE!")
    print(f"Results saved to: {output_json}")
    print(f"Report saved to: {output_report}")
    print("\nQUICK SUMMARY:")
    print(f"  Found {results['summary']['anomalous_kernels']} reduction kernels without R0_BLOCK")
    print(f"  Out of {results['summary']['total_reduction_kernels']} total reduction kernels")
    print(f"  Affects {len(results['summary']['files_affected'])} files")

    if results['summary']['anomalous_kernels'] > 0:
        print("\nTOP PATTERNS:")
        for pattern, count in list(results['summary']['common_patterns'].items())[:3]:
            print(f"  {pattern}: {count} kernels")

if __name__ == "__main__":
    main()