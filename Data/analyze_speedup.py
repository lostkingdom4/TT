#!/usr/bin/env python3
"""
Analyze speedup metrics from Triton kernel auto-tuning results.

This script reads all JSON files in the performance folder, finds the default
configuration runtime and compares it with the best runtime to compute speedup.
Also compares with most frequent config by kernel type and global most frequent config.
Reports minimum, average, and maximum speedup across all kernels.
"""

import json
import os
import glob
from typing import Dict, List, Optional, Tuple
from collections import Counter
import statistics
import math

def find_default_config_runtime(all_configs: List[Dict], default_config: Dict) -> Optional[float]:
    """
    Find the runtime (lat_ms) for the default configuration in all_configs.

    Args:
        all_configs: List of all tested configurations with their runtimes
        default_config: The default configuration parameters to match

    Returns:
        Runtime in milliseconds if found, None otherwise
    """
    # Filter out non-tunable parameters (grid is computed, not tuned)
    tunable_params = {k: v for k, v in default_config.items() if k != 'grid'}

    for config_result in all_configs:
        if config_result.get('status') != 'success':
            continue

        config = config_result.get('config', {})

        # Check if this config matches the default config (only tunable params)
        match = True
        for key, value in tunable_params.items():
            if config.get(key) != value:
                match = False
                break

        if match:
            return config_result.get('lat_ms')

    return None

def detect_kernel_type(kernel_name: str, default_config: Dict) -> str:
    """Detect kernel type based on kernel name and config parameters."""
    if 'triton_poi_' in kernel_name:
        return '1d'
    elif 'triton_per_' in kernel_name:
        return 'reduction_fixed'
    elif 'triton_red_' in kernel_name:
        return 'reduction'
    elif 'R0_BLOCK' in default_config:
        return 'reduction'
    elif 'YBLOCK' in default_config:
        return '2d'
    elif 'XBLOCK' in default_config:
        return '1d'
    return 'unknown'

def config_to_tuple(config: Dict) -> tuple:
    """Convert config dict to hashable tuple for frequency counting."""
    return tuple(sorted(config.items()))

def analyze_performance_file(filepath: str, verbose: bool = False) -> Optional[Dict]:
    """
    Analyze a single performance JSON file to extract speedup metrics.

    Args:
        filepath: Path to the JSON performance file
        verbose: If True, print detailed warning messages

    Returns:
        Dictionary containing analysis results or None if analysis failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Extract key information
        kernel_name = data.get('kernel', os.path.basename(filepath))
        default_config = data.get('default_config', {})
        best_runtime = data.get('best', {}).get('lat_ms')
        all_configs = data.get('all_configs', [])

        if not default_config or not best_runtime or not all_configs:
            if verbose:
                print(f"Warning: Missing data in {filepath}")
            return None

        # Find default config runtime
        default_runtime = find_default_config_runtime(all_configs, default_config)

        if default_runtime is None:
            if verbose:
                print(f"Warning: Default config not tested in {os.path.basename(filepath)}")
                print(f"  Default config: {default_config}")
                # Get search space to see what was tested
                search_space = data.get('search_space', {})
                if search_space:
                    print(f"  Tested XBLOCK values: {search_space.get('XBLOCKS', 'N/A')}")
                    print(f"  Tested WARPS values: {search_space.get('WARPS', 'N/A')}")
            return None

        # Detect kernel type
        kernel_type = detect_kernel_type(kernel_name, default_config)

        # Calculate speedup
        speedup = default_runtime / best_runtime

        result = {
            'kernel_name': kernel_name,
            'kernel_type': kernel_type,
            'filepath': filepath,
            'default_runtime_ms': default_runtime,
            'best_runtime_ms': best_runtime,
            'speedup': speedup,
            'default_config': default_config,
            'best_config': data.get('best', {}).get('cfg', {}),
            'total_configs': len(all_configs),
            'successful_configs': len([c for c in all_configs if c.get('status') == 'success']),
            'all_configs': all_configs
        }

        return result

    except Exception as e:
        if verbose:
            print(f"Error analyzing {filepath}: {e}")
        return None

def main():
    """Main function to analyze all performance files and generate speedup report."""

    # Get current directory and performance folder path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    performance_dir = os.path.join(current_dir, 'performance')

    if not os.path.exists(performance_dir):
        print(f"Error: Performance directory not found at {performance_dir}")
        return

    # Find all JSON files in performance directory
    json_files = glob.glob(os.path.join(performance_dir, '*.json'))

    if not json_files:
        print(f"Error: No JSON files found in {performance_dir}")
        return

    print(f"Found {len(json_files)} JSON files to analyze")
    print("=" * 80)

    # Analyze each file
    results = []
    failed_files = []
    verbose = False  # Set to True for detailed warnings

    for filepath in sorted(json_files):
        result = analyze_performance_file(filepath, verbose=verbose)
        if result:
            results.append(result)
        else:
            failed_files.append(filepath)

    if not results:
        print("Error: No files could be analyzed successfully")
        return

    # Group results by kernel type
    results_by_type = {}
    for result in results:
        ktype = result['kernel_type']
        if ktype not in results_by_type:
            results_by_type[ktype] = []
        results_by_type[ktype].append(result)

    # Find most frequent config for each kernel type
    most_frequent_by_type = {}
    for ktype, ktype_results in results_by_type.items():
        config_counter = Counter()
        for result in ktype_results:
            config_tuple = config_to_tuple(result['default_config'])
            config_counter[config_tuple] += 1

        if config_counter:
            most_common = config_counter.most_common(1)[0]
            most_frequent_by_type[ktype] = dict(most_common[0])

    # Generate "best practice" config for each kernel type
    # For each parameter, find the value that most frequently appears in top-performing configs
    best_practice_by_type = {}
    for ktype, ktype_results in results_by_type.items():
        # Get top 25% performers by speedup
        sorted_by_speedup = sorted(ktype_results, key=lambda x: x['speedup'], reverse=True)
        top_25_percent = max(1, len(sorted_by_speedup) // 4)
        top_performers = sorted_by_speedup[:top_25_percent]

        # Count parameter values in top performers
        param_counters = {}
        for result in top_performers:
            for param, value in result['best_config'].items():
                if param not in param_counters:
                    param_counters[param] = Counter()
                # Convert lists to tuples for hashing
                hashable_value = tuple(value) if isinstance(value, list) else value
                param_counters[param][hashable_value] += 1

        # Build best practice config with most frequent values
        best_practice_config = {}
        for param, counter in param_counters.items():
            if counter:
                most_common_value = counter.most_common(1)[0][0]
                # Convert tuple back to list if it was originally a list
                best_practice_config[param] = list(most_common_value) if isinstance(most_common_value, tuple) else most_common_value

        best_practice_by_type[ktype] = best_practice_config

    # Calculate speedups for different baseline configs
    for result in results:
        ktype = result['kernel_type']
        all_configs = result['all_configs']
        best_runtime = result['best_runtime_ms']

        # Speedup vs most frequent config for this kernel type
        if ktype in most_frequent_by_type:
            freq_config = most_frequent_by_type[ktype]
            freq_runtime = find_default_config_runtime(all_configs, freq_config)
            if freq_runtime:
                result['freq_type_runtime_ms'] = freq_runtime
                result['speedup_vs_freq_type'] = freq_runtime / best_runtime
            else:
                result['freq_type_runtime_ms'] = None
                result['speedup_vs_freq_type'] = None

        # Speedup vs best practice config for this kernel type
        if ktype in best_practice_by_type:
            bp_config = best_practice_by_type[ktype]
            bp_runtime = find_default_config_runtime(all_configs, bp_config)
            if bp_runtime:
                result['best_practice_runtime_ms'] = bp_runtime
                result['speedup_vs_best_practice'] = bp_runtime / best_runtime
            else:
                result['best_practice_runtime_ms'] = None
                result['speedup_vs_best_practice'] = None

    # Calculate statistics
    speedups = [r['speedup'] for r in results]
    min_speedup = min(speedups)
    max_speedup = max(speedups)
    avg_speedup = statistics.mean(speedups)
    median_speedup = statistics.median(speedups)
    geomean_speedup = math.exp(statistics.mean(math.log(s) for s in speedups))

    # Calculate statistics for freq type baseline
    speedups_vs_freq_type = [r['speedup_vs_freq_type'] for r in results if r.get('speedup_vs_freq_type')]
    if speedups_vs_freq_type:
        avg_speedup_vs_freq_type = statistics.mean(speedups_vs_freq_type)
        median_speedup_vs_freq_type = statistics.median(speedups_vs_freq_type)
        geomean_speedup_vs_freq_type = math.exp(statistics.mean(math.log(s) for s in speedups_vs_freq_type))
    else:
        avg_speedup_vs_freq_type = None
        median_speedup_vs_freq_type = None
        geomean_speedup_vs_freq_type = None

    # Calculate statistics for best practice baseline
    speedups_vs_bp = [r['speedup_vs_best_practice'] for r in results if r.get('speedup_vs_best_practice')]
    if speedups_vs_bp:
        avg_speedup_vs_bp = statistics.mean(speedups_vs_bp)
        median_speedup_vs_bp = statistics.median(speedups_vs_bp)
        geomean_speedup_vs_bp = math.exp(statistics.mean(math.log(s) for s in speedups_vs_bp))
    else:
        avg_speedup_vs_bp = None
        median_speedup_vs_bp = None
        geomean_speedup_vs_bp = None

    # Print detailed results
    print(f"\n{'Kernel Analysis Results':<50} {'vs Default':>12} {'vs FreqType':>12} {'vs BestPrac':>12}")
    print("=" * 90)

    for result in sorted(results, key=lambda x: x['speedup']):
        kernel_name = result['kernel_name'][:45]  # Truncate long names
        speedup_default = result['speedup']
        speedup_freq_type = result.get('speedup_vs_freq_type', 0) or 0
        speedup_bp = result.get('speedup_vs_best_practice', 0) or 0
        print(f"{kernel_name:<50} {speedup_default:>10.2f}x {speedup_freq_type:>10.2f}x {speedup_bp:>10.2f}x")

    # Print summary statistics
    print("\n" + "=" * 90)
    print(f"SUMMARY STATISTICS ({len(results)} kernels analyzed)")
    print("=" * 90)
    print(f"{'Baseline:':<30} {'vs Default':>12} {'vs FreqType':>12} {'vs BestPrac':>12}")
    print(f"{'Minimum speedup:':<30} {min_speedup:>10.2f}x {min(speedups_vs_freq_type) if speedups_vs_freq_type else 0:>10.2f}x {min(speedups_vs_bp) if speedups_vs_bp else 0:>10.2f}x")
    print(f"{'Average speedup:':<30} {avg_speedup:>10.2f}x {avg_speedup_vs_freq_type if avg_speedup_vs_freq_type else 0:>10.2f}x {avg_speedup_vs_bp if avg_speedup_vs_bp else 0:>10.2f}x")
    print(f"{'Geometric mean speedup:':<30} {geomean_speedup:>10.2f}x {geomean_speedup_vs_freq_type if geomean_speedup_vs_freq_type else 0:>10.2f}x {geomean_speedup_vs_bp if geomean_speedup_vs_bp else 0:>10.2f}x")
    print(f"{'Median speedup:':<30} {median_speedup:>10.2f}x {median_speedup_vs_freq_type if median_speedup_vs_freq_type else 0:>10.2f}x {median_speedup_vs_bp if median_speedup_vs_bp else 0:>10.2f}x")
    print(f"{'Maximum speedup:':<30} {max_speedup:>10.2f}x {max(speedups_vs_freq_type) if speedups_vs_freq_type else 0:>10.2f}x {max(speedups_vs_bp) if speedups_vs_bp else 0:>10.2f}x")

    # Print most frequent configs
    print("\n" + "=" * 100)
    print("MOST FREQUENT DEFAULT CONFIGURATIONS BY KERNEL TYPE")
    print("=" * 100)
    for ktype, config in most_frequent_by_type.items():
        print(f"\n{ktype.upper()}:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")

    # Print best practice configs
    print("\n" + "=" * 90)
    print("BEST PRACTICE CONFIGURATIONS BY KERNEL TYPE")
    print("(Most frequent parameters in top 25% performers)")
    print("=" * 90)
    for ktype, config in best_practice_by_type.items():
        print(f"\n{ktype.upper()}:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")

    # Print detailed breakdown for best and worst performing kernels
    print(f"\n" + "=" * 90)
    print(f"BEST PERFORMING KERNEL:")
    best_result = max(results, key=lambda x: x['speedup'])
    print(f"  Kernel: {best_result['kernel_name']}")
    print(f"  Type: {best_result['kernel_type']}")
    print(f"  Speedup vs default: {best_result['speedup']:.2f}x")
    print(f"  Speedup vs freq type: {best_result.get('speedup_vs_freq_type', 'N/A')}")
    print(f"  Speedup vs best practice: {best_result.get('speedup_vs_best_practice', 'N/A')}")
    print(f"  Default runtime: {best_result['default_runtime_ms']:.4f} ms")
    print(f"  Best runtime: {best_result['best_runtime_ms']:.4f} ms")
    print(f"  Configs tested: {best_result['successful_configs']}/{best_result['total_configs']}")

    print(f"\nWORST PERFORMING KERNEL:")
    worst_result = min(results, key=lambda x: x['speedup'])
    print(f"  Kernel: {worst_result['kernel_name']}")
    print(f"  Type: {worst_result['kernel_type']}")
    print(f"  Speedup vs default: {worst_result['speedup']:.2f}x")
    print(f"  Speedup vs freq type: {worst_result.get('speedup_vs_freq_type', 'N/A')}")
    print(f"  Speedup vs best practice: {worst_result.get('speedup_vs_best_practice', 'N/A')}")
    print(f"  Default runtime: {worst_result['default_runtime_ms']:.4f} ms")
    print(f"  Best runtime: {worst_result['best_runtime_ms']:.4f} ms")
    print(f"  Configs tested: {worst_result['successful_configs']}/{worst_result['total_configs']}")

    # Report failures
    if failed_files:
        print(f"\n" + "=" * 90)
        print(f"FAILED TO ANALYZE ({len(failed_files)} files):")
        for filepath in failed_files[:10]:  # Show first 10
            print(f"  {os.path.basename(filepath)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Save detailed results to JSON
    output_file = os.path.join(current_dir, 'speedup_analysis_results.json')

    # Clean results for JSON serialization (remove all_configs to reduce size)
    results_for_json = []
    for r in results:
        r_copy = r.copy()
        r_copy.pop('all_configs', None)
        results_for_json.append(r_copy)

    summary = {
        'summary_stats': {
            'total_kernels_analyzed': len(results),
            'min_speedup': min_speedup,
            'avg_speedup': avg_speedup,
            'geomean_speedup': geomean_speedup,
            'median_speedup': median_speedup,
            'max_speedup': max_speedup,
            'avg_speedup_vs_freq_type': avg_speedup_vs_freq_type,
            'geomean_speedup_vs_freq_type': geomean_speedup_vs_freq_type,
            'median_speedup_vs_freq_type': median_speedup_vs_freq_type,
            'avg_speedup_vs_best_practice': avg_speedup_vs_bp,
            'geomean_speedup_vs_best_practice': geomean_speedup_vs_bp,
            'median_speedup_vs_best_practice': median_speedup_vs_bp,
        },
        'most_frequent_by_type': most_frequent_by_type,
        'best_practice_by_type': best_practice_by_type,
        'detailed_results': results_for_json,
        'failed_files': failed_files
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()