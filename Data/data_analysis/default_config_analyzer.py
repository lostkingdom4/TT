#!/usr/bin/env python3
"""
Default Configuration Analyzer for Triton Kernels

This script analyzes all Triton kernel files in data_64 to extract their default
configurations and perform statistical analysis by kernel type.
"""

import json
import os
import sys
import re
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import argparse

# Add parent directory to path to import tune_one_file
sys.path.insert(0, str(Path(__file__).parent.parent))
from tune_one_file import extract_default_config, read_text, detect_kernel_type, infer_numel_for_kernel

def detect_kernel_type(param_names: List[str], kernel_name: str = "", default_config: Optional[Dict] = None) -> str:
    """
    Kernel type detection that distinguishes reduction types.

    Returns:
        - '1d': Standard 1D pointwise kernels
        - '2d': 2D kernels with both X and Y dimensions  
        - 'reduction': Reduction kernels with tunable R0_BLOCK parameter
        - 'reduction_fixed': Reduction kernels with fixed R0_BLOCK (not tunable)
        - 'unknown': Cannot determine type
    """
    has_ynumel = "ynumel" in param_names
    has_xnumel = "xnumel" in param_names
    has_r0_numel = "r0_numel" in param_names
    has_yblock = any("YBLOCK" in p for p in param_names)
    has_xblock = any("XBLOCK" in p for p in param_names)
    has_r0_block = any("R0_BLOCK" in p for p in param_names)

    # Check for reduction patterns - distinguish between tunable and fixed R0_BLOCK
    if has_r0_numel or has_r0_block:
        if has_r0_block:
            return "reduction"  # Has tunable R0_BLOCK parameter
        else:
            return "reduction_fixed"  # Has r0_numel but R0_BLOCK is fixed/predefined

    # Standard 2D kernels
    if has_ynumel and has_xnumel and has_yblock and has_xblock:
        return "2d"

    # Standard 1D kernels
    if has_xnumel and has_xblock:
        return "1d"

    # If we have default_config info, use it to make better decisions
    if default_config:
        config_keys = set(default_config.keys())
        if "R0_BLOCK" in config_keys:
            return "reduction"  # Has tunable R0_BLOCK parameter
        elif has_r0_numel and "R0_BLOCK" not in config_keys:
            return "reduction_fixed"  # Has r0_numel but R0_BLOCK is fixed/predefined
        elif "YBLOCK" in config_keys and "XBLOCK" in config_keys:
            return "2d"
        elif "XBLOCK" in config_keys:
            return "1d"

    return "unknown"


class DefaultConfigAnalyzer:
    """Analyzes default configurations from Triton kernel files"""

    def __init__(self, data_dir: str = "../data_64"):
        self.data_dir = Path(data_dir).resolve()
        self.triton_modules_dir = self.data_dir / "triton_modules"
        self.output_dir = Path(__file__).parent

        # Storage for results
        self.kernel_configs = {}
        self.kernel_types = {}
        self.extraction_stats = {
            "total_files": 0,
            "processed_files": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "kernels_found": 0
        }

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def analyze_all_kernels(self) -> Dict[str, Any]:
        """Analyze all Triton kernel files and extract default configurations"""

        if not self.triton_modules_dir.exists():
            self.logger.error(f"Triton modules directory not found: {self.triton_modules_dir}")
            return {}

        triton_files = list(self.triton_modules_dir.glob("*_triton.py"))
        self.extraction_stats["total_files"] = len(triton_files)

        self.logger.info(f"Found {len(triton_files)} Triton kernel files in {self.triton_modules_dir}")

        if not triton_files:
            self.logger.warning("No Triton files found")
            return {}

        # Process each file
        for triton_file in triton_files:
            try:
                self._analyze_single_file(triton_file)
                self.extraction_stats["processed_files"] += 1
                self.logger.info(f"✅ Processed: {triton_file.name}")
            except Exception as e:
                self.logger.error(f"❌ Error processing {triton_file.name}: {e}")

        # Generate analysis results
        results = {
            "extraction_stats": self.extraction_stats,
            "kernel_configs": self.kernel_configs,
            "kernel_types": self.kernel_types,
            "statistical_analysis": self._perform_statistical_analysis()
        }

        return results

    def _analyze_single_file(self, triton_file: Path) -> None:
        """Analyze a single Triton kernel file"""

        self.logger.debug(f"Processing file: {triton_file}")

        # Read source code
        src = read_text(triton_file)

        # Find all kernel functions in the file
        kernel_pattern = r"@triton\.jit\s+def\s+(\w+)\s*\("
        kernel_matches = re.findall(kernel_pattern, src)

        if not kernel_matches:
            self.logger.warning(f"No Triton kernels found in {triton_file.name}")
            return

        file_base_name = triton_file.stem.replace("_triton", "")

        for kernel_name in kernel_matches:
            try:
                self.extraction_stats["kernels_found"] += 1

                # Extract default configuration
                default_config = extract_default_config(src, kernel_name)

                # Infer kernel parameters for type detection
                numel_param_names, numel_values, other_param_names = infer_numel_for_kernel(src, kernel_name)

                if numel_param_names and numel_values and other_param_names:
                    param_names = numel_param_names + other_param_names
                    # Use enhanced detection that can distinguish per-element reductions
                    kernel_type = detect_kernel_type(param_names, kernel_name, default_config)
                else:
                    kernel_type = "unknown"

                # Store results
                kernel_id = f"{file_base_name}::{kernel_name}"

                self.kernel_configs[kernel_id] = {
                    "file": triton_file.name,
                    "kernel_name": kernel_name,
                    "default_config": default_config,
                    "extraction_success": default_config is not None,
                    "numel_params": dict(zip(numel_param_names or [], numel_values or [])),
                    "other_params": other_param_names or [],
                    "kernel_type": kernel_type
                }

                self.kernel_types[kernel_id] = kernel_type

                if default_config is not None:
                    self.extraction_stats["successful_extractions"] += 1
                    self.logger.debug(f"  ✅ {kernel_name}: {default_config}")
                else:
                    self.extraction_stats["failed_extractions"] += 1
                    self.logger.debug(f"  ❌ {kernel_name}: Failed to extract config")

            except Exception as e:
                self.extraction_stats["failed_extractions"] += 1
                self.logger.error(f"  ❌ Error processing kernel {kernel_name}: {e}")

    def _perform_statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis on extracted configurations"""

        # Group kernels by type
        kernels_by_type = defaultdict(list)
        configs_by_type = defaultdict(list)

        for kernel_id, kernel_info in self.kernel_configs.items():
            kernel_type = kernel_info["kernel_type"]
            kernels_by_type[kernel_type].append(kernel_id)

            if kernel_info["default_config"]:
                configs_by_type[kernel_type].append(kernel_info["default_config"])

        # Analyze configuration patterns by kernel type
        type_analysis = {}

        for kernel_type, configs in configs_by_type.items():
            if not configs:
                continue

            # Convert configs to comparable strings for frequency analysis
            config_strings = []
            param_frequencies = defaultdict(Counter)

            for config in configs:
                # Create standardized config string
                sorted_items = sorted(config.items()) if isinstance(config, dict) else []
                config_str = str(dict(sorted_items))
                config_strings.append(config_str)

                # Track individual parameter frequencies
                if isinstance(config, dict):
                    for param, value in config.items():
                        param_frequencies[param][value] += 1

            # Calculate statistics
            config_counter = Counter(config_strings)

            type_analysis[kernel_type] = {
                "total_kernels": len(kernels_by_type[kernel_type]),
                "kernels_with_configs": len(configs),
                "unique_configurations": len(config_counter),
                "most_common_configs": dict(config_counter.most_common(5)),
                "parameter_frequencies": {
                    param: dict(counter.most_common())
                    for param, counter in param_frequencies.items()
                },
                "config_diversity": len(config_counter) / max(len(configs), 1)
            }

        # Overall statistics
        total_successful = sum(1 for info in self.kernel_configs.values()
                             if info["extraction_success"])

        overall_stats = {
            "total_kernels": len(self.kernel_configs),
            "successful_extractions": total_successful,
            "extraction_success_rate": total_successful / max(len(self.kernel_configs), 1),
            "kernel_type_distribution": dict(Counter(self.kernel_types.values())),
            "by_kernel_type": type_analysis
        }

        return overall_stats

    def save_results(self, results: Dict[str, Any], output_file: str = "default_config_analysis.json") -> None:
        """Save analysis results to JSON file"""

        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"💾 Results saved to: {output_path}")

    def print_analysis_summary(self, results: Dict[str, Any]) -> None:
        """Print formatted summary of analysis results"""

        print("\n" + "="*70)
        print("🔍 TRITON KERNEL DEFAULT CONFIGURATION ANALYSIS")
        print("="*70)

        # Extraction statistics
        stats = results["extraction_stats"]
        print(f"\n📊 Extraction Summary:")
        print(f"   • Files processed: {stats['processed_files']}/{stats['total_files']}")
        print(f"   • Kernels found: {stats['kernels_found']}")
        print(f"   • Successful extractions: {stats['successful_extractions']}")
        print(f"   • Failed extractions: {stats['failed_extractions']}")

        success_rate = (stats['successful_extractions'] / max(stats['kernels_found'], 1)) * 100
        print(f"   • Success rate: {success_rate:.1f}%")

        # Overall analysis
        analysis = results["statistical_analysis"]
        print(f"\n📈 Overall Statistics:")
        print(f"   • Total unique kernels: {analysis['total_kernels']}")
        print(f"   • Kernels with configs: {analysis['successful_extractions']}")
        print(f"   • Extraction success rate: {analysis['extraction_success_rate']:.1%}")

        # Kernel type distribution
        print(f"\n🏷️  Kernel Type Distribution:")
        for kernel_type, count in analysis["kernel_type_distribution"].items():
            percentage = (count / analysis['total_kernels']) * 100
            print(f"   • {kernel_type}: {count} kernels ({percentage:.1f}%)")

        # Analysis by kernel type
        print(f"\n📋 Analysis by Kernel Type:")
        for kernel_type, type_info in analysis["by_kernel_type"].items():
            print(f"\n  🔸 {kernel_type.upper()} Kernels:")
            print(f"     • Total kernels: {type_info['total_kernels']}")
            print(f"     • Kernels with configs: {type_info['kernels_with_configs']}")
            print(f"     • Unique configurations: {type_info['unique_configurations']}")
            print(f"     • Configuration diversity: {type_info['config_diversity']:.2f}")

            # Show most common configurations
            if type_info['most_common_configs']:
                print(f"     • Most common configs:")
                for i, (config_str, count) in enumerate(list(type_info['most_common_configs'].items())[:3], 1):
                    # Parse config string back for display
                    try:
                        config_dict = eval(config_str)
                        config_display = ", ".join(f"{k}={v}" for k, v in config_dict.items())
                    except:
                        config_display = config_str[:60] + "..." if len(config_str) > 60 else config_str

                    percentage = (count / type_info['kernels_with_configs']) * 100
                    print(f"       {i}. {config_display} → {count} times ({percentage:.1f}%)")

            # Show parameter frequencies
            if type_info['parameter_frequencies']:
                print(f"     • Parameter frequencies:")
                for param, freq_dict in list(type_info['parameter_frequencies'].items())[:3]:
                    most_common = list(freq_dict.items())[0]
                    print(f"       - {param}: {most_common[0]} ({most_common[1]} times)")

def create_kernel_type_classifier(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a detailed classification analysis of kernel types and their configurations
    """

    classifier_results = {
        "classification_rules": {
            "1d": "Kernels with xnumel and XBLOCK parameters (pointwise operations)",
            "2d": "Kernels with xnumel, ynumel, XBLOCK, and YBLOCK parameters (2D operations)",
            "reduction": "Kernels with tunable R0_BLOCK parameters (traditional reductions)",
            "reduction_fixed": "Kernels with r0_numel but fixed/predefined R0_BLOCK (per-element reductions)",
            "per_element": "Final-stage reduction kernels (triton_per_* without R0_BLOCK)",
            "unknown": "Kernels that don't match standard patterns"
        },
        "type_signatures": {},
        "configuration_patterns": {},
        "recommendations": {}
    }

    analysis = results.get("statistical_analysis", {})

    for kernel_type, type_info in analysis.get("by_kernel_type", {}).items():
        # Extract parameter signatures
        param_frequencies = type_info.get("parameter_frequencies", {})

        classifier_results["type_signatures"][kernel_type] = {
            "common_parameters": list(param_frequencies.keys()),
            "total_kernels": type_info.get("total_kernels", 0),
            "config_diversity": type_info.get("config_diversity", 0)
        }

        # Analyze configuration patterns
        most_common_configs = type_info.get("most_common_configs", {})
        if most_common_configs:
            # Get the most frequent configuration
            most_frequent_config_str = list(most_common_configs.keys())[0]
            most_frequent_count = list(most_common_configs.values())[0]

            try:
                most_frequent_config = eval(most_frequent_config_str)

                classifier_results["configuration_patterns"][kernel_type] = {
                    "most_frequent_config": most_frequent_config,
                    "frequency": most_frequent_count,
                    "percentage": (most_frequent_count / type_info.get("kernels_with_configs", 1)) * 100,
                    "parameter_analysis": param_frequencies
                }

            except Exception as e:
                classifier_results["configuration_patterns"][kernel_type] = {
                    "error": f"Could not parse config: {e}",
                    "raw_config": most_frequent_config_str
                }

        # Generate recommendations
        config_diversity = type_info.get("config_diversity", 0)
        total_kernels = type_info.get("total_kernels", 0)

        if config_diversity < 0.3 and total_kernels > 5:
            recommendation = f"Low diversity ({config_diversity:.2f}) - consider using common config as default"
        elif config_diversity > 0.8:
            recommendation = f"High diversity ({config_diversity:.2f}) - may need more sophisticated auto-tuning"
        else:
            recommendation = f"Moderate diversity ({config_diversity:.2f}) - current approach seems reasonable"

        classifier_results["recommendations"][kernel_type] = recommendation

    return classifier_results

def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Analyze default configurations in Triton kernels")
    parser.add_argument("--data-dir", default="../data_64",
                       help="Directory containing triton_modules subdirectory")
    parser.add_argument("--output", default="default_config_analysis.json",
                       help="Output file for results")
    parser.add_argument("--classifier-output", default="kernel_type_classifier.json",
                       help="Output file for kernel type classification")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create analyzer
    analyzer = DefaultConfigAnalyzer(args.data_dir)

    # Run analysis
    print("🚀 Starting default configuration extraction and analysis...")
    results = analyzer.analyze_all_kernels()

    if results and results.get("extraction_stats", {}).get("kernels_found", 0) > 0:
        # Print summary
        analyzer.print_analysis_summary(results)

        # Save main results
        analyzer.save_results(results, args.output)

        # Create and save kernel type classification
        classifier_results = create_kernel_type_classifier(results)

        classifier_path = analyzer.output_dir / args.classifier_output
        with open(classifier_path, 'w') as f:
            json.dump(classifier_results, f, indent=2, default=str)

        print(f"🏷️  Kernel type classification saved to: {classifier_path}")

        print(f"\n✅ Analysis complete!")

        # Additional insights
        if args.verbose:
            print(f"\n🔬 Additional Insights:")

            analysis = results["statistical_analysis"]

            # Check for patterns
            total_kernels = analysis.get("total_kernels", 0)
            successful_extractions = analysis.get("successful_extractions", 0)

            if successful_extractions / max(total_kernels, 1) < 0.7:
                print(f"   • Low extraction success rate ({successful_extractions/max(total_kernels, 1):.1%})")
                print(f"   • Consider improving the extraction algorithm")

            # Analyze kernel type balance
            type_dist = analysis.get("kernel_type_distribution", {})
            if type_dist.get("unknown", 0) > total_kernels * 0.2:
                print(f"   • High number of unknown kernel types ({type_dist.get('unknown', 0)})")
                print(f"   • May need to extend kernel type detection logic")

            # Check configuration diversity
            for kernel_type, type_info in analysis.get("by_kernel_type", {}).items():
                diversity = type_info.get("config_diversity", 0)
                if diversity < 0.2:
                    print(f"   • {kernel_type} kernels have very low config diversity ({diversity:.2f})")
                    print(f"   • Consider using a standard default configuration")

    else:
        print("❌ No kernels found or analysis failed")
        return 1

    return 0


def test():
	src = read_text(Path("../data_64/triton_modules/PermEqui2_mean_triton.py"))
	kernel_name = "triton_per_fused_mean_0"
	numel_param_names, numel_values, other_param_names = infer_numel_for_kernel(src, kernel_name)
	print("Numel Params:", numel_param_names)
	print("Numel Values:", numel_values)
	print("Other Params:", other_param_names)
	if numel_param_names and other_param_names:
		param_names = numel_param_names + other_param_names
		kernel_type = detect_kernel_type(param_names, kernel_name)
	else:
		kernel_type = "unknown"
	print("Detected Kernel Type:", kernel_type)

if __name__ == "__main__":
    sys.exit(main())