#!/usr/bin/env python3
"""
Configuration Frequency Analyzer for Performance Results

This script analyzes all JSON files in the performance directory to identify
the frequency of each Triton kernel configuration across all tuning results.
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional
import argparse

class ConfigFrequencyAnalyzer:
    """Analyzes configuration frequency across performance JSON files"""

    def __init__(self, performance_dir: str = "performance"):
        self.performance_dir = Path(performance_dir)
        self.config_counter = Counter()
        self.kernel_configs = defaultdict(list)
        self.best_configs = Counter()
        self.files_processed = 0

    def analyze_all_files(self) -> Dict[str, Any]:
        """Analyze all JSON files in the performance directory"""

        if not self.performance_dir.exists():
            print(f"❌ Performance directory not found: {self.performance_dir}")
            return {}

        json_files = list(self.performance_dir.glob("*.json"))
        print(f"📁 Found {len(json_files)} JSON files in {self.performance_dir}")

        if not json_files:
            print("⚠️  No JSON files found")
            return {}

        results = {
            "total_files": len(json_files),
            "processed_files": 0,
            "config_frequencies": {},
            "best_config_frequencies": {},
            "kernel_analysis": {},
            "summary_stats": {}
        }

        for json_file in json_files:
            try:
                self._analyze_single_file(json_file)
                self.files_processed += 1
                print(f"✅ Processed: {json_file.name}")
            except Exception as e:
                print(f"❌ Error processing {json_file.name}: {e}")

        results["processed_files"] = self.files_processed
        results["config_frequencies"] = dict(self.config_counter.most_common())
        results["best_config_frequencies"] = dict(self.best_configs.most_common())
        results["kernel_analysis"] = self._generate_kernel_analysis()
        results["summary_stats"] = self._generate_summary_stats()

        return results

    def _analyze_single_file(self, json_file: Path) -> None:
        """Analyze a single JSON performance file"""

        with open(json_file, 'r') as f:
            data = json.load(f)

        kernel_name = json_file.stem.replace('_tune', '')

        # Handle the specific format from auto-tuning results
        if isinstance(data, dict):
            # Check for the auto-tuning result format
            if 'best' in data and 'cfg' in data['best']:
                # This is the auto-tuning result format
                best_cfg = data['best']['cfg']
                best_time = data['best'].get('lat_ms', 0)

                # Extract best configuration
                config_key = self._extract_config_key_from_cfg(best_cfg)
                if config_key:
                    self.config_counter[config_key] += 1
                    self.kernel_configs[kernel_name].append(config_key)
                    self.best_configs[config_key] += 1

                # Also process all attempted configurations if available
                if 'all_results' in data:
                    for result in data['all_results']:
                        if isinstance(result, dict) and 'cfg' in result:
                            cfg_key = self._extract_config_key_from_cfg(result['cfg'])
                            if cfg_key:
                                self.config_counter[cfg_key] += 1
                                self.kernel_configs[kernel_name].append(cfg_key)

                return

            # Handle other possible formats
            if 'configurations' in data:
                configs = data['configurations']
            elif 'results' in data:
                configs = data['results']
            elif 'configs' in data:
                configs = data['configs']
            else:
                # Try to extract from the dict itself
                configs = [data]
        elif isinstance(data, list):
            configs = data
        else:
            print(f"⚠️  Unknown data structure in {json_file.name}")
            return

        # Fallback processing for other formats
        best_config = None
        best_time = float('inf')

        for config in configs:
            if not isinstance(config, dict):
                continue

            # Extract configuration parameters
            config_key = self._extract_config_key(config)
            if config_key:
                self.config_counter[config_key] += 1
                self.kernel_configs[kernel_name].append(config_key)

                # Track best configuration (lowest time)
                time_value = self._extract_time_value(config)
                if time_value is not None and time_value < best_time:
                    best_time = time_value
                    best_config = config_key

        # Record best configuration for this kernel
        if best_config:
            self.best_configs[best_config] += 1

    def _extract_config_key(self, config: Dict[str, Any]) -> str:
        """Extract a standardized configuration key from config data"""

        # Common parameter names to look for
        param_keys = [
            'BLOCK_SIZE', 'XBLOCK', 'YBLOCK', 'ZBLOCK',
            'num_warps', 'num_stages', 'num_ctas',
            'BLOCK_M', 'BLOCK_N', 'BLOCK_K',
            'GROUP_M', 'SPLIT_K'
        ]

        config_parts = []

        # Look for parameters in different possible locations
        config_dict = config
        if 'config' in config:
            config_dict = config['config']
        elif 'kwargs' in config:
            config_dict = config['kwargs']
        elif 'meta' in config:
            config_dict = config['meta']

        for key in param_keys:
            if key in config_dict:
                config_parts.append(f"{key}={config_dict[key]}")

        # Also check for triton.Config style parameters
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if key.upper() in [k.upper() for k in param_keys]:
                    config_parts.append(f"{key}={value}")

        return ", ".join(sorted(config_parts)) if config_parts else "unknown_config"

    def _extract_config_key_from_cfg(self, cfg: Dict[str, Any]) -> str:
        """Extract configuration key from auto-tuning cfg format"""

        config_parts = []

        # Common parameters in cfg
        param_keys = [
            'XBLOCK', 'YBLOCK', 'ZBLOCK', 'BLOCK_SIZE',
            'num_warps', 'num_stages', 'num_ctas',
            'BLOCK_M', 'BLOCK_N', 'BLOCK_K',
            'GROUP_M', 'SPLIT_K'
        ]

        for key in param_keys:
            if key in cfg:
                config_parts.append(f"{key}={cfg[key]}")

        # Add any other parameters not in the standard list
        for key, value in cfg.items():
            if key not in param_keys and key != 'grid':  # Skip grid as it's derived
                config_parts.append(f"{key}={value}")

        return ", ".join(sorted(config_parts)) if config_parts else "unknown_config"

    def _extract_time_value(self, config: Dict[str, Any]) -> Optional[float]:
        """Extract timing value from configuration"""

        # Common time field names
        time_fields = ['time', 'latency', 'duration', 'runtime', 'ms', 'us', 'ns']

        for field in time_fields:
            if field in config:
                try:
                    return float(config[field])
                except (ValueError, TypeError):
                    continue

        # Look in nested dictionaries
        for key, value in config.items():
            if isinstance(value, dict):
                for time_field in time_fields:
                    if time_field in value:
                        try:
                            return float(value[time_field])
                        except (ValueError, TypeError):
                            continue

        return None

    def _generate_kernel_analysis(self) -> Dict[str, Any]:
        """Generate per-kernel configuration analysis"""

        analysis = {}

        for kernel_name, configs in self.kernel_configs.items():
            config_counter = Counter(configs)
            analysis[kernel_name] = {
                "total_configs": len(configs),
                "unique_configs": len(config_counter),
                "most_common_configs": dict(config_counter.most_common(5)),
                "config_diversity": len(config_counter) / max(len(configs), 1)
            }

        return analysis

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""

        total_configs = sum(self.config_counter.values())
        unique_configs = len(self.config_counter)

        return {
            "total_configurations": total_configs,
            "unique_configurations": unique_configs,
            "average_configs_per_kernel": total_configs / max(self.files_processed, 1),
            "most_popular_config": self.config_counter.most_common(1)[0] if self.config_counter else None,
            "config_distribution_entropy": self._calculate_entropy(self.config_counter),
            "kernels_analyzed": len(self.kernel_configs)
        }

    def _calculate_entropy(self, counter: Counter) -> float:
        """Calculate Shannon entropy of configuration distribution"""

        import math

        total = sum(counter.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability)

        return entropy

    def print_analysis(self, results: Dict[str, Any]) -> None:
        """Print formatted analysis results"""

        print("\n" + "="*60)
        print("🔍 TRITON CONFIGURATION FREQUENCY ANALYSIS")
        print("="*60)

        # Summary
        stats = results["summary_stats"]
        print(f"\n📊 Summary:")
        print(f"   • Files processed: {results['processed_files']}/{results['total_files']}")
        print(f"   • Total configurations: {stats['total_configurations']}")
        print(f"   • Unique configurations: {stats['unique_configurations']}")
        print(f"   • Average configs per kernel: {stats['average_configs_per_kernel']:.1f}")
        print(f"   • Configuration diversity (entropy): {stats['config_distribution_entropy']:.2f}")

        # Most popular configurations
        print(f"\n🏆 Top 10 Most Frequent Configurations:")
        for i, (config, count) in enumerate(list(results["config_frequencies"].items())[:10], 1):
            percentage = (count / stats['total_configurations']) * 100
            print(f"   {i:2d}. {config} → {count} times ({percentage:.1f}%)")

        # Best performing configurations
        print(f"\n⚡ Top 10 Best Performing Configurations:")
        for i, (config, count) in enumerate(list(results["best_config_frequencies"].items())[:10], 1):
            percentage = (count / results['processed_files']) * 100
            print(f"   {i:2d}. {config} → Best in {count} kernels ({percentage:.1f}%)")

        # Per-kernel analysis
        print(f"\n📈 Per-Kernel Analysis (Top 5):")
        kernel_analysis = results["kernel_analysis"]
        sorted_kernels = sorted(kernel_analysis.items(),
                              key=lambda x: x[1]["total_configs"], reverse=True)

        for i, (kernel, analysis) in enumerate(sorted_kernels[:5], 1):
            print(f"   {i}. {kernel}")
            print(f"      • Total configs: {analysis['total_configs']}")
            print(f"      • Unique configs: {analysis['unique_configs']}")
            print(f"      • Diversity: {analysis['config_diversity']:.2f}")

            # Show top config for this kernel
            top_config = list(analysis["most_common_configs"].items())[0]
            print(f"      • Most common: {top_config[0]} ({top_config[1]} times)")

    def save_results(self, results: Dict[str, Any], output_file: str = "config_frequency_analysis.json") -> None:
        """Save results to JSON file"""

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Analyze Triton configuration frequencies")
    parser.add_argument("--performance-dir", default="performance",
                       help="Directory containing performance JSON files")
    parser.add_argument("--output", default="config_frequency_analysis.json",
                       help="Output file for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ConfigFrequencyAnalyzer(args.performance_dir)

    # Run analysis
    print("🚀 Starting configuration frequency analysis...")
    results = analyzer.analyze_all_files()

    if results:
        # Print results
        analyzer.print_analysis(results)

        # Save results
        analyzer.save_results(results, args.output)

        print(f"\n✅ Analysis complete!")

        # Additional insights
        if args.verbose:
            print(f"\n🔬 Additional Insights:")
            stats = results["summary_stats"]

            if stats["unique_configurations"] > 100:
                print(f"   • High configuration diversity detected ({stats['unique_configurations']} unique configs)")
                print(f"   • Consider configuration space reduction techniques")

            if stats["most_popular_config"]:
                config, count = stats["most_popular_config"]
                popularity = (count / stats["total_configurations"]) * 100
                if popularity > 20:
                    print(f"   • Dominant configuration found: {config} ({popularity:.1f}%)")
                    print(f"   • This config might be near-optimal for many kernels")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()