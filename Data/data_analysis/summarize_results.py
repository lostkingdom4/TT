#!/usr/bin/env python3
"""
Configuration Pattern Summarizer

Generate concise summaries of default configuration analysis results.
"""

import json
import sys
from pathlib import Path

def load_results(analysis_file, classifier_file):
    """Load analysis results from JSON files"""
    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)

    with open(classifier_file, 'r') as f:
        classifier_data = json.load(f)

    return analysis_data, classifier_data

def print_executive_summary(analysis_data, classifier_data):
    """Print a concise executive summary"""

    stats = analysis_data.get("statistical_analysis", {})

    print("="*60)
    print("📋 TRITON KERNEL CONFIGURATION SUMMARY")
    print("="*60)

    # High-level metrics
    total_kernels = stats.get("total_kernels", 0)
    success_rate = stats.get("extraction_success_rate", 0)

    print(f"\n📊 Key Metrics:")
    print(f"   • Total Triton kernels analyzed: {total_kernels:,}")
    print(f"   • Configuration extraction success: {success_rate:.1%}")
    print(f"   • Kernel types identified: {len(stats.get('by_kernel_type', {}))}")

    # Dominant patterns
    print(f"\n🎯 Dominant Configuration Patterns:")

    type_analysis = stats.get("by_kernel_type", {})

    for kernel_type, data in type_analysis.items():
        if data.get("kernels_with_configs", 0) == 0:
            continue

        most_common = data.get("most_common_configs", {})
        if not most_common:
            continue

        # Get the most frequent config
        top_config = list(most_common.keys())[0]
        top_count = list(most_common.values())[0]
        total_with_configs = data.get("kernels_with_configs", 1)
        percentage = (top_count / total_with_configs) * 100

        print(f"\n   🔸 {kernel_type.upper()} Kernels ({data.get('total_kernels', 0):,} total):")

        # Parse and display the config nicely
        try:
            config_dict = eval(top_config)
            config_parts = []

            # Order parameters logically
            param_order = ['XBLOCK', 'YBLOCK', 'ZBLOCK', 'R0_BLOCK', 'num_warps', 'num_stages']

            for param in param_order:
                if param in config_dict:
                    config_parts.append(f"{param}={config_dict[param]}")

            # Add any remaining parameters
            for param, value in config_dict.items():
                if param not in param_order:
                    config_parts.append(f"{param}={value}")

            config_str = ", ".join(config_parts)
            print(f"      Most common: {config_str}")
            print(f"      Usage: {top_count:,} kernels ({percentage:.1f}%)")

        except:
            print(f"      Most common: {top_config[:60]}...")
            print(f"      Usage: {top_count} kernels ({percentage:.1f}%)")

    # Recommendations
    print(f"\n💡 Key Recommendations:")

    recommendations = classifier_data.get("recommendations", {})
    for kernel_type, rec in recommendations.items():
        if "Low diversity" in rec and "common config as default" in rec:
            # Extract the suggested config
            type_data = type_analysis.get(kernel_type, {})
            if type_data.get("most_common_configs"):
                print(f"   • {kernel_type.upper()}: Use standardized default configuration")

    # Configuration space insights
    print(f"\n🔍 Configuration Space Analysis:")
    for kernel_type, data in type_analysis.items():
        diversity = data.get("config_diversity", 0)
        if diversity < 0.1:
            print(f"   • {kernel_type.upper()}: Very low diversity ({diversity:.3f}) - single config dominates")
        elif diversity > 0.5:
            print(f"   • {kernel_type.upper()}: High diversity ({diversity:.3f}) - needs careful tuning")

def print_recommended_defaults(analysis_data):
    """Print recommended default configurations for each kernel type"""

    print(f"\n" + "="*60)
    print("🛠️  RECOMMENDED DEFAULT CONFIGURATIONS")
    print("="*60)

    type_analysis = analysis_data.get("statistical_analysis", {}).get("by_kernel_type", {})

    for kernel_type, data in type_analysis.items():
        most_common = data.get("most_common_configs", {})
        if not most_common:
            continue

        print(f"\n🔸 {kernel_type.upper()} Kernels:")

        # Show top 3 configurations
        configs = list(most_common.items())[:3]
        for i, (config_str, count) in enumerate(configs, 1):
            try:
                config_dict = eval(config_str)

                # Format configuration nicely
                parts = []
                for param in ['XBLOCK', 'YBLOCK', 'ZBLOCK', 'R0_BLOCK', 'num_warps', 'num_stages']:
                    if param in config_dict:
                        parts.append(f"{param}={config_dict[param]}")

                config_display = ", ".join(parts)
                percentage = (count / data.get("kernels_with_configs", 1)) * 100

                if i == 1:
                    print(f"   ✅ PRIMARY: {config_display} ({percentage:.1f}%)")
                else:
                    print(f"   {i}. ALTERNATIVE: {config_display} ({percentage:.1f}%)")

            except:
                print(f"   {i}. {config_str[:50]}... ({count} uses)")

def main():
    """Main function"""

    # Default file paths
    analysis_file = Path(__file__).parent / "default_config_analysis.json"
    classifier_file = Path(__file__).parent / "kernel_type_classifier.json"

    # Check if files exist
    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        print("   Run default_config_analyzer.py first")
        return 1

    if not classifier_file.exists():
        print(f"❌ Classifier file not found: {classifier_file}")
        print("   Run default_config_analyzer.py first")
        return 1

    try:
        # Load data
        analysis_data, classifier_data = load_results(analysis_file, classifier_file)

        # Print summaries
        print_executive_summary(analysis_data, classifier_data)
        print_recommended_defaults(analysis_data)

        print(f"\n" + "="*60)
        print("📄 For detailed analysis, see:")
        print(f"   • {analysis_file}")
        print(f"   • {classifier_file}")
        print("="*60)

        return 0

    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())