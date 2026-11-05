#!/usr/bin/env python3
"""
Configuration Pattern Visualizer

Creates visualizations and detailed analysis of Triton configuration patterns
from the frequency analysis results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

class ConfigPatternVisualizer:
    """Visualizes configuration patterns from frequency analysis"""

    def __init__(self, analysis_file: str = "config_frequency_analysis.json"):
        self.analysis_file = Path(analysis_file)
        self.data = {}
        self.load_data()

    def load_data(self):
        """Load analysis data from JSON file"""
        if self.analysis_file.exists():
            with open(self.analysis_file, 'r') as f:
                self.data = json.load(f)
        else:
            print(f"❌ Analysis file not found: {self.analysis_file}")

    def create_parameter_distribution_plots(self):
        """Create distribution plots for each parameter"""

        configs = self.data.get("config_frequencies", {})
        if not configs:
            print("No configuration data found")
            return

        # Parse configuration parameters
        param_data = defaultdict(list)

        for config_str, frequency in configs.items():
            if config_str == "unknown_config":
                continue

            # Parse config string: "XBLOCK=256, num_stages=2, num_warps=4"
            for param in config_str.split(", "):
                if "=" in param:
                    key, value = param.split("=")
                    try:
                        param_data[key].extend([int(value)] * frequency)
                    except ValueError:
                        param_data[key].extend([value] * frequency)

        # Create subplots
        num_params = len(param_data)
        if num_params == 0:
            print("No parseable parameters found")
            return

        cols = 3
        rows = (num_params + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes if num_params > 1 else [axes]
        else:
            axes = axes.flatten()

        for i, (param, values) in enumerate(param_data.items()):
            ax = axes[i]

            if isinstance(values[0], int):
                # Numeric parameter - histogram
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.set_title(f'{param} Distribution')
                ax.set_xlabel(param)
                ax.set_ylabel('Frequency')
            else:
                # Categorical parameter - bar plot
                value_counts = Counter(values)
                ax.bar(value_counts.keys(), value_counts.values(), alpha=0.7)
                ax.set_title(f'{param} Distribution')
                ax.set_xlabel(param)
                ax.set_ylabel('Frequency')
                plt.setp(ax.get_xticklabels(), rotation=45)

        # Hide unused subplots
        for i in range(num_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('config_parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Parameter distribution plots saved as 'config_parameter_distributions.png'")

    def create_config_popularity_chart(self, top_n: int = 15):
        """Create a horizontal bar chart of most popular configurations"""

        configs = self.data.get("config_frequencies", {})
        if not configs:
            return

        # Get top N configurations
        top_configs = dict(list(configs.items())[:top_n])

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        y_pos = np.arange(len(top_configs))
        values = list(top_configs.values())
        labels = [self._shorten_config_label(config) for config in top_configs.keys()]

        bars = ax.barh(y_pos, values, alpha=0.8, color='skyblue', edgecolor='navy')

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{value}', va='center', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Frequency')
        ax.set_title(f'Top {top_n} Most Popular Configurations')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('config_popularity_chart.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Configuration popularity chart saved as 'config_popularity_chart.png'")

    def create_parameter_correlation_heatmap(self):
        """Create correlation heatmap between parameters"""

        configs = self.data.get("config_frequencies", {})
        if not configs:
            return

        # Build parameter matrix
        param_vectors = defaultdict(list)
        config_names = []

        for config_str, frequency in configs.items():
            if config_str == "unknown_config":
                continue

            config_names.append(self._shorten_config_label(config_str))
            param_dict = {}

            for param in config_str.split(", "):
                if "=" in param:
                    key, value = param.split("=")
                    try:
                        param_dict[key] = int(value)
                    except ValueError:
                        # Handle categorical values
                        param_dict[key] = hash(value) % 100

            # Ensure all parameters have values
            for param in param_vectors.keys():
                if param not in param_dict:
                    param_dict[param] = 0

            for param, value in param_dict.items():
                param_vectors[param].append(value)

        # Create DataFrame
        df = pd.DataFrame(dict(param_vectors), index=config_names[:len(list(param_vectors.values())[0])])

        if df.empty or len(df.columns) < 2:
            print("Not enough numeric parameters for correlation analysis")
            return

        # Calculate correlation matrix
        correlation_matrix = df.corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig('parameter_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Parameter correlation heatmap saved as 'parameter_correlation_heatmap.png'")

    def analyze_kernel_type_patterns(self):
        """Analyze configuration patterns by kernel type"""

        kernel_analysis = self.data.get("kernel_analysis", {})
        if not kernel_analysis:
            return

        # Group kernels by type
        kernel_types = defaultdict(list)

        for kernel_name, analysis in kernel_analysis.items():
            # Extract kernel type from name
            if "poi_fused" in kernel_name:
                kernel_type = "pointwise"
            elif "red_fused" in kernel_name:
                kernel_type = "reduction"
            elif "per_fused" in kernel_name:
                kernel_type = "persistent"
            else:
                kernel_type = "other"

            kernel_types[kernel_type].append((kernel_name, analysis))

        # Analyze patterns per type
        print("\n🔍 Kernel Type Configuration Analysis:")
        print("=" * 50)

        for kernel_type, kernels in kernel_types.items():
            print(f"\n📊 {kernel_type.upper()} Kernels ({len(kernels)} kernels)")

            # Collect configurations for this type
            type_configs = Counter()
            total_configs = 0

            for kernel_name, analysis in kernels:
                for config, count in analysis["most_common_configs"].items():
                    type_configs[config] += count
                    total_configs += count

            # Show top configurations for this type
            print(f"   Total configurations: {total_configs}")
            print(f"   Top configurations:")

            for i, (config, count) in enumerate(type_configs.most_common(3), 1):
                percentage = (count / total_configs) * 100 if total_configs > 0 else 0
                print(f"      {i}. {config} → {count} ({percentage:.1f}%)")

    def generate_optimization_insights(self):
        """Generate optimization insights from the analysis"""

        print("\n💡 OPTIMIZATION INSIGHTS")
        print("=" * 40)

        stats = self.data.get("summary_stats", {})
        configs = self.data.get("config_frequencies", {})
        best_configs = self.data.get("best_config_frequencies", {})

        # Configuration diversity analysis
        if stats.get("config_distribution_entropy", 0) > 4:
            print("\n🎯 High Configuration Diversity Detected")
            print("   • Many different configurations are being used")
            print("   • Consider automated hyperparameter optimization")
            print("   • Look for parameter space reduction opportunities")

        # Dominant configuration analysis
        if configs:
            top_config, top_count = list(configs.items())[0]
            popularity = (top_count / stats.get("total_configurations", 1)) * 100

            if popularity > 15:
                print(f"\n🔥 Dominant Configuration Found ({popularity:.1f}%)")
                print(f"   • Configuration: {top_config}")
                print(f"   • This might be near-optimal for many kernels")
                print(f"   • Consider using as default configuration")

        # Performance consistency analysis
        if best_configs and configs:
            consistent_performers = []
            for config in best_configs:
                if config in configs:
                    performance_ratio = best_configs[config] / configs[config]
                    if performance_ratio > 0.8:  # Best in >80% of appearances
                        consistent_performers.append((config, performance_ratio))

            if consistent_performers:
                print(f"\n⚡ Consistently High-Performing Configurations:")
                for config, ratio in sorted(consistent_performers, key=lambda x: x[1], reverse=True)[:3]:
                    print(f"   • {config} (best in {ratio:.1%} of uses)")

    def _shorten_config_label(self, config_str: str, max_length: int = 40) -> str:
        """Shorten configuration labels for display"""
        if len(config_str) <= max_length:
            return config_str

        # Try to abbreviate parameter names
        abbreviated = config_str.replace("num_stages", "stages")
        abbreviated = abbreviated.replace("num_warps", "warps")
        abbreviated = abbreviated.replace("XBLOCK", "X")
        abbreviated = abbreviated.replace("R0_BLOCK", "R0")

        if len(abbreviated) <= max_length:
            return abbreviated

        return config_str[:max_length-3] + "..."

def main():
    """Main function"""

    print("📊 Configuration Pattern Visualization")
    print("=" * 40)

    # Check if analysis file exists
    if not Path("config_frequency_analysis.json").exists():
        print("❌ Analysis file not found. Please run analyze_config_frequency.py first.")
        return

    # Create visualizer
    visualizer = ConfigPatternVisualizer()

    if not visualizer.data:
        print("❌ No data loaded")
        return

    # Generate visualizations
    try:
        print("\n📈 Creating parameter distribution plots...")
        visualizer.create_parameter_distribution_plots()

        print("\n📊 Creating configuration popularity chart...")
        visualizer.create_config_popularity_chart()

        print("\n🔥 Creating parameter correlation heatmap...")
        visualizer.create_parameter_correlation_heatmap()

        print("\n🔍 Analyzing kernel type patterns...")
        visualizer.analyze_kernel_type_patterns()

        print("\n💡 Generating optimization insights...")
        visualizer.generate_optimization_insights()

        print(f"\n✅ Visualization complete! Check the generated PNG files.")

    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Install with: pip install matplotlib seaborn pandas")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")

if __name__ == "__main__":
    main()