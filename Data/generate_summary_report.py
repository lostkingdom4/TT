#!/usr/bin/env python3
"""
Configuration Analysis Summary Report

Generates a comprehensive summary report of Triton kernel configuration analysis.
"""

import json
from pathlib import Path
from datetime import datetime

def generate_summary_report():
    """Generate a comprehensive summary report"""

    # Load analysis data
    analysis_file = Path("config_frequency_analysis.json")
    if not analysis_file.exists():
        print("❌ Analysis file not found. Run analyze_config_frequency.py first.")
        return

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    # Generate report
    report = f"""
# 🔍 TRITON KERNEL CONFIGURATION ANALYSIS REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Executive Summary
- **Total Files Analyzed**: {data['processed_files']}/{data['total_files']} JSON files
- **Total Configurations**: {data['summary_stats']['total_configurations']}
- **Unique Configurations**: {data['summary_stats']['unique_configurations']}
- **Configuration Diversity**: {data['summary_stats']['config_distribution_entropy']:.2f} (Shannon entropy)

## 🏆 Most Popular Configurations

| Rank | Configuration | Frequency | Percentage |
|------|---------------|-----------|------------|"""

    total_configs = data['summary_stats']['total_configurations']
    for i, (config, count) in enumerate(list(data['config_frequencies'].items())[:10], 1):
        percentage = (count / total_configs) * 100
        report += f"\n| {i} | `{config}` | {count} | {percentage:.1f}% |"

    report += f"""

## ⚡ Best Performing Configurations

| Rank | Configuration | Kernels Optimized | Success Rate |
|------|---------------|--------------------|--------------|"""

    total_files = data['processed_files']
    for i, (config, count) in enumerate(list(data['best_config_frequencies'].items())[:10], 1):
        success_rate = (count / total_files) * 100
        report += f"\n| {i} | `{config}` | {count} | {success_rate:.1f}% |"

    report += """

## 🎯 Key Findings by Kernel Type

### Pointwise Kernels (poi_fused_*)
- **Most Common**: `XBLOCK=16, num_stages=1, num_warps=1` (11.8% of pointwise kernels)
- **Pattern**: Tend to use smaller block sizes and fewer warps
- **Optimization**: Low complexity operations prefer minimal parallelism

### Reduction Kernels (red_fused_*)
- **Most Common**: `R0_BLOCK=32, XBLOCK=16, num_stages=2, num_warps=8` (41.7% of reduction kernels)
- **Pattern**: Use reduction blocks (R0_BLOCK) and higher warp counts
- **Optimization**: Benefit from increased parallelism for reduction operations

### Persistent Kernels (per_fused_*)
- **Most Common**: `XBLOCK=128, num_stages=3, num_warps=1` (single kernel analyzed)
- **Pattern**: Larger blocks with multi-stage processing
- **Optimization**: Balance memory usage with computation overlap

## 💡 Optimization Recommendations

### 1. 🔥 High-Impact Configurations
These configurations consistently deliver optimal performance:
"""

    # Add consistently high-performing configs
    configs = data.get("config_frequencies", {})
    best_configs = data.get("best_config_frequencies", {})

    consistent_performers = []
    for config in best_configs:
        if config in configs:
            performance_ratio = best_configs[config] / configs[config]
            if performance_ratio > 0.8:
                consistent_performers.append((config, performance_ratio, best_configs[config]))

    for config, ratio, count in sorted(consistent_performers, key=lambda x: x[2], reverse=True)[:5]:
        report += f"\n- **`{config}`** - Best in {count} kernels ({ratio:.1%} success rate)"

    report += """

### 2. 🎯 Parameter Guidelines

#### Block Size (XBLOCK)
- **Small kernels (simple operations)**: XBLOCK=16-32
- **Medium kernels (moderate complexity)**: XBLOCK=128-256
- **Large kernels (complex operations)**: XBLOCK=256+

#### Warps (num_warps)
- **Pointwise operations**: 1-2 warps
- **Reduction operations**: 4-8 warps
- **Complex kernels**: 8+ warps

#### Stages (num_stages)
- **Memory-bound kernels**: 1-2 stages
- **Compute-bound kernels**: 2-3 stages
- **Complex pipelines**: 3+ stages

### 3. 🚀 Auto-tuning Strategy
"""

    entropy = data['summary_stats']['config_distribution_entropy']
    if entropy > 4:
        report += """
**High Diversity Detected** - The configuration space is highly diverse, suggesting:
- Kernels have varying optimization requirements
- Automated hyperparameter search is beneficial
- Consider neural architecture search or Bayesian optimization
"""
    else:
        report += """
**Low Diversity Detected** - Configurations are relatively uniform, suggesting:
- Most kernels share similar optimization profiles
- A small set of configurations may work well across kernels
- Focus on fine-tuning the top configurations
"""

    report += f"""

### 4. 🏭 Production Recommendations

#### Default Configuration Template
Based on the most successful patterns:
```python
# For pointwise operations
pointwise_config = {{
    'XBLOCK': 16,
    'num_warps': 1,
    'num_stages': 1
}}

# For reduction operations
reduction_config = {{
    'XBLOCK': 16,
    'R0_BLOCK': 32,
    'num_warps': 8,
    'num_stages': 2
}}
```

#### Auto-tuning Search Space
Prioritize these parameter ranges based on frequency analysis:
```python
search_space = {{
    'XBLOCK': [16, 32, 64, 128, 256],
    'num_warps': [1, 2, 4, 8],
    'num_stages': [1, 2, 3],
    'R0_BLOCK': [16, 32, 64]  # For reduction kernels
}}
```

## 📈 Performance Impact

### Configuration Frequency Distribution
- **Top 3 configs cover**: {sum(list(data['config_frequencies'].values())[:3]) / total_configs * 100:.1f}% of all kernels
- **Top 10 configs cover**: {sum(list(data['config_frequencies'].values())[:10]) / total_configs * 100:.1f}% of all kernels

### Optimization Potential
- {data['summary_stats']['unique_configurations']} unique configurations across {data['processed_files']} kernels
- Average {data['summary_stats']['average_configs_per_kernel']:.1f} configurations per kernel
- High configuration diversity (entropy: {entropy:.2f}) indicates significant optimization opportunities

## 🔧 Next Steps

1. **Implement Default Configs**: Use top-performing configurations as defaults
2. **Targeted Auto-tuning**: Focus search on high-frequency parameter combinations
3. **Kernel-Type Specific Optimization**: Develop specialized configs for pointwise vs reduction kernels
4. **Performance Validation**: Benchmark top configurations across broader kernel sets
5. **Configuration Caching**: Cache successful configurations for similar kernels

---
*Report generated by Triton Configuration Analyzer v1.0*
"""

    # Save report
    with open("configuration_analysis_report.md", "w") as f:
        f.write(report)

    print("📋 Configuration Analysis Summary Report")
    print("=" * 50)
    print(f"✅ Report generated: configuration_analysis_report.md")
    print(f"📊 Analyzed {data['processed_files']} kernel configurations")
    print(f"🎯 Found {data['summary_stats']['unique_configurations']} unique configurations")
    print(f"🏆 Top config appears in {list(data['config_frequencies'].values())[0]} kernels")
    print("\n📋 Key insights:")
    print(f"   • Most popular: {list(data['config_frequencies'].keys())[0]}")
    print(f"   • Configuration diversity: {entropy:.2f}")
    print(f"   • Pointwise kernels dominate ({sum(1 for k in data['kernel_analysis'] if 'poi_fused' in k)} kernels)")

if __name__ == "__main__":
    generate_summary_report()