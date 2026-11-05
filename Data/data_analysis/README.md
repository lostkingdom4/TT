# Triton Kernel Configuration Analysis

This directory contains tools for analyzing default configurations extracted from Triton kernels in the `data_64` dataset.

## Quick Start

```bash
# Run the full analysis
python default_config_analyzer.py --verbose

# View summary of results
python summarize_results.py

# Run with custom data directory
python default_config_analyzer.py --data-dir ../other_data --output my_analysis.json
```

## Generated Files

- **`default_config_analysis.json`** - Complete analysis results including all extracted configurations
- **`kernel_type_classifier.json`** - Kernel type classification patterns and recommendations
- **Analysis summary** - Printed to console by `summarize_results.py`

## Key Findings

From the analysis of **19,020 Triton kernels** with **83.7% successful extraction rate**:

### 1D Kernels (50.7% of total)
- **Recommended default:** `XBLOCK=1024, num_warps=4, num_stages=1`
- Used by 35.1% of 1D kernels
- Very low configuration diversity (0.002)

### Reduction Kernels (40.5% of total)
- **Recommended default:** `XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1`
- Used by 37.3% of reduction kernels
- Very low configuration diversity (0.004)

### 2D Kernels (8.7% of total)
- **Recommended default:** `XBLOCK=32, YBLOCK=32, num_warps=4, num_stages=1`
- Used by 35.3% of 2D kernels
- Low configuration diversity (0.020)

## Usage in Auto-tuning

These default configurations can be used as:
1. **Starting points** for auto-tuning searches
2. **Fallback configurations** when extraction fails
3. **Baseline comparisons** for performance evaluation
4. **Default values** in kernel generation pipelines

## Scripts Overview

- **`default_config_analyzer.py`** - Main analysis engine that processes all Triton files
- **`run_default_analysis.py`** - Simple wrapper for running the analyzer
- **`summarize_results.py`** - Generates concise summary reports
- **`analyze_config_frequency.py`** - Legacy frequency analysis tool

## Implementation Details

The analyzer uses the `extract_default_config()` function from `tune_one_file.py` to:

1. Parse Triton kernel source code
2. Identify `tl.constexpr` parameters
3. Extract actual values from kernel invocations
4. Map parameters to their configured values
5. Classify kernels by type (1D, 2D, reduction)
6. Compute statistical patterns across all kernels

## Configuration Extraction Process

```python
# Example of extracted configuration
{
    "XBLOCK": 1024,
    "num_warps": 4,
    "num_stages": 1
}
```

The extraction handles:
- **Positional arguments** in kernel calls
- **Keyword arguments** (`num_warps=4`)
- **Multiple kernel invocations** per file
- **Different kernel types** with varying parameter patterns

## Error Handling

Common reasons for extraction failures (16.3% of kernels):
- Missing `tl.constexpr` annotations
- Complex parameter passing patterns
- Non-standard kernel invocation formats
- Missing or malformed kernel calls

## Future Enhancements

- Support for more kernel types (e.g., matrix multiplication)
- Performance correlation analysis
- Auto-tuning integration
- Configuration recommendation engine