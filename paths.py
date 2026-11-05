"""Centralized path definitions for the project.

All file system paths that depend on the project layout are defined here.
Other modules import from this module rather than computing paths via
``__file__`` traversals.
"""

import os
from pathlib import Path

# Project root is the directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent

# Tritonbench root: either from environment variable or default to tb/.
# The env var allows overriding for CI or development setups where
# tritonbench is installed elsewhere.
TRITONBENCH_ROOT = Path(
    os.environ.get("TRITONBENCH_ROOT", str(PROJECT_ROOT / "tb"))
).resolve()

# Tritonbench Python package directory (contains operators/, utils/, etc.)
TRITONBENCH_PACKAGE_DIR = TRITONBENCH_ROOT / "tritonbench"

# Metadata directory within tritonbench
METADATA_DIR = TRITONBENCH_PACKAGE_DIR / "metadata"

# The oss_cuda_kernels.yaml file
OSS_CUDA_KERNELS_YAML = METADATA_DIR / "oss_cuda_kernels.yaml"

# Tritonbench's run.py entry point
TRITONBENCH_RUN_PY = TRITONBENCH_ROOT / "run.py"
