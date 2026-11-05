"""
Benchmark a Triton kernel with both original and LLM-recommended configs,
then report the comparison.

Reads the output from multiagent_triton_runner.py and the captured kernel
shapes to reconstruct inputs, run benchmarks, and measure speedup.
Iterates over all config pairs in the output directory, resolving the
shape_path from each output JSON to find the corresponding shapes file.

Usage:
    # Per-pair shape resolution (shape_path from each output JSON):
    python compare_configs.py \
        --output-dir output/fp16_softmax_fwd_3inputs \
        --kernel-module ../../tritonbench/operators/softmax/operator.py \
        --kernel-name softmax_kernel \
        --kernel-class Operator

    # Global shapes file override (all pairs use the same shapes):
    python compare_configs.py \
        --output-dir output/fp16_softmax_fwd \
        --shapes-file extract/ir_output/softmax/softmax_softmax_kernel_shapes.json \
        --kernel-module ../../tritonbench/operators/softmax/operator.py \
        --kernel-name softmax_kernel \
        --kernel-class Operator \
        --output-json comparison_results.json
"""

import argparse
import contextlib
import importlib.util
import json
import logging
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.testing

# Triton 3.6 removed AttrsDescriptor from triton.compiler.compiler.
# Captured inductor kernels (from older Triton) import it at module level.
# Provide a shim so those imports succeed.
try:
    from triton.compiler.compiler import AttrsDescriptor as _AD  # noqa: F401
except ImportError:
    import triton.compiler.compiler as _tcc
    import hashlib as _hashlib

    class _AttrsDescriptorShim:
        """Minimal shim for AttrsDescriptor removed in Triton 3.6."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @classmethod
        def from_dict(cls, d):
            return cls(**d)

        def __hash__(self):
            return int(_hashlib.md5(str(sorted(self.__dict__.items())).encode()).hexdigest(), 16)

        def __eq__(self, other):
            return isinstance(other, _AttrsDescriptorShim) and self.__dict__ == other.__dict__

    _tcc.AttrsDescriptor = _AttrsDescriptorShim

# Allow importing sibling modules
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_utils import device_sync
from gpu_lock import get_gpu_lock
from tensor_utils import create_kernel_inputs, load_tensor_shapes

logger = logging.getLogger("compare_configs")


# ---------------------------------------------------------------------------
# Block size name mapping
# ---------------------------------------------------------------------------

# Alias table: maps a generic LLM name to known alternative names that real
# kernels may use.  When the LLM produces a generic name but the kernel uses
# one of its aliases, we can resolve the mapping without a fragile positional
# fallback.
BLOCK_NAME_ALIASES: Dict[str, List[str]] = {
    "XBLOCK": ["BLOCK_SIZE", "BLOCK_SIZE_X", "BLOCK_M", "BLOCK_X"],
    "YBLOCK": ["BLOCK_SIZE_Y", "BLOCK_N", "BLOCK_Y"],
    "ZBLOCK": ["BLOCK_SIZE_Z", "BLOCK_K", "BLOCK_Z"],
}


def _map_block_sizes(
    recommended_block_sizes: Dict[str, int],
    shapes_kwargs: Dict[str, Any],
) -> Dict[str, int]:
    """Map LLM block size names to actual kernel parameter names.

    The LLM may produce generic names (XBLOCK, YBLOCK, ZBLOCK) from the TTGIR
    extraction layer, but the kernel may use different constexpr parameter names
    (e.g. BLOCK_SIZE for softmax).  The shapes JSON kwargs field contains the
    *actual* names used by the kernel at capture time.

    Resolution order:
    1. Direct match -- LLM keys already match kernel params.
    2. Alias match -- LLM key has a known alias among the kernel params.
    3. Positional fallback -- map by canonical ordering.

    Args:
        recommended_block_sizes: e.g. {"XBLOCK": 128} or {"BLOCK_SIZE": 256}
        shapes_kwargs: Raw kwargs dict from the shapes JSON, where each value
            is {"type": ..., "value": ...}.

    Returns:
        Dict mapping actual kernel param names to recommended values.
    """
    if not recommended_block_sizes:
        return {}

    # Extract actual block-like params from shapes kwargs
    actual_block_params = {
        k: v["value"]
        for k, v in shapes_kwargs.items()
        if "BLOCK" in k.upper()
    }

    if not actual_block_params:
        # No block params in shapes -- pass through as-is and hope for the best
        logger.warning(
            "No BLOCK params found in shapes kwargs; using LLM names directly: %s",
            recommended_block_sizes,
        )
        return dict(recommended_block_sizes)

    # 1) Direct match: LLM keys are a subset of (or equal to) actual keys
    if set(recommended_block_sizes.keys()).issubset(set(actual_block_params.keys())):
        return dict(recommended_block_sizes)

    # 2) Alias match: try to resolve each LLM key via BLOCK_NAME_ALIASES
    actual_set = set(actual_block_params.keys())
    alias_mapped: Dict[str, int] = {}
    unresolved: Dict[str, int] = {}

    for llm_key, value in recommended_block_sizes.items():
        if llm_key in actual_set:
            # Already a direct match
            alias_mapped[llm_key] = value
        else:
            # Check aliases
            aliases = BLOCK_NAME_ALIASES.get(llm_key, [])
            matched_alias = None
            for alias in aliases:
                if alias in actual_set and alias not in alias_mapped:
                    matched_alias = alias
                    break
            if matched_alias:
                alias_mapped[matched_alias] = value
                logger.warning(
                    "Mapped LLM block size %s=%d -> kernel param %s (alias match)",
                    llm_key, value, matched_alias,
                )
            else:
                unresolved[llm_key] = value

    # If alias resolution handled everything, return
    if not unresolved:
        return alias_mapped

    # 3) Positional mapping fallback for remaining unresolved keys
    llm_order = ["XBLOCK", "YBLOCK", "ZBLOCK"] + [f"R{i}_BLOCK" for i in range(10)]

    # Remaining actual params not yet mapped
    remaining_actual = sorted(k for k in actual_block_params if k not in alias_mapped)

    unresolved_sorted = sorted(
        unresolved.keys(),
        key=lambda k: llm_order.index(k) if k in llm_order else 999,
    )

    for i, llm_key in enumerate(unresolved_sorted):
        if i < len(remaining_actual):
            actual_key = remaining_actual[i]
            alias_mapped[actual_key] = unresolved[llm_key]
            logger.warning(
                "Mapped LLM block size %s=%d -> kernel param %s (positional fallback)",
                llm_key,
                unresolved[llm_key],
                actual_key,
            )
        else:
            logger.warning(
                "LLM recommended %s=%d but no matching kernel param found; skipping",
                llm_key,
                unresolved[llm_key],
            )

    return alias_mapped


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_multiagent_configs(
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Load all multiagent output JSONs and extract config pairs.

    Returns a list of dicts, each with keys:
        source_file, original_config, recommended_config, shape_path
    """
    results: List[Dict[str, Any]] = []
    seen_shapes: set = set()

    for jf in sorted(output_dir.glob("*.json")):
        if jf.name == "summary.json":
            continue
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", jf, exc)
            continue

        if not data.get("success"):
            logger.info("Skipping %s (success=false)", jf.name)
            continue

        inner = data.get("result", {})
        if isinstance(inner, dict):
            inner = inner.get("result", inner)

        original = inner.get("original_config")
        recommended = inner.get("recommended_config")
        if not original or not recommended:
            logger.warning("Skipping %s: missing original/recommended config", jf.name)
            continue

        shape_path = data.get("shape_path", "")

        # Deduplicate by shape_path
        if shape_path in seen_shapes:
            logger.info("Skipping duplicate shape_path %s in %s", shape_path, jf.name)
            continue
        seen_shapes.add(shape_path)

        results.append(
            {
                "source_file": str(jf),
                "original_config": original,
                "recommended_config": recommended,
                "shape_path": shape_path,
            }
        )

    return results


def _resolve_shape_path(
    shape_path_str: str,
    script_dir: Path,
) -> Optional[Path]:
    """Resolve a shape_path from an output JSON to an absolute file path.

    The shape_path stored in multiagent output JSONs is relative to SCRIPT_DIR
    (e.g. 'extract/ir_output/softmax/softmax_Operator.softmax_kernel_input0_shapes.json').

    Returns the resolved Path if the file exists, or None if it does not.
    """
    if not shape_path_str:
        return None
    p = Path(shape_path_str)
    if not p.is_absolute():
        p = (script_dir / p).resolve()
    if p.is_file():
        return p
    logger.warning("Shape file does not exist: %s", p)
    return None


def _build_kwargs(
    config: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    shapes_kwargs: Dict[str, Any],
    skip_block_sizes: bool = False,
) -> Dict[str, Any]:
    """Build kernel kwargs by overlaying a config on top of captured kwargs.

    Args:
        skip_block_sizes: If True, skip block-size mapping (used for template
            kernels where block sizes are handled via source-level patching).
    """
    kw = dict(base_kwargs)
    # Remove captured keys that are not kernel parameters
    kw.pop("warmup", None)

    # Override num_warps / num_stages
    if "num_warps" in config:
        kw["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        kw["num_stages"] = config["num_stages"]

    # Map and apply block sizes (skip for template kernels — handled via source patching)
    if not skip_block_sizes:
        mapped = _map_block_sizes(config.get("block_size", {}), shapes_kwargs)
        kw.update(mapped)

    return kw


def _filter_kwargs_for_kernel(kernel_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove kwargs that the JITFunction doesn't accept as parameters.

    Some kernels have block sizes hardcoded (not constexpr), so passing
    LLM-recommended block-size kwargs like XBLOCK/YBLOCK causes
    ``Keyword argument was specified but unrecognised``.  This filters
    them out while keeping valid params like num_warps, num_stages, etc.
    """
    try:
        from triton.runtime.jit import JITFunction
        if not isinstance(kernel_fn, JITFunction):
            return kwargs

        valid_names = {p.name for p in kernel_fn.params}
        # num_warps, num_stages, num_ctas are always accepted by the launcher
        always_valid = {"num_warps", "num_stages", "num_ctas"}
        allowed = valid_names | always_valid

        filtered = {}
        dropped = []
        for k, v in kwargs.items():
            if k in allowed:
                filtered[k] = v
            else:
                dropped.append(k)

        if dropped:
            logger.info(
                "Dropped unrecognised kwargs for kernel: %s",
                ", ".join(sorted(dropped)),
            )
        return filtered
    except ImportError:
        return kwargs


# ---------------------------------------------------------------------------
# Source-level patching for template kernels
# ---------------------------------------------------------------------------

import hashlib
import math
import re
import tempfile


def _is_template_kernel(source: str) -> bool:
    """Check if kernel source uses ``@triton_heuristics.template`` decorator."""
    return "triton_heuristics.template(" in source


def _extract_hardcoded_constexprs(source: str) -> Dict[str, int]:
    """Extract ``BLOCK_X : tl.constexpr = VALUE`` from function body.

    Returns dict like ``{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}``.
    Only returns block-size-like params (containing ``BLOCK``).
    """
    pattern = r"(\w*BLOCK\w*)\s*:\s*tl\.constexpr\s*=\s*(\d+)"
    return {m.group(1): int(m.group(2)) for m in re.finditer(pattern, source)}


def _patch_kernel_source(source: str, block_sizes: Dict[str, int]) -> str:
    """Replace ``BLOCK_X : tl.constexpr = OLD`` with new values in *source*.

    Only patches params present in *block_sizes* dict.  Returns modified source.
    """
    result = source
    for name, value in block_sizes.items():
        pattern = rf"({re.escape(name)}\s*:\s*tl\.constexpr\s*=\s*)\d+"
        result = re.sub(pattern, rf"\g<1>{value}", result)
    return result


def _load_patched_kernel(
    module_path: Path,
    kernel_name: str,
    block_sizes: Dict[str, int],
    kernel_class: Optional[str] = None,
):
    """Load kernel with patched block sizes via temp-file reimport."""
    source = module_path.read_text()
    patched = _patch_kernel_source(source, block_sizes)

    if patched == source:
        return _load_kernel(module_path, kernel_name, kernel_class)

    config_hash = hashlib.md5(
        str(sorted(block_sizes.items())).encode()
    ).hexdigest()[:8]
    temp_path = (
        Path(tempfile.gettempdir())
        / f"{module_path.stem}_patched_{config_hash}.py"
    )
    temp_path.write_text(patched)

    logger.info(
        "Wrote patched kernel to %s (block_sizes=%s)", temp_path, block_sizes
    )
    return _load_kernel(temp_path, kernel_name, kernel_class)


_TEMPLATE_BLOCK_ALIASES = {
    "XBLOCK": "BLOCK_M",
    "YBLOCK": "BLOCK_N",
    "ZBLOCK": "BLOCK_K",
    "BLOCK_SIZE": "BLOCK_M",
    "BLOCK_SIZE_M": "BLOCK_M",
    "BLOCK_SIZE_N": "BLOCK_N",
    "BLOCK_SIZE_K": "BLOCK_K",
}


def _map_template_block_sizes(
    llm_blocks: Dict[str, int],
    kernel_constexprs: Dict[str, int],
) -> Dict[str, int]:
    """Map LLM block-size names to template kernel's actual constexpr names.

    Returns a copy of *kernel_constexprs* with values overridden by
    *llm_blocks* where a name match or alias match is found.
    """
    if not llm_blocks:
        return dict(kernel_constexprs)

    result = dict(kernel_constexprs)
    for llm_name, value in llm_blocks.items():
        if llm_name in kernel_constexprs:
            result[llm_name] = value
        elif llm_name in _TEMPLATE_BLOCK_ALIASES:
            target = _TEMPLATE_BLOCK_ALIASES[llm_name]
            if target in kernel_constexprs:
                result[target] = value
    return result


def _compute_template_grid(
    source: str,
    block_sizes: Dict[str, int],
    original_grid: Tuple[int, ...],
    original_block_sizes: Dict[str, int],
) -> Tuple[int, ...]:
    """Recompute the launch grid for a template kernel with new block sizes.

    For matmul templates: extracts hardcoded ``M``, ``N`` from source and
    computes ``ceil(M/BLOCK_M) * ceil(N/BLOCK_N)``.  Falls back to
    proportional scaling of the original grid.
    """
    # Try to extract M and N from source (pattern: M = 20211)
    dims: Dict[str, int] = {}
    for var in ("M", "N"):
        m = re.search(rf"^\s*{var}\s*=\s*(\d+)", source, re.MULTILINE)
        if m:
            dims[var] = int(m.group(1))

    # Matmul template: grid = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
    if (
        "M" in dims
        and "N" in dims
        and "BLOCK_M" in block_sizes
        and "BLOCK_N" in block_sizes
    ):
        grid_m = math.ceil(dims["M"] / block_sizes["BLOCK_M"])
        grid_n = math.ceil(dims["N"] / block_sizes["BLOCK_N"])
        new_grid = (grid_m * grid_n, 1, 1)
        logger.info(
            "Template grid: M=%d N=%d BLOCK_M=%d BLOCK_N=%d → grid=%s",
            dims["M"],
            dims["N"],
            block_sizes["BLOCK_M"],
            block_sizes["BLOCK_N"],
            new_grid,
        )
        return new_grid

    # Fallback: scale original grid proportionally
    if original_block_sizes and block_sizes:
        ratio = 1.0
        for key in block_sizes:
            if key in original_block_sizes and original_block_sizes[key] > 0:
                ratio *= original_block_sizes[key] / block_sizes[key]
        new_total = max(1, int(math.ceil(original_grid[0] * ratio)))
        return (new_total,) + original_grid[1:]

    return original_grid


# ---------------------------------------------------------------------------
# Kernel loading
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _add_path(path: str):
    """Temporarily add *path* to the front of ``sys.path``."""
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _get_submodule_path(module_path: Path) -> Optional[Path]:
    """Return the submodule root if *module_path* lives under ``submodules/<name>/``.

    Tritonbench submodules reside in ``<repo>/submodules/<name>/``.  When a
    kernel file is inside a submodule, the submodule root must be on
    ``sys.path`` so that the kernel's top-level imports can resolve.
    """
    parts = module_path.parts
    for i, part in enumerate(parts):
        if part == "submodules" and i + 1 < len(parts):
            submod_root = Path(*parts[: i + 2])
            if submod_root.is_dir():
                return submod_root
    return None


def _load_kernel(
    module_path: Path,
    kernel_name: str,
    kernel_class: Optional[str] = None,
):
    """Dynamically import a Triton JIT kernel function.

    Tritonbench operator modules must be imported through the package path
    (tritonbench.operators.<op>.operator) because the @register_benchmark
    decorator inspects the module name.  We try the package import first,
    falling back to importlib for standalone kernel files.

    Args:
        module_path: Path to the Python file containing the kernel.
        kernel_name: Name of the @triton.jit function.
        kernel_class: If the kernel is a class method (e.g. Operator.softmax_kernel),
            pass the class name here.

    Returns:
        The JITFunction object that can be called as ``kernel_fn[grid](*args, **kw)``.
    """
    module_path = module_path.resolve()

    # Try to derive a tritonbench package-style module name from the file path
    # e.g. .../tritonbench/operators/softmax/operator.py -> tritonbench.operators.softmax.operator
    parts = module_path.parts
    mod = None
    for i, part in enumerate(parts):
        if part == "tritonbench" and i + 2 < len(parts) and parts[i + 1] == "operators":
            dotted = ".".join(parts[i:]).removesuffix(".py")
            try:
                mod = importlib.import_module(dotted)
                logger.info("Imported kernel module via package: %s", dotted)
                break
            except ImportError as exc:
                logger.debug("Package import failed (%s), trying importlib: %s", dotted, exc)

    # Fallback: importlib from file path
    if mod is None:
        spec = importlib.util.spec_from_file_location("_kernel_module", str(module_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        mod = importlib.util.module_from_spec(spec)

        # If the kernel lives inside a submodule, temporarily add the
        # submodule root to sys.path so top-level imports can resolve
        # (e.g. ``from generative_recommenders.common import ...``).
        submod_root = _get_submodule_path(module_path)
        if submod_root is not None:
            logger.info("Adding submodule root to sys.path for import: %s", submod_root)
            with _add_path(str(submod_root)):
                spec.loader.exec_module(mod)
        else:
            spec.loader.exec_module(mod)
        logger.info("Imported kernel module via file: %s", module_path)

    if kernel_class:
        cls = getattr(mod, kernel_class)
        return getattr(cls, kernel_name)
    return getattr(mod, kernel_name)


def _is_inductor_autotuner(kernel_fn) -> bool:
    """Check if a kernel is a PyTorch Inductor CachingAutotuner."""
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        return isinstance(kernel_fn, CachingAutotuner)
    except ImportError:
        return False


def _unwrap_autotuner(kernel_fn):
    """Unwrap Triton Autotuner layers so we can pass explicit constexpr kwargs.

    The Autotuner's ``_bench()`` method raises *Conflicting meta-parameters* if
    kwargs contain keys that overlap with its managed configs (BLOCK_M, etc.).
    For direct benchmarking with explicit config values we bypass the Autotuner
    and call the underlying JITFunction (or Heuristics wrapper) directly.

    Heuristics and other KernelInterface subclasses are preserved because they
    do not reject explicit kwargs.

    Note: Inductor's CachingAutotuner is NOT unwrapped because it manages its
    own dispatching.  Instead, callers pass ``stream=`` when invoking it.

    Returns:
        ``(unwrapped_kernel, autotuner_config_keys)`` where
        *autotuner_config_keys* is a set of param names that the Autotuner
        manages (empty set if the kernel was not autotuned).
    """
    # Unwrap inductor's CachingAutotuner to its underlying JITFunction.
    # CachingAutotuner.__getitem__ injects warmup=False which the compiled
    # launcher does not accept, so we must bypass it entirely.
    if _is_inductor_autotuner(kernel_fn):
        autotuner_keys: set = set()
        for config in kernel_fn.configs:
            autotuner_keys.update(config.kwargs.keys())
        logger.info(
            "Unwrapping Inductor CachingAutotuner (managed params: %s)",
            ", ".join(sorted(autotuner_keys)) or "(none)",
        )
        return kernel_fn.fn, autotuner_keys

    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return kernel_fn, set()

    autotuner_keys: set = set()
    while isinstance(kernel_fn, Autotuner):
        for config in kernel_fn.configs:
            autotuner_keys.update(config.kwargs.keys())
        logger.info(
            "Unwrapping Autotuner layer (managed params: %s)",
            ", ".join(sorted(autotuner_keys)) or "(none)",
        )
        kernel_fn = kernel_fn.fn

    return kernel_fn, autotuner_keys


def _evaluate_grid_formula(
    metadata: Dict[str, Any],
    kwargs: Dict[str, Any],
) -> Optional[Tuple[int, ...]]:
    """Evaluate a callable grid formula with the given kwargs (block sizes).

    The shapes JSON may store the grid as a callable::

        {"type": "callable",
         "source": "grid = lambda meta: (triton.cdiv(M, meta[\\"BLOCK_M\\"]), ...)"}

    We reconstruct a namespace from scalar args (M, N, K) and kwargs
    (BLOCK_M, BLOCK_N, etc.) then eval the lambda to get concrete grid dims.

    Returns a tuple of ints or ``None`` if evaluation fails.
    """
    import math

    raw = metadata.get("grid")
    if not isinstance(raw, dict) or raw.get("type") != "callable":
        return None

    source = raw.get("source", "")
    if not source or "lambda" not in source:
        return None

    # Build meta dict — use provided kwargs (which contain the actual block sizes)
    meta: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if not isinstance(v, torch.Tensor):
            meta[k] = v

    # Build namespace with scalar arg values (closure variables like M, N, K)
    namespace: Dict[str, Any] = dict(meta)
    for sa in metadata.get("scalar_args", []):
        namespace[f"arg{sa['arg_index']}"] = sa["value"]

    # Infer common dimension names from tensor shapes
    tensor_args = metadata.get("tensor_args", [])
    if len(tensor_args) >= 2:
        shapes = [t["shape"] for t in tensor_args]
        if len(shapes[0]) == 2 and len(shapes[1]) == 2:
            namespace.setdefault("M", shapes[0][0])
            namespace.setdefault("N", shapes[1][1])
            namespace.setdefault("K", shapes[0][1])
    if len(tensor_args) >= 1:
        first_shape = tensor_args[0]["shape"]
        if len(first_shape) == 1:
            namespace.setdefault("N", first_shape[0])
            # Common alias used in elementwise kernels
            namespace.setdefault("n_elements", first_shape[0])
        # For any shape, total numel is a common grid variable
        numel = 1
        for d in first_shape:
            numel *= d
        namespace.setdefault("n_elements", numel)

    # triton.cdiv shim
    class _triton:
        @staticmethod
        def cdiv(a, b):
            return math.ceil(a / b)

    namespace["triton"] = _triton()
    namespace["math"] = math

    try:
        idx = source.find("lambda")
        if idx < 0:
            return None
        lambda_src = source[idx:].strip()
        # Strip inline comments
        lines = lambda_src.split("\n")
        cleaned = []
        for line in lines:
            comment_idx = line.find("#")
            if comment_idx >= 0:
                line = line[:comment_idx]
            cleaned.append(line)
        lambda_src = "\n".join(cleaned).rstrip()

        eval_globals = {"__builtins__": {}}
        eval_globals.update(namespace)
        grid_fn = eval(lambda_src, eval_globals)  # noqa: S307
        result = grid_fn(meta)
        if isinstance(result, (tuple, list)):
            return tuple(int(v) for v in result)
        return (int(result),)
    except Exception as exc:
        logger.warning("Failed to evaluate grid formula: %s", exc)
        return None


def _resolve_grid(
    metadata: Dict[str, Any],
    original_config: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[int, ...]:
    """Resolve the kernel launch grid from shapes metadata.

    Shapes JSON stores the grid in one of two formats:

    - Static list: ``[629, 8]`` — used directly.
    - Callable dict: ``{"type": "callable", "source": "lambda meta: ..."}`` —
      evaluated using *kwargs* (which contain the actual block sizes for this
      benchmark run).  Falls back to the pre-evaluated grid in
      *original_config* if evaluation fails.

    When *kwargs* is provided, the callable grid is re-evaluated with the
    actual block sizes so the grid matches the config being benchmarked.
    """
    raw = metadata.get("grid")

    # Static grid — most common for simple kernels.
    # If XBLOCK changed, recompute grid.x proportionally (grid.x = xnumel / XBLOCK).
    if isinstance(raw, (list, tuple)):
        grid = list(int(v) for v in raw)
        if kwargs is not None and len(grid) >= 1:
            # Check if XBLOCK changed vs original
            orig_bs = original_config.get("block_size", {})
            for bk in ("XBLOCK", "BLOCK_M"):
                new_val = kwargs.get(bk)
                orig_val = orig_bs.get(bk)
                if new_val and orig_val and new_val != orig_val:
                    # Recompute grid.x: original_grid.x * original_XBLOCK / new_XBLOCK
                    new_grid_x = int(grid[0] * orig_val / new_val)
                    if new_grid_x != grid[0]:
                        logger.info(
                            "Recomputing grid.x for %s change: %d -> %d (grid %d -> %d)",
                            bk, orig_val, new_val, grid[0], new_grid_x,
                        )
                        grid[0] = new_grid_x
                    break
        return tuple(grid)

    # Callable grid — try to evaluate with the actual kwargs
    if kwargs is not None:
        evaluated = _evaluate_grid_formula(metadata, kwargs)
        if evaluated is not None:
            logger.info("Evaluated callable grid with kwargs: %s", evaluated)
            return evaluated

    # Fallback: use the pre-evaluated grid from original_config
    cfg_grid = original_config.get("grid", {})
    if isinstance(cfg_grid, dict) and "x" in cfg_grid:
        dims = [int(cfg_grid["x"])]
        y = int(cfg_grid.get("y", 1))
        z = int(cfg_grid.get("z", 1))
        if z > 1:
            dims.extend([y, z])
        elif y > 1:
            dims.append(y)
        logger.info("Resolved callable grid from original_config: %s", tuple(dims))
        return tuple(dims)

    if isinstance(cfg_grid, (list, tuple)):
        return tuple(int(v) for v in cfg_grid)

    raise ValueError(
        f"Cannot resolve grid: shapes metadata has {type(raw).__name__}, "
        f"original_config grid={cfg_grid}"
    )


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _compute_stats(times: list) -> Dict[str, float]:
    """Compute basic statistics from a list of timing measurements (ms)."""
    if not times:
        return {}
    s = sorted(times)
    n = len(s)
    return {
        "median": s[n // 2],
        "mean": statistics.mean(s),
        "min": s[0],
        "max": s[-1],
        "std_dev": statistics.stdev(s) if n > 1 else 0.0,
        "samples": n,
    }


_BENCHMARK_FAILURE_LOG = Path("benchmark_failures.log")


def _log_benchmark_failure(label: str, exc: Exception, grid, kwargs):
    """Append benchmark failure details to a persistent log file."""
    import datetime
    try:
        with open(_BENCHMARK_FAILURE_LOG, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"[{datetime.datetime.now().isoformat()}] Benchmark FAILED ({label})\n")
            f.write(f"  Error: {exc}\n")
            f.write(f"  Grid:  {grid}\n")
            # Log config params (num_warps, num_stages, block sizes)
            config_params = {k: v for k, v in kwargs.items()
                            if k in ('num_warps', 'num_stages', 'XBLOCK', 'RBLOCK',
                                     'YBLOCK', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K',
                                     'BLOCK_SIZE', 'shared_memory_bytes')}
            f.write(f"  Config: {config_params}\n")
            if 'CUDA error' in str(exc) or 'illegal memory' in str(exc):
                f.write(f"  LIKELY CAUSE: shapes/kernel mismatch or grid not recomputed for new block size\n")
        print(f"    [BENCHMARK FAILURE] {label}: {exc}", flush=True)
    except Exception:
        pass  # never let logging break the benchmark flow


def _flush_l2_cache(device: str = "cuda") -> None:
    """Flush the GPU L2 cache by reading a large tensor.

    This ensures each benchmark starts from the same cold-cache state,
    eliminating measurement bias from stale cache contents left by
    previous kernels or other processes.  V100 has 6 MB L2; A100 has
    40 MB.  We use 48 MB to cover all architectures.
    """
    flush_size = 48 * 1024 * 1024 // 4  # 48 MB of float32
    try:
        buf = torch.empty(flush_size, dtype=torch.float32, device=device)
        buf.zero_()
        torch.cuda.synchronize()
        del buf
    except Exception:
        pass


def _benchmark_config(
    kernel_fn,
    grid: Tuple[int, ...],
    args: list,
    kwargs: Dict[str, Any],
    warmup: int,
    rep: int,
    label: str,
) -> Dict[str, float]:
    """Benchmark a single config and return stats dict."""
    logger.info("Benchmarking %s config (warmup=%d, rep=%d) ...", label, warmup, rep)

    # Capture args/kwargs in the closure explicitly to avoid late-binding issues
    _args = tuple(args)
    _kwargs = dict(kwargs)
    _grid = grid
    _fn = kernel_fn

    def kernel_call():
        _fn[_grid](*_args, **_kwargs)

    try:
        # Flush L2 cache and sync GPU so every config starts from the
        # same cold-cache state — prevents the first config (original)
        # from appearing slower due to stale cache from other processes.
        _flush_l2_cache()
        torch.cuda.synchronize()
        times = triton.testing.do_bench(kernel_call, warmup=warmup, rep=rep, return_mode="all")
    except Exception as exc:
        logger.error("Benchmark failed for %s: %s", label, exc)
        # Log failure details to a persistent file for diagnostics
        _log_benchmark_failure(label, exc, _grid, _kwargs)
        return {"error": str(exc)}

    stats = _compute_stats(times)
    logger.info(
        "  %s: median=%.4f ms  mean=%.4f ms  (n=%d)",
        label,
        stats.get("median", float("inf")),
        stats.get("mean", float("inf")),
        stats.get("samples", 0),
    )
    return stats


# ---------------------------------------------------------------------------
# Output correctness validation
# ---------------------------------------------------------------------------

def _clone_tensor_args(
    args: list,
    kwargs: Dict[str, Any],
) -> Tuple[list, Dict[str, Any]]:
    """Deep-clone all tensor arguments so each config run has independent buffers.

    Triton kernels write output tensors in-place.  Without cloning, the
    second config's outputs overwrite the first, making comparison impossible.
    """
    cloned_args = [
        a.clone() if isinstance(a, torch.Tensor) else a
        for a in args
    ]
    cloned_kwargs = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in kwargs.items()
    }
    return cloned_args, cloned_kwargs


# Tolerance defaults by dtype category: (rtol, atol)
_DTYPE_TOLERANCES: Dict[str, Tuple[float, float]] = {
    "fp16": (1e-2, 1e-2),
    "bf16": (1e-2, 1e-2),
    "fp32": (1e-4, 1e-4),
    "fp64": (1e-5, 1e-5),
    "int":  (0.0, 0.0),
    "bool": (0.0, 0.0),
}


def _get_dtype_tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    """Return ``(rtol, atol)`` appropriate for *dtype*.

    Different block sizes cause different FP accumulation orders, so
    half-precision outputs can legitimately differ more than full-precision.
    """
    if dtype == torch.float16:
        return _DTYPE_TOLERANCES["fp16"]
    if dtype == torch.bfloat16:
        return _DTYPE_TOLERANCES["bf16"]
    if dtype == torch.float32:
        return _DTYPE_TOLERANCES["fp32"]
    if dtype == torch.float64:
        return _DTYPE_TOLERANCES["fp64"]
    if dtype == torch.bool:
        return _DTYPE_TOLERANCES["bool"]
    # Integer types: exact match
    return _DTYPE_TOLERANCES["int"]


def _compare_tensors(
    orig: torch.Tensor,
    rec: torch.Tensor,
    name: str,
) -> Optional[Dict[str, Any]]:
    """Compare two tensors and return mismatch info, or ``None`` if they match.

    NaN handling: NaN in the same positions in both tensors is considered
    matching (both configs produce undefined results there).  NaN in one
    tensor but not the other is always flagged.
    """
    if orig.shape != rec.shape:
        return {
            "name": name,
            "reason": "shape_mismatch",
            "original_shape": list(orig.shape),
            "recommended_shape": list(rec.shape),
        }

    rtol, atol = _get_dtype_tolerances(orig.dtype)

    # Check for NaN divergence first (only meaningful for float types)
    if orig.is_floating_point():
        orig_nan = torch.isnan(orig)
        rec_nan = torch.isnan(rec)
        if orig_nan.any() or rec_nan.any():
            nan_mismatch = orig_nan != rec_nan
            if nan_mismatch.any():
                return {
                    "name": name,
                    "reason": "nan_mismatch",
                    "original_nan_count": int(orig_nan.sum().item()),
                    "recommended_nan_count": int(rec_nan.sum().item()),
                    "nan_mismatch_count": int(nan_mismatch.sum().item()),
                }
            # NaN positions match — compare only non-NaN elements
            mask = ~orig_nan
            if mask.any():
                orig_clean = orig[mask].float()
                rec_clean = rec[mask].float()
                if not torch.allclose(orig_clean, rec_clean, rtol=rtol, atol=atol):
                    diff = (orig_clean - rec_clean).abs()
                    return {
                        "name": name,
                        "reason": "value_mismatch",
                        "dtype": str(orig.dtype),
                        "rtol": rtol,
                        "atol": atol,
                        "max_abs_diff": float(diff.max().item()),
                        "mean_abs_diff": float(diff.mean().item()),
                        "num_mismatched": int(
                            (diff > atol + rtol * rec_clean.abs()).sum().item()
                        ),
                        "total_elements": int(mask.sum().item()),
                    }
            return None  # All NaN in both — trivially matching

    # Standard comparison (cast to float32 for the diff to avoid fp16 overflow)
    if not torch.allclose(orig, rec, rtol=rtol, atol=atol):
        diff = (orig.float() - rec.float()).abs()
        return {
            "name": name,
            "reason": "value_mismatch",
            "dtype": str(orig.dtype),
            "rtol": rtol,
            "atol": atol,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
            "num_mismatched": int(
                (diff > atol + rtol * rec.float().abs()).sum().item()
            ),
            "total_elements": int(orig.numel()),
        }

    return None


def _validate_outputs(
    kernel_fn,
    orig_grid: Tuple[int, ...],
    rec_grid: Tuple[int, ...],
    input_args: list,
    orig_kwargs: Dict[str, Any],
    rec_kwargs: Dict[str, Any],
    device: str = "cuda",
    rec_kernel_fn=None,
) -> Dict[str, Any]:
    """Run both configs once on cloned inputs and compare output tensors.

    Returns a dict with:
        ``valid``  – ``True`` if all tensors match within tolerance.
        ``mismatches`` – Per-tensor mismatch details (empty when valid).
        ``error``  – Error string if validation itself crashed.
        ``num_tensors_checked`` – How many tensor arguments were compared.

    Args:
        rec_kernel_fn: Optional separate kernel function for the recommended
            config (used for source-patched template kernels).
    """
    _rec_fn = rec_kernel_fn or kernel_fn

    # Clone inputs independently for each config
    orig_args, orig_kw = _clone_tensor_args(input_args, orig_kwargs)
    rec_args, rec_kw = _clone_tensor_args(input_args, rec_kwargs)

    # Run original config once
    try:
        kernel_fn[orig_grid](*orig_args, **orig_kw)
        device_sync(device)
    except Exception as exc:
        return {
            "valid": False,
            "mismatches": [],
            "error": f"original config execution failed: {exc}",
            "num_tensors_checked": 0,
        }

    # Run recommended config once
    try:
        _rec_fn[rec_grid](*rec_args, **rec_kw)
        device_sync(device)
    except Exception as exc:
        return {
            "valid": False,
            "mismatches": [],
            "error": f"recommended config execution failed: {exc}",
            "num_tensors_checked": 0,
        }

    # Compare all tensor arguments
    mismatches = []
    num_checked = 0

    for i, (orig_val, rec_val) in enumerate(zip(orig_args, rec_args)):
        if not isinstance(orig_val, torch.Tensor):
            continue
        num_checked += 1
        mismatch = _compare_tensors(orig_val, rec_val, f"arg[{i}]")
        if mismatch is not None:
            mismatches.append(mismatch)

    for key in orig_kw:
        orig_val = orig_kw[key]
        rec_val = rec_kw.get(key)
        if not isinstance(orig_val, torch.Tensor) or not isinstance(rec_val, torch.Tensor):
            continue
        num_checked += 1
        mismatch = _compare_tensors(orig_val, rec_val, f"kwarg[{key}]")
        if mismatch is not None:
            mismatches.append(mismatch)

    return {
        "valid": len(mismatches) == 0,
        "mismatches": mismatches,
        "error": None,
        "num_tensors_checked": num_checked,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_table(
    kernel_name: str,
    tensor_shape: list,
    tensor_dtype: str,
    grid: tuple,
    orig_config: dict,
    rec_config: dict,
    orig_stats: dict,
    rec_stats: dict,
    mapped_block_names: dict,
) -> str:
    """Format a human-readable comparison table."""
    lines = []
    sep = "=" * 72
    lines.append(sep)
    lines.append(f"CONFIG COMPARISON: {kernel_name}")
    lines.append(sep)
    lines.append(
        f"Shape: {tensor_shape} {tensor_dtype} | Grid: {grid}"
    )
    lines.append("")

    # Determine which block size columns to show
    block_cols = list(mapped_block_names.keys()) if mapped_block_names else []
    header_parts = ["Config".ljust(16), "num_warps".center(10)]
    for bc in block_cols:
        header_parts.append(bc.center(12))
    header_parts.extend(["Median (ms)".center(14), "Mean (ms)".center(14)])
    lines.append(" | ".join(header_parts))
    lines.append("-" * len(lines[-1]))

    def _row(label, config, stats, mapped_names):
        parts = [label.ljust(16)]
        parts.append(str(config.get("num_warps", "?")).center(10))
        bs = config.get("block_size", {})
        for bc in block_cols:
            # Try mapped name first, then direct
            val = mapped_names.get(bc, bs.get(bc, "?"))
            parts.append(str(val).center(12))
        if "error" in stats:
            parts.extend(["FAILED".center(14), "FAILED".center(14)])
        else:
            parts.append(f"{stats['median']:.4f}".center(14))
            parts.append(f"{stats['mean']:.4f}".center(14))
        return " | ".join(parts)

    # For original, the mapped block sizes are the original_config's block_size
    # mapped through _map_block_sizes
    orig_mapped = {k: orig_config.get("block_size", {}).get(k, "?") for k in block_cols}
    rec_mapped = {k: mapped_block_names.get(k, rec_config.get("block_size", {}).get(k, "?")) for k in block_cols}

    lines.append(_row("Original", orig_config, orig_stats, orig_mapped))
    lines.append(_row("Recommended", rec_config, rec_stats, rec_mapped))
    lines.append("")

    if "error" not in orig_stats and "error" not in rec_stats:
        speedup = orig_stats["median"] / rec_stats["median"] if rec_stats["median"] > 0 else float("inf")
        pct = (1 - rec_stats["median"] / orig_stats["median"]) * 100 if orig_stats["median"] > 0 else 0
        direction = "improvement" if pct > 0 else "regression"
        lines.append(f"Speedup: {speedup:.2f}x ({abs(pct):.1f}% {direction})")
    else:
        lines.append("Speedup: N/A (benchmark error)")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reusable benchmark entry point (used by closed-loop optimization)
# ---------------------------------------------------------------------------

def benchmark_config_pair(
    original_config: Dict[str, Any],
    recommended_config: Dict[str, Any],
    shapes_file: Path,
    kernel_module: Path,
    kernel_name: str,
    kernel_class: Optional[str] = None,
    device: str = "cuda",
    warmup: int = 5,
    rep: int = 20,
    seed: int = 42,
    validate_outputs: bool = True,
) -> Dict[str, Any]:
    """Benchmark original vs recommended config and return comparison stats.

    This is the primary entry point for closed-loop benchmark verification
    inside the multi-agent optimization loop.  It loads the kernel once,
    creates inputs from the shapes file, benchmarks both configs, and returns
    a result dict with timing statistics and speedup.

    Args:
        original_config: Original kernel config (num_warps, block_size, ...).
        recommended_config: LLM-recommended config to verify.
        shapes_file: Path to ``*_shapes.json`` with captured tensor metadata.
        kernel_module: Path to Python file containing the Triton kernel.
        kernel_name: Name of the ``@triton.jit`` kernel function.
        kernel_class: Class name if kernel is a class method (e.g. "Operator").
        device: Device for benchmarking ("cuda", "xpu", "cpu").
        warmup: Warmup iterations for ``triton.testing.do_bench``.
        rep: Repetitions for ``triton.testing.do_bench``.
        seed: Random seed for reproducible input generation.
        validate_outputs: When ``True`` (default), run both configs once on
            cloned inputs after benchmarking and verify that the recommended
            config produces outputs matching the original within dtype-aware
            tolerances.  Configs that fail validation are rejected as errors.

    Returns:
        Dict with keys: original_stats, recommended_stats, speedup,
        improvement_pct, original_config, recommended_config, grid,
        validation (correctness check result, ``None`` if skipped).
        On error, includes an "error" key with a description string.
    """
    shapes_file = Path(shapes_file)
    kernel_module = Path(kernel_module)

    # Load shapes metadata
    metadata = load_tensor_shapes(shapes_file)
    shapes_kwargs = metadata.get("kwargs", {})

    # Create inputs on CPU first; moved to GPU inside the benchmark lock
    # to avoid 32 parallel workers each holding GPU tensors while waiting.
    input_args, input_kwargs = create_kernel_inputs(
        shapes_file, device="cpu", values="random", seed=seed,
    )

    # Load kernel (unwrap Autotuner so explicit constexpr kwargs don't conflict)
    kernel_fn = _load_kernel(kernel_module, kernel_name, kernel_class)
    kernel_fn, _at_keys = _unwrap_autotuner(kernel_fn)

    # Source-level patching for kernels with hardcoded block sizes
    # (template kernels, inductor persistent/reduction/pointwise kernels)
    orig_kernel_fn = kernel_fn  # default: baked kernel for original
    rec_kernel_fn = kernel_fn   # default: same kernel for both configs
    _template_orig_grid = None
    _template_rec_grid = None
    _needs_source_patch = False
    orig_constexprs = {}
    mapped_orig = {}
    mapped_rec = {}

    try:
        source = kernel_module.read_text()
    except (OSError, IOError):
        source = ""

    # Detect hardcoded constexpr block params in source.
    # This covers template kernels (BLOCK_M/N/K), inductor reduction
    # kernels (XBLOCK/RBLOCK), persistent kernels, etc.
    orig_constexprs = _extract_hardcoded_constexprs(source) if source else {}
    _is_template = _is_template_kernel(source)
    _needs_source_patch = bool(orig_constexprs)

    if _needs_source_patch:
        # --- Patch the ORIGINAL config's kernel when needed ---
        # Inductor templates have block sizes baked as tl.constexpr; passing
        # num_warps via kwargs doesn't resize the tile.  So when the original
        # config's block_size differs from baked (e.g. BASELINE_USE_AUTOTUNE_FIRST
        # swaps in 32x64 while baked is 128x128), we must source-patch a second
        # kernel so block_size + num_warps stay mutually consistent — otherwise
        # the compiled kernel is invalid and silently writes NaN.
        orig_block_sizes = original_config.get("block_size", {})
        mapped_orig = _map_template_block_sizes(orig_block_sizes, orig_constexprs)

        if mapped_orig != orig_constexprs:
            logger.info(
                "Source patching ORIGINAL block sizes %s → %s",
                orig_constexprs,
                mapped_orig,
            )
            try:
                orig_kernel_fn = _load_patched_kernel(
                    kernel_module, kernel_name, mapped_orig, kernel_class,
                )
                orig_kernel_fn, _ = _unwrap_autotuner(orig_kernel_fn)
                if _is_template:
                    _base_grid_for_template = _resolve_grid(
                        metadata, original_config, {},
                    )
                    _template_orig_grid = _compute_template_grid(
                        source, mapped_orig, _base_grid_for_template, orig_constexprs,
                    )
            except Exception as exc:
                logger.warning(
                    "Original source patching failed, falling back to baked: %s",
                    exc,
                )
                orig_kernel_fn = kernel_fn

        # --- Patch the RECOMMENDED config's kernel when needed ---
        rec_block_sizes = recommended_config.get("block_size", {})
        mapped_rec = _map_template_block_sizes(rec_block_sizes, orig_constexprs)

        if mapped_rec != orig_constexprs:
            logger.info(
                "Source patching RECOMMENDED block sizes %s → %s",
                orig_constexprs,
                mapped_rec,
            )
            try:
                rec_kernel_fn = _load_patched_kernel(
                    kernel_module, kernel_name, mapped_rec, kernel_class,
                )
                rec_kernel_fn, _ = _unwrap_autotuner(rec_kernel_fn)

                # Recompute grid for template kernels (block sizes affect grid)
                if _is_template:
                    _base_grid_for_template = _resolve_grid(
                        metadata, original_config, {},
                    )
                    _template_rec_grid = _compute_template_grid(
                        source, mapped_rec, _base_grid_for_template, orig_constexprs,
                    )
            except Exception as exc:
                logger.warning(
                    "Recommended source patching failed, falling back to baked: %s",
                    exc,
                )
                rec_kernel_fn = kernel_fn

    # Build kwargs for each config (skip block-size mapping when using source patching)
    orig_kwargs = _build_kwargs(
        original_config, input_kwargs, shapes_kwargs,
        skip_block_sizes=_needs_source_patch,
    )
    rec_kwargs = _build_kwargs(
        recommended_config, input_kwargs, shapes_kwargs,
        skip_block_sizes=_needs_source_patch,
    )

    # Filter out kwargs the kernel doesn't accept (e.g. LLM-recommended
    # block sizes for kernels with hardcoded block dims)
    orig_kwargs = _filter_kwargs_for_kernel(kernel_fn, orig_kwargs)
    rec_kwargs = _filter_kwargs_for_kernel(rec_kernel_fn, rec_kwargs)

    # For reporting: use source-patched names or standard mapping
    if _needs_source_patch:
        mapped_blocks = mapped_rec if mapped_rec else orig_constexprs
    else:
        mapped_blocks = _map_block_sizes(
            recommended_config.get("block_size", {}), shapes_kwargs,
        )

    # Resolve grids per-config (block sizes affect grid dimensions)
    orig_grid = _template_orig_grid or _resolve_grid(
        metadata, original_config, orig_kwargs,
    )
    rec_grid = _template_rec_grid or _resolve_grid(
        metadata, recommended_config, rec_kwargs,
    )

    # Benchmark both configs (may use different kernel fns for template kernels).
    # Acquire GPU lock so only one benchmark runs at a time when multiple
    # kernel-optimization subprocesses execute in parallel.
    gpu_lock = get_gpu_lock()
    with gpu_lock:
        # Flush stale GPU memory from this process before benchmarking.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Move inputs to GPU only while holding the lock so that idle
        # workers don't occupy GPU memory and inflate baseline timings.
        input_args = [
            a.to(device) if isinstance(a, torch.Tensor) else a
            for a in input_args
        ]

        torch.cuda.synchronize()

        orig_stats = _benchmark_config(
            orig_kernel_fn, orig_grid, input_args, orig_kwargs, warmup, rep, "original",
        )
        rec_stats = _benchmark_config(
            rec_kernel_fn, rec_grid, input_args, rec_kwargs, warmup, rep, "recommended",
        )

        # Check for benchmark errors
        if "error" in orig_stats:
            err_msg = f"original config benchmark failed: {orig_stats['error']}"
            logger.error(err_msg)
            print(f"    [BENCHMARK ERROR] {err_msg}", flush=True)
            print(f"      kernel={kernel_name}  grid={list(orig_grid)}  config={original_config}", flush=True)
            print(f"      shapes_file={shapes_file}", flush=True)
            return {
                "error": err_msg,
                "original_stats": orig_stats,
                "recommended_stats": rec_stats,
                "original_config": original_config,
                "recommended_config": recommended_config,
                "grid": list(orig_grid),
            }
        if "error" in rec_stats:
            err_msg = f"recommended config benchmark failed: {rec_stats['error']}"
            logger.error(err_msg)
            print(f"    [BENCHMARK ERROR] {err_msg}", flush=True)
            print(f"      kernel={kernel_name}  grid={list(rec_grid)}  config={recommended_config}", flush=True)
            print(f"      shapes_file={shapes_file}", flush=True)
            return {
                "error": err_msg,
                "original_stats": orig_stats,
                "recommended_stats": rec_stats,
                "original_config": original_config,
                "recommended_config": recommended_config,
                "grid": list(orig_grid),
            }

        # Output correctness validation
        validation_result = None
        if validate_outputs:
            logger.info("Validating output correctness ...")
            validation_result = _validate_outputs(
                kernel_fn=orig_kernel_fn,
                orig_grid=orig_grid,
                rec_grid=rec_grid,
                input_args=input_args,
                orig_kwargs=orig_kwargs,
                rec_kwargs=rec_kwargs,
                device=device,
                rec_kernel_fn=rec_kernel_fn if rec_kernel_fn is not orig_kernel_fn else None,
            )
            if not validation_result["valid"]:
                error_detail = validation_result.get("error") or (
                    f"output mismatch in {len(validation_result['mismatches'])} tensor(s): "
                    + "; ".join(
                        f"{m['name']}: {m['reason']}"
                        + (f" (max_diff={m.get('max_abs_diff', 'N/A')})"
                           if m.get("max_abs_diff") is not None else "")
                        for m in validation_result["mismatches"]
                    )
                )
                logger.warning("Output validation FAILED: %s", error_detail)
                return {
                    "error": f"output validation failed: {error_detail}",
                    "validation_result": validation_result,
                    "original_stats": orig_stats,
                    "recommended_stats": rec_stats,
                    "original_config": original_config,
                    "recommended_config": recommended_config,
                    "grid": list(orig_grid),
                }
            logger.info(
                "Output validation PASSED (%d tensor(s) checked)",
                validation_result["num_tensors_checked"],
            )
    # GPU lock released — speedup computation and cleanup don't need GPU

    # Compute speedup.
    # Guard against timer-quantization noise: CUDA events have ~1 μs
    # (0.001 ms) resolution.  If the absolute difference between original
    # and recommended medians is within one timer tick, treat them as
    # identical and report speedup = 1.0.
    _TIMER_TICK_MS = 0.00105  # ~1 μs (CUDA event tick is 1024 ns)
    orig_med = orig_stats["median"]
    rec_med = rec_stats["median"]

    if rec_med > 0 and abs(orig_med - rec_med) <= _TIMER_TICK_MS:
        speedup = 1.0
        improvement_pct = 0.0
    elif rec_med > 0:
        speedup = orig_med / rec_med
        improvement_pct = (1 - rec_med / orig_med) * 100 if orig_med > 0 else None
    else:
        speedup = None
        improvement_pct = None

    # Free GPU memory after benchmarking to prevent OOM on long runs
    del input_args, input_kwargs
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "original_stats": orig_stats,
        "recommended_stats": rec_stats,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "original_config": original_config,
        "recommended_config": recommended_config,
        "block_size_mapping": mapped_blocks,
        "grid": list(orig_grid),
        "validation": validation_result,
    }


# ---------------------------------------------------------------------------
# Per-pair benchmarking (CLI-oriented wrapper)
# ---------------------------------------------------------------------------

def _benchmark_one_pair(
    pair: Dict[str, Any],
    shapes_file: Path,
    kernel_fn,
    args,
) -> Optional[Dict[str, Any]]:
    """Benchmark a single original-vs-recommended config pair.

    Returns a result dict suitable for inclusion in the output JSON,
    or None if the benchmark could not be run.
    """
    original_config = pair["original_config"]
    recommended_config = pair["recommended_config"]

    logger.info("--- Benchmarking pair from %s ---", pair.get("source_file", "?"))
    logger.info("  Shapes file: %s", shapes_file)
    logger.info("  Original config: %s", json.dumps(original_config, indent=2))
    logger.info("  Recommended config: %s", json.dumps(recommended_config, indent=2))

    # Load shapes and create random inputs
    try:
        metadata = load_tensor_shapes(shapes_file)
    except Exception as exc:
        logger.error("Failed to load shapes from %s: %s", shapes_file, exc)
        return None

    shapes_kwargs = metadata.get("kwargs", {})

    try:
        input_args, input_kwargs = create_kernel_inputs(
            shapes_file, device=args.device, values="random", seed=args.seed
        )
    except Exception as exc:
        logger.error("Failed to create inputs from %s: %s", shapes_file, exc)
        return None

    # Extract tensor info for reporting
    tensor_args_meta = metadata.get("tensor_args", [])
    tensor_shape = tensor_args_meta[0]["shape"] if tensor_args_meta else []
    tensor_dtype = tensor_args_meta[0]["dtype"] if tensor_args_meta else "unknown"

    # Build kwargs for each config
    orig_kwargs = _build_kwargs(original_config, input_kwargs, shapes_kwargs)
    rec_kwargs = _build_kwargs(recommended_config, input_kwargs, shapes_kwargs)

    # Filter out kwargs the kernel doesn't accept
    orig_kwargs = _filter_kwargs_for_kernel(kernel_fn, orig_kwargs)
    rec_kwargs = _filter_kwargs_for_kernel(kernel_fn, rec_kwargs)

    mapped_blocks = _map_block_sizes(
        recommended_config.get("block_size", {}), shapes_kwargs
    )

    # Resolve grids per-config (block sizes affect grid dimensions)
    orig_grid = _resolve_grid(metadata, original_config, orig_kwargs)
    rec_grid = _resolve_grid(metadata, original_config, rec_kwargs)

    logger.info("Original grid: %s", orig_grid)
    logger.info("Recommended grid: %s", rec_grid)
    logger.info("Tensor shape: %s %s", tensor_shape, tensor_dtype)
    logger.info("Original kwargs: %s", {k: v for k, v in orig_kwargs.items() if not isinstance(v, torch.Tensor)})
    logger.info("Recommended kwargs: %s", {k: v for k, v in rec_kwargs.items() if not isinstance(v, torch.Tensor)})

    # Benchmark both configs
    orig_stats = _benchmark_config(
        kernel_fn, orig_grid, input_args, orig_kwargs, args.warmup, args.rep, "original"
    )
    rec_stats = _benchmark_config(
        kernel_fn, rec_grid, input_args, rec_kwargs, args.warmup, args.rep, "recommended"
    )

    # Print comparison table
    table = _format_table(
        kernel_name=args.kernel_name,
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
        grid=orig_grid,
        orig_config=original_config,
        rec_config=recommended_config,
        orig_stats=orig_stats,
        rec_stats=rec_stats,
        mapped_block_names=mapped_blocks,
    )
    print("\n" + table + "\n")

    # Build result dict
    speedup = None
    improvement_pct = None
    if "error" not in orig_stats and "error" not in rec_stats:
        speedup = (
            orig_stats["median"] / rec_stats["median"]
            if rec_stats["median"] > 0 else None
        )
        improvement_pct = (
            (1 - rec_stats["median"] / orig_stats["median"]) * 100
            if orig_stats["median"] > 0 else None
        )

    return {
        "source_file": pair.get("source_file", ""),
        "kernel": args.kernel_name,
        "kernel_class": args.kernel_class,
        "device": args.device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tensor_shape": tensor_shape,
        "tensor_dtype": tensor_dtype,
        "grid": list(orig_grid),
        "original_config": original_config,
        "recommended_config": recommended_config,
        "block_size_mapping": mapped_blocks,
        "original_stats": orig_stats,
        "recommended_stats": rec_stats,
        "speedup": speedup,
        "improvement_pct": improvement_pct,
        "settings": {
            "warmup": args.warmup,
            "rep": args.rep,
            "seed": args.seed,
            "shapes_file": str(shapes_file),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark a Triton kernel with original vs LLM-recommended configs."
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing multiagent_triton_runner output JSONs.",
    )
    p.add_argument(
        "--shapes-file",
        type=str,
        required=False,
        default=None,
        help="Path to the *_shapes.json file. When provided, overrides shape_path "
             "from all output JSONs. When omitted, each output JSON's shape_path is "
             "resolved relative to the script directory.",
    )
    p.add_argument(
        "--kernel-module",
        type=str,
        required=True,
        help="Path to the Python file containing the Triton kernel.",
    )
    p.add_argument(
        "--kernel-name",
        type=str,
        required=True,
        help="Name of the @triton.jit kernel function.",
    )
    p.add_argument(
        "--kernel-class",
        type=str,
        default=None,
        help="Class name if the kernel is a class method (e.g. Operator).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for benchmarking (default: cuda).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for triton.testing.do_bench (default: 25).",
    )
    p.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Repetitions for triton.testing.do_bench (default: 100).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input tensor generation (default: 42).",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON comparison results (optional).",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = get_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    kernel_module = Path(args.kernel_module)

    # Resolve relative paths against script directory
    if not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    if not kernel_module.is_absolute():
        kernel_module = (SCRIPT_DIR / kernel_module).resolve()

    # Resolve global shapes-file override if provided
    global_shapes_file: Optional[Path] = None
    if args.shapes_file:
        global_shapes_file = Path(args.shapes_file)
        if not global_shapes_file.is_absolute():
            global_shapes_file = (SCRIPT_DIR / global_shapes_file).resolve()
        if not global_shapes_file.is_file():
            logger.error("Shapes file does not exist: %s", global_shapes_file)
            sys.exit(1)

    # Validate paths
    if not output_dir.is_dir():
        logger.error("Output dir does not exist: %s", output_dir)
        sys.exit(1)
    if not kernel_module.is_file():
        logger.error("Kernel module does not exist: %s", kernel_module)
        sys.exit(1)

    # ---- 1. Load configs from multiagent output ----
    config_pairs = _load_multiagent_configs(output_dir)
    if not config_pairs:
        logger.error("No valid config pairs found in %s", output_dir)
        sys.exit(1)

    # ---- 2. Load kernel (once, shared across all pairs) ----
    logger.info("Loading kernel %s from %s ...", args.kernel_name, kernel_module)
    kernel_fn = _load_kernel(kernel_module, args.kernel_name, args.kernel_class)
    kernel_fn, _at_keys = _unwrap_autotuner(kernel_fn)

    # ---- 3. Iterate over all config pairs ----
    all_results: List[Dict[str, Any]] = []
    skipped = 0

    for i, pair in enumerate(config_pairs):
        logger.info("=== Config pair %d/%d ===", i + 1, len(config_pairs))

        # Determine shapes file for this pair
        if global_shapes_file is not None:
            shapes_file = global_shapes_file
        else:
            shapes_file = _resolve_shape_path(pair["shape_path"], SCRIPT_DIR)
            if shapes_file is None:
                logger.warning(
                    "Skipping pair %d/%d from %s: shapes file not found (%s)",
                    i + 1, len(config_pairs),
                    pair.get("source_file", "?"),
                    pair["shape_path"],
                )
                skipped += 1
                continue

        result = _benchmark_one_pair(pair, shapes_file, kernel_fn, args)
        if result is not None:
            all_results.append(result)

    # ---- 4. Summary ----
    logger.info(
        "Completed: %d benchmarked, %d skipped out of %d total config pairs",
        len(all_results), skipped, len(config_pairs),
    )

    if not all_results:
        logger.error("No benchmarks completed successfully.")
        sys.exit(1)

    # Print aggregate summary
    print("\n" + "=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)
    for r in all_results:
        sp = r.get("speedup")
        sp_str = f"{sp:.2f}x" if sp is not None else "N/A"
        pct = r.get("improvement_pct")
        pct_str = f"{abs(pct):.1f}%" if pct is not None else "N/A"
        direction = ""
        if pct is not None:
            direction = " improvement" if pct > 0 else " regression"
        print(
            f"  {r['tensor_shape']} {r['tensor_dtype']}: "
            f"speedup={sp_str} ({pct_str}{direction})"
        )
    print("=" * 72)

    # ---- 5. Write output JSON ----
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = (SCRIPT_DIR / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        output_payload = {
            "kernel": args.kernel_name,
            "kernel_class": args.kernel_class,
            "device": args.device,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_pairs_total": len(config_pairs),
            "num_pairs_benchmarked": len(all_results),
            "num_pairs_skipped": skipped,
            "results": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(output_payload, f, indent=2)
        logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
