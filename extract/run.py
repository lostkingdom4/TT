"""
Scan oss_cuda_kernels.yaml for triton-tagged benchmarks, dump IR for all
(or N) input configurations, record the corresponding input shapes, and
create runner-compatible naming for multiagent_triton_runner.py.

Usage:
    # Process all triton-tagged operators (all inputs)
    python extract/run.py --gpu v100 --output-dir ./ir_output

    # Process specific operators only
    python extract/run.py --gpu v100 --output-dir ./ir_output --ops addmm,gemm

    # Limit to first 3 input configurations
    python extract/run.py --gpu v100 --output-dir ./ir_output --num-inputs 3
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

# Import centralized paths (project root is one level up from extract/)
sys.path.insert(0, str(SCRIPT_DIR.parent))
from paths import TRITONBENCH_ROOT as REPO_PATH, METADATA_DIR, OSS_CUDA_KERNELS_YAML

logger = logging.getLogger("extract")


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def parse_triton_ops(yaml_path: Path) -> Dict[str, List[str]]:
    """Return {op_name: [benchmark_names]} for every triton-tagged variant."""
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f) or {}

    triton_ops: Dict[str, List[str]] = {}
    for op_name, variants in metadata.items():
        if not isinstance(variants, dict):
            continue
        subops = [
            name
            for name, meta in variants.items()
            if isinstance(meta, dict) and "triton" in meta.get("tags", [])
        ]
        if subops:
            triton_ops[op_name] = subops
    return triton_ops


def _kernel_names_for_op(yaml_path: Path, op_name: str) -> List[str]:
    """Return triton kernel names for a given operator from the YAML metadata."""
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f) or {}
    variants = metadata.get(op_name, {})
    if not isinstance(variants, dict):
        return []
    kernels: List[str] = []
    for _variant, meta in variants.items():
        if not isinstance(meta, dict):
            continue
        tags = meta.get("tags", []) or []
        if "triton" not in tags:
            continue
        for k in meta.get("kernels", []) or []:
            kernels.append(str(k))
    return kernels


# ---------------------------------------------------------------------------
# Runner-compatible file generation
# ---------------------------------------------------------------------------

def _create_runner_compat_files(op_name: str, output_dir: Path) -> None:
    """Create symlinks compatible with multiagent_triton_runner.py.

    The runner discovers kernels by YAML kernel names (e.g. 'Operator.softmax_kernel')
    and expects TTGIR filenames to contain those names.  The extract pipeline produces
    filenames using bare kernel names (e.g. 'softmax_kernel').  This function bridges
    the gap by creating appropriately named symlinks.
    """
    kernel_names = _kernel_names_for_op(OSS_CUDA_KERNELS_YAML, op_name)
    if not kernel_names:
        return

    for kname in kernel_names:
        # Derive bare name (e.g. "Operator.softmax_kernel" -> "softmax_kernel")
        bare = kname.split(".")[-1] if "." in kname else kname

        # Create shapes symlinks so _find_shapes() can discover them by
        # YAML kernel name (e.g. *Operator.softmax_kernel_shapes.json).
        # Shape files live directly in output_dir with bare kernel names.
        for shapes_file in sorted(output_dir.glob(f"*{bare}_shapes.json")):
            if kname not in shapes_file.name:
                compat_name = shapes_file.name.replace(bare, kname)
                compat_path = shapes_file.parent / compat_name
                if not compat_path.exists():
                    compat_path.symlink_to(shapes_file.name)
                    logger.info("Created runner-compat shapes symlink %s", compat_path)

        # Also create symlinks for per-input shape files
        for shapes_file in sorted(output_dir.glob(f"*{bare}_input*_shapes.json")):
            if kname not in shapes_file.name:
                compat_name = shapes_file.name.replace(bare, kname)
                compat_path = shapes_file.parent / compat_name
                if not compat_path.exists():
                    compat_path.symlink_to(shapes_file.name)
                    logger.info("Created runner-compat per-input shapes symlink %s", compat_path)

    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir() or op_name not in subdir.name:
            continue

        for kname in kernel_names:
            bare = kname.split(".")[-1] if "." in kname else kname

            # Create TTGIR symlinks containing the full YAML kernel name.
            # Two symlinks are needed because different code paths use different
            # glob patterns:
            #   multiagent_triton_runner._find_ttgir: *{name}*.ttgir  (substring)
            #   optimizer_functions.analyze_kernel_from_cache: *{name}.ttgir (suffix)
            for ttgir in sorted(subdir.rglob("*.ttgir")):
                if bare in ttgir.stem and kname not in ttgir.stem:
                    # Substring-match symlink (e.g. Operator.softmax_kernel_k0_input_x0.ttgir)
                    compat_name = ttgir.name.replace(bare, kname)
                    compat_path = ttgir.parent / compat_name
                    if not compat_path.exists():
                        compat_path.symlink_to(ttgir.name)
                        logger.info("Created runner-compat TTGIR symlink %s", compat_path)
                    # Also create symlink for the co-located JSON metadata
                    json_src = ttgir.with_suffix('.json')
                    if json_src.exists():
                        json_dst = compat_path.with_suffix('.json')
                        if not json_dst.exists():
                            json_dst.symlink_to(json_src.name)
                            logger.info("Created runner-compat JSON symlink %s", json_dst)

                    # Exact-suffix symlink (e.g. Operator.softmax_kernel.ttgir)
                    suffix_path = ttgir.parent / f"{kname}.ttgir"
                    if not suffix_path.exists():
                        suffix_path.symlink_to(ttgir.name)
                        logger.info("Created runner-compat TTGIR suffix symlink %s", suffix_path)
                    # Also create JSON metadata symlink for the suffix path
                    json_src = ttgir.with_suffix('.json')
                    if json_src.exists():
                        json_suffix = suffix_path.with_suffix('.json')
                        if not json_suffix.exists():
                            json_suffix.symlink_to(json_src.name)
                            logger.info("Created runner-compat JSON suffix symlink %s", json_suffix)


# ---------------------------------------------------------------------------
# Child-process work: dump IR + capture kernel shapes for one operator
# ---------------------------------------------------------------------------

def _shape_signature(tensor_args: list, scalar_args: list, grid) -> str:
    """Compute a hashable signature from tensor shapes/dtypes and scalar values.

    Two kernel invocations with the same signature are considered to have
    identical input configurations (e.g., warmup vs timed run within the
    same input_id, or autotuning runs with different configs but same tensors).
    Grid is NOT included because autotuned kernels launch with varying grids.
    """
    parts = []
    for t in tensor_args:
        parts.append(f"{t['arg_index']}:{t['shape']}:{t['dtype']}")
    for s in scalar_args:
        parts.append(f"{s['arg_index']}:{s['value']}")
    return "|".join(parts)


def _normalize_grid(grid) -> object:
    """Convert a Triton grid value to a JSON-serializable format."""
    if isinstance(grid, (tuple, list)):
        return list(grid)
    if callable(grid):
        try:
            import inspect
            return {"type": "callable", "source": inspect.getsource(grid)}
        except Exception:
            return {"type": "callable", "repr": str(grid)}
    return str(grid)


def _normalize_kwargs(kwargs: dict) -> dict:
    """Convert kernel kwargs to a JSON-serializable dict, excluding 'grid'."""
    import torch

    info: Dict = {}
    for key, val in kwargs.items():
        if key == "grid":
            continue
        if isinstance(val, torch.Tensor):
            info[key] = {
                "type": "tensor",
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "device": str(val.device),
            }
        elif isinstance(val, (int, float, bool, str)):
            info[key] = {
                "type": "scalar",
                "value": val,
                "value_type": type(val).__name__,
            }
        else:
            info[key] = {
                "type": "other",
                "value": str(val),
                "value_type": type(val).__name__,
            }
    return info


def _dump_ir_with_shape_capture(extra_args: List[str]) -> Dict[str, List[dict]]:
    """Run tritonbench with --dump-ir while capturing Triton kernel launch args.

    Hooks ``JITFunction.run`` so that every Triton kernel launch records its
    tensor arguments, scalar arguments, and grid.  The hook composes safely
    with ``dump_ir``'s own ``mock.patch.object`` hook.

    Captures **all** unique input configurations per kernel (deduplicated by
    shape signature), not just the first invocation.

    Returns:
        Dict mapping kernel function name -> list of shape metadata dicts,
        one per unique input configuration.  Each dict has keys:
        ``kernel``, ``tensor_args``, ``scalar_args``, ``kwargs``, ``grid``,
        ``total_args``, ``input_index``.
    """
    import torch
    from triton.runtime.jit import JITFunction
    from tritonbench.utils.run_utils import tritonbench_run

    captured: Dict[str, List[dict]] = {}
    seen_sigs: Dict[str, set] = {}
    original_run = JITFunction.run

    def capture_run(self, *args, grid, **kwargs):
        try:
            kname = getattr(getattr(self, "fn", None), "__name__", "unknown")
            tensor_args = []
            scalar_args = []
            for idx, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    tensor_args.append({
                        "arg_index": idx,
                        "shape": list(arg.shape),
                        "dtype": str(arg.dtype),
                        "device": str(arg.device),
                        "numel": int(arg.numel()),
                        "stride": list(arg.stride()),
                        "is_contiguous": bool(arg.is_contiguous()),
                    })
                elif isinstance(arg, (int, float, bool)):
                    scalar_args.append({
                        "arg_index": idx,
                        "value": arg,
                        "type": type(arg).__name__,
                    })

            sig = _shape_signature(tensor_args, scalar_args, grid)

            if kname not in seen_sigs:
                seen_sigs[kname] = set()
                captured[kname] = []

            if sig not in seen_sigs[kname]:
                seen_sigs[kname].add(sig)
                captured[kname].append({
                    "kernel": kname,
                    "tensor_args": tensor_args,
                    "scalar_args": scalar_args,
                    "kwargs": _normalize_kwargs(kwargs),
                    "grid": _normalize_grid(grid),
                    "total_args": len(args),
                    "input_index": len(captured[kname]),
                })
        except Exception:
            pass  # never let capture failure break the actual kernel run
        return original_run(self, *args, grid=grid, **kwargs)

    JITFunction.run = capture_run
    try:
        tritonbench_run(extra_args)
    finally:
        JITFunction.run = original_run

    return captured


def _save_kernel_shapes(captured: Dict[str, List[dict]], op_name: str,
                        output_dir: Path) -> None:
    """Write per-kernel, per-input shape files.

    File naming:
      - Backward-compat default: ``{op_name}_{kernel_name}_shapes.json``
        (always a copy of input 0, so existing globs still work)
      - Per-input files: ``{op_name}_{kernel_name}_input{i}_shapes.json``

    ``multiagent_triton_runner._find_shapes()`` discovers the default file
    via its ``*{kernel_name}_shapes.json`` glob pattern, and per-input files
    via ``*{kernel_name}_input*_shapes.json`` when ``--all-shapes`` is used.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for kname, shape_list in captured.items():
        if not shape_list:
            continue

        # 1. Backward-compatible default file (= input 0)
        default_filename = f"{op_name}_{kname}_shapes.json"
        default_path = output_dir / default_filename
        with open(default_path, "w") as f:
            json.dump(shape_list[0], f, indent=2)
        logger.info("Saved default kernel shapes to %s", default_path)

        # 2. Per-input files for all captured inputs
        for i, meta in enumerate(shape_list):
            per_input_filename = f"{op_name}_{kname}_input{i}_shapes.json"
            per_input_path = output_dir / per_input_filename
            with open(per_input_path, "w") as f:
                json.dump(meta, f, indent=2)

        logger.info("Saved %d per-input shape files for %s", len(shape_list), kname)


def run_child(args: argparse.Namespace, extra_args: List[str]) -> None:
    """Child-process entry: dump IR and capture kernel shapes for a single operator."""
    output_dir = Path(args.output_dir)
    op_name = args.op
    op_dir = output_dir / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    subops = args.only.split(",") if args.only else []

    num_inputs = getattr(args, "num_inputs", 0)

    # 1. Dump IR + capture kernel shapes in a single pass
    logger.info("[extract] Dumping IR for %s (benchmarks: %s, num_inputs=%s) ...",
                op_name, subops, num_inputs or "all")
    ir_args = [
        "--op", op_name,
        "--dump-ir", str(op_dir),
    ]
    if num_inputs > 0:
        ir_args.extend(["--num-inputs", str(num_inputs)])
    if subops:
        ir_args.extend(["--only", ",".join(subops)])
    ir_args.extend(extra_args)

    captured_shapes: Dict[str, List[dict]] = {}
    try:
        captured_shapes = _dump_ir_with_shape_capture(ir_args)
    except Exception as exc:
        logger.warning("[extract] Failed to dump IR for %s: %s", op_name, exc)

    # 2. Save captured kernel shapes as per-kernel JSON files
    if captured_shapes:
        logger.info("[extract] Saving kernel shapes for %s (%d kernels) ...",
                    op_name, len(captured_shapes))
        try:
            _save_kernel_shapes(captured_shapes, op_name, op_dir)
        except Exception as exc:
            logger.warning(
                "[extract] Failed to save shapes for %s: %s", op_name, exc
            )
    else:
        logger.warning("[extract] No kernel shapes captured for %s", op_name)

    # 3. Create files compatible with multiagent_triton_runner.py naming conventions
    logger.info("[extract] Creating runner-compatible files for %s ...", op_name)
    try:
        _create_runner_compat_files(op_name, op_dir)
    except Exception as exc:
        logger.warning(
            "[extract] Failed to create compat files for %s: %s", op_name, exc
        )


# ---------------------------------------------------------------------------
# Parent-process work: orchestrate children
# ---------------------------------------------------------------------------

def spawn_child(op_name: str, subops: List[str], output_dir: Path,
                num_inputs: int = 0, gpu: str = "") -> None:
    """Spawn a child subprocess for a single operator."""
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--run-child",
        "--gpu", gpu,
        "--op", op_name,
        "--only", ",".join(subops),
        "--output-dir", str(output_dir),
        "--num-inputs", str(num_inputs),
    ]
    logger.info("[extract] Starting: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        subprocess.check_call(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=str(REPO_PATH),
        )
        elapsed = time.perf_counter() - start
        logger.info("[extract] Finished %s in %.1fs", op_name, elapsed)
    except subprocess.CalledProcessError:
        logger.warning("[extract] Failed for %s, continuing ...", op_name)
    except KeyboardInterrupt:
        logger.warning("[extract] Interrupted, exiting.")
        sys.exit(1)


def _resolve_output_dir(raw: str) -> Path:
    """Resolve output dir relative to the script's directory (not cwd)."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def run_parent(args: argparse.Namespace) -> None:
    """Parse YAML, filter operators, and spawn a child per operator."""
    output_dir = _resolve_output_dir(args.output_dir) / args.gpu
    output_dir.mkdir(parents=True, exist_ok=True)

    triton_ops = parse_triton_ops(Path(args.metadata))
    if not triton_ops:
        logger.error("[extract] No triton-tagged operators found in %s", args.metadata)
        sys.exit(0)

    if args.ops:
        requested = set(args.ops.split(","))
        triton_ops = {k: v for k, v in triton_ops.items() if k in requested}
        missing = requested - set(triton_ops.keys())
        if missing:
            logger.warning("[extract] Operators not found in YAML: %s", missing)

    logger.info("[extract] Found %d triton-tagged operators:", len(triton_ops))
    for op_name, subops in triton_ops.items():
        logger.info("  %s: %s", op_name, subops)

    for op_name, subops in triton_ops.items():
        logger.info("\n[extract] ===== %s =====", op_name)
        spawn_child(op_name, subops, output_dir, num_inputs=args.num_inputs,
                    gpu=args.gpu)

    logger.info("\n[extract] Done. Results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan oss_cuda_kernels.yaml for triton-tagged benchmarks, "
            "dump IR for all (or N) input configurations, and record per-input shapes."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="GPU name to label this run (e.g. v100, a100, h100). Used as a subfolder under output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ir_output",
        help="Output directory for IR files and input shapes (relative to script dir).",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated list of operators to process (default: all).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(OSS_CUDA_KERNELS_YAML),
        help="Path to oss_cuda_kernels.yaml.",
    )
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=0,
        help="Number of input configurations to process (0 = all available).",
    )
    parser.add_argument(
        "--run-child",
        action="store_true",
        help="(internal) Run as a child process for a single operator.",
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="(internal) Operator name when running as child.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="(internal) Comma-separated benchmark names when running as child.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Triton JIT cache (~/.triton/cache/) before extraction. "
             "Needed when tritonbench was re-cloned or moved so IR loc() "
             "paths reflect the current source locations.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = get_parser()
    args, extra_args = parser.parse_known_args()

    if getattr(args, "clear_cache", False) and not args.run_child:
        cache_dir = Path.home() / ".triton" / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            logger.info("[extract] Cleared Triton cache: %s", cache_dir)

    if args.run_child:
        run_child(args, extra_args)
    else:
        run_parent(args)
