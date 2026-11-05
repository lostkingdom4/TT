"""Discover operators and kernels from an IR output directory.

Scans ``extract/ir_output/<device>/`` for operators that have
``*_input*_shapes.json`` files and checks whether the corresponding
TritonBench operator kernels use ``@triton.autotune``.

Also supports filtering by a run-list JSON
(e.g. ``run_lists/v100_improved_only.json``) to select specific
(operator, kernel) pairs for the paper's 19-kernel evaluation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path so we can import paths.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from paths import TRITONBENCH_ROOT
from utils import get_device_name, parse_shapes_file  # noqa: F401 — re-exported

log = logging.getLogger(__name__)


def _detect_autotune(operator: str) -> bool:
    """Return True if any file for this operator uses ``@triton.autotune``
    or ``max_autotune=True``."""
    op_dir = TRITONBENCH_ROOT / "tritonbench" / "operators" / operator
    if not op_dir.is_dir():
        return False
    for py_file in op_dir.rglob("*.py"):
        try:
            text = py_file.read_text(errors="ignore")
            if ("@triton.autotune" in text or "@autotune" in text
                    or "max_autotune=True" in text):
                return True
        except OSError:
            continue
    return False


def discover_operators(ir_output_dir: Path) -> List[Dict[str, Any]]:
    """Scan ``ir_output_dir`` and return operator-level entries.

    Each entry contains:
        name           — operator name (e.g. ``softmax``)
        ir_dir         — Path to the operator's IR directory
        num_shapes     — number of captured input shapes
        shapes_files   — sorted list of ``*_input*_shapes.json`` paths
        has_autotune   — whether the TritonBench op uses ``@triton.autotune``
        kernel_name    — name of the kernel (from the first shapes file)
        in_tritonbench — whether the operator exists in
                         ``tritonbench/operators/``
    """
    ir_output_dir = Path(ir_output_dir).resolve()
    if not ir_output_dir.is_dir():
        raise FileNotFoundError(
            f"IR output directory not found: {ir_output_dir}"
        )

    operators: List[Dict[str, Any]] = []
    for entry in sorted(ir_output_dir.iterdir()):
        if not entry.is_dir():
            continue
        shapes_files = sorted(entry.glob("*_input*_shapes.json"))
        if not shapes_files:
            continue

        op_name = entry.name
        tb_op_dir = TRITONBENCH_ROOT / "tritonbench" / "operators" / op_name
        in_tritonbench = tb_op_dir.is_dir()
        has_autotune = _detect_autotune(op_name) if in_tritonbench else False

        meta = parse_shapes_file(shapes_files[0])
        kernel_name = meta["kernel_name"] if meta else "unknown"

        operators.append({
            "name": op_name,
            "ir_dir": entry,
            "num_shapes": len(shapes_files),
            "shapes_files": shapes_files,
            "has_autotune": has_autotune,
            "kernel_name": kernel_name,
            "in_tritonbench": in_tritonbench,
        })

    return operators


def discover_kernels(
    ir_output_dir: Path,
    operators: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return per-(operator, kernel) entries rather than per-operator.

    Each operator directory may contain multiple kernels (e.g. addmm has
    ``_addmm_fwd``, ``triton_mm``, ``triton_tem_fused_addmm_0``).  This
    helper reads every ``*_input0_shapes.json`` file to enumerate the
    distinct ``kernel`` values per operator.

    Fields per entry::

        operator, kernel, ir_dir, num_shapes, shapes_files,
        has_autotune, in_tritonbench
    """
    if operators is None:
        operators = discover_operators(ir_output_dir)

    kernels: List[Dict[str, Any]] = []
    for op in operators:
        # Group shape files by kernel name.
        by_kernel: Dict[str, List[Path]] = {}
        for sf in op["shapes_files"]:
            meta = parse_shapes_file(sf)
            kname = meta["kernel_name"] if meta else "unknown"
            by_kernel.setdefault(kname, []).append(sf)

        for kname, files in sorted(by_kernel.items()):
            kernels.append({
                "operator": op["name"],
                "kernel": kname,
                "ir_dir": op["ir_dir"],
                "num_shapes": len(files),
                "shapes_files": sorted(files),
                "has_autotune": op["has_autotune"],
                "in_tritonbench": op["in_tritonbench"],
            })
    return kernels


def load_run_list(run_list_path: Path) -> List[Tuple[str, str]]:
    """Parse a run-list JSON and return a list of ``(operator, kernel)``
    pairs.  Mode is always ``include`` for the paper's run lists."""
    data = json.loads(Path(run_list_path).read_text())
    out = []
    for entry in data.get("kernels", []):
        op = entry.get("operator")
        kern = entry.get("kernel")
        if op and kern:
            out.append((op, kern))
    return out


def filter_by_run_list(
    kernels: List[Dict[str, Any]],
    run_list: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Return only ``kernels`` entries whose ``(operator, kernel)`` is in
    the run-list."""
    wanted = set(run_list)
    return [k for k in kernels if (k["operator"], k["kernel"]) in wanted]


def print_discovery_summary(
    kernels_or_operators: List[Dict[str, Any]],
    device: str,
) -> None:
    """Human-readable summary; accepts either the operator-level or the
    kernel-level entry shape."""
    entries = kernels_or_operators
    per_kernel = entries and "kernel" in entries[0]

    total_shapes = sum(e["num_shapes"] for e in entries)
    autotuned = [e for e in entries if e["has_autotune"]]
    in_tb = [e for e in entries if e.get("in_tritonbench", True)]

    print(f"\n{'=' * 70}")
    label = "Kernels" if per_kernel else "Operators"
    print(f"  Discovery Summary — device: {device}")
    print(f"{'=' * 70}")
    print(f"  {label} with shapes:   {len(entries)}")
    print(f"  In TritonBench:         {len(in_tb)}")
    print(f"  With @triton.autotune:  {len(autotuned)}")
    print(f"  Total shapes:           {total_shapes}")
    print(f"{'=' * 70}")

    if per_kernel:
        print(f"  {'Operator':<22} {'Kernel':<50} {'Shp':>5} {'AT':>3}")
        print(f"  {'-' * 22} {'-' * 50} {'-' * 5} {'-' * 3}")
        for e in entries:
            at = "y" if e["has_autotune"] else "n"
            k = e["kernel"][:50]
            print(f"  {e['operator']:<22} {k:<50} {e['num_shapes']:>5} {at:>3}")
    else:
        print(f"  {'Operator':<25} {'Shapes':>7} {'Autotune':>10} {'In TB':>7}")
        print(f"  {'-' * 25} {'-' * 7} {'-' * 10} {'-' * 7}")
        for op in entries:
            at = "yes" if op["has_autotune"] else "no"
            tb = "yes" if op["in_tritonbench"] else "no"
            print(f"  {op['name']:<25} {op['num_shapes']:>7} {at:>10} {tb:>7}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Discover operators / kernels in IR output")
    parser.add_argument("ir_dir", type=Path,
                        help="IR output directory (e.g. extract/ir_output/v100)")
    parser.add_argument("--run-list", type=Path, default=None,
                        help="Optional run-list JSON to filter by")
    parser.add_argument("--per-kernel", action="store_true",
                        help="Print per-(operator, kernel) entries")
    args = parser.parse_args()

    device = get_device_name(args.ir_dir)
    if args.per_kernel or args.run_list:
        entries = discover_kernels(args.ir_dir)
        if args.run_list:
            entries = filter_by_run_list(entries, load_run_list(args.run_list))
    else:
        entries = discover_operators(args.ir_dir)
    print_discovery_summary(entries, device)
