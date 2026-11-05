#!/usr/bin/env python3
"""Batch experiment runner for the multi-agent Triton kernel optimizer.

Auto-discovers all operator/kernel combinations from an IR output directory
and runs multiagent_triton_runner.py for each in an isolated subprocess,
tracking progress in a resumable JSON manifest.

Usage examples::

    # Discover all kernels and print commands (no execution)
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 --dry-run

    # Run all discovered kernels
    python batch_runner.py --gpu v100 --output-dir output/batch_v100

    # Run only specific operators
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --operators softmax addmm

    # Resume after interruption (default behavior)
    python batch_runner.py --gpu v100 --output-dir output/batch_v100

    # Force re-run everything
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 --no-resume

    # All shapes, with debug output
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --all-shapes --debug-mode

    # Cap inductor kernels to 5 per operator
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --max-kernels-per-op 5

    # Run only inductor-generated kernels (torch.compile/pt2)
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --kernel-source inductor

    # Run only hand-written Triton kernels
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --kernel-source handwritten

    # Run only kernels listed in a JSON run-list file
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --run-list run_lists/v100_threshold_include.json

    # Exclude kernels listed in a JSON run-list file
    python batch_runner.py --gpu v100 --output-dir output/batch_v100 \
        --run-list run_lists/v100_threshold_exclude.json
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import GPU_ARCH_MAP, get_device_name, parse_shapes_file

log = logging.getLogger(__name__)

# Path to TritonBench metadata YAML (lists kernel names and tags)
_METADATA_YAML = _PROJECT_ROOT / "tb" / "tritonbench" / "metadata" / "oss_cuda_kernels.yaml"

# Inductor codegen naming pattern: triton_{poi,red,per,tem}_fused_*
_INDUCTOR_PATTERN = re.compile(r"^triton_(poi|red|per|tem)_fused_")


# ---------------------------------------------------------------------------
# Kernel source classification (inductor vs hand-written)
# ---------------------------------------------------------------------------

def _load_handwritten_kernel_names() -> Set[str]:
    """Load hand-written Triton kernel names from oss_cuda_kernels.yaml.

    Returns the set of kernel names that appear under entries tagged with
    'triton' (but not 'pt2'), which are hand-written Triton kernels.
    """
    if not _METADATA_YAML.exists():
        log.warning("Metadata YAML not found at %s — kernel classification "
                     "will fall back to naming patterns only", _METADATA_YAML)
        return set()

    try:
        import yaml
        with open(_METADATA_YAML) as f:
            meta = yaml.safe_load(f)
    except Exception as e:
        log.warning("Failed to load metadata YAML: %s", e)
        return set()

    handwritten: Set[str] = set()
    for _operator, variants in meta.items():
        for _variant_name, variant_info in variants.items():
            tags = variant_info.get("tags", [])
            kernels = variant_info.get("kernels", [])
            # Hand-written = has 'triton' tag but NOT 'pt2'
            if "triton" in tags and "pt2" not in tags and kernels:
                for k in kernels:
                    # Skip torch.ops.* entries (native ops, not Triton JIT)
                    if not k.startswith("torch.ops.") and not k.startswith("<"):
                        handwritten.add(k)
    return handwritten


# Lazily populated on first use
_HANDWRITTEN_NAMES: Optional[Set[str]] = None


def classify_kernel_source(kernel_name: str) -> str:
    """Classify a kernel as 'inductor' or 'handwritten'.

    Uses the oss_cuda_kernels.yaml metadata (primary) and inductor naming
    patterns (fallback).

    Returns:
        'handwritten' or 'inductor'.
    """
    global _HANDWRITTEN_NAMES
    if _HANDWRITTEN_NAMES is None:
        _HANDWRITTEN_NAMES = _load_handwritten_kernel_names()

    # 1. Check metadata: if kernel name is listed under a triton-tagged entry
    if kernel_name in _HANDWRITTEN_NAMES:
        return "handwritten"

    # 2. Check inductor naming pattern
    if _INDUCTOR_PATTERN.match(kernel_name):
        return "inductor"

    # 3. Fallback heuristic: if name starts with common inductor prefixes
    #    but wasn't caught by the pattern, treat as inductor
    if kernel_name.startswith("triton_") and "_fused_" in kernel_name:
        return "inductor"

    # 4. Default to handwritten (non-inductor kernels don't match the pattern)
    return "handwritten"


# ---------------------------------------------------------------------------
# Run-list loading
# ---------------------------------------------------------------------------

def load_run_list(path: Path) -> Dict[str, Any]:
    """Load a JSON run-list file specifying kernels to include or exclude.

    Expected JSON format::

        {
          "mode": "include" | "exclude",
          "kernels": [
            {"operator": "layer_norm", "kernel": "_layer_norm_fwd_fused_no_bias"},
            ...
          ]
        }

    Returns the parsed dict with a "lookup" key added: a set of
    (operator, kernel_name) tuples for fast matching.  When kernel names
    contain "..." (abbreviated in docs), they are treated as prefix patterns.
    """
    with open(path) as f:
        data = json.load(f)

    mode = data.get("mode", "include")
    if mode not in ("include", "exclude"):
        raise ValueError(f"run-list mode must be 'include' or 'exclude', got '{mode}'")

    kernels = data.get("kernels", [])
    lookup: Set[tuple] = set()
    prefixes: List[tuple] = []

    for entry in kernels:
        op = entry["operator"]
        kname = entry["kernel"]
        if "..." in kname:
            # Treat as prefix: "triton_poi_fused_..._softmax_1" → not exact match,
            # store as prefix pattern for fuzzy matching
            prefixes.append((op, kname))
        else:
            lookup.add((op, kname))

    data["_lookup"] = lookup
    data["_prefixes"] = prefixes
    return data


def run_list_matches(run_list: Dict[str, Any], operator: str, kernel_name: str) -> bool:
    """Check if an (operator, kernel) pair matches the run-list.

    Returns True if the pair is found in the run-list's kernel set (exact
    match or prefix match for abbreviated names).
    """
    lookup = run_list["_lookup"]
    prefixes = run_list["_prefixes"]

    # Exact match
    if (operator, kernel_name) in lookup:
        return True

    # Prefix/pattern match for abbreviated kernel names (contain "...")
    for op, pattern in prefixes:
        if operator != op:
            continue
        # "triton_poi_fused_..._add_div_mul_sqrt_sub_1" →
        # split on "..." and check start/end
        parts = pattern.split("...")
        if len(parts) == 2:
            if kernel_name.startswith(parts[0]) and kernel_name.endswith(parts[1]):
                return True

    return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    """A single kernel to optimize."""
    id: str                     # "{operator}__{kernel_name}"
    operator: str
    kernel_name: str
    cache_dir: str              # stored relative to project root
    shapes_dir: str             # stored relative to project root
    num_shapes_available: int
    status: str = "pending"     # pending | running | success | failed | skipped
    num_shapes_to_process: int = 1
    command: str = ""
    output_dir: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    per_shape_results: List[Dict[str, Any]] = field(default_factory=list)
    best_speedup: Optional[float] = None
    avg_speedup: Optional[float] = None
    shapes_improved: int = 0
    shapes_regressed: int = 0
    shapes_kept_original: int = 0
    shapes_failed: int = 0
    error: Optional[str] = None
    runner_exit_code: Optional[int] = None
    completed_shapes: List[str] = field(default_factory=list)


@dataclass
class ShapeWorkItem:
    """A single (kernel, shape) pair — the unit of work for --parallel-shapes."""
    kernel_id: str        # references WorkItem.id
    operator: str
    kernel_name: str
    cache_dir: str        # relative to project root
    shapes_dir: str       # relative to project root
    shape_file: str       # specific shape file path, relative to project root
    output_dir: str       # per-kernel output dir (from WorkItem)
    shape_index: int      # 0-based index among this kernel's shapes


@dataclass
class BatchManifest:
    """Tracks batch experiment progress."""
    version: int = 1
    gpu: str = ""
    target_gpu_arch: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)
    work_items: List[WorkItem] = field(default_factory=list)

    def __post_init__(self):
        # RLock for thread-safe manifest access in parallel mode.
        # Not a dataclass field so asdict() ignores it.
        self._lock = threading.RLock()

    def save(self, path: Path) -> None:
        """Atomically save manifest to disk (thread-safe)."""
        with self._lock:
            data = {
                "version": self.version,
                "gpu": self.gpu,
                "target_gpu_arch": self.target_gpu_arch,
                "config": self.config,
                "summary": self._compute_summary(),
                "work_items": [asdict(w) for w in self.work_items],
            }
        # Write to temp file first, then rename for atomicity
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=".manifest_"
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path) -> "BatchManifest":
        """Load manifest from disk."""
        with open(path) as f:
            data = json.load(f)
        manifest = cls(
            version=data.get("version", 1),
            gpu=data.get("gpu", ""),
            target_gpu_arch=data.get("target_gpu_arch", ""),
            config=data.get("config", {}),
        )
        for item_data in data.get("work_items", []):
            manifest.work_items.append(WorkItem(**item_data))
        return manifest

    def _compute_summary(self) -> Dict[str, Any]:
        counts = defaultdict(int)
        total_shapes = 0
        total_shapes_to_process = 0
        for w in self.work_items:
            counts[w.status] += 1
            total_shapes += w.num_shapes_available
            total_shapes_to_process += w.num_shapes_to_process
        return {
            "total_kernels": len(self.work_items),
            "total_shapes_available": total_shapes,
            "total_shapes_to_process": total_shapes_to_process,
            "pending": counts["pending"],
            "running": counts["running"],
            "success": counts["success"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
        }

    def get_item(self, item_id: str) -> Optional[WorkItem]:
        for w in self.work_items:
            if w.id == item_id:
                return w
        return None


class KernelResultCollector:
    """Thread-safe per-kernel result aggregator for shape-level parallelism.

    When shapes of the same kernel run on different threads, this class
    collects their results and finalises the parent WorkItem once all
    shapes for that kernel have completed.
    """

    def __init__(self, manifest: BatchManifest, manifest_path: Path):
        self._manifest = manifest
        self._manifest_path = manifest_path
        self._lock = threading.Lock()
        self._pending_counts: Dict[str, int] = {}   # kernel_id -> remaining shapes
        self._results: Dict[str, List[Dict[str, Any]]] = {}  # kernel_id -> per_shape results

    def register_kernel(self, kernel_id: str, num_shapes: int):
        """Register a kernel with the number of shapes to expect."""
        with self._lock:
            self._pending_counts[kernel_id] = num_shapes
            self._results.setdefault(kernel_id, [])

    def pre_populate(self, kernel_id: str, existing_results: List[Dict[str, Any]]):
        """Pre-populate results from a previous run (for resume support)."""
        with self._lock:
            self._results.setdefault(kernel_id, []).extend(existing_results)

    def record_shape_result(self, kernel_id: str, shape_result: Dict[str, Any]):
        """Record the result of a single shape and finalise the kernel if complete."""
        with self._lock:
            self._results.setdefault(kernel_id, []).append(shape_result)
            self._pending_counts[kernel_id] = self._pending_counts.get(kernel_id, 1) - 1
            remaining = self._pending_counts[kernel_id]

            item = self._manifest.get_item(kernel_id)
            if item:
                item.per_shape_results = list(self._results[kernel_id])
                shape_stem = shape_result.get("shape_file", "")
                if shape_stem and shape_stem not in item.completed_shapes:
                    item.completed_shapes.append(shape_stem)

            if remaining <= 0:
                self._finalize_kernel(kernel_id)

            self._manifest.save(self._manifest_path)

    def _finalize_kernel(self, kernel_id: str):
        """Compute aggregate stats for a completed kernel (called under lock)."""
        item = self._manifest.get_item(kernel_id)
        if not item:
            return

        results = self._results.get(kernel_id, [])
        speedups: List[float] = []
        improved = regressed = kept_original = failed = 0

        for r in results:
            status = r.get("status", "failed")
            if status == "failed":
                failed += 1
            elif status == "kept_original":
                kept_original += 1
                speedups.append(r.get("speedup", 1.0))
            elif status == "improved":
                improved += 1
                if r.get("speedup") is not None:
                    speedups.append(r["speedup"])
            elif status == "regressed":
                regressed += 1
                if r.get("speedup") is not None:
                    speedups.append(r["speedup"])
            else:
                # unchanged
                if r.get("speedup") is not None:
                    speedups.append(r["speedup"])

        item.shapes_improved = improved
        item.shapes_regressed = regressed
        item.shapes_kept_original = kept_original
        item.shapes_failed = failed
        item.best_speedup = max(speedups) if speedups else None
        item.avg_speedup = round(sum(speedups) / len(speedups), 4) if speedups else None
        item.completed_at = datetime.now(timezone.utc).isoformat()

        if item.started_at:
            try:
                t0 = datetime.fromisoformat(item.started_at)
                item.duration_seconds = round(
                    (datetime.now(timezone.utc) - t0).total_seconds(), 1,
                )
            except (ValueError, TypeError):
                pass

        # At least one shape succeeded → kernel success
        item.status = "success" if (improved + regressed + kept_original) > 0 else "failed"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_kernels(ir_dir: Path) -> List[WorkItem]:
    """Scan an IR output directory and return a WorkItem per (operator, kernel).

    Discovery is shapes-file-driven: we find all *_input*_shapes.json at
    each operator root, parse the JSON to get the kernel name, group by
    kernel, then verify a corresponding TTGIR exists.
    """
    if not ir_dir.is_dir():
        raise FileNotFoundError(f"IR directory not found: {ir_dir}")

    items: List[WorkItem] = []

    for operator_dir in sorted(ir_dir.iterdir()):
        if not operator_dir.is_dir():
            continue

        operator = operator_dir.name

        # Find per-input shapes files at operator root (non-recursive)
        shapes_files = sorted(operator_dir.glob("*_input*_shapes.json"))
        if not shapes_files:
            log.debug("Skipping %s — no shapes files", operator)
            continue

        # Group shapes files by kernel name
        kernel_shapes: Dict[str, List[Path]] = defaultdict(list)
        for sf in shapes_files:
            meta = parse_shapes_file(sf)
            if meta and meta["kernel_name"] != "unknown":
                kernel_shapes[meta["kernel_name"]].append(sf)
            else:
                log.debug("Skipping unparseable shapes file: %s", sf)

        # Create one work item per kernel
        for kernel_name, shape_paths in sorted(kernel_shapes.items()):
            # Verify TTGIR exists
            ttgir_candidates = sorted(operator_dir.rglob(f"*{kernel_name}*.ttgir"))
            if not ttgir_candidates:
                log.warning(
                    "Skipping %s/%s — no matching TTGIR found",
                    operator, kernel_name,
                )
                items.append(WorkItem(
                    id=f"{operator}__{kernel_name}",
                    operator=operator,
                    kernel_name=kernel_name,
                    cache_dir=str(operator_dir.relative_to(_PROJECT_ROOT)),
                    shapes_dir=str(operator_dir.relative_to(_PROJECT_ROOT)),
                    num_shapes_available=len(shape_paths),
                    status="skipped",
                    error="No matching TTGIR file found",
                ))
                continue

            items.append(WorkItem(
                id=f"{operator}__{kernel_name}",
                operator=operator,
                kernel_name=kernel_name,
                cache_dir=str(operator_dir.relative_to(_PROJECT_ROOT)),
                shapes_dir=str(operator_dir.relative_to(_PROJECT_ROOT)),
                num_shapes_available=len(shape_paths),
            ))

    return items


def _auto_detect_gpu(ir_root: Path) -> str:
    """Auto-detect GPU directory name when there's only one."""
    candidates = [d.name for d in ir_root.iterdir() if d.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No GPU directories found in {ir_root}")
    raise ValueError(
        f"Multiple GPU directories found: {candidates}. "
        f"Use --gpu to specify which one."
    )


# ---------------------------------------------------------------------------
# Shape-level work queue
# ---------------------------------------------------------------------------

def flatten_to_shape_items(
    manifest: BatchManifest,
    all_shapes: bool,
    num_shapes: int,
) -> List[ShapeWorkItem]:
    """Expand pending WorkItems into a flat list of ShapeWorkItems.

    Each ShapeWorkItem represents a single (kernel, shape_file) pair so that
    a shared thread pool can balance work across kernels at shape granularity.
    """
    shape_items: List[ShapeWorkItem] = []

    for item in manifest.work_items:
        if item.status in ("success", "skipped"):
            continue

        shapes_dir = _PROJECT_ROOT / item.shapes_dir
        # Reuse the same discovery pattern as discover_kernels():
        # find per-input shapes files and filter to this kernel.
        all_shapes_files = sorted(shapes_dir.glob("*_input*_shapes.json"))
        kernel_shapes: List[Path] = []
        for sf in all_shapes_files:
            meta = parse_shapes_file(sf)
            if meta and meta["kernel_name"] == item.kernel_name:
                kernel_shapes.append(sf)

        # Apply shape limit
        if not all_shapes:
            kernel_shapes = kernel_shapes[:num_shapes]

        # Skip already-completed shapes (resume support)
        completed_stems = set(item.completed_shapes)

        for idx, sf in enumerate(kernel_shapes):
            if sf.stem in completed_stems:
                continue
            shape_items.append(ShapeWorkItem(
                kernel_id=item.id,
                operator=item.operator,
                kernel_name=item.kernel_name,
                cache_dir=item.cache_dir,
                shapes_dir=item.shapes_dir,
                shape_file=str(sf.relative_to(_PROJECT_ROOT)),
                output_dir=item.output_dir,
                shape_index=idx,
            ))

    return shape_items


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------

def build_command(item: WorkItem, args: argparse.Namespace) -> List[str]:
    """Build the multiagent_triton_runner.py command for a work item."""
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "multiagent_triton_runner.py"),
        "--cache-dir", str(_PROJECT_ROOT / item.cache_dir),
        "--input-shapes-dir", str(_PROJECT_ROOT / item.shapes_dir),
        "--kernels", item.kernel_name,
        "--output-dir", str(_PROJECT_ROOT / item.output_dir),
        "--target-gpu", args.target_gpu_arch,
        "--llm-provider", args.llm_provider,
        "--num-candidates", str(args.num_candidates),
        "--max-iterations", str(args.max_iterations),
    ]

    if args.all_shapes:
        cmd.append("--all-shape")
    else:
        cmd.extend(["--num-shapes", str(args.num_shapes)])

    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])

    if args.debug_mode:
        cmd.append("--debug-mode")

    if getattr(args, "skip_resource_budget", False):
        cmd.append("--skip-resource-budget")

    if getattr(args, "skip_data_movement", False):
        cmd.append("--skip-data-movement")

    if getattr(args, "include_agent_proposals", False):
        cmd.append("--include-agent-proposals")

    if getattr(args, "disable_negotiation", False):
        cmd.append("--disable-negotiation")

    return cmd


def build_shape_command(
    shape_item: ShapeWorkItem,
    args: argparse.Namespace,
    shape_output_dir: Path,
) -> List[str]:
    """Build the multiagent_triton_runner.py command for a single shape."""
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "multiagent_triton_runner.py"),
        "--cache-dir", str(_PROJECT_ROOT / shape_item.cache_dir),
        "--input-shapes-dir", str(_PROJECT_ROOT / shape_item.shapes_dir),
        "--kernels", shape_item.kernel_name,
        "--shape-file", str(_PROJECT_ROOT / shape_item.shape_file),
        "--output-dir", str(shape_output_dir),
        "--target-gpu", args.target_gpu_arch,
        "--llm-provider", args.llm_provider,
        "--num-candidates", str(args.num_candidates),
        "--max-iterations", str(args.max_iterations),
        "--num-shapes", "1",
    ]

    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])

    if args.debug_mode:
        cmd.append("--debug-mode")

    if getattr(args, "skip_resource_budget", False):
        cmd.append("--skip-resource-budget")

    if getattr(args, "skip_data_movement", False):
        cmd.append("--skip-data-movement")

    if getattr(args, "include_agent_proposals", False):
        cmd.append("--include-agent-proposals")

    if getattr(args, "disable_negotiation", False):
        cmd.append("--disable-negotiation")

    return cmd


def run_shape(
    shape_item: ShapeWorkItem,
    args: argparse.Namespace,
    collector: KernelResultCollector,
) -> None:
    """Execute multiagent_triton_runner.py for a single shape work item."""
    shape_output_dir = _PROJECT_ROOT / shape_item.output_dir / f"shape_{shape_item.shape_index}"
    shape_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_shape_command(shape_item, args, shape_output_dir)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=args.timeout,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)},
        )
        elapsed = round(time.time() - t0, 1)

        parsed = parse_runner_output(shape_output_dir)
        if parsed["per_shape_results"]:
            shape_result = parsed["per_shape_results"][0]
        elif proc.returncode != 0:
            shape_result = {
                "status": "failed",
                "error": f"exit code {proc.returncode}",
            }
        else:
            shape_result = {"status": "failed", "error": "no output parsed"}

    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - t0, 1)
        shape_result = {"status": "failed", "error": f"timeout after {elapsed}s"}
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        shape_result = {"status": "failed", "error": str(e)}

    shape_result["shape_file"] = Path(shape_item.shape_file).stem
    shape_result["duration_seconds"] = elapsed
    collector.record_shape_result(shape_item.kernel_id, shape_result)


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_runner_output(output_dir: Path) -> Dict[str, Any]:
    """Parse the runner's output to extract per-shape results.

    Reads live_summary.json (written by multiagent_triton_runner after each
    shape) to get per-shape speedups and aggregate stats.
    """
    result: Dict[str, Any] = {
        "per_shape_results": [],
        "best_speedup": None,
        "avg_speedup": None,
        "shapes_improved": 0,
        "shapes_regressed": 0,
        "shapes_kept_original": 0,
        "shapes_failed": 0,
    }

    live_summary = output_dir / "live_summary.json"
    if not live_summary.exists():
        return result

    try:
        data = json.loads(live_summary.read_text())
    except (json.JSONDecodeError, OSError):
        return result

    per_shape = data.get("per_shape", [])
    speedups: List[float] = []

    for entry in per_shape:
        shape_result: Dict[str, Any] = {
            "shape_file": entry.get("shape", ""),
            "success": entry.get("success", False),
        }

        if not entry.get("success"):
            shape_result["status"] = "failed"
            shape_result["error"] = entry.get("error")
            result["shapes_failed"] += 1
        else:
            speedup = entry.get("speedup")
            recommendation = entry.get("recommendation", "optimize")
            shape_result["speedup"] = speedup
            shape_result["recommendation"] = recommendation

            if recommendation == "keep_original":
                shape_result["status"] = "kept_original"
                shape_result["speedup"] = speedup if speedup is not None else 1.0
                result["shapes_kept_original"] += 1
                speedups.append(speedup if speedup is not None else 1.0)
            elif speedup is not None and speedup > 1.0:
                shape_result["status"] = "improved"
                result["shapes_improved"] += 1
            elif speedup is not None and speedup < 1.0:
                shape_result["status"] = "regressed"
                result["shapes_regressed"] += 1
            else:
                shape_result["status"] = "unchanged"

            if speedup is not None:
                speedups.append(speedup)

        result["per_shape_results"].append(shape_result)

    if speedups:
        result["best_speedup"] = max(speedups)
        result["avg_speedup"] = round(sum(speedups) / len(speedups), 4)

    return result


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_kernel(
    item: WorkItem,
    args: argparse.Namespace,
    manifest: BatchManifest,
    manifest_path: Path,
) -> None:
    """Execute multiagent_triton_runner.py for a single kernel work item."""
    with manifest._lock:
        item.started_at = datetime.now(timezone.utc).isoformat()
        item.status = "running"
    manifest.save(manifest_path)

    cmd = build_command(item, args)
    item.command = " ".join(cmd)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=args.timeout,
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            env={
                **os.environ,
                "PYTHONPATH": str(_PROJECT_ROOT),
            },
        )
        item.runner_exit_code = proc.returncode

        # Print subprocess output for visibility
        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines()[-10:]:
                print(f"    {line}")
        if proc.returncode != 0 and proc.stderr.strip():
            stderr_lines = proc.stderr.strip().splitlines()
            for line in stderr_lines[-5:]:
                log.warning("  [stderr] %s", line)

        # Parse results
        kernel_output_dir = _PROJECT_ROOT / item.output_dir
        parsed = parse_runner_output(kernel_output_dir)

        with manifest._lock:
            item.per_shape_results = parsed["per_shape_results"]
            item.best_speedup = parsed["best_speedup"]
            item.avg_speedup = parsed["avg_speedup"]
            item.shapes_improved = parsed["shapes_improved"]
            item.shapes_regressed = parsed["shapes_regressed"]
            item.shapes_kept_original = parsed["shapes_kept_original"]
            item.shapes_failed = parsed["shapes_failed"]

            successful_shapes = len(parsed["per_shape_results"]) - parsed["shapes_failed"]
            if proc.returncode == 0:
                item.status = "success"
            elif successful_shapes > 0:
                # Partial success: some shapes worked, some failed.
                # Record as success so the good results are counted.
                # Failed shapes are recorded in per_shape_results for inspection.
                item.status = "success"
                item.error = (
                    f"{parsed['shapes_failed']} of "
                    f"{len(parsed['per_shape_results'])} shapes failed"
                )
            else:
                item.status = "failed"
                item.error = f"Exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        item.status = "failed"
        item.error = f"Timeout after {args.timeout}s"
    except Exception as e:
        item.status = "failed"
        item.error = str(e)

    with manifest._lock:
        item.completed_at = datetime.now(timezone.utc).isoformat()
        item.duration_seconds = round(time.time() - t0, 1)
    manifest.save(manifest_path)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_per_kernel_result(item: WorkItem, index: int, total: int) -> None:
    """Print inline result after a kernel completes."""
    status_str = item.status.upper()
    duration = f"{item.duration_seconds:.1f}s" if item.duration_seconds else "—"
    num_shapes = len(item.per_shape_results) if item.per_shape_results else 0
    num_ok = num_shapes - item.shapes_failed

    # Show "x/y shapes" when there are failures
    if item.shapes_failed > 0:
        shapes_str = f"{num_ok}/{num_shapes} shapes ok, {item.shapes_failed} failed"
    else:
        shapes_str = f"{num_shapes} shapes"

    print(f"\n[{index}/{total}] {item.operator} / {item.kernel_name} "
          f"({shapes_str}) — {duration} [{status_str}]")

    # Per-shape detail
    for sr in item.per_shape_results:
        shape_name = sr.get("shape_file", "?")
        if sr.get("status") == "failed":
            print(f"    {shape_name}: FAILED — {sr.get('error', 'unknown error')}")
        else:
            speedup = sr.get("speedup")
            speedup_str = f"{speedup:.2f}x" if speedup is not None else "—"
            status = sr.get("status", "?")
            print(f"    {shape_name}: {speedup_str} ({status})")

    # Aggregate (only count successful shapes)
    if item.best_speedup is not None:
        avg_str = f"{item.avg_speedup:.2f}x" if item.avg_speedup else "—"
        print(f"    Best: {item.best_speedup:.2f}x | Avg: {avg_str} | "
              f"Improved: {item.shapes_improved}/{num_ok}")


def print_batch_summary(manifest: BatchManifest) -> None:
    """Print the final batch-level summary."""
    summary = manifest._compute_summary()

    print(f"\n{'='*90}")
    print(f"  Batch Summary — {manifest.gpu} ({manifest.target_gpu_arch})")
    print(f"{'='*90}")

    # Header
    print(f"  {'Operator':<22} {'Kernel':<40} {'Status':<9} "
          f"{'Best':>6} {'Avg':>6} {'Shapes':>8}")
    print(f"  {'-'*22} {'-'*40} {'-'*9} {'-'*6} {'-'*6} {'-'*8}")

    all_speedups = []
    total_shapes_improved = 0
    total_shapes_processed = 0

    for item in manifest.work_items:
        if item.status == "skipped":
            continue

        best_str = f"{item.best_speedup:.2f}x" if item.best_speedup else "—"
        avg_str = f"{item.avg_speedup:.2f}x" if item.avg_speedup else "—"
        num_shapes = len(item.per_shape_results) if item.per_shape_results else 0
        shapes_ok = item.shapes_improved + item.shapes_kept_original + item.shapes_regressed
        shapes_str = f"{shapes_ok}/{num_shapes}" if num_shapes else "—"

        # Truncate long kernel names
        kernel_display = item.kernel_name[:38] + ".." if len(item.kernel_name) > 40 else item.kernel_name

        print(f"  {item.operator:<22} {kernel_display:<40} {item.status:<9} "
              f"{best_str:>6} {avg_str:>6} {shapes_str:>8}")

        if item.best_speedup is not None:
            all_speedups.append(item.best_speedup)
        total_shapes_improved += item.shapes_improved
        total_shapes_processed += num_shapes

    print(f"{'='*90}")
    print(f"  Total: {summary['total_kernels']} | "
          f"Success: {summary['success']} | "
          f"Failed: {summary['failed']} | "
          f"Skipped: {summary['skipped']}")

    if all_speedups:
        avg_best = sum(all_speedups) / len(all_speedups)
        print(f"  Improved shapes: {total_shapes_improved}/{total_shapes_processed} | "
              f"Avg best speedup: {avg_best:.2f}x")

    print(f"{'='*90}")


def print_discovery_summary(items: List[WorkItem], gpu: str) -> None:
    """Print summary of discovered kernels."""
    operators = defaultdict(list)
    for item in items:
        operators[item.operator].append(item)

    total_shapes = sum(i.num_shapes_available for i in items)
    runnable = [i for i in items if i.status != "skipped"]

    # Count by source type
    inductor_count = sum(1 for i in runnable if classify_kernel_source(i.kernel_name) == "inductor")
    handwritten_count = sum(1 for i in runnable if classify_kernel_source(i.kernel_name) == "handwritten")

    print(f"\n{'='*80}")
    print(f"  Discovery Summary — GPU: {gpu}")
    print(f"{'='*80}")
    print(f"  Operators:        {len(operators)}")
    print(f"  Total kernels:    {len(items)}")
    print(f"  Runnable:         {len(runnable)} (inductor: {inductor_count}, handwritten: {handwritten_count})")
    print(f"  Skipped:          {len(items) - len(runnable)}")
    print(f"  Total shapes:     {total_shapes}")
    print(f"{'='*80}")
    print(f"  {'Operator':<25} {'Kernels':>8} {'Shapes':>8}  {'Source'}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}  {'-'*12}")
    for op_name, op_items in sorted(operators.items()):
        shapes = sum(i.num_shapes_available for i in op_items)
        # Show source types for this operator's runnable kernels
        sources = set(
            classify_kernel_source(i.kernel_name)
            for i in op_items if i.status != "skipped"
        )
        source_str = ", ".join(sorted(sources)) if sources else "skipped"
        print(f"  {op_name:<25} {len(op_items):>8} {shapes:>8}  {source_str}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for multi-agent Triton optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root output directory for batch results",
    )

    # Discovery
    parser.add_argument(
        "--ir-dir", type=Path, default=_PROJECT_ROOT / "extract" / "ir_output",
        help="IR output root directory (default: extract/ir_output)",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU folder name: v100, A40, l40s (auto-detect if only one)",
    )

    # Filtering
    parser.add_argument(
        "--operators", nargs="+", default=None,
        help="Only process these operators",
    )
    parser.add_argument(
        "--exclude-operators", nargs="+", default=None,
        help="Skip these operators",
    )
    parser.add_argument(
        "--max-kernels-per-op", type=int, default=None,
        help="Max kernels per operator (useful for capping inductor kernel count)",
    )
    parser.add_argument(
        "--kernel-source", type=str, default="all",
        choices=["all", "inductor", "handwritten"],
        help="Filter by kernel source: 'inductor' (torch.compile/pt2 generated), "
             "'handwritten' (manually written Triton), or 'all' (default: all)",
    )
    parser.add_argument(
        "--run-list", type=Path, default=None,
        help="JSON file specifying kernels to include or exclude. "
             "See run_lists/ for examples.",
    )

    # Shapes
    parser.add_argument(
        "--num-shapes", type=int, default=1,
        help="Number of input shapes per kernel (default: 1)",
    )
    parser.add_argument(
        "--all-shapes", action="store_true",
        help="Process all available shapes per kernel",
    )

    # Optimizer
    parser.add_argument(
        "--llm-provider", type=str, default="deepseek",
        choices=["deepseek", "openai", "ollama", "vllm"],
        help="LLM provider (default: deepseek)",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=3,
        help="Candidates per optimization round (default: 3)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1,
        help="Max refinement rounds (default: 1)",
    )
    parser.add_argument(
        "--skip-resource-budget", action="store_true",
        help="Ablation A1a: remove ResourceBudgetAgent (single-agent DM only)",
    )
    parser.add_argument(
        "--skip-data-movement", action="store_true",
        help="Ablation A1b: remove DataMovementAgent (single-agent RB only)",
    )
    parser.add_argument(
        "--include-agent-proposals", action="store_true",
        help="Inject each agent's preferred config as benchmark candidates",
    )
    parser.add_argument(
        "--disable-negotiation", action="store_true",
        help="Skip negotiation sub-loop between agents",
    )
    parser.add_argument(
        "--target-gpu-arch", type=str, default=None,
        help="Override GPU arch code (e.g., cuda:70). Auto-detected from --gpu.",
    )

    # Execution
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover and print commands without executing",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="Resume from existing manifest (default: True)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Force re-run everything",
    )
    parser.add_argument(
        "--timeout", type=int, default=21600,
        help="Per-kernel timeout in seconds (default: 21600 = 6h)",
    )
    parser.add_argument(
        "--debug-mode", action="store_true",
        help="Pass --debug-mode to the runner",
    )
    parser.add_argument(
        "--parallel-kernels", type=int, default=1,
        help="Number of kernels to process in parallel (default: 1, sequential). "
             "GPU benchmarks are automatically serialized via file lock.",
    )
    parser.add_argument(
        "--parallel-shapes", type=int, default=None,
        help="Number of shapes to process in parallel across all kernels. "
             "Creates a flat queue of (kernel, shape) pairs for maximum "
             "worker utilization when kernels have unequal shape counts. "
             "Mutually exclusive with --parallel-kernels > 1.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()

    if args.parallel_shapes and args.parallel_kernels > 1:
        parser.error("--parallel-shapes and --parallel-kernels > 1 are mutually exclusive")

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Resolve GPU ---
    ir_root = args.ir_dir.resolve()
    if args.gpu:
        gpu_name = args.gpu
    else:
        gpu_name = _auto_detect_gpu(ir_root)

    gpu_dir = ir_root / gpu_name
    if not gpu_dir.is_dir():
        print(f"Error: GPU directory not found: {gpu_dir}")
        sys.exit(1)

    # Resolve target GPU arch
    if args.target_gpu_arch:
        target_arch = args.target_gpu_arch
    else:
        gpu_lower = gpu_name.lower()
        if gpu_lower in GPU_ARCH_MAP:
            target_arch = GPU_ARCH_MAP[gpu_lower]
        else:
            print(f"Error: Unknown GPU '{gpu_name}'. "
                  f"Use --target-gpu-arch to specify arch code.")
            print(f"Known GPUs: {list(GPU_ARCH_MAP.keys())}")
            sys.exit(1)

    args.target_gpu_arch = target_arch

    # --- Discovery ---
    print(f"\nBatch Experiment Runner")
    print(f"  IR directory:   {gpu_dir}")
    print(f"  GPU:            {gpu_name} ({target_arch})")
    print(f"  Output:         {args.output_dir}")
    print(f"  LLM provider:   {args.llm_provider}")
    print(f"  Kernel source:  {args.kernel_source}")
    print(f"  Resume:         {args.resume}")

    items = discover_kernels(gpu_dir)
    if not items:
        print("\nNo kernels discovered. Nothing to do.")
        return

    # --- Apply filters ---
    if args.operators:
        op_set = set(args.operators)
        items = [i for i in items if i.operator in op_set]
    if args.exclude_operators:
        exclude_set = set(args.exclude_operators)
        items = [i for i in items if i.operator not in exclude_set]

    # Filter by kernel source (inductor vs handwritten)
    if args.kernel_source != "all":
        filtered = []
        for item in items:
            if item.status == "skipped":
                filtered.append(item)
                continue
            source = classify_kernel_source(item.kernel_name)
            if source == args.kernel_source:
                filtered.append(item)
            else:
                log.debug("Filtering out %s/%s (source=%s, want=%s)",
                          item.operator, item.kernel_name, source, args.kernel_source)
        items = filtered
        if not items:
            print(f"\nNo {args.kernel_source} kernels found after filtering.")
            return

    # Filter by run-list (JSON file)
    if args.run_list:
        run_list = load_run_list(args.run_list)
        mode = run_list.get("mode", "include")
        desc = run_list.get("description", args.run_list.name)
        print(f"  Run-list:       {args.run_list} (mode={mode})")
        print(f"                  {desc}")

        filtered = []
        included_count = 0
        excluded_count = 0
        for item in items:
            if item.status == "skipped":
                filtered.append(item)
                continue
            matches = run_list_matches(run_list, item.operator, item.kernel_name)
            if mode == "include":
                if matches:
                    filtered.append(item)
                    included_count += 1
                else:
                    log.debug("Run-list exclude (not in include list): %s/%s",
                              item.operator, item.kernel_name)
                    excluded_count += 1
            else:  # exclude
                if matches:
                    item.status = "skipped"
                    item.error = f"Excluded by run-list: {args.run_list.name}"
                    filtered.append(item)
                    excluded_count += 1
                else:
                    filtered.append(item)
                    included_count += 1
        items = filtered
        print(f"  Run-list filter: {included_count} kept, {excluded_count} removed")
        if not [i for i in items if i.status != "skipped"]:
            print(f"\nNo kernels remaining after run-list filter.")
            return

    # Cap kernels per operator
    if args.max_kernels_per_op is not None:
        filtered = []
        op_counts: Dict[str, int] = defaultdict(int)
        for item in items:
            if item.status == "skipped":
                filtered.append(item)
                continue
            if op_counts[item.operator] < args.max_kernels_per_op:
                filtered.append(item)
                op_counts[item.operator] += 1
            else:
                item.status = "skipped"
                item.error = f"Capped at {args.max_kernels_per_op} kernels per operator"
                filtered.append(item)
        items = filtered

    print_discovery_summary(items, gpu_name)

    # Set output dirs and num_shapes for each item
    for item in items:
        item.output_dir = str(
            args.output_dir / item.operator / item.kernel_name
        )
        if args.all_shapes:
            item.num_shapes_to_process = item.num_shapes_available
        else:
            item.num_shapes_to_process = min(args.num_shapes, item.num_shapes_available)

    # --- Manifest ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "batch_manifest.json"

    if args.resume and manifest_path.exists():
        manifest = BatchManifest.load(manifest_path)
        log.info("Loaded existing manifest with %d items", len(manifest.work_items))

        # Merge: add new items, preserve status of existing ones
        existing_ids = {w.id for w in manifest.work_items}
        for item in items:
            if item.id not in existing_ids:
                manifest.work_items.append(item)

        # Reset any "running" items (crash recovery)
        for w in manifest.work_items:
            if w.status == "running":
                log.info("Resetting crashed item: %s", w.id)
                w.status = "pending"
    else:
        manifest = BatchManifest(
            gpu=gpu_name,
            target_gpu_arch=target_arch,
            config={
                "llm_provider": args.llm_provider,
                "llm_model": args.llm_model,
                "num_candidates": args.num_candidates,
                "max_iterations": args.max_iterations,
                "num_shapes": args.num_shapes,
                "all_shapes": args.all_shapes,
                "debug_mode": args.debug_mode,
                "timeout": args.timeout,
                "parallel_kernels": args.parallel_kernels,
                "parallel_shapes": args.parallel_shapes,
            },
            work_items=items,
        )

    manifest.save(manifest_path)

    # --- Dry run ---
    if args.dry_run:
        print("\n--- DRY RUN: Commands that would be executed ---\n")
        runnable = [w for w in manifest.work_items
                    if w.status not in ("success", "skipped")]
        for i, item in enumerate(runnable, 1):
            cmd = build_command(item, args)
            print(f"[{i}/{len(runnable)}] {item.operator} / {item.kernel_name}")
            print(f"  {' '.join(cmd)}")
            print()
        print(f"Total: {len(runnable)} kernels to run")
        return

    # --- Execute ---
    pending = [w for w in manifest.work_items
               if w.status in ("pending",)]
    total = len(manifest.work_items)
    skip_count = sum(1 for w in manifest.work_items if w.status == "success")
    if skip_count:
        print(f"\nResuming: skipping {skip_count} already-completed kernels")

    # Set GPU benchmark lock path so child subprocesses inherit it.
    # All parallel subprocesses share this lock to serialize GPU benchmarks.
    lock_path = str(args.output_dir.resolve() / ".gpu_benchmark.lock")
    os.environ["GPU_BENCHMARK_LOCK_PATH"] = lock_path

    n_parallel = args.parallel_kernels
    n_shape_parallel = args.parallel_shapes

    if n_shape_parallel:
        print(f"\nRunning shapes from {len(pending)} kernels "
              f"({n_shape_parallel} parallel shape workers)...\n")
    elif n_parallel > 1:
        print(f"\nRunning {len(pending)} kernels "
              f"({n_parallel} parallel workers)...\n")
    else:
        print(f"\nRunning {len(pending)} kernels...\n")

    try:
        if n_shape_parallel:
            # --- Shape-level parallel execution ---
            # Mark all pending kernels as "running" and ensure output dirs
            for item in pending:
                with manifest._lock:
                    item.status = "running"
                    item.started_at = datetime.now(timezone.utc).isoformat()
                (_PROJECT_ROOT / item.output_dir).mkdir(
                    parents=True, exist_ok=True,
                )
            manifest.save(manifest_path)

            # Build flat work queue of (kernel, shape) pairs
            shape_items = flatten_to_shape_items(
                manifest, args.all_shapes, args.num_shapes,
            )

            if not shape_items:
                print("No shapes to process (all completed or none discovered).")
            else:
                # Initialise collector and register kernels
                collector = KernelResultCollector(manifest, manifest_path)
                kernel_shape_counts: Dict[str, int] = defaultdict(int)
                for si in shape_items:
                    kernel_shape_counts[si.kernel_id] += 1
                for kid, count in kernel_shape_counts.items():
                    collector.register_kernel(kid, count)
                    # Pre-populate with existing results on resume
                    item = manifest.get_item(kid)
                    if item and item.per_shape_results:
                        collector.pre_populate(kid, item.per_shape_results)

                total_shapes = len(shape_items)
                print(f"  {total_shapes} shapes across "
                      f"{len(kernel_shape_counts)} kernels\n")

                completed_count = 0
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_shape_parallel,
                ) as executor:
                    futures = {
                        executor.submit(run_shape, si, args, collector): si
                        for si in shape_items
                    }
                    for future in concurrent.futures.as_completed(futures):
                        si = futures[future]
                        completed_count += 1
                        try:
                            future.result()
                        except Exception as e:
                            collector.record_shape_result(si.kernel_id, {
                                "status": "failed",
                                "error": str(e),
                                "shape_file": Path(si.shape_file).stem,
                            })
                        print(f"  [{completed_count}/{total_shapes}] "
                              f"{si.operator}/{si.kernel_name} "
                              f"shape {si.shape_index}")

                # Print per-kernel summaries
                for item in pending:
                    global_idx = next(
                        idx for idx, w in enumerate(manifest.work_items, 1)
                        if w.id == item.id
                    )
                    print_per_kernel_result(item, global_idx, total)

        elif n_parallel > 1:
            # --- Kernel-level parallel execution ---
            # Ensure all output dirs exist before submitting
            for item in pending:
                (_PROJECT_ROOT / item.output_dir).mkdir(
                    parents=True, exist_ok=True,
                )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_parallel,
            ) as executor:
                futures = {}
                for item in pending:
                    future = executor.submit(
                        run_kernel, item, args, manifest, manifest_path,
                    )
                    futures[future] = item

                for future in concurrent.futures.as_completed(futures):
                    item = futures[future]
                    global_idx = next(
                        idx for idx, w in enumerate(manifest.work_items, 1)
                        if w.id == item.id
                    )
                    try:
                        future.result()
                    except Exception as e:
                        with manifest._lock:
                            item.status = "failed"
                            item.error = str(e)
                        manifest.save(manifest_path)
                    print_per_kernel_result(item, global_idx, total)
        else:
            # --- Sequential execution (original path) ---
            for i, item in enumerate(pending, 1):
                global_idx = next(
                    idx for idx, w in enumerate(manifest.work_items, 1)
                    if w.id == item.id
                )
                print(f"\n--- [{global_idx}/{total}] "
                      f"{item.operator} / {item.kernel_name} "
                      f"({item.num_shapes_to_process} shapes) ---")

                # Ensure output dir exists
                kernel_output = _PROJECT_ROOT / item.output_dir
                kernel_output.mkdir(parents=True, exist_ok=True)

                run_kernel(item, args, manifest, manifest_path)
                print_per_kernel_result(item, global_idx, total)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving manifest...")
        # Mark current running items as failed
        with manifest._lock:
            for w in manifest.work_items:
                if w.status == "running":
                    w.status = "failed"
                    w.error = "Interrupted by user"
                    w.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.save(manifest_path)
        print(f"Manifest saved to {manifest_path}")
        print("Re-run with --resume to continue.")
        sys.exit(130)

    # --- Final summary ---
    manifest.save(manifest_path)
    print_batch_summary(manifest)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
