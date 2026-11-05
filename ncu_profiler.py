"""Optional NCU profiling for the multi-agent feedback loop.

When enabled via ``--enable-profiling``, the orchestrator profiles the best
benchmark candidate with NVIDIA Nsight Compute (``ncu``) and extracts
hardware metrics that ``construct_agent_feedback()`` routes to each agent.

The current timing-based benchmark (``benchmark_config_pair``) remains the
default and always runs.  NCU profiling supplements it with detailed metrics
so agents can understand *why* a config regressed, not just that it did.

Requires ``ncu`` on PATH.  Fails gracefully (returns ``None``) when NCU is
unavailable or profiling errors occur.
"""

import csv
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NCU metric name -> our internal key
# ---------------------------------------------------------------------------
# Keys starting with "_" are intermediates used to compute derived metrics.
_NCU_METRIC_MAP: Dict[str, str] = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "dram__bytes.sum.per_second": "_dram_bytes_per_sec",
    # L2 hit/miss (sector-level)
    "lts__t_sectors_lookup_hit.sum": "_l2_hit",
    "lts__t_sectors_lookup_miss.sum": "_l2_miss",
    # L1/TEX hit/miss for global loads
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum": "_l1_hit",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum": "_l1_miss",
    # Warp stall reasons (as % of peak sustained)
    "smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active": "_stall_memory",
    "smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active": "_stall_short_scoreboard",
    "smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_active": "_stall_barrier",
    "smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active": "_stall_not_selected",
}

# Reverse map for quick lookup during CSV parsing
_METRIC_NAME_TO_KEY = {v: k for k, v in _NCU_METRIC_MAP.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_ncu_available() -> bool:
    """Check if the ``ncu`` command is available on PATH."""
    return shutil.which("ncu") is not None


def profile_with_ncu(
    config: Dict[str, Any],
    shapes_file: Path,
    kernel_module: Path,
    kernel_name: str,
    kernel_class: Optional[str] = None,
    device: str = "cuda",
    peak_memory_bandwidth_gb_s: float = 900.0,
) -> Optional[Dict[str, Any]]:
    """Profile a single kernel config with NCU and extract agent-relevant metrics.

    Returns a dict with keys matching ``construct_agent_feedback()`` expectations:

    - ``achieved_occupancy_pct`` (float): Warp occupancy as % of peak.
    - ``registers_per_thread`` (int): Registers allocated per thread.
    - ``global_memory_throughput_pct`` (float): DRAM throughput as % of peak.
    - ``l2_hit_rate`` (float): L2 cache hit rate %.
    - ``l1_hit_rate`` (float): L1/TEX cache hit rate %.
    - ``warp_stall_reasons`` (dict): Stall breakdown ``{reason: "X%", ...}``.

    Returns ``None`` on failure (NCU not available, profiling error, etc.).
    """
    if not is_ncu_available():
        logger.warning("NCU not found on PATH; skipping profiling")
        return None

    try:
        script = _generate_profiling_script(
            kernel_module=kernel_module,
            kernel_name=kernel_name,
            kernel_class=kernel_class,
            shapes_file=shapes_file,
            config=config,
            device=device,
        )
        csv_text = _run_ncu(script)
        if not csv_text:
            return None

        raw = _parse_ncu_csv(csv_text)
        if not raw:
            logger.warning("NCU CSV parsing returned no metrics")
            return None

        return _compute_derived_metrics(raw, peak_memory_bandwidth_gb_s)
    except Exception as e:
        logger.warning("NCU profiling failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _generate_profiling_script(
    kernel_module: Path,
    kernel_name: str,
    kernel_class: Optional[str],
    shapes_file: Path,
    config: Dict[str, Any],
    device: str,
) -> str:
    """Generate a Python script that loads and runs the kernel twice (warmup + profiled)."""
    agent_dir = str(Path(__file__).resolve().parent)
    kernel_class_repr = repr(kernel_class) if kernel_class else "None"
    config_json = json.dumps(config)

    return f'''\
import sys
sys.path.insert(0, {repr(agent_dir)})

from pathlib import Path
import torch
from compare_configs import _load_kernel, _build_kwargs, _map_block_sizes, _unwrap_autotuner
from tensor_utils import create_kernel_inputs, load_tensor_shapes

# Load kernel
shapes_file = Path({repr(str(shapes_file))})
kernel_module = Path({repr(str(kernel_module))})
metadata = load_tensor_shapes(shapes_file)
grid = tuple(metadata["grid"])
shapes_kwargs = metadata.get("kwargs", {{}})

input_args, input_kwargs = create_kernel_inputs(
    shapes_file, device={repr(device)}, values="random", seed=42,
)
kernel_fn = _load_kernel(kernel_module, {repr(kernel_name)}, {kernel_class_repr})
kernel_fn, _ = _unwrap_autotuner(kernel_fn)

config = {config_json}
kwargs = _build_kwargs(config, input_kwargs, shapes_kwargs)

# Warmup invocation (JIT compile + cache)
kernel_fn[grid](*input_args, **kwargs)
torch.cuda.synchronize()

# Profiled invocation
kernel_fn[grid](*input_args, **kwargs)
torch.cuda.synchronize()
'''


def _run_ncu(script_content: str) -> Optional[str]:
    """Write script to temp file, run under ``ncu --csv --set full``, return CSV stdout."""
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="ncu_profile_")
        with os.fdopen(tmp_fd, "w") as f:
            f.write(script_content)
        tmp_fd = None  # closed by os.fdopen

        cmd = [
            "ncu",
            "--csv",
            "--set", "full",
            "--launch-skip", "1",
            "--launch-count", "1",
            sys.executable if hasattr(sys, "executable") else "python",
            tmp_path,
        ]

        logger.info("Running NCU: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            logger.warning(
                "NCU exited with code %d.\nstderr: %s",
                proc.returncode,
                proc.stderr[:500] if proc.stderr else "(empty)",
            )
            # NCU may still produce valid CSV even with non-zero exit
            if not proc.stdout.strip():
                return None

        return proc.stdout

    except subprocess.TimeoutExpired:
        logger.warning("NCU profiling timed out after 120s")
        return None
    except FileNotFoundError:
        logger.warning("NCU command not found")
        return None
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Need sys for sys.executable in _run_ncu
import sys  # noqa: E402


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse NCU ``--csv`` output and extract target metric values.

    NCU CSV format has a header row that may be preceded by comment lines
    starting with ``==``.  The data rows contain columns including
    "Metric Name" and "Metric Value".

    For metrics that appear multiple times (e.g., from multiple kernel
    launches), we take the **last** value (the profiled invocation).
    """
    result: Dict[str, float] = {}

    # Strip NCU comment/banner lines (start with "==")
    lines = [
        line for line in csv_text.splitlines()
        if line.strip() and not line.startswith("==")
    ]
    if not lines:
        return result

    csv_content = "\n".join(lines)
    reader = csv.DictReader(io.StringIO(csv_content))

    # Find the column names for metric name and value
    # NCU uses quoted headers; DictReader handles this
    metric_name_col = None
    metric_value_col = None

    if reader.fieldnames:
        for col in reader.fieldnames:
            col_lower = col.strip().lower().replace('"', '')
            if col_lower == "metric name":
                metric_name_col = col
            elif col_lower == "metric value":
                metric_value_col = col

    if not metric_name_col or not metric_value_col:
        logger.warning(
            "NCU CSV missing 'Metric Name' or 'Metric Value' columns. "
            "Available: %s",
            reader.fieldnames,
        )
        return result

    for row in reader:
        ncu_metric = row.get(metric_name_col, "").strip()
        if ncu_metric in _NCU_METRIC_MAP:
            our_key = _NCU_METRIC_MAP[ncu_metric]
            raw_value = row.get(metric_value_col, "").strip()
            try:
                # NCU may format numbers with commas (e.g., "1,234,567")
                cleaned = raw_value.replace(",", "")
                result[our_key] = float(cleaned)
            except (ValueError, TypeError):
                logger.debug(
                    "Could not parse metric %s value %r", ncu_metric, raw_value,
                )

    return result


def _compute_derived_metrics(
    raw: Dict[str, float],
    peak_bw_gb_s: float,
) -> Dict[str, Any]:
    """Compute agent-facing metrics from raw NCU values.

    Returns a dict with keys matching ``construct_agent_feedback()``
    expectations.  Only includes metrics for which raw data is available.
    """
    result: Dict[str, Any] = {}

    # --- Direct pass-through metrics ---
    if "achieved_occupancy_pct" in raw:
        result["achieved_occupancy_pct"] = raw["achieved_occupancy_pct"]

    if "registers_per_thread" in raw:
        result["registers_per_thread"] = int(raw["registers_per_thread"])

    # --- Global memory throughput as % of peak ---
    dram_bps = raw.get("_dram_bytes_per_sec")
    if dram_bps is not None and peak_bw_gb_s > 0:
        result["global_memory_throughput_pct"] = (
            dram_bps / (peak_bw_gb_s * 1e9) * 100.0
        )

    # --- L2 cache hit rate ---
    l2_hit = raw.get("_l2_hit", 0)
    l2_miss = raw.get("_l2_miss", 0)
    if l2_hit + l2_miss > 0:
        result["l2_hit_rate"] = l2_hit / (l2_hit + l2_miss) * 100.0

    # --- L1/TEX cache hit rate ---
    l1_hit = raw.get("_l1_hit", 0)
    l1_miss = raw.get("_l1_miss", 0)
    if l1_hit + l1_miss > 0:
        result["l1_hit_rate"] = l1_hit / (l1_hit + l1_miss) * 100.0

    # --- Warp stall reasons ---
    stall_map = {
        "_stall_memory": "memory",
        "_stall_short_scoreboard": "short_scoreboard",
        "_stall_barrier": "barrier",
        "_stall_not_selected": "not_selected",
    }
    stalls: Dict[str, str] = {}
    for raw_key, label in stall_map.items():
        if raw_key in raw:
            stalls[label] = f"{raw[raw_key]:.1f}%"
    if stalls:
        result["warp_stall_reasons"] = stalls

    return result
