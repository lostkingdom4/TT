"""TritonBench operator setup helpers.

Centralizes the ``--device / --precision / --metrics`` args,
``fresh_triton_cache`` context, and operator-class instantiation used
by every baseline's subprocess runner.
"""

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from paths import TRITONBENCH_ROOT  # noqa: E402

_TB_ROOT = str(TRITONBENCH_ROOT)
if _TB_ROOT not in sys.path:
    sys.path.insert(0, _TB_ROOT)

log = logging.getLogger(__name__)


def build_tb_args(
    device: str = "cuda",
    precision: str = "fp16",
    metrics: str = "latency,best_config",
    warmup: int = 5,
    rep: int = 20,
):
    """Parse TritonBench CLI args with the paper's standard settings."""
    from tritonbench.utils.parser import get_parser
    parser = get_parser()
    tb_args, extra_args = parser.parse_known_args([
        "--device", device,
        "--precision", precision,
        "--metrics", metrics,
    ])
    tb_args.warmup = warmup
    tb_args.rep = rep
    return tb_args, extra_args


def load_op_class(operator: str):
    """Import TritonBench, return the op class for ``operator``."""
    from tritonbench.operators.op import load_opbench_by_name
    OpClass = load_opbench_by_name(operator)
    if OpClass is None:
        raise RuntimeError(
            f"Operator '{operator}' not found in TritonBench"
        )
    return OpClass


def fresh_cache_context():
    """No-op context manager — retained so call sites don't change.

    We used to wrap each per-shape run in ``fresh_triton_cache()`` which
    installs a tmpdir ``TRITON_CACHE_DIR`` and forces every (kernel,
    config) to recompile.  That has **no effect on measured latency**
    (the JIT is deterministic per config hash), but it makes sweeps
    30-60 s slower per config per subprocess restart and mismatches the
    Full system, which reuses the persistent ``~/.triton/cache/``.
    Matching Full's cache behavior is the fair-comparison default.
    Warmup + repeated-rep median + the L2 flush inside
    ``benchmark_config_pair`` handle runtime-state contamination.
    """
    return contextlib.nullcontext()


def extract_latency(metrics) -> float:
    """Median latency (ms) from a ``BenchmarkOperatorMetrics`` row."""
    lat = getattr(metrics, "latency", None)
    if lat is None:
        return None
    if hasattr(lat, "p50"):
        return float(lat.p50)
    try:
        return float(lat)
    except (TypeError, ValueError):
        return None


def jsonable(v) -> Any:
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)
