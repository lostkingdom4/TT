"""Latency cache backed by the autotune baseline's exhaustive sweep.

The autotune baseline
(``baselines/results/<device>/autotune/<op>__<kernel>.json``) records the
latency of every config it evaluated per shape.  When ``config_source``
is ``"common_search_space"`` the autotune runner exhaustively measured
every config in ``baselines.common.search_space.get_search_space(op,
kernel)`` once per shape.

This module loads that cache and exposes a single ``lookup(op, kernel,
shape_id, config)`` that returns the cached latency without touching
the GPU.  Used by ``baselines.run_cached`` to replay strategies off the
cache on a busy machine.

Kernels whose cache ``config_source`` is ``"autotune_decorator"`` (the
autotune ran @triton.autotune's hand-picked config list, e.g. the 8
default GEMM configs) expose a restricted coverage that will NOT cover
the full ``common_search_space``.  ``coverage(op, kernel)`` reports
this so callers can skip such kernels cleanly.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger(__name__)


def _normalize_cfg_key(cfg: Dict[str, Any]) -> Tuple:
    """Hashable + type-normalized key for a config dict.

    JSON-serialized configs may carry ``XBLOCK=64`` as ``int`` but
    ``num_warps="4"`` as ``str`` (or vice versa).  We normalize to the
    Python type that matches the most common representation -- int when
    the value is integral, otherwise the original.
    """
    out = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, bool):
            out.append((k, v))
        elif isinstance(v, (int, float)):
            if float(v).is_integer():
                out.append((k, int(v)))
            else:
                out.append((k, float(v)))
        elif isinstance(v, str):
            try:
                out.append((k, int(v)))
            except ValueError:
                try:
                    out.append((k, float(v)))
                except ValueError:
                    out.append((k, v))
        else:
            out.append((k, v))
    return tuple(out)


@dataclass
class ShapeCache:
    input_id: Any
    x_val: Optional[str]
    original_latency_ms: Optional[float]
    reference_config: Optional[Dict[str, Any]] = None
    config_source: Optional[str] = None
    n_configs_expected: Optional[int] = None
    latencies: Dict[Tuple, float] = field(default_factory=dict)
    failures: Set[Tuple] = field(default_factory=set)
    triton_impl_name: Optional[str] = None

    @property
    def coverage(self) -> int:
        return len(self.latencies) + len(self.failures)


class AutotuneLatencyCache:
    """Cache lookup for the autotune baseline's per-config latencies."""

    def __init__(self):
        # {(operator, kernel): {shape_id: ShapeCache}}
        self._by_kernel: Dict[Tuple[str, str], Dict[Any, ShapeCache]] = {}

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, autotune_dir: Path) -> "AutotuneLatencyCache":
        cache = cls()
        autotune_dir = Path(autotune_dir).resolve()
        if not autotune_dir.is_dir():
            raise FileNotFoundError(f"Not a directory: {autotune_dir}")

        for path in sorted(autotune_dir.glob("*.json")):
            if path.name in ("index.json", "summary.json"):
                continue
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                log.warning("failed to parse %s: %s", path, exc)
                continue
            op = data.get("operator")
            kern = data.get("kernel")
            if op is None or kern is None:
                continue

            by_shape: Dict[Any, ShapeCache] = {}
            for shape in data.get("shapes", []):
                sid = shape.get("input_id")
                if sid is None:
                    continue
                sc = ShapeCache(
                    input_id=sid,
                    x_val=shape.get("x_val"),
                    original_latency_ms=shape.get("original_config_latency_ms"),
                    reference_config=shape.get("reference_config"),
                    config_source=shape.get("config_source"),
                    n_configs_expected=shape.get("n_configs"),
                    triton_impl_name=shape.get("triton_impl_name"),
                )
                for entry in shape.get("trial_history", []):
                    cfg = entry.get("config")
                    if cfg is None:
                        continue
                    key = _normalize_cfg_key(cfg)
                    lat = entry.get("latency_ms")
                    status = entry.get("status")
                    if (status == "success" and lat is not None
                            and lat == lat and lat > 0
                            and lat != float("inf")):
                        sc.latencies[key] = float(lat)
                    else:
                        sc.failures.add(key)
                by_shape[sid] = sc
            if by_shape:
                cache._by_kernel[(op, kern)] = by_shape

        log.info("Loaded autotune cache: %d kernels, %d shape entries total",
                 len(cache._by_kernel),
                 sum(len(v) for v in cache._by_kernel.values()))
        return cache

    # ------------------------------------------------------------------
    # Lookup API
    # ------------------------------------------------------------------

    def has_kernel(self, op: str, kern: Optional[str]) -> bool:
        return (op, kern) in self._by_kernel

    def shape_ids(self, op: str, kern: Optional[str]) -> List[Any]:
        by_shape = self._by_kernel.get((op, kern), {})
        # Sort by numeric input_id when possible (stable per-shape order)
        def _k(x):
            try:
                return (0, int(x))
            except (TypeError, ValueError):
                return (1, str(x))
        return sorted(by_shape.keys(), key=_k)

    def shape(self, op: str, kern: Optional[str], shape_id: Any) -> Optional[ShapeCache]:
        return self._by_kernel.get((op, kern), {}).get(shape_id)

    def lookup(
        self, op: str, kern: Optional[str], shape_id: Any,
        config: Dict[str, Any],
    ) -> Tuple[str, Optional[float]]:
        """Return ``(status, latency_or_None)``.

        * ``"success"`` - cached successful measurement (``latency`` is a float)
        * ``"failure"`` - cached known failure (``latency`` is ``None``)
        * ``"miss"``    - config not present in the cache (``latency`` is ``None``)
        """
        sc = self.shape(op, kern, shape_id)
        if sc is None:
            return ("miss", None)
        key = _normalize_cfg_key(config)
        if key in sc.latencies:
            return ("success", sc.latencies[key])
        if key in sc.failures:
            return ("failure", None)
        return ("miss", None)

    def coverage(
        self, op: str, kern: Optional[str],
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Diagnostic: how many configs are cached vs the expected space."""
        by_shape = self._by_kernel.get((op, kern), {})
        if not by_shape:
            return {"cached_shapes": 0, "sources": {}, "hit_rate": 0.0}
        sources = {}
        total_hits = 0
        total_tried = 0
        max_expected = 0
        for sc in by_shape.values():
            src = sc.config_source or "?"
            sources[src] = sources.get(src, 0) + 1
            total_hits += len(sc.latencies)
            total_tried += sc.coverage
            max_expected = max(max_expected, sc.n_configs_expected or 0)
        return {
            "cached_shapes": len(by_shape),
            "sources": sources,
            "n_configs_expected": max_expected,
            "hit_rate": total_hits / total_tried if total_tried else 0.0,
            "success_entries": total_hits,
            "total_entries": total_tried,
        }

    def is_common_space_kernel(self, op: str, kern: Optional[str]) -> bool:
        """True iff every cached shape's ``config_source == 'common_search_space'``."""
        by_shape = self._by_kernel.get((op, kern), {})
        if not by_shape:
            return False
        return all(sc.config_source == "common_search_space"
                   for sc in by_shape.values())

    def kernels(self) -> List[Tuple[str, str]]:
        return sorted(self._by_kernel.keys())
