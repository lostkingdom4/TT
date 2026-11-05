"""Generic results store shared by every baseline.

Directory layout produced by a baseline named ``<baseline>``::

    <results_root>/
        <device>/
            <baseline>/
                index.json                   # progress tracker
                <operator>__<kernel>.json    # per-kernel envelope
                <operator>.json              # legacy per-operator file
                summary.json                 # aggregate statistics
                summary.csv                  # flat per-shape export

Each subclass defines a ``FIELD_SCHEMA`` class attribute (list of
``(json_key, csv_column)`` tuples) that determines which per-shape
fields land in ``summary.csv``.  Subclasses may also override
``_operator_summary_extras`` to add per-baseline summary columns.
"""

import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class BaseResultsStore:
    """On-disk results store with resume support."""

    #: subclasses override: list of (shape-dict key, CSV column name)
    FIELD_SCHEMA: List[tuple] = [
        ("operator", "operator"),
        ("kernel", "kernel"),
        ("input_id", "input_id"),
        ("x_val", "x_val"),
        ("triton_impl_name", "impl_name"),
        ("latency_ms", "latency_ms"),
        ("verified", "verified"),
        ("verification_error", "verification_error"),
        ("status", "status"),
        ("error", "error"),
    ]

    #: baseline identifier used as the leaf directory under <device>/
    BASELINE_NAME: str = "base"

    def __init__(
        self,
        results_root: Path,
        device: str,
        baseline_name: Optional[str] = None,
        extra_index: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.baseline = baseline_name or self.BASELINE_NAME
        self.device_dir = Path(results_root) / device / self.baseline
        self.device_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.device_dir / "index.json"
        self._ensure_index(extra_index or {})

    # ------------------------------------------------------------------
    # per-kernel file naming
    # ------------------------------------------------------------------

    def _file_name(self, operator: str, kernel: Optional[str]) -> str:
        if kernel:
            # Disambiguate operators with multiple kernels (e.g. addmm).
            return f"{operator}__{kernel}.json"
        return f"{operator}.json"

    def _result_key(self, operator: str, kernel: Optional[str]) -> str:
        return f"{operator}__{kernel}" if kernel else operator

    # ------------------------------------------------------------------
    # index management
    # ------------------------------------------------------------------

    def _ensure_index(self, extra: Dict[str, Any]):
        if not self.index_path.exists():
            payload = {
                "device": self.device,
                "baseline": self.baseline,
                "started_at": _now_iso(),
                "last_updated": _now_iso(),
                "operators_completed": [],
                "operators_failed": [],
                **extra,
            }
            self._write_index(payload)

    def _read_index(self) -> Dict[str, Any]:
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, data: Dict[str, Any]):
        data["last_updated"] = _now_iso()
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def is_completed(self, operator: str, kernel: Optional[str] = None) -> bool:
        idx = self._read_index()
        key = self._result_key(operator, kernel)
        return key in idx.get("operators_completed", [])

    # Back-compat alias used by the existing random runner
    is_operator_completed = is_completed

    # ------------------------------------------------------------------
    # per-operator / per-kernel results
    # ------------------------------------------------------------------

    def save_operator_results(
        self,
        operator: str,
        shape_results: List[Dict[str, Any]],
        status: str = "completed",
        kernel: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        payload = {
            "operator": operator,
            "kernel": kernel,
            "device": self.device,
            "baseline": self.baseline,
            "num_shapes": len(shape_results),
            "timestamp": _now_iso(),
            "status": status,
            "shapes": shape_results,
        }
        if extra:
            payload.update(extra)

        out_path = self.device_dir / self._file_name(operator, kernel)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        log.info("Saved %d shape results to %s", len(shape_results), out_path)

        idx = self._read_index()
        completed = set(idx.get("operators_completed", []))
        failed = set(idx.get("operators_failed", []))
        key = self._result_key(operator, kernel)
        if status == "completed":
            completed.add(key); failed.discard(key)
        else:
            failed.add(key); completed.discard(key)
        idx["operators_completed"] = sorted(completed)
        idx["operators_failed"] = sorted(failed)
        self._write_index(idx)

    def load_operator_results(
        self, operator: str, kernel: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        path = self.device_dir / self._file_name(operator, kernel)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def operator_file_path(
        self, operator: str, kernel: Optional[str] = None,
    ) -> Path:
        return self.device_dir / self._file_name(operator, kernel)

    def write_partial_envelope(
        self,
        operator: str,
        shape_results: List[Dict[str, Any]],
        kernel: Optional[str] = None,
        status: str = "in_progress",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Atomically write the per-kernel envelope without touching index.json.

        Used by runners that flush partial progress after every trial so a
        ``Ctrl-C`` never loses more than one trial's work.  The index is
        still the coarse per-kernel tracker updated by
        :meth:`save_operator_results` when the kernel completes.
        """
        payload = {
            "operator": operator,
            "kernel": kernel,
            "device": self.device,
            "baseline": self.baseline,
            "num_shapes": len(shape_results),
            "timestamp": _now_iso(),
            "status": status,
            "shapes": shape_results,
        }
        if extra:
            payload.update(extra)

        out_path = self.operator_file_path(operator, kernel)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, out_path)
        return out_path

    def all_operator_files(self) -> List[Path]:
        return sorted(
            p for p in self.device_dir.glob("*.json")
            if p.name not in ("index.json", "summary.json")
        )

    # ------------------------------------------------------------------
    # aggregation & export
    # ------------------------------------------------------------------

    def export_csv(self, csv_path: Optional[Path] = None) -> Path:
        if csv_path is None:
            csv_path = self.device_dir / "summary.csv"

        rows: List[Dict[str, Any]] = []
        extra_keys = set()
        for path in self.all_operator_files():
            data = json.loads(path.read_text())
            for shape in data.get("shapes", []):
                row = {"device": self.device}
                for json_key, csv_col in self.FIELD_SCHEMA:
                    v = shape.get(json_key, data.get(json_key, ""))
                    if isinstance(v, (dict, list)):
                        v = json.dumps(v, default=str)
                    row[csv_col] = v
                # Capture any extra keys the subclass hasn't declared.
                for k in shape:
                    if k not in {jk for jk, _ in self.FIELD_SCHEMA}:
                        extra_keys.add(k)
                rows.append(row)

        fieldnames = ["device"] + [csv_col for _, csv_col in self.FIELD_SCHEMA]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        log.info("Exported %d rows to %s", len(rows), csv_path)
        return csv_path

    def export_summary(self) -> Dict[str, Any]:
        idx = self._read_index()
        total_shapes = 0
        total_ok = 0
        total_err = 0
        total_corr_err = 0
        total_verified = 0
        operator_summaries = []

        for path in self.all_operator_files():
            data = json.loads(path.read_text())
            shapes = data.get("shapes", [])
            ok = sum(1 for s in shapes if s.get("status") == "success")
            err = sum(1 for s in shapes if s.get("status") == "error")
            corr_err = sum(
                1 for s in shapes if s.get("status") == "correctness_error"
            )
            verified = sum(1 for s in shapes if s.get("verified"))

            total_shapes += len(shapes)
            total_ok += ok
            total_err += err
            total_corr_err += corr_err
            total_verified += verified

            latencies = [
                s["latency_ms"] for s in shapes
                if s.get("latency_ms") is not None
            ]

            summary = {
                "operator": data.get("operator", path.stem),
                "kernel": data.get("kernel"),
                "has_autotune": data.get("has_autotune", False),
                "num_shapes": len(shapes),
                "successful": ok,
                "errors": err,
                "correctness_errors": corr_err,
                "verified": verified,
                "min_latency_ms": min(latencies) if latencies else None,
                "max_latency_ms": max(latencies) if latencies else None,
                "median_latency_ms": _median(latencies) if latencies else None,
            }
            summary.update(self._operator_summary_extras(data, shapes))
            operator_summaries.append(summary)

        return {
            "device": self.device,
            "baseline": self.baseline,
            "operators_completed": len(idx.get("operators_completed", [])),
            "operators_failed": len(idx.get("operators_failed", [])),
            "total_shapes": total_shapes,
            "total_ok": total_ok,
            "total_errors": total_err,
            "total_correctness_errors": total_corr_err,
            "total_verified": total_verified,
            "operators": operator_summaries,
        }

    def _operator_summary_extras(
        self, data: Dict[str, Any], shapes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Hook for baselines to add per-operator summary columns."""
        return {}

    def print_summary(self):
        summary = self.export_summary()
        print(f"\n{'=' * 70}")
        print(
            f"  {self.baseline.capitalize()} Baseline Summary — "
            f"device: {summary['device']}"
        )
        print(f"{'=' * 70}")
        print(f"  Operators completed:      {summary['operators_completed']}")
        print(f"  Operators failed:         {summary['operators_failed']}")
        print(f"  Total shapes benchmarked: {summary['total_shapes']}")
        print(f"  Successful:               {summary['total_ok']}")
        print(f"  Errors:                   {summary['total_errors']}")
        print(f"  Correctness errors:       {summary['total_correctness_errors']}")
        print(f"  Verified:                 {summary['total_verified']}")
        print(f"{'=' * 70}")
        print(
            f"  {'Operator/Kernel':<45} {'AT':>3} {'Shp':>5} "
            f"{'OK':>4} {'Err':>4} {'Vfy':>4} {'Med ms':>10}"
        )
        print(f"  {'-' * 45} {'-' * 3} {'-' * 5} "
              f"{'-' * 4} {'-' * 4} {'-' * 4} {'-' * 10}")
        for op in summary["operators"]:
            at = "Y" if op["has_autotune"] else "N"
            label = (f"{op['operator']}/{op['kernel']}"
                     if op.get("kernel") else op["operator"])[:45]
            med = (
                f"{op['median_latency_ms']:.4f}"
                if op["median_latency_ms"] else "N/A"
            )
            print(
                f"  {label:<45} {at:>3} {op['num_shapes']:>5} "
                f"{op['successful']:>4} {op['errors']:>4} "
                f"{op['verified']:>4} {med:>10}"
            )
        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0
