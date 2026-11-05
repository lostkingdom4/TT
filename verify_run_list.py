#!/usr/bin/env python3
"""Verify IR + metadata availability for every entry in a batch_runner run-list.

Mirrors the checks batch_runner.discover_kernels() performs, so the output
answers: "if I point batch_runner at this run-list and IR dir, which
(operator, kernel) entries will it actually be able to run?"

For each run-list entry we verify:

  1. The operator directory exists under ``<ir-dir>/<gpu>/<operator>/``.
  2. At least one ``*_input*_shapes.json`` file in that directory parses as
     JSON and has a ``kernel`` field that matches the run-list kernel name
     (exact match, or prefix/suffix match for ``...`` abbreviations).
  3. A matching TTGIR file is reachable via ``rglob("*<kernel>*.ttgir")``
     from the operator directory.

Exit code is 0 only when every entry passes all three checks.

Usage::

    python verify_run_list.py run_lists/v100_threshold_include_no_jagged_ln.json
    python verify_run_list.py run_lists/v100_threshold_include.json --gpu v100
    python verify_run_list.py <run-list> --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils import parse_shapes_file


@dataclass
class EntryReport:
    operator: str
    kernel: str
    source: Optional[str]
    operator_dir_exists: bool = False
    shapes_matches: List[Path] = field(default_factory=list)
    ttgir_matches: List[Path] = field(default_factory=list)
    kernel_source_paths: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.operator_dir_exists
            and bool(self.shapes_matches)
            and bool(self.ttgir_matches)
            and bool(self.kernel_source_paths)
            and not self.errors
        )


def _kernel_name_matches(shapes_kernel: str, pattern: str) -> bool:
    """Replicate run_list_matches() semantics for a single kernel field.

    ``...`` inside the run-list kernel name is a prefix/suffix wildcard.
    """
    if pattern == shapes_kernel:
        return True
    if "..." in pattern:
        parts = pattern.split("...")
        if len(parts) == 2:
            prefix, suffix = parts
            return shapes_kernel.startswith(prefix) and shapes_kernel.endswith(suffix)
    return False


# Mirrors ConfigurationOptimizer._extract_source_location: TTGIR loc() directives
# look like `loc("path/to/file.py":line:col)`. The benchmark step in
# compare_configs._load_kernel imports whichever .py the most-common loc points
# to, so a stale or doubled path baked into the TTGIR makes every alternate-config
# benchmark fail with FileNotFoundError.
_LOC_RE = re.compile(r'loc\("([^"]+\.py)":')
_LIBRARY_MARKERS = ("site-packages", "dist-packages", "/lib/python")


def _extract_loc_source(ttgir_path: Path) -> Optional[str]:
    """Return the most-referenced user .py path embedded in TTGIR loc() directives.

    Library/framework files (site-packages, dist-packages, stdlib) are filtered
    out — they're inlined dependencies, not the user kernel.
    """
    try:
        content = ttgir_path.read_text(errors="replace")
    except OSError:
        return None
    counts: dict[str, int] = {}
    for path in _LOC_RE.findall(content):
        counts[path] = counts.get(path, 0) + 1
    if not counts:
        return None
    user = {p: n for p, n in counts.items() if not any(m in p for m in _LIBRARY_MARKERS)}
    pool = user or counts
    return max(pool.items(), key=lambda kv: kv[1])[0]


def _resolve_kernel_module(
    loc_source: str, shapes_file: Path, kernel_name: str
) -> Tuple[Optional[Path], str]:
    """Mirror optimizer_functions.py's bench_source_file resolution.

    Returns ``(resolved_path, label)`` where ``label`` is "per-input" if a
    matching ``<kernel>_input<id>.py`` exists next to the base loc source, else
    "base" if the loc source itself exists, else ``(None, "missing")``.
    """
    base = Path(loc_source)
    # Parse input_id from shapes filename: ..._input<id>_shapes.json
    m = re.search(r"_input(\d+)_shapes\.json$", shapes_file.name)
    if m:
        per_input = base.parent / f"{kernel_name}_input{m.group(1)}.py"
        if per_input.is_file():
            return per_input, "per-input"
    if base.is_file():
        return base, "base"
    return None, "missing"


def check_entry(entry: dict, gpu_dir: Path) -> EntryReport:
    operator = entry["operator"]
    kernel = entry["kernel"]
    source = entry.get("source")
    report = EntryReport(operator=operator, kernel=kernel, source=source)

    operator_dir = gpu_dir / operator
    if not operator_dir.is_dir():
        report.errors.append(f"operator dir missing: {operator_dir}")
        return report
    report.operator_dir_exists = True

    # --- Shapes files ---
    shapes_files = sorted(operator_dir.glob("*_input*_shapes.json"))
    if not shapes_files:
        report.errors.append("no *_input*_shapes.json in operator dir")
    else:
        matched: List[Path] = []
        for sf in shapes_files:
            meta = parse_shapes_file(sf)
            if not meta:
                continue
            name = meta.get("kernel_name", "unknown")
            if name == "unknown":
                continue
            if _kernel_name_matches(name, kernel):
                matched.append(sf)
        if not matched:
            report.errors.append(
                f"no shapes file with kernel == '{kernel}' "
                f"(scanned {len(shapes_files)} files)"
            )
        else:
            report.shapes_matches = matched

    # --- TTGIR ---
    ttgir_matches = sorted(operator_dir.rglob(f"*{kernel}*.ttgir"))
    if not ttgir_matches:
        report.errors.append(f"no TTGIR matching *{kernel}*.ttgir under operator dir")
    else:
        report.ttgir_matches = ttgir_matches

    # --- Kernel source (.py) reachable from TTGIR loc() ---
    # Replicates compare_configs._load_kernel(module_path=...) input: the path
    # is built from the most-common loc("...py":n:m) directive in the TTGIR,
    # then optionally remapped to a per-input variant by optimizer_functions.py.
    if report.ttgir_matches and report.shapes_matches:
        # Use the same resolved kernel name the runner would (split off class prefix).
        bench_kernel = kernel.rsplit(".", 1)[-1] if "." in kernel else kernel
        # Pick the TTGIR most likely to be loaded: the one whose stem most closely
        # matches the kernel name. Falls back to the first match.
        ttgir_path = next(
            (t for t in report.ttgir_matches if bench_kernel in t.stem),
            report.ttgir_matches[0],
        )
        loc_source = _extract_loc_source(ttgir_path)
        if loc_source is None:
            report.errors.append(
                f"no loc(\"...py\") directive found in {ttgir_path.name}"
            )
        else:
            resolved: List[Path] = []
            missing_for: List[str] = []
            for sf in report.shapes_matches:
                path, label = _resolve_kernel_module(loc_source, sf, bench_kernel)
                if path is None:
                    missing_for.append(sf.stem)
                else:
                    resolved.append(path)
            report.kernel_source_paths = resolved
            if missing_for:
                report.errors.append(
                    f"kernel source unreachable for {len(missing_for)}/"
                    f"{len(report.shapes_matches)} shape(s); loc() points to "
                    f"{loc_source} (file does not exist, and no "
                    f"{bench_kernel}_input<id>.py beside it)"
                )

    return report


def _format_count(n: int, label: str) -> str:
    return f"{n} {label}" if n != 1 else f"{n} {label[:-1] if label.endswith('s') else label}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify IR + shapes metadata for every kernel in a batch_runner run-list",
    )
    parser.add_argument(
        "run_list", type=Path,
        help="Path to run-list JSON (same format as batch_runner --run-list)",
    )
    parser.add_argument(
        "--ir-dir", type=Path,
        default=_PROJECT_ROOT / "extract" / "ir_output",
        help="IR output root (default: extract/ir_output)",
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="GPU folder name (e.g. v100, A40). Inferred from run-list filename if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print the matched shapes/TTGIR files for OK entries too",
    )
    args = parser.parse_args()

    if not args.run_list.is_file():
        print(f"error: run-list not found: {args.run_list}", file=sys.stderr)
        return 2

    with open(args.run_list) as f:
        run_list = json.load(f)

    if run_list.get("mode", "include") != "include":
        print(
            f"warning: run-list mode is '{run_list.get('mode')}' — "
            f"this tool only makes sense for 'include' lists",
            file=sys.stderr,
        )

    entries = run_list.get("kernels", [])
    if not entries:
        print("error: run-list has no 'kernels' entries", file=sys.stderr)
        return 2

    # --- Resolve GPU directory ---
    gpu_name = args.gpu
    if gpu_name is None:
        stem = args.run_list.stem.lower()
        for candidate in ("v100", "a40", "a100", "l40s", "h100"):
            if stem.startswith(candidate) or f"_{candidate}_" in stem:
                gpu_name = candidate
                break
    if gpu_name is None:
        sub = [d.name for d in args.ir_dir.iterdir() if d.is_dir()]
        if len(sub) == 1:
            gpu_name = sub[0]
        else:
            print(
                f"error: could not infer GPU; pass --gpu. Candidates under "
                f"{args.ir_dir}: {sub}",
                file=sys.stderr,
            )
            return 2

    # Preserve folder-name casing (e.g. "A40")
    gpu_dir = None
    for d in args.ir_dir.iterdir():
        if d.is_dir() and d.name.lower() == gpu_name.lower():
            gpu_dir = d
            break
    if gpu_dir is None or not gpu_dir.is_dir():
        print(f"error: GPU dir not found: {args.ir_dir}/{gpu_name}", file=sys.stderr)
        return 2

    print(f"Verifying {args.run_list.name}")
    print(f"  IR dir:  {gpu_dir}")
    print(f"  entries: {len(entries)}")
    print()

    reports: List[EntryReport] = [check_entry(e, gpu_dir) for e in entries]

    ok_entries = [r for r in reports if r.ok]
    bad_entries = [r for r in reports if not r.ok]

    col_w = max(len(f"{r.operator}/{r.kernel}") for r in reports) + 2

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(_PROJECT_ROOT))
        except ValueError:
            return str(p)

    for r in reports:
        tag = "OK  " if r.ok else "FAIL"
        label = f"{r.operator}/{r.kernel}".ljust(col_w)
        summary = (
            f"{_format_count(len(r.shapes_matches), 'shapes')}, "
            f"{_format_count(len(r.ttgir_matches), 'ttgirs')}, "
            f"{_format_count(len(r.kernel_source_paths), 'sources')}"
        )
        print(f"  [{tag}] {label} {summary}")
        if args.verbose and r.ok:
            for sf in r.shapes_matches:
                print(f"          shapes: {_rel(sf)}")
            for tf in r.ttgir_matches:
                print(f"          ttgir : {_rel(tf)}")
            for kf in r.kernel_source_paths:
                print(f"          source: {_rel(kf)}")
        for err in r.errors:
            print(f"          -> {err}")

    print()
    print(f"Result: {len(ok_entries)} OK, {len(bad_entries)} FAIL "
          f"(of {len(reports)} total)")

    return 0 if not bad_entries else 1


if __name__ == "__main__":
    sys.exit(main())
