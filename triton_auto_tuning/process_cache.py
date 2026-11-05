#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search inside a Triton cache directory.

Features:
- Scan recursively (default: $TRITON_CACHE_DIR or ./.triton_cache)
- Match by name substring (--name) or regex (--name-regex)
- Filter by extension(s) (--exts .ttir .ttgir .spv .ptx .so .py)
- Optional content search (--contains "pattern") with safe text detection
- Size and mtime filters
- Prints path, size, and mtime for matches

Examples:
  python find_in_triton_cache.py --name softmax
  python find_in_triton_cache.py --exts .ttir .ttgir
  python find_in_triton_cache.py --contains "BLOCK_SIZE"
  TRITON_CACHE_DIR=/path/to/cache python find_in_triton_cache.py --name-regex "triton_.*kernel"

"""
import os
import re
import sys
import argparse
import datetime as dt
from pathlib import Path
import pprint
# plot_triton_configs_nocli.py
import json, statistics
from pathlib import Path
import matplotlib.pyplot as plt  # plain matplotlib

import matplotlib.pyplot as plt  # no seaborn

def human_bytes(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def is_probably_text(p: Path, sample_bytes: int = 4096) -> bool:
    try:
        with p.open("rb") as f:
            chunk = f.read(sample_bytes)
        # Heuristic: if there is a NUL byte, likely binary.
        return b"\x00" not in chunk
    except Exception:
        return False

def file_contains(p: Path, needle: str, ignore_case: bool, max_bytes: int) -> bool:
    # Only search text files by default.
    if not is_probably_text(p):
        return False
    flags = re.IGNORECASE if ignore_case else 0
    pat = re.compile(re.escape(needle), flags)
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            read = 0
            for line in f:
                if pat.search(line):
                    return True
                read += len(line.encode("utf-8", errors="ignore"))
                if max_bytes and read > max_bytes:
                    break
    except Exception:
        return False
    return False

def parse_args():
    default_cache = os.environ.get("TRITON_CACHE_DIR", str(Path("triton_cache").resolve()))
    ap = argparse.ArgumentParser(description="Search Triton cache for files.")
    ap.add_argument("--cache-dir", default=default_cache, help="Root cache dir (default: $TRITON_CACHE_DIR or ./.triton_cache)")
    ap.add_argument("--name", help="Substring to match in filename (case-insensitive by default)")
    ap.add_argument("--name-regex", help="Regex to match against filename")
    ap.add_argument("--exts", nargs="*", default=[], help="Only include files with these extensions, e.g. .ttir .ttgir .spv .ptx .so .py")
    ap.add_argument("--contains", help="Search for this text inside files (safe text-only scan)")
    ap.add_argument("--ignore-case", action="store_true", default=True, help="Case-insensitive name/content matching (default on)")
    ap.add_argument("--case-sensitive", dest="ignore_case", action="store_false", help="Case-sensitive matching")
    ap.add_argument("--size-min", type=int, default=0, help="Minimum file size in bytes")
    ap.add_argument("--size-max", type=int, default=0, help="Maximum file size in bytes (0 = no limit)")
    ap.add_argument("--mtime-since", type=int, default=0, help="Only files modified in the last N days")
    ap.add_argument("--limit", type=int, default=0, help="Stop after printing N matches (0 = no limit)")
    ap.add_argument("--max-bytes", type=int, default=4_000_000, help="Max bytes to scan per file for --contains (default 4MB)")
    return ap.parse_args()

def search_for_file(args):
    root = Path(args.cache_dir).expanduser().resolve()
    if not root.exists():
        print(f"[error] cache dir not found: {root}", file=sys.stderr)
        sys.exit(2)

    name_sub = (args.name or "")
    name_re = re.compile(args.name_regex) if args.name_regex else None
    exts = {e.lower() for e in args.exts}
    now = dt.datetime.now().timestamp()
    min_mtime = 0
    if args.mtime_since > 0:
        min_mtime = now - args.mtime_since * 86400

    ci = args.ignore_case
    if ci:
        name_sub = name_sub.lower()

    scanned, matched = 0, 0
    match_files_name = []
    try:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            scanned += 1

            # extension filter
            if exts and p.suffix.lower() not in exts:
                continue

            # size filters
            try:
                sz = p.stat().st_size
                mt = p.stat().st_mtime
            except FileNotFoundError:
                continue
            if args.size_min and sz < args.size_min:
                continue
            if args.size_max and args.size_max > 0 and sz > args.size_max:
                continue
            if min_mtime and mt < min_mtime:
                continue

            fname = p.name
            # name substring / regex filters
            if args.name:
                check = fname.lower() if ci else fname
                if name_sub not in check:
                    continue
            if name_re and not name_re.search(fname):
                continue

            # content filter
            if args.contains:
                if not file_contains(p, args.contains, args.ignore_case, args.max_bytes):
                    continue

            # print match
            when = dt.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{p}  |  {human_bytes(sz):>8}  |  {when}")
            matched += 1
            match_files_name.append(str(p))
            
            if args.limit and matched >= args.limit:
                break
    except KeyboardInterrupt:
        pass

    print(f"\n[done] scanned {scanned} files under {root}")
    print(f"[done] matched {matched} file(s)")
    return match_files_name

    
    
class config:
    def __init__(self,
                 cache_dir=os.environ.get("TRITON_CACHE_DIR", str(Path(".triton_cache").resolve())),
                 name=None,
                 name_regex=None,
                 exts=None,
                 contains=None,
                 ignore_case=True,
                 size_min=0,
                 size_max=0,
                 mtime_since=0,
                 limit=0,
                 max_bytes=4_000_000):
        self.cache_dir = cache_dir
        self.name = name
        self.name_regex = name_regex
        self.exts = [] if exts is None else exts
        self.contains = contains
        self.ignore_case = ignore_case
        self.size_min = size_min
        self.size_max = size_max
        self.mtime_since = mtime_since
        self.limit = limit
        self.max_bytes = max_bytes

        
def create_config(args):
    return config(args.cache_dir, args.name, args.name_regex, args.exts, args.contains, args.ignore_case, args.size_min, args.size_max, args.mtime_since, args.limit, args.max_bytes)


# ---- data loading ----------------------------------------------------------

def load_records(json_paths):
    """Read one or more JSON files and return flat records."""
    if isinstance(json_paths, (str, Path)):
        json_paths = [json_paths]
    records = []
    for p in map(Path, json_paths):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("configs_timings") or data.get("config_timings") or []
        for pair in items:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            cfg, t = pair
            kwargs = (cfg or {}).get("kwargs") or {}
            records.append({
                "file": str(p),
                "block_size": kwargs.get("BLOCK_SIZE"),
                "num_warps": (cfg or {}).get("num_warps"),
                "num_stages": (cfg or {}).get("num_stages"),
                "time_s": float(t),  # assumes seconds
            })
    # normalize ints
    for r in records:
        for k in ("block_size", "num_warps", "num_stages"):
            if r[k] is not None:
                r[k] = int(r[k])
    return records

def filter_fixed(records, fixed=None):
    """Keep only records matching fixed {dim: value} (e.g., {'num_warps':8})."""
    if not fixed:
        return records
    out = []
    for r in records:
        ok = True
        for k, v in fixed.items():
            if r.get(k) != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out

def aggregate_by_x(records, x_key, agg="min"):
    """Group by x_key and aggregate time (min/mean/median)."""
    groups = {}
    for r in records:
        x = r.get(x_key)
        if x is None:
            continue
        groups.setdefault(x, []).append(r["time_s"])
    xs, ys = [], []
    for x, ts in groups.items():
        if   agg == "min":    val = min(ts)
        elif agg == "mean":   val = statistics.fmean(ts)
        elif agg == "median": val = statistics.median(ts)
        else: raise ValueError("agg must be min|mean|median")
        xs.append(x); ys.append(val)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    return xs, ys

def scale_times(times_s, unit="ms"):
    if unit == "s":   return times_s, "Seconds"
    if unit == "ms":  return [t*1e3 for t in times_s], "Milliseconds"
    if unit == "us":  return [t*1e6 for t in times_s], "Microseconds"
    raise ValueError("unit must be s|ms|us")

# ---- plotting --------------------------------------------------------------

def save_plot(xs, ys, x_label, y_unit, outpath, title=None):
    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys, marker="o")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.xlabel(x_label)
    plt.ylabel(f"Time ({y_unit})")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[saved] {out}")


def plot_one_axis(json_paths, x_axis, fixed=None, agg="min", unit="ms", outfile=None, title=None):
    """
    x_axis: 'block_size' | 'num_warps' | 'num_stages'
    fixed : dict for the other dims, e.g. {'num_warps':8,'num_stages':2}
    agg   : 'min'|'mean'|'median' (collapse duplicates / multiple files)
    unit  : 's'|'ms'|'us'  (input assumed seconds)
    title : optional plot title (string). If None, a sensible default is used.
    """
    records = load_records(json_paths)
    data = filter_fixed(records, fixed)
    if not data:
        raise ValueError("No records after applying 'fixed'. Check your constraints.")
    xs, ys_s = aggregate_by_x(data, x_axis, agg=agg)
    if not xs:
        raise ValueError("Nothing to plot—did you pick the right x_axis and fixed?")
    ys, y_unit = scale_times(ys_s, unit)

    if outfile is None:
        outfile = f"plots/by_{x_axis}.png"

    # Default title if none provided
    if title is None:
        fixed_str = ", ".join(f"{k}={v}" for k, v in (fixed or {}).items()) or "aggregated over others"
        title = f"{x_axis.replace('_',' ').title()} vs Time ({agg}, {unit}); {fixed_str}"

    save_plot(xs, ys,
              x_label=x_axis.replace("_", " ").title(),
              y_unit=y_unit,
              outpath=outfile,
              title=title)


def main():
    args = parse_args()
    config = create_config(args)
    match_files_name = search_for_file(config)
    
    # Replace 'results.json' with your file path (or a list of files)
    jf = match_files_name

    # 1) Block size vs time (fastest across warps/stages)
    plot_one_axis(jf, x_axis="block_size", fixed={"num_warps": 2, "num_stages": 2}, agg="min", unit="ms",
                  outfile="plots/by_block_size.png",
                  title="Softmax (XPU): Block Size @ warps=2, stages=2")

    # 2) Num warps vs time at fixed block_size & num_stages
    plot_one_axis(jf, x_axis="num_warps",
                  fixed={"block_size": 256, "num_stages": 1},
                  agg="min", unit="ms",
                  outfile="plots/by_num_warps.png",
                  title="Softmax (XPU): Num Warps @ BLOCK=256, stages=1")

    # 3) Num stages vs time at fixed block_size & num_warps
    plot_one_axis(jf, x_axis="num_stages",
                  fixed={"block_size": 64, "num_warps": 8},
                  agg="min", unit="ms",
                  outfile="plots/by_num_stages.png",
                  title="Softmax (XPU): Num Stages @ BLOCK=64, warps=8")

if __name__ == "__main__":
    main()
