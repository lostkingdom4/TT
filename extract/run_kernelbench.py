"""Capture Triton kernels emitted by PyTorch Inductor when compiling
KernelBench (https://github.com/ScalingIntelligence/KernelBench) problems,
filter out kernels below a latency threshold, then dump TTGIR + shapes in the
same on-disk format used for TritonBench operators.

Pipeline per problem (see plan for details):
    1. Import the Model / get_inputs / get_init_inputs.
    2. torch.compile(Model(), mode="max-autotune") and run forward.
    3. Hook CachingAutotuner.run to record (a) per-kernel median latency and
       (b) tensor/scalar args + grid + config.
    4. Drop kernels whose median latency < --threshold-ms (default 0.01 ms).
    5. For each surviving kernel, extract TTGIR from the Triton cache (reusing
       extract/run_inductor.py helpers) and save per-input shape JSON.
    6. Correctness check: torch.allclose against eager output.
    7. Emit kernelbench_kernels.yaml and run_lists/<gpu>_kernelbench_include.json.

Usage:
    python extract/run_kernelbench.py --gpu v100 --level 2 \\
        --problem 14_Gemm_Divide_Sum_Scaling
    python extract/run_kernelbench.py --gpu v100 --levels 1,2,3 \\
        --output-dir ir_output --threshold-ms 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

import yaml

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from paths import PROJECT_ROOT, METADATA_DIR  # noqa: E402
from _autotuner_hooks import combined_autotuner_hook  # noqa: E402
from kernelbench_utils import (  # noqa: E402
    KernelBenchProblem,
    SourceHashTable,
    discover_problems,
    load_problem_module,
)
from run_inductor import (  # noqa: E402 — reuse existing IR writer
    _extract_and_save_ir,
    _save_kernel_shapes,
)

logger = logging.getLogger("extract_kernelbench")

DEFAULT_KERNELBENCH_ROOT = (
    SCRIPT_DIR / "kernelbench_src" / "KernelBench"
)
DEFAULT_METADATA_OUT = METADATA_DIR / "kernelbench_kernels.yaml"
DEFAULT_THRESHOLD_MS = 0.01
SCREENING_ITERS = 6  # forward passes for latency median


# ---------------------------------------------------------------------------
# Per-problem capture (one process per problem for state isolation)
# ---------------------------------------------------------------------------

def _run_eager(model, inputs) -> "torch.Tensor | tuple":
    import torch
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, torch.Tensor):
        return out.detach().clone()
    if isinstance(out, (tuple, list)):
        return tuple(o.detach().clone() if hasattr(o, "detach") else o
                     for o in out)
    return out


def _all_close(a, b, atol=1e-2, rtol=1e-2) -> bool:
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        # Compare on CPU: autotune leaves the GPU close to full on some
        # L1 problems (e.g. 19_ReLU at 4096x393216 fp32), and allclose's
        # scratch can OOM. CPU RAM has no such constraint here.
        return bool(torch.allclose(
            a.detach().to("cpu", torch.float32),
            b.detach().to("cpu", torch.float32),
            atol=atol, rtol=rtol,
        ))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return all(_all_close(x, y, atol, rtol) for x, y in zip(a, b))
    return a == b


def _capture_problem(
    problem: KernelBenchProblem,
    output_dir: Path,
    threshold_ms: float,
    device: str,
    dtype: str,
) -> Dict[str, Any]:
    """Screen + capture one KernelBench problem. Returns a summary dict."""
    import torch
    import torch._inductor.config as inductor_config

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    module = load_problem_module(problem)
    op_name = problem.operator_name
    op_dir = output_dir / op_name
    op_dir.mkdir(parents=True, exist_ok=True)

    init_args = module.get_init_inputs()
    raw_inputs = module.get_inputs()

    eager_model = module.Model(*init_args).to(device=device, dtype=torch_dtype)
    eager_model.eval()
    inputs = [
        x.to(device=device, dtype=torch_dtype) if hasattr(x, "to") else x
        for x in raw_inputs
    ]

    try:
        eager_out = _run_eager(eager_model, inputs)
    except Exception as exc:
        logger.warning("[kb] eager forward failed for %s: %s", op_name, exc)
        return {
            "operator": op_name, "status": "eager_failed",
            "error": str(exc), "kernels": [],
        }

    triton_cache_dir = tempfile.mkdtemp(prefix="triton_cache_kb_")
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    torch._dynamo.reset()

    compiled_out = None
    try:
        with inductor_config.patch(
                max_autotune=True,
                max_autotune_gemm_backends="TRITON",
        ):
            compiled_model = torch.compile(eager_model, mode="max-autotune-no-cudagraphs")
            # Warmup forward: triggers autotuning, which benchmarks throwaway
            # candidate templates (triton_mm_0..N). We do NOT want those in the
            # capture — only the kernels Inductor actually dispatches to in
            # steady state. So warm up WITHOUT the hook, then install it.
            _ = compiled_model(*inputs)
            torch.cuda.synchronize()

            with combined_autotuner_hook(op_dir) as state:
                for i in range(SCREENING_ITERS):
                    compiled_out = compiled_model(*inputs)
                    torch.cuda.synchronize()
    except Exception as exc:
        logger.warning("[kb] compile/capture failed for %s: %s", op_name, exc)
        traceback.print_exc()
        _cleanup_cache(triton_cache_dir)
        return {
            "operator": op_name, "status": "compile_failed",
            "error": str(exc), "kernels": [],
        }

    latencies: Dict[str, List[float]] = state["latencies"]
    captured: Dict[str, List[dict]] = state["captured"]

    if not captured:
        logger.warning("[kb] no Triton kernels emitted for %s (likely cuBLAS/cuDNN)",
                       op_name)
        _cleanup_cache(triton_cache_dir)
        return {"operator": op_name, "status": "no_triton_kernels", "kernels": []}

    # All timings are post-warmup/post-autotune so they're usable directly.
    screened: Dict[str, float] = {
        kname: float(median(series)) for kname, series in latencies.items()
    }

    kept: List[str] = []
    filtered: List[Dict[str, Any]] = []
    for kname, med in screened.items():
        if med >= threshold_ms:
            kept.append(kname)
        else:
            filtered.append({"kernel": kname, "median_ms": round(med, 6)})

    kept_captured = {k: captured[k] for k in kept if k in captured}

    # Persist screening log regardless of whether anything was kept.
    latency_log = {
        "operator": op_name,
        "threshold_ms": threshold_ms,
        "kernels": [
            {"kernel": k, "median_ms": round(screened[k], 6),
             "n_calls": len(latencies[k]),
             "min_ms": round(min(latencies[k]), 6),
             "max_ms": round(max(latencies[k]), 6),
             "kept": k in kept}
            for k in screened
        ],
    }
    (op_dir / "latencies.json").write_text(
        json.dumps(latency_log, indent=2, sort_keys=True))

    # Correctness — free GPU cache first so CPU transfer isn't fighting
    # for pinned staging buffers on near-full GPUs.
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    correctness_ok = _all_close(compiled_out, eager_out)
    if not correctness_ok:
        logger.warning("[kb] allclose failed for %s — kernels still saved", op_name)

    # Extract TTGIR + shapes for kept kernels
    saved_kernels: List[str] = []
    src_hash_table = SourceHashTable(output_dir.parent / "kb_source_hashes.json")
    dedup_hits: List[Dict[str, str]] = []

    if kept_captured:
        _save_kernel_shapes(kept_captured, op_name, op_dir)
        for kname in kept_captured:
            try:
                _extract_and_save_ir(kname, triton_cache_dir, op_dir, op_name)
            except Exception as exc:
                logger.warning("[kb] TTGIR extract failed for %s.%s: %s",
                               op_name, kname, exc)
                continue

            ttgir_path = op_dir / f"{op_name}_{kname}.ttgir"
            src_path = op_dir / f"{kname}.py"
            if src_path.exists():
                src_hash = SourceHashTable.hash_source(src_path.read_text())
                prior = src_hash_table.record(
                    src_hash, op_name, kname, str(ttgir_path))
                if prior is not None:
                    dedup_hits.append({
                        "kernel": kname, "src_hash": src_hash,
                        "first_seen_operator": prior["operator"],
                        "first_seen_kernel": prior["kernel"],
                    })
            saved_kernels.append(kname)
        src_hash_table.save()

    _cleanup_cache(triton_cache_dir)

    summary = {
        "operator": op_name, "status": "ok",
        "kernels": saved_kernels,
        "filtered": filtered,
        "correctness": correctness_ok,
        "dedup_hits": dedup_hits,
        "level": problem.level, "pid": problem.pid, "slug": problem.slug,
    }
    (op_dir / "problem_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _cleanup_cache(cache_dir: str) -> None:
    try:
        shutil.rmtree(cache_dir)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Output aggregation: metadata YAML + run-list JSON
# ---------------------------------------------------------------------------

def _write_metadata_yaml(
    metadata_path: Path,
    per_problem: List[Dict[str, Any]],
) -> None:
    existing: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            existing = yaml.safe_load(metadata_path.read_text()) or {}
        except Exception:
            existing = {}

    for entry in per_problem:
        if entry.get("status") != "ok" or not entry.get("kernels"):
            continue
        op = entry["operator"]
        existing[op] = {
            "pt2_kernelbench": {
                "kernels": sorted(entry["kernels"]),
                "tags": ["pt2", "kernelbench"],
                "level": entry["level"],
            }
        }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        yaml.dump(existing, default_flow_style=False, sort_keys=True))
    logger.info("[kb] Wrote metadata YAML: %s", metadata_path)


def _write_run_list(
    run_list_path: Path,
    per_problem: List[Dict[str, Any]],
    threshold_ms: float,
    gpu: str,
) -> None:
    kernels = []
    for entry in per_problem:
        if entry.get("status") != "ok":
            continue
        op = entry["operator"]
        for kname in entry["kernels"]:
            kernels.append({"operator": op, "kernel": kname, "source": "inductor"})
    payload = {
        "description": (
            f"KernelBench kernels on {gpu} with median latency >= {threshold_ms} ms"
        ),
        "mode": "include",
        "threshold_ms": threshold_ms,
        "kernels": sorted(kernels, key=lambda e: (e["operator"], e["kernel"])),
    }
    run_list_path.parent.mkdir(parents=True, exist_ok=True)
    run_list_path.write_text(json.dumps(payload, indent=2))
    logger.info("[kb] Wrote run-list JSON: %s (%d kernels)",
                run_list_path, len(kernels))


# ---------------------------------------------------------------------------
# Child-process entry: handle one problem
# ---------------------------------------------------------------------------

def run_child(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve() / args.gpu
    kb_root = Path(args.kernelbench_root).resolve()

    problems = discover_problems(
        kb_root,
        levels=[args.level],
        problem_filter=[args.problem_stem],
    )
    if not problems:
        logger.error("[kb-child] No problem matching level=%d stem=%s",
                     args.level, args.problem_stem)
        sys.exit(2)

    summary = _capture_problem(
        problems[0], output_dir,
        threshold_ms=args.threshold_ms,
        device=args.device,
        dtype=args.dtype,
    )
    # Child writes its own summary; parent will pick it up via per-problem file.
    print(json.dumps({"child_summary": summary}), flush=True)


# ---------------------------------------------------------------------------
# Parent-process: orchestrate children
# ---------------------------------------------------------------------------

def _spawn_child(problem: KernelBenchProblem,
                 output_dir: Path,
                 args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--run-child",
        "--gpu", args.gpu,
        "--output-dir", str(output_dir),
        "--kernelbench-root", str(args.kernelbench_root),
        "--level", str(problem.level),
        "--problem-stem", problem.module_path.stem,
        "--threshold-ms", str(args.threshold_ms),
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    logger.info("[kb] Spawning: %s", " ".join(cmd))
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError as exc:
        logger.warning("[kb] Child failed for %s (rc=%d), continuing ...",
                       problem.operator_name, exc.returncode)

    logger.info("[kb] Finished %s in %.1fs",
                problem.operator_name, time.perf_counter() - start)
    summary_path = output_dir / args.gpu / problem.operator_name / "problem_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            pass
    return {"operator": problem.operator_name, "status": "child_no_summary",
            "kernels": []}


def run_parent(args: argparse.Namespace) -> None:
    levels = [int(x) for x in args.levels.split(",")] if args.levels else [args.level]
    problem_filter: Optional[List[str]] = None
    if args.problem:
        problem_filter = [p.strip() for p in args.problem.split(",") if p.strip()]

    kb_root = Path(args.kernelbench_root).resolve()
    if not kb_root.is_dir():
        logger.error("[kb] KernelBench root not found: %s", kb_root)
        sys.exit(1)

    problems = discover_problems(kb_root, levels=levels, problem_filter=problem_filter)
    if not problems:
        logger.error("[kb] No problems matched (levels=%s, filter=%s)",
                     levels, problem_filter)
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[kb] %d problems selected; output -> %s/%s/",
                len(problems), output_dir, args.gpu)

    per_problem: List[Dict[str, Any]] = []
    for p in problems:
        per_problem.append(_spawn_child(p, output_dir, args))

    _write_metadata_yaml(Path(args.metadata_out), per_problem)
    run_list_path = (PROJECT_ROOT / "run_lists"
                     / f"{args.gpu}_kernelbench_include.json")
    _write_run_list(run_list_path, per_problem, args.threshold_ms, args.gpu)

    # Aggregate run summary
    summary = {
        "gpu": args.gpu, "threshold_ms": args.threshold_ms,
        "total_problems": len(per_problem),
        "ok": sum(1 for e in per_problem if e.get("status") == "ok"),
        "no_triton_kernels": sum(1 for e in per_problem
                                 if e.get("status") == "no_triton_kernels"),
        "eager_failed": sum(1 for e in per_problem
                            if e.get("status") == "eager_failed"),
        "compile_failed": sum(1 for e in per_problem
                              if e.get("status") == "compile_failed"),
        "total_kernels": sum(len(e.get("kernels", [])) for e in per_problem),
        "correctness_pass": sum(1 for e in per_problem if e.get("correctness")),
        "per_problem": per_problem,
    }
    summary_path = output_dir / args.gpu / "kernelbench_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("[kb] Run summary: %s", summary_path)
    logger.info("[kb] OK=%d  no_triton=%d  eager_failed=%d  compile_failed=%d  kernels=%d",
                summary["ok"], summary["no_triton_kernels"],
                summary["eager_failed"], summary["compile_failed"],
                summary["total_kernels"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True, help="GPU name subfolder, e.g. v100")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "ir_output"))
    p.add_argument("--kernelbench-root", default=str(DEFAULT_KERNELBENCH_ROOT))
    p.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    p.add_argument("--threshold-ms", type=float, default=DEFAULT_THRESHOLD_MS,
                   help="Drop kernels whose median latency is below this (ms).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32", choices=["fp16", "fp32", "bf16"],
                   help="Default fp32 matches KernelBench's reference convention; "
                        "use fp16/bf16 if your target corpus is half-precision.")
    p.add_argument("--level", type=int, default=2,
                   help="Single level to process (default 2).")
    p.add_argument("--levels", default=None,
                   help="Comma-separated levels, overrides --level.")
    p.add_argument("--problem", default=None,
                   help="Comma-separated problem stems (e.g. 14_Gemm_Divide_Sum_Scaling)")

    # Child-mode flags
    p.add_argument("--run-child", action="store_true")
    p.add_argument("--problem-stem", default=None,
                   help="(internal) Single problem stem for child mode")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = get_parser().parse_args()
    if args.run_child:
        run_child(args)
    else:
        run_parent(args)
