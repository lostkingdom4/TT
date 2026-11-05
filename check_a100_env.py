#!/usr/bin/env python3
"""Check that the current environment is ready to run extract/run_inductor.py
against an NVIDIA A100. Run from the repo root:

    python check_a100_env.py

Prints a section per check and a final PASS/FAIL.
"""
import os
import subprocess
import sys
from pathlib import Path

FAIL = False


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def fail(msg: str) -> None:
    global FAIL
    FAIL = True
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  OK: {msg}")


section("GPU (nvidia-smi)")
try:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    print(out)
    if "A100" not in out:
        fail("no A100 visible — check CUDA_VISIBLE_DEVICES")
    else:
        ok("A100 visible")
except Exception as exc:
    fail(f"nvidia-smi not runnable: {exc}")

section("CUDA_VISIBLE_DEVICES")
print(os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

section("PyTorch")
try:
    import torch

    print("torch:", torch.__version__)
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
    else:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]: {p.name}  sm_{p.major}{p.minor}  mem={p.total_memory // 2**30}GB")
            if p.major == 8 and p.minor == 0:
                ok(f"gpu[{i}] is sm_80 (A100)")
except Exception as exc:
    fail(f"torch import failed: {exc}")

section("Triton")
try:
    import triton

    print("triton:", triton.__version__)
    ok("triton importable")
except Exception as exc:
    fail(f"triton import failed: {exc}")

section("TritonBench import")
try:
    from tritonbench.operators import load_opbench_by_name

    targets = [
        "mamba2_chunk_scan",
        "mamba2_chunk_state",
        "rope",
        "rms_norm",
        "kl_div",
        "layer_norm",
        "sum",
    ]
    for op in targets:
        cls = load_opbench_by_name(op)
        if cls:
            ok(f"{op}: loadable")
        else:
            fail(f"{op}: load_opbench_by_name returned None")
except Exception as exc:
    fail(f"tritonbench import failed: {exc}")

section("Metadata yaml")
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
try:
    from paths import OSS_CUDA_KERNELS_YAML

    print("path:", OSS_CUDA_KERNELS_YAML)
    if Path(OSS_CUDA_KERNELS_YAML).exists():
        ok("yaml exists")
    else:
        fail("yaml missing")
except Exception as exc:
    fail(f"paths.OSS_CUDA_KERNELS_YAML unavailable: {exc}")

section("extract.run_inductor.parse_inductor_ops dry check")
try:
    from extract.run_inductor import parse_inductor_ops

    ops = parse_inductor_ops(Path(OSS_CUDA_KERNELS_YAML))
    for op in [
        "mamba2_chunk_scan",
        "mamba2_chunk_state",
        "rope",
        "rms_norm",
        "kl_div",
    ]:
        impls = ops.get(op)
        if impls:
            ok(f"{op}: pt2 impls = {impls}")
        else:
            fail(f"{op}: NOT pt2-tagged in yaml")
except Exception as exc:
    fail(f"parse_inductor_ops failed: {exc}")

section("ir_output/a100 state")
root = repo_root / "extract" / "ir_output" / "a100"
for op in [
    "mamba2_chunk_scan",
    "mamba2_chunk_state",
    "rope",
    "rms_norm",
    "kl_div",
    "layer_norm",
    "sum",
]:
    d = root / op
    if not d.exists():
        print(f"  {op}: MISSING DIR")
    else:
        ttgir = list(d.rglob("*.ttgir"))
        print(f"  {op}: {len(ttgir)} ttgir files{' (EMPTY)' if not ttgir else ''}")

print()
if FAIL:
    print("=== RESULT: FAIL — fix the FAILs above before running extract/run_inductor.py ===")
    sys.exit(1)
else:
    print("=== RESULT: PASS — ready to run extract/run_inductor.py ===")
    sys.exit(0)
