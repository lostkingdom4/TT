#!/usr/bin/env python3
"""Capture per-kernel @triton.autotune candidate configs for KernelBench.

Motivation
----------
``extract/run_kernelbench.py`` drives Inductor with ``max_autotune=True`` and
hooks ``CachingAutotuner.run`` — which only fires for the **winning** template
candidate. So the ``.py`` we save on disk is the post-autotune winner, with
one specific ``BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages`` baked in.

``optimizer_functions.BASELINE_USE_AUTOTUNE_FIRST=1`` wants the *first*
candidate's config as the baseline — but the first-candidate config isn't
recorded anywhere for KernelBench templates.

Approach
--------
Monkey-patch ``sa.TritonTemplateCaller.precompile`` (same hook point as
``extract/capture_triton_mm_only.py``). Inductor precompiles *every*
autotune candidate before benchmarking them, so this hook sees all of them
regardless of who wins. For each (template kernel name), we record every
candidate's config parsed from its generated ``.py`` source, in the order
Inductor emitted them. First candidate == "first config".

Output
------
Writes ``{op_dir}/{kname}_autotune_configs.json`` per template kernel:

    {
      "kernel": "triton_tem_fused_mm_0",
      "operator": "kb_L1_10_3D_tensor_matrix_multiplication",
      "candidates": [
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16, "GROUP_M": 8,
         "num_warps": 2, "num_stages": 5},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch._inductor.config as inductor_config
import torch._inductor.select_algorithm as sa

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "extract"))
from kernelbench_utils import KernelBenchProblem, discover_problems, load_problem_module

logger = logging.getLogger("capture_kb_autotune")

# ``BLOCK_M : tl.constexpr = 64`` at the top of the kernel body
_CONSTEXPR_RX = re.compile(
    r"^\s*(BLOCK_[A-Z]+|GROUP_M|EVEN_K|ALLOW_TF32)\s*:\s*tl\.constexpr\s*=\s*(\S+)",
    re.MULTILINE,
)
# ``num_stages=4,`` / ``num_warps=8,`` inside @triton_heuristics.template(...)
_NUM_WARPS_RX = re.compile(r"\bnum_warps\s*=\s*(\d+)")
_NUM_STAGES_RX = re.compile(r"\bnum_stages\s*=\s*(\d+)")


def _parse_config_from_source(src: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    m = _NUM_STAGES_RX.search(src)
    if m:
        cfg["num_stages"] = int(m.group(1))
    m = _NUM_WARPS_RX.search(src)
    if m:
        cfg["num_warps"] = int(m.group(1))
    for name, val in _CONSTEXPR_RX.findall(src):
        v: Any
        try:
            v = int(val)
        except ValueError:
            if val == "True":
                v = True
            elif val == "False":
                v = False
            else:
                v = val
        cfg[name] = v
    return cfg


def _install_hook(candidates: Dict[str, List[Dict[str, Any]]]):
    """Return ``original_precompile`` so the caller can restore it."""
    original = sa.TritonTemplateCaller.precompile

    def _patched(self):
        result = original(self)
        try:
            kname = self.bmreq.kernel_name
            src_path = Path(self.bmreq.module_path)
            exists = src_path.exists()
            logger.debug("hook: kname=%s src=%s exists=%s",
                         kname, src_path, exists)
            if exists:
                src = src_path.read_text()
                cfg = _parse_config_from_source(src)
                cfg["_source_path"] = str(src_path)
                candidates.setdefault(kname, []).append(cfg)
            else:
                logger.warning("hook: src_path missing for %s -> %s",
                               kname, src_path)
        except Exception as e:
            logger.warning("hook parse failed: %s", e)
        return result

    sa.TritonTemplateCaller.precompile = _patched
    return original


def _capture_problem(problem: KernelBenchProblem, output_dir: Path) -> Dict[str, Any]:
    op_name = problem.operator_name
    op_dir = output_dir / op_name
    if not op_dir.is_dir():
        return {"operator": op_name, "status": "no_op_dir", "kernels_captured": 0}

    module = load_problem_module(problem)
    init_args = module.get_init_inputs()
    inputs = [t.cuda() if torch.is_tensor(t) else t for t in module.get_inputs()]
    model = module.Model(*init_args).cuda().eval()

    candidates: Dict[str, List[Dict[str, Any]]] = {}
    original = _install_hook(candidates)
    try:
        torch._dynamo.reset()
        # Disable Inductor's compile caches so the precompile hook actually
        # fires on every run (otherwise re-running on the same shape short-
        # circuits and loads the cached winner without re-autotuning).
        with inductor_config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            force_disable_caches=True,
            autotune_local_cache=False,
            autotune_remote_cache=False,
            fx_graph_cache=False,
        ):
            compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
            with torch.no_grad():
                _ = compiled(*inputs)
        logger.info("  hook fired for %d kernel name(s): %s",
                    len(candidates), list(candidates.keys()))
    finally:
        sa.TritonTemplateCaller.precompile = original
        del model, inputs
        torch.cuda.empty_cache()

    # Inductor's precompile hook sees the *template* name (e.g. "triton_mm",
    # "triton_bmm"), but by the time run_kernelbench saved the .py it was
    # renamed to the *fused* name (e.g. "triton_tem_fused_convolution_1" even
    # when the template was "triton_mm" for a 1x1 conv). We map captured
    # candidates to every matching template kernel file in the op_dir.
    written = 0
    # All captured template kernels have the same Triton search space
    # (BLOCK_M/N/K, num_warps, num_stages, GROUP_M). If exact-name match fails,
    # use the first captured family's candidates for all final template kernels.
    all_final_kernels = sorted({
        py.stem for py in op_dir.glob("triton_tem_fused_*.py")
        if "_input" not in py.stem
    })
    for kname, cfgs in candidates.items():
        base = kname[len("triton_"):] if kname.startswith("triton_") else kname
        payload = {
            "kernel_template": kname,
            "operator": op_name,
            "candidates": [{k: v for k, v in c.items() if k != "_source_path"}
                           for c in cfgs],
            "candidate_sources": [c.get("_source_path") for c in cfgs],
        }
        # Prefer exact family match; fall back to every template file when
        # no name match (e.g. triton_mm template fused into convolution_1).
        matched = sorted({
            py.stem for py in op_dir.glob(f"triton_tem_fused_{base}_*.py")
            if "_input" not in py.stem
        })
        final_kernels = matched if matched else all_final_kernels
        if not final_kernels:
            logger.warning(
                "[%s] captured template %s but no triton_tem_fused_*.py "
                "present in %s — skipping",
                op_name, kname, op_dir,
            )
            continue
        note = "exact-match" if matched else "fallback-all"
        for final in final_kernels:
            out_payload = dict(payload)
            out_payload["kernel"] = final
            out_payload["mapping_mode"] = note
            out = op_dir / f"{final}_autotune_configs.json"
            out.write_text(json.dumps(out_payload, indent=2))
            written += 1
            logger.info("[%s] %s (template=%s, %s): %d candidates -> %s",
                        op_name, final, kname, note, len(cfgs), out.name)

    return {"operator": op_name, "status": "ok", "kernels_captured": written,
            "kernel_names": sorted(candidates)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernelbench-root", type=Path,
                   default=_PROJECT_ROOT / "extract" / "kernelbench_src"
                           / "KernelBench" / "KernelBench",
                   help="Where level{1,2,3}/*.py live")
    p.add_argument("--gpu", default="v100",
                   help="Subdir of extract/ir_output/ to update")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override extract/ir_output/<gpu>/ output root")
    p.add_argument("--level", type=int, action="append", default=None,
                   help="Levels to process (repeatable); default 1 2 3")
    p.add_argument("--problem", action="append", default=None,
                   help="Exact pid-slug or operator_name filter (repeatable)")
    p.add_argument("--only-templates", action="store_true",
                   default=True,
                   help="Skip problems without a triton_tem_* kernel on disk "
                        "(default). Saves time on pure-pointwise problems.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = args.output_dir or (_PROJECT_ROOT / "extract" / "ir_output" / args.gpu)
    if not output_dir.is_dir():
        print(f"Output dir not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    levels = args.level or [1, 2, 3]
    problems = discover_problems(args.kernelbench_root, levels=levels,
                                 problem_filter=args.problem)
    if not problems:
        print(f"No problems matched (root={args.kernelbench_root}, "
              f"levels={levels}, filter={args.problem})", file=sys.stderr)
        sys.exit(1)

    # Filter to problems whose op_dir contains a triton_tem_* source.
    if args.only_templates:
        kept = []
        for prob in problems:
            op_dir = output_dir / prob.operator_name
            if any(op_dir.glob("triton_tem_*.py")):
                # Don't re-run problems that already have config JSON for every
                # template kernel present.
                need = False
                for py in op_dir.glob("triton_tem_*.py"):
                    kname = py.stem.split("_input")[0]  # strip _input{N}
                    cfg_json = op_dir / f"{kname}_autotune_configs.json"
                    if not cfg_json.exists():
                        need = True
                        break
                if need:
                    kept.append(prob)
        logger.info("Filtered to %d/%d problems with missing template configs",
                    len(kept), len(problems))
        problems = kept

    stats = {"ok": 0, "failed": 0, "total_kernels": 0, "no_op_dir": 0}
    # Subprocess-per-problem isolates Inductor/Triton caches so the precompile
    # hook fires fresh for every problem. Running multiple problems in one
    # Python process caused the second-and-later problems with the same shape
    # to short-circuit precompile to 0.0001s (cache hit) and the hook to
    # produce no candidates.
    if len(problems) > 1 and not os.environ.get("KB_AUTOTUNE_SINGLE_PROCESS"):
        for i, prob in enumerate(problems, 1):
            logger.info("[%d/%d] %s (subprocess)", i, len(problems),
                        prob.operator_name)
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--gpu", args.gpu,
                "--level", str(prob.level),
                "--problem", f"{prob.pid}_{prob.slug}",
                "--log-level", args.log_level,
                "--kernelbench-root", str(args.kernelbench_root),
            ]
            if args.output_dir:
                cmd.extend(["--output-dir", str(args.output_dir)])
            env = dict(os.environ, KB_AUTOTUNE_SINGLE_PROCESS="1")
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                logger.error("[%s] subprocess failed rc=%d\n%s",
                             prob.operator_name, r.returncode, r.stderr[-2000:])
                stats["failed"] += 1
                continue
            stats["ok"] += 1
            # Count JSONs written by the subprocess in this op_dir
            op_dir = output_dir / prob.operator_name
            stats["total_kernels"] += len(list(op_dir.glob("*_autotune_configs.json")))
        logger.info("Done. %s", stats)
        return
    # Single-problem / subprocess-internal path
    for i, prob in enumerate(problems, 1):
        logger.info("[%d/%d] %s", i, len(problems), prob.operator_name)
        try:
            r = _capture_problem(prob, output_dir)
        except Exception as e:
            logger.exception("[%s] capture failed: %s", prob.operator_name, e)
            stats["failed"] += 1
            continue
        if r["status"] == "ok":
            stats["ok"] += 1
            stats["total_kernels"] += r["kernels_captured"]
        elif r["status"] == "no_op_dir":
            stats["no_op_dir"] += 1
    logger.info("Done. %s", stats)


if __name__ == "__main__":
    main()
