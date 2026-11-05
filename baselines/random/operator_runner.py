"""Run a single TritonBench operator with randomly generated configs.

Delegates injection / verification / latency extraction to
``baselines.common``.  Invoked as a subprocess by ``run_all.py``.
"""

import argparse
import gc
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from baselines.common.injector import _RandomConfigInjector
from baselines.common.tritonbench_setup import (
    build_tb_args,
    extract_latency,
    fresh_cache_context,
    jsonable,
    load_op_class,
)
from baselines.common.verify import _verify_outputs

log = logging.getLogger(__name__)


def run_operator(
    operator: str,
    seed: int = 42,
    device: str = "cuda",
    warmup: int = 25,
    rep: int = 100,
    has_autotune: bool = False,
    kernel: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Benchmark all shapes of ``operator`` using random configs."""
    import torch

    tb_args, extra_args = build_tb_args(
        device=device, precision="fp16",
        metrics="latency,best_config",
        warmup=warmup, rep=rep,
    )

    OpClass = load_op_class(operator)

    results: List[Dict[str, Any]] = []

    with fresh_cache_context():
        try:
            op = OpClass(tb_args=tb_args, extra_args=extra_args)
        except Exception as exc:
            log.error("Failed to instantiate %s: %s", operator, exc)
            return [_error_result(operator, -1, seed, has_autotune,
                                  str(exc), kernel=kernel)]

        injector = _RandomConfigInjector(
            operator, seed, has_autotune, kernel=kernel,
        )
        try:
            injector.__enter__()
            op.run(warmup=warmup, rep=rep)
        except Exception as exc:
            log.error("op.run() failed for %s: %s", operator, exc)
        finally:
            injector.__exit__(None, None, None)

        output = getattr(op, "output", None)
        if output is None or not hasattr(output, "result"):
            log.warning("No output from operator %s", operator)
            return [_error_result(operator, -1, seed, has_autotune,
                                  "No output produced", kernel=kernel)]

        config_idx = 0
        for x_val, y_vals in output.result:
            for impl_name, metrics in y_vals.items():
                latency_ms = extract_latency(metrics)
                error_msg = getattr(metrics, "error_msg", None)
                random_cfg = injector.get_config(config_idx)
                config_idx += 1

                status = "success" if latency_ms is not None else "error"
                if error_msg:
                    status = "error"
                verified = status == "success"

                results.append({
                    "operator": operator,
                    "kernel": kernel,
                    "input_id": x_val,
                    "x_val": str(x_val),
                    "random_seed": seed,
                    "has_autotune": has_autotune,
                    "random_config": (
                        {str(k): jsonable(v) for k, v in random_cfg.items()}
                        if random_cfg else None
                    ),
                    "latency_ms": latency_ms,
                    "verified": verified,
                    "verification_error": error_msg if not verified else None,
                    "triton_impl_name": impl_name,
                    "status": status,
                    "error": error_msg,
                })

        log.info("Running output verification for %s ...", operator)
        results = _verify_outputs(op, results)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    return results


def _error_result(
    operator: str, input_id: int, seed: int, has_autotune: bool,
    error: str, kernel: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "operator": operator,
        "kernel": kernel,
        "input_id": input_id,
        "x_val": str(input_id),
        "random_seed": seed,
        "has_autotune": has_autotune,
        "random_config": None,
        "latency_ms": None,
        "verified": False,
        "verification_error": error,
        "triton_impl_name": None,
        "status": "error",
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a TritonBench operator with random configs"
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--kernel", default=None,
                        help="Kernel name (enables per-kernel search space)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--has-autotune", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("Running %s (kernel=%s, seed=%d, autotune=%s)",
             args.operator, args.kernel, args.seed, args.has_autotune)

    try:
        results = run_operator(
            operator=args.operator,
            seed=args.seed,
            device=args.device,
            warmup=args.warmup,
            rep=args.rep,
            has_autotune=args.has_autotune,
            kernel=args.kernel,
        )
    except Exception as exc:
        log.error("Fatal error running %s: %s", args.operator, exc)
        traceback.print_exc()
        results = [_error_result(args.operator, -1, args.seed,
                                 args.has_autotune, f"Fatal: {exc}",
                                 kernel=args.kernel)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    ok = sum(1 for r in results if r["status"] == "success")
    err = sum(1 for r in results if r["status"] == "error")
    corr_err = sum(1 for r in results if r["status"] == "correctness_error")
    verified = sum(1 for r in results if r.get("verified"))
    print(
        f"[{args.operator}] shapes={len(results)}, ok={ok}, errors={err}, "
        f"correctness_errors={corr_err}, verified={verified}"
    )


if __name__ == "__main__":
    main()
