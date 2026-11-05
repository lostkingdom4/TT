"""Post-benchmark output verification.

Compares outputs from the baseline implementation (PyTorch / eager)
against the Triton implementation for each shape, using a dtype-aware
tolerance.  Mutates per-shape result dicts in place, setting
``verified`` / ``verification_error`` / ``status`` keys.
"""

import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def _verify_outputs(op, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Post-benchmark output verification.

    Iterates over ``op.output.result`` and compares outputs across
    implementations (baseline vs triton) when both are available.
    """
    import torch

    output = getattr(op, "output", None)
    if output is None or not hasattr(output, "result"):
        return results

    # Identify baseline and triton impl names (from the first shape's entry).
    baseline_name = None
    triton_name = None
    for _x_val, y_vals in output.result:
        for impl_name in y_vals:
            lower = impl_name.lower()
            if "torch" in lower or "eager" in lower or "baseline" in lower:
                baseline_name = impl_name
            elif "triton" in lower:
                triton_name = impl_name
        break

    if not baseline_name or not triton_name:
        log.debug(
            "Skipping cross-impl verification (baseline=%s, triton=%s)",
            baseline_name, triton_name,
        )
        return results

    for x_val, y_vals in output.result:
        baseline_metrics = y_vals.get(baseline_name)
        triton_metrics = y_vals.get(triton_name)
        if baseline_metrics is None or triton_metrics is None:
            continue

        baseline_out = getattr(baseline_metrics, "output", None)
        triton_out = getattr(triton_metrics, "output", None)
        if baseline_out is None or triton_out is None:
            continue

        try:
            if isinstance(baseline_out, torch.Tensor) and isinstance(
                triton_out, torch.Tensor
            ):
                if baseline_out.dtype in (torch.float16, torch.bfloat16):
                    atol, rtol = 1e-2, 1e-2
                elif baseline_out.dtype == torch.float32:
                    atol, rtol = 1e-4, 1e-4
                else:
                    atol, rtol = 0, 0

                is_close = torch.allclose(
                    baseline_out.float(), triton_out.float(),
                    atol=atol, rtol=rtol,
                )

                for r in results:
                    if (
                        str(r.get("input_id")) == str(x_val)
                        and r.get("triton_impl_name") == triton_name
                    ):
                        r["verified"] = is_close
                        if not is_close:
                            max_diff = (
                                (baseline_out.float() - triton_out.float())
                                .abs().max().item()
                            )
                            r["verification_error"] = (
                                f"Output mismatch: max_diff={max_diff:.6e}"
                            )
                            r["status"] = "correctness_error"
                        break
        except Exception as exc:
            log.debug("Verification comparison failed for shape %s: %s",
                      x_val, exc)

    return results
