"""Per-kernel budget policy for the scaled-budget baseline comparison.

The LLM multi-agent system runs at a fixed budget of 9 evaluations per
shape.  The classical baselines (random / BO / AUC-Bandit) scale their
budget with the size of the valid search space so that they get a fair
shot on kernels where the space is large (GEMMs) without wasting
evaluations on kernels where the space is tiny (layer_norm's 16 configs).

Policy::

    budget(space_size) = clip(round(0.10 * space_size), 9, 150)

Consequences across the paper's 19 kernels:

* Spaces <= 96 (15 kernels)  -> floored to 9  (matched-budget regime)
* Spaces 288, 420, 576        -> 29, 42, 58
* Spaces >= 1500 (2 GEMMs)    -> capped at 150

Analysis computes "matched @ 9" by truncating the per-trial history to
the first 9 entries (no re-running).
"""

from typing import Any, Dict, Iterable, List, Optional

from .search_space import get_search_space


def space_size(search_space: Dict[str, Iterable[Any]]) -> int:
    """Return the Cartesian-product size of a ``{param: values}`` space."""
    if not search_space:
        return 0
    n = 1
    for values in search_space.values():
        n *= len(list(values))
    return n


def scaled_budget(
    size: int,
    floor: int = 9,
    ceil: int = 150,
    fraction: float = 0.10,
) -> int:
    """Scaled budget for one search space of the given size."""
    if size <= 0:
        return floor
    return max(floor, min(ceil, round(fraction * size)))


def budget_for(
    operator: str,
    kernel: Optional[str] = None,
    floor: int = 9,
    ceil: int = 150,
    fraction: float = 0.10,
) -> int:
    """Resolve the search space for ``(operator, kernel)`` and scale."""
    space = get_search_space(operator, kernel)
    return scaled_budget(space_size(space), floor, ceil, fraction)


def budget_table(kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one row per ``(operator, kernel)`` with size + scaled budget.

    Each input entry must carry ``operator`` and (optionally) ``kernel``
    keys -- i.e. the shape produced by ``discover_kernels`` /
    ``load_run_list``.
    """
    rows = []
    for k in kernels:
        op = k.get("operator") or k.get("name")
        kern = k.get("kernel")
        space = get_search_space(op, kern)
        size = space_size(space)
        rows.append({
            "operator": op,
            "kernel": kern,
            "space_size": size,
            "budget": scaled_budget(size),
        })
    return rows


def print_budget_table(kernels: List[Dict[str, Any]]) -> None:
    rows = budget_table(kernels)
    print(f"\n{'=' * 78}")
    print(f"  Scaled-budget table  (policy: clip(round(0.10 * size), 9, 150))")
    print(f"{'=' * 78}")
    print(f"  {'Operator':<22} {'Kernel':<42} {'Size':>6} {'Budget':>8}")
    print(f"  {'-' * 22} {'-' * 42} {'-' * 6} {'-' * 8}")
    for r in rows:
        label = r["kernel"][:42] if r["kernel"] else "-"
        print(f"  {r['operator']:<22} {label:<42} {r['space_size']:>6} {r['budget']:>8}")
    print(f"{'=' * 78}")
    total = sum(r["budget"] for r in rows)
    floored = sum(1 for r in rows if r["budget"] == 9)
    capped = sum(1 for r in rows if r["budget"] == 150)
    print(f"  kernels: {len(rows)}  total budget/seed: {total}  "
          f"floored@9: {floored}  capped@150: {capped}")
    print()
