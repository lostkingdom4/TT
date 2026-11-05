"""Shared Triton configuration search space.

All baselines (random, autotune, BO, future EA) call
``get_search_space(operator, kernel)`` so they tune over the same
parameter space — a prerequisite for apples-to-apples comparison in the
paper's Table 2.

Design
------
One unified value table, ``UNIFIED_VALUE_RANGES``, defines the full
set of values every knob can take.  Per-kernel entries in
``KERNEL_ACTIVE_KNOBS`` name *which* knobs a kernel actually uses;
``get_search_space`` builds the per-kernel dict by pulling those knobs
from the unified table.  This way:

* Widening a knob's value set is a one-line edit in one place.
* Adding a new kernel means naming its knobs, not duplicating values.
* Per-kernel Cartesian products stay bounded (the autotune runner only
  enumerates the knobs the kernel actually uses).

The value ranges are the superset of (a) values already tuned by any
existing kernel, (b) values proposed by the multi-agent system (see
``multiagent.py:_generate_candidate_configs`` and observed
``live_summary.json`` outputs under
``output/v100_improved_all_shapes_A5_proposals/``), and (c) nearby
powers of two so random/BO can explore in both directions.

Unknown knobs are **harmless**: the injector (see
``baselines/common/injector.py``) only installs a knob on an
autotune-wrapped kernel if it appears in ``template.kwargs``, and on a
JIT-patched kernel if it appears in the launch ``kwargs``.  Everything
else is silently dropped.  Callers may therefore pass configs that
contain knobs a given kernel doesn't use.

Lookup order for ``get_search_space(operator, kernel)``:

1. ``KERNEL_ACTIVE_KNOBS[(operator, kernel)]`` — per-kernel knob set.
2. ``OPERATOR_ACTIVE_KNOBS[operator]`` — per-operator fallback for the
   standard TritonBench operators (softmax, addmm, etc.) so that
   legacy runs of ``baselines.random.run_all`` without ``--kernel``
   keep working.
3. ``DEFAULT_ACTIVE_KNOBS`` — last resort (num_warps × num_stages only).

``CONFIG_LEVEL_PARAMS`` marks which keys belong on ``triton.Config``
(num_warps, num_stages, num_ctas) versus the kernel's constexpr kwargs.
"""

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import triton

# -----------------------------------------------------------------------
# Parameters that live on triton.Config (not constexpr kwargs)
# -----------------------------------------------------------------------
CONFIG_LEVEL_PARAMS = {"num_warps", "num_stages", "num_ctas"}


# =======================================================================
# ONE unified value table — widen here, everything else propagates.
# =======================================================================
UNIFIED_VALUE_RANGES: Dict[str, List] = {
    # ---- triton.Config-level knobs -----------------------------------
    "num_warps":  [1, 2, 4, 8, 16, 32],
    "num_stages": [1, 2, 3, 4, 5],
    "num_ctas":   [1],

    # ---- Inductor pointwise / 1-D block knob -------------------------
    "XBLOCK": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
               8192, 16384],
    # ---- Inductor 2-D pointwise (rare in current run list) -----------
    "YBLOCK": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    # ---- Inductor reduction ------------------------------------------
    "RBLOCK": [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],

    # ---- Handwritten layer_norm (BLOCK_SIZE must cover all columns) --
    # Agents propose BLOCK_SIZE up to 16384; wider than strictly
    # necessary, but the injector drops values that don't compile.
    "BLOCK_SIZE": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],

    # ---- GEMM --------------------------------------------------------
    "BLOCK_M": [16, 32, 64, 128, 256],
    "BLOCK_N": [16, 32, 64, 128, 256],
    "BLOCK_K": [16, 32, 64, 128],
    "GROUP_M": [1, 2, 4, 8, 16],

    # ---- Jagged operators (TritonBench) ------------------------------
    "BLOCK_SIZE_RAGGED": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    "BLOCK_SIZE_M":      [8, 16, 32, 64, 128],

    # ---- sum-op (TritonBench) ----------------------------------------
    "BLOCK_SIZE_NON_REDUCE_DIM": [1, 2, 4, 8, 16, 32, 64],
    "BLOCK_SIZE_REDUCE_DIM":     [1, 2, 4, 8, 16, 32, 64],
    "BLOCK_SIZE_N":              [1, 4, 16, 64, 256, 1024],
    "BLOCK_SIZE_K":              [1, 4, 16, 64, 256],
}


# =======================================================================
# Per-kernel active knob sets — names only; values come from the table.
# =======================================================================
# Covers the 19 kernels in run_lists/v100_improved_only.json plus the
# TritonBench operators used by legacy (no-kernel) calls.
_PW1D = {"XBLOCK", "num_warps", "num_stages"}               # Inductor pointwise / 1-D
_RED  = {"XBLOCK", "RBLOCK", "num_warps", "num_stages"}     # Inductor reduction
_GEMM_GROUP = {"BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M",
               "num_warps", "num_stages"}
_GEMM       = {"BLOCK_M", "BLOCK_N", "BLOCK_K",
               "num_warps", "num_stages"}

KERNEL_ACTIVE_KNOBS: Dict[Tuple[str, str], Set[str]] = {
    # -- Handwritten / Inductor-templated GEMMs ------------------------
    ("addmm", "_addmm_fwd"):               _GEMM_GROUP,
    ("addmm", "triton_mm"):                _GEMM_GROUP,
    ("addmm", "triton_tem_fused_addmm_0"): _GEMM,

    # -- Inductor pointwise kernels ------------------------------------
    ("embedding", "triton_poi_fused_embedding_0"):                      _PW1D,
    ("geglu",     "triton_poi_fused_gelu_mul_0"):                       _PW1D,
    ("int4_gemm", "triton_poi_fused___lshift___bitwise_and_bitwise_or_0"): _PW1D,

    ("jagged_layer_norm",
     "triton_poi_fused__jagged_to_padded_dense_forward__padded_dense_to_"
     "jagged_forward_add_div_mul_sqrt_sub_view_2"): _PW1D,

    ("mamba2_chunk_scan",  "triton_poi_fused_add_mul_4"): _PW1D,
    ("mamba2_chunk_scan",  "triton_poi_fused_bmm_1"):     _PW1D,
    ("mamba2_chunk_scan",  "triton_poi_fused_clone_2"):   _PW1D,
    ("mamba2_chunk_scan",  "triton_poi_fused_clone_3"):   _PW1D,
    ("mamba2_chunk_scan",  "triton_poi_fused_mul_0"):     _PW1D,
    ("mamba2_chunk_state", "triton_poi_fused_clone_0"):   _PW1D,
    ("mamba2_chunk_state", "triton_poi_fused_clone_1"):   _PW1D,

    ("rope", "triton_poi_fused_add_cat_mul_0"): _PW1D,
    ("rope", "triton_poi_fused_add_cat_mul_1"): _PW1D,

    # -- Inductor reduction kernels ------------------------------------
    ("jagged_layer_norm",
     "triton_red_fused__jagged_to_padded_dense_forward_div_mul_pow_sub_"
     "sum_0"): _RED,
    ("jagged_softmax",
     "triton_red_fused__jagged_to_padded_dense_forward__softmax_0"): _RED,

    # -- Handwritten layer_norm ----------------------------------------
    # BLOCK_SIZE was previously not tuned (semantic — must cover all
    # columns).  Multi-agent proposals include BLOCK_SIZE up to 16384,
    # so it is now part of the search space; the injector drops values
    # that fail to compile.
    ("layer_norm", "_layer_norm_fwd_fused_no_bias"): {
        "BLOCK_SIZE", "num_warps", "num_stages",
    },
}


# =======================================================================
# Per-operator fallback (used when caller does not pass kernel).
# =======================================================================
OPERATOR_ACTIVE_KNOBS: Dict[str, Set[str]] = {
    "softmax":    {"num_warps", "num_stages"},
    "addmm":      _GEMM_GROUP,
    "jagged_softmax": {"BLOCK_SIZE_RAGGED", "BLOCK_SIZE_M",
                       "num_warps", "num_stages"},
    "jagged_sum":  {"BLOCK_SIZE_RAGGED", "BLOCK_SIZE_M",
                    "num_warps", "num_stages"},
    "jagged_mean": {"BLOCK_SIZE_RAGGED", "BLOCK_SIZE_M",
                    "num_warps", "num_stages"},
    "layer_norm": {"num_warps", "num_stages"},
    "sum":        {"BLOCK_SIZE_NON_REDUCE_DIM", "BLOCK_SIZE_REDUCE_DIM",
                   "BLOCK_SIZE_N", "BLOCK_SIZE_K",
                   "num_warps", "num_stages"},
    "vector_add":      {"BLOCK_SIZE", "num_warps", "num_stages"},
    "vector_exp":      {"BLOCK_SIZE", "num_warps", "num_stages"},
    "low_mem_dropout": {"BLOCK_SIZE", "num_warps", "num_stages"},
}


DEFAULT_ACTIVE_KNOBS: Set[str] = {"num_warps", "num_stages"}


def _materialize(knobs: Set[str]) -> Dict[str, List]:
    """Build {knob: values} by pulling each name from the unified table."""
    return {k: list(UNIFIED_VALUE_RANGES[k]) for k in knobs
            if k in UNIFIED_VALUE_RANGES}


# -----------------------------------------------------------------------
# Back-compat aliases: older code imported these dicts directly.
# -----------------------------------------------------------------------
KERNEL_SEARCH_SPACES: Dict[Tuple[str, str], Dict[str, List]] = {
    key: _materialize(knobs) for key, knobs in KERNEL_ACTIVE_KNOBS.items()
}
OPERATOR_SEARCH_SPACES: Dict[str, Dict[str, List]] = {
    op: _materialize(knobs) for op, knobs in OPERATOR_ACTIVE_KNOBS.items()
}
DEFAULT_SEARCH_SPACE: Dict[str, List] = _materialize(DEFAULT_ACTIVE_KNOBS)


# -----------------------------------------------------------------------
# Per-(operator, kernel) explicit config lists that mirror each kernel's
# own @triton.autotune(configs=[...]) decorator verbatim.
#
# Used by the "Autotune" baseline so we never go through Triton's
# built-in autotuner selection (which has been observed to mispick by
# ~1.45× on V100 for GEMM shapes due to its short internal benchmark
# window). Every measurement — including t_orig — goes through
# _FixedConfigInjector + do_bench, yielding a measurement methodology
# identical to batch_runner / compare_configs.py.
#
# KERNEL_ACTIVE_KNOBS above is kept intact for a future "full-grid"
# baseline that does not restrict itself to the decorator's list.
# -----------------------------------------------------------------------
KERNEL_AUTOTUNE_CONFIGS: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
    ("addmm", "_addmm_fwd"): [
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
    ],
    ("addmm", "triton_mm"): [
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
    ],
    ("addmm", "triton_tem_fused_addmm_0"): [
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 2, "num_stages": 5},
    ],
    # NOTE: int4_gemm's ``triton_poi_fused___lshift___bitwise_and_bitwise_or_0``
    # is an Inductor pointwise (XBLOCK) kernel, NOT the int4-GEMM matmul.
    # It is not @triton.autotune-wrapped itself — operator-level detection
    # is fooled by the GEMM kernel in the same directory. We intentionally
    # do NOT list it here so the baseline falls back to the pointwise
    # search space built from _PW1D.
    #
    # jagged_softmax @triton.autotune from tb/tritonbench/operators/jagged_softmax/kernels.py:
    #   itertools.product(BLOCK_SIZES_RAGGED=[8,128,2048], BLOCK_SIZES_M=[8,64],
    #                     NUM_WARPS=[4,8], NUM_STAGES=[2,4]) = 24 configs
    ("jagged_softmax",
     "triton_red_fused__jagged_to_padded_dense_forward__softmax_0"): [
        {"BLOCK_SIZE_RAGGED": b_r, "BLOCK_SIZE_M": b_m,
         "num_warps": w, "num_stages": s}
        for b_r in [8, 128, 2048]
        for b_m in [8, 64]
        for w in [4, 8]
        for s in [2, 4]
    ],
}


# -----------------------------------------------------------------------
# public helpers
# -----------------------------------------------------------------------

def get_search_space(
    operator: str,
    kernel: Optional[str] = None,
) -> Dict[str, List]:
    """Return the search space for ``(operator, kernel)``.

    Lookup order:
      1. ``KERNEL_ACTIVE_KNOBS[(operator, kernel)]`` if ``kernel`` given.
      2. ``OPERATOR_ACTIVE_KNOBS[operator]``.
      3. ``DEFAULT_ACTIVE_KNOBS``.

    Values come from the single ``UNIFIED_VALUE_RANGES`` table.
    """
    if kernel is not None:
        knobs = KERNEL_ACTIVE_KNOBS.get((operator, kernel))
        if knobs is not None:
            return _materialize(knobs)
    knobs = OPERATOR_ACTIVE_KNOBS.get(operator)
    if knobs is not None:
        return _materialize(knobs)
    return _materialize(DEFAULT_ACTIVE_KNOBS)


def get_autotune_configs(
    operator: str,
    kernel: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Return the hardcoded `@triton.autotune` config list for this kernel.

    Returns ``None`` if the kernel is not in ``KERNEL_AUTOTUNE_CONFIGS``
    (i.e. it isn't wrapped with ``@triton.autotune`` as far as we know).
    """
    if kernel is None:
        return None
    return KERNEL_AUTOTUNE_CONFIGS.get((operator, kernel))


def generate_random_config(
    operator: str,
    rng: random.Random,
    kernel: Optional[str] = None,
) -> Dict[str, Any]:
    """Pick one value at random for every parameter in the search space."""
    space = get_search_space(operator, kernel)
    return {param: rng.choice(values) for param, values in space.items()}


def generate_random_triton_config(
    operator: str,
    rng: random.Random,
    template_config: Optional[triton.Config] = None,
    kernel: Optional[str] = None,
) -> triton.Config:
    """Generate a ``triton.Config`` with randomly chosen values.

    If ``template_config`` is given, any parameter present in the template
    but absent from our search space is carried over unchanged (preserves
    kernel-specific constexprs we don't know about).
    """
    flat = generate_random_config(operator, rng, kernel=kernel)

    kwargs: Dict[str, Any] = {}
    if template_config is not None:
        for k, v in template_config.kwargs.items():
            if k not in flat:
                kwargs[k] = v
    for k, v in flat.items():
        if k not in CONFIG_LEVEL_PARAMS:
            kwargs[k] = v

    return triton.Config(
        kwargs,
        num_warps=flat.get("num_warps", 4),
        num_stages=flat.get("num_stages", 2),
        num_ctas=getattr(template_config, "num_ctas", 1)
        if template_config
        else 1,
    )
