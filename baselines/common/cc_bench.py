"""Strategy-agnostic per-shape search loop using the compare_configs harness.

This is the compare_configs analogue of the inline TritonBench-based
loop in each scaled-budget runner's ``operator_runner.py``.  It loads
the extracted kernel source + per-input ``*_shapes.json`` and drives
``compare_configs.benchmark_config_pair`` per trial, matching the
paper's Full-system measurement path exactly.  The harness used is
recorded in every shape envelope as ``"harness": "compare_configs"``
for downstream consistency checks.

Used by ``baselines.bo``, ``baselines.random_search``, and
``baselines.auc_bandit`` when ``--harness compare_configs`` is set.
"""

import glob
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from baselines.common.budget import scaled_budget, space_size
from baselines.common.search_space import get_search_space

log = logging.getLogger(__name__)


def _median(xs: List[float]) -> Optional[float]:
    """Median of a list, or None if empty.  Pure helper for clarity."""
    return float(statistics.median(xs)) if xs else None


def _aggregate_orig_lat(samples: List[float]) -> Optional[Dict[str, float]]:
    """Cross-trial aggregate stats for the original-config measurements.

    Returns ``None`` when no fresh measurements were collected.  The
    median is what feeds the speedup ratio; the rest characterizes the
    do_bench noise floor (min / max / mean / stdev / CV).
    """
    if not samples:
        return None
    n = len(samples)
    mean = float(statistics.mean(samples))
    stdev = float(statistics.stdev(samples)) if n > 1 else 0.0
    return {
        "n_samples": n,
        "min":    float(min(samples)),
        "max":    float(max(samples)),
        "median": float(statistics.median(samples)),
        "mean":   mean,
        "stdev":  stdev,
        "cv":     (stdev / mean) if mean > 0 else 0.0,
    }


def enumerate_shape_ids(ir_dir: Path, operator: str, kernel: str) -> Set[int]:
    """Return the set of input IDs for a (operator, kernel) pair.

    Matches the same glob pattern used inside ``run_operator_cc`` so the
    caller (run_all's shape-level retry loop) can tell which shape IDs are
    expected to show up in a subprocess's output.  When a subprocess dies
    mid-run (SIGABRT / OOM / timeout) shapes it never reached are missing
    from its output entirely; comparing against this set lets the retry
    loop spawn a fresh subprocess for exactly those IDs.
    """
    pat = str(Path(ir_dir) / operator / f"*_{kernel}_input*_shapes.json")
    ids: Set[int] = set()
    for f in glob.glob(pat):
        m = re.search(r"_input(\d+)_shapes\.json$", f)
        if m:
            ids.add(int(m.group(1)))
    return ids


# ---------------------------------------------------------------------------
# Strategy dispatch (mirrors baselines.run_cached)
# ---------------------------------------------------------------------------

def _make_strategy(
    name: str, search_space: Dict[str, Any], n_trials: int, seed: int,
):
    if name == "random_search":
        from baselines.random_search.random_search import RandomSearch
        return RandomSearch(search_space, n_trials=n_trials, seed=seed)
    if name == "bo":
        from baselines.bo.bo_search import BOSearch
        return BOSearch(search_space, n_trials=n_trials, seed=seed)
    if name == "auc_bandit":
        from baselines.auc_bandit.search import AUCBanditSearch
        return AUCBanditSearch(search_space, n_trials=n_trials, seed=seed)
    raise ValueError(f"Unknown strategy: {name}")


# ---------------------------------------------------------------------------
# Per-shape loop
# ---------------------------------------------------------------------------

def run_operator_cc(
    operator: str,
    kernel: Optional[str],
    strategy_name: str,
    ir_dir: Path,
    *,
    budget_policy: str = "scaled",
    fixed_n_trials: int = 9,
    seed: int = 42,
    warmup: int = 5,
    rep: int = 20,
    has_autotune: bool = False,
    device: str = "cuda",
    validate_outputs: bool = False,
    skip_shape_ids: Optional[set] = None,
    incremental_output: Optional[Path] = None,
    per_shape_timeout_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Strategy-driven search over every shape of (operator, kernel).

    Returns a list of per-shape result dicts in the same schema as the
    TritonBench runners, with ``"harness": "compare_configs"`` and the
    strategy-specific trial-history key (``"bo_trial_history"`` for
    ``bo``, ``"trial_history"`` for everyone else).

    * ``incremental_output``: path for atomic per-trial JSON flushes.  Every
      trial (not just every shape) updates the row, so a SIGABRT / OOM /
      timeout mid-shape loses at most the one in-flight trial's latency.
      On process restart, ``run_operator_cc`` re-reads the file, finds any
      rows with partial ``trial_history``, and replays them into a fresh
      strategy to resume from the exact next trial — assumes deterministic
      strategies (seeded Optuna TPE / AUC-Bandit / Random all qualify).

    * ``per_shape_timeout_s``: wall-clock cap per shape.  Checked at each
      trial boundary.  If exceeded, remaining trials are recorded as
      ``error="shape exceeded Xs wall-clock timeout"`` with ``latency_ms=None``,
      the strategy still ``observe(None)`` for each (fair-budget
      accounting), and the shape is finalized with whatever was found.
    """
    import json as _json
    import os as _os
    import tempfile as _tempfile
    import time as _time
    import torch  # Late import so dry-run on CPU still loads the module.

    def _flush_incremental(rows: List[Dict[str, Any]]) -> None:
        if incremental_output is None:
            return
        try:
            out = Path(incremental_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with _tempfile.NamedTemporaryFile(
                mode="w", suffix=".tmp", prefix=f"{out.name}.",
                dir=str(out.parent), delete=False,
            ) as tmp:
                _json.dump(rows, tmp, default=str)
                tmp.flush()
                _os.fsync(tmp.fileno())
                tmp_path = tmp.name
            _os.replace(tmp_path, str(out))
        except OSError as e:
            log.warning("Failed to write incremental output to %s: %s",
                        incremental_output, e)

    def _load_prior_rows() -> Dict[int, Dict[str, Any]]:
        """Load existing per-shape rows from the incremental file (if any),
        keyed by ``input_id``, for mid-shape trial-history replay."""
        if incremental_output is None:
            return {}
        p = Path(incremental_output)
        if not (p.exists() and p.stat().st_size > 0):
            return {}
        try:
            with open(p) as f:
                prior = _json.load(f)
            return {r["input_id"]: r for r in prior
                    if isinstance(r, dict) and r.get("input_id") is not None}
        except Exception as e:
            log.warning("Could not load prior rows from %s for replay: %s", p, e)
            return {}

    prior_rows_by_id = _load_prior_rows()

    # Deferred imports avoid pulling tempfile/triton at module load time
    # when the analyzer (no-GPU) imports baselines.common transitively.
    from compare_configs import (
        _extract_hardcoded_constexprs,
        _is_template_kernel,
        benchmark_config_pair,
    )
    from baselines.autotune.operator_runner_cc import (
        _find_kernel_py,
        _load_baked_in_config,
        _shape_idx_from_path,
        _to_cc_config,
    )

    ir_dir = Path(ir_dir).resolve()
    search_space = get_search_space(operator, kernel)
    if not search_space:
        return [_error_result(
            operator, kernel, -1, fixed_n_trials,
            f"No search space for {operator}/{kernel}",
            strategy_name,
        )]

    if budget_policy == "scaled":
        effective_trials = scaled_budget(space_size(search_space))
    else:
        effective_trials = int(fixed_n_trials)

    if per_shape_timeout_s is None:
        # Default: 30 + 3*budget s, clipped to [60, 600].  Each trial calls
        # benchmark_config_pair which runs two do_bench(warmup=5, rep=20)
        # calls (~1 s each per the paper's measured timing), so a clean
        # trial costs ~2 s.  3 s/trial gives 1.5x headroom for slow configs
        # (small XBLOCK / large grid stretches rep=20 to ~3 s).  The 30 s
        # additive term absorbs first-trial JIT compile (TRITON_CACHE_DIR
        # is fresh per subprocess; pointwise compile ~5-15 s, GEMM
        # templates ~15-30 s).  60 s floor keeps tiny budgets sane; 600 s
        # ceiling caps pathological compile-hang cases.  At the paper's
        # K values this gives 111 s (K=27), 165 s (K=45), 480 s (K=150).
        per_shape_timeout_s = max(60.0, min(600.0, 30.0 + 3.0 * effective_trials))

    # Discover shapes files for this (op, kernel).  Pattern matches the
    # autotune cc runner: <op>/<op>_<kernel>_input<N>_shapes.json.
    if kernel is None:
        raise ValueError("compare_configs harness requires --kernel")
    shape_files = sorted(glob.glob(
        str(ir_dir / operator / f"*_{kernel}_input*_shapes.json")
    ))
    if not shape_files:
        return [_error_result(
            operator, kernel, -1, effective_trials,
            f"No *_input*_shapes.json files under {ir_dir / operator}",
            strategy_name,
        )]

    log.info(
        "Running %s/%s on %s: strategy=%s policy=%s trials=%d shapes=%d",
        operator, kernel, device, strategy_name, budget_policy,
        effective_trials, len(shape_files),
    )

    def _cuda_poisoned(msg: Optional[str]) -> bool:
        if not msg:
            return False
        low = msg.lower()
        return ("illegal memory access" in low
                or "cuda error" in low
                or "cuda-capable device(s) is/are busy" in low
                or "an illegal instruction" in low)

    skip_shape_ids = set(skip_shape_ids or ())
    results: List[Dict[str, Any]] = []
    cuda_dead = False  # once set, remaining shapes are skipped without calling the GPU
    for shape_file in shape_files:
        shape_path = Path(shape_file)
        shape_idx = _shape_idx_from_path(shape_file)
        if shape_idx is None:
            continue

        if shape_idx in skip_shape_ids:
            log.info("shape %d: already completed in prior attempt, skipping", shape_idx)
            continue

        baked_cfg = _load_baked_in_config(shape_path)
        if baked_cfg is None:
            log.warning("shape %d: missing baked-in config", shape_idx)
            continue

        kernel_py = _find_kernel_py(ir_dir, operator, kernel, shape_idx)
        if kernel_py is None:
            log.warning("shape %d: missing kernel .py", shape_idx)
            results.append(_error_result(
                operator, kernel, shape_idx, effective_trials,
                f"no kernel .py found under {ir_dir / operator}",
                strategy_name,
            ))
            continue

        # Template kernels (@triton_heuristics.template) hardcode M/N/K
        # strides and derived constants like ``N*idx_m`` as integer literals
        # inside the @triton.jit body, so every captured shape needs its own
        # ``{kernel}_input{N}.py`` snapshot.  If this kernel is a template
        # *and* we fell back to the shared ``{kernel}.py``, benchmarking
        # would silently measure the wrong problem (and produce OOB writes
        # whenever M differs).  Fail the shape loudly instead.
        try:
            src = kernel_py.read_text()
        except OSError:
            src = ""
        is_template = bool(src) and _is_template_kernel(src)
        used_per_shape = kernel_py.name == f"{kernel}_input{shape_idx}.py"
        if is_template and not used_per_shape:
            log.error(
                "shape %d of template kernel %s/%s: per-shape snapshot "
                "%s_input%d.py missing; shared %s.py bakes literals from "
                "a different shape and would corrupt the benchmark. Skipping.",
                shape_idx, operator, kernel, kernel, shape_idx, kernel,
            )
            results.append(_error_result(
                operator, kernel, shape_idx, effective_trials,
                f"template kernel missing per-shape snapshot "
                f"{kernel}_input{shape_idx}.py",
                strategy_name,
            ))
            continue

        if is_template:
            hc = _extract_hardcoded_constexprs(src)
            for k, v in hc.items():
                baked_cfg.setdefault(k, v)
        baked_cc = _to_cc_config(baked_cfg)

        log.info("shape %d: using %s (template=%s, per_shape=%s)",
                 shape_idx, kernel_py.name, is_template, used_per_shape)

        # ---- Run the strategy for this shape ----
        strategy = _make_strategy(
            strategy_name, search_space,
            n_trials=effective_trials, seed=seed,
        )

        original_stats: Optional[Dict[str, float]] = None
        original_lat: Optional[float] = None
        # All fresh per-trial measurements of the original config on this
        # shape.  do_bench has ~5% CV on V100, so the first trial's reading
        # is a biased sample; we record every fresh measurement and use
        # their median as the canonical baseline for the speedup ratio.
        original_lat_samples: List[float] = []
        trial_history: List[Dict[str, Any]] = []
        per_trial_errors = 0

        # ---- Mid-shape resume: replay trial_history from prior crashed run ----
        # If this shape has a partial row saved from a prior subprocess that
        # died mid-shape, re-observe each saved (cfg, lat) into the fresh
        # strategy to reconstruct identical internal state, then continue
        # from the next trial.  All strategies (BO=TPE, AUC-Bandit, Random)
        # are deterministic given seed + observed history, so propose() on
        # replay yields the same cfg as was saved (sanity-checked below).
        recovery_attempted = False
        resume_row = prior_rows_by_id.get(shape_idx)
        # CUDA-recovery retry: a prior attempt's saved history is "full"
        # (len == effective_trials) but every trial failed because trial 0
        # poisoned CUDA and the rest were cascaded skips.  Reset the
        # trial_history but keep the saved entries available for cfg_cache
        # seeding below — that way the new attempt's strategy proposes
        # the poison cfg again, hits the cache, short-circuits to FAIL
        # (no GPU launch, CUDA stays alive), and trials 1+ run normally.
        if resume_row is not None:
            saved_hist_full = (resume_row.get("bo_trial_history")
                               if strategy_name == "bo"
                               else resume_row.get("trial_history")) or []
            # Recovery fires for any shape whose saved history is all
            # FAIL and contains a CUDA-related error (real root or
            # downstream cascade).  A fresh subprocess respawn supplies
            # a clean CUDA context; the cfg_cache-seeding step then
            # short-circuits any specific poison cfgs we observed.
            if (saved_hist_full and len(saved_hist_full) >= effective_trials
                    and all(h.get("latency_ms") is None for h in saved_hist_full)
                    and any(
                        ("cuda error" in (h.get("error") or "").lower()
                         or "illegal memory access" in (h.get("error") or "").lower()
                         or "an illegal instruction" in (h.get("error") or "").lower()
                         or "context poisoned" in (h.get("error") or "").lower())
                        for h in saved_hist_full
                    )):
                log.info("shape %d: CUDA-recovery retry — resetting trial_history, "
                         "seeding cfg_cache from %d saved entries",
                         shape_idx, len(saved_hist_full))
                # Mutate resume_row so the cfg_cache seeding below picks
                # the saved entries up via the normal fallback path.
                # Then clear bo_trial_history so the resume-replay branch
                # starts fresh (it requires len < effective_trials).
                if strategy_name == "bo":
                    resume_row["_recovery_saved_hist"] = list(saved_hist_full)
                    resume_row["bo_trial_history"] = []
                else:
                    resume_row["_recovery_saved_hist"] = list(saved_hist_full)
                    resume_row["trial_history"] = []
                recovery_attempted = True
        if resume_row is not None:
            saved_hist = (resume_row.get("bo_trial_history")
                          if strategy_name == "bo"
                          else resume_row.get("trial_history")) or []
            if saved_hist and len(saved_hist) < effective_trials:
                for entry in saved_hist:
                    try:
                        replayed_cfg = strategy.propose()
                        strategy.observe(entry.get("latency_ms"))
                    except Exception as e:
                        log.warning("shape %d: replay aborted after %d/%d "
                                    "trials: %s; restarting shape fresh",
                                    shape_idx, len(trial_history), len(saved_hist), e)
                        strategy = _make_strategy(
                            strategy_name, search_space,
                            n_trials=effective_trials, seed=seed,
                        )
                        trial_history = []
                        per_trial_errors = 0
                        original_stats = None
                        original_lat = None
                        original_lat_samples = []
                        break
                    if entry.get("latency_ms") is None:
                        per_trial_errors += 1
                    # Rebuild the running median from the saved per-trial
                    # original_latency_ms values, but only count fresh
                    # (non-cached) measurements — cached entries replayed
                    # the same value and would skew the median.
                    if (not entry.get("cached")
                            and entry.get("original_latency_ms") is not None):
                        original_lat_samples.append(
                            float(entry["original_latency_ms"])
                        )
                    trial_history.append(entry)
                if len(trial_history) == len(saved_hist):
                    original_stats = resume_row.get("original_stats")
                    original_lat = (
                        _median(original_lat_samples)
                        or resume_row.get("original_config_latency_ms")
                    )
                    log.info("shape %d: resumed from trial %d (budget=%d)",
                             shape_idx, len(trial_history), effective_trials)

        # NOTE: when cuda_dead is set from an earlier shape, we *don't*
        # wholesale-skip and we *don't* break early.  The per-trial
        # cuda_dead branch in the trial loop below records a cascade
        # error for each trial, producing a full trial_history that
        # marks this shape as cuda-cascade-recoverable.  The next
        # subprocess respawn (with a fresh CUDA context) sees the
        # recoverable history and gives the shape a clean retry.

        shape_start_ts = _time.time()
        shape_timed_out = False
        start_trial_id = len(trial_history)

        # Per-shape config cache.  Maps a hashable cfg key to the prior
        # benchmark result so duplicate proposals (TPE exploitation,
        # AUC-Bandit greedy mutation) skip the GPU round-trip.  Caches
        # both successes and intrinsic failures (OOM, compile error,
        # illegal-memory-access) so a config that crashed on a prior
        # subprocess attempt of THIS shape short-circuits to a cached
        # FAIL on the next attempt — strategy still observes the FAIL
        # (so its surrogate updates), budget slot is still consumed,
        # but the GPU isn't relaunched, preserving CUDA context for
        # downstream trials.  Scope is per-shape only: failures don't
        # leak across shapes (matched-budget premise — Full's K=9 LLM
        # proposals don't share knowledge across shapes either).
        cfg_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        # When this is a CUDA-recovery retry, seed cfg_cache from the
        # cleared-but-saved history so the poison config short-circuits.
        if recovery_attempted and resume_row is not None:
            for h in resume_row.get("_recovery_saved_hist", []):
                prior_cfg = h.get("config")
                if not prior_cfg:
                    continue
                cfg_key = tuple(sorted(prior_cfg.items()))
                err = (h.get("error") or "")[:200]
                if not err.startswith("skipped: CUDA") \
                        and "context poisoned" not in err.lower():
                    cfg_cache.setdefault(
                        cfg_key,
                        {
                            "lat_ms": None,
                            "trial_orig_ms": None,
                            "err": f"prior CUDA fault: {err}" if err else "prior CUDA fault",
                        },
                    )

        # Seed from this shape's resumed trial history (across subprocess
        # respawns of the same shape).
        for h in trial_history:
            prior_cfg = h.get("config")
            if not prior_cfg:
                continue
            cfg_key = tuple(sorted(prior_cfg.items()))
            prior_lat = h.get("latency_ms")
            if prior_lat is not None:
                cfg_cache.setdefault(
                    cfg_key,
                    {
                        "lat_ms": float(prior_lat),
                        "trial_orig_ms": h.get("original_latency_ms"),
                        "err": h.get("error"),
                    },
                )
            else:
                # Failed prior trial — cache so the next subprocess
                # attempt on this shape skips the GPU launch and
                # preserves CUDA context.  Skip cascaded "skipped:
                # CUDA" entries — those aren't the cfg's fault.
                err = (h.get("error") or "")[:200]
                if not err.startswith("skipped: CUDA") \
                        and "context poisoned" not in err.lower():
                    cfg_cache.setdefault(
                        cfg_key,
                        {
                            "lat_ms": None,
                            "trial_orig_ms": None,
                            "err": f"prior failure: {err}" if err else "prior failure",
                        },
                    )

        for trial_id in range(start_trial_id, effective_trials):
            cfg = strategy.propose()
            cfg_cc = _to_cc_config(cfg)
            lat_ms: Optional[float] = None
            err: Optional[str] = None
            trial_orig_ms: Optional[float] = None
            cached = False

            cfg_key = tuple(sorted(cfg.items()))

            if (per_shape_timeout_s is not None
                    and _time.time() - shape_start_ts > per_shape_timeout_s):
                shape_timed_out = True
                err = (f"shape exceeded {per_shape_timeout_s:.0f}s "
                       "wall-clock timeout")

            elif cfg_key in cfg_cache:
                # Skip the GPU round-trip: this config was benchmarked
                # earlier in the same shape.  Reuse the prior latency
                # so the strategy gets a deterministic observation and
                # the trial slot is still consumed.
                cached_entry = cfg_cache[cfg_key]
                lat_ms = cached_entry["lat_ms"]
                trial_orig_ms = cached_entry.get("trial_orig_ms")
                err = cached_entry.get("err")
                cached = True

            elif cuda_dead:
                # CUDA context is already dead from a previous trial in this
                # shape.  Every benchmark_config_pair call would raise the
                # same error, so skip the GPU round-trip and record the
                # proposed config as a failed trial that still consumes one
                # budget slot.  This matches Full's accounting, where a
                # failed candidate (validation / compile / OOM) is counted
                # against the per-shape budget rather than retried for free.
                err = "CUDA context poisoned earlier in shape; config not measured"
            else:
                try:
                    r = benchmark_config_pair(
                        original_config=baked_cc,
                        recommended_config=cfg_cc,
                        shapes_file=shape_path,
                        kernel_module=kernel_py,
                        kernel_name=kernel,
                        device=device,
                        warmup=warmup, rep=rep, seed=seed,
                        validate_outputs=validate_outputs,
                    )
                    os_r = r.get("original_stats")
                    if (os_r and "error" not in os_r
                            and os_r.get("median") is not None):
                        trial_orig_ms = float(os_r["median"])
                        original_lat_samples.append(trial_orig_ms)
                        if original_stats is None:
                            # Keep the first do_bench's per-call stats for
                            # back-compat; the cross-trial aggregate
                            # (min/max/median/...) is recorded separately
                            # via _aggregate_orig_lat below.
                            original_stats = dict(os_r)
                        # original_lat is the running median of all fresh
                        # per-trial originals, refreshed each measurement.
                        original_lat = _median(original_lat_samples)

                    rec_stats = r.get("recommended_stats")
                    if rec_stats and "error" not in rec_stats:
                        med = rec_stats.get("median")
                        if med is not None and med > 0 and med != float("inf"):
                            lat_ms = float(med)
                    elif rec_stats:
                        err = str(rec_stats.get("error"))[:200]
                        if _cuda_poisoned(err):
                            cuda_dead = True

                    if "error" in r:
                        err = err or str(r["error"])[:200]
                        if _cuda_poisoned(err):
                            cuda_dead = True
                except Exception as exc:
                    err = str(exc)[:200]
                    if _cuda_poisoned(err):
                        cuda_dead = True

                # Cache the freshly-measured result.  Successful configs
                # cache the latency; intrinsic failures (OOM, compile
                # error, illegal-memory-access) are also cached so we
                # don't re-launch them on subprocess respawns — saving
                # GPU time and preserving CUDA context for downstream
                # trials.  Cascaded "skipped: CUDA" failures are NOT
                # cached (they're not the cfg's fault).
                if lat_ms is not None:
                    cfg_cache[cfg_key] = {
                        "lat_ms": lat_ms,
                        "trial_orig_ms": trial_orig_ms,
                        "err": err,
                    }
                else:
                    err_msg = (err or "")[:200]
                    is_cascaded = (err_msg.startswith("skipped: CUDA")
                                   or "context poisoned" in err_msg.lower())
                    if not is_cascaded:
                        cfg_cache[cfg_key] = {
                            "lat_ms": None,
                            "trial_orig_ms": trial_orig_ms,
                            "err": err_msg or "failure",
                        }

            if lat_ms is None:
                per_trial_errors += 1

            strategy.observe(lat_ms)
            trial_speedup = (
                (trial_orig_ms / lat_ms)
                if (trial_orig_ms is not None and lat_ms is not None
                    and lat_ms > 0) else None
            )
            entry = {
                "trial_id": trial_id,
                "config": {str(k): v for k, v in cfg.items()},
                "latency_ms": lat_ms,
                "original_latency_ms": trial_orig_ms,
                "speedup": trial_speedup,
                "status": "success" if lat_ms is not None else "error",
                "error": err,
                "state": "COMPLETE" if lat_ms is not None else "FAIL",
                "cached": cached,
            }
            if strategy_name == "auc_bandit":
                hist = strategy.history()
                if trial_id < len(hist):
                    entry["proposer"] = hist[trial_id].get("proposer")
                    entry["auc_scores"] = hist[trial_id].get("auc_scores")
                    entry["is_new_best"] = hist[trial_id].get("is_new_best")
            trial_history.append(entry)

            # Per-trial incremental flush: the shape's row is rewritten with
            # the updated trial_history after every trial, so a crash after
            # trial K leaves trials 0..K on disk; the next run replays them
            # and continues from K+1 instead of restarting the shape.
            if incremental_output is not None:
                partial_row: Dict[str, Any] = {
                    "operator": operator, "kernel": kernel,
                    "input_id": shape_idx, "x_val": str(shape_idx),
                    "has_autotune": has_autotune, "triton_impl_name": None,
                    "harness": "compare_configs",
                    "kernel_py_used": kernel_py.name,
                    "is_template_kernel": is_template,
                    "per_shape_snapshot": used_per_shape,
                    "n_trials": effective_trials,
                    "budget": effective_trials,
                    "budget_policy": budget_policy,
                    "space_size": space_size(search_space),
                    "reference_config": dict(baked_cfg),
                    "original_config_latency_ms": original_lat,
                    "original_stats": original_stats,
                    "original_lat_aggregate": _aggregate_orig_lat(original_lat_samples),
                    "shape_wall_s": _time.time() - shape_start_ts,
                    "n_fresh_benches": sum(
                        1 for h in trial_history if not h.get("cached")
                    ),
                    "trial_errors": per_trial_errors,
                    "status": "in_progress",
                    "error": None,
                    "verified": False,
                    "verification_error": None,
                }
                if strategy_name == "bo":
                    partial_row["bo_trial_history"] = list(trial_history)
                else:
                    partial_row["trial_history"] = list(trial_history)
                _flush_incremental(results + [partial_row])

        best_cfg, best_lat, best_id = strategy.best()
        speedup = (
            (original_lat / best_lat)
            if (original_lat is not None and best_lat is not None and best_lat > 0)
            else None
        )

        if best_cfg is None:
            status = "error"
            error_msg = f"All {effective_trials} trials failed"
        else:
            status = "success"
            error_msg = None

        # Snapshot the best-so-far at trial 9 so the paper's "matched @ 9"
        # column can be produced directly from the envelope without rerunning
        # at budget=9.  When effective_trials < 9 the snapshot covers all
        # trials (i.e. == the full-budget best).
        def _cfg_strkeys(c):
            return {str(k): v for k, v in c.items()} if c else None

        snapshot_k = min(9, effective_trials)
        best9_id = None
        best9_lat: Optional[float] = None
        best9_cfg: Optional[Dict[str, Any]] = None
        for entry in trial_history[:snapshot_k]:
            lm = entry.get("latency_ms")
            if lm is None:
                continue
            if best9_lat is None or lm < best9_lat:
                best9_lat = float(lm)
                best9_id = entry.get("trial_id")
                best9_cfg = dict(entry.get("config", {}))
        speedup9 = (
            (original_lat / best9_lat)
            if (original_lat is not None and best9_lat is not None and best9_lat > 0)
            else None
        )
        snapshot_at_9 = {
            "k": snapshot_k,
            "best_latency_ms": best9_lat,
            "best_trial_id": best9_id,
            "best_config": _cfg_strkeys(best9_cfg) if best9_cfg else None,
            "speedup": speedup9,
            "trial_errors": sum(
                1 for e in trial_history[:snapshot_k]
                if e.get("status") != "success"
            ),
        }

        row: Dict[str, Any] = {
            "operator": operator,
            "kernel": kernel,
            "input_id": shape_idx,
            "x_val": str(shape_idx),
            "has_autotune": has_autotune,
            "triton_impl_name": None,
            "harness": "compare_configs",
            "kernel_py_used": kernel_py.name,
            "is_template_kernel": is_template,
            "per_shape_snapshot": used_per_shape,
            "n_trials": effective_trials,
            "budget": effective_trials,
            "budget_policy": budget_policy,
            "space_size": space_size(search_space),
            "reference_config": dict(baked_cfg),
            "original_config_latency_ms": original_lat,
            "original_stats": original_stats,
            "original_lat_aggregate": _aggregate_orig_lat(original_lat_samples),
            "shape_wall_s": _time.time() - shape_start_ts,
            "n_fresh_benches": sum(
                1 for h in trial_history if not h.get("cached")
            ),
            "recovery_attempted": recovery_attempted,
            "best_latency_ms": best_lat,
            "latency_ms": best_lat,
            "speedup": speedup,
            "snapshot_at_9": snapshot_at_9,
            "trial_errors": per_trial_errors,
            "status": status,
            "error": error_msg,
            "verified": status == "success",
            "verification_error": None,
        }

        # Strategy-specific fields so each runner's results.py CSV
        # schema finds what it expects.
        if strategy_name == "bo":
            row["bo_config"] = _cfg_strkeys(best_cfg)
            row["bo_best_trial_id"] = best_id
            row["bo_trial_history"] = trial_history
            row["bo_trial_errors"] = per_trial_errors
            row["n_startup_trials"] = getattr(strategy, "n_startup_trials", None)
        else:
            row["best_config"] = _cfg_strkeys(best_cfg)
            row["best_trial_id"] = best_id
            row["trial_history"] = trial_history

        if strategy_name == "auc_bandit":
            row["window"] = getattr(strategy, "window", None)
            row["de_pop_size"] = getattr(strategy, "de_pop_size", None)
            row["technique_trials"] = strategy.technique_trials()
            if best_id is not None and 0 <= best_id < len(trial_history):
                row["best_proposer"] = trial_history[best_id].get("proposer")
            else:
                row["best_proposer"] = None

        results.append(row)
        _flush_incremental(results)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def _error_result(
    operator: str, kernel: Optional[str], input_id: int,
    n_trials: int, error: str, strategy_name: str,
) -> Dict[str, Any]:
    row = {
        "operator": operator,
        "kernel": kernel,
        "input_id": input_id,
        "x_val": str(input_id),
        "has_autotune": False,
        "triton_impl_name": None,
        "harness": "compare_configs",
        "n_trials": n_trials,
        "budget": n_trials,
        "reference_config": None,
        "original_config_latency_ms": None,
        "original_stats": None,
        "best_latency_ms": None,
        "latency_ms": None,
        "speedup": None,
        "trial_errors": n_trials,
        "status": "error",
        "error": error,
        "verified": False,
        "verification_error": error,
    }
    if strategy_name == "bo":
        row["bo_config"] = None
        row["bo_best_trial_id"] = None
        row["bo_trial_history"] = []
        row["bo_trial_errors"] = n_trials
    else:
        row["best_config"] = None
        row["best_trial_id"] = None
        row["trial_history"] = []
    if strategy_name == "auc_bandit":
        row["best_proposer"] = None
        row["technique_trials"] = {}
    return row
