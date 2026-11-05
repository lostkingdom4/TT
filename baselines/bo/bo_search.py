"""Optuna TPE wrapper used by the BO baseline.

Pure logic — no Triton / PyTorch imports.  The runner loop is::

    search = BOSearch(space, n_trials=9, seed=42)
    for _ in range(search.n_trials):
        cfg = search.propose()
        latency = measure(cfg)
        search.observe(latency)
    best_cfg, best_lat, best_id = search.best()
    history = search.history()

If a trial fails (``latency_ms`` is ``None`` or NaN) it's reported to
Optuna as a ``FAIL`` trial so TPE can learn which regions to avoid.

Each knob is sampled as an ordered integer index into its value list
(``suggest_int(0, len-1)``) rather than ``suggest_categorical``.  This
matches Bergstra et al. 2011's TPE formulation, which uses Parzen-window
KDEs over ordinal/integer dimensions: an observation at ``XBLOCK=128``
generalises to neighboring values like 64 and 256, which the categorical
estimator (uniform prior + frequency count) cannot do.
"""

from typing import Any, Dict, List, Optional, Tuple

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState


class BOSearch:
    """TPE-driven ordinal search over a discrete kernel config space.

    ``n_startup_trials`` defaults to 10, matching Optuna's
    ``TPESampler`` default.  Hyperopt / Bergstra 2011 used 20; we
    use Optuna's value because it leaves more budget for TPE-driven
    trials at our K range (9–150).  At ``n_trials < 10`` every trial
    is a startup trial, so the sampler degenerates to uniform random
    — that matches the matched-budget regime where TPE has no
    information advantage anyway.
    """

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        n_trials: int = 9,
        seed: int = 42,
        n_startup_trials: Optional[int] = None,
    ):
        if not search_space:
            raise ValueError("search_space must be non-empty")
        self.search_space = dict(search_space)
        self.n_trials = int(n_trials)
        if n_startup_trials is None:
            n_startup_trials = 10  # Optuna TPESampler default
        self.n_startup_trials = int(n_startup_trials)

        # Internal name for the integer-index suggestion of each knob.
        self._idx_param = {p: f"{p}__idx" for p in self.search_space}

        # Silence the per-trial info logs — the runner prints its own summary.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._sampler = TPESampler(
            seed=int(seed), n_startup_trials=self.n_startup_trials,
        )
        self._study = optuna.create_study(
            direction="minimize", sampler=self._sampler,
        )
        self._pending: Optional[optuna.trial.Trial] = None
        self._trials_run = 0

    # ------------------------------------------------------------------

    def _decode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map ``{param__idx: int}`` back to ``{param: value}``."""
        out: Dict[str, Any] = {}
        for param, values in self.search_space.items():
            idx_key = self._idx_param[param]
            if idx_key in params:
                idx = int(params[idx_key])
                if 0 <= idx < len(values):
                    out[param] = values[idx]
        return out

    def propose(self) -> Dict[str, Any]:
        """Pull the next config from Optuna.  Call once per iteration."""
        trial = self._study.ask()
        cfg = {}
        for param, values in self.search_space.items():
            idx = trial.suggest_int(self._idx_param[param], 0, len(values) - 1)
            cfg[param] = values[idx]
        self._pending = trial
        return cfg

    def observe(self, latency_ms: Optional[float]) -> None:
        """Report the trial's latency back to TPE."""
        if self._pending is None:
            raise RuntimeError("observe() called before propose()")
        trial = self._pending
        self._pending = None
        self._trials_run += 1

        if (latency_ms is None
                or latency_ms != latency_ms          # NaN
                or latency_ms <= 0
                or latency_ms == float("inf")):
            self._study.tell(trial, state=TrialState.FAIL)
        else:
            self._study.tell(trial, float(latency_ms))

    # ------------------------------------------------------------------

    def best(self) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[int]]:
        """Return ``(config, latency_ms, trial_id)`` for the best trial.

        Returns ``(None, None, None)`` if every trial failed.
        """
        completed = [t for t in self._study.trials
                     if t.state == TrialState.COMPLETE]
        if not completed:
            return None, None, None
        best = min(completed, key=lambda t: t.value)
        return self._decode(best.params), float(best.value), int(best.number)

    def history(self) -> List[Dict[str, Any]]:
        """Full per-trial history for the result JSON."""
        return [
            {
                "trial_id": int(t.number),
                "config": self._decode(t.params),
                "latency_ms": (float(t.value) if t.value is not None else None),
                "state": t.state.name,
            }
            for t in self._study.trials
        ]

    @property
    def trials_run(self) -> int:
        return self._trials_run
