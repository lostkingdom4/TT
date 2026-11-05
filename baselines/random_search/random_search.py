"""Uniform random search over a discrete Triton-config space.

Mirrors the ``propose()/observe()/best()/history()`` interface of
``BOSearch`` so the runner loop is identical.  Samples without
replacement by default -- if the budget exceeds the enumerated space we
wrap around and allow repeats (logged in the history).
"""

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple


def enumerate_space(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Return all configurations in the Cartesian product of ``search_space``."""
    if not search_space:
        return []
    keys = list(search_space.keys())
    lists = [list(search_space[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*lists)]


class RandomSearch:
    """Uniform random sampling without replacement from the valid space."""

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        n_trials: int = 9,
        seed: int = 42,
    ):
        if not search_space:
            raise ValueError("search_space must be non-empty")
        self.search_space = dict(search_space)
        self.n_trials = int(n_trials)
        self._rng = random.Random(int(seed))

        # Enumerate the full space once so we can sample without replacement.
        self._all_configs: List[Dict[str, Any]] = enumerate_space(self.search_space)
        self._rng.shuffle(self._all_configs)
        self._cursor = 0

        self._history: List[Dict[str, Any]] = []
        self._pending_cfg: Optional[Dict[str, Any]] = None
        self._trials_run = 0

    # ------------------------------------------------------------------

    def propose(self) -> Dict[str, Any]:
        """Pull the next config. Wraps around if budget > |space|."""
        if not self._all_configs:
            raise RuntimeError("Empty search space")
        idx = self._cursor % len(self._all_configs)
        # When wrapping, reshuffle so repeats aren't in the same order.
        if self._cursor > 0 and idx == 0:
            self._rng.shuffle(self._all_configs)
        cfg = dict(self._all_configs[idx])
        self._cursor += 1
        self._pending_cfg = cfg
        return cfg

    def observe(self, latency_ms: Optional[float]) -> None:
        if self._pending_cfg is None:
            raise RuntimeError("observe() called before propose()")
        cfg = self._pending_cfg
        self._pending_cfg = None

        if (latency_ms is None
                or latency_ms != latency_ms          # NaN
                or latency_ms <= 0
                or latency_ms == float("inf")):
            state = "FAIL"
            lat = None
        else:
            state = "COMPLETE"
            lat = float(latency_ms)

        self._history.append({
            "trial_id": self._trials_run,
            "config": cfg,
            "latency_ms": lat,
            "state": state,
        })
        self._trials_run += 1

    # ------------------------------------------------------------------

    def best(self) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[int]]:
        completed = [h for h in self._history if h["state"] == "COMPLETE"]
        if not completed:
            return None, None, None
        winner = min(completed, key=lambda h: h["latency_ms"])
        return dict(winner["config"]), float(winner["latency_ms"]), int(winner["trial_id"])

    def history(self) -> List[Dict[str, Any]]:
        return [dict(h) for h in self._history]

    @property
    def trials_run(self) -> int:
        return self._trials_run
