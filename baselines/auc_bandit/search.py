"""AUC Bandit meta-controller.

From-scratch reimplementation of Ansel et al. 2014 section 3.  Exposes
the same ``propose() / observe() / best() / history()`` interface as
``baselines.bo.bo_search.BOSearch`` so the per-shape runner can be a
near-copy of the BO runner.

Per-step algorithm::

    for tech in sub_techniques:
        score[tech] = auc(tech_history) + sqrt(2 * ln(total) / trials[tech])
    picked = argmax_tech(score)
    cfg    = sub_techniques[picked].propose(rng)
    # after measurement:
    is_new_best = (latency < global_best_latency)
    for tech in sub_techniques:
        tech.observe(cfg, latency, is_new_best)
    tech_history[picked].append(1 if is_new_best else 0)     # window <= W

The per-technique history is its OWN sequence of 0/1 improvement
events (OpenTuner-style), not a cross-technique timeline.  Older
entries weigh less than recent ones via the
``sum(i * h[i]) / (n*(n-1)/2)`` recency-weighted AUC formula.
"""

import collections
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from baselines.auc_bandit.sub_techniques import (
    DifferentialEvolution,
    GreedyMutation,
    UniformRandom,
)
from baselines.random_search.random_search import enumerate_space


class AUCBanditSearch:
    """AUC-Bandit ensemble: DE + greedy mutation + uniform random."""

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        n_trials: int = 9,
        seed: int = 42,
        window: Optional[int] = None,
        de_pop_size: Optional[int] = None,
        de_F: float = 0.5,
        de_CR: float = 0.9,
    ):
        if not search_space:
            raise ValueError("search_space must be non-empty")
        self.search_space = dict(search_space)
        self.n_trials = int(n_trials)
        self._rng = random.Random(int(seed))

        self._all = enumerate_space(self.search_space)
        if not self._all:
            raise RuntimeError("Empty enumerated space")

        self.window = int(window) if window else min(self.n_trials, 20)
        # DE population scales with budget; floor 3 so DE/rand/1 works.
        self.de_pop_size = int(de_pop_size) if de_pop_size else max(
            3, min(self.n_trials // 2, 10),
        )

        self._de = DifferentialEvolution(
            search_space=self.search_space,
            all_configs=self._all,
            pop_size=self.de_pop_size,
            F=de_F, CR=de_CR,
            rng=random.Random(int(seed) + 1),
        )
        self._gm = GreedyMutation(
            search_space=self.search_space,
            all_configs=self._all,
        )
        self._ur = UniformRandom(all_configs=self._all)

        self._techs = {
            self._de.name: self._de,
            self._gm.name: self._gm,
            self._ur.name: self._ur,
        }

        # Per-technique window of 0/1 "was this an improvement" events.
        self._tech_hist: Dict[str, collections.deque] = {
            name: collections.deque(maxlen=self.window)
            for name in self._techs
        }
        self._tech_trials: Dict[str, int] = {
            name: 0 for name in self._techs
        }

        self._history: List[Dict[str, Any]] = []
        self._best_lat: Optional[float] = None
        self._best_cfg: Optional[Dict[str, Any]] = None
        self._best_trial_id: Optional[int] = None
        self._pending: Optional[Tuple[str, Dict[str, Any]]] = None
        self._trials_run = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _auc(self, tech_name: str) -> float:
        h = list(self._tech_hist[tech_name])
        n = len(h)
        if n == 0:
            return 1.0  # favor unused techniques (OpenTuner convention)
        if n == 1:
            return float(h[0])
        normalizer = (n - 1) * n / 2.0
        return sum(i * h[i] for i in range(n)) / normalizer

    def _ucb_bonus(self, tech_name: str) -> float:
        trials = self._tech_trials[tech_name]
        if trials == 0:
            return float("inf")
        total = max(sum(self._tech_trials.values()), 1)
        return math.sqrt(2.0 * math.log(total) / trials)

    def _score(self, tech_name: str) -> float:
        return self._auc(tech_name) + self._ucb_bonus(tech_name)

    def _pick_technique(self) -> str:
        scores = {name: self._score(name) for name in self._techs}
        # Deterministic tiebreak via RNG-seeded random choice among max.
        best_score = max(scores.values())
        winners = [n for n, s in scores.items() if s == best_score]
        return self._rng.choice(winners)

    # ------------------------------------------------------------------
    # Core loop interface (mirrors BOSearch)
    # ------------------------------------------------------------------

    def propose(self) -> Dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("propose() called twice without observe()")
        name = self._pick_technique()
        cfg = self._techs[name].propose(self._rng)
        self._pending = (name, cfg)
        return dict(cfg)

    def observe(self, latency_ms: Optional[float]) -> None:
        if self._pending is None:
            raise RuntimeError("observe() called before propose()")
        name, cfg = self._pending
        self._pending = None

        valid = (latency_ms is not None
                 and latency_ms == latency_ms  # not NaN
                 and latency_ms > 0
                 and latency_ms != float("inf"))
        lat = float(latency_ms) if valid else None
        state = "COMPLETE" if valid else "FAIL"

        is_new_best = False
        if valid and (self._best_lat is None or lat < self._best_lat):
            is_new_best = True
            self._best_lat = lat
            self._best_cfg = dict(cfg)
            self._best_trial_id = self._trials_run

        # Broadcast observation to *all* sub-techniques so they can
        # update population / best-known state.
        for tech in self._techs.values():
            tech.observe(cfg, lat, is_new_best)

        # Bandit bookkeeping: only the picked technique logs a 0/1
        # event; unchosen techniques' windows are untouched.
        self._tech_hist[name].append(1 if is_new_best else 0)
        self._tech_trials[name] += 1

        self._history.append({
            "trial_id": self._trials_run,
            "proposer": name,
            "config": dict(cfg),
            "latency_ms": lat,
            "state": state,
            "is_new_best": is_new_best,
            "auc_scores": {
                n: round(self._auc(n), 4) for n in self._techs
            },
        })
        self._trials_run += 1

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def best(self) -> Tuple[Optional[Dict[str, Any]], Optional[float], Optional[int]]:
        if self._best_cfg is None:
            return None, None, None
        return dict(self._best_cfg), float(self._best_lat), int(self._best_trial_id)

    def history(self) -> List[Dict[str, Any]]:
        return [dict(h) for h in self._history]

    def technique_trials(self) -> Dict[str, int]:
        return dict(self._tech_trials)

    @property
    def trials_run(self) -> int:
        return self._trials_run
