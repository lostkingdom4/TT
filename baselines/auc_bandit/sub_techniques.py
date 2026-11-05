"""Sub-techniques used by the AUC Bandit ensemble.

Each sub-technique exposes ``propose(rng)`` (returns a dict config) and
receives ``observe(config, latency, is_new_best)`` to update any
internal state (DE's population, for instance).  They never see the
raw bandit scores -- the meta-controller handles selection.

Shared invariants:

* ``search_space`` is a ``{param: [values]}`` dict (values already
  sorted for meaningful "neighboring" semantics).
* ``all_configs`` is the pre-enumerated list of valid ``dict`` configs
  (one-shot work done by the meta-controller).
* Configs returned are always in ``all_configs`` -- sub-techniques
  fall back to a random valid config after bounded retries.
"""

import random
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_key(cfg: Dict[str, Any]) -> Tuple:
    """Hashable key for a config (used for set membership)."""
    return tuple(sorted(cfg.items()))


# ---------------------------------------------------------------------------
# Uniform random sampler
# ---------------------------------------------------------------------------

class UniformRandom:
    """Sample uniformly from the enumerated valid space."""

    name = "uniform_random"

    def __init__(self, all_configs: List[Dict[str, Any]]):
        self._all = all_configs

    def propose(self, rng: random.Random) -> Dict[str, Any]:
        return dict(rng.choice(self._all))

    def observe(self, config, latency, is_new_best):
        pass


# ---------------------------------------------------------------------------
# Greedy mutation on the global best
# ---------------------------------------------------------------------------

class GreedyMutation:
    """Flip 1-2 params of the current global best to neighboring values."""

    name = "greedy_mutation"

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        all_configs: List[Dict[str, Any]],
    ):
        self._keys = list(search_space.keys())
        self._values = {k: list(search_space[k]) for k in self._keys}
        self._all = all_configs
        self._all_keys = {_cfg_key(c) for c in all_configs}
        self._best_cfg: Optional[Dict[str, Any]] = None
        self._best_lat: Optional[float] = None

    def propose(self, rng: random.Random) -> Dict[str, Any]:
        if self._best_cfg is None:
            return dict(rng.choice(self._all))

        for _ in range(10):
            n_flips = rng.randint(1, min(2, len(self._keys)))
            params = rng.sample(self._keys, n_flips)
            cand = dict(self._best_cfg)
            any_change = False
            for p in params:
                values = self._values[p]
                cur_idx = values.index(self._best_cfg[p])
                neighbors = []
                if cur_idx > 0:
                    neighbors.append(cur_idx - 1)
                if cur_idx < len(values) - 1:
                    neighbors.append(cur_idx + 1)
                if not neighbors:
                    continue
                cand[p] = values[rng.choice(neighbors)]
                if cand[p] != self._best_cfg[p]:
                    any_change = True
            if any_change and _cfg_key(cand) in self._all_keys:
                return cand
        return dict(rng.choice(self._all))

    def observe(self, config, latency, is_new_best):
        if (latency is None or latency != latency or latency <= 0
                or latency == float("inf")):
            return
        if self._best_lat is None or latency < self._best_lat:
            self._best_lat = float(latency)
            self._best_cfg = dict(config)


# ---------------------------------------------------------------------------
# Differential evolution (DE/rand/1/bin) in ordered-index space
# ---------------------------------------------------------------------------

class DifferentialEvolution:
    """DE/rand/1/bin over the discrete search space.

    Each config is projected into an integer-index vector where the
    i'th coordinate is the index of the chosen value in the ordered
    value list for parameter i.  Mutation is performed in index space,
    then rounded back to nearest valid integer indices, and finally
    mapped back to a config dict.  Crossover probability ``CR`` and
    mutation factor ``F`` follow the standard defaults (0.9, 0.5).
    """

    name = "de"

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        all_configs: List[Dict[str, Any]],
        pop_size: int,
        F: float = 0.5,
        CR: float = 0.9,
        rng: Optional[random.Random] = None,
    ):
        self._keys = list(search_space.keys())
        self._values = {k: list(search_space[k]) for k in self._keys}
        self._all = all_configs
        self._all_keys = {_cfg_key(c) for c in all_configs}
        self._F = float(F)
        self._CR = float(CR)
        self._rng = rng or random.Random()

        self._pop_size = max(3, int(pop_size))
        self._pop_size = min(self._pop_size, len(all_configs))

        # Initial population: random distinct configs
        init = self._rng.sample(all_configs, self._pop_size)
        self._pop: List[Dict[str, Any]] = [dict(c) for c in init]
        self._pop_lat: List[Optional[float]] = [None] * self._pop_size
        self._last_target_idx: Optional[int] = None

    def _cfg_to_idx(self, cfg: Dict[str, Any]) -> List[int]:
        return [self._values[k].index(cfg[k]) for k in self._keys]

    def _idx_to_cfg(self, idx: List[float]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, k in enumerate(self._keys):
            vals = self._values[k]
            # Clamp + round to valid index range.
            j = int(round(idx[i]))
            if j < 0:
                j = 0
            elif j >= len(vals):
                j = len(vals) - 1
            out[k] = vals[j]
        return out

    def propose(self, rng: random.Random) -> Dict[str, Any]:
        n = self._pop_size
        # Pick target (round-robin would bias early pop; use random).
        target_i = rng.randrange(n)
        target_idx = self._cfg_to_idx(self._pop[target_i])

        # r1, r2, r3: 3 distinct members.  At pop_size=3 they are all
        # three members (including possibly the target); that's a minor
        # deviation from strict DE/rand/1 but keeps the algorithm
        # functional at the smallest budgets.
        if n >= 4:
            choices = [i for i in range(n) if i != target_i]
            r1, r2, r3 = rng.sample(choices, 3)
        else:
            r1, r2, r3 = rng.sample(range(n), 3)

        r1_idx = self._cfg_to_idx(self._pop[r1])
        r2_idx = self._cfg_to_idx(self._pop[r2])
        r3_idx = self._cfg_to_idx(self._pop[r3])

        for _ in range(10):
            mutant = [
                r1_idx[i] + self._F * (r2_idx[i] - r3_idx[i])
                for i in range(len(self._keys))
            ]
            j_rand = rng.randrange(len(self._keys))
            trial = [
                mutant[i] if (rng.random() < self._CR or i == j_rand)
                else target_idx[i]
                for i in range(len(self._keys))
            ]
            cand = self._idx_to_cfg(trial)
            if _cfg_key(cand) in self._all_keys:
                self._last_target_idx = target_i
                return cand

        # Fallback: random-valid
        self._last_target_idx = target_i
        return dict(rng.choice(self._all))

    def observe(self, config, latency, is_new_best):
        if (latency is None or latency != latency or latency <= 0
                or latency == float("inf")):
            return
        lat = float(latency)

        # DE replacement: if this trial beats the (last-proposed)
        # target, replace it in the population.  Because all observers
        # see every trial (even those originated by other techniques),
        # we only do replacement when this technique's DE engine was
        # the last proposer.
        tgt = self._last_target_idx
        if tgt is not None and (
            self._pop_lat[tgt] is None or lat < self._pop_lat[tgt]
        ):
            self._pop[tgt] = dict(config)
            self._pop_lat[tgt] = lat
            self._last_target_idx = None
            return

        # If *another* technique found a new global best, inject it into
        # the population slot with the worst latency.  This keeps the DE
        # pool current with the global best without overwriting measured
        # members for no reason.
        if is_new_best:
            worst_i = max(
                range(self._pop_size),
                key=lambda i: (
                    float("inf") if self._pop_lat[i] is None
                    else self._pop_lat[i]
                ),
            )
            if (self._pop_lat[worst_i] is None
                    or lat < self._pop_lat[worst_i]):
                self._pop[worst_i] = dict(config)
                self._pop_lat[worst_i] = lat

    @property
    def pop_size(self) -> int:
        return self._pop_size
