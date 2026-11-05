"""AUC Bandit meta-technique (from-scratch reimplementation).

Reference: Ansel et al. 2014, "OpenTuner: An Extensible Framework for
Program Autotuning", section 3.  Cross-checked against the canonical
implementation at github.com/jansel/opentuner.

Sub-techniques (ensemble members):

1. **Differential Evolution** (DE/rand/1/bin) -- maintains a population
   of configs, each generation mutates a random member using
   ``r1 + F*(r2 - r3)`` in the ordered-index space and crossover with
   CR=0.9.
2. **Greedy Mutation** -- flips 1-2 params of the global-best config to
   neighboring values in the ordered list.
3. **Uniform Random** -- unbiased baseline over the enumerated valid
   space.

Meta-controller: at each step, compute
``score(t) = AUC(history_t) + sqrt(2 * ln(total_trials) / trials_t)``
where ``AUC`` is the recency-weighted win fraction within a sliding
window of ``W = min(budget, 20)`` observations (1 = this technique
produced a new global best, 0 = it didn't).  Pick the argmax technique
to propose the next config.  All sub-techniques observe all points so
the DE population + global-best pool stay in sync across proposers.
"""

from baselines.auc_bandit.search import AUCBanditSearch  # noqa: F401
