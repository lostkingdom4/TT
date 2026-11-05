"""Scaled-budget uniform random search baseline.

Uses the same propose/observe/best/history interface as
``baselines.bo.bo_search.BOSearch`` so the per-shape runner can be a
near-copy of the BO runner.  For each shape:

    budget = clip(round(0.10 * |search_space|), 9, 150)
    for trial in range(budget):
        cfg = random.choice(enumerate_valid(search_space))  # no replacement
        latency = measure(cfg)
    best_cfg = argmin(observed latencies)

Results land in ``baselines/results/<device>/random_search/``.
"""
