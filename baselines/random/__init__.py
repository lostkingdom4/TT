"""Random-config baseline.

Draws each kernel launch's config uniformly at random from the shared
per-kernel search space in ``baselines.common.search_space``.

CLI::

    python -m baselines.random.run_all \\
        --ir-dir extract/ir_output/v100 --seed 42 --export-csv
"""
