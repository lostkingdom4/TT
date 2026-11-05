"""Shared logging setup for baseline runners.

Every long-running baseline sweep should write a durable log file so
progress can be inspected while the run is in flight (e.g. ``tail -f``
the file).  Stdout alone is buffered by Python when the process is
running under nohup / `>` redirection, so `--log-level INFO` lines may
not reach disk for minutes.  Using a ``FileHandler`` with
``logging.INFO`` forces a flush on every record.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_run_logging(
    level: str,
    log_file: Optional[Path] = None,
    stream: bool = True,
) -> Optional[Path]:
    """Configure the root logger for a baseline run.

    Installs:
      - a ``StreamHandler`` on stderr (unless ``stream=False``)
      - a ``FileHandler`` at ``log_file`` if provided

    Both handlers share the same format and level.  File handler is
    opened in append mode so ``--resume`` runs keep history across
    re-launches.  Returns the resolved log file path (or ``None``).
    """
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    # Clear any basicConfig-installed handlers from earlier callers so
    # we don't end up double-logging.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(lvl)

    if stream:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setLevel(lvl)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    resolved: Optional[Path] = None
    if log_file is not None:
        resolved = Path(log_file).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(resolved, mode="a")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Defeat stdout/stderr buffering so ``print()`` calls from the
    # runner land on disk immediately when output is redirected.
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except AttributeError:
        # Python <3.7: fall back to env var
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

    return resolved


def default_log_path(
    output_dir: Path, device: str, baseline_name: str,
) -> Path:
    """Durable log path: <output_dir>/<device>/<baseline>/run.log."""
    return Path(output_dir) / device / baseline_name / "run.log"
