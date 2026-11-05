#!/usr/bin/env python3
"""Bootstrap script to set up the tritonbench dependency.

Usage:
    # Clone tritonbench into tb/
    python setup_tb.py

    # Clone a specific branch
    python setup_tb.py --branch my-branch

    # Symlink an existing tritonbench checkout instead of cloning
    python setup_tb.py --link /path/to/existing/tritonbench

    # Clone from a custom repo URL
    python setup_tb.py --repo https://github.com/my-fork/tritonbench.git
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TB_DIR = SCRIPT_DIR / "tb"

DEFAULT_REPO = "https://github.com/pytorch-labs/tritonbench.git"
DEFAULT_BRANCH = "main"


def link_tritonbench(target: Path) -> None:
    """Symlink an existing tritonbench checkout into tb/."""
    target = target.resolve()
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)
    if TB_DIR.exists() or TB_DIR.is_symlink():
        print(f"Removing existing tb/ -> {TB_DIR.resolve()}")
        if TB_DIR.is_symlink():
            TB_DIR.unlink()
        else:
            print("Error: tb/ exists and is not a symlink. Remove it manually.")
            sys.exit(1)
    TB_DIR.symlink_to(target)
    print(f"Symlinked tb/ -> {target}")


def clone_tritonbench(repo: str, branch: str) -> None:
    """Clone tritonbench into tb/."""
    if TB_DIR.exists():
        print("tb/ already exists. Pulling latest...")
        subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=str(TB_DIR),
            check=False,
        )
    else:
        print(f"Cloning {repo} (branch: {branch}) into tb/...")
        subprocess.run(
            ["git", "clone", "--branch", branch, repo, str(TB_DIR)],
            check=True,
        )


def init_submodules() -> None:
    """Initialize and update git submodules inside tb/.

    Tritonbench kernels (e.g. addmm) live inside submodules such as
    ``generative-recommenders``.  Without this step the submodule
    directories are empty and kernel benchmarking fails with
    FileNotFoundError.
    """
    if not TB_DIR.exists():
        print("Warning: tb/ does not exist, skipping submodule init")
        return
    print("Initializing git submodules in tb/ ...")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=str(TB_DIR),
        check=True,
    )
    print("Submodules initialized.")


def clear_triton_cache() -> None:
    """Remove the Triton JIT compilation cache.

    Triton caches compiled kernels under ``~/.triton/cache/``.  The cached
    IR embeds the absolute source path at compile time, so after moving or
    re-cloning tritonbench the stale cache produces IR with wrong ``loc()``
    paths.  Clearing it forces a fresh recompilation.
    """
    cache_dir = Path.home() / ".triton" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared Triton cache: {cache_dir}")
    else:
        print("Triton cache not found, nothing to clear.")


def install_editable() -> None:
    """Install tritonbench in editable mode."""
    print("Installing tritonbench in editable mode...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(TB_DIR)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up tritonbench dependency")
    parser.add_argument(
        "--link",
        type=Path,
        default=None,
        help="Symlink an existing tritonbench checkout instead of cloning",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Git repo URL to clone (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Branch to clone (default: {DEFAULT_BRANCH})",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip pip install -e (just clone/link)",
    )
    parser.add_argument(
        "--no-submodules",
        action="store_true",
        help="Skip git submodule init/update",
    )
    parser.add_argument(
        "--clear-triton-cache",
        action="store_true",
        help="Clear ~/.triton/cache/ to force IR recompilation with correct paths",
    )
    args = parser.parse_args()

    if args.link is not None:
        link_tritonbench(args.link)
    else:
        clone_tritonbench(args.repo, args.branch)

    if not args.no_submodules:
        init_submodules()

    if not args.no_install:
        install_editable()

    if args.clear_triton_cache:
        clear_triton_cache()

    print()
    print(f"Setup complete. tritonbench is available at: {TB_DIR.resolve()}")


if __name__ == "__main__":
    main()
