"""KernelBench problem loading + bookkeeping utilities.

KernelBench problems are independent Python modules under
``KernelBench/level{1,2,3}/<pid>_<slug>.py`` that each define:
    class Model(nn.Module): ...
    def get_inputs() -> list[Tensor]
    def get_init_inputs() -> list[Any]

This helper discovers them, loads them via importlib, and builds the slug we
use as the "operator" name in our metadata YAML so downstream tooling treats
each problem like a TritonBench operator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class KernelBenchProblem:
    level: int
    pid: int
    slug: str
    module_path: Path

    @property
    def operator_name(self) -> str:
        """Used as the 'operator' key in kernelbench_kernels.yaml."""
        return f"kb_L{self.level}_{self.pid}_{self.slug}"


_PID_SLUG_RE = re.compile(r"^(\d+)_(.+)$")


def discover_problems(
    kernelbench_root: Path,
    levels: Iterable[int] = (1, 2, 3),
    problem_filter: Optional[Iterable[str]] = None,
) -> List[KernelBenchProblem]:
    """Walk KernelBench/level{N}/*.py and return a sorted list of problems.

    ``problem_filter`` matches against the stem (e.g. "14_Gemm_Divide_Sum_Scaling")
    or the operator_name (e.g. "kb_L2_14_Gemm_Divide_Sum_Scaling"). Exact match
    only; pass multiple entries to include multiple problems.
    """
    wanted = set(problem_filter) if problem_filter else None
    problems: List[KernelBenchProblem] = []
    for level in levels:
        level_dir = kernelbench_root / f"level{level}"
        if not level_dir.is_dir():
            continue
        for py in sorted(level_dir.glob("*.py")):
            stem = py.stem
            m = _PID_SLUG_RE.match(stem)
            if not m:
                continue
            pid = int(m.group(1))
            slug = m.group(2)
            prob = KernelBenchProblem(level=level, pid=pid, slug=slug, module_path=py)
            if wanted is not None and stem not in wanted \
                    and prob.operator_name not in wanted:
                continue
            problems.append(prob)
    return problems


def load_problem_module(problem: KernelBenchProblem):
    """Import a problem module (Model / get_inputs / get_init_inputs)."""
    spec = importlib.util.spec_from_file_location(
        f"kernelbench_problem_{problem.level}_{problem.pid}",
        str(problem.module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {problem.module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for attr in ("Model", "get_inputs", "get_init_inputs"):
        if not hasattr(module, attr):
            raise RuntimeError(
                f"{problem.module_path.name} missing required attribute '{attr}'"
            )
    return module


class SourceHashTable:
    """Track which kernel source hashes have already been emitted so we don't
    re-run the LLM on identical Inductor-emitted bodies across KernelBench L3
    problems (e.g. the same ``triton_poi_fused_add_mul`` body recurs in every
    ResNet variant). Persisted as JSON so the table is shared across driver
    invocations.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, dict] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                self._data = {}

    @staticmethod
    def hash_source(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def record(self, src_hash: str, operator: str, kernel: str,
               ttgir_path: str) -> Optional[dict]:
        """Return the prior entry if this hash was already seen, else register."""
        if src_hash in self._data:
            return self._data[src_hash]
        self._data[src_hash] = {
            "operator": operator,
            "kernel": kernel,
            "ttgir": ttgir_path,
        }
        return None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
