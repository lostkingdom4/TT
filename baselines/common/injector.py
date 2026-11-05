"""Config injection via Triton monkeypatching.

Two injectors share the same patch surface but source their configs
differently:

* ``_RandomConfigInjector`` — draws a random config from the search
  space every time a kernel launches.  Used by ``baselines.random``.
* ``_FixedConfigInjector`` — installs one caller-supplied config and
  uses it for every launch until ``__exit__``.  Used by
  ``baselines.bo`` to evaluate a single Optuna trial.

Both support the two kernel types found in TritonBench:

* **Autotuned kernels** — patches ``Autotuner.run`` so the autotuner's
  config list is replaced with a single ``triton.Config``; the
  autotuner has no choice but to pick that one.
* **Non-autotuned JIT kernels** — patches ``JITFunction.__getitem__``
  so each kernel launch has ``num_warps`` / ``num_stages`` (and any
  block-size kwargs the kernel already accepts) overridden.

Chosen configs are recorded in ``self.injected_configs`` for
traceability.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from .search_space import (
    CONFIG_LEVEL_PARAMS,
    generate_random_config,
    get_search_space,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base: shared patch install / restore
# ---------------------------------------------------------------------------

class _BaseConfigInjector:
    """Shared autotuner / JIT patch plumbing."""

    def __init__(self, operator: str, has_autotune: bool, kernel: Optional[str] = None):
        self.operator = operator
        self.kernel = kernel
        self.has_autotune = has_autotune
        self._original_autotuner_run = None
        self._original_jit_getitem = None
        self.injected_configs: List[Dict[str, Any]] = []

    # -- context manager --------------------------------------------------

    def __enter__(self):
        if self.has_autotune:
            self._install_autotuner_patch()
        else:
            self._install_jit_patch()
        return self

    def __exit__(self, *exc):
        if self._original_autotuner_run is not None:
            from triton.runtime.autotuner import Autotuner
            Autotuner.run = self._original_autotuner_run
        if self._original_jit_getitem is not None:
            from triton.runtime.jit import JITFunction
            JITFunction.__getitem__ = self._original_jit_getitem
        return False

    # -- subclass hooks ---------------------------------------------------

    def _next_config(
        self,
        expected_kwargs: Optional[set] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the config to install on the next kernel launch.

        ``expected_kwargs``/``template_kwargs`` are supplied in autotuner
        mode so the subclass can adapt to whichever param names the
        autotuner's template configs carry.
        """
        raise NotImplementedError

    # -- autotuner patch --------------------------------------------------

    def _install_autotuner_patch(self):
        from triton.runtime.autotuner import Autotuner
        import triton

        self._original_autotuner_run = Autotuner.run
        injector = self

        def _patched_run(autotuner_self, *args, **kwargs):
            if autotuner_self.configs:
                template = autotuner_self.configs[0]
                expected_kwargs = set(template.kwargs.keys())
            else:
                template = None
                expected_kwargs = set()

            full = injector._next_config(
                expected_kwargs=expected_kwargs,
                template_kwargs=template.kwargs if template else None,
            )

            block_kwargs = {
                k: v for k, v in full.items() if k not in CONFIG_LEVEL_PARAMS
            }
            # Fall back to template values for params the injector
            # didn't specify (keeps kernel-specific constexprs intact).
            if template is not None:
                for k, v in template.kwargs.items():
                    block_kwargs.setdefault(k, v)

            new_config = triton.Config(
                block_kwargs,
                num_warps=full.get(
                    "num_warps", getattr(template, "num_warps", 4)
                ),
                num_stages=full.get(
                    "num_stages", getattr(template, "num_stages", 2)
                ),
                num_ctas=getattr(template, "num_ctas", 1) if template else 1,
            )

            autotuner_self.configs = [new_config]
            autotuner_self.cache = {}

            injector.injected_configs.append({
                **block_kwargs,
                "num_warps": new_config.num_warps,
                "num_stages": new_config.num_stages,
            })
            log.debug("Injected autotuner config: %s", injector.injected_configs[-1])

            return injector._original_autotuner_run(
                autotuner_self, *args, **kwargs
            )

        Autotuner.run = _patched_run

    # -- JIT patch --------------------------------------------------------

    def _install_jit_patch(self):
        from triton.runtime.jit import JITFunction

        self._original_jit_getitem = JITFunction.__getitem__
        injector = self

        def _patched_getitem(jit_self, grid):
            launcher = injector._original_jit_getitem(jit_self, grid)
            rc = injector._next_config()

            def wrapped_launcher(*args, **kwargs):
                if "num_warps" in rc:
                    kwargs["num_warps"] = rc["num_warps"]
                if "num_stages" in rc:
                    kwargs["num_stages"] = rc["num_stages"]

                # Inject block-size params only if the kernel already
                # accepts them (prevents breaking unrelated kernels).
                for key, value in rc.items():
                    if key not in CONFIG_LEVEL_PARAMS and key in kwargs:
                        kwargs[key] = value

                injector.injected_configs.append(rc)
                log.debug("Injected JIT config: %s", rc)
                return launcher(*args, **kwargs)

            return wrapped_launcher

        JITFunction.__getitem__ = _patched_getitem

    # -- helpers ----------------------------------------------------------

    def get_config(self, index: int) -> Optional[Dict[str, Any]]:
        if 0 <= index < len(self.injected_configs):
            return self.injected_configs[index]
        return None


# ---------------------------------------------------------------------------
# Random injector
# ---------------------------------------------------------------------------

class _RandomConfigInjector(_BaseConfigInjector):
    """Draws a fresh random config from the search space on every launch."""

    def __init__(
        self,
        operator: str,
        seed: int,
        has_autotune: bool,
        kernel: Optional[str] = None,
    ):
        super().__init__(operator, has_autotune, kernel=kernel)
        self.seed = seed
        self.rng = random.Random(seed)
        self.search_space = get_search_space(operator, kernel)

    def _next_config(
        self,
        expected_kwargs: Optional[set] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if expected_kwargs is not None:
            # Autotuner mode: adapt to the kernel's actual param names.
            random_kwargs: Dict[str, Any] = {}
            for param in expected_kwargs:
                if param in self.search_space:
                    random_kwargs[param] = self.rng.choice(
                        self.search_space[param]
                    )
                elif template_kwargs is not None and param in template_kwargs:
                    random_kwargs[param] = template_kwargs[param]
            random_kwargs["num_warps"] = self.rng.choice(
                self.search_space.get("num_warps", [4])
            )
            random_kwargs["num_stages"] = self.rng.choice(
                self.search_space.get("num_stages", [2])
            )
            return random_kwargs
        # JIT mode: just sample the whole space.
        return generate_random_config(
            self.operator, self.rng, kernel=self.kernel
        )


# ---------------------------------------------------------------------------
# Fixed injector (used by BO to install one trial's config)
# ---------------------------------------------------------------------------

class _FixedConfigInjector(_BaseConfigInjector):
    """Installs a single caller-supplied config for every kernel launch."""

    def __init__(
        self,
        operator: str,
        config: Dict[str, Any],
        has_autotune: bool,
        kernel: Optional[str] = None,
    ):
        super().__init__(operator, has_autotune, kernel=kernel)
        self.config = dict(config)

    def _next_config(
        self,
        expected_kwargs: Optional[set] = None,
        template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if expected_kwargs is not None:
            # Autotuner mode: project the fixed config onto whichever
            # keys the autotuner's template config actually carries.
            out: Dict[str, Any] = {}
            for param in expected_kwargs:
                if param in self.config:
                    out[param] = self.config[param]
                elif template_kwargs is not None and param in template_kwargs:
                    out[param] = template_kwargs[param]
            out["num_warps"] = self.config.get(
                "num_warps",
                template_kwargs.get("num_warps", 4) if template_kwargs else 4,
            )
            out["num_stages"] = self.config.get(
                "num_stages",
                template_kwargs.get("num_stages", 2) if template_kwargs else 2,
            )
            return out
        return dict(self.config)
