from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from macroscopic.solvers.base import BaseMPCSolver

logger = logging.getLogger(__name__)

# Sentinel so swap_solver raises clearly when the pool hasn't been built.
_NOT_BUILT = object()


class SolverPoolMixin:
    """
    Pre-builds one solver per discrete alpha value and hot-swaps `self.solver`
    under `self._solver_lock` when the agent selects a new action.

    Must be mixed in with AsyncMPCController (which owns _solver_lock and .solver).
    Call build_solver_pool() before the SUMO loop starts.
    """

    def build_solver_pool(
        self,
        alpha_values: list[float],
        solver_factory: Callable[[float], "BaseMPCSolver"],
    ) -> None:
        """
        Build a solver for every alpha value in alpha_values.
        solver_factory(alpha) must return a fully setup() BaseMPCSolver.
        """
        logger.info(
            "[SolverPool] Building %d solver(s) for alpha values %s ...",
            len(alpha_values),
            alpha_values,
        )
        self._solver_pool: dict[float, "BaseMPCSolver"] = {}
        for alpha in alpha_values:
            logger.info("[SolverPool]   α=%.2f ...", alpha)
            self._solver_pool[alpha] = solver_factory(alpha)
        # Point the live solver at the median alpha to begin with
        mid = alpha_values[len(alpha_values) // 2]
        self.solver = self._solver_pool[mid]
        logger.info("[SolverPool] Pool ready. Initial solver: α=%.2f", mid)

    def swap_solver(self, alpha: float) -> None:
        """
        Replace the active solver with the pre-built variant for alpha.
        Must be called while holding self._solver_lock.
        """
        if not hasattr(self, "_solver_pool"):
            raise RuntimeError(
                "SolverPoolMixin.build_solver_pool() must be called before swap_solver()."
            )
        if alpha not in self._solver_pool:
            raise KeyError(
                f"No pre-built solver for α={alpha}. "
                f"Available: {sorted(self._solver_pool)}"
            )
        self.solver = self._solver_pool[alpha]
        logger.debug("[SolverPool] Swapped to α=%.2f", alpha)


class RebuildSolverMixin:
    """
    Alternative to SolverPoolMixin: rebuilds the solver from scratch on every swap.
    Slower but stateless — useful when testing without the memory cost of N solvers.
    Provide rebuild_strategy='rebuild' in the rl config to use this.
    """

    def build_solver_pool(
        self,
        alpha_values: list[float],
        solver_factory: Callable[[float], "BaseMPCSolver"],
    ) -> None:
        self._solver_factory = solver_factory
        self._alpha_values = alpha_values
        # Build initial solver at mid alpha
        mid = alpha_values[len(alpha_values) // 2]
        logger.info("[RebuildSolver] Building initial solver α=%.2f ...", mid)
        self.solver = solver_factory(mid)
        logger.info("[RebuildSolver] Ready (will rebuild on each swap).")

    def swap_solver(self, alpha: float) -> None:
        """Rebuild from scratch for the requested alpha — slow but always fresh."""
        logger.info("[RebuildSolver] Rebuilding solver for α=%.2f ...", alpha)
        self.solver = self._solver_factory(alpha)
        logger.debug("[RebuildSolver] Swapped to α=%.2f", alpha)


def solver_swap_mixin_from_config(rebuild_strategy: str) -> type:
    """Return the appropriate mixin class based on config string."""
    if rebuild_strategy == "rebuild":
        return RebuildSolverMixin
    if rebuild_strategy == "pool":
        return SolverPoolMixin
    raise ValueError(
        f"Unknown rebuild_strategy '{rebuild_strategy}'. Choose 'pool' or 'rebuild'."
    )
