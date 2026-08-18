from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from microscopic.controller.base import BaseSUMOController, MPCInitialiserMixin
from macroscopic.simulation.environment import MacroEnvironment
from microscopic.simulation.environment import MicroEnvironment
from macroscopic.logger import LOGGER

if TYPE_CHECKING:
    from macroscopic.solvers.base import BaseMPCSolver


class SynchronousMPCController(MPCInitialiserMixin, BaseSUMOController):
    """
    Closed-loop controller: feeds SUMO state into the macroscopic MPC solver
    and applies resulting speed limits and metering rates back to SUMO.

    Becomes fully functional once BaseMPCSolver.solve_single_step() is implemented.
    """

    def __init__(
        self,
        macro_env: MacroEnvironment,
        micro_env: MicroEnvironment,
        solver: BaseMPCSolver,
        solve_interval_steps: int = 100,
    ):
        super().__init__(macro_env, micro_env, solve_interval_steps)
        self.solver = solver
        self._current_step: int = 0

    def compute_action(self, state: dict) -> dict:
        """
        Map SUMO state → MPC solve → control dict {'v_ctrl', 'r'}.
        Returns {} until solve_single_step() is implemented on the solver.
        """
        demands_forecast = self._get_demands_forecast()

        # try:
        solution = self.solver.solve_single_step(
            rho=state["rho"],
            v=state["v"],
            w=state["w"],
            demands_forecast=demands_forecast,
            current_time=self._current_step,
        )
        # except NotImplementedError:
        #     LOGGER.warning("solve single step is not implemented for class %s " % self.solver.__class__)
        #     return {}

        controls: dict = {}

        if "v_ctrl" in solution:
            v_ctrl = np.atleast_1d(solution["v_ctrl"])
            # round to nearest 10 km/h — standard VSL sign posting
            controls["v_ctrl"] = np.round(v_ctrl / 10) * 10

        if "r" in solution:
            controls["r"] = np.atleast_1d(solution["r"])

        return controls

    def _get_demands_forecast(self, *args) -> np.ndarray:
        """Slice the macro demand array over the MPC horizon from the current step."""
        # convert micro step → macro row (T_macro / step_length micro steps per macro step)
        T_macro_s = self.macro_env.parameters.T * 3600
        macro_step = int(self._current_step * self.micro_env.step_length / T_macro_s)
        horizon = getattr(self.solver, "Np", 7) * getattr(self.solver, "M", 6)

        d_hat = self.macro_env.demands[macro_step : macro_step + horizon, :]
        if d_hat.shape[0] < horizon:
            pad_width = horizon - d_hat.shape[0]
            if d_hat.shape[0] == 0:
                d_hat = np.pad(d_hat, ((0, pad_width), (0, 0)), "constant")
            else:
                d_hat = np.pad(d_hat, ((0, pad_width), (0, 0)), "edge")
        return d_hat

    def on_step(self, step: int, env: MicroEnvironment, state: dict) -> None:
        self._current_step = step
        super().on_step(step, env, state)
