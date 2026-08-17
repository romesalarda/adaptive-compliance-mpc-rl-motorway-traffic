from macroscopic.solvers.base import BaseMPCSolver

from typing import TypeVar

import casadi as cs
import numpy as np

SymType = TypeVar("SymType", cs.SX, cs.MX)

from macroscopic.logger import LOGGER
class MpcRMSimulation(BaseMPCSolver):
    """
    A class for running simulations using the Model Predictive Control (MPC) solver.
    Inherits from BaseMPCSolver and provides methods for simulating traffic flow dynamics over time.
    """

    def _simulate(self):

        sol_prev = None
        F = self.get_function()

        rho = self.environment.parameters.rho_init
        v = self.environment.parameters.v_free_init
        w = self.environment.parameters.w_init

        r_last = cs.DM.ones(self.r.size1(), 1)

        for k in range(self.environment.demands.shape[0]):

            d_hat = self.calculate_demands_over_horizon(k)

            if k % self.M == 0:
                sol = self.MPC.solve(
                    pars={
                        "rho_0": rho,
                        "v_0": v,
                        "w_0": w,
                        "d": d_hat.T,
                        "r_last": r_last,
                    },
                    vals0=sol_prev,
                )
                sol_prev = sol.vals
                r_last = sol.vals["r"][0]
                print(sol.vals["r"])
            
            rho, v, w, q, q_on = self.get_next_states(rho, v, w, r_last, k, F=F)

            self.store_step(rho, v, w, q, q_on, r_last=r_last, omit_if_none=True)

            if k % 100 == 0:
                LOGGER.info(f"step {k} of {self.environment.demands.shape[0]}")

    def solve_single_step(self, rho, v, w, demands_forecast, current_time=None):

        F = self.get_function()
        r_last = cs.DM.ones(self.r.size1(), 1) if self._r_last_value is None else self._r_last_value

        d_hat = demands_forecast

        rho_safe = cs.fmax(rho, 1e-3)
        v_safe = cs.fmax(v, 1e-3)
        w_safe = cs.fmax(w, 0)

        rho_safe = np.squeeze(rho_safe.full())
        v_safe   = np.squeeze(v_safe.full())
        w_safe   = np.squeeze(w_safe.full())

        sol = self.MPC.solve(
            pars={
                "rho_0": rho_safe,
                "v_0": v_safe,
                "w_0": w_safe,
                "d": d_hat.T,
                "r_last": r_last,
            },
        )
        # sol_prev = sol.vals
        r_first = sol.vals["r"][:, 0]
        if r_first[0] < 0:
            r_first = [0]

        self._r_last_value = r_first

        rho, v, w, q, q_on = self.get_next_states(rho, v, w, r_first, current_time, F=F)

        self.store_step(rho, v, w, q, q_on, r_last=r_last)

        # sol.vals state arrays are shape (n_seg/n_orig, horizon+1); col 0 = initial state
        result: dict = {"r": r_first[0]}
        result.update(self.parse_predictions(sol.vals))
        return result