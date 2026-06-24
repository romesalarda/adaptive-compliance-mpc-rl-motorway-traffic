from macroscopic.simulation.environment import Environment
from macroscopic.solvers.base import BaseMPCSolver


from typing import TypeVar

import casadi as cs
import numpy as np

SymType = TypeVar("SymType", cs.SX, cs.MX)

from macroscopic.simulation.environment import Environment
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
            
            rho, v, w, q, q_on = self.get_next_states(rho, v, w, r_last, k, F=F)

            self.store_step(rho, v, w, q, q_on, r_last)

            if k % 100 == 0:
                LOGGER.info(f"step {k} of {self.environment.demands.shape[0]}")

