from sym_metanet import MeteredOnRamp

from macroscopic.simulation.parameters import VslParameters
from macroscopic.solvers.vsl import MpcVSLSimulation


from typing import TypeVar

import casadi as cs
import numpy as np

SymType = TypeVar("SymType", cs.SX, cs.MX)

from macroscopic.simulation.environment import MacroEnvironment
from macroscopic.logger import LOGGER

class MpcCoordinatedSimulation(MpcVSLSimulation):
    """
    A class for running simulations using the Model Predictive Control (MPC) solver.
        Inherits from BaseMPCSolver and provides methods for simulating traffic flow dynamics over time with coordinated control of Variable Speed Limits (VSL) and ramp metering.
    """

    def init_control_states(self):

        parameters: VslParameters = self.environment.parameters
        
        self.v_ctrl, _ = self.MPC.action(
            "v_ctrl", len(parameters.segments_with_vsl),
            lb=parameters.minimum_speed_limit, 
            ub=parameters.maximum_speed_limit
        ) 

        n_metered = sum(1 for o in self.environment.network.origins if isinstance(o, MeteredOnRamp))
        self.r, _ = self.MPC.action("r", n_metered, lb=0, ub=1)
        
        self.v_ctrl_last = self.MPC.parameter("v_ctrl_last", (self.v_ctrl.size1(), 1))
        self.r_last = self.MPC.parameter("r_last", (self.r.size1(), 1))

    def init_objective(self):

        # ---------------- old -----------------
        # temporal_smoothness_weight = self.objective_weights.get("temporal_smoothness", 10)
        # temporal_smoothness = temporal_smoothness_weight * cs.sumsqr(cs.diff(cs.horzcat(self.v_ctrl_last, self.v_ctrl), 1, 1) / parameters.v_free)

        # # TTT = sum of all segment densities + queue lengths, multiplied by T

        # self.MPC.minimize(
        #     parameters.T * cs.sum2(cs.sum1(self.rho * parameters.L * parameters.lanes) + cs.sum1(self.w))
        #     + temporal_smoothness
        # )
        # ---------------- old -----------------
        parameters: VslParameters = self.environment.parameters

        r_weight = self.use_weight_parameter("r_weight", 1.0)
        temporal_smoothness_weight = self.use_weight_parameter("temporal_smoothness", 0.4)

        temporal_smoothness = temporal_smoothness_weight * cs.sumsqr(cs.diff(cs.horzcat(self.v_ctrl_last, self.v_ctrl), 1, 1) / parameters.v_free)
        tts_term = cs.sum2(cs.sum1(self.rho * parameters.L * parameters.lanes) + cs.sum1(self.w))

        # T was ommitted here, add T in future

        self.MPC.minimize(
            self.environment.parameters.T * tts_term
            + temporal_smoothness
            + r_weight * cs.sumsqr(cs.diff(cs.horzcat(self.r_last, self.r)))
            # + seg4_safety_penalty
        )

    def _simulate(self):

        sol_prev = None
        F = self.get_function()

        rho = self.environment.parameters.rho_init
        v = self.environment.parameters.v_free_init
        w = self.environment.parameters.w_init
        parameters: VslParameters = self.environment.parameters

        v_ctrl_last = parameters.v_ctrl
        r_last = cs.DM.ones(self.r.size1(), 1)  # ramp metering rates

        for k in range(self.environment.demands.shape[0]):

            d_hat = self.calculate_demands_over_horizon(k)
            if k % self.M == 0:
                sol = self.MPC.solve(
                    pars={
                        "rho_0": rho,
                        "v_0": v,
                        "w_0": w,
                        "d": d_hat.T,
                        "v_ctrl_last": v_ctrl_last,
                        "r_last": r_last,
                    },
                    vals0=sol_prev,
                )
                sol_prev = sol.vals
                v_ctrl_last = sol.vals["v_ctrl"][:, 0]

                v_ctrl_py = [float(v_ctrl_last[i]) for i in range(v_ctrl_last.numel())]
                v_ctrl_last = [round(v / 10) * 10 for v in v_ctrl_py]

                r_last = sol.vals["r"][:, 0]
            
            rho, v, w, q, q_on = self.get_next_states(rho, v, w, v_ctrl_last, r_last, k, F=F)

            self.store_step(rho, v, w, q, q_on, v_ctrl_last=v_ctrl_last, r_last=r_last, omit_if_none=True)

            if k % 100 == 0:
                LOGGER.info(f"step {k} of {self.environment.demands.shape[0]}")

    def solve_single_step(self, rho, v, w, demands_forecast, current_time=None, v_ctrl_last=None, r_last=None):
        """v_ctrl_last/r_last should be the controller's true last-applied control (avoids stale continuity when solvers are swapped by RL)."""

        fallback_r_last = cs.DM.ones(self.r.size1(), 1) if self._r_last_value is None else self._r_last_value
        r_last = self._coerce_last_control(r_last, self.r.size1(), fallback_r_last)

        fallback_v_ctrl_last = self.environment.parameters.v_ctrl if self._v_ctrl_last_value is None else self._v_ctrl_last_value
        v_ctrl_last = self._coerce_last_control(v_ctrl_last, self.v_ctrl.size1(), fallback_v_ctrl_last)

        d_hat = demands_forecast

        rho_safe = cs.fmax(rho, 1e-3)
        v_safe = cs.fmax(v, 1e-3)
        w_safe = cs.fmax(w, 0)

        rho_safe = np.squeeze(rho_safe.full())
        v_safe   = np.squeeze(v_safe.full())
        w_safe   = np.squeeze(w_safe.full())

        try:
            sol = self.MPC.solve(
                pars={
                    "rho_0": rho_safe,
                    "v_0": v_safe,
                    "w_0": w_safe,
                    "d": d_hat.T,
                    "r_last": r_last,
                    "v_ctrl_last": v_ctrl_last,
                },
            )
        except Exception as exc:
            LOGGER.error("[RampSolver] MPC.solve failed (%s); falling back to r_last=%s", exc, r_last)
            raise exc
            r_fallback = float(np.clip(np.array(r_last).flatten()[0], 0.0, 1.0))
            return {"r": r_fallback}

        r_first = np.clip(np.array(sol.vals["r"][:, 0]).flatten(), 0.0, 1.0)
        self._v_ctrl_last_value = sol.vals["v_ctrl"][:, 0]

        LOGGER.warning("Values for single step solution RM %s", sol.vals["r"])

        self._r_last_value = cs.DM(r_first)

        result: dict = {"v_ctrl": self._v_ctrl_last_value, "r": r_first[0]}
        
        result.update(self.parse_predictions(sol.vals))

        return result