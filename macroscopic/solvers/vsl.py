from sym_metanet import MeteredOnRamp

from macroscopic.simulation.environment import MacroEnvironment
from macroscopic.simulation.parameters import VslParameters
from macroscopic.solvers.base import BaseMPCSolver


from typing import TypeVar

import casadi as cs
import numpy as np

SymType = TypeVar("SymType", cs.SX, cs.MX)

from macroscopic.simulation.environment import MacroEnvironment
from macroscopic.logger import LOGGER

class MpcVSLSimulation(BaseMPCSolver):
    """
    A class for running simulations using the Model Predictive Control (MPC) solver.
    Inherits from BaseMPCSolver and provides methods for simulating traffic flow dynamics over time with Variable Speed Limits (VSL).
    """

    def __init__(self, environment: MacroEnvironment, Np=None, Nc=None, M=None, **kwargs):
        super().__init__(environment, Np, Nc, M, **kwargs)
        self._v_ctrl_last = None  # Initialize the last speed limit control variable
        self._v_ctrl = None  # Initialize the speed limit control variable

        self._v_ctrl_last_value = None  # Store the last value of v_ctrl for use in the next iteration

    @property
    def v_ctrl_last(self):
        '''
        Last speed limit control variable from the previous MPC iteration.
        '''
        return self._v_ctrl_last    
    
    @v_ctrl_last.setter
    def v_ctrl_last(self, value):
        self._v_ctrl_last = value
    
    @property
    def v_ctrl(self):
        '''
        Current speed limit control variable for the ongoing MPC iteration.
        '''
        return self._v_ctrl
    
    @v_ctrl.setter
    def v_ctrl(self, value):
        self._v_ctrl = value

    def init_control_states(self):

        parameters: VslParameters = self.environment.parameters

        self.v_ctrl, _ = self.MPC.action(
            "v_ctrl", len(parameters.segments_with_vsl),
            lb=parameters.minimum_speed_limit, 
            ub=parameters.maximum_speed_limit
        ) 

        n_metered = sum(1 for o in self.environment.network.origins if isinstance(o, MeteredOnRamp))
        self.r, _ = self.MPC.action(
            "r", 
            n_metered, 
            lb=self.use_weight_parameter("r_lower_bound", 1),
            ub=self.use_weight_parameter("r_upper_bound", 1)
        )
        
        self.v_ctrl_last = self.MPC.parameter("v_ctrl_last", (self.v_ctrl.size1(), 1))

    def init_constraints(self):


        delta_v_max = self.use_weight_parameter("delta_v_max", 10)  # Max drop allowed
        delta_v_increase = self.use_weight_parameter("delta_v_increase", 0)  # Change to 0 to allow flat profiles
        max_temporal_change = self.use_weight_parameter("max_temporal_change", 10) # i.e. between time steps
        mpc = self.MPC
        v_ctrl = self.v_ctrl
        v_ctrl_last = self.v_ctrl_last

        for i in range(v_ctrl.size1()):
            mpc.constraint(f"temporal_pos_{i}_0", v_ctrl_last[i] - v_ctrl[i, 0], "<=", max_temporal_change, soft=False)
            mpc.constraint(f"temporal_neg_{i}_0", v_ctrl[i, 0] - v_ctrl_last[i], "<=", max_temporal_change, soft=False)
            
            for k in range(v_ctrl.size2() - 1):
                mpc.constraint(f"temporal_pos_{i}_{k+1}", v_ctrl[i, k] - v_ctrl[i, k+1], "<=", max_temporal_change, soft=False)
                mpc.constraint(f"temporal_neg_{i}_{k+1}", v_ctrl[i, k+1] - v_ctrl[i, k], "<=", max_temporal_change, soft=False)

        for k in range(v_ctrl.size2()): # i.e. beteen segments at each time step (spatial smoothness)
            for i in range(v_ctrl.size1() - 1): 
                mpc.constraint(
                    f"vsl_downstream_lower_{i}_{k}",
                    v_ctrl[i + 1, k] - v_ctrl[i, k],  
                    "<=",
                    delta_v_increase,
                    soft=False
                )
                
                mpc.constraint(
                    f"vsl_max_drop_{i}_{k}",
                    v_ctrl[i, k] - v_ctrl[i + 1, k],
                    "<=",
                    delta_v_max,
                    soft=False
                )

    def init_objective(self):

        parameters: VslParameters = self.environment.parameters
        temporal_smoothness_weight = self.use_weight_parameter("temporal_smoothness", 10)
        temporal_smoothness = temporal_smoothness_weight * cs.sumsqr(cs.diff(cs.horzcat(self.v_ctrl_last, self.v_ctrl), 1, 1) / parameters.v_free)

        self.MPC.minimize(
            parameters.T * cs.sum2(cs.sum1(self.rho * parameters.L * parameters.lanes) + cs.sum1(self.w))
            + temporal_smoothness
        )

    def get_next_states(self, rho, v, w, v_ctrl_last, r_last, k, F):

        x_next, q_all = F(cs.vertcat(rho, v, w), cs.vertcat(v_ctrl_last, r_last), self.environment.demands[k, :])
        rho, v, w = cs.vertsplit(x_next, (0, self.number_of_segments, 2 * self.number_of_segments, 2 * self.number_of_segments + self.number_of_origins))
        q, q_on = cs.vertsplit(q_all, (0, self.number_of_segments, self.number_of_segments + self.number_of_origins))
        return rho, v, w, q, q_on

    def _simulate(self):

        sol_prev = None
        F = self.get_function()

        rho = self.environment.parameters.rho_init
        v = self.environment.parameters.v_free_init
        w = self.environment.parameters.w_init
        parameters: VslParameters = self.environment.parameters

        v_ctrl_last = parameters.v_ctrl
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
                        "v_ctrl_last": v_ctrl_last,
                    },
                    vals0=sol_prev,
                )
                sol_prev = sol.vals
                v_ctrl_last = sol.vals["v_ctrl"][:, 0]
                r_last = sol.vals["r"][:, 0]
            
            rho, v, w, q, q_on = self.get_next_states(rho, v, w, v_ctrl_last, r_last, k, F=F)

            self.store_step(rho, v, w, q, q_on, v_ctrl_last=v_ctrl_last, omit_if_none=True)

            if k % 100 == 0:
                LOGGER.info(f"step {k} of {self.environment.demands.shape[0]}")

    def solve_single_step(self, rho, v, w, demands_forecast, current_time=None):
    
        self._v_ctrl_last_value = self.environment.parameters.v_ctrl if self._v_ctrl_last_value is None else self._v_ctrl_last_value

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
                "v_ctrl_last": self._v_ctrl_last_value,
            },
        )
        LOGGER.info(f"\nsolve_single_step: v_ctrl={sol.vals['v_ctrl']} for cycle at t={current_time}")

        self._v_ctrl_last_value = sol.vals["v_ctrl"][:, 0]
        r_first = sol.vals["r"][:, 0]

        LOGGER.info(f"solve_single_step: v_ctrl={self._v_ctrl_last_value}, r={r_first} at t={current_time}")

        # sol.vals state arrays are shape (n_seg/n_orig, horizon+1); col 0 = initial state
        result: dict = {"v_ctrl": self._v_ctrl_last_value, "r": r_first[0]}

        result.update(self.parse_predictions(sol.vals))
        return result