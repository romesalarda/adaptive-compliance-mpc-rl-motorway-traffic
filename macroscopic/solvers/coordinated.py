from sym_metanet import MeteredOnRamp

from macroscopic.simulation.environment import Environment
from macroscopic.simulation.parameters import VslParameters
from macroscopic.solvers.vsl import MpcVSLSimulation


from typing import TypeVar

import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)

from macroscopic.simulation.environment import Environment
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

    def init_constraints(self):

        delta_v_max = 10  # Max drop allowed
        delta_v_increase = 0  # Change to 0 to allow flat profiles
        max_temporal_change = 10 # i.e. between time steps
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

        r_weight = 0.4
        temporal_smoothness_weight = 0.4
        seg4_safety_weight = 1.0
        threshold = 25
        seg_4_idx = 3  # Assuming segment 4 is at index 3 (0-based indexing)

        temporal_smoothness = temporal_smoothness_weight * cs.sumsqr(cs.diff(cs.horzcat(self.v_ctrl_last, self.v_ctrl), 1, 1) / parameters.v_free)
        # solid with 40 - 60 as temporal smoothness
        tts_term = cs.sum2(cs.sum1(self.rho * parameters.L * parameters.lanes) + cs.sum1(self.w))
    
        # congestion_indicator = cs.fmax(0, self.w[1, 0] - threshold)
        congestion_indicator = cs.fmax(0, self.w[1, 0] - threshold)
        seg4_safety_penalty = seg4_safety_weight * congestion_indicator * cs.sumsqr(self.v_ctrl[seg_4_idx, :] / parameters.v_free)

        self.MPC.minimize(
            parameters.T * tts_term
            + r_weight * cs.sumsqr(cs.diff(cs.horzcat(self.r_last, self.r)))
            + temporal_smoothness
            + seg4_safety_penalty
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
                r_last = sol.vals["r"][:, 0]
            
            rho, v, w, q, q_on = self.get_next_states(rho, v, w, v_ctrl_last, r_last, k, F=F)

            self.store_step(rho, v, w, q, q_on, v_ctrl_last=v_ctrl_last, r_last=r_last, omit_if_none=True)

            if k % 100 == 0:
                LOGGER.info(f"step {k} of {self.environment.demands.shape[0]}")