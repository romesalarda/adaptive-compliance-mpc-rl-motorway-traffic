
from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from macroscopic.simulation.environment import MacroEnvironment
from microscopic.simulation.environment import MicroEnvironment

if TYPE_CHECKING:
    pass

from macroscopic.logger import LOGGER


class BaseSUMOController:
    """
    Decides what controls to apply based on the current micro/macro state.
    Does NOT apply controls directly — delegates to MicroEnvironment's actuation methods.

    Designed to be called synchronously inside the simulation loop.
    Each instance is owned by exactly one SUMOBaseSimulation (one process).
    """

    def __init__(
        self,
        macro_env: MacroEnvironment,
        micro_env: MicroEnvironment,
        solve_interval_steps: int = 100,
    ):
        self.macro_env = macro_env
        self.micro_env = micro_env
        self.solve_interval_steps = solve_interval_steps

        self._last_controls: dict = {}
        self._last_solve_time: float = 0.0
        self._controls_received_this_step: bool = False

    def should_solve(self, step: int) -> bool:
        return step % self.solve_interval_steps == 0

    def setup(self) -> None:
        """Called once before the simulation loop starts."""
        pass

    @abstractmethod
    def compute_action(self, state: dict) -> dict:
        """
        Given state {'rho', 'v', 'w'}, return controls {'v_ctrl', 'r'}.
        Return an empty dict to skip control application this step.
        """
        raise NotImplementedError

    def apply_controls(self, controls: dict, env: MicroEnvironment) -> None:
        """Push computed controls into the environment."""
        if "v_ctrl" in controls and env.vsl_edge_ids:
            for edge_id, speed_kmh in zip(env.vsl_edge_ids, controls["v_ctrl"]):
                print(f"[BaseSUMOController] Applying VSL {float(speed_kmh):.1f} km/h to edge {edge_id}")
                env.set_variable_speed_limit(edge_id, float(speed_kmh))

        if "r" in controls and env.ramp_tl_map:
            rates = np.atleast_1d(controls["r"])
            for tl_id, rate in zip(env.ramp_tl_map.values(), rates):
                print(f"[BaseSUMOController] Applying ramp metering rate {float(rate):.2f} to TL {tl_id}")
                env.set_ramp_metering_rate(tl_id, float(rate))

    def on_step(self, step: int, env: MicroEnvironment, state: dict) -> None:
        """Called every simulation step; runs the solve+apply cycle when due."""
        self._controls_received_this_step = False
        if not self.should_solve(step):
            return
        t0 = time.perf_counter()
        controls = self.compute_action(state)
        self._last_solve_time = time.perf_counter() - t0
        if controls:
            self.apply_controls(controls, env)
            self._last_controls = controls
            self._controls_received_this_step = True

    # ------------------------------------------------------------------
    # Manual override
    # ------------------------------------------------------------------

    def surrender_ramp_to_manual(self, tl_id: str) -> None:
        """Yield RM control for this TL; MPC will no longer actuate it."""
        self.micro_env.surrender_ramp_to_manual(tl_id)

    def restore_ramp_control(self, tl_id: str) -> None:
        self.micro_env.restore_ramp_control(tl_id)

    def stop(self, timeout: float = 5.0) -> None:
        """No-op for synchronous controllers; overridden by AsyncMPCController."""


class MPCInitialiserMixin:
    '''
    Mixin for controllers that need to compute an initial solution before the simulation loop starts.
    '''
    def setup(self) -> None:

        LOGGER.info("\nBeginning init solution")
        start_time = time.perf_counter()

        solution = self.solver.solve_single_step(
            rho=self.macro_env.parameters.rho_init,
            v=self.macro_env.parameters.v_free_init,
            w=self.macro_env.parameters.w_init,
            demands_forecast=self._get_demands_forecast(0),
            current_time=0,
        )

        end_time = time.perf_counter()

        controls: dict = {"solve_time": end_time - start_time}

        if "v_ctrl" in solution:
            v_ctrl = np.atleast_1d(solution["v_ctrl"])
            controls["v_ctrl"] = np.round(v_ctrl / 10) * 10
        if "r" in solution:
            controls["r"] = np.atleast_1d(solution["r"])
        # Pass predicted state trajectories for recording and visualisation
        for key in ("rho_pred", "v_pred", "w_pred"):
            if key in solution:
                controls[key] = solution[key]

        self.apply_controls(controls, self.micro_env)

        self._last_controls = controls

        LOGGER.info("\nFinsihed setup of init solution")
    