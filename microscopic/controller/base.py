
from __future__ import annotations

import time
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from macroscopic.simulation.environment import MacroEnvironment
from microscopic.simulation.environment import MicroEnvironment

if TYPE_CHECKING:
    pass


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
                env.set_variable_speed_limit(edge_id, float(speed_kmh))

        if "r" in controls and env.ramp_tl_map:
            rates = np.atleast_1d(controls["r"])
            for tl_id, rate in zip(env.ramp_tl_map.values(), rates):
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

    