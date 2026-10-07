from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from macroscopic.logger import LOGGER
from macroscopic.simulation.environment import MacroEnvironment
from microscopic.controller.base import BaseSUMOController, MPCInitialiserMixin
from microscopic.simulation.environment import MicroEnvironment
from microscopic.statistics.async_stats import AsyncMPCSolveRecord, AsyncMPCStats

if TYPE_CHECKING:
    from macroscopic.solvers.base import BaseMPCSolver


class AsyncMPCController(MPCInitialiserMixin, BaseSUMOController):
    """
    Non-blocking MPC controller: SUMO never waits for the solver.

    A daemon thread continuously pulls state snapshots from a queue, calls
    solve_single_step(), and returns controls via a second queue.  The main
    (SUMO) thread pushes state at every solve interval and receives the latest
    available controls — no blocking, no dropped steps.

    Communication pattern:
        main thread  →  _state_queue  →  MPC thread
        MPC thread   →  _control_queue →  main thread
    """

    def __init__(
        self,
        macro_env: MacroEnvironment,
        micro_env: MicroEnvironment,
        solver: BaseMPCSolver,
        solve_interval_steps: int = 100,
        step_length: float = 0.1,
    ):
        super().__init__(macro_env, micro_env, solve_interval_steps)
        self.solver = solver
        self.step_length = step_length
        # How far apart (in simulation seconds) state pushes occur
        self._solve_interval_s: float = solve_interval_steps * step_length

        # maxsize=1: only the most recent state/controls matter
        self._state_queue: queue.Queue = queue.Queue(maxsize=1)
        self._control_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        # Serialise solver access in case of future re-entrancy
        self._solver_lock = threading.RLock()

        # Main-thread-only timing state (no lock needed)
        self._last_push_sim: float = float("-inf")
        self._cycle_id: int = 0
        self._consecutive_dropped_pushes: int = 0

        self.async_stats = AsyncMPCStats()

        self._preflight_checks()

        self._mpc_thread = threading.Thread(
            target=self._mpc_worker, daemon=True, name="mpc-solver"
        )
        self._mpc_thread.start()

    def _preflight_checks(self) -> None:
        """Catch solve-timing misalignments early instead of letting them silently skew MPC predictions."""
        if self.step_length <= 0 or self._solve_interval_s <= 0:
            raise ValueError(
                f"Invalid controller timing: step_length={self.step_length}, "
                f"solve_interval_s={self._solve_interval_s}"
            )
        macro_dt_s = float(self.macro_env.parameters.T) * 3600.0
        macro_steps_per_solve = self._solve_interval_s / macro_dt_s
        if not np.isclose(macro_steps_per_solve, round(macro_steps_per_solve), atol=1e-6):
            LOGGER.warning(
                "[AsyncMPC] solve interval (%.1fs) is not an integer multiple of the macro model's "
                "time step T=%.1fs (%.3f macro steps) — demand-forecast slicing (_get_demands_forecast) "
                "will misalign with real simulation time.",
                self._solve_interval_s, macro_dt_s, macro_steps_per_solve,
            )
            return
        M = getattr(self.solver, "M", None)
        n_macro_steps = int(round(macro_steps_per_solve))
        if M and n_macro_steps % M != 0:
            LOGGER.warning(
                "[AsyncMPC] solve interval spans %d macro time-step(s) but the solver's control "
                "spacing M=%d does not divide it evenly — solve cadence is misaligned with the "
                "MPC's own control-update boundary.",
                n_macro_steps, M,
            )

    # ------------------------------------------------------------------
    # Background MPC thread
    # ------------------------------------------------------------------

    def _mpc_worker(self) -> None:
        LOGGER.info("[AsyncMPC] worker thread started (interval=%.1fs)", self._solve_interval_s)

        while not self._stop_event.is_set():
            try:
                state = self._state_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            try:
                with self._solver_lock:

                    LOGGER.info("[AsyncMPC] cycle %d: solving for sim t=%.1fs (wall t=%.1fs)",
                                state["_cycle_id"],
                                state["_push_sim_time"],
                                state["_push_wall_time"],
                            )
                    
                    LOGGER.info("[AsyncMPC] state: \nrho=%s, \nv=%s, \nw=%s",
                                state["rho"].tolist(),
                                state["v"].tolist(),
                                state["w"].tolist(),
                            )

                    solution = self.solver.solve_single_step(
                        rho=state["rho"],
                        v=state["v"],
                        w=state["w"],
                        demands_forecast=state["demands_forecast"],
                        current_time=state["current_step"],
                        v_ctrl_last=state.get("v_ctrl_last"),
                        r_last=state.get("r_last"),
                    )
                    
            except NotImplementedError:
                raise RuntimeError(
                    "[AsyncMPC] solve_single_step() is not implemented for class %s"
                    % self.solver.__class__
                )
            except Exception as exc:
                LOGGER.error("[AsyncMPC] solve_single_step failed (cycle %d): %s", state["_cycle_id"], exc)
                raise exc                

            LOGGER.info("[AsyncMPC] cycle %d solved in %.3fs (sim t=%.1fs)",
                        state["_cycle_id"],
                        time.perf_counter() - t0,
                        state["_push_sim_time"],
                    )

            # LOGGER.info("[AsyncMPC] solutions: %s", solution)

            solve_time = time.perf_counter() - t0

            controls: dict = {"solve_time": solve_time}
            if "v_ctrl" in solution:
                v_ctrl = np.atleast_1d(solution["v_ctrl"])
                controls["v_ctrl"] = np.round(v_ctrl / 10) * 10
            if "r" in solution:
                controls["r"] = np.atleast_1d(solution["r"])
            # Pass predicted state trajectories for recording and visualisation
            for key in ("rho_pred", "v_pred", "w_pred"):
                if key in solution:
                    controls[key] = solution[key]

            # Bundle push-time metadata so the main thread can compute delays
            meta = {
                "_cycle_id": state["_cycle_id"],
                "_push_sim_time": state["_push_sim_time"],
                "_push_wall_time": state["_push_wall_time"],
                "_state_queue_was_full": state["_state_queue_was_full"],
            }

            payload = (controls, meta)
            # Replace stale result if the queue is full
            try:
                self._control_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self._control_queue.get_nowait()
                except queue.Empty:
                    pass
                self._control_queue.put_nowait(payload)

            LOGGER.info(
                "[AsyncMPC] cycle %d solved in %.3fs (sim t=%.1fs)",
                state["_cycle_id"],
                solve_time,
                state["_push_sim_time"],
            )

        LOGGER.info("[AsyncMPC] worker thread stopped")

    # ------------------------------------------------------------------
    # Main-thread helpers
    # ------------------------------------------------------------------

    def _get_demands_forecast(self, step: int) -> np.ndarray:
        """Slice macro demands over the MPC horizon, starting at *step*."""
        # convert micro step → macro row (T_macro / step_length micro steps per macro step)
        T_macro_s = self.macro_env.parameters.T * 3600
        macro_step = int(step * self.step_length / T_macro_s)
        horizon = getattr(self.solver, "Np", 7) * getattr(self.solver, "M", 6)
        d_hat = self.macro_env.demands[macro_step: macro_step + horizon, :]
        if d_hat.shape[0] < horizon:
            pad = horizon - d_hat.shape[0]
            mode = "constant" if d_hat.shape[0] == 0 else "edge"
            d_hat = np.pad(d_hat, ((0, pad), (0, 0)), mode)
        return d_hat

    # ------------------------------------------------------------------
    # Simulation loop hook (called every SUMO step by SUMOBaseSimulation)
    # ------------------------------------------------------------------

    def on_step(self, step: int, env: MicroEnvironment, state: dict) -> None:
        self._controls_received_this_step = False
        sim_time = step * self.step_length

        # ---- push state if solve interval has elapsed ----
        if sim_time - self._last_push_sim >= self._solve_interval_s:
            self._cycle_id += 1
            push_wall = time.perf_counter()

            state_msg: dict = {
                "rho": state["rho"],
                "v": state["v"],
                "w": state["w"],
                "current_step": step,
                "demands_forecast": self._get_demands_forecast(step),
                # single source of truth for 'what's actually applied right now' — never read solver-private state
                "v_ctrl_last": self._last_controls.get("v_ctrl"),
                "r_last": self._last_controls.get("r"),
                "_cycle_id": self._cycle_id,
                "_push_sim_time": sim_time,
                "_push_wall_time": push_wall,
                "_state_queue_was_full": False,
            }

            try:
                self._state_queue.put_nowait(state_msg)
                self._consecutive_dropped_pushes = 0
            except queue.Full:
                # Worker is busy; replace the unconsumed state
                try:
                    self._state_queue.get_nowait()
                except queue.Empty:
                    pass
                state_msg["_state_queue_was_full"] = True
                self._state_queue.put_nowait(state_msg)
                self.async_stats.queue_events.append({
                    "event_type": "state_queue_full",
                    "time_sim": sim_time,
                    "time_wall": push_wall,
                })
                self._consecutive_dropped_pushes += 1
                LOGGER.warning("[AsyncMPC] state queue full at t=%.1fs (cycle %d)", sim_time, self._cycle_id)
                if self._consecutive_dropped_pushes >= 3:
                    LOGGER.error(
                        "[AsyncMPC] %d consecutive solve cycles dropped — solver can't keep up with the "
                        "%.1fs interval; applied controls are now stale relative to the live plant.",
                        self._consecutive_dropped_pushes, self._solve_interval_s,
                    )

            self._last_push_sim = sim_time

        # ---- receive controls if the worker has a result ready ----
        try:
            controls, meta = self._control_queue.get_nowait()
        except queue.Empty:
            return
        else:
            LOGGER.info("\n[AsyncMPC] received controls for cycle %d at t=%.1fs", meta["_cycle_id"], sim_time)

            v_ctrl = controls.get("v_ctrl", None)
            r = controls.get("r", None)
            if v_ctrl is not None:
                LOGGER.info("[AsyncMPC] received v_ctrl=%s for cycle %d at t=%.1fs", v_ctrl.tolist(), meta["_cycle_id"], sim_time)
            if r is not None:
                LOGGER.info("[AsyncMPC] received r=%s for cycle %d at t=%.1fs", r.tolist(), meta["_cycle_id"], sim_time)
                
        receive_wall = time.perf_counter()

        # Update tracked state used by SUMOBaseSimulation for stats recording
        self._last_controls = controls
        self._last_solve_time = controls.get("solve_time", 0.0)
        self._controls_received_this_step = True

        self.async_stats.records.append(AsyncMPCSolveRecord(
            cycle_id=meta["_cycle_id"],
            push_time_sim=meta["_push_sim_time"],
            push_time_wall=meta["_push_wall_time"],
            receive_time_sim=sim_time,
            receive_time_wall=receive_wall,
            solve_time_reported=controls.get("solve_time", 0.0),
            solve_time_wall=receive_wall - meta["_push_wall_time"],
            state_rho=state["rho"].tolist(),
            state_v=state["v"].tolist(),
            state_w=state["w"].tolist(),
            state_queue_was_full=meta["_state_queue_was_full"],
            rho_pred=controls["rho_pred"].tolist() if "rho_pred" in controls else [],
            v_pred=controls["v_pred"].tolist() if "v_pred" in controls else [],
            w_pred=controls["w_pred"].tolist() if "w_pred" in controls else [],
            status="completed",
        ))

        self.apply_controls(controls, env)

    def compute_action(self, state: dict) -> dict:
        # on_step() is overridden directly; this path is never taken
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self, timeout: float = 5.0) -> None:
        if not self._mpc_thread.is_alive():
            return
        self._stop_event.set()
        self._mpc_thread.join(timeout=timeout)
        if self._mpc_thread.is_alive():
            LOGGER.warning("[AsyncMPC] thread did not stop within %.1fs", timeout)
        else:
            LOGGER.info("[AsyncMPC] thread stopped cleanly")
        n_incomplete = self._cycle_id - len(self.async_stats.records)
        if n_incomplete > 0:
            LOGGER.info("[AsyncMPC] %d cycle(s) pushed but never resolved", n_incomplete)
