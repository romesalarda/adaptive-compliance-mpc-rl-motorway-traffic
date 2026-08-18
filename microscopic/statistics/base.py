from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from macroscopic.logger import LOGGER


@dataclass
class MicroRunSimulationResults:
    """Per-timestep data recorded during a single simulation run."""

    steps: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    # (n_steps, n_segments) after finalise — lists during recording
    density: list[np.ndarray] = field(default_factory=list)   # veh/km/lane
    speed: list[np.ndarray] = field(default_factory=list)     # km/h
    queue: list[np.ndarray] = field(default_factory=list)     # vehicles per origin

    # Recorded only at MPC solve steps (sparse)
    v_ctrl_applied: list[np.ndarray] = field(default_factory=list)
    r_applied: list[np.ndarray] = field(default_factory=list)
    mpc_solve_times: list[float] = field(default_factory=list)
    solve_steps: list[int] = field(default_factory=list)         # which steps triggered a solve
    solve_timestamps: list[float] = field(default_factory=list)  # simulation time (s) at each solve reception


@dataclass
class MicroOverviewSimulationResults:
    """Aggregated KPIs computed after the run completes."""

    total_time_spent: float = 0.0          # veh·h (TTS)
    total_distance_travelled: float = 0.0  # veh·km (TTD)
    total_waiting: float = 0.0             # veh·h (time spent queued)
    total_collisions: int = 0
    mean_density: float = 0.0              # veh/km/lane, space-time average
    mean_speed: float = 0.0                # km/h, space-time average
    mean_mpc_solve_time: float = 0.0       # s
    runtime_s: float = 0.0                 # wall-clock time for the full run


@dataclass
class MicroSimulationResults:
    """Container for one complete simulation run — live data + KPIs + metadata."""

    run: MicroRunSimulationResults = field(default_factory=MicroRunSimulationResults)
    overview: MicroOverviewSimulationResults = field(default_factory=MicroOverviewSimulationResults)

    scenario_id: str = ""
    solver_type: str = ""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Set by finalize(); numpy arrays with shape (n_steps, n_*)
    density: Optional[np.ndarray] = None
    speed: Optional[np.ndarray] = None
    queue: Optional[np.ndarray] = None

    def record_step(
        self,
        step: int,
        time: float,
        state: dict,
        controls: Optional[dict] = None,
        solve_time: Optional[float] = None,
    ) -> None:
        r = self.run
        r.steps.append(step)
        r.timestamps.append(time)
        r.density.append(state["rho"].copy())
        r.speed.append(state["v"].copy())
        r.queue.append(state["w"].copy())


        if controls is not None:
            has_controls = False
            if "v_ctrl" in controls:
                r.v_ctrl_applied.append(np.atleast_1d(controls["v_ctrl"]).ravel().copy())
                has_controls = True
            if "r" in controls:
                r.r_applied.append(np.atleast_1d(controls["r"]).ravel().copy())
                has_controls = True
            if has_controls:
                r.solve_steps.append(step)
                r.solve_timestamps.append(time)
            else:
                LOGGER.error("No controls found in record_step() call, but controls dict was not None.")

        if solve_time is not None:
            r.mpc_solve_times.append(solve_time)

    def finalize(
        self,
        step_length: float = 0.1,
        n_steps: int = 0,
        segment_lengths_km: Optional[list[float]] = None,
        segment_lanes: Optional[list[int]] = None,
        runtime_s: float = 0.0,
    ) -> None:
        """Convert lists to arrays and compute overview KPIs."""
        r = self.run

        if not r.steps:
            return

        self.density = np.array(r.density, dtype=np.float32)   # (n_steps, n_segments)
        self.speed = np.array(r.speed, dtype=np.float32)
        self.queue = np.array(r.queue, dtype=np.float32)        # (n_steps, n_origins)

        ov = self.overview
        ov.runtime_s = runtime_s

        dt_h = step_length / 3600.0  # step duration in hours

        if segment_lengths_km is not None and segment_lanes is not None:
            L = np.array(segment_lengths_km)
            lam = np.array(segment_lanes)
            # TTS = T * sum_t( sum_i(rho_i * L_i * lam_i) + sum_j(w_j) )
            ov.total_time_spent = float(
                dt_h * np.sum(self.density * (L * lam)[np.newaxis, :])
                + dt_h * np.sum(self.queue)
            )
            ov.total_distance_travelled = float(
                dt_h * np.sum(self.speed * self.density * (L * lam)[np.newaxis, :])
            )
        else:
            # Coarse estimate without geometry (useful before scenario meta is available)
            ov.total_time_spent = float(dt_h * np.sum(self.density) + dt_h * np.sum(self.queue))

        ov.total_waiting = float(dt_h * np.sum(self.queue))
        ov.mean_density = float(np.mean(self.density))
        ov.mean_speed = float(np.mean(self.speed[self.speed > 0])) if np.any(self.speed > 0) else 0.0
        ov.mean_mpc_solve_time = float(np.mean(r.mpc_solve_times)) if r.mpc_solve_times else 0.0
