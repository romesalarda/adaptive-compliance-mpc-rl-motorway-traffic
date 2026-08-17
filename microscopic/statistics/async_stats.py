from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AsyncMPCSolveRecord:
    """Timing and state snapshot for one complete MPC solve cycle."""

    cycle_id: int = 0
    push_time_sim: float = 0.0       # simulation time when state was pushed (s)
    push_time_wall: float = 0.0      # perf_counter value at push
    receive_time_sim: float = 0.0    # simulation time when controls arrived
    receive_time_wall: float = 0.0   # perf_counter value at receive
    solve_time_reported: float = 0.0 # elapsed inside solve_single_step()
    solve_time_wall: float = 0.0     # push_time_wall → receive_time_wall
    state_rho: list[float] = field(default_factory=list)
    state_v: list[float] = field(default_factory=list)
    state_w: list[float] = field(default_factory=list)
    state_queue_was_full: bool = False
    status: str = "incomplete"       # 'completed' | 'incomplete'


@dataclass
class AsyncMPCStats:
    """Per-cycle solve records and queue events from one async MPC run."""

    records: list[AsyncMPCSolveRecord] = field(default_factory=list)
    # state_queue_full events logged when the worker hasn't consumed a state in time
    queue_events: list[dict] = field(default_factory=list)


def save_async_stats_hdf5(stats: AsyncMPCStats, path: str | Path) -> None:
    """Write AsyncMPCStats to an HDF5 file."""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        completed = sum(1 for r in stats.records if r.status == "completed")
        mg = f.create_group("meta")
        mg.attrs["total_cycles"] = len(stats.records)
        mg.attrs["completed_cycles"] = completed
        mg.attrs["incomplete_cycles"] = len(stats.records) - completed

        if stats.records:
            sg = f.create_group("statistics")

            scalar_float = [
                "push_time_sim", "push_time_wall",
                "receive_time_sim", "receive_time_wall",
                "solve_time_reported", "solve_time_wall",
            ]
            for key in scalar_float:
                vals = np.array([getattr(r, key) for r in stats.records], dtype=np.float64)
                sg.create_dataset(key, data=vals)

            sg.create_dataset(
                "cycle_id",
                data=np.array([r.cycle_id for r in stats.records], dtype=np.int64),
            )
            sg.create_dataset(
                "state_queue_was_full",
                data=np.array([r.state_queue_was_full for r in stats.records], dtype=bool),
            )

            str_dt = h5py.string_dtype()
            sg.create_dataset(
                "status",
                data=[r.status for r in stats.records],
                dtype=str_dt,
            )

            for key in ("state_rho", "state_v", "state_w"):
                arrays = [getattr(r, key) for r in stats.records]
                if any(arrays):
                    max_len = max((len(a) for a in arrays), default=0)
                    padded = np.full((len(arrays), max_len), np.nan, dtype=np.float32)
                    for i, a in enumerate(arrays):
                        if a:
                            padded[i, : len(a)] = a
                    sg.create_dataset(key, data=padded, compression="gzip")

        if stats.queue_events:
            eg = f.create_group("queue_events")
            str_dt = h5py.string_dtype()
            for key in stats.queue_events[0]:
                vals = [ev.get(key) for ev in stats.queue_events]
                if key == "event_type":
                    eg.create_dataset(key, data=vals, dtype=str_dt)
                else:
                    eg.create_dataset(key, data=np.array(vals, dtype=np.float64))
