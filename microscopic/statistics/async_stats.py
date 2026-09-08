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
    # MPC predicted trajectories from solver; shape (horizon, n_seg/n_orig) stored as list-of-lists
    rho_pred: list = field(default_factory=list)
    v_pred: list = field(default_factory=list)
    w_pred: list = field(default_factory=list)
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

            # Predicted trajectories: (n_cycles, horizon, n_seg) — only stored when solver exposes them
            for key in ("rho_pred", "v_pred", "w_pred"):
                arrays = [getattr(r, key) for r in stats.records]
                if any(a for a in arrays):
                    arr_3d = np.array([np.array(a, dtype=np.float32) if a else np.full((1, 1), np.nan, dtype=np.float32) for a in arrays], dtype=object)
                    # All shapes equal when solver returns predictions; stack safely
                    try:
                        stacked = np.stack([np.array(a, dtype=np.float32) for a in arrays])
                        sg.create_dataset(key, data=stacked, compression="gzip")
                    except (ValueError, TypeError):
                        pass  # skip if shapes are inconsistent

        if stats.queue_events:
            eg = f.create_group("queue_events")
            str_dt = h5py.string_dtype()
            for key in stats.queue_events[0]:
                vals = [ev.get(key) for ev in stats.queue_events]
                if key == "event_type":
                    eg.create_dataset(key, data=vals, dtype=str_dt)
                else:
                    eg.create_dataset(key, data=np.array(vals, dtype=np.float64))


def load_async_stats_hdf5(path: str | Path) -> AsyncMPCStats:
    """Read an AsyncMPCStats previously written by save_async_stats_hdf5()."""
    import h5py

    path = Path(path)
    stats = AsyncMPCStats()

    with h5py.File(path, "r") as f:
        if "statistics" not in f:
            return stats

        sg = f["statistics"]
        n = int(f["meta"].attrs.get("total_cycles", 0))

        scalar_float = [
            "push_time_sim", "push_time_wall",
            "receive_time_sim", "receive_time_wall",
            "solve_time_reported", "solve_time_wall",
        ]
        scalars = {key: sg[key][:] if key in sg else np.zeros(n) for key in scalar_float}
        cycle_id = sg["cycle_id"][:] if "cycle_id" in sg else np.arange(n)
        queue_full = sg["state_queue_was_full"][:] if "state_queue_was_full" in sg else np.zeros(n, dtype=bool)
        status = sg["status"].asstr()[:] if "status" in sg else ["completed"] * n

        padded = {key: sg[key][:] if key in sg else None for key in ("state_rho", "state_v", "state_w")}
        preds = {key: sg[key][:] if key in sg else None for key in ("rho_pred", "v_pred", "w_pred")}

        for i in range(n):
            record = AsyncMPCSolveRecord(
                cycle_id=int(cycle_id[i]),
                push_time_sim=float(scalars["push_time_sim"][i]),
                push_time_wall=float(scalars["push_time_wall"][i]),
                receive_time_sim=float(scalars["receive_time_sim"][i]),
                receive_time_wall=float(scalars["receive_time_wall"][i]),
                solve_time_reported=float(scalars["solve_time_reported"][i]),
                solve_time_wall=float(scalars["solve_time_wall"][i]),
                state_queue_was_full=bool(queue_full[i]),
                status=str(status[i]),
            )
            for key, arr in padded.items():
                if arr is not None:
                    row = arr[i]
                    setattr(record, key, row[~np.isnan(row)].tolist())
            for key, arr in preds.items():
                if arr is not None:
                    setattr(record, key, arr[i].tolist())
            stats.records.append(record)

        if "queue_events" in f:
            eg = f["queue_events"]
            keys = list(eg.keys())
            n_events = len(eg[keys[0]]) if keys else 0
            for i in range(n_events):
                event = {}
                for key in keys:
                    val = eg[key][i]
                    event[key] = val.decode() if isinstance(val, bytes) else (
                        val if not hasattr(val, "item") else val.item()
                    )
                stats.queue_events.append(event)

    return stats
